"""로컬 LLM(Ollama - Qwen2.5 7B/14B 등)을 활용한 슬라이딩 윈도우 문맥 기반 자막 번역 모듈.

특징:
1. 문장 1개씩 번역할 때 발생하는 문맥 단절, 반말/존댓말 불일치, 대명사 오역을 방지하기 위해
   직전 N개 대사(문맥 참조용) + 현재 번역할 M개 대사 묶음을 JSON 스키마로 전달합니다.
2. 배치 완료 시마다 작업 폴더(`translation_checkpoint.json`)에 즉시 체크포인트를 저장하며,
   재실행 시 이미 번역된 문장은 자동으로 건너뛰고 미번역 배치부터 이어서 번역합니다.
3. 스트리밍 수신 중 `cancel_event`를 실시간 감지하여 사용자가 [즉시 중단] 클릭 시 0.1초 이내에 즉시 중단합니다.
4. Ollama `/api/ps` 및 응답 메타데이터를 분석하여 실제 사용된 연산 장치(CPU vs GPU)와 초당 토큰 속도(t/s)를 로깅합니다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from pipeline.models import SubtitleSegment
from pipeline.telemetry import query_ollama_processor_status
from pipeline.workspace import JobCancelledError

logger = logging.getLogger(__name__)

LANG_KO_NAMES = {
    "zh": "중국어",
    "ja": "일본어",
    "en": "영어",
    "fr": "프랑스어",
    "es": "스페인어",
    "de": "독일어",
    "ru": "러시아어",
    "it": "이탈리아어",
}


class LocalLLMTranslator:
    """Ollama 로컬 API를 이용한 영화 자막 전문 번역기 (즉시 중단, 배치 체크포인트 이어하기, 텔레메트리 내장)."""

    def __init__(self, llm_config: Dict):
        self.base_url = llm_config.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = llm_config.get("model", "qwen2.5:7b")
        self.batch_size = int(llm_config.get("batch_size", 10))
        self.context_size = int(llm_config.get("context_size", 5))
        self.temperature = float(llm_config.get("temperature", 0.2))
        self.timeout_sec = int(llm_config.get("timeout_sec", 180))
        self.movie_context = llm_config.get("movie_context", "").strip()
        self.last_telemetry: Dict[str, Any] = {}

    def _try_start_ollama_server(self, force_restart: bool = False) -> bool:
        """Ollama 서버가 꺼져 있거나 CPU 단독 모드일 때 Intel Arc iGPU Vulkan 가속(OLLAMA_VULKAN=1, OLLAMA_IGPU_ENABLE=1)으로 구동합니다."""
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.exists():
                ollama_bin = str(candidate)
        if not ollama_bin:
            return False

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )

        if force_restart:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama app.exe", "/IM", "ollama.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=5,
                )
                time.sleep(1.0)
            except Exception as exc:
                logger.debug("기존 Ollama 프로세스 종료 중 예외(무시): %s", exc)

        env = os.environ.copy()
        env["OLLAMA_VULKAN"] = "1"
        env["OLLAMA_IGPU_ENABLE"] = "1"
        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            for _ in range(14):
                time.sleep(0.5)
                try:
                    r = requests.get(f"{self.base_url}/api/tags", timeout=2)
                    if r.status_code == 200:
                        logger.info(
                            "Ollama 백그라운드 서버 자동 시작 완료 (OLLAMA_VULKAN=1, OLLAMA_IGPU_ENABLE=1)"
                        )
                        return True
                except requests.RequestException:
                    continue
        except Exception as exc:
            logger.debug("Ollama 자동 시작 실패: %s", exc)
        return False

    def check_server_and_model(self) -> tuple[bool, str]:
        """Ollama 서버 실행 여부 및 모델 다운로드 상태를 확인합니다 (미실행 시 자동 시작)."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
        except requests.RequestException:
            if not self._try_start_ollama_server():
                return (
                    False,
                    f"로컬 Ollama 서버({self.base_url})에 연결할 수 없습니다.\n"
                    f"`setup_env.ps1`을 실행하여 Ollama를 설치 및 시작해 주세요.",
                )
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()

        try:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            target_prefix = self.model.split(":")[0]
            if any(m == self.model or m.startswith(self.model) for m in models):
                return True, f"Ollama 연결 정상 (모델: {self.model}, Vulkan 가속 설정됨)"
            if any(m.startswith(target_prefix) for m in models):
                matched = next(m for m in models if m.startswith(target_prefix))
                self.model = matched
                return True, f"Ollama 연결 정상 (감지된 모델: {self.model}, Vulkan 가속 설정됨)"

            return (
                False,
                f"Ollama 서버는 실행 중이나 '{self.model}' 모델이 아직 설치되지 않았습니다.\n"
                f"GUI의 [모델 준비/점검] 버튼을 누르거나 `ollama pull {self.model}`을 실행해 주세요.",
            )
        except requests.RequestException as exc:
            return (
                False,
                f"로컬 Ollama 서버({self.base_url}) 응답 오류: {exc}",
            )

    def translate_segments(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str = "한국어",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        checkpoint_callback: Optional[Callable[[List[SubtitleSegment]], None]] = None,
    ) -> List[SubtitleSegment]:
        """전체 자막 세그먼트를 슬라이딩 윈도우 배치 방식으로 한국어 번역하고 체크포인트 및 텔레메트리를 기록합니다."""
        total = len(segments)
        if total == 0:
            return segments

        t_stage_start = time.perf_counter()
        total_load_ns = 0
        total_prompt_tokens = 0
        total_prompt_ns = 0
        total_eval_tokens = 0
        total_eval_ns = 0
        batches_count = 0
        skipped_batches = 0
        fallback_count = 0
        ps_info: Dict[str, Any] = {}

        src_lang_name = LANG_KO_NAMES.get(source_lang, source_lang)
        already_done = sum(1 for s in segments if s.translated_text)
        logger.info(
            "로컬 LLM 자막 번역 시작: 모델=%s, 원문=%s -> %s, 총 %d개 세그먼트 (기존 완료=%d, 배치=%d, 문맥=%d)",
            self.model, src_lang_name, target_lang, total, already_done, self.batch_size, self.context_size,
        )

        for start_idx in range(0, total, self.batch_size):
            if cancel_event and cancel_event.is_set():
                if checkpoint_callback:
                    checkpoint_callback(segments)
                done_now = sum(1 for s in segments if s.translated_text)
                raise JobCancelledError(
                    f"LLM 번역 중 사용자에 의해 작업이 즉시 중단되었습니다 ({done_now}/{total}문장 번역 저장됨)."
                )

            end_idx = min(start_idx + self.batch_size, total)
            current_batch = segments[start_idx:end_idx]

            # 이미 이전 실행에서 번역 완료된 배치는 건너뛰기 (Resume)
            if all(bool(seg.translated_text) for seg in current_batch):
                skipped_batches += 1
                if progress_callback:
                    pct = (end_idx / total) * 100.0
                    progress_callback(
                        pct,
                        f"[{end_idx}/{total} | 중간 생산물 복원] {current_batch[-1].translated_text[:32]}",
                    )
                continue

            ctx_start = max(0, start_idx - self.context_size)
            context_segments = segments[ctx_start:start_idx]

            batch_translations, stats = self._translate_batch(
                current_batch,
                context_segments,
                src_lang_name,
                target_lang,
                cancel_event=cancel_event,
            )
            batches_count += 1
            total_load_ns += stats.get("load_duration", 0)
            total_prompt_tokens += stats.get("prompt_eval_count", 0)
            total_prompt_ns += stats.get("prompt_eval_duration", 0)
            total_eval_tokens += stats.get("eval_count", 0)
            total_eval_ns += stats.get("eval_duration", 0)

            # 첫 번째 실제 배치가 메모리에 적재된 직후 /api/ps를 조회하여 실제 GPU/CPU 오프로딩 상태 확인
            if batches_count == 1:
                ps_info = query_ollama_processor_status(self.base_url, self.model)
                if ps_info.get("gpu_offload_pct", 0.0) == 0.0:
                    logger.warning(
                        "[LLM 가속 자동 복구] 기존 Ollama 서버가 내장 GPU 제외(CPU 단독) 모드로 실행 중입니다. "
                        "Intel Arc iGPU 가속(OLLAMA_VULKAN=1, OLLAMA_IGPU_ENABLE=1)을 적용하여 서버를 재시작합니다..."
                    )
                    if self._try_start_ollama_server(force_restart=True):
                        # 두 번째 배치에서 GPU 적재 상태를 다시 조회하도록 유도
                        ps_info = {}
                if ps_info:
                    logger.info(
                        "[LLM 하드웨어 가속 상태] 장치=%s | 상세=%s",
                        ps_info.get("accelerator"),
                        ps_info.get("processor_summary"),
                    )
            elif batches_count == 2 and not ps_info:
                ps_info = query_ollama_processor_status(self.base_url, self.model)
                logger.info(
                    "[LLM 하드웨어 가속 상태 (재시작 후)] 장치=%s | 상세=%s",
                    ps_info.get("accelerator"),
                    ps_info.get("processor_summary"),
                )

            for seg in current_batch:
                if seg.translated_text:
                    continue
                if cancel_event and cancel_event.is_set():
                    if checkpoint_callback:
                        checkpoint_callback(segments)
                    done_now = sum(1 for s in segments if s.translated_text)
                    raise JobCancelledError(
                        f"LLM 번역 중 사용자에 의해 작업이 즉시 중단되었습니다 ({done_now}/{total}문장 번역 저장됨)."
                    )

                translated = batch_translations.get(seg.id)
                if not translated:
                    fallback_count += 1
                    translated, fb_stats = self._translate_single_fallback(
                        seg.original_text, src_lang_name, target_lang, cancel_event=cancel_event
                    )
                    total_eval_tokens += fb_stats.get("eval_count", 0)
                    total_eval_ns += fb_stats.get("eval_duration", 0)
                seg.translated_text = translated.strip()

            # 배치 완료 즉시 작업 폴더에 체크포인트 저장
            if checkpoint_callback:
                checkpoint_callback(segments)

            if progress_callback:
                pct = (end_idx / total) * 100.0
                elapsed_now = time.perf_counter() - t_stage_start
                tok_s = (total_eval_tokens / (total_eval_ns / 1e9)) if total_eval_ns > 0 else 0.0
                accel_short = ps_info.get("accelerator", "LLM")
                sample = current_batch[-1].translated_text or ""
                progress_callback(
                    pct,
                    f"[{end_idx}/{total} | {accel_short} | {tok_s:.1f} tok/s | 경과 {elapsed_now:.0f}s] {sample[:32]}",
                )

        total_elapsed = time.perf_counter() - t_stage_start
        model_load_sec = total_load_ns / 1e9
        compute_sec = max(0.0, total_elapsed - model_load_sec)
        eval_tok_per_sec = (total_eval_tokens / (total_eval_ns / 1e9)) if total_eval_ns > 0 else 0.0
        prompt_tok_per_sec = (
            (total_prompt_tokens / (total_prompt_ns / 1e9)) if total_prompt_ns > 0 else 0.0
        )

        if not ps_info:
            ps_info = query_ollama_processor_status(self.base_url, self.model)

        accel_type = ps_info.get("accelerator", "CPU/GPU")
        proc_summary = ps_info.get("processor_summary", f"Ollama ({self.model})")

        self.last_telemetry = {
            "accelerator_type": accel_type,
            "backend_detail": f"{proc_summary} [model={self.model}, batch={self.batch_size}]",
            "elapsed_sec": round(total_elapsed, 2),
            "model_load_sec": round(model_load_sec, 2),
            "compute_sec": round(compute_sec, 2),
            "items_processed": total,
            "throughput_label": (
                f"생성 {eval_tok_per_sec:.1f} tok/s (프롬프트 {prompt_tok_per_sec:.1f} tok/s, "
                f"문장당 {total_elapsed / max(1, total):.2f}초)"
            ),
            "extra_metrics": {
                "model": self.model,
                "gpu_offload_pct": ps_info.get("gpu_offload_pct", 0.0),
                "size_total_mb": ps_info.get("size_total_mb", 0.0),
                "size_vram_mb": ps_info.get("size_vram_mb", 0.0),
                "prompt_tokens": total_prompt_tokens,
                "eval_tokens": total_eval_tokens,
                "eval_tok_per_sec": round(eval_tok_per_sec, 2),
                "prompt_tok_per_sec": round(prompt_tok_per_sec, 2),
                "batches_count": batches_count,
                "skipped_batches": skipped_batches,
                "fallback_count": fallback_count,
            },
        }

        return segments

    def _translate_batch(
        self,
        batch: List[SubtitleSegment],
        context_segments: List[SubtitleSegment],
        src_lang_name: str,
        target_lang: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[Dict[int, str], Dict[str, int]]:
        """배치 단위 JSON 구조화 번역을 스트리밍으로 수행하여 즉시 중단(cancel_event)을 지원합니다."""
        context_lines = []
        for c in context_segments:
            context_lines.append(
                f"- (원문) {c.original_text} -> (번역) {c.translated_text or ''}"
            )
        context_block = "\n".join(context_lines) if context_lines else "(이전 대사 없음)"

        input_items = [{"id": s.id, "source": s.original_text} for s in batch]
        extra_hint = f"\n추가 작품 배경 정보: {self.movie_context}" if self.movie_context else ""

        system_prompt = (
            f"당신은 {src_lang_name} 영화를 자연스러운 {target_lang}로 번역하는 전문 영상 번역가입니다.\n"
            "반드시 다음 원칙을 지키세요:\n"
            "1. 직역투를 피하고 영화 등장인물의 감정과 상황에 맞는 자연스러운 한국어 구어체로 번역하세요.\n"
            "2. 이전 대사 문맥을 참고하여 화자의 존댓말/반말 톤과 고유명사를 일관되게 유지하세요.\n"
            "3. 음성 인식(STT) 과정에서 발생한 경미한 동음이의어 오타가 있다면 문맥에 맞게 교정하여 번역하세요.\n"
            "4. 입력된 각 항목의 `id`를 절대 변경하거나 합치지 말고, 1:1로 대응하는 JSON 형식으로만 응답하세요.\n"
            '5. 응답 JSON 형식: {"translations": [{"id": 숫자, "translation": "한국어 번역문"}, ...]}'
            f"{extra_hint}"
        )

        user_prompt = (
            f"[직전 대사 문맥 (참고용, 번역 출력 제외)]\n{context_block}\n\n"
            f"[이번에 번역할 {src_lang_name} 대사 목록]\n"
            f"{json.dumps(input_items, ensure_ascii=False, indent=2)}\n\n"
            "위 대사 목록을 빠짐없이 번역하여 지정된 JSON 형식으로만 출력하세요."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096,
            },
        }

        chunks: List[str] = []
        stats: Dict[str, int] = {}

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_sec,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        resp.close()
                        raise JobCancelledError("LLM 배치 번역 중 사용자에 의해 작업이 즉시 중단되었습니다.")
                    if not raw_line:
                        continue
                    part = json.loads(raw_line.decode("utf-8", errors="replace"))
                    piece = part.get("message", {}).get("content", "")
                    if piece:
                        chunks.append(piece)
                    if part.get("done"):
                        stats = {
                            "load_duration": int(part.get("load_duration", 0)),
                            "prompt_eval_count": int(part.get("prompt_eval_count", 0)),
                            "prompt_eval_duration": int(part.get("prompt_eval_duration", 0)),
                            "eval_count": int(part.get("eval_count", 0)),
                            "eval_duration": int(part.get("eval_duration", 0)),
                        }
                        break

            content = "".join(chunks)
            return self._parse_json_response(content), stats
        except JobCancelledError:
            raise
        except Exception as exc:
            logger.warning("배치 번역 중 오류 발생 (개별 폴백 번역으로 전환): %s", exc)
            return {}, {}

    @staticmethod
    def _parse_json_response(content: str) -> Dict[int, str]:
        """LLM의 JSON 응답에서 id -> 번역문 딕셔너리를 안전하게 추출합니다."""
        result: Dict[int, str] = {}
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)

            items = []
            if isinstance(data, dict):
                for key in ("translations", "items", "data", "subtitles", "result"):
                    if isinstance(data.get(key), list):
                        items = data[key]
                        break
                if not items and "id" in data and ("translation" in data or "text" in data):
                    items = [data]
            elif isinstance(data, list):
                items = data

            for item in items:
                if not isinstance(item, dict):
                    continue
                seg_id = item.get("id")
                trans = (
                    item.get("translation")
                    or item.get("text")
                    or item.get("korean")
                    or item.get("target")
                )
                if seg_id is not None and trans:
                    result[int(seg_id)] = str(trans).strip()
        except Exception as exc:
            logger.debug("JSON 파싱 실패: %s | 원문: %s", exc, content[:200])

        return result

    def _translate_single_fallback(
        self,
        text: str,
        src_lang_name: str,
        target_lang: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, Dict[str, int]]:
        """배치 JSON에서 누락된 단일 문장을 개별 번역합니다."""
        if cancel_event and cancel_event.is_set():
            raise JobCancelledError("단일 문장 번역 중 작업이 중단되었습니다.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"당신은 영화 자막 번역가입니다. 주어진 {src_lang_name} 대사를 자연스러운 "
                        f"{target_lang} 구어체 자막으로만 번역하세요. 부연 설명 없이 번역문만 출력하세요."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
            out = body.get("message", {}).get("content", "").strip()
            stats = {
                "eval_count": int(body.get("eval_count", 0)),
                "eval_duration": int(body.get("eval_duration", 0)),
            }
            return (out if out else text), stats
        except Exception as exc:
            logger.error("단일 문장 번역 실패 (원문 유지): %s", exc)
            return text, {}
