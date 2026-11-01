"""전체 자막 생성 파이프라인(오디오 추출 -> STT -> 로컬 LLM 번역 -> 자막 파일 생성) 실행 오케스트레이터.

각 단계별 연산 장치(CPU / GPU / NPU) 및 소요 시간(Telemetry)을 기록하여
GUI 로그창, CLI 콘솔, 그리고 `logs/perf_report.log` 및 `logs/perf_history.jsonl`에 자동 저장합니다.
"""

from __future__ import annotations

import copy
import datetime
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from pipeline.extractor import (
    SUPPORTED_VIDEO_EXTENSIONS,
    extract_audio_to_wav,
    inspect_video_metadata,
)
from pipeline.formatter import write_ass, write_srt
from pipeline.telemetry import (
    JobTelemetryReport,
    StageMetric,
    detect_hardware_inventory,
    save_telemetry_report,
)
from pipeline.transcriber import transcribe_audio
from pipeline.translator import LocalLLMTranslator

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """YAML 설정 파일을 로드합니다."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: Optional[Path] = None) -> None:
    """변경된 설정을 YAML 파일에 저장합니다."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def process_video_file(
    video_path: Path,
    config: Dict,
    track_index: Optional[int] = None,
    language_override: Optional[str] = None,
    output_dir: Optional[Path] = None,
    skip_translation: bool = False,
    status_callback: Optional[Callable[[str, float, str], None]] = None,
) -> List[Path]:
    """단일 영화 파일에 대해 자막 생성 전 과정을 수행하고 성능 리포트를 기록한 뒤 생성된 파일 경로를 반환합니다."""
    t_job_start = time.perf_counter()
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"영화 파일을 찾을 수 없습니다: {video_path}")

    hw_inv = detect_hardware_inventory()
    media_info = inspect_video_metadata(video_path)
    stage_metrics: List[StageMetric] = []

    stt_cfg = copy.deepcopy(config.get("stt", {}))
    llm_cfg = copy.deepcopy(config.get("llm", {}))
    sub_cfg = copy.deepcopy(config.get("subtitle", {}))

    if language_override and language_override != "default":
        stt_cfg["language"] = language_override

    preferred_lang = stt_cfg.get("language", "auto")
    bilingual = bool(sub_cfg.get("bilingual", False))
    formats = sub_cfg.get("formats", ["srt"])
    save_orig = bool(sub_cfg.get("save_original_srt", True))
    target_lang = sub_cfg.get("target_language", "한국어")

    target_dir = Path(output_dir).resolve() if output_dir else video_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    base_target = target_dir / video_path.name

    generated_files: List[Path] = []

    with tempfile.TemporaryDirectory(prefix="subgen_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{video_path.stem}_16k.wav"

        # =====================================================================
        # 1. 오디오 추출 및 코덱 정규화 (16kHz Mono WAV)
        # =====================================================================
        t_ext_start = time.perf_counter()
        ext_backend = "System FFmpeg CLI" if shutil.which("ffmpeg") else "PyAV (내장 FFmpeg C-API)"
        if status_callback:
            status_callback(
                "1/4 오디오 추출",
                0.0,
                f"[장치: CPU | 백엔드: {ext_backend}] 오디오 스트림 디코딩 시작: {video_path.name}",
            )

        _, selected_track = extract_audio_to_wav(
            video_path=video_path,
            output_wav_path=wav_path,
            preferred_language=preferred_lang,
            track_index=track_index,
            progress_callback=(
                (lambda p: status_callback("1/4 오디오 추출", p, f"[CPU 디코딩] {p:.0f}% 완료"))
                if status_callback else None
            ),
        )
        ext_elapsed = time.perf_counter() - t_ext_start
        wav_size_mb = wav_path.stat().st_size / (1024 * 1024) if wav_path.exists() else 0.0
        ext_rtf = (media_info.duration_sec / ext_elapsed) if ext_elapsed > 0 and media_info.duration_sec > 0 else 0.0

        stage_metrics.append(
            StageMetric(
                stage_name="1/4 오디오 추출",
                accelerator_type="CPU",
                backend_detail=f"{ext_backend} ({selected_track.codec_name.upper()} -> 16kHz Mono PCM, {wav_size_mb:.1f}MB)",
                elapsed_sec=round(ext_elapsed, 2),
                compute_sec=round(ext_elapsed, 2),
                throughput_label=f"{ext_rtf:.1f}x 실시간 배속" if ext_rtf > 0 else f"{wav_size_mb:.1f}MB 추출",
            )
        )

        if status_callback:
            status_callback(
                "1/4 오디오 추출",
                100.0,
                f"완료 ({ext_elapsed:.2f}초, [CPU] {ext_backend}): {selected_track.display_label()}",
            )

        # =====================================================================
        # 2. 다국어 음성 인식 (Faster-Whisper + Silero VAD)
        # =====================================================================
        stt_dev = stt_cfg.get("device", "cpu").upper()
        stt_threads = stt_cfg.get("cpu_threads", 12)
        stt_compute = stt_cfg.get("compute_type", "int8")
        if status_callback:
            status_callback(
                "2/4 음성 인식 (STT)",
                0.0,
                f"[장치: {stt_dev} ({stt_compute}, {stt_threads}스레드)] Whisper ({stt_cfg.get('model_size', 'large-v3-turbo')}) 로드 중...",
            )

        stt_telemetry: Dict[str, Any] = {}
        segments, detected_lang = transcribe_audio(
            wav_path=wav_path,
            stt_config=stt_cfg,
            progress_callback=(
                (lambda p, msg: status_callback("2/4 음성 인식 (STT)", p, msg))
                if status_callback else None
            ),
            telemetry_out=stt_telemetry,
        )

        stage_metrics.append(
            StageMetric(
                stage_name="2/4 음성 인식 (STT)",
                accelerator_type=stt_telemetry.get("accelerator_type", "CPU"),
                backend_detail=stt_telemetry.get("backend_detail", "Faster-Whisper"),
                elapsed_sec=stt_telemetry.get("elapsed_sec", 0.0),
                model_load_sec=stt_telemetry.get("model_load_sec", 0.0),
                compute_sec=stt_telemetry.get("compute_sec", 0.0),
                items_processed=stt_telemetry.get("items_processed", len(segments)),
                throughput_label=stt_telemetry.get("throughput_label", ""),
                extra_metrics=stt_telemetry.get("extra_metrics", {}),
            )
        )

    if not segments:
        logger.warning("인식된 대사가 없습니다: %s", video_path)
        return generated_files

    # 원문 자막 별도 저장 옵션
    if save_orig:
        orig_srt_path = base_target.with_suffix(f".{detected_lang}.orig.srt")
        write_srt(segments, orig_srt_path, bilingual=False, use_original_only=True)
        generated_files.append(orig_srt_path)

    # =====================================================================
    # 3. 로컬 LLM 문맥 번역 (한국어 원문이거나 번역 생략 옵션인 경우 건너뜀)
    # =====================================================================
    if not skip_translation and detected_lang != "ko":
        translator = LocalLLMTranslator(llm_cfg)
        ok, msg = translator.check_server_and_model()
        if not ok:
            logger.warning("%s\n-> 원문 자막만 우선 생성합니다.", msg)
            if status_callback:
                status_callback("3/4 LLM 번역 건너뜀", 100.0, msg)
        else:
            if status_callback:
                status_callback(
                    "3/4 로컬 LLM 번역",
                    0.0,
                    f"{translator.model} 모델로 문맥 기반 한국어 번역 시작 (총 {len(segments)}문장)...",
                )
            segments = translator.translate_segments(
                segments=segments,
                source_lang=detected_lang,
                target_lang=target_lang,
                progress_callback=(
                    (lambda p, detail: status_callback("3/4 로컬 LLM 번역", p, detail))
                    if status_callback else None
                ),
            )
            llm_tel = translator.last_telemetry
            if llm_tel:
                stage_metrics.append(
                    StageMetric(
                        stage_name="3/4 로컬 LLM 번역",
                        accelerator_type=llm_tel.get("accelerator_type", "CPU/GPU"),
                        backend_detail=llm_tel.get("backend_detail", f"Ollama ({translator.model})"),
                        elapsed_sec=llm_tel.get("elapsed_sec", 0.0),
                        model_load_sec=llm_tel.get("model_load_sec", 0.0),
                        compute_sec=llm_tel.get("compute_sec", 0.0),
                        items_processed=llm_tel.get("items_processed", len(segments)),
                        throughput_label=llm_tel.get("throughput_label", ""),
                        extra_metrics=llm_tel.get("extra_metrics", {}),
                    )
                )
                if status_callback:
                    status_callback(
                        "3/4 로컬 LLM 번역",
                        100.0,
                        f"번역 완료 ({llm_tel.get('elapsed_sec', 0.0):.1f}초 | 장치={llm_tel.get('accelerator_type')} | {llm_tel.get('throughput_label')})",
                    )

    # =====================================================================
    # 4. 최종 자막 파일(.srt / .ass) 포맷팅 및 저장
    # =====================================================================
    t_fmt_start = time.perf_counter()
    if status_callback:
        status_callback("4/4 자막 파일 저장", 50.0, "타임스탬프 보정 및 자막 파일 기록 중...")

    if "srt" in formats:
        srt_path = base_target.with_suffix(".ko.srt" if not skip_translation else ".srt")
        write_srt(segments, srt_path, bilingual=bilingual, use_original_only=skip_translation)
        generated_files.append(srt_path)

    if "ass" in formats:
        ass_path = base_target.with_suffix(".ko.ass" if not skip_translation else ".ass")
        write_ass(segments, ass_path, bilingual=bilingual)
        generated_files.append(ass_path)

    fmt_elapsed = time.perf_counter() - t_fmt_start
    stage_metrics.append(
        StageMetric(
            stage_name="4/4 자막 파일 저장",
            accelerator_type="CPU + Disk I/O",
            backend_detail=f"SRT/ASS 포맷팅 ({', '.join(p.name for p in generated_files)})",
            elapsed_sec=round(fmt_elapsed, 3),
            compute_sec=round(fmt_elapsed, 3),
            items_processed=len(segments),
        )
    )

    total_job_elapsed = time.perf_counter() - t_job_start

    # 자동 병목 진단 및 향후 CPU / GPU / NPU 최적화 제안 생성
    opt_notes: List[str] = []
    for sm in stage_metrics:
        if sm.stage_name == "2/4 음성 인식 (STT)" and sm.accelerator_type == "CPU":
            opt_notes.append(
                "STT(음성 인식) 단계가 현재 CTranslate2 CPU(INT8)로 동작했습니다. "
                "Intel Arc GPU(Vulkan `whisper.cpp`) 또는 Intel NPU/iGPU(`OpenVINO Whisper`) 백엔드로 전환 시 "
                "CPU 점유율을 낮추고 STT 속도를 약 2~3배 단축할 수 있습니다."
            )
        if sm.stage_name == "3/4 로컬 LLM 번역":
            gpu_pct = sm.extra_metrics.get("gpu_offload_pct", 0.0)
            if gpu_pct == 0.0:
                opt_notes.append(
                    "LLM 번역 단계에서 Ollama가 GPU VRAM 오프로딩 없이 100% CPU로 구동되었습니다. "
                    "윈도우 트레이의 Ollama 앱을 종료 후 `OLLAMA_VULKAN=1` 환경에서 재시작하거나 "
                    "Intel IPEX-LLM(SYCL) 백엔드를 적용하면 GPU 가속으로 토큰 속도를 2배 이상 높일 수 있습니다."
                )
            else:
                opt_notes.append(
                    f"LLM 번역 단계에서 Intel Arc GPU가 {gpu_pct:.0f}% 가속 참여했습니다."
                )

    report = JobTelemetryReport(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        video_name=video_path.name,
        video_codec=media_info.video_codec,
        audio_codec=selected_track.codec_name.upper(),
        video_duration_sec=round(media_info.duration_sec, 2),
        detected_language=detected_lang,
        hardware=hw_inv,
        stages=stage_metrics,
        total_elapsed_sec=round(total_job_elapsed, 2),
        optimization_notes=opt_notes,
    )

    report_path = save_telemetry_report(report)
    summary_str = report.format_summary_text()
    logger.info("%s\n[성능 로그 저장 위치] %s", summary_str, report_path)

    if status_callback:
        status_callback(
            "완료",
            100.0,
            f"자막 생성 완료 (총 {total_job_elapsed/60:.1f}분 소요): {', '.join(p.name for p in generated_files)}\n{summary_str}",
        )

    return generated_files


def collect_video_files(target_path: Path) -> List[Path]:
    """단일 파일 또는 폴더 내 모든 지원 영화 파일을 탐색합니다."""
    target_path = Path(target_path).resolve()
    if target_path.is_file():
        return [target_path]
    if target_path.is_dir():
        files = [
            p for p in sorted(target_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        return files
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target_path}")
