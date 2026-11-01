"""전체 자막 생성 파이프라인(오디오 추출 -> STT -> 로컬 LLM 번역 -> 자막 파일 생성) 실행 오케스트레이터."""

from __future__ import annotations

import copy
import logging
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from pipeline.extractor import SUPPORTED_VIDEO_EXTENSIONS, extract_audio_to_wav
from pipeline.formatter import write_ass, write_srt
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
    """단일 영화 파일에 대해 자막 생성 전 과정을 수행하고 생성된 자막 파일 경로 목록을 반환합니다.

    status_callback(stage_name, percentage, detail_message)
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"영화 파일을 찾을 수 없습니다: {video_path}")

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

        # 1. 오디오 추출 및 코덱 정규화 (16kHz Mono WAV)
        if status_callback:
            status_callback("1/4 오디오 추출", 0.0, f"오디오 스트림 분석 및 디코딩 중: {video_path.name}")

        _, selected_track = extract_audio_to_wav(
            video_path=video_path,
            output_wav_path=wav_path,
            preferred_language=preferred_lang,
            track_index=track_index,
            progress_callback=(
                (lambda p: status_callback("1/4 오디오 추출", p, f"오디오 디코딩 중 ({p:.0f}%)"))
                if status_callback else None
            ),
        )
        if status_callback:
            status_callback("1/4 오디오 추출", 100.0, f"완료: {selected_track.display_label()}")

        # 2. 다국어 음성 인식 (Faster-Whisper + Silero VAD)
        if status_callback:
            status_callback(
                "2/4 음성 인식 (STT)",
                0.0,
                f"Whisper ({stt_cfg.get('model_size', 'large-v3-turbo')}) 모델 로드 및 음성 인식 시작...",
            )

        segments, detected_lang = transcribe_audio(
            wav_path=wav_path,
            stt_config=stt_cfg,
            progress_callback=(
                (lambda p, msg: status_callback("2/4 음성 인식 (STT)", p, msg))
                if status_callback else None
            ),
        )

    if not segments:
        logger.warning("인식된 대사가 없습니다: %s", video_path)
        return generated_files

    # 원문 자막 별도 저장 옵션
    if save_orig:
        orig_srt_path = base_target.with_suffix(f".{detected_lang}.orig.srt")
        write_srt(segments, orig_srt_path, bilingual=False, use_original_only=True)
        generated_files.append(orig_srt_path)

    # 3. 로컬 LLM 문맥 번역 (한국어 원문이거나 번역 생략 옵션인 경우 건너뜀)
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
                    f"{translator.model} 모델로 문맥 기반 한국어 번역 중...",
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

    # 4. 최종 자막 파일(.srt / .ass) 포맷팅 및 저장
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

    if status_callback:
        status_callback(
            "완료",
            100.0,
            f"생성 완료: {', '.join(p.name for p in generated_files)}",
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
