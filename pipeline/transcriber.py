"""Faster-Whisper 및 Silero VAD를 이용한 다국어(중국어/일본어/영어) 음성 인식 모듈.

Intel Core Ultra 7 155H + 32GB RAM 환경에 맞춰:
- INT8 양자화 및 멀티스레드(12 스레드) 추론으로 빠른 처리
- Silero VAD 필터로 영화 배경음악(BGM)/효과음 구간의 환각(Hallucination) 차단
- 음성 인식 완료 직후 모델을 메모리에서 명시적으로 해제(Sequential Offloading)하여
  다음 단계인 로컬 LLM(Qwen2.5)이 메모리를 온전히 사용할 수 있도록 관리합니다.
"""

from __future__ import annotations

import gc
import logging
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import av
import numpy as np
from faster_whisper import WhisperModel

from pipeline.models import SubtitleSegment

logger = logging.getLogger(__name__)

# PyAV 19+에서 제거된 metadata_errors 인자를 faster-whisper가 호출할 때 발생하는 호환성 오류 방어 패치
_orig_av_open = av.open


def _compat_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _compat_av_open

LANGUAGE_NAME_MAP = {
    "zh": "중국어 (Chinese)",
    "ja": "일본어 (Japanese)",
    "en": "영어 (English)",
    "ko": "한국어 (Korean)",
    "fr": "프랑스어 (French)",
}


def load_wav_as_float32(wav_path: Path) -> np.ndarray:
    """16kHz 16-bit Mono PCM WAV 파일을 Whisper 입력용 1D float32 배열(-1.0 ~ 1.0)로 즉시 로드합니다."""
    with wave.open(str(wav_path), "rb") as wf:
        num_frames = wf.getnframes()
        pcm_bytes = wf.readframes(num_frames)
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_audio(
    wav_path: Path,
    stt_config: Dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[SubtitleSegment], str]:
    """16kHz WAV 파일에서 타임스탬프가 정렬된 원문 자막 세그먼트를 추출합니다.

    Returns:
        (세그먼트 리스트, 감지된 언어 코드)
    """
    model_size = stt_config.get("model_size", "large-v3-turbo")
    device = stt_config.get("device", "cpu")
    compute_type = stt_config.get("compute_type", "int8")
    cpu_threads = int(stt_config.get("cpu_threads", 12))
    language = stt_config.get("language", "auto")
    beam_size = int(stt_config.get("beam_size", 5))
    vad_filter = bool(stt_config.get("vad_filter", True))
    vad_parameters = stt_config.get(
        "vad_parameters",
        {"min_silence_duration_ms": 500, "speech_pad_ms": 250},
    )

    if language == "auto" or not language:
        whisper_lang = None
    else:
        whisper_lang = language.lower()

    logger.info(
        "Whisper 모델 로드 중: model=%s, device=%s, compute_type=%s, threads=%d",
        model_size, device, compute_type, cpu_threads,
    )

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )

    segments_out: List[SubtitleSegment] = []
    detected_lang = whisper_lang or "en"

    try:
        # PyAV 재디코딩을 거치지 않고 16kHz float32 파형 배열을 직접 전달하여 속도 향상 및 호환성 보장
        audio_waveform = load_wav_as_float32(wav_path)

        segments_gen, info = model.transcribe(
            audio_waveform,
            language=whisper_lang,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,  # 영화 BGM 구간에서 동일 대사 무한 반복 환각 방지
        )

        detected_lang = info.language
        total_duration = info.duration or (len(audio_waveform) / 16000.0)
        lang_label = LANGUAGE_NAME_MAP.get(detected_lang, detected_lang)
        logger.info(
            "음성 언어 감지 완료: %s (확률: %.1f%%), 전체 길이: %.1f초",
            lang_label, info.language_probability * 100, total_duration,
        )

        seg_id = 1
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            segments_out.append(
                SubtitleSegment(
                    id=seg_id,
                    start=float(seg.start),
                    end=float(seg.end),
                    original_text=text,
                )
            )
            seg_id += 1

            if progress_callback and total_duration > 0:
                pct = min(100.0, (seg.end / total_duration) * 100.0)
                progress_callback(pct, f"[{seg.end:.1f}s / {total_duration:.1f}s] {text[:40]}")

        if progress_callback:
            progress_callback(100.0, f"음성 인식 완료 (총 {len(segments_out)}개 대사)")

    finally:
        # Sequential Offloading: LLM 구동 전 Whisper 모델 메모리 즉시 반환
        del model
        gc.collect()
        logger.info("Whisper 모델 메모리 해제 완료 (LLM 번역 준비)")

    return segments_out, detected_lang
