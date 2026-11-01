"""Faster-Whisper 및 Silero VAD를 이용한 다국어(중국어/일본어/영어) 음성 인식 모듈.

특징:
- INT8 양자화 및 멀티스레드(12 스레드) 추론으로 빠른 처리
- Silero VAD 필터로 영화 배경음악(BGM)/효과음 구간의 환각(Hallucination) 차단
- CPU / GPU / NPU 사용 백엔드 및 모델 로드/추론 소요 시간(Telemetry) 정밀 측정
- 작업 중 즉시 중단(`cancel_event`) 및 중간 세그먼트 저장(`stt_segments.partial.json`)을 통한
  오디오 오프셋 이어하기(Resume) 지원
- 음성 인식 완료 직후 모델을 메모리에서 명시적으로 해제(Sequential Offloading)
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import av
import numpy as np
from faster_whisper import WhisperModel

from pipeline.models import SubtitleSegment
from pipeline.workspace import JobCancelledError

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
    "es": "스페인어 (Spanish)",
    "de": "독일어 (German)",
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
    telemetry_out: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[threading.Event] = None,
    initial_segments: Optional[List[SubtitleSegment]] = None,
    initial_lang: Optional[str] = None,
    partial_save_callback: Optional[Callable[[List[SubtitleSegment], str], None]] = None,
) -> Tuple[List[SubtitleSegment], str]:
    """16kHz WAV 파일에서 타임스탬프가 정렬된 원문 자막 세그먼트를 추출하고 성능 지표를 기록합니다.

    이전에 부분 저장된 `initial_segments`가 주어지면 마지막 타임스탬프 이후 오디오 파형부터 이어서 인식합니다.
    """
    if cancel_event and cancel_event.is_set():
        raise JobCancelledError("음성 인식 시작 전 작업이 중단되었습니다.")

    t_stage_start = time.perf_counter()

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
        whisper_lang = initial_lang if initial_lang else None
    else:
        whisper_lang = language.lower()

    accel_label = "GPU (CUDA)" if device.lower() == "cuda" else "CPU"
    vad_provider = "CPUExecutionProvider" if vad_filter else "Disabled"
    backend_desc = (
        f"Faster-Whisper (CTranslate2: device={device.upper()}, compute={compute_type}, "
        f"threads={cpu_threads}) + Silero VAD ({vad_provider})"
    )

    logger.info(
        "[STT 하드웨어/백엔드] 연산장치=%s | 상세=%s | 모델=%s",
        accel_label, backend_desc, model_size,
    )

    t_load_start = time.perf_counter()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    model_load_sec = time.perf_counter() - t_load_start
    logger.info("[STT 타이밍] Whisper 모델 로드 완료: %.2f초 소요", model_load_sec)

    segments_out: List[SubtitleSegment] = list(initial_segments) if initial_segments else []
    detected_lang = initial_lang or whisper_lang or "en"
    resume_offset_sec = float(segments_out[-1].end) if segments_out else 0.0
    total_duration = 0.0
    t_infer_start = time.perf_counter()

    try:
        full_waveform = load_wav_as_float32(wav_path)
        total_duration = len(full_waveform) / 16000.0

        if resume_offset_sec > 0.0 and resume_offset_sec < total_duration - 1.0:
            start_sample = int(resume_offset_sec * 16000)
            audio_waveform = full_waveform[start_sample:]
            logger.info(
                "[STT 이어하기] 기존 %d개 문장 복원 완료 -> %.1f초 지점부터 이어서 음성 인식을 진행합니다.",
                len(segments_out), resume_offset_sec,
            )
        else:
            resume_offset_sec = 0.0
            audio_waveform = full_waveform

        segments_gen, info = model.transcribe(
            audio_waveform,
            language=whisper_lang,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,  # 영화 BGM 구간에서 동일 대사 무한 반복 환각 방지
        )

        detected_lang = info.language or detected_lang
        lang_label = LANGUAGE_NAME_MAP.get(detected_lang, detected_lang)
        logger.info(
            "음성 언어 감지 완료: %s (확률: %.1f%%), 전체 오디오 길이: %.1f초",
            lang_label, info.language_probability * 100, total_duration,
        )

        seg_id = len(segments_out) + 1
        new_count = 0

        for seg in segments_gen:
            if cancel_event and cancel_event.is_set():
                if partial_save_callback and segments_out:
                    partial_save_callback(segments_out, detected_lang)
                raise JobCancelledError(
                    f"음성 인식 중 사용자에 의해 작업이 즉시 중단되었습니다 (현재까지 {len(segments_out)}문장 중간 저장됨)."
                )

            text = seg.text.strip()
            if not text:
                continue

            abs_start = float(seg.start) + resume_offset_sec
            abs_end = float(seg.end) + resume_offset_sec

            segments_out.append(
                SubtitleSegment(
                    id=seg_id,
                    start=abs_start,
                    end=abs_end,
                    original_text=text,
                )
            )
            seg_id += 1
            new_count += 1

            # 15문장마다 작업 폴더에 중간 산출물(stt_segments.partial.json) 자동 백업
            if partial_save_callback and new_count % 15 == 0:
                partial_save_callback(segments_out, detected_lang)

            if progress_callback and total_duration > 0:
                pct = min(100.0, (abs_end / total_duration) * 100.0)
                elapsed_now = max(0.001, time.perf_counter() - t_infer_start)
                processed_audio_sec = max(0.1, abs_end - resume_offset_sec)
                speed_x = processed_audio_sec / elapsed_now
                progress_callback(
                    pct,
                    f"[{accel_label} | {abs_end:.0f}s/{total_duration:.0f}s | 경과 {elapsed_now:.0f}s ({speed_x:.1f}x배속)] {text[:35]}",
                )

        infer_sec = time.perf_counter() - t_infer_start
        total_sec = time.perf_counter() - t_stage_start
        processed_duration = max(0.0, total_duration - resume_offset_sec)
        speed_factor = (processed_duration / infer_sec) if infer_sec > 0 else 0.0

        if progress_callback:
            progress_callback(
                100.0,
                f"음성 인식 완료: 총 {len(segments_out)}문장 | 장치={accel_label} | 소요={total_sec:.1f}초 ({speed_factor:.2f}x 실시간 배속)",
            )

        if telemetry_out is not None:
            telemetry_out.update(
                {
                    "accelerator_type": accel_label,
                    "backend_detail": backend_desc,
                    "elapsed_sec": round(total_sec, 2),
                    "model_load_sec": round(model_load_sec, 2),
                    "compute_sec": round(infer_sec, 2),
                    "items_processed": len(segments_out),
                    "throughput_label": f"{speed_factor:.2f}x 실시간 배속 (오디오 {processed_duration:.0f}초 / 추론 {infer_sec:.1f}초)",
                    "extra_metrics": {
                        "model_size": model_size,
                        "device": device,
                        "compute_type": compute_type,
                        "cpu_threads": cpu_threads,
                        "vad_provider": vad_provider,
                        "audio_duration_sec": round(total_duration, 2),
                        "resumed_from_sec": round(resume_offset_sec, 2),
                        "realtime_speed_x": round(speed_factor, 2),
                        "detected_language": detected_lang,
                    },
                }
            )

    finally:
        # Sequential Offloading: LLM 구동 전 Whisper 모델 메모리 즉시 반환
        del model
        gc.collect()
        logger.info("Whisper 모델 메모리 해제 완료")

    return segments_out, detected_lang
