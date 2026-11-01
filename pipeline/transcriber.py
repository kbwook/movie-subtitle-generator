"""Intel OpenVINO (Intel Arc GPU / Intel AI Boost NPU) 및 Faster-Whisper (CPU) 통합 다국어 음성 인식 모듈.

지원 연산 장치 (`stt_config["device"]`):
- "GPU"    : Intel(R) Arc(TM) Graphics 내장 GPU 가속 (`openvino_genai.WhisperPipeline` + Silero VAD)
- "NPU"    : Intel(R) AI Boost NPU 신경망 가속기 (`openvino_genai.WhisperPipeline` + Silero VAD)
- "CPU-OV" : OpenVINO CPU 가속 모드
- "CPU"    : Faster-Whisper (CTranslate2 INT8 멀티스레드 CPU)

특징:
- Silero VAD 필터로 영화 배경음악(BGM)/무음 구간을 사전 제거하여 환각(Hallucination) 원천 차단 및 추론 시간 단축
- CPU / GPU / NPU 사용 백엔드 및 모델 로드/추론 소요 시간(Telemetry) 정밀 측정
- 작업 중 즉시 중단(`cancel_event`) 및 중간 세그먼트 저장(`stt_segments.partial.json`)을 통한 이어하기(Resume) 지원
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
from faster_whisper.vad import VadOptions, get_speech_timestamps
from huggingface_hub import snapshot_download

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

# 모델 크기별 공식 OpenVINO IR 모델 저장소 매핑 (GPU / NPU 가속용)
OPENVINO_MODEL_MAP = {
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "large-v3-turbo-fp16": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "base": "OpenVINO/whisper-base-fp16-ov",
    "tiny": "OpenVINO/whisper-tiny-fp16-ov",
}


def load_wav_as_float32(wav_path: Path) -> np.ndarray:
    """16kHz 16-bit Mono PCM WAV 파일을 Whisper 입력용 1D float32 배열(-1.0 ~ 1.0)로 즉시 로드합니다."""
    with wave.open(str(wav_path), "rb") as wf:
        num_frames = wf.getnframes()
        pcm_bytes = wf.readframes(num_frames)
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _build_speech_windows(
    audio_waveform: np.ndarray,
    vad_filter: bool,
    vad_parameters: Dict[str, Any],
    max_window_sec: float = 28.0,
    max_merge_gap_sec: float = 1.5,
) -> List[Tuple[int, int]]:
    """Silero VAD를 이용해 발화 구간만 검출한 뒤 Whisper 입력에 최적인 <=28초 윈도우(샘플 인덱스 쌍)로 병합합니다."""
    total_samples = len(audio_waveform)
    max_win_samples = int(max_window_sec * 16000)
    max_gap_samples = int(max_merge_gap_sec * 16000)

    if not vad_filter or total_samples < 16000:
        return [
            (start, min(start + max_win_samples, total_samples))
            for start in range(0, total_samples, max_win_samples)
        ]

    vad_opts = VadOptions(
        min_silence_duration_ms=int(vad_parameters.get("min_silence_duration_ms", 500)),
        speech_pad_ms=int(vad_parameters.get("speech_pad_ms", 250)),
    )
    raw_chunks = get_speech_timestamps(audio_waveform, vad_opts)
    if not raw_chunks:
        return []

    windows: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for ch in raw_chunks:
        s = int(ch["start"])
        e = int(ch["end"])

        # 단일 발화 구간이 28초를 초과하면 28초 단위로 분할
        while (e - s) > max_win_samples:
            if cur_start is not None and cur_end is not None:
                windows.append((cur_start, cur_end))
                cur_start, cur_end = None, None
            windows.append((s, s + max_win_samples))
            s += max_win_samples

        if cur_start is None or cur_end is None:
            cur_start, cur_end = s, e
        else:
            if (s - cur_end) <= max_gap_samples and (e - cur_start) <= max_win_samples:
                cur_end = e
            else:
                windows.append((cur_start, cur_end))
                cur_start, cur_end = s, e

    if cur_start is not None and cur_end is not None:
        windows.append((cur_start, cur_end))

    return windows


def _transcribe_with_openvino(
    wav_path: Path,
    stt_config: Dict,
    ov_device: str,
    progress_callback: Optional[Callable[[float, str], None]],
    telemetry_out: Optional[Dict[str, Any]],
    cancel_event: Optional[threading.Event],
    initial_segments: Optional[List[SubtitleSegment]],
    initial_lang: Optional[str],
    partial_save_callback: Optional[Callable[[List[SubtitleSegment], str], None]],
) -> Tuple[List[SubtitleSegment], str]:
    """OpenVINO WhisperPipeline을 이용해 Intel Arc GPU 또는 Intel AI Boost NPU에서 고속 음성 인식을 수행합니다."""
    import openvino_genai as ov_genai

    t_stage_start = time.perf_counter()
    model_size = stt_config.get("model_size", "large-v3-turbo")
    language = stt_config.get("language", "auto")
    vad_filter = bool(stt_config.get("vad_filter", True))
    vad_parameters = stt_config.get(
        "vad_parameters",
        {"min_silence_duration_ms": 500, "speech_pad_ms": 250},
    )

    if language == "auto" or not language:
        whisper_lang = initial_lang if initial_lang else None
    else:
        whisper_lang = language.lower()

    hf_repo_id = OPENVINO_MODEL_MAP.get(model_size, model_size)
    if "/" not in hf_repo_id:
        hf_repo_id = OPENVINO_MODEL_MAP["large-v3-turbo"]

    device_Names = {
        "GPU": "GPU (Intel Arc Graphics)",
        "NPU": "NPU (Intel AI Boost)",
        "CPU": "CPU (OpenVINO)",
    }
    accel_label = device_Names.get(ov_device, f"OpenVINO ({ov_device})")
    cache_dir = Path(__file__).resolve().parent.parent / "logs" / "ov_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(
            0.0,
            f"[{accel_label}] OpenVINO 모델({hf_repo_id}) 준비 및 커널 컴파일 중...",
        )

    t_load_start = time.perf_counter()
    model_dir = snapshot_download(hf_repo_id)

    actual_ov_device = ov_device
    try:
        pipe = ov_genai.WhisperPipeline(str(model_dir), actual_ov_device, CACHE_DIR=str(cache_dir))
    except Exception as exc:
        if actual_ov_device == "NPU":
            logger.warning(
                "NPU에서 해당 Whisper 모델 컴파일 제한 감지 (%s) -> Intel Arc GPU로 자동 전환합니다.", exc
            )
            actual_ov_device = "GPU"
            accel_label = "GPU (Intel Arc Graphics - NPU 폴백)"
            pipe = ov_genai.WhisperPipeline(str(model_dir), actual_ov_device, CACHE_DIR=str(cache_dir))
        else:
            raise

    model_load_sec = time.perf_counter() - t_load_start
    backend_desc = f"OpenVINO WhisperPipeline (device={actual_ov_device}, model={hf_repo_id}) + Silero VAD"
    logger.info(
        "[STT OpenVINO 가속] 연산장치=%s | 상세=%s | 모델 로드=%.2f초",
        accel_label, backend_desc, model_load_sec,
    )

    segments_out: List[SubtitleSegment] = list(initial_segments) if initial_segments else []
    detected_lang = initial_lang or whisper_lang or "en"
    resume_offset_sec = float(segments_out[-1].end) if segments_out else 0.0
    t_infer_start = time.perf_counter()

    try:
        full_waveform = load_wav_as_float32(wav_path)
        total_duration = len(full_waveform) / 16000.0

        if resume_offset_sec > 0.0 and resume_offset_sec < total_duration - 1.0:
            start_sample = int(resume_offset_sec * 16000)
            audio_waveform = full_waveform[start_sample:]
            logger.info(
                "[STT 이어하기] 기존 %d개 문장 복원 -> %.1f초 지점부터 %s 가속 인식을 재개합니다.",
                len(segments_out), resume_offset_sec, accel_label,
            )
        else:
            resume_offset_sec = 0.0
            audio_waveform = full_waveform

        windows = _build_speech_windows(audio_waveform, vad_filter, vad_parameters)
        logger.info(
            "[VAD 발화 구간 검출 완료] 총 %d개 발화 윈도우 생성 (전체 오디오 %.1f초)",
            len(windows), total_duration,
        )

        seg_id = len(segments_out) + 1
        new_count = 0
        lang_locked = whisper_lang

        for win_idx, (w_start, w_end) in enumerate(windows, start=1):
            if cancel_event and cancel_event.is_set():
                if partial_save_callback and segments_out:
                    partial_save_callback(segments_out, detected_lang)
                raise JobCancelledError(
                    f"음성 인식 중 사용자에 의해 작업이 즉시 중단되었습니다 (현재까지 {len(segments_out)}문장 중간 저장됨)."
                )

            win_audio = audio_waveform[w_start:w_end]
            if len(win_audio) < 2400:  # 0.15초 미만 노이즈 생략
                continue

            win_start_sec = resume_offset_sec + (w_start / 16000.0)
            win_end_sec = resume_offset_sec + (w_end / 16000.0)

            gen_kwargs: Dict[str, Any] = {
                "return_timestamps": True,
                "task": "transcribe",
            }
            if lang_locked:
                gen_kwargs["language"] = f"<|{lang_locked}|>"

            res = pipe.generate(win_audio, **gen_kwargs)

            if not lang_locked:
                raw_lang = getattr(res, "language", None)
                if raw_lang:
                    clean_lang = str(raw_lang).replace("<|", "").replace("|>", "").strip()
                    if clean_lang:
                        detected_lang = clean_lang
                        lang_locked = clean_lang
                        logger.info("OpenVINO 음성 언어 자동 감지 완료: %s", detected_lang)

            chunks = getattr(res, "chunks", None) or []
            last_text = ""
            for ch in chunks:
                text = (getattr(ch, "text", "") or "").strip()
                if not text:
                    continue
                c_start = max(0.0, float(getattr(ch, "start_ts", 0.0)))
                c_end = float(getattr(ch, "end_ts", 0.0))
                if c_end <= c_start:
                    c_end = min(len(win_audio) / 16000.0, c_start + 2.5)

                abs_start = win_start_sec + c_start
                abs_end = min(win_end_sec + 0.5, win_start_sec + c_end)

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
                last_text = text

            if partial_save_callback and new_count >= 10:
                partial_save_callback(segments_out, detected_lang)
                new_count = 0

            if progress_callback and total_duration > 0:
                pct = min(99.9, (win_end_sec / total_duration) * 100.0)
                elapsed_now = max(0.001, time.perf_counter() - t_infer_start)
                processed_audio_sec = max(0.1, win_end_sec - resume_offset_sec)
                speed_x = processed_audio_sec / elapsed_now
                progress_callback(
                    pct,
                    f"[{accel_label} | {win_end_sec:.0f}s/{total_duration:.0f}s | 경과 {elapsed_now:.0f}s ({speed_x:.1f}x배속)] {last_text[:32]}",
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
                        "ov_model": hf_repo_id,
                        "device": actual_ov_device,
                        "audio_duration_sec": round(total_duration, 2),
                        "resumed_from_sec": round(resume_offset_sec, 2),
                        "realtime_speed_x": round(speed_factor, 2),
                        "detected_language": detected_lang,
                    },
                }
            )
    finally:
        del pipe
        gc.collect()
        logger.info("OpenVINO WhisperPipeline 메모리 해제 완료")

    return segments_out, detected_lang


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
    """설정된 연산 장치(`GPU`, `NPU`, `CPU-OV`, `CPU`)에 따라 최적의 백엔드로 원문 자막을 추출합니다."""
    if cancel_event and cancel_event.is_set():
        raise JobCancelledError("음성 인식 시작 전 작업이 중단되었습니다.")

    raw_device = str(stt_config.get("device", "GPU")).strip().upper()
    if raw_device.startswith("GPU"):
        return _transcribe_with_openvino(
            wav_path, stt_config, "GPU", progress_callback, telemetry_out,
            cancel_event, initial_segments, initial_lang, partial_save_callback,
        )
    if raw_device.startswith("NPU"):
        return _transcribe_with_openvino(
            wav_path, stt_config, "NPU", progress_callback, telemetry_out,
            cancel_event, initial_segments, initial_lang, partial_save_callback,
        )
    if raw_device == "CPU-OV":
        return _transcribe_with_openvino(
            wav_path, stt_config, "CPU", progress_callback, telemetry_out,
            cancel_event, initial_segments, initial_lang, partial_save_callback,
        )

    # 기본 CPU(CTranslate2 Faster-Whisper) 또는 NVIDIA CUDA 경로
    t_stage_start = time.perf_counter()

    model_size = stt_config.get("model_size", "large-v3-turbo")
    device = "cuda" if raw_device == "CUDA" else "cpu"
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

    accel_label = "GPU (CUDA)" if device == "cuda" else "CPU (CTranslate2)"
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
            condition_on_previous_text=False,
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
        del model
        gc.collect()
        logger.info("Whisper 모델 메모리 해제 완료")

    return segments_out, detected_lang
