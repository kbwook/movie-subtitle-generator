"""전체 자막 생성 파이프라인(오디오 추출 -> STT -> 로컬 LLM 번역 -> 자막 파일 생성) 실행 오케스트레이터.

특징:
1. 각 영상 파일 처리 시 `<영화이름>_work/` 전용 작업 폴더(`MovieWorkspace`)를 생성하고
   단계별 중간 생산물(`audio_16k.wav`, `stt_segments.json`, `translation_checkpoint.json`)과
   최종 산출물(`.orig.srt`, `.ko.srt`, `.ko.ass`, `perf_report.txt`)을 저장합니다.
2. 이전 실행에서 저장된 중간 생산물이 존재하면 완료된 단계나 번역 배치를 자동으로 건너뛰고(Resume)
   나머지 작업만 이어서 수행합니다.
3. `cancel_event`를 통해 작업 진행 도중 언제든 즉시 중단(`JobCancelledError`)을 지원하며,
   중단 시점까지의 중간 생산물을 안전하게 보존합니다.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from pipeline.extractor import (
    SUPPORTED_VIDEO_EXTENSIONS,
    extract_audio_to_wav,
    get_ffmpeg_binary,
    inspect_video_metadata,
    select_best_audio_track,
)
from pipeline.formatter import write_ass, write_srt
from pipeline.models import SubtitleSegment
from pipeline.telemetry import (
    JobTelemetryReport,
    StageMetric,
    detect_hardware_inventory,
    save_telemetry_report,
)
from pipeline.transcriber import transcribe_audio
from pipeline.translator import LocalLLMTranslator
from pipeline.workspace import JobCancelledError, MovieWorkspace

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
    resume: bool = True,
    cancel_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str, float, str], None]] = None,
) -> List[Path]:
    """단일 영화 파일에 대해 전용 작업 폴더(`<영화이름>_work/`) 기반 자막 생성 및 이어하기를 수행합니다."""
    t_job_start = time.perf_counter()
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"영화 파일을 찾을 수 없습니다: {video_path}")

    if cancel_event and cancel_event.is_set():
        raise JobCancelledError("작업 시작 전 중단 요청이 감지되었습니다.")

    workspace = MovieWorkspace(video_path=video_path, output_dir=output_dir)
    if not resume:
        workspace.clear_artifacts()
    work_dir = workspace.ensure_dir()

    # 실시간 파일 로거 핸들러 등록 (공용 logs/app_execution.log 및 작업폴더 내 job_execution.log 즉시 생성)
    project_logs_dir = Path(__file__).resolve().parent.parent / "logs"
    project_logs_dir.mkdir(parents=True, exist_ok=True)
    app_log_file = project_logs_dir / "app_execution.log"
    job_log_file = work_dir / "job_execution.log"

    def _write_realtime_log(stage_str: str, pct_val: float, msg_str: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{video_path.name} | {stage_str} | {pct_val:5.1f}%] {msg_str}\n"
        for log_p in (app_log_file, job_log_file):
            try:
                with open(log_p, "a", encoding="utf-8") as lf:
                    lf.write(line)
                    lf.flush()
            except OSError:
                pass

    last_logged_bucket: Dict[str, int] = {}

    orig_status_callback = status_callback

    def wrapped_status_cb(stage_str: str, pct_val: float, msg_str: str) -> None:
        bucket = int(pct_val // 5)  # 5% 단위로 실시간 디스크 로그 기록
        if (
            pct_val in (0.0, 100.0)
            or "완료" in stage_str
            or last_logged_bucket.get(stage_str) != bucket
        ):
            last_logged_bucket[stage_str] = bucket
            _write_realtime_log(stage_str, pct_val, msg_str)
        if orig_status_callback:
            orig_status_callback(stage_str, pct_val, msg_str)

    status_callback = wrapped_status_cb
    _write_realtime_log("작업 시작", 0.0, f"작업 폴더 생성 및 파이프라인 시작: {work_dir}")

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

    generated_files: List[Path] = []
    wav_path = workspace.wav_path

    # =========================================================================
    # 1. 오디오 추출 및 코덱 정규화 (`<영화이름>_work/audio_16k.wav`)
    # =========================================================================
    t_ext_start = time.perf_counter()
    if resume and (workspace.has_valid_audio(track_index) or workspace.has_valid_stt()):
        if workspace.audio_meta_path.exists():
            selected_track = workspace.load_audio_meta()
        else:
            selected_track = select_best_audio_track(media_info.audio_tracks, preferred_lang, track_index)
        ext_elapsed = time.perf_counter() - t_ext_start
        wav_size_mb = wav_path.stat().st_size / (1024 * 1024) if wav_path.exists() else 0.0

        stage_metrics.append(
            StageMetric(
                stage_name="1/4 오디오 추출",
                accelerator_type="Cache (재사용)",
                backend_detail=f"작업 폴더 중간 생산물 재사용 ({wav_path.name}, {wav_size_mb:.1f}MB)",
                elapsed_sec=round(ext_elapsed, 3),
                compute_sec=0.0,
                throughput_label="즉시 복원 (Skip)",
            )
        )
        if status_callback:
            status_callback(
                "1/4 오디오 추출",
                100.0,
                f"[중간 생산물 재사용] {work_dir.name}/{wav_path.name} ({selected_track.display_label()})",
            )
    else:
        ext_backend = "Native FFmpeg C CLI" if get_ffmpeg_binary() else "PyAV (내장 FFmpeg C-API)"
        if status_callback:
            status_callback(
                "1/4 오디오 추출",
                0.0,
                f"[장치: CPU | 작업폴더: {work_dir.name}] 오디오 디코딩 시작...",
            )

        _, selected_track = extract_audio_to_wav(
            video_path=video_path,
            output_wav_path=wav_path,
            preferred_language=preferred_lang,
            track_index=track_index,
            progress_callback=(
                (lambda p: status_callback("1/4 오디오 추출", p, f"[CPU 디코딩 -> {work_dir.name}/audio_16k.wav] {p:.0f}%"))
                if status_callback else None
            ),
            cancel_event=cancel_event,
        )
        workspace.save_audio_meta(selected_track)

        ext_elapsed = time.perf_counter() - t_ext_start
        wav_size_mb = wav_path.stat().st_size / (1024 * 1024) if wav_path.exists() else 0.0
        ext_rtf = (
            (media_info.duration_sec / ext_elapsed)
            if ext_elapsed > 0 and media_info.duration_sec > 0
            else 0.0
        )

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
                f"완료 ({ext_elapsed:.2f}초, 저장: {work_dir.name}/{wav_path.name}): {selected_track.display_label()}",
            )

    # =========================================================================
    # 2. 다국어 음성 인식 (`<영화이름>_work/stt_segments.json`)
    # =========================================================================
    t_stt_start = time.perf_counter()
    if resume and workspace.has_valid_stt():
        segments, detected_lang = workspace.load_stt_segments()
        stt_elapsed = time.perf_counter() - t_stt_start
        stage_metrics.append(
            StageMetric(
                stage_name="2/4 음성 인식 (STT)",
                accelerator_type="Cache (재사용)",
                backend_detail=f"작업 폴더 중간 생산물 재사용 ({workspace.stt_json_path.name}, 총 {len(segments)}문장)",
                elapsed_sec=round(stt_elapsed, 3),
                compute_sec=0.0,
                items_processed=len(segments),
                throughput_label="즉시 복원 (Skip)",
            )
        )
        if status_callback:
            status_callback(
                "2/4 음성 인식 (STT)",
                100.0,
                f"[중간 생산물 재사용] {work_dir.name}/stt_segments.json (감지 언어={detected_lang}, 총 {len(segments)}문장 복원)",
            )
    else:
        partial_segments: Optional[List[SubtitleSegment]] = None
        partial_lang: Optional[str] = None
        if resume and workspace.stt_partial_path.exists():
            try:
                p_data = json.loads(workspace.stt_partial_path.read_text(encoding="utf-8"))
                partial_lang = p_data.get("detected_language")
                partial_segments = [SubtitleSegment(**item) for item in p_data.get("segments", [])]
            except Exception:
                partial_segments = None

        stt_dev = stt_cfg.get("device", "cpu").upper()
        stt_threads = stt_cfg.get("cpu_threads", 12)
        stt_compute = stt_cfg.get("compute_type", "int8")
        resume_msg = (
            f" (기존 {len(partial_segments)}문장 이후부터 이어하기)"
            if partial_segments else ""
        )
        if status_callback:
            status_callback(
                "2/4 음성 인식 (STT)",
                0.0,
                f"[장치: {stt_dev} ({stt_compute}, {stt_threads}스레드)] Whisper ({stt_cfg.get('model_size', 'large-v3-turbo')}) 시작{resume_msg}...",
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
            cancel_event=cancel_event,
            initial_segments=partial_segments,
            initial_lang=partial_lang,
            partial_save_callback=lambda segs, lang: workspace.save_stt_segments(segs, lang, partial=True),
        )

        # STT 완료 시 작업 폴더에 stt_segments.json 확정 저장
        workspace.save_stt_segments(segments, detected_lang, partial=False)

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

    # 원문 자막(.orig.srt)을 작업 폴더 및 대상 폴더에 저장
    if save_orig:
        work_orig_srt = work_dir / f"{video_path.stem}.{detected_lang}.orig.srt"
        write_srt(segments, work_orig_srt, bilingual=False, use_original_only=True)
        outer_orig_srt = workspace.base_dir / f"{video_path.stem}.{detected_lang}.orig.srt"
        shutil.copy2(work_orig_srt, outer_orig_srt)
        generated_files.append(outer_orig_srt)

    # =========================================================================
    # 3. 로컬 LLM 문맥 번역 (`<영화이름>_work/translation_checkpoint.json`)
    # =========================================================================
    if not skip_translation and detected_lang != "ko":
        restored_count = 0
        if resume:
            segments, restored_count = workspace.merge_translation_checkpoint(segments)
            if restored_count > 0 and status_callback:
                status_callback(
                    "3/4 로컬 LLM 번역",
                    (restored_count / len(segments)) * 100.0,
                    f"[중간 생산물 복원] 기존 번역 체크포인트에서 {restored_count}/{len(segments)}문장 복원 완료",
                )

        if restored_count >= len(segments):
            stage_metrics.append(
                StageMetric(
                    stage_name="3/4 로컬 LLM 번역",
                    accelerator_type="Cache (재사용)",
                    backend_detail=f"작업 폴더 번역 체크포인트 100% 재사용 ({len(segments)}문장)",
                    elapsed_sec=0.01,
                    compute_sec=0.0,
                    items_processed=len(segments),
                    throughput_label="즉시 복원 (Skip)",
                )
            )
        else:
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
                        (restored_count / len(segments)) * 100.0,
                        f"{translator.model} 모델로 한국어 번역 진행 중 (기존 {restored_count}/{len(segments)}문장 완료)...",
                    )
                segments = translator.translate_segments(
                    segments=segments,
                    source_lang=detected_lang,
                    target_lang=target_lang,
                    progress_callback=(
                        (lambda p, detail: status_callback("3/4 로컬 LLM 번역", p, detail))
                        if status_callback else None
                    ),
                    cancel_event=cancel_event,
                    checkpoint_callback=lambda segs: workspace.save_translation_checkpoint(
                        segs, detected_lang, target_lang, translator.model
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

    # =========================================================================
    # 4. 최종 자막 파일(.srt / .ass) 포맷팅 및 저장 (작업 폴더 + 영상 폴더 동시 저장)
    # =========================================================================
    t_fmt_start = time.perf_counter()
    if status_callback:
        status_callback("4/4 자막 파일 저장", 50.0, "타임스탬프 보정 및 작업/출력 폴더에 자막 파일 기록 중...")

    sub_suffix = ".ko" if not skip_translation else ""
    if "srt" in formats:
        work_srt = work_dir / f"{video_path.stem}{sub_suffix}.srt"
        outer_srt = workspace.base_dir / f"{video_path.stem}{sub_suffix}.srt"
        write_srt(segments, work_srt, bilingual=bilingual, use_original_only=skip_translation)
        shutil.copy2(work_srt, outer_srt)
        generated_files.append(outer_srt)

    if "ass" in formats:
        work_ass = work_dir / f"{video_path.stem}{sub_suffix}.ass"
        outer_ass = workspace.base_dir / f"{video_path.stem}{sub_suffix}.ass"
        write_ass(segments, work_ass, bilingual=bilingual)
        shutil.copy2(work_ass, outer_ass)
        generated_files.append(outer_ass)

    fmt_elapsed = time.perf_counter() - t_fmt_start
    stage_metrics.append(
        StageMetric(
            stage_name="4/4 자막 파일 저장",
            accelerator_type="CPU + Disk I/O",
            backend_detail=f"작업폴더({work_dir.name}) 및 출력폴더 저장 ({', '.join(p.name for p in generated_files)})",
            elapsed_sec=round(fmt_elapsed, 3),
            compute_sec=round(fmt_elapsed, 3),
            items_processed=len(segments),
        )
    )

    total_job_elapsed = time.perf_counter() - t_job_start

    opt_notes: List[str] = []
    for sm in stage_metrics:
        if sm.stage_name == "2/4 음성 인식 (STT)" and sm.accelerator_type == "CPU":
            opt_notes.append(
                "STT(음성 인식) 단계가 현재 CTranslate2 CPU(INT8)로 동작했습니다. "
                "Intel Arc GPU(Vulkan `whisper.cpp`) 또는 Intel NPU/iGPU(`OpenVINO Whisper`) 백엔드로 전환 시 "
                "CPU 점유율을 낮추고 STT 속도를 약 2~3배 단축할 수 있습니다."
            )
        if sm.stage_name == "3/4 로컬 LLM 번역" and sm.accelerator_type != "Cache (재사용)":
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
    workspace.perf_report_path.write_text(summary_str + "\n", encoding="utf-8")
    logger.info("%s\n[작업 폴더] %s\n[성능 로그] %s", summary_str, work_dir, report_path)

    if status_callback:
        status_callback(
            "완료",
            100.0,
            f"자막 생성 완료 (총 {total_job_elapsed/60:.1f}분, 작업폴더: {work_dir.name}): {', '.join(p.name for p in generated_files)}\n{summary_str}",
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
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
            and not p.parent.name.endswith("_work")
        ]
        return files
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target_path}")
