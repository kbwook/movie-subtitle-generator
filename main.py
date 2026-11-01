r"""로컬 AI 영화 자막 생성기 CLI 및 GUI 통합 진입점.

사용 예시:
1. 데스크톱 GUI 실행 (인자 없이 실행하거나 --gui 옵션):
   .\.venv\Scripts\python.exe main.py --gui

2. 단일 영화 파일 자막 생성 (자동으로 <영화이름>_work 폴더에 중간 생산물 저장 및 이어하기 지원):
   .\.venv\Scripts\python.exe main.py "D:\Movies\sample_movie.mkv"

3. 중간 생산물을 무시하고 처음부터 새로 생성 (--fresh):
   .\.venv\Scripts\python.exe main.py "D:\Movies\sample_movie.mkv" --fresh

4. 특정 언어(중국어/일본어/영어/프랑스어) 및 이중 자막(한글+원문) 지정:
   .\.venv\Scripts\python.exe main.py "D:\Movies\jp_movie.mp4" --lang ja --bilingual

5. 고품질 14B 모델 사용:
   .\.venv\Scripts\python.exe main.py "D:\Movies\cn_movie.mkv" --lang zh --llm-model qwen2.5:14b

6. 영화 파일 내 오디오 트랙 목록 확인:
   .\.venv\Scripts\python.exe main.py "D:\Movies\multi_audio.mkv" --list-tracks
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from pipeline.extractor import list_audio_tracks
from pipeline.runner import collect_video_files, load_config, process_video_file
from pipeline.telemetry import detect_hardware_inventory
from pipeline.translator import LocalLLMTranslator
from pipeline.workspace import JobCancelledError, MovieWorkspace


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure") and sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure") and sys.stderr is not None:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    app_log_path = logs_dir / "app_execution.log"

    handlers: list[logging.Handler] = [
        logging.FileHandler(app_log_path, encoding="utf-8"),
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="다양한 코덱의 중·일·영 영화 파일로부터 한국어 자막(.srt/.ass)을 생성합니다."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="자막을 생성할 영화 파일 경로 또는 영화 폴더 경로 (생략 시 GUI 실행)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="데스크톱 그래픽 인터페이스(GUI)를 실행합니다.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="설정 파일 경로 (기본값: config.yaml)",
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "zh", "ja", "en", "fr"],
        default=None,
        help="원본 영화 음성 언어 (auto: 자동 감지, zh: 중국어, ja: 일본어, en: 영어, fr: 프랑스어)",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Faster-Whisper 모델 크기 (예: large-v3-turbo, large-v3, medium)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama 로컬 LLM 모델 태그 (예: qwen2.5:7b, qwen2.5:14b)",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="추출할 오디오 트랙 번호 (0부터 시작, --list-tracks로 확인 가능)",
    )
    parser.add_argument(
        "--list-tracks",
        action="store_true",
        help="영상 파일 내 오디오 트랙(코덱/언어) 목록만 출력하고 종료합니다.",
    )
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="한국어 번역문과 외국어 원문을 동시에 표시하는 이중 자막을 생성합니다.",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="LLM 번역을 건너뛰고 원문 자막(STT)만 생성합니다.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="작업 폴더(<영화이름>_work)에 저장된 중간 생산물을 무시하고 처음부터 새로 실행합니다.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="현재 노트북의 파이프라인 준비 상태 및 로컬 LLM 연결 상태를 점검합니다.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)

    if args.lang:
        config["stt"]["language"] = args.lang
    if args.whisper_model:
        config["stt"]["model_size"] = args.whisper_model
    if args.llm_model:
        config["llm"]["model"] = args.llm_model
    if args.bilingual:
        config["subtitle"]["bilingual"] = True

    if args.check_env:
        hw = detect_hardware_inventory()
        print("=== 로컬 AI 영화 자막 생성기 하드웨어 및 환경 점검 ===")
        print(f"- CPU: {hw.cpu_name} ({hw.logical_cores} 논리 코어)")
        print(f"- GPU: {', '.join(hw.gpus) or '감지 안 됨'}")
        print(f"- NPU: {', '.join(hw.npus) or '감지 안 됨'}")
        print(f"- RAM: 가용 {hw.avail_ram_gb:.1f} GB / 전체 {hw.total_ram_gb:.1f} GB")
        print(f"- ONNX 프로바이더: {', '.join(hw.onnx_providers)}")
        print(f"- STT 모델 설정: {config['stt']['model_size']} (device={config['stt'].get('device', 'cpu')}, threads={config['stt']['cpu_threads']}, compute={config['stt']['compute_type']})")
        translator = LocalLLMTranslator(config.get("llm", {}))
        ok, msg = translator.check_server_and_model()
        print(f"- LLM 상태: {'[정상]' if ok else '[조치 필요]'} {msg}")
        return 0

    if args.gui or not args.target:
        from gui import launch_gui
        launch_gui()
        return 0

    target_path = Path(args.target).resolve()
    if args.list_tracks:
        tracks = list_audio_tracks(target_path)
        print(f"=== 오디오 트랙 목록: {target_path.name} (총 {len(tracks)}개) ===")
        for t in tracks:
            print(" ", t.display_label())
        return 0

    video_files = collect_video_files(target_path)
    if not video_files:
        logging.error("처리할 동영상 파일이 없습니다: %s", target_path)
        return 1

    logging.info("총 %d개의 영화 파일을 처리합니다.", len(video_files))

    cancel_event = threading.Event()

    def _sigint_handler(_sig, _frame):
        print("\n[🛑 즉시 중단 요청] 현재 작업 단계의 중간 생산물을 저장하고 즉시 종료합니다...")
        cancel_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    def cli_status(stage: str, pct: float, detail: str) -> None:
        if pct in (0.0, 100.0) or int(pct) % 20 == 0:
            print(f"[{stage} | {pct:5.1f}%] {detail}")

    for idx, video_file in enumerate(video_files, start=1):
        ws = MovieWorkspace(video_file)
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(video_files)}] 영화 처리 시작: {video_file.name}")
        print(f" -> 작업 폴더: {ws.work_dir} (현재 상태: {ws.inspect_resume_status()})")
        print("=" * 70)
        try:
            out_files = process_video_file(
                video_path=video_file,
                config=config,
                track_index=args.track,
                skip_translation=args.no_translate,
                resume=not args.fresh,
                cancel_event=cancel_event,
                status_callback=cli_status,
            )
            print("-> 생성된 자막 파일:")
            for f in out_files:
                print(f"   * {f}")
        except JobCancelledError as exc:
            print(f"\n[중단 완료] {exc}")
            print(f" -> 보존된 중간 생산물 상태: {ws.inspect_resume_status()}")
            return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
