"""로컬 AI 다국어 영화 자막 생성기 - 멀티 파일 대기열(Queue) 관리 및 상세 옵션 GUI."""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

from pipeline.extractor import SUPPORTED_VIDEO_EXTENSIONS, inspect_video_metadata
from pipeline.models import VideoMediaInfo
from pipeline.runner import collect_video_files, load_config, process_video_file, save_config
from pipeline.telemetry import detect_hardware_inventory
from pipeline.translator import LocalLLMTranslator


@dataclass
class QueueItem:
    """대기열(리스트)에 등록된 개별 영화 작업 항목."""
    item_id: str
    video_path: Path
    media_info: Optional[VideoMediaInfo] = None
    track_index: Optional[int] = None      # None이면 자동 선택
    language_override: str = "default"     # "default", "auto", "zh", "ja", "en"
    custom_hint: str = ""                  # 개별 영화 배경 힌트
    status: str = "대기 중"
    generated_files: List[Path] = field(default_factory=list)

    def track_lang_label(self) -> str:
        t_str = "자동트랙" if self.track_index is None else f"트랙#{self.track_index}"
        l_str = "기본언어" if self.language_override == "default" else self.language_override
        return f"{t_str} / {l_str}"


class SubtitleGeneratorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("로컬 AI 다국어 영화 자막 생성기 (Intel Core Ultra 7 + Arc 최적화)")
        self.root.geometry("1080x820")
        self.root.minsize(960, 740)

        self.config_path = Path(__file__).resolve().parent / "config.yaml"
        self.config = load_config(self.config_path)

        self.queue_items: Dict[str, QueueItem] = {}
        self.queue_order: List[str] = []
        self._item_counter = 0

        self.is_running = False
        self.stop_requested = False

        self._configure_styles()
        self._build_ui()
        self._check_llm_status_async()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=26, font=("맑은 고딕", 9))
        style.configure("Treeview.Heading", font=("맑은 고딕", 9, "bold"))
        style.configure("Accent.TButton", font=("맑은 고딕", 10, "bold"))

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 5}

        # =====================================================================
        # 1. 상단: 자막 생성 대상 영화 리스트(Queue) 관리 영역
        # =====================================================================
        queue_frame = ttk.LabelFrame(
            self.root,
            text="1. 자막 생성 영화 리스트 관리 (다양한 코덱 지원: MKV, MP4, AVI, TS, MOV, WMV 등)",
        )
        queue_frame.pack(fill="both", expand=True, **pad)

        # 리스트 상단 툴바 버튼들
        toolbar = ttk.Frame(queue_frame)
        toolbar.pack(fill="x", padx=8, pady=(6, 4))

        ttk.Button(toolbar, text="➕ 영화 파일 추가", command=self._on_add_files).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📁 폴더 통째로 추가", command=self._on_add_folder).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        ttk.Button(toolbar, text="🔼 위로", width=7, command=lambda: self._on_move_item(-1)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🔽 아래로", width=7, command=lambda: self._on_move_item(1)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🗑️ 선택 삭제", command=self._on_remove_selected).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🧹 완료 항목 정리", command=self._on_clear_completed).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🔄 전체 비우기", command=self._on_clear_all).pack(side="left", padx=2)

        ttk.Button(toolbar, text="📊 성능 리포트 열기", command=self._on_open_perf_log).pack(side="right", padx=2)
        ttk.Button(toolbar, text="📂 선택 파일 폴더 열기", command=self._on_open_folder).pack(side="right", padx=2)
        self.queue_count_var = tk.StringVar(value="등록된 영화: 0편")
        ttk.Label(toolbar, textvariable=self.queue_count_var, font=("맑은 고딕", 9, "bold")).pack(
            side="right", padx=10
        )

        # Treeview 테이블
        tree_container = ttk.Frame(queue_frame)
        tree_container.pack(fill="both", expand=True, padx=8, pady=4)

        columns = ("no", "filename", "video_info", "audio_tracks", "override", "status")
        self.tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=7,
        )
        self.tree.heading("no", text="#")
        self.tree.heading("filename", text="영화 파일명")
        self.tree.heading("video_info", text="비디오 코덱 / 길이 / 용량")
        self.tree.heading("audio_tracks", text="감지된 오디오 트랙 / 코덱")
        self.tree.heading("override", text="적용 트랙 / 언어")
        self.tree.heading("status", text="진행 상태")

        self.tree.column("no", width=40, anchor="center", stretch=False)
        self.tree.column("filename", width=260, anchor="w")
        self.tree.column("video_info", width=185, anchor="center", stretch=False)
        self.tree.column("audio_tracks", width=250, anchor="w")
        self.tree.column("override", width=125, anchor="center", stretch=False)
        self.tree.column("status", width=150, anchor="center", stretch=False)

        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # 선택된 영화 개별 오디오 트랙/언어 지정 바
        item_opt_frame = ttk.Frame(queue_frame)
        item_opt_frame.pack(fill="x", padx=8, pady=(2, 6))

        ttk.Label(item_opt_frame, text="▶ 선택 영화 개별 설정:").pack(side="left", padx=(2, 6))
        ttk.Label(item_opt_frame, text="오디오 트랙:").pack(side="left")
        self.item_track_var = tk.StringVar(value="자동 선택 (중·일·영 우선)")
        self.item_track_combo = ttk.Combobox(
            item_opt_frame,
            textvariable=self.item_track_var,
            state="readonly",
            width=36,
            values=["자동 선택 (중·일·영 우선)"],
        )
        self.item_track_combo.pack(side="left", padx=(4, 10))

        ttk.Label(item_opt_frame, text="개별 언어:").pack(side="left")
        self.item_lang_var = tk.StringVar(value="default (공통 설정 따름)")
        self.item_lang_combo = ttk.Combobox(
            item_opt_frame,
            textvariable=self.item_lang_var,
            state="readonly",
            width=20,
            values=[
                "default (공통 설정 따름)",
                "auto (자동 감지)",
                "zh (중국어)",
                "ja (일본어)",
                "en (영어)",
            ],
        )
        self.item_lang_combo.pack(side="left", padx=(4, 10))

        ttk.Label(item_opt_frame, text="개별 힌트:").pack(side="left")
        self.item_hint_var = tk.StringVar(value="")
        ttk.Entry(item_opt_frame, textvariable=self.item_hint_var, width=18).pack(side="left", padx=(4, 8))

        ttk.Button(
            item_opt_frame,
            text="선택 항목에 적용",
            command=self._on_apply_item_override,
        ).pack(side="left", padx=2)

        # =====================================================================
        # 2. 중단: 음성 인식(STT) / 로컬 LLM 번역 / 자막 포맷 상세 옵션 영역 (탭/그룹)
        # =====================================================================
        options_container = ttk.Frame(self.root)
        options_container.pack(fill="x", **pad)

        # 좌측: 음성 인식(STT) & 자막 출력 옵션
        stt_frame = ttk.LabelFrame(options_container, text="2. 음성 인식 (STT) 및 자막 출력 옵션")
        stt_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ttk.Label(stt_frame, text="기본 원본 언어:").grid(row=0, column=0, sticky="w", padx=8, pady=5)
        init_lang = self.config["stt"].get("language", "auto")
        lang_map = {
            "auto": "auto (자동 감지)",
            "zh": "zh (중국어)",
            "ja": "ja (일본어)",
            "en": "en (영어)",
        }
        self.global_lang_var = tk.StringVar(value=lang_map.get(init_lang, "auto (자동 감지)"))
        ttk.Combobox(
            stt_frame,
            textvariable=self.global_lang_var,
            state="readonly",
            values=list(lang_map.values()),
            width=18,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(stt_frame, text="Whisper 모델:").grid(row=0, column=2, sticky="w", padx=8, pady=5)
        self.whisper_var = tk.StringVar(value=self.config["stt"].get("model_size", "large-v3-turbo"))
        ttk.Combobox(
            stt_frame,
            textvariable=self.whisper_var,
            state="readonly",
            values=["large-v3-turbo", "large-v3", "medium", "small"],
            width=16,
        ).grid(row=0, column=3, sticky="w", padx=6, pady=5)

        ttk.Label(stt_frame, text="CPU 스레드:").grid(row=1, column=0, sticky="w", padx=8, pady=5)
        self.threads_var = tk.IntVar(value=int(self.config["stt"].get("cpu_threads", 12)))
        ttk.Spinbox(stt_frame, from_=2, to=22, textvariable=self.threads_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=5
        )

        self.vad_var = tk.BooleanVar(value=bool(self.config["stt"].get("vad_filter", True)))
        ttk.Checkbutton(
            stt_frame,
            text="Silero VAD 필터 (배경음악 환각 방지)",
            variable=self.vad_var,
        ).grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=5)

        # 자막 포맷 체크박스들
        fmt_list = self.config["subtitle"].get("formats", ["srt", "ass"])
        self.fmt_srt_var = tk.BooleanVar(value=("srt" in fmt_list))
        self.fmt_ass_var = tk.BooleanVar(value=("ass" in fmt_list))
        self.bilingual_var = tk.BooleanVar(value=bool(self.config["subtitle"].get("bilingual", False)))
        self.save_orig_var = tk.BooleanVar(value=bool(self.config["subtitle"].get("save_original_srt", True)))

        chk_row = ttk.Frame(stt_frame)
        chk_row.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=4)
        ttk.Label(chk_row, text="출력 형식:").pack(side="left", padx=(0, 6))
        ttk.Checkbutton(chk_row, text=".srt", variable=self.fmt_srt_var).pack(side="left", padx=4)
        ttk.Checkbutton(chk_row, text=".ass (스타일 자막)", variable=self.fmt_ass_var).pack(side="left", padx=4)
        ttk.Checkbutton(chk_row, text="원문(.orig.srt) 보관", variable=self.save_orig_var).pack(side="left", padx=8)

        chk_row2 = ttk.Frame(stt_frame)
        chk_row2.grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(
            chk_row2,
            text="한글 + 외국어 원문 동시 표기 (이중 자막)",
            variable=self.bilingual_var,
        ).pack(side="left", padx=(0, 12))

        self.skip_trans_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            chk_row2,
            text="번역 생략 (원문 STT만 추출)",
            variable=self.skip_trans_var,
        ).pack(side="left")

        # 저장 폴더 지정
        out_row = ttk.Frame(stt_frame)
        out_row.grid(row=4, column=0, columnspan=4, sticky="we", padx=8, pady=5)
        ttk.Label(out_row, text="자막 저장 폴더:").pack(side="left")
        self.output_dir_var = tk.StringVar(value="")
        ttk.Entry(out_row, textvariable=self.output_dir_var, width=28).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(out_row, text="폴더 선택", width=9, command=self._on_browse_output_dir).pack(side="left", padx=2)
        ttk.Button(
            out_row,
            text="원본 위치",
            width=8,
            command=lambda: self.output_dir_var.set(""),
        ).pack(side="left")

        # 우측: 로컬 LLM(Ollama) 옵션
        llm_frame = ttk.LabelFrame(options_container, text="3. 로컬 LLM 번역 설정 (Ollama / Intel Arc GPU 가속)")
        llm_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        ttk.Label(llm_frame, text="LLM 모델:").grid(row=0, column=0, sticky="w", padx=8, pady=5)
        self.llm_var = tk.StringVar(value=self.config["llm"].get("model", "qwen2.5:7b"))
        self.llm_combo = ttk.Combobox(
            llm_frame,
            textvariable=self.llm_var,
            values=["qwen2.5:7b", "qwen2.5:14b", "gemma2:9b", "exaone3.5:7.8b"],
            width=18,
        )
        self.llm_combo.grid(row=0, column=1, sticky="w", padx=6, pady=5)

        btn_pull = ttk.Button(llm_frame, text="⬇ 모델 준비/점검", command=self._on_pull_or_check_model)
        btn_pull.grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=5)

        ttk.Label(llm_frame, text="번역 배치 크기:").grid(row=1, column=0, sticky="w", padx=8, pady=5)
        self.batch_size_var = tk.IntVar(value=int(self.config["llm"].get("batch_size", 10)))
        ttk.Spinbox(llm_frame, from_=3, to=25, textvariable=self.batch_size_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=5
        )

        ttk.Label(llm_frame, text="이전 문맥 참조:").grid(row=1, column=2, sticky="w", padx=8, pady=5)
        self.ctx_size_var = tk.IntVar(value=int(self.config["llm"].get("context_size", 5)))
        ttk.Spinbox(llm_frame, from_=0, to=15, textvariable=self.ctx_size_var, width=8).grid(
            row=1, column=3, sticky="w", padx=6, pady=5
        )

        ttk.Label(llm_frame, text="Temperature:").grid(row=2, column=0, sticky="w", padx=8, pady=5)
        self.temp_var = tk.DoubleVar(value=float(self.config["llm"].get("temperature", 0.2)))
        ttk.Spinbox(
            llm_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.temp_var, width=8
        ).grid(row=2, column=1, sticky="w", padx=6, pady=5)

        ttk.Button(
            llm_frame,
            text="💾 현재 옵션 기본값 저장",
            command=self._on_save_config,
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=5)

        ttk.Label(llm_frame, text="공통 작품 힌트:").grid(row=3, column=0, sticky="w", padx=8, pady=5)
        self.global_hint_var = tk.StringVar(value=self.config["llm"].get("movie_context", ""))
        ttk.Entry(llm_frame, textvariable=self.global_hint_var, width=36).grid(
            row=3, column=1, columnspan=3, sticky="we", padx=6, pady=5
        )

        self.llm_status_var = tk.StringVar(value="Ollama 상태 확인 중...")
        ttk.Label(llm_frame, textvariable=self.llm_status_var, foreground="#0055aa").grid(
            row=4, column=0, columnspan=4, sticky="w", padx=8, pady=4
        )
        llm_frame.columnconfigure(3, weight=1)

        # =====================================================================
        # 3. 하단: 전체 대기열 진행률 / 현재 파일 진행률 및 실행 제어
        # =====================================================================
        prog_frame = ttk.LabelFrame(self.root, text="4. 작업 진행 상태 및 실행 제어")
        prog_frame.pack(fill="x", **pad)

        # 대기열 전체 진행률
        q_row = ttk.Frame(prog_frame)
        q_row.pack(fill="x", padx=10, pady=(4, 2))
        self.queue_prog_label_var = tk.StringVar(value="전체 대기열 진행률: 대기 중 (0 / 0)")
        ttk.Label(q_row, textvariable=self.queue_prog_label_var, width=32).pack(side="left")
        self.queue_prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(q_row, variable=self.queue_prog_var, maximum=100.0).pack(
            side="left", fill="x", expand=True, padx=6
        )

        # 현재 파일 세부 진행률
        f_row = ttk.Frame(prog_frame)
        f_row.pack(fill="x", padx=10, pady=2)
        self.stage_label_var = tk.StringVar(value="현재 파일 단계: 대기 중")
        ttk.Label(f_row, textvariable=self.stage_label_var, width=32, font=("맑은 고딕", 9, "bold")).pack(
            side="left"
        )
        self.file_prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(f_row, variable=self.file_prog_var, maximum=100.0).pack(
            side="left", fill="x", expand=True, padx=6
        )

        # 상태 메시지 및 시작/중지 버튼
        ctrl_row = ttk.Frame(prog_frame)
        ctrl_row.pack(fill="x", padx=10, pady=(4, 6))

        self.detail_label_var = tk.StringVar(
            value="영화를 리스트에 추가하고 [대기열 자막 생성 시작]을 눌러주세요."
        )
        ttk.Label(ctrl_row, textvariable=self.detail_label_var).pack(side="left", fill="x", expand=True)

        self.stop_btn = ttk.Button(
            ctrl_row,
            text="⏹ 현재 파일 완료 후 중지",
            state="disabled",
            command=self._on_stop_queue,
        )
        self.stop_btn.pack(side="right", padx=4, ipadx=8, ipady=3)

        self.start_btn = ttk.Button(
            ctrl_row,
            text="▶ 대기열 자막 생성 시작",
            style="Accent.TButton",
            command=self._on_start_queue,
        )
        self.start_btn.pack(side="right", padx=4, ipadx=14, ipady=3)

        # =====================================================================
        # 4. 최하단: 실시간 작업 로그
        # =====================================================================
        log_frame = ttk.LabelFrame(self.root, text="실시간 작업 로그")
        log_frame.pack(fill="both", expand=False, **pad)

        self.log_box = scrolledtext.ScrolledText(log_frame, height=7, state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, padx=8, pady=6)

    # -------------------------------------------------------------------------
    # 유틸리티 및 로그 메서드
    # -------------------------------------------------------------------------
    def _append_log(self, msg: str) -> None:
        def _update():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.root.after(0, _update)

    def _sync_ui_to_config(self) -> Dict:
        lang_code = self.global_lang_var.get().split()[0]
        self.config["stt"]["language"] = lang_code
        self.config["stt"]["model_size"] = self.whisper_var.get().strip()
        self.config["stt"]["cpu_threads"] = int(self.threads_var.get())
        self.config["stt"]["vad_filter"] = bool(self.vad_var.get())

        self.config["llm"]["model"] = self.llm_var.get().strip()
        self.config["llm"]["batch_size"] = int(self.batch_size_var.get())
        self.config["llm"]["context_size"] = int(self.ctx_size_var.get())
        self.config["llm"]["temperature"] = float(self.temp_var.get())
        self.config["llm"]["movie_context"] = self.global_hint_var.get().strip()

        formats = []
        if self.fmt_srt_var.get():
            formats.append("srt")
        if self.fmt_ass_var.get():
            formats.append("ass")
        if not formats:
            formats = ["srt"]
            self.fmt_srt_var.set(True)

        self.config["subtitle"]["formats"] = formats
        self.config["subtitle"]["bilingual"] = bool(self.bilingual_var.get())
        self.config["subtitle"]["save_original_srt"] = bool(self.save_orig_var.get())
        return self.config

    def _on_save_config(self) -> None:
        self._sync_ui_to_config()
        save_config(self.config, self.config_path)
        self._append_log(f"[설정 저장] 현재 옵션이 {self.config_path.name}에 저장되었습니다.")
        messagebox.showinfo("설정 저장 완료", "현재 설정이 기본값(config.yaml)으로 저장되었습니다.")

    def _check_llm_status_async(self) -> None:
        def _worker():
            self._sync_ui_to_config()
            hw = detect_hardware_inventory()
            self._append_log(
                f"[하드웨어 진단] CPU: {hw.cpu_name} ({hw.logical_cores}코어) | "
                f"GPU: {', '.join(hw.gpus) or '없음'} | "
                f"NPU: {', '.join(hw.npus) or '없음'} | "
                f"RAM: 가용 {hw.avail_ram_gb:.1f}GB/{hw.total_ram_gb:.1f}GB"
            )
            translator = LocalLLMTranslator(self.config.get("llm", {}))
            ok, msg = translator.check_server_and_model()
            short_msg = msg.splitlines()[0]
            self.root.after(0, lambda: self.llm_status_var.set(f"{'✅' if ok else '⚠️'} {short_msg}"))
            self._append_log(f"[시스템 점검] {msg}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_open_perf_log(self) -> None:
        log_file = Path(__file__).resolve().parent / "logs" / "perf_report.log"
        if log_file.exists():
            os.startfile(str(log_file))
        else:
            messagebox.showinfo(
                "성능 리포트 없음",
                "아직 기록된 성능 리포트(logs/perf_report.log)가 없습니다.\n영화 자막 생성이 완료되면 자동으로 기록됩니다.",
            )

    def _on_pull_or_check_model(self) -> None:
        model_name = self.llm_var.get().strip()
        if not model_name:
            return

        def _worker():
            self.root.after(0, lambda: self.llm_status_var.set(f"⏳ '{model_name}' 모델 상태 점검/다운로드 중..."))
            self._append_log(f"[Ollama] '{model_name}' 모델 준비 확인 중...")
            ollama_exe = "ollama"
            local_ollama = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            if local_ollama.exists():
                ollama_exe = str(local_ollama)

            try:
                proc = subprocess.run(
                    [ollama_exe, "pull", model_name],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if proc.returncode == 0:
                    self._append_log(f"[Ollama] '{model_name}' 준비 완료!")
                else:
                    self._append_log(f"[Ollama 오류] {proc.stderr.strip()}")
            except Exception as exc:
                self._append_log(f"[Ollama 실행 실패] {exc} (setup_env.ps1을 먼저 실행해 주세요)")

            self._check_llm_status_async()

        threading.Thread(target=_worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # 영화 리스트(Queue) 추가/삭제/정렬 관리
    # -------------------------------------------------------------------------
    def _add_paths_to_queue(self, paths: List[Path]) -> None:
        existing_paths = {item.video_path for item in self.queue_items.values()}
        added_count = 0

        for p in paths:
            p = p.resolve()
            if p in existing_paths or not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                continue

            self._item_counter += 1
            item_id = f"item_{self._item_counter}"
            q_item = QueueItem(item_id=item_id, video_path=p)

            try:
                info = inspect_video_metadata(p)
                q_item.media_info = info
                video_str = f"{info.video_codec} | {info.format_duration()} | {info.format_size()}"
                audio_str = info.audio_summary()
            except Exception as exc:
                video_str = "분석 오류"
                audio_str = str(exc)[:30]

            self.queue_items[item_id] = q_item
            self.queue_order.append(item_id)
            existing_paths.add(p)
            added_count += 1

            self.tree.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    len(self.queue_order),
                    p.name,
                    video_str,
                    audio_str,
                    q_item.track_lang_label(),
                    q_item.status,
                ),
            )

        self._refresh_tree_numbers()
        if added_count > 0:
            self._append_log(f"[리스트 추가] 영화 {added_count}편이 대기열에 등록되었습니다.")

    def _refresh_tree_numbers(self) -> None:
        for idx, item_id in enumerate(self.queue_order, start=1):
            q_item = self.queue_items[item_id]
            current_vals = list(self.tree.item(item_id, "values"))
            if current_vals:
                current_vals[0] = idx
                current_vals[4] = q_item.track_lang_label()
                current_vals[5] = q_item.status
                self.tree.item(item_id, values=current_vals)
        self.queue_count_var.set(f"등록된 영화: {len(self.queue_order)}편")

    def _on_add_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="자막을 생성할 영화 파일 선택 (다중 선택 가능)",
            filetypes=[
                ("동영상 파일", "*.mkv *.mp4 *.avi *.mov *.ts *.m2ts *.wmv *.flv *.webm *.mpg *.mpeg *.m4v"),
                ("모든 파일", "*.*"),
            ],
        )
        if file_paths:
            self._add_paths_to_queue([Path(p) for p in file_paths])

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="영화 파일이 들어있는 폴더 선택")
        if not folder:
            return
        files = collect_video_files(Path(folder))
        if not files:
            messagebox.showinfo("파일 없음", "선택한 폴더 내에 지원하는 동영상 파일이 없습니다.")
            return
        self._add_paths_to_queue(files)

    def _on_move_item(self, direction: int) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        for item_id in (selected if direction < 0 else reversed(selected)):
            idx = self.queue_order.index(item_id)
            new_idx = idx + direction
            if 0 <= new_idx < len(self.queue_order):
                self.queue_order[idx], self.queue_order[new_idx] = (
                    self.queue_order[new_idx],
                    self.queue_order[idx],
                )
                self.tree.move(item_id, "", new_idx)
        self._refresh_tree_numbers()

    def _on_remove_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        for item_id in selected:
            if item_id in self.queue_items:
                del self.queue_items[item_id]
            if item_id in self.queue_order:
                self.queue_order.remove(item_id)
            self.tree.delete(item_id)
        self._refresh_tree_numbers()

    def _on_clear_completed(self) -> None:
        to_remove = [
            iid for iid, item in self.queue_items.items() if item.status.startswith("✅")
        ]
        for iid in to_remove:
            del self.queue_items[iid]
            self.queue_order.remove(iid)
            self.tree.delete(iid)
        self._refresh_tree_numbers()

    def _on_clear_all(self) -> None:
        if self.is_running:
            messagebox.showwarning("작업 실행 중", "자막 생성이 진행 중일 때는 전체 비우기를 할 수 없습니다.")
            return
        self.queue_items.clear()
        self.queue_order.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._refresh_tree_numbers()

    def _on_open_folder(self) -> None:
        selected = self.tree.selection()
        target_dir = None
        if selected and selected[0] in self.queue_items:
            target_dir = self.queue_items[selected[0]].video_path.parent
        elif self.output_dir_var.get().strip():
            target_dir = Path(self.output_dir_var.get().strip())
        elif self.queue_order:
            target_dir = self.queue_items[self.queue_order[0]].video_path.parent

        if target_dir and target_dir.exists():
            os.startfile(str(target_dir))

    def _on_browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="생성된 자막을 저장할 폴더 선택")
        if folder:
            self.output_dir_var.set(folder)

    # -------------------------------------------------------------------------
    # 개별 영화 오디오 트랙 & 언어 오버라이드 설정
    # -------------------------------------------------------------------------
    def _on_tree_select(self, _event: object = None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        first_id = selected[0]
        q_item = self.queue_items.get(first_id)
        if not q_item:
            return

        tracks = q_item.media_info.audio_tracks if q_item.media_info else []
        values = ["자동 선택 (중·일·영 우선)"] + [t.display_label() for t in tracks]
        self.item_track_combo["values"] = values

        if q_item.track_index is None or q_item.track_index >= len(tracks):
            self.item_track_combo.current(0)
        else:
            self.item_track_combo.current(q_item.track_index + 1)

        lang_prefix = q_item.language_override
        for idx, val in enumerate(self.item_lang_combo["values"]):
            if val.startswith(lang_prefix):
                self.item_lang_combo.current(idx)
                break

        self.item_hint_var.set(q_item.custom_hint)

    def _on_apply_item_override(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("항목 미선택", "먼저 리스트에서 개별 설정을 적용할 영화를 선택해 주세요.")
            return

        track_combo_idx = self.item_track_combo.current()
        track_idx: Optional[int] = None if track_combo_idx <= 0 else (track_combo_idx - 1)
        lang_code = self.item_lang_var.get().split()[0]
        custom_hint = self.item_hint_var.get().strip()

        for item_id in selected:
            q_item = self.queue_items.get(item_id)
            if not q_item:
                continue
            q_item.track_index = track_idx
            q_item.language_override = lang_code
            q_item.custom_hint = custom_hint

        self._refresh_tree_numbers()
        self._append_log(
            f"[개별 설정 적용] 선택한 {len(selected)}개 영화에 ({track_idx if track_idx is not None else '자동트랙'} / {lang_code}) 설정이 반영되었습니다."
        )

    # -------------------------------------------------------------------------
    # 대기열 순차 실행 및 중지 제어
    # -------------------------------------------------------------------------
    def _on_stop_queue(self) -> None:
        if not self.is_running:
            return
        self.stop_requested = True
        self.stop_btn.configure(state="disabled")
        self._append_log("[중지 예약] 현재 처리 중인 영화의 자막 생성이 끝나면 대기열 작업을 중지합니다.")

    def _on_start_queue(self) -> None:
        if self.is_running:
            return

        pending_ids = [
            iid for iid in self.queue_order if not self.queue_items[iid].status.startswith("✅")
        ]
        if not pending_ids:
            messagebox.showwarning(
                "처리할 영화 없음",
                "대기열에 처리할 영화가 없습니다. 영화 파일을 추가하거나 완료 항목을 초기화해 주세요.",
            )
            return

        self._sync_ui_to_config()
        skip_translation = bool(self.skip_trans_var.get())
        out_dir_str = self.output_dir_var.get().strip()
        output_dir = Path(out_dir_str) if out_dir_str else None

        self.is_running = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        total_jobs = len(pending_ids)
        self._append_log("=" * 72)
        self._append_log(f"[대기열 시작] 총 {total_jobs}편의 영화 자막 생성을 시작합니다.")

        def _worker() -> None:
            completed_count = 0
            try:
                for idx, item_id in enumerate(pending_ids, start=1):
                    if self.stop_requested:
                        self._append_log("[작업 중지됨] 사용자 요청으로 대기열 실행을 멈췄습니다.")
                        break

                    q_item = self.queue_items.get(item_id)
                    if not q_item:
                        continue

                    # 개별 영화용 config 사본 구성
                    job_cfg = dataclasses.replace if False else {
                        "stt": dict(self.config["stt"]),
                        "llm": dict(self.config["llm"]),
                        "subtitle": dict(self.config["subtitle"]),
                    }
                    if q_item.custom_hint:
                        base_hint = job_cfg["llm"].get("movie_context", "")
                        job_cfg["llm"]["movie_context"] = (
                            f"{base_hint} / {q_item.custom_hint}".strip(" /")
                        )

                    def update_item_status(status_text: str, iid: str = item_id) -> None:
                        if iid in self.queue_items:
                            self.queue_items[iid].status = status_text
                            self.root.after(0, self._refresh_tree_numbers)

                    def status_cb(
                        stage: str,
                        pct: float,
                        detail: str,
                        job_idx: int = idx,
                        iid: str = item_id,
                    ) -> None:
                        overall_pct = ((job_idx - 1) + (pct / 100.0)) / total_jobs * 100.0

                        def _ui():
                            self.queue_prog_label_var.set(
                                f"전체 대기열 진행률: {job_idx} / {total_jobs}편 ({overall_pct:.1f}%)"
                            )
                            self.queue_prog_var.set(overall_pct)
                            self.stage_label_var.set(f"{stage} ({pct:.1f}%)")
                            self.file_prog_var.set(pct)
                            self.detail_label_var.set(detail.splitlines()[0] if detail else "")

                        self.root.after(0, _ui)
                        update_item_status(f"{stage} ({pct:.0f}%)", iid)
                        if pct in (0.0, 100.0) or "완료" in stage:
                            self._append_log(f"[{q_item.video_path.name} | {stage}] {detail}")

                    self._append_log(f"\n---> [{idx}/{total_jobs}] '{q_item.video_path.name}' 처리 시작")
                    try:
                        out_files = process_video_file(
                            video_path=q_item.video_path,
                            config=job_cfg,
                            track_index=q_item.track_index,
                            language_override=q_item.language_override,
                            output_dir=output_dir,
                            skip_translation=skip_translation,
                            status_callback=status_cb,
                        )
                        q_item.generated_files = out_files
                        update_item_status("✅ 완료", item_id)
                        completed_count += 1
                    except Exception as exc:
                        logging.exception("파일 처리 중 오류: %s", q_item.video_path)
                        update_item_status("❌ 오류", item_id)
                        self._append_log(f"[오류 | {q_item.video_path.name}] {exc}")

                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "대기열 작업 완료",
                        f"총 {total_jobs}편 중 {completed_count}편의 자막 생성이 완료되었습니다!",
                    ),
                )
            finally:
                self.is_running = False
                self.stop_requested = False
                self.root.after(0, lambda: self.start_btn.configure(state="normal"))
                self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))

        threading.Thread(target=_worker, daemon=True).start()


def launch_gui() -> None:
    root = tk.Tk()
    SubtitleGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
