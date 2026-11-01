"""CPU / GPU / NPU 하드웨어 가속기 사용 현황 및 단계별 소요 시간(Performance Telemetry) 로깅 모듈.

지속적인 성능 개선(CPU vs Intel Arc GPU vs Intel AI Boost NPU 비교 분석)을 위해:
1. 시스템에 장착된 CPU, GPU, NPU 장치 및 가용 추론 백엔드(ONNXRuntime, CTranslate2, Ollama Vulkan 등)를 식별합니다.
2. 파이프라인의 4단계(1: 오디오 추출, 2: STT 음성 인식, 3: 로컬 LLM 번역, 4: 자막 포맷팅)별로
   - 실제로 사용된 연산 장치 (CPU / GPU / NPU 및 세부 백엔드)
   - 모델 로드 시간, 순수 추론/디코딩 시간, 전체 소요 시간
   - 처리 효율 지표 (실시간 대비 배속 RTF, 초당 토큰 생성 속도 t/s, 문장당 소요 시간)
   을 정밀 측정합니다.
3. 측정 결과를 콘솔/GUI 실시간 로그에 표 형태로 출력하고,
   `logs/perf_history.jsonl` 및 `logs/perf_report.log`에 누적 기록하여 전후 성능을 비교할 수 있게 합니다.
"""

from __future__ import annotations

import ctypes
import datetime
import json
import logging
import os
import winreg
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnxruntime as ort
import requests

logger = logging.getLogger(__name__)


@dataclass
class HardwareInventory:
    """현재 노트북의 CPU / GPU / NPU 하드웨어 및 추론 프로바이더 감지 정보."""
    cpu_name: str = "Unknown CPU"
    logical_cores: int = 0
    total_ram_gb: float = 0.0
    avail_ram_gb: float = 0.0
    gpus: List[str] = field(default_factory=list)
    npus: List[str] = field(default_factory=list)
    onnx_providers: List[str] = field(default_factory=list)


@dataclass
class StageMetric:
    """단일 파이프라인 단계의 하드웨어 사용 및 소요 시간 측정치."""
    stage_name: str
    accelerator_type: str          # 예: "CPU", "GPU (Vulkan)", "NPU", "CPU + GPU"
    backend_detail: str            # 예: "CTranslate2 INT8 (12 threads)", "Ollama (Vulkan GPU 100%)"
    elapsed_sec: float = 0.0
    model_load_sec: float = 0.0
    compute_sec: float = 0.0
    items_processed: int = 0       # 프레임 수, 대사 문장 수, 또는 토큰 수
    throughput_label: str = ""     # 예: "5.2x 실시간 배속", "18.4 tok/s"
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobTelemetryReport:
    """영화 1편 자막 생성 전체 과정의 종합 성능 리포트."""
    timestamp: str
    video_name: str
    video_codec: str
    audio_codec: str
    video_duration_sec: float
    detected_language: str
    hardware: HardwareInventory
    stages: List[StageMetric] = field(default_factory=list)
    total_elapsed_sec: float = 0.0
    optimization_notes: List[str] = field(default_factory=list)

    def format_summary_text(self) -> str:
        """GUI 로그창 및 CLI 콘솔에 출력할 가독성 높은 성능·하드웨어 분석 리포트."""
        lines = [
            "",
            "========================================================================",
            f" 📊 [성능 및 하드웨어 가속 리포트] {self.video_name}",
            "========================================================================",
            f" • 측정 시각      : {self.timestamp}",
            f" • 미디어 정보    : 영상={self.video_codec} / 오디오={self.audio_codec} / 길이={self.video_duration_sec:.1f}초 ({self.video_duration_sec/60:.1f}분)",
            f" • 감지된 하드웨어:",
            f"   - CPU : {self.hardware.cpu_name} ({self.hardware.logical_cores} 논리 코어)",
            f"   - GPU : {', '.join(self.hardware.gpus) if self.hardware.gpus else '감지 안 됨'}",
            f"   - NPU : {', '.join(self.hardware.npus) if self.hardware.npus else 'Intel AI Boost (드라이버/백엔드 미연결)'}",
            f"   - RAM : 가용 {self.hardware.avail_ram_gb:.1f} GB / 전체 {self.hardware.total_ram_gb:.1f} GB",
            "------------------------------------------------------------------------",
            " [단계별 사용 연산 장치(CPU/GPU/NPU) 및 소요 시간]",
        ]

        for s in self.stages:
            pct = (s.elapsed_sec / self.total_elapsed_sec * 100.0) if self.total_elapsed_sec > 0 else 0.0
            load_info = f" (로드 {s.model_load_sec:.2f}s / 연산 {s.compute_sec:.2f}s)" if s.model_load_sec > 0 else ""
            tp_info = f" | 처리속도: {s.throughput_label}" if s.throughput_label else ""
            lines.append(
                f"  ▸ {s.stage_name:<18} : {s.elapsed_sec:7.2f}초 ({pct:4.1f}%){load_info}"
            )
            lines.append(
                f"    └ 사용 장치: [{s.accelerator_type}] {s.backend_detail}{tp_info}"
            )

        lines.append("------------------------------------------------------------------------")
        overall_rtf = (
            f" (영화 길이 대비 {self.video_duration_sec / self.total_elapsed_sec:.2f}x 배속 처리)"
            if self.total_elapsed_sec > 0 and self.video_duration_sec > 0
            else ""
        )
        lines.append(
            f" ⏱ 총 소요 시간   : {self.total_elapsed_sec:.2f}초 ({self.total_elapsed_sec/60:.2f}분){overall_rtf}"
        )

        if self.optimization_notes:
            lines.append("------------------------------------------------------------------------")
            lines.append(" 💡 [향후 성능 개선 진단 및 제안]")
            for note in self.optimization_notes:
                lines.append(f"  * {note}")

        lines.append("========================================================================")
        return "\n".join(lines)


def _read_reg_str(root, subkey: str, value_name: str) -> Optional[str]:
    try:
        with winreg.OpenKey(root, subkey) as key:
            val, _ = winreg.QueryValueEx(key, value_name)
            return str(val).strip()
    except OSError:
        return None


def detect_hardware_inventory() -> HardwareInventory:
    """레지스트리 및 런타임 API를 통해 CPU, GPU, NPU 및 메모리 상태를 안전하게 조회합니다."""
    inv = HardwareInventory()
    inv.logical_cores = os.cpu_count() or 1

    cpu_name = _read_reg_str(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    if cpu_name:
        inv.cpu_name = cpu_name

    # 메모리 용량 확인
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        inv.total_ram_gb = round(m.ullTotalPhys / (1024 ** 3), 2)
        inv.avail_ram_gb = round(m.ullAvailPhys / (1024 ** 3), 2)
    except Exception:
        pass

    # 디스플레이 어댑터(GPU) 조회
    for idx in range(4):
        desc = _read_reg_str(
            winreg.HKEY_LOCAL_MACHINE,
            rf"SYSTEM\CurrentControlSet\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\000{idx}",
            "DriverDesc",
        )
        if desc and desc not in inv.gpus:
            inv.gpus.append(desc)

    # Intel Core Ultra (Meteor Lake / Lunar Lake) 내장 NPU 존재 여부 확인
    if "Ultra" in inv.cpu_name:
        inv.npus.append("Intel(R) AI Boost NPU (Core Ultra 내장)")

    try:
        inv.onnx_providers = list(ort.get_available_providers())
    except Exception:
        inv.onnx_providers = ["CPUExecutionProvider"]

    return inv


def query_ollama_processor_status(base_url: str, model_name: str) -> Dict[str, Any]:
    """Ollama `/api/ps` 엔드포인트를 조회하여 현재 모델이 CPU와 GPU(VRAM) 중 어디에 적재되었는지 확인합니다."""
    info: Dict[str, Any] = {
        "accelerator": "CPU",
        "gpu_offload_pct": 0.0,
        "size_total_mb": 0.0,
        "size_vram_mb": 0.0,
        "processor_summary": "Ollama 상태 미확인",
    }
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        target_prefix = model_name.split(":")[0]

        matched = None
        for m in models:
            name = m.get("name", "")
            if name == model_name or name.startswith(target_prefix):
                matched = m
                break
        if not matched and models:
            matched = models[0]

        if matched:
            size_total = float(matched.get("size", 0))
            size_vram = float(matched.get("size_vram", 0))
            info["size_total_mb"] = round(size_total / (1024 ** 2), 1)
            info["size_vram_mb"] = round(size_vram / (1024 ** 2), 1)

            if size_total > 0 and size_vram > 0:
                pct = round((size_vram / size_total) * 100.0, 1)
                info["gpu_offload_pct"] = pct
                if pct >= 95.0:
                    info["accelerator"] = "GPU (Intel Arc iGPU)"
                else:
                    info["accelerator"] = f"GPU+CPU 하이브리드 (GPU {pct}%)"
                info["processor_summary"] = (
                    f"Ollama GPU 오프로딩 {pct}% (VRAM {info['size_vram_mb']:.0f}MB / 전체 {info['size_total_mb']:.0f}MB)"
                )
            else:
                info["accelerator"] = "CPU (100% 시스템 RAM)"
                info["processor_summary"] = (
                    f"Ollama CPU 단독 구동 중 (GPU VRAM 적재 0MB / 전체 {info['size_total_mb']:.0f}MB)"
                )
    except Exception as exc:
        logger.debug("Ollama /api/ps 조회 실패: %s", exc)

    return info


def save_telemetry_report(report: JobTelemetryReport, project_root: Optional[Path] = None) -> Path:
    """성능 리포트를 `logs/perf_history.jsonl` 및 `logs/perf_report.log`에 영구 기록합니다."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / "perf_history.jsonl"
    txt_path = logs_dir / "perf_report.log"

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")

    summary_text = report.format_summary_text()
    with open(txt_path, "a", encoding="utf-8") as tf:
        tf.write(summary_text + "\n")

    return txt_path
