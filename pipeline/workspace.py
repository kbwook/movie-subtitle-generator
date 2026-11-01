"""개별 영화 작업 폴더(`<영화이름>_work/`) 및 중간 생산물(오디오 WAV, STT JSON, 번역 체크포인트) 관리 모듈.

역할:
1. 각 영상 파일 처리 시 `<영화이름>_work/` 전용 작업 폴더를 생성합니다.
2. 단계별 중간 생산물을 원자적(Atomic)으로 저장하고 로드합니다:
   - 1단계 산출물: `audio_16k.wav` + `audio_meta.json` (추출된 오디오 및 트랙 메타데이터)
   - 2단계 산출물: `stt_segments.json` + `<영화이름>.<언어>.orig.srt` (Whisper STT 원문 세그먼트)
   - 3단계 산출물: `translation_checkpoint.json` (배치 단위로 실시간 저장되는 LLM 번역 체크포인트)
   - 4단계 산출물: `<영화이름>.ko.srt`, `<영화이름>.ko.ass`, `perf_report.txt` (작업 폴더 및 원본 폴더에 저장)
3. 작업 중단(`JobCancelledError`) 후 재실행 시 이미 저장된 중간 생산물을 감지하여
   완료된 단계는 건너뛰고(Resume) 중단된 지점부터 즉시 이어서 진행합니다.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.models import AudioTrackInfo, SubtitleSegment


class JobCancelledError(Exception):
    """사용자가 '현재 작업 즉시 중단'을 요청했을 때 발생하는 예외."""


class MovieWorkspace:
    """단일 영화 파일의 작업 폴더 및 중간 생산물 입출력 관리자."""

    def __init__(self, video_path: Path, output_dir: Optional[Path] = None):
        self.video_path = Path(video_path).resolve()
        self.stem = self.video_path.stem
        self.base_dir = Path(output_dir).resolve() if output_dir else self.video_path.parent
        self.work_dir = self.base_dir / f"{self.stem}_work"

        # 단계별 중간 생산물 파일 경로
        self.wav_path = self.work_dir / "audio_16k.wav"
        self.wav_tmp_path = self.work_dir / "audio_16k.wav.part"
        self.audio_meta_path = self.work_dir / "audio_meta.json"

        self.stt_json_path = self.work_dir / "stt_segments.json"
        self.stt_partial_path = self.work_dir / "stt_segments.partial.json"

        self.trans_ckpt_path = self.work_dir / "translation_checkpoint.json"
        self.perf_report_path = self.work_dir / "perf_report.txt"

    def ensure_dir(self) -> Path:
        """작업 폴더를 생성하고 경로를 반환합니다."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir

    # -------------------------------------------------------------------------
    # 1단계: 오디오 WAV 및 트랙 메타데이터 저장/로드
    # -------------------------------------------------------------------------
    def has_valid_audio(self, expected_track_index: Optional[int] = None) -> bool:
        if not self.wav_path.exists() or not self.audio_meta_path.exists():
            return False
        if self.wav_path.stat().st_size < 1024:
            return False
        try:
            meta = json.loads(self.audio_meta_path.read_text(encoding="utf-8"))
            if expected_track_index is not None and meta.get("index") != expected_track_index:
                return False
            return True
        except Exception:
            return False

    def save_audio_meta(self, track: AudioTrackInfo) -> None:
        self.ensure_dir()
        self.audio_meta_path.write_text(
            json.dumps(dataclasses.asdict(track), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_audio_meta(self) -> AudioTrackInfo:
        data = json.loads(self.audio_meta_path.read_text(encoding="utf-8"))
        return AudioTrackInfo(**data)

    # -------------------------------------------------------------------------
    # 2단계: STT 원문 세그먼트 저장/로드 (부분 저장 및 완료 저장)
    # -------------------------------------------------------------------------
    def has_valid_stt(self) -> bool:
        return self.stt_json_path.exists() and self.stt_json_path.stat().st_size > 10

    def save_stt_segments(
        self,
        segments: List[SubtitleSegment],
        detected_lang: str,
        partial: bool = False,
    ) -> Path:
        self.ensure_dir()
        payload = {
            "video_name": self.video_path.name,
            "detected_language": detected_lang,
            "partial": partial,
            "segment_count": len(segments),
            "segments": [dataclasses.asdict(s) for s in segments],
        }
        target_path = self.stt_partial_path if partial else self.stt_json_path
        tmp_path = target_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(target_path)

        if not partial and self.stt_partial_path.exists():
            try:
                self.stt_partial_path.unlink()
            except OSError:
                pass
        return target_path

    def load_stt_segments(self) -> Tuple[List[SubtitleSegment], str]:
        data = json.loads(self.stt_json_path.read_text(encoding="utf-8"))
        detected_lang = data.get("detected_language", "en")
        segments = [SubtitleSegment(**item) for item in data.get("segments", [])]
        return segments, detected_lang

    # -------------------------------------------------------------------------
    # 3단계: 로컬 LLM 번역 체크포인트 저장/로드 (배치 단위 이어하기 지원)
    # -------------------------------------------------------------------------
    def save_translation_checkpoint(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        model_name: str,
    ) -> Path:
        self.ensure_dir()
        completed = sum(1 for s in segments if s.translated_text)
        payload = {
            "video_name": self.video_path.name,
            "source_language": source_lang,
            "target_language": target_lang,
            "model": model_name,
            "completed_count": completed,
            "total_count": len(segments),
            "segments": [dataclasses.asdict(s) for s in segments],
        }
        tmp_path = self.trans_ckpt_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.trans_ckpt_path)
        return self.trans_ckpt_path

    def load_translation_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.trans_ckpt_path.exists():
            return None
        try:
            data = json.loads(self.trans_ckpt_path.read_text(encoding="utf-8"))
            return data
        except Exception:
            return None

    def merge_translation_checkpoint(self, segments: List[SubtitleSegment]) -> Tuple[List[SubtitleSegment], int]:
        """기존 번역 체크포인트가 있으면 번역 완료된 문장들을 세그먼트에 복원하고 복원된 개수를 반환합니다."""
        ckpt = self.load_translation_checkpoint()
        if not ckpt:
            return segments, 0

        ckpt_map: Dict[int, str] = {}
        for item in ckpt.get("segments", []):
            sid = item.get("id")
            t_text = item.get("translated_text")
            if sid is not None and t_text:
                ckpt_map[int(sid)] = str(t_text)

        restored = 0
        for seg in segments:
            if seg.id in ckpt_map and ckpt_map[seg.id]:
                seg.translated_text = ckpt_map[seg.id]
                restored += 1

        return segments, restored

    # -------------------------------------------------------------------------
    # 상태 조회 및 초기화
    # -------------------------------------------------------------------------
    def inspect_resume_status(self) -> str:
        """현재 작업 폴더 내 중간 생산물 보유 상태를 사람이 읽기 쉬운 문자열로 반환합니다."""
        if not self.work_dir.exists():
            return "대기 중 (새 작업)"

        final_srt = self.work_dir / f"{self.stem}.ko.srt"
        if final_srt.exists():
            return "✅ 완료 (산출물 보유)"

        ckpt = self.load_translation_checkpoint()
        if ckpt:
            done = ckpt.get("completed_count", 0)
            total = ckpt.get("total_count", 0)
            if total > 0 and done >= total:
                return "이어하기 (3/4 번역 완료)"
            if done > 0:
                return f"이어하기 (번역 {done}/{total})"

        if self.has_valid_stt():
            try:
                data = json.loads(self.stt_json_path.read_text(encoding="utf-8"))
                cnt = data.get("segment_count", 0)
                return f"이어하기 (2/4 STT 완료: {cnt}문장)"
            except Exception:
                return "이어하기 (2/4 STT 완료)"

        if self.stt_partial_path.exists():
            try:
                data = json.loads(self.stt_partial_path.read_text(encoding="utf-8"))
                cnt = data.get("segment_count", 0)
                return f"중단됨 (1/4 오디오 완료, STT {cnt}문장)"
            except Exception:
                pass

        if self.has_valid_audio():
            return "이어하기 (1/4 오디오 추출됨)"

        return "대기 중 (작업 폴더 존재)"

    def clear_artifacts(self) -> None:
        """작업 폴더 내 중간 생산물을 모두 삭제하여 처음부터 다시 실행할 수 있게 합니다."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
