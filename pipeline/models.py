"""자막 세그먼트, 오디오 트랙 및 비디오 미디어 메타데이터 모델."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AudioTrackInfo:
    """컨테이너 내 개별 오디오 스트림 메타데이터."""
    index: int
    stream_index: int
    codec_name: str
    language: str
    title: str
    channels: int
    sample_rate: int

    def display_label(self) -> str:
        lang = self.language or "und"
        title_part = f" - {self.title}" if self.title else ""
        return (
            f"트랙 #{self.index} (스트림 {self.stream_index}): "
            f"[{lang}] {self.codec_name.upper()} / {self.channels}ch / {self.sample_rate}Hz{title_part}"
        )

    def short_label(self) -> str:
        lang = self.language or "und"
        return f"#{self.index}[{lang}:{self.codec_name.upper()}]"


@dataclass
class VideoMediaInfo:
    """영화 파일의 컨테이너, 비디오 코덱, 길이 및 오디오 트랙 종합 정보."""
    path: Path
    file_size_bytes: int
    duration_sec: float
    video_codec: str
    resolution: str
    audio_tracks: List[AudioTrackInfo] = field(default_factory=list)

    def format_duration(self) -> str:
        total = max(0, int(round(self.duration_sec)))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def format_size(self) -> str:
        mb = self.file_size_bytes / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.2f} GB"
        return f"{mb:.1f} MB"

    def audio_summary(self) -> str:
        if not self.audio_tracks:
            return "오디오 없음"
        joined = ", ".join(t.short_label() for t in self.audio_tracks)
        return f"{len(self.audio_tracks)}개 ({joined})"


@dataclass
class SubtitleSegment:
    """단일 자막 구간(타임스탬프, 원문, 번역문)."""
    id: int
    start: float  # 초 단위 시작 시간
    end: float    # 초 단위 종료 시간
    original_text: str
    translated_text: Optional[str] = None
