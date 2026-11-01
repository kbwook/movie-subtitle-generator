"""다양한 코덱의 영화 파일로부터 미디어 정보 및 오디오 트랙을 검색하고 16kHz Mono WAV로 추출하는 모듈.

내장 PyAV(FFmpeg C 라이브러리 바인딩)를 기본 사용하여 시스템에 별도 FFmpeg가 설치되어 있지 않아도
MKV, MP4, AVI, TS, MOV 등의 컨테이너와 AAC, AC3, E-AC3, DTS, FLAC, MP3, Opus 등
거의 모든 오디오 코덱을 디코딩합니다. 시스템 FFmpeg가 있으면 고속 CLI 경로도 지원합니다.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Callable, List, Optional

import av
import numpy as np

from pipeline.models import AudioTrackInfo, VideoMediaInfo

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m2ts", ".mpg", ".mpeg", ".m4v"
}

LANGUAGE_ALIASES = {
    "zh": {"zh", "chi", "zho", "cmn", "yue", "chs", "cht", "chinese"},
    "ja": {"ja", "jpn", "japanese"},
    "en": {"en", "eng", "english"},
    "ko": {"ko", "kor", "korean"},
}


def inspect_video_metadata(video_path: Path) -> VideoMediaInfo:
    """영상 파일의 비디오 코덱, 해상도, 재생 시간 및 오디오 스트림 목록을 종합 조회합니다."""
    video_path = Path(video_path).resolve()
    file_size = video_path.stat().st_size if video_path.exists() else 0
    tracks: List[AudioTrackInfo] = []
    video_codec = "unknown"
    resolution = "-"
    duration_sec = 0.0

    with av.open(str(video_path)) as container:
        if container.duration:
            duration_sec = float(container.duration / av.time_base)

        if container.streams.video:
            v_stream = container.streams.video[0]
            if v_stream.codec_context:
                video_codec = (v_stream.codec_context.name or "unknown").upper()
                w = v_stream.codec_context.width or 0
                h = v_stream.codec_context.height or 0
                if w and h:
                    resolution = f"{w}x{h}"

        for idx, stream in enumerate(container.streams.audio):
            metadata = stream.metadata or {}
            lang = (
                metadata.get("language")
                or metadata.get("LANGUAGE")
                or metadata.get("lang")
                or "und"
            )
            title = metadata.get("title") or metadata.get("TITLE") or ""
            codec_name = stream.codec_context.name if stream.codec_context else "unknown"
            channels = stream.codec_context.channels if stream.codec_context else 0
            sample_rate = stream.codec_context.sample_rate if stream.codec_context else 0
            tracks.append(
                AudioTrackInfo(
                    index=idx,
                    stream_index=stream.index,
                    codec_name=codec_name,
                    language=lang.lower(),
                    title=title,
                    channels=channels,
                    sample_rate=sample_rate,
                )
            )

    return VideoMediaInfo(
        path=video_path,
        file_size_bytes=file_size,
        duration_sec=duration_sec,
        video_codec=video_codec,
        resolution=resolution,
        audio_tracks=tracks,
    )


def list_audio_tracks(video_path: Path) -> List[AudioTrackInfo]:
    """영상 파일 내의 모든 오디오 스트림 정보를 조회합니다."""
    return inspect_video_metadata(video_path).audio_tracks


def select_best_audio_track(
    tracks: List[AudioTrackInfo],
    preferred_language: str = "auto",
    track_index: Optional[int] = None,
) -> AudioTrackInfo:
    """선호 언어 또는 지정된 인덱스에 따라 최적의 오디오 트랙을 선택합니다."""
    if not tracks:
        raise ValueError("영상 파일에 오디오 트랙이 존재하지 않습니다.")

    if track_index is not None:
        if 0 <= track_index < len(tracks):
            return tracks[track_index]
        raise ValueError(f"유효하지 않은 오디오 트랙 번호입니다: {track_index} (전체 {len(tracks)}개)")

    if preferred_language and preferred_language != "auto":
        aliases = LANGUAGE_ALIASES.get(preferred_language.lower(), {preferred_language.lower()})
        for track in tracks:
            if track.language in aliases:
                return track

    # 중·일·영 트랙이 태그되어 있으면 우선 선택
    target_langs = LANGUAGE_ALIASES["zh"] | LANGUAGE_ALIASES["ja"] | LANGUAGE_ALIASES["en"]
    for track in tracks:
        if track.language in target_langs:
            return track

    return tracks[0]


def extract_audio_to_wav(
    video_path: Path,
    output_wav_path: Path,
    preferred_language: str = "auto",
    track_index: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[Path, AudioTrackInfo]:
    """선택된 오디오 트랙을 Whisper 최적 포맷(16kHz, Mono, 16-bit PCM WAV)으로 추출합니다."""
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = list_audio_tracks(video_path)
    selected_track = select_best_audio_track(tracks, preferred_language, track_index)
    logger.info("선택된 오디오 트랙: %s", selected_track.display_label())

    # 시스템에 ffmpeg 실행 파일이 있는 경우 고속 서브프로세스 추출 시도
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(video_path),
                "-map", f"0:a:{selected_track.index}",
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_wav_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if progress_callback:
                progress_callback(100.0)
            return output_wav_path, selected_track
        except Exception as exc:
            logger.warning("시스템 FFmpeg 호출 실패, 내장 PyAV 디코더로 전환합니다: %s", exc)

    # 내장 PyAV 디코더 + 리샘플러 (외부 FFmpeg 설치 불필요)
    with av.open(str(video_path)) as container:
        audio_stream = container.streams.audio[selected_track.index]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        duration_sec = float(container.duration / av.time_base) if container.duration else 0.0

        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(16000)

            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        pcm_bytes = r_frame.to_ndarray().astype(np.int16).tobytes()
                        wav_file.writeframes(pcm_bytes)

                    if progress_callback and duration_sec > 0 and frame.pts and frame.time_base:
                        current_sec = float(frame.pts * frame.time_base)
                        pct = min(100.0, (current_sec / duration_sec) * 100.0)
                        progress_callback(pct)

            # 남은 리샘플러 버퍼 플러시
            for r_frame in resampler.resample(None):
                pcm_bytes = r_frame.to_ndarray().astype(np.int16).tobytes()
                wav_file.writeframes(pcm_bytes)

    if progress_callback:
        progress_callback(100.0)

    return output_wav_path, selected_track
