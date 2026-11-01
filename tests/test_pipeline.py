"""자막 생성 파이프라인 및 GUI 대기열 관리 단위/통합 검증 테스트."""

from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import av
import numpy as np

from gui import SubtitleGeneratorGUI
from pipeline.extractor import (
    extract_audio_to_wav,
    inspect_video_metadata,
    list_audio_tracks,
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
from pipeline.transcriber import load_wav_as_float32
from pipeline.translator import LocalLLMTranslator
from pipeline.workspace import JobCancelledError, MovieWorkspace


def _create_dummy_mkv_with_multi_audio(path: Path) -> None:
    """테스트용 다국어(ja, zh) 멀티 오디오 트랙을 가진 MKV 컨테이너 파일을 생성합니다."""
    with av.open(str(path), mode="w") as container:
        v_stream = container.add_stream("libx264", rate=24)
        v_stream.width = 320
        v_stream.height = 240
        v_stream.pix_fmt = "yuv420p"

        a_stream_ja = container.add_stream("aac", rate=48000)
        a_stream_ja.layout = "stereo"
        a_stream_ja.metadata["language"] = "jpn"
        a_stream_ja.metadata["title"] = "Japanese Stereo"

        a_stream_zh = container.add_stream("ac3", rate=48000)
        a_stream_zh.layout = "stereo"
        a_stream_zh.metadata["language"] = "zho"
        a_stream_zh.metadata["title"] = "Chinese AC3"

        for _ in range(24):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            v_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in v_stream.encode(v_frame):
                container.mux(pkt)

        for stream in (a_stream_ja, a_stream_zh):
            samples = np.zeros((2, 1024), dtype=np.float32)
            for _ in range(45):
                a_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
                a_frame.sample_rate = 48000
                for pkt in stream.encode(a_frame):
                    container.mux(pkt)

        for pkt in v_stream.encode():
            container.mux(pkt)
        for pkt in a_stream_ja.encode():
            container.mux(pkt)
        for pkt in a_stream_zh.encode():
            container.mux(pkt)


class PipelineTest(unittest.TestCase):
    def test_multi_codec_audio_extraction_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mkv_path = Path(tmp) / "test_movie.mkv"
            wav_path = Path(tmp) / "extracted.wav"
            _create_dummy_mkv_with_multi_audio(mkv_path)

            meta = inspect_video_metadata(mkv_path)
            self.assertEqual(meta.video_codec, "H264")
            self.assertEqual(meta.resolution, "320x240")
            self.assertEqual(len(meta.audio_tracks), 2)

            tracks = list_audio_tracks(mkv_path)
            self.assertEqual(tracks[0].language, "jpn")
            self.assertEqual(tracks[1].language, "zho")

            zh_track = select_best_audio_track(tracks, preferred_language="zh")
            self.assertEqual(zh_track.index, 1)

            out_wav, chosen = extract_audio_to_wav(mkv_path, wav_path, preferred_language="ja")
            self.assertTrue(out_wav.exists())
            self.assertGreater(out_wav.stat().st_size, 16000)
            self.assertEqual(chosen.index, 0)

            # faster-whisper 호환 float32 배열 로드 및 PyAV 19 metadata_errors 패치 검증
            waveform = load_wav_as_float32(out_wav)
            self.assertEqual(waveform.dtype, np.float32)
            self.assertEqual(waveform.ndim, 1)
            self.assertGreater(len(waveform), 8000)

            with av.open(str(out_wav), mode="r", metadata_errors="ignore") as c:
                self.assertEqual(len(c.streams.audio), 1)

    def test_llm_json_parsing_and_formatting(self) -> None:
        raw_json = """```json
        {
          "translations": [
            {"id": 1, "translation": "오랜만이야, 잘 지냈어?"},
            {"id": 2, "translation": "응, 덕분에 무사히 도착했어."}
          ]
        }
        ```"""
        parsed = LocalLLMTranslator._parse_json_response(raw_json)
        self.assertEqual(parsed[1], "오랜만이야, 잘 지냈어?")
        self.assertEqual(parsed[2], "응, 덕분에 무사히 도착했어.")

        segments = [
            SubtitleSegment(id=1, start=1.0, end=3.5, original_text="好久不见，你还好吗？", translated_text=parsed[1]),
            SubtitleSegment(id=2, start=3.2, end=5.8, original_text="嗯，托你的福平安到了。", translated_text=parsed[2]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            srt_path = Path(tmp) / "movie.ko.srt"
            ass_path = Path(tmp) / "movie.ko.ass"
            write_srt(segments, srt_path, bilingual=True)
            write_ass(segments, ass_path, bilingual=True)

            srt_text = srt_path.read_text(encoding="utf-8-sig")
            ass_text = ass_path.read_text(encoding="utf-8-sig")
            self.assertIn("오랜만이야, 잘 지냈어?", srt_text)
            self.assertIn("好久不见，你还好吗？", srt_text)
            self.assertIn("OriginalSub", ass_text)

    def test_gui_queue_management(self) -> None:
        root = tk.Tk()
        root.withdraw()
        try:
            app = SubtitleGeneratorGUI(root)
            with tempfile.TemporaryDirectory() as tmp:
                mkv1 = Path(tmp) / "movie_jp.mkv"
                mkv2 = Path(tmp) / "movie_zh.mkv"
                _create_dummy_mkv_with_multi_audio(mkv1)
                _create_dummy_mkv_with_multi_audio(mkv2)

                app._add_paths_to_queue([mkv1, mkv2])
                self.assertEqual(len(app.queue_order), 2)

                # 첫 번째 항목 선택 및 개별 오버라이드 적용 테스트
                first_id = app.queue_order[0]
                app.tree.selection_set(first_id)
                app._on_tree_select()
                app.item_track_combo.current(2)  # 트랙 #1 (Chinese AC3)
                app.item_lang_combo.current(2)   # zh (중국어)
                app.item_hint_var.set("무협 시대극")
                app._on_apply_item_override()

                q_item = app.queue_items[first_id]
                self.assertEqual(q_item.track_index, 1)
                self.assertEqual(q_item.language_override, "zh")
                self.assertEqual(q_item.custom_hint, "무협 시대극")

                # 순서 아래로 이동 테스트
                app._on_move_item(1)
                self.assertEqual(app.queue_order[1], first_id)

                # 전체 비우기 테스트
                app._on_clear_all()
                self.assertEqual(len(app.queue_order), 0)
        finally:
            root.destroy()

    def test_hardware_telemetry_and_report_logging(self) -> None:
        hw = detect_hardware_inventory()
        self.assertGreater(hw.logical_cores, 0)
        self.assertGreater(hw.total_ram_gb, 0.0)

        report = JobTelemetryReport(
            timestamp="2026-11-01 12:55:00",
            video_name="sample_movie.mkv",
            video_codec="H264",
            audio_codec="AAC",
            video_duration_sec=3600.0,
            detected_language="fr",
            hardware=hw,
            stages=[
                StageMetric(
                    stage_name="1/4 오디오 추출",
                    accelerator_type="CPU",
                    backend_detail="PyAV (16kHz Mono PCM)",
                    elapsed_sec=8.5,
                    compute_sec=8.5,
                    throughput_label="423.5x 실시간 배속",
                ),
                StageMetric(
                    stage_name="2/4 음성 인식 (STT)",
                    accelerator_type="CPU",
                    backend_detail="Faster-Whisper INT8 (12 threads)",
                    elapsed_sec=600.0,
                    model_load_sec=2.1,
                    compute_sec=597.9,
                    items_processed=850,
                    throughput_label="6.0x 실시간 배속",
                ),
            ],
            total_elapsed_sec=608.5,
            optimization_notes=["STT 단계 OpenVINO GPU/NPU 전환 가능"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            txt_path = save_telemetry_report(report, project_root=Path(tmp))
            self.assertTrue(txt_path.exists())
            content = txt_path.read_text(encoding="utf-8")
            self.assertIn("성능 및 하드웨어 가속 리포트", content)
            self.assertIn("Faster-Whisper INT8", content)

    def test_workspace_intermediate_artifacts_and_cancel(self) -> None:
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            mkv_path = Path(tmp) / "sample_film.mkv"
            _create_dummy_mkv_with_multi_audio(mkv_path)

            ws = MovieWorkspace(mkv_path)
            self.assertEqual(ws.inspect_resume_status(), "대기 중 (새 작업)")

            # 1단계: 오디오 추출 및 메타 저장 검증
            out_wav, track = extract_audio_to_wav(mkv_path, ws.wav_path, preferred_language="ja")
            ws.save_audio_meta(track)
            self.assertTrue(ws.has_valid_audio(expected_track_index=0))
            self.assertEqual(ws.inspect_resume_status(), "이어하기 (1/4 오디오 추출됨)")

            # 2단계: STT 세그먼트 저장 및 복원 검증
            segs = [
                SubtitleSegment(1, 0.0, 2.0, "Bonjour"),
                SubtitleSegment(2, 2.5, 4.5, "Comment ça va?"),
            ]
            ws.save_stt_segments(segs, "fr", partial=False)
            self.assertTrue(ws.has_valid_stt())
            self.assertIn("2/4 STT 완료: 2문장", ws.inspect_resume_status())

            loaded_segs, lang = ws.load_stt_segments()
            self.assertEqual(lang, "fr")
            self.assertEqual(len(loaded_segs), 2)

            # 3단계: 부분 번역 체크포인트 저장 및 이어하기 병합 검증
            loaded_segs[0].translated_text = "안녕하세요"
            ws.save_translation_checkpoint(loaded_segs, "fr", "한국어", "qwen2.5:7b")
            self.assertEqual(ws.inspect_resume_status(), "이어하기 (번역 1/2)")

            fresh_segs, _ = ws.load_stt_segments()
            merged_segs, restored_cnt = ws.merge_translation_checkpoint(fresh_segs)
            self.assertEqual(restored_cnt, 1)
            self.assertEqual(merged_segs[0].translated_text, "안녕하세요")
            self.assertIsNone(merged_segs[1].translated_text)

            # 즉시 중단(cancel_event) 동작 검증
            cancel_ev = threading.Event()
            cancel_ev.set()
            with self.assertRaises(JobCancelledError):
                extract_audio_to_wav(
                    mkv_path,
                    Path(tmp) / "aborted.wav",
                    cancel_event=cancel_ev,
                )


if __name__ == "__main__":
    unittest.main()
