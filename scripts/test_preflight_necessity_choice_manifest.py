from __future__ import annotations

import sys
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_necessity_choice_manifest as preflight  # noqa: E402


class DistinctFramePreflightTests(unittest.TestCase):
    def _mock_capture(self, fps):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = fps
        cap.read.side_effect = [(True, object()), (True, object()), (False, None)]
        module = SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_FPS=5)
        return cap, module

    def test_invalid_source_fps_is_rejected_and_capture_released(self):
        for fps in (0.0, -1.0, float("nan"), float("inf"), None, True, "30"):
            with self.subTest(fps=fps):
                cap, cv2 = self._mock_capture(fps)
                with patch.dict(sys.modules, {"cv2": cv2}):
                    with self.assertRaisesRegex(RuntimeError, "invalid source FPS"):
                        preflight.decodable_video_info("synthetic.mp4")
                cap.release.assert_called_once()
                cap.read.assert_not_called()

    def test_valid_source_fps_is_returned_without_substitution(self):
        cap, cv2 = self._mock_capture(29.97)
        with patch.dict(sys.modules, {"cv2": cv2}):
            self.assertEqual(preflight.decodable_video_info("synthetic.mp4"), (2, 29.97))
        cap.release.assert_called_once()

    def test_capture_is_released_if_fps_metadata_read_raises(self):
        cap, cv2 = self._mock_capture(29.97)
        cap.get.side_effect = RuntimeError("synthetic metadata error")
        with patch.dict(sys.modules, {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "synthetic metadata error"):
                preflight.decodable_video_info("synthetic.mp4")
        cap.release.assert_called_once()

    def test_capture_is_released_if_open_fails(self):
        cap, cv2 = self._mock_capture(29.97)
        cap.isOpened.return_value = False
        with patch.dict(sys.modules, {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "Could not open video"):
                preflight.decodable_video_info("synthetic.mp4")
        cap.release.assert_called_once()
        cap.get.assert_not_called()
        cap.read.assert_not_called()

    def test_short_clip_with_duplicate_sampled_indices_is_excluded_pre_score(self):
        sample = {"video_id": "synthetic-video", "qid": "q1", "video": "synthetic.mp4",
                  "grounding": [[0.0, 0.0]]}
        with patch.object(preflight, "decodable_video_info", return_value=(15, 1.0)):
            selected, records = preflight.audit_samples([sample], num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual(records[0]["status"], "insufficient_distinct_sample_frames")
        self.assertEqual(records[0]["requested_frame_count"], 16)
        self.assertEqual(records[0]["sampled_frame_count"], 15)
        self.assertEqual(len(records[0]["sampled_indices"]), 15)
        self.assertEqual(records[0]["evidence_indices"], [])
        self.assertEqual(records[0]["control_indices"], [])

    def test_full_distinct_sample_can_proceed_to_coverage_check(self):
        sample = {"video_id": "synthetic-video", "qid": "q1", "video": "synthetic.mp4",
                  "grounding": [[0.0, 0.0]]}
        with patch.object(preflight, "decodable_video_info", return_value=(16, 1.0)):
            selected, records = preflight.audit_samples([sample], num_frames=16)

        self.assertEqual(selected, [sample])
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["requested_frame_count"], 16)
        self.assertEqual(records[0]["sampled_frame_count"], 16)
        self.assertEqual(len(records[0]["evidence_indices"]), 1)
        self.assertEqual(len(records[0]["control_indices"]), 1)

    def test_invalid_assignment_geometry_has_distinct_fail_closed_status(self):
        sample = {"video_id": "synthetic-video", "qid": "q1", "video": "synthetic.mp4",
                  "grounding": [[2.0, 1.0]]}
        with patch.object(preflight, "decodable_video_info", return_value=(16, 1.0)):
            selected, records = preflight.audit_samples([sample], num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual(records[0]["status"], "invalid_assignment_geometry")
        self.assertIn("nonnegative-duration", records[0]["error"])

    def test_decoder_failure_remains_separate_from_assignment_geometry(self):
        sample = {"video_id": "synthetic-video", "qid": "q1", "video": "synthetic.mp4",
                  "grounding": [[0.0, 0.0]]}
        with patch.object(preflight, "decodable_video_info",
                          side_effect=RuntimeError("synthetic decoder failure")):
            selected, records = preflight.audit_samples([sample], num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual(records[0]["status"], "video_or_metadata_error")
        self.assertIn("decoder failure", records[0]["error"])

    def test_missing_grounding_is_classified_as_invalid_assignment_geometry(self):
        sample = {"video_id": "synthetic-video", "qid": "q1", "video": "synthetic.mp4"}
        with patch.object(preflight, "decodable_video_info", return_value=(16, 1.0)):
            selected, records = preflight.audit_samples([sample], num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual(records[0]["status"], "invalid_assignment_geometry")
        self.assertIn("intervals must be a list", records[0]["error"])

    def test_cli_counts_geometry_failures_and_writes_only_valid_selection(self):
        valid = {"video_id": "synthetic-valid", "qid": "q1", "video": "valid.mp4",
                 "grounding": [[0.0, 0.0]]}
        invalid = {"video_id": "synthetic-invalid", "qid": "q2", "video": "invalid.mp4",
                   "grounding": [[2.0, 1.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "input.json"
            selected_path = root / "selected.json"
            smoke_path = root / "smoke.json"
            audit_path = root / "audit.json"
            samples_path.write_text(json.dumps([valid, invalid, None]))
            argv = ["preflight", "--samples-file", str(samples_path), "--n-valid", "1",
                    "--n-smoke", "0", "--output-samples", str(selected_path),
                    "--smoke-samples-output", str(smoke_path), "--audit-output", str(audit_path)]
            with patch.object(sys, "argv", argv), patch.object(
                    preflight, "decodable_video_info", return_value=(16, 1.0)), \
                    redirect_stdout(StringIO()):
                preflight.main()

            report = json.loads(audit_path.read_text())
            self.assertEqual(report["status_counts"], {
                "invalid_assignment_geometry": 1, "invalid_manifest_record": 1, "ok": 1})
            self.assertEqual(json.loads(selected_path.read_text()), [valid])
            self.assertEqual(json.loads(smoke_path.read_text()), [])

    def test_invalid_manifest_rows_are_recorded_without_aborting_audit(self):
        malformed = [None, 17, "bad", {}, {"video": "no-id.mp4"},
                     {"video_id": " padded ", "video": "padded.mp4"}]
        selected, records = preflight.audit_samples(malformed, num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual([record["status"] for record in records],
                         ["invalid_manifest_record"] * len(malformed))
        self.assertEqual([record["sample_id"] for record in records],
                         [f"{index}:{index}" for index in range(len(malformed))])

    def test_duplicate_video_rows_are_rejected_before_video_decode(self):
        samples = [
            {"video_id": "same-video", "qid": "q1", "video": "a.mp4",
             "grounding": [[0.0, 0.0]]},
            {"video_id": "same-video", "qid": "q2", "video": "b.mp4",
             "grounding": [[1.0, 1.0]]},
        ]
        with patch.object(preflight, "decodable_video_info") as decode:
            selected, records = preflight.audit_samples(samples, num_frames=16)

        self.assertEqual(selected, [])
        self.assertEqual([record["status"] for record in records],
                         ["duplicate_video_id", "duplicate_video_id"])
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
