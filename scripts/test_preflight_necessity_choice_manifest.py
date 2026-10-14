from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
