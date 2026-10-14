from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_necessity_choice_manifest as preflight  # noqa: E402


class DistinctFramePreflightTests(unittest.TestCase):
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
