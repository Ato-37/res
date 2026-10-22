from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_claim_likelihood_human_review_v2 import build, PRIVATE


class ReviewArtifactV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PRIVATE)
        self.root = Path(self.tmp.name)
        video = self.root / "local.mp4"
        video.write_bytes(b"fixture video bytes")
        video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
        records, result_rows = [], []
        for i in range(3):
            claim = f"Claim {i}."
            claim_hash = hashlib.sha256(claim.encode()).hexdigest()
            pts = [float(n) for n in range(32)]
            records.append({"generated_parsed": {"claim": claim},
                            "sampled_frame_relative_pts_seconds": pts,
                            "local_video_path": str(video), "video_sha256": video_hash,
                            "candidate_pair_sha256": f"{i:064x}", "question": f"Question {i}?"})
            result_rows.append({"source_video_sha256": video_hash,
                                "saved_claim_sha256": claim_hash})
        self.packet = self.root / "packet.json"
        self.result = self.root / "result.json"
        self.output = self.root / "artifact.html"
        self.packet.write_text(json.dumps({"schema_version": 1, "frames_per_video": 32,
                                           "effective_videos": 3, "records": records}))
        self.result.write_text(json.dumps({"completed_forwards": 18, "attempted_forwards": 18,
                                          "frame_count": 32, "cohort_sha256":
                                          hashlib.sha256(self.packet.read_bytes()).hexdigest(),
                                          "records": result_rows}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_binds_frozen_inputs_and_displays_32_indices(self):
        with patch("build_claim_likelihood_human_review_v2.frame_jpeg", return_value=b"\xff\xd8x"):
            build(self.packet, self.result, self.output)
        html = self.output.read_text()
        marker = 'name="review-artifact-sha256" content="'
        begin = html.index(marker) + len(marker)
        fingerprint = html[begin:begin + 64]
        canonical = html[:begin] + "__FP__" + html[begin + 64:]
        self.assertEqual(fingerprint, hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(os.stat(self.output).st_mode & 0o777, 0o600)
        self.assertIn('packet_sha256', html)
        self.assertIn('result_sha256', html)
        self.assertIn('model_frame_indices', html)
        self.assertIn('respondents', html)
        self.assertIn('次のレビュー担当者', html)
        self.assertIn('frame index ${x.model_frame_indices[i]}', html)
        self.assertIn('"model_frame_indices":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]', html)

    def test_refuses_misaligned_claim_binding(self):
        result = json.loads(self.result.read_text())
        result["records"][0]["saved_claim_sha256"] = "0" * 64
        self.result.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            build(self.packet, self.result, self.output)

    def test_refuses_output_outside_private_root(self):
        with self.assertRaisesRegex(ValueError, "must remain"):
            build(self.packet, self.result, Path("/tmp/review-v2.html"))

    def test_refuses_overwrite_and_preserves_existing_artifact(self):
        self.output.write_text("existing", encoding="utf-8")
        with patch("build_claim_likelihood_human_review_v2.frame_jpeg",
                   return_value=b"\xff\xd8x"):
            with self.assertRaises(FileExistsError):
                build(self.packet, self.result, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
