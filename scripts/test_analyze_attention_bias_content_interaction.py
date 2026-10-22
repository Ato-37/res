from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_attention_bias_content_interaction import summarize


class InteractionSummaryTests(unittest.TestCase):
    def test_recomputes_interaction_and_excludes_failed_receipt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = {
                "status": "completed_one_row_posthoc_attention_bias_content_interaction",
                "direct_faithfulness": False,
                "human_review": {"review_actually_performed": False},
                "requested_records": 1, "effective_records": 1,
                "requested_forwards": 1, "attempted_forwards": 1, "completed_forwards": 1,
                "row_token_sha256": "a" * 64,
                "reused_frozen_receipts": {"original_baseline_mean": -1.0,
                    "original_bias_mean": -0.8, "gray_baseline_mean": -1.5},
                "new_gray_video_biased_score": {"target_logprob_mean": -1.6},
                "primary_descriptive_interaction": {
                    "baseline_original_minus_gray_nats_per_token": .5,
                    "biased_original_minus_gray_nats_per_token": .8,
                    "bias_by_video_content_interaction_nats_per_token": .3}}
            good = root / "good.json"; good.write_text(json.dumps(valid))
            bad = root / "failed.json"; bad.write_text(json.dumps({"status": "failed", "completed_forwards": 0}))
            result = summarize([good, bad])
            self.assertEqual(result["completed_rows"], 1)
            self.assertAlmostEqual(result["interaction_mean"], .3)
            self.assertEqual(len(result["excluded_receipts"]), 1)
            self.assertFalse(result["direct_faithfulness"])

    def test_rejects_duplicate_rows_or_bad_arithmetic(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            payload = {"status": "completed_one_row_posthoc_attention_bias_content_interaction",
                "direct_faithfulness": False, "human_review": {"review_actually_performed": False},
                "requested_records": 1, "effective_records": 1, "requested_forwards": 1,
                "attempted_forwards": 1, "completed_forwards": 1, "row_token_sha256": "b" * 64,
                "reused_frozen_receipts": {"original_baseline_mean": -1.0, "original_bias_mean": -.8,
                    "gray_baseline_mean": -1.5}, "new_gray_video_biased_score": {"target_logprob_mean": -1.6},
                "primary_descriptive_interaction": {"baseline_original_minus_gray_nats_per_token": .5,
                    "biased_original_minus_gray_nats_per_token": .8,
                    "bias_by_video_content_interaction_nats_per_token": .31}}
            p.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "arithmetic_mismatch"):
                summarize([p])


if __name__ == "__main__":
    unittest.main()
