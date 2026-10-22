from __future__ import annotations

import unittest

from analyze_claim_text_qa_interval_key_bias_summary import BRANCHES, summarize


def _receipt() -> dict:
    rows = []
    for i in range(3):
        scores = {
            "baseline_qa_pixel_mask": [-1.0],
            "baseline_control_pixel_mask": [-0.8],
            "qa_bias_qa_pixel_mask": [-1.0],
            "qa_bias_control_pixel_mask": [-0.7],
            "sham_bias_qa_pixel_mask": [-1.0],
            "sham_bias_control_pixel_mask": [-0.75],
        }
        means = {name: values[0] for name, values in scores.items()}
        rows.append({
            "row_token_sha256": f"{i:064x}",
            "target_token_count": 1,
            "branch_token_logprobs_nats": scores,
            "branch_mean_logprob_nats_per_token": means,
            "primary": {
                "f_baseline_nats": 0.2,
                "f_qa_bias_nats": 0.3,
                "f_sham_bias_nats": 0.25,
                "delta_qa_bias_minus_baseline_nats": 0.1,
                "specificity_qa_minus_sham_bias_nats": 0.05,
            },
            "forward_count": {"attempted": 6, "completed": 6},
        })
    return {
        "status": "completed_exploratory_claim_text_qa_interval_key_bias_cohort",
        "mode": "full", "direct_faithfulness": False,
        "faithfulness_result": False,
        "requested_records": 3, "effective_records": 3,
        "requested_forwards": 18, "attempted_forwards": 18,
        "completed_forwards": 18, "failures": [], "frame_count": 32,
        "key_bias_windows": "fixed", "records": rows,
    }


class AggregateSummaryTests(unittest.TestCase):
    def test_recomputes_finite_contrasts_without_identifiers(self):
        result = summarize(_receipt(), "a" * 64)
        self.assertEqual(result["n_rows"], 3)
        self.assertAlmostEqual(
            result["primary_QA_bias_minus_baseline"]["mean_nat_per_token"], 0.1)
        self.assertAlmostEqual(
            result["specificity_QA_bias_minus_matched_sham"]["mean_nat_per_token"], 0.05)
        self.assertNotIn("video", str(result).lower())
        self.assertNotIn("claim_text", str(result).lower())

    def test_rejects_inconsistent_saved_primary(self):
        receipt = _receipt()
        receipt["records"][0]["primary"]["specificity_qa_minus_sham_bias_nats"] = 0.5
        with self.assertRaisesRegex(ValueError, "primary_arithmetic_mismatch"):
            summarize(receipt, "b" * 64)

    def test_rejects_direct_faithfulness_overclaim(self):
        receipt = _receipt()
        receipt["direct_faithfulness"] = True
        with self.assertRaisesRegex(ValueError, "full_receipt_contract_invalid"):
            summarize(receipt, "c" * 64)


if __name__ == "__main__":
    unittest.main()
