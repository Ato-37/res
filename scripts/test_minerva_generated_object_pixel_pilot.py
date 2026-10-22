"""Focused CPU-only contract tests for the generated pixel pilot runner."""
import unittest

from run_minerva_generated_object_pixel_pilot import (
    _generated_target,
    _prompt,
    _record_for_row,
    _validate_proxy_anchor_binding,
    compute_pixel_bias_interaction,
)


class GeneratedTargetTests(unittest.TestCase):
    def test_proxy_anchor_must_match_official_frame_and_safe_pts(self):
        cohort = {"candidate_preflight_status": "metadata_only_proxy_not_preflighted",
                  "rationale_interval_boundary_margin_seconds": 0.05}
        selected = {"proxy_anchor": {"frame_number": 303,
                                      "interval_seconds": [10, 11]}}
        actual = {"mask_frame_number": 303,
                  "mask_time_seconds": 10.1,
                  "timestamp_support_interval_seconds": [10, 11]}
        _validate_proxy_anchor_binding(cohort, selected, actual)
        with self.assertRaisesRegex(ValueError, "proxy_anchor_differs"):
            _validate_proxy_anchor_binding(cohort, selected,
                {**actual, "mask_frame_number": 304})
        with self.assertRaisesRegex(ValueError, "outside_safe_interval"):
            _validate_proxy_anchor_binding(cohort, selected,
                {**actual, "mask_time_seconds": 10.01})

    def test_prompt_requests_one_short_visible_observation(self):
        prompt = _prompt({
            "question": "What is happening?",
            **{f"answer_choice_{i}": f"choice {i}" for i in range(5)},
        }, "red cup", 16.0)
        self.assertIn("exactly one short", prompt)
        self.assertIn("directly visible observation", prompt)
        self.assertIn("red cup", prompt)
        self.assertIn("16.0 seconds", prompt)

    def test_generated_sentence_does_not_require_numeric_time(self):
        _generated_target("The cup moves toward the person.", [3, 4], "red cup")

    def test_generated_target_rejects_missing_object_and_multiple_sentences(self):
        with self.assertRaisesRegex(ValueError, "generated_target_unclear"):
            _generated_target("Unclear.", [1], "red cup")
        with self.assertRaisesRegex(ValueError, "missing_object_name"):
            _generated_target("A person reaches forward.", [1], "red cup")
        with self.assertRaisesRegex(ValueError, "multiple_sentences"):
            _generated_target("The cup moves. The person turns.", [1, 2], "red cup")

    def test_generated_target_rejects_empty_and_overlong(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _generated_target("  ", [1], "cup")
        with self.assertRaisesRegex(ValueError, "short_single_line"):
            _generated_target("The cup moves.", list(range(65)), "cup")


class CohortReceiptBindingTests(unittest.TestCase):
    def test_exact_unique_passing_preflight_record(self):
        receipt = {"status": "completed_frozen_cohort_cpu_preflight",
                   "cohort_sha256": "cohort", "records": [
                       {"video_id": "v1", "row_key_sha256": "row",
                        "status": "pass"}]}
        self.assertEqual(_record_for_row(receipt, cohort_sha256="cohort",
            row_key_sha256="row", video_id="v1")["status"], "pass")

    def test_receipt_rejects_wrong_cohort_duplicate_and_failed_row(self):
        record = {"video_id": "v1", "row_key_sha256": "row", "status": "pass"}
        receipt = {"status": "completed_frozen_cohort_cpu_preflight",
                   "cohort_sha256": "cohort", "records": [record]}
        with self.assertRaisesRegex(ValueError, "binding_mismatch"):
            _record_for_row(receipt, cohort_sha256="other",
                row_key_sha256="row", video_id="v1")
        receipt["records"].append(dict(record))
        with self.assertRaisesRegex(ValueError, "not_unique"):
            _record_for_row(receipt, cohort_sha256="cohort",
                row_key_sha256="row", video_id="v1")
        receipt["records"] = [dict(record, status="fail")]
        with self.assertRaisesRegex(ValueError, "not_unique"):
            _record_for_row(receipt, cohort_sha256="cohort",
                row_key_sha256="row", video_id="v1")


class ContrastTests(unittest.TestCase):
    def test_six_branch_primary_and_secondary_differences(self):
        scores = {
            "support_none": -2.0, "sham_none": -1.5,
            "support_bbox_bias": -1.0, "sham_bbox_bias": -0.2,
            "support_spatial_sham_bias": -1.2,
            "sham_spatial_sham_bias": -0.4,
        }
        contrasts = compute_pixel_bias_interaction(scores)
        self.assertAlmostEqual(contrasts["D_no_bias_nats_per_token"], 0.5)
        self.assertAlmostEqual(contrasts["D_bbox_bias_nats_per_token"], 0.8)
        self.assertAlmostEqual(contrasts["D_spatial_sham_bias_nats_per_token"], 0.8)
        self.assertAlmostEqual(contrasts[
            "primary_bbox_minus_spatial_sham_interaction_nats_per_token"], 0.0)
        self.assertAlmostEqual(contrasts[
            "secondary_bbox_minus_no_bias_interaction_nats_per_token"], 0.3)

    def test_contrasts_reject_missing_nonfinite_or_boolean_scores(self):
        valid = {f"{fill}_{key}": -1.0 for fill in ("support", "sham")
                 for key in ("none", "bbox_bias", "spatial_sham_bias")}
        for broken in ({k: v for k, v in valid.items() if k != "sham_none"},
                       dict(valid, sham_none=float("nan")),
                       dict(valid, sham_none=True)):
            with self.subTest(broken=broken), self.assertRaisesRegex(
                    ValueError, "branch_score_set_invalid"):
                compute_pixel_bias_interaction(broken)


if __name__ == "__main__":
    unittest.main()
