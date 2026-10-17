from __future__ import annotations

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import qta_synthetic_cohort_preflight as preflight_module
from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort
from qta_prescore_mapping_binding import (
    audit_qta_prescore_mapping_bundle, interval_list_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def synthetic_mapping_inputs() -> dict:
    """Build three clearly synthetic rows for a structural integration check."""
    tokens = [f"synthetic-token-{i:02d}" for i in range(3)]
    receipt_sha, review_sha = _sha("synthetic receipt"), _sha("synthetic review")
    video_hashes = {token: _sha(f"synthetic video {i}")
                    for i, token in enumerate(tokens)}
    batch_hashes = {token: _sha(f"synthetic batch {i}")
                    for i, token in enumerate(tokens)}
    sampling_hashes = {token: _sha(f"synthetic sampling {i}")
                       for i, token in enumerate(tokens)}
    reviews, clocks, processors, baselines = [], [], [], []
    calibration_hashes, source_pins = {}, {}
    for i, token in enumerate(tokens):
        video_sha = video_hashes[token]
        payload = {"source_video_sha256": video_sha, "slope": "1",
                   "intercept": "0", "anchor_pts_min_seconds": "0",
                   "anchor_pts_max_seconds": "1", "guard_pts_seconds": "1/20"}
        calibration_hashes[token] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")).hexdigest()
        intervals = [[0.1, 0.4]]
        reviews.append({
            "review_token": token, "source_video_sha256": video_sha,
            "review_payload_sha256": review_sha,
            "support_decision": "supports_claim",
            "evidence_intervals_seconds": intervals,
        })
        clocks.append({
            "review_token": token, "source_video_sha256": video_sha,
            "video_claim_review_receipt_sha256": receipt_sha,
            "calibration_sha256": calibration_hashes[token],
            "calibration_binding_payload": payload,
            "slope": "1", "intercept": "0",
            "anchor_pts_min_seconds": "0", "anchor_pts_max_seconds": "1",
            "guard_pts_seconds": "1/20",
        })
        processors.append({
            "review_token": token, "source_video_sha256": video_sha,
            "typed_batch_sha256": batch_hashes[token],
            "sampling_metadata_sha256": sampling_hashes[token],
            "video_grid_thw": [5, 1, 1],
            "reviewed_intervals_rational_seconds": [["1/10", "2/5"]],
            "interval_boundary_guard_rational_seconds": "1/20",
            "processor_uniform_grid_assignment_check": (
                "passed_actual_pts_and_scalar_fps_group_sets_equal"),
            "evidence_group_indices": [1, 3], "control_group_indices": [0, 4],
            "temporal_group_token_positions": [[0], [1], [2], [3], [4]],
        })
        baselines.append({
            "review_token": token, "source_video_sha256": video_sha,
            "video_claim_review_receipt_sha256": receipt_sha,
            "video_support_decision": "supports_claim",
            "reviewed_intervals_sha256": interval_list_sha256(intervals),
            "question_sha256": _sha(f"synthetic question {i}"),
            "options_sha256": _sha(f"synthetic options {i}"),
            "claim_sha256": _sha(f"synthetic claim {i}"),
        })
        source_pins[token] = {key: baselines[-1][key]
                              for key in ("question_sha256", "options_sha256", "claim_sha256")}

    return {
        "expected_cohort_tokens": tokens,
        "expected_video_review_receipt_sha256": receipt_sha,
        "expected_review_payload_sha256": review_sha,
        "expected_video_sha256_by_token": video_hashes,
        "expected_processor_batch_sha256_by_token": batch_hashes,
        "expected_sampling_metadata_sha256_by_token": sampling_hashes,
        "expected_processor_groups_by_token": {
            token: {"evidence_group_indices": [1, 3],
                    "control_group_indices": [0, 4], "video_grid_thw": [5, 1, 1],
                    "temporal_group_token_positions": [[0], [1], [2], [3], [4]]}
            for token in tokens
        },
        "expected_calibration_sha256_by_token": calibration_hashes,
        "expected_baseline_source_sha256_by_token": source_pins,
        "review_rows": reviews, "clock_rows": clocks,
        "processor_rows": processors, "baseline_rows": baselines,
    }


class SyntheticQtaCohortPreflightTests(unittest.TestCase):
    def _assert_non_authorizing(self, report):
        self.assertEqual(report["requested_records"], 3)
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["branch_forwards"], 0)
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["clock_joint_freeze_authenticated"])
        self.assertEqual(report["prediction_link_authentication"],
                         "self_asserted_unauthenticated")

    def test_three_synthetic_rows_integrate_but_remain_unvalidated(self):
        report = preflight_synthetic_qta_cohort(**synthetic_mapping_inputs())
        self.assertEqual(report["status"],
                         "structural_row_binding_passed_unvalidated")
        self.assertEqual(report["verified_structural_rows"], 3)
        self.assertEqual(len(report["row_binding_sha256"]), 3)
        self.assertIsNotNone(report["bundle_binding_sha256"])
        self.assertIsNone(report["failure_code"])
        self.assertEqual(report["scope"], "structural_row_binding_integration_only")
        self.assertEqual(report["data_provenance_status"],
                         "not_verified_by_this_preflight")
        self.assertNotIn("synthetic_only", report)
        self._assert_non_authorizing(report)

    def test_incomplete_collection_short_circuits_and_keeps_denominator(self):
        inputs = synthetic_mapping_inputs()
        inputs["processor_rows"] = inputs["processor_rows"][:2]
        with patch.object(preflight_module, "audit_qta_prescore_mapping_bundle") as binder:
            report = preflight_synthetic_qta_cohort(**inputs)
        binder.assert_not_called()
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["failure_code"], "cohort_row_count_mismatch")
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self._assert_non_authorizing(report)

    def test_mismatched_row_fails_closed_without_complete_case_reduction(self):
        inputs = synthetic_mapping_inputs()
        inputs["processor_rows"] = copy.deepcopy(inputs["processor_rows"])
        inputs["processor_rows"][1]["review_token"] = "wrong-synthetic-token"
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["failure_code"], "review_token_binding_mismatch")
        self.assertEqual(report["verified_structural_rows"], 1)
        self.assertEqual(report["row_binding_sha256"], [])
        self._assert_non_authorizing(report)

    def test_inconsistent_success_summary_is_not_promoted_to_success(self):
        inputs = synthetic_mapping_inputs()
        valid = audit_qta_prescore_mapping_bundle(**inputs)
        fewer_rows = copy.deepcopy(valid)
        fewer_rows["verified_structural_rows"] = 2
        bad_digests = copy.deepcopy(valid)
        bad_digests["row_binding_sha256"][1] = "not-a-hash"
        bad_bundle = copy.deepcopy(valid)
        bad_bundle["bundle_binding_sha256"] = "0" * 64
        invalid_summaries = (
            (fewer_rows, "structural_join_success_summary_invalid"),
            (bad_digests, "structural_join_success_summary_invalid"),
            (bad_bundle, "structural_join_bundle_digest_mismatch"),
        )
        for summary, expected_code in invalid_summaries:
            with self.subTest(summary=summary):
                with patch.object(preflight_module,
                                  "audit_qta_prescore_mapping_bundle",
                                  return_value=summary):
                    report = preflight_synthetic_qta_cohort(**inputs)
                self.assertEqual(report["status"],
                                 "blocked_structural_row_binding_integration")
                self.assertEqual(report["failure_code"], expected_code)
                self.assertEqual(report["row_binding_sha256"], [])
                self._assert_non_authorizing(report)


if __name__ == "__main__":
    unittest.main()
