from __future__ import annotations

import hashlib
import json
import unittest

from qta_prescore_mapping_binding import (
    audit_qta_prescore_mapping_bundle,
    interval_list_sha256,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class QtaPrescoreMappingBindingTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [f"opaque-token-{i}" for i in range(3)]
        self.receipt_sha = sha("video-review-receipt")
        self.review_payload_sha = sha("review-payload")
        self.video_hashes = {token: sha(f"video-{i}")
                             for i, token in enumerate(self.tokens)}
        self.batch_hashes = {token: sha(f"batch-{i}")
                             for i, token in enumerate(self.tokens)}
        self.sampling_hashes = {token: sha(f"sampling-{i}")
                                for i, token in enumerate(self.tokens)}
        self.calibration_payloads = {
            token: {"source_video_sha256": self.video_hashes[token], "slope": "1",
                    "intercept": "0", "anchor_pts_min_seconds": "0",
                    "anchor_pts_max_seconds": "1", "guard_pts_seconds": "1/20"}
            for token in self.tokens
        }
        self.calibration_hashes = {
            token: hashlib.sha256(json.dumps(payload, sort_keys=True,
                separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
            for token, payload in self.calibration_payloads.items()
        }
        self.review_rows = []
        self.clock_rows = []
        self.processor_rows = []
        self.baseline_rows = []
        for i, token in enumerate(self.tokens):
            intervals = [[0.1, 0.4]]
            interval_sha = interval_list_sha256(intervals)
            self.review_rows.append({
                "review_token": token,
                "source_video_sha256": self.video_hashes[token],
                "review_payload_sha256": self.review_payload_sha,
                "support_decision": "supports_claim",
                "evidence_intervals_seconds": intervals,
            })
            self.clock_rows.append({
                "review_token": token,
                "source_video_sha256": self.video_hashes[token],
                "video_claim_review_receipt_sha256": self.receipt_sha,
                "calibration_sha256": self.calibration_hashes[token],
                "calibration_binding_payload": self.calibration_payloads[token],
                "slope": "1", "intercept": "0",
                "anchor_pts_min_seconds": "0",
                "anchor_pts_max_seconds": "1",
                "guard_pts_seconds": "1/20",
            })
            self.processor_rows.append({
                "review_token": token,
                "source_video_sha256": self.video_hashes[token],
                "typed_batch_sha256": self.batch_hashes[token],
                "sampling_metadata_sha256": self.sampling_hashes[token],
                "video_grid_thw": [5, 1, 1],
                "reviewed_intervals_rational_seconds": [["1/10", "2/5"]],
                "interval_boundary_guard_rational_seconds": "1/20",
                "evidence_group_indices": [1, 3],
                "control_group_indices": [0, 4],
                "temporal_group_token_positions": [[0], [1], [2], [3], [4]],
            })
            self.baseline_rows.append({
                "review_token": token,
                "source_video_sha256": self.video_hashes[token],
                "video_claim_review_receipt_sha256": self.receipt_sha,
                "video_support_decision": "supports_claim",
                "reviewed_intervals_sha256": interval_sha,
                "question_sha256": sha(f"question-{i}"),
                "options_sha256": sha(f"options-{i}"),
                "claim_sha256": sha(f"claim-{i}"),
            })
        self.baseline_pins = {
            token: {field: self.baseline_rows[i][field]
                    for field in ("question_sha256", "options_sha256", "claim_sha256")}
            for i, token in enumerate(self.tokens)
        }

    def _audit(self, **overrides):
        inputs = {
            "expected_cohort_tokens": self.tokens,
            "expected_video_review_receipt_sha256": self.receipt_sha,
            "expected_review_payload_sha256": self.review_payload_sha,
            "expected_video_sha256_by_token": self.video_hashes,
            "expected_processor_batch_sha256_by_token": self.batch_hashes,
            "expected_sampling_metadata_sha256_by_token": self.sampling_hashes,
            "expected_processor_groups_by_token": {
                token: {"evidence_group_indices": [1, 3],
                        "control_group_indices": [0, 4],
                        "video_grid_thw": [5, 1, 1],
                        "temporal_group_token_positions": [[0], [1], [2], [3], [4]]}
                for token in self.tokens
            },
            "expected_calibration_sha256_by_token": self.calibration_hashes,
            "expected_baseline_source_sha256_by_token": self.baseline_pins,
            "review_rows": self.review_rows,
            "clock_rows": self.clock_rows,
            "processor_rows": self.processor_rows,
            "baseline_rows": self.baseline_rows,
        }
        inputs.update(overrides)
        return audit_qta_prescore_mapping_bundle(**inputs)

    def test_valid_three_row_join_is_still_non_authorizing(self):
        self.assertEqual(interval_list_sha256([[0.1, 0.4]]),
                         hashlib.sha256(b'[["1/10","2/5"]]').hexdigest())
        self.assertNotEqual(interval_list_sha256([[0.1, 0.4], [0.1, 0.4]]),
                            interval_list_sha256([[0.1, 0.4]]))
        report = self._audit()
        self.assertEqual(report["status"],
                         "structurally_bound_pending_independent_score_freeze")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 3)
        self.assertEqual(len(report["row_binding_sha256"]), 3)
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["branch_forwards"], 0)
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["reviewer_identity_verified"])
        self.assertFalse(report["semantic_review_authenticated"])

    def test_interval_digest_and_exact_pts_map_are_recomputed(self):
        broken = [dict(row) for row in self.baseline_rows]
        broken[1] = {**broken[1], "reviewed_intervals_sha256": sha("forged")}
        report = self._audit(baseline_rows=broken)
        self.assertEqual(report["failure_code"], "reviewed_interval_digest_mismatch")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 1)
        self.assertEqual(report["branch_forwards"], 0)

        mapped = [dict(row) for row in self.processor_rows]
        mapped[0] = {**mapped[0],
                     "reviewed_intervals_rational_seconds": [["7", "8"]]}
        report = self._audit(processor_rows=mapped)
        self.assertEqual(report["failure_code"], "processor_interval_mapping_mismatch")

    def test_same_interval_wrong_source_or_batch_is_rejected(self):
        wrong_source = [dict(row) for row in self.processor_rows]
        wrong_source[1] = {**wrong_source[1], "source_video_sha256": sha("other-video")}
        self.assertEqual(self._audit(processor_rows=wrong_source)["failure_code"],
                         "source_video_binding_mismatch")

        wrong_batch = [dict(row) for row in self.processor_rows]
        wrong_batch[1] = {**wrong_batch[1], "typed_batch_sha256": sha("other-batch")}
        self.assertEqual(self._audit(processor_rows=wrong_batch)["failure_code"],
                         "processor_batch_binding_mismatch")

        wrong_calibration = [dict(row) for row in self.clock_rows]
        wrong_payload = {**wrong_calibration[0]["calibration_binding_payload"],
                         "guard_pts_seconds": "1/10"}
        wrong_hash = hashlib.sha256(json.dumps(wrong_payload, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        wrong_calibration[0] = {**wrong_calibration[0],
                                "calibration_sha256": wrong_hash,
                                "calibration_binding_payload": wrong_payload,
                                "guard_pts_seconds": "1/10"}
        self.assertEqual(self._audit(clock_rows=wrong_calibration)["failure_code"],
                         "calibration_binding_mismatch")

        changed_clock_value = [dict(row) for row in self.clock_rows]
        changed_clock_value[0] = {**changed_clock_value[0], "slope": "2"}
        self.assertEqual(self._audit(clock_rows=changed_clock_value)["failure_code"],
                         "calibration_clock_value_mismatch")

        changed_payload = [dict(row) for row in self.clock_rows]
        changed_payload[0] = {
            **changed_payload[0],
            "calibration_binding_payload": {
                **changed_payload[0]["calibration_binding_payload"], "slope": "2"},
        }
        self.assertEqual(self._audit(clock_rows=changed_payload)["failure_code"],
                         "calibration_payload_digest_mismatch")

        wrong_sampling = [dict(row) for row in self.processor_rows]
        wrong_sampling[0] = {**wrong_sampling[0], "sampling_metadata_sha256": sha("other-sampling")}
        self.assertEqual(self._audit(processor_rows=wrong_sampling)["failure_code"],
                         "sampling_metadata_binding_mismatch")

        out_of_range = [dict(row) for row in self.processor_rows]
        out_of_range[0] = {**out_of_range[0], "evidence_group_indices": [1, 5]}
        self.assertEqual(self._audit(processor_rows=out_of_range)["failure_code"],
                         "processor_group_index_out_of_range")

        wrong_mapping = [dict(row) for row in self.processor_rows]
        wrong_mapping[0] = {**wrong_mapping[0], "evidence_group_indices": [1, 2]}
        self.assertEqual(self._audit(processor_rows=wrong_mapping)["failure_code"],
                         "processor_group_mapping_mismatch")

        wrong_temporal = [dict(row) for row in self.processor_rows]
        wrong_temporal[0] = {**wrong_temporal[0],
                             "temporal_group_token_positions": [[0], [1], [2], [3], [8]]}
        self.assertEqual(self._audit(processor_rows=wrong_temporal)["failure_code"],
                         "processor_temporal_groups_geometry_invalid")

        gap_temporal = [dict(row) for row in self.processor_rows]
        gap_temporal[0] = {**gap_temporal[0],
                           "temporal_group_token_positions": [[0], [2], [3], [4], [5]]}
        self.assertEqual(self._audit(processor_rows=gap_temporal)["failure_code"],
                         "processor_temporal_groups_geometry_invalid")

        wrong_baseline = [dict(row) for row in self.baseline_rows]
        wrong_baseline[0] = {**wrong_baseline[0], "claim_sha256": sha("other-claim")}
        self.assertEqual(self._audit(baseline_rows=wrong_baseline)["failure_code"],
                         "baseline_source_binding_mismatch")

    def test_clock_or_group_contract_failure_blocks_whole_requested_cohort(self):
        clock = [dict(row) for row in self.clock_rows]
        clock[2] = {**clock[2], "anchor_pts_max_seconds": "1/5"}
        report = self._audit(clock_rows=clock)
        self.assertEqual(report["failure_code"], "calibration_clock_value_mismatch")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 2)

        processor = [dict(row) for row in self.processor_rows]
        processor[0] = {**processor[0], "control_group_indices": [1, 4]}
        self.assertEqual(self._audit(processor_rows=processor)["failure_code"],
                         "evidence_control_group_contract_invalid")

    def test_missing_or_reordered_row_never_becomes_complete_case_cohort(self):
        report = self._audit(processor_rows=self.processor_rows[:2])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["failure_code"], "cohort_row_count_mismatch")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 0)

        swapped = [self.processor_rows[1], self.processor_rows[0], self.processor_rows[2]]
        self.assertEqual(self._audit(processor_rows=swapped)["failure_code"],
                         "review_token_binding_mismatch")

        with self.assertRaisesRegex(ValueError,
                                    "expected_three_unique_cohort_tokens_required"):
            self._audit(expected_cohort_tokens="abc")

    def test_non_supporting_first_stage_decision_blocks(self):
        rows = [dict(row) for row in self.review_rows]
        rows[0] = {**rows[0], "support_decision": "ambiguous"}
        self.assertEqual(self._audit(review_rows=rows)["failure_code"],
                         "video_claim_not_supported")

    def test_review_payload_digest_must_match_external_pin(self):
        rows = [dict(row) for row in self.review_rows]
        rows[1] = {**rows[1], "review_payload_sha256": sha("substituted-review")}
        self.assertEqual(self._audit(review_rows=rows)["failure_code"],
                         "review_payload_binding_mismatch")

    def test_duplicate_intervals_are_hashed_as_recorded_and_grouped_once(self):
        reviews = [dict(row) for row in self.review_rows]
        reviews[0] = {**reviews[0],
                      "evidence_intervals_seconds": [[0.1, 0.4], [0.1, 0.4]]}
        baseline = [dict(row) for row in self.baseline_rows]
        baseline[0] = {**baseline[0], "reviewed_intervals_sha256":
                        interval_list_sha256(reviews[0]["evidence_intervals_seconds"])}
        processor = [dict(row) for row in self.processor_rows]
        processor[0] = {**processor[0],
                        "reviewed_intervals_rational_seconds":
                        [["1/10", "2/5"], ["1/10", "2/5"]]}
        report = self._audit(review_rows=reviews, baseline_rows=baseline,
                             processor_rows=processor)
        self.assertEqual(report["status"],
                         "structurally_bound_pending_independent_score_freeze")


if __name__ == "__main__":
    unittest.main()
