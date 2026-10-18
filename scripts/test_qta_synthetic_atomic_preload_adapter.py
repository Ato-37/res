from __future__ import annotations

import hashlib
import json
import unittest
from collections.abc import Mapping

from qta_synthetic_atomic_preload_adapter import (
    ROW_COUNT, bind_synthetic_atomic_inputs,
)
from test_qta_actual_prebaseline_diagnostic import combined_inputs


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class QtaSyntheticAtomicPreloadAdapterTests(unittest.TestCase):
    def denied(self, report):
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["score_ready"], False)
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["forward_count"], 0)
        self.assertEqual(report["score_calls"], 0)

    def test_exact_synthetic_bytes_join_as_one_ordered_three_row_unit(self):
        source, review, private_values = combined_inputs()
        report = bind_synthetic_atomic_inputs(actual_source_inputs=source,
                                              review_bundle_inputs=review)
        self.assertEqual(report["status"],
                         "three_row_structural_binding_passed_unvalidated")
        self.assertEqual(report["requested_rows"], ROW_COUNT)
        self.assertEqual(report["validated_rows"], ROW_COUNT)
        self.assertEqual(len(report["ordered_row_binding_sha256"]), ROW_COUNT)
        self.assertEqual(len(set(report["ordered_row_binding_sha256"])), ROW_COUNT)
        self.assertEqual(len(report["cohort_binding_sha256"]), 64)
        self.assertEqual(report["input_classification"],
                         "caller_supplied_fixture_not_independently_authenticated")
        public = json.dumps(report, ensure_ascii=False)
        for value in private_values + ["claim-", "video-", "question 0?", "answer 0"]:
            self.assertNotIn(value, public)
        self.denied(report)

    def test_missing_source_or_review_fails_the_entire_cohort(self):
        source, review, _ = combined_inputs()
        for source_inputs, review_inputs in ((None, review), (source, None)):
            with self.subTest(source=source_inputs is not None,
                              review=review_inputs is not None):
                report = bind_synthetic_atomic_inputs(
                    actual_source_inputs=source_inputs,
                    review_bundle_inputs=review_inputs)
                self.assertEqual(report["status"], "blocked_atomic_three_row_binding")
                self.assertEqual(report["requested_rows"], ROW_COUNT)
                self.assertEqual(report["validated_rows"], 0)
                self.assertEqual(report["ordered_row_binding_sha256"], [])
                self.assertIsNone(report["cohort_binding_sha256"])
                self.denied(report)

    def test_review_row_swap_fails_without_partial_bindings(self):
        source, review, _ = combined_inputs()
        review = dict(review)
        payload = json.loads(review["review_raw"])
        payload["records"][0], payload["records"][1] = (
            payload["records"][1], payload["records"][0])
        review_raw = encode(payload)
        index = json.loads(review["review_index_raw"])
        index["review_payload_sha256"] = sha(review_raw)
        review["review_raw"] = review_raw
        review["review_index_raw"] = encode(index)
        report = bind_synthetic_atomic_inputs(actual_source_inputs=source,
                                              review_bundle_inputs=review)
        self.assertEqual(report["status"], "blocked_atomic_three_row_binding")
        self.assertEqual(report["validated_rows"], 0)
        self.assertEqual(report["ordered_row_binding_sha256"], [])
        self.denied(report)

    def test_tampered_source_pin_fails_before_any_partial_acceptance(self):
        source, review, _ = combined_inputs()
        source = dict(source)
        source["expected_qta_config_sha256"] = "0" * 64
        report = bind_synthetic_atomic_inputs(actual_source_inputs=source,
                                              review_bundle_inputs=review)
        self.assertEqual(report["status"], "blocked_atomic_three_row_binding")
        self.assertEqual(report["validated_rows"], 0)
        self.assertEqual(report["ordered_row_binding_sha256"], [])
        self.assertIsNone(report["source_input_sha256"])
        self.denied(report)

    def test_stateful_mappings_are_snapshotted_once_before_hash_and_validation(self):
        source, review, _ = combined_inputs()
        stable = bind_synthetic_atomic_inputs(actual_source_inputs=source,
                                              review_bundle_inputs=review)

        class ChangeOnSecondRead(Mapping):
            def __init__(self, original):
                self.original = original
                self.read_counts = {key: 0 for key in original}

            def __iter__(self):
                return iter(self.original)

            def __len__(self):
                return len(self.original)

            def __getitem__(self, key):
                self.read_counts[key] += 1
                if self.read_counts[key] == 1:
                    return self.original[key]
                value = self.original[key]
                if isinstance(value, bytes):
                    return b"changed-after-snapshot"
                return "0" * 64

        stateful_source = ChangeOnSecondRead(source)
        stateful_review = ChangeOnSecondRead(review)
        report = bind_synthetic_atomic_inputs(
            actual_source_inputs=stateful_source,
            review_bundle_inputs=stateful_review)
        self.assertEqual(report["status"],
                         "three_row_structural_binding_passed_unvalidated")
        self.assertEqual(report["source_input_sha256"], stable["source_input_sha256"])
        self.assertEqual(report["review_input_sha256"], stable["review_input_sha256"])
        self.assertEqual(report["ordered_row_binding_sha256"],
                         stable["ordered_row_binding_sha256"])
        self.assertTrue(all(count == 1 for count in stateful_source.read_counts.values()))
        self.assertTrue(all(count == 1 for count in stateful_review.read_counts.values()))
        self.denied(report)

    def test_arbitrary_mapping_exception_text_is_not_returned(self):
        class RaisingMapping(Mapping):
            def __iter__(self):
                raise ValueError("private video path must not leak")

            def __len__(self):
                return 0

            def __getitem__(self, key):
                raise KeyError(key)

        report = bind_synthetic_atomic_inputs(actual_source_inputs=RaisingMapping(),
                                              review_bundle_inputs=None)
        self.assertEqual(report["status"], "blocked_atomic_three_row_binding")
        self.assertEqual(report["failure_code"], "atomic_binding_failed")
        self.assertNotIn("private video path", json.dumps(report))
        self.assertNotIn("must not leak", json.dumps(report))
        self.denied(report)


if __name__ == "__main__":
    unittest.main()
