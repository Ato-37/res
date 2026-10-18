from __future__ import annotations

import hashlib
import unittest

from nextgqa_atomic_batch_freeze import FrozenBatch
from qta_baseline_output_provenance import (
    derive_synthetic_baseline_fixture,
    verify_synthetic_baseline_binding,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class BaselineOutputFixtureTests(unittest.TestCase):
    def setUp(self):
        self.run_digests = {key: sha(key) for key in (
            "protocol_sha256", "model_sha256", "tokenizer_sha256",
            "scorer_sha256", "processor_sha256")}
        self.rows = [{
            "record_id": f"row-{i}",
            "prompt": f"synthetic prompt {i}",
            "input_ids": [1, 20 + i],
            "score_position": 1,
            "answer_token_ids": [10, 11, 12, 13, 14],
            "gold_index": 0,
            "batch": FrozenBatch({"input_ids": [[1, 20 + i]],
                                  "pixel_values": [[float(i + 1)]]}),
            "fixture_schema": "synthetic_logits_1x1_vocab_v1",
            "fixture_logits": [[[0.0] * 10 + [3.0, 2.0, 1.0, 0.5, 0.0]]],
        } for i in range(3)]

    def _derive(self, **kwargs):
        return derive_synthetic_baseline_fixture(
            rows=self.rows, run_digests=self.run_digests, **kwargs)

    def test_fixture_math_and_order_are_explicitly_non_authorizing(self):
        result = self._derive()
        self.assertEqual(result["fixture_row_count"], 3)
        self.assertTrue(verify_synthetic_baseline_binding(
            result, expected_record_ids=["row-0", "row-1", "row-2"],
            expected_run_digests=self.run_digests))
        record = result["payload"]["records"][0]
        self.assertEqual(record["p_index"], 0)
        self.assertEqual(record["q_index"], 1)
        self.assertLessEqual(max(record["option_log_probabilities"]), 0)
        self.assertFalse(result["scoring_authorized"])
        self.assertFalse(result["cryptographic_attestation"])

    def test_swapped_rows_changed_pins_and_payload_tampering_fail(self):
        result = self._derive()
        with self.assertRaisesRegex(ValueError, "binding_row_order_mismatch"):
            verify_synthetic_baseline_binding(
                result, expected_record_ids=["row-1", "row-0", "row-2"],
                expected_run_digests=self.run_digests)
        wrong = {**self.run_digests, "model_sha256": sha("other model")}
        with self.assertRaisesRegex(ValueError, "binding_run_digests_mismatch"):
            verify_synthetic_baseline_binding(
                result, expected_record_ids=["row-0", "row-1", "row-2"],
                expected_run_digests=wrong)
        result["payload"]["records"][0]["option_log_probabilities"][0] += 0.25
        with self.assertRaisesRegex(ValueError, "binding_payload_digest_mismatch"):
            verify_synthetic_baseline_binding(
                result, expected_record_ids=["row-0", "row-1", "row-2"],
                expected_run_digests=self.run_digests)

    def test_contradictory_outer_scope_fields_fail(self):
        for field, value in (("execution_authorization", "granted"),
                             ("fixture_row_count", 2),
                             ("fixture_logits_are_synthetic", False),
                             ("fixture_inputs_unverified", False)):
            result = self._derive()
            result[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "binding_scope_invalid"):
                verify_synthetic_baseline_binding(
                    result, expected_record_ids=["row-0", "row-1", "row-2"],
                    expected_run_digests=self.run_digests)

    def test_row_three_mismatch_fails_atomic_preflight(self):
        self.rows[2]["input_ids"] = [1, 999]
        with self.assertRaisesRegex(ValueError, "row_3_batch_input_ids_mismatch"):
            self._derive()

    def test_cpu_tensor_input_ids_are_compared(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        self.rows[1]["batch"] = FrozenBatch({
            "input_ids": torch.tensor([[1, 21]], device="cpu"),
            "pixel_values": [[2.0]],
        })
        self.assertEqual(self._derive()["fixture_row_count"], 3)

    def test_fixture_shape_is_exact(self):
        for malformed in (
            [0.0] * 15,
            [[0.0] * 15],
            [[[0.0] * 15, [0.0] * 15]],
            [[[[0.0] * 15]]],
        ):
            self.rows[0]["fixture_logits"] = malformed
            with self.subTest(shape=malformed), self.assertRaisesRegex(
                    ValueError, "synthetic_fixture_logits_shape_invalid"):
                self._derive()

    def test_public_api_has_no_model_or_callable_argument(self):
        calls = []

        def model(**_kwargs):
            calls.append("called")

        with self.assertRaises(TypeError):
            derive_synthetic_baseline_fixture(
                rows=self.rows, run_digests=self.run_digests, model=model)
        self.assertEqual(calls, [])

    def test_invalid_late_row_blocks_whole_fixture_cohort(self):
        self.rows[2]["fixture_schema"] = "model_logits"
        with self.assertRaisesRegex(ValueError, "row_3_synthetic_fixture_schema_invalid"):
            self._derive()


if __name__ == "__main__":
    unittest.main()
