"""Synthetic unit checks for the exact three-condition scoring calculation."""
from __future__ import annotations

import unittest
import json

try:
    import torch
except ImportError:  # pragma: no cover - this repo's research env includes torch
    torch = None

from nextgqa_claim_mask_scoring import (
    CONDITIONS,
    score_preflighted_triplets,
)


class Batch(dict):
    def to(self, _device):
        return self


class CpuFailBatch(Batch):
    def to(self, device):
        if device == "cpu":
            raise RuntimeError("synthetic materialization failure")
        return self


class Output:
    def __init__(self, logits):
        self.logits = logits


class TokenTensorFailTorch:
    def tensor(self, *_args, **_kwargs):
        raise RuntimeError("synthetic device failure with private detail")


class FakeModel:
    def __init__(self, fail_at=None, nonfinite_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.nonfinite_at = nonfinite_at

    def __call__(self, synthetic_logits, **_kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic failure")
        logits = synthetic_logits.clone()
        if self.calls == self.nonfinite_at:
            logits[0] = float("nan")
        return Output(logits.reshape(1, 1, 5))


def triplets():
    rows = []
    for gold in range(3):
        vectors = {}
        for condition, gold_score, foil_score in (
                ("full", 2.0, 0.0),
                ("claim_evidence_black", 0.0, 2.0),
                ("temporal_control_black", 1.0, 0.0)):
            vector = [0.0] * 5
            vector[gold] = gold_score
            vector[(gold + 1) % 5] = foil_score
            vectors[condition] = torch.tensor(vector)
        rows.append({"gold_index": gold,
                     "batches": {condition: Batch(input_ids=torch.tensor([[1]]),
                                                  synthetic_logits=vectors[condition])
                                 for condition in CONDITIONS}})
    return rows


@unittest.skipIf(torch is None, "torch unavailable")
class NextgqaClaimMaskScoringTests(unittest.TestCase):
    def test_success_runs_exactly_nine_forwards_and_only_finite_d3(self):
        model = FakeModel()
        result = score_preflighted_triplets(
            model=model, triplets=triplets(), token_ids=[0, 1, 2, 3, 4],
            torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 9)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_forwards"], 9)
        self.assertEqual(len(result["records"]), 3)
        self.assertAlmostEqual(result["D3_control_minus_claim_mask_nats"], 3.0)
        self.assertTrue(all(abs(row["d_control_minus_claim_mask_nats"] - 3.0) < 1e-6
                            for row in result["records"]))
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["confidence_interval"])

    def test_bad_ninth_condition_stops_without_survivor_average(self):
        rows = triplets()
        rows[2]["batches"].pop("temporal_control_black")
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "record_3_condition_set_invalid"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_cpu_materializes_all_nine_batches_before_any_forward(self):
        rows = triplets()
        rows[2]["batches"]["temporal_control_black"] = CpuFailBatch(input_ids=torch.tensor([[1]]))
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "record_3_temporal_control_black_cpu_materialization_failed"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_forward_failure_keeps_d3_null_and_records_counts(self):
        model = FakeModel(fail_at=8)
        result = score_preflighted_triplets(
            model=model, triplets=triplets(), token_ids=[0, 1, 2, 3, 4],
            torch_module=torch, device="cpu")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(model.calls, 8)
        self.assertEqual(result["attempted_forwards"], 8)
        self.assertEqual(result["completed_forwards"], 7)
        self.assertEqual(result["cohort_denominator"], 3)
        self.assertEqual(result["completed_triplets"], 2)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])
        self.assertEqual([row["d_control_minus_claim_mask_nats"] is not None
                          for row in result["records"]], [True, True, False])

    def test_duplicate_answer_tokens_rejected_before_any_forward(self):
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "answer_label_token_contract_invalid"):
            score_preflighted_triplets(
                model=model, triplets=triplets(), token_ids=[0, 1, 2, 3, 3],
                torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_nonfinite_logits_stop_and_return_no_d3(self):
        result = score_preflighted_triplets(
            model=FakeModel(nonfinite_at=4), triplets=triplets(),
            token_ids=[0, 1, 2, 3, 4], torch_module=torch, device="cpu")
        self.assertEqual(result["failure_code"], "nonfinite_next_token_logits")
        self.assertEqual(result["attempted_forwards"], 4)
        self.assertEqual(result["completed_forwards"], 3)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_resource_guard_exception_is_sanitized_and_stops(self):
        def guard(row, condition_position, _condition):
            if (row, condition_position) == (3, 2):
                raise RuntimeError("sensitive path must not escape")

        result = score_preflighted_triplets(
            model=FakeModel(), triplets=triplets(), token_ids=[0, 1, 2, 3, 4],
            torch_module=torch, device="cpu", before_forward=guard)
        self.assertEqual(result["failure_code"], "forward_or_score_failed_RuntimeError")
        self.assertEqual(result["attempted_forwards"], 7)
        self.assertNotIn("sensitive path", json.dumps(result))
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_answer_token_tensor_device_failure_is_sanitized_before_forward(self):
        model = FakeModel()
        result = score_preflighted_triplets(
            model=model, triplets=triplets(), token_ids=[0, 1, 2, 3, 4],
            torch_module=TokenTensorFailTorch(), device="cuda")
        self.assertEqual(result["failure_code"], "answer_token_tensor_creation_failed_RuntimeError")
        self.assertEqual(result["attempted_forwards"], 0)
        self.assertNotIn("private detail", json.dumps(result))
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])


if __name__ == "__main__":
    unittest.main()
