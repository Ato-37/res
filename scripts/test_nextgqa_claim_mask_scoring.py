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
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from nextgqa_model_call_boundary import ModelCallBoundary


Batch = FrozenBatch


class ArbitraryBatch(dict):
    def to(self, _device):
        return self


class MutableBatchSubclass(FrozenBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_calls = 0

    def to(self, device):
        self.to_calls += 1
        return super().to(device)


class DeviceTensorLike:
    device = "cuda:0"

    def detach(self):
        return self

    def cpu(self):
        raise AssertionError("CPU validation must reject before copying")

    def tolist(self):
        return [1]


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


class HookableFakeModel(torch.nn.Module):
    """PyTorch module so tests exercise real global/local pre-hook ordering."""
    def __init__(self, fail_at=None, nonfinite_at=None):
        super().__init__()
        self.calls = 0
        self.fail_at = fail_at
        self.nonfinite_at = nonfinite_at

    def forward(self, synthetic_logits, input_ids, use_cache, logits_to_keep):
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


def hash_binding(rows):
    ids = [f"r{i}" for i in range(1, 4)]
    hashes = {record_id: {condition: batch_sha256(rows[position]["batches"][condition])
                          for condition in CONDITIONS}
              for position, record_id in enumerate(ids)}
    return {"record_ids": ids, "expected_batch_hashes": hashes}


@unittest.skipIf(torch is None, "torch unavailable")
class NextgqaClaimMaskScoringTests(unittest.TestCase):
    def test_single_record_smoke_never_returns_three_record_aggregate(self):
        rows=triplets()[:1]; ids=["smoke-row"]
        hashes={ids[0]:{condition:batch_sha256(rows[0]["batches"][condition])
                         for condition in CONDITIONS}}
        model=HookableFakeModel()
        result=score_preflighted_triplets(model=model,triplets=rows,token_ids=[0,1,2,3,4],
            record_ids=ids,expected_batch_hashes=hashes,torch_module=torch,device="cpu",
            requested_records=1)
        self.assertEqual(result["status"],"completed")
        self.assertEqual(result["scope"],"single_record_smoke_only")
        self.assertEqual(result["completed_forwards"],3)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_exact_final_kwargs_are_verified_at_root_entry_for_all_nine_calls(self):
        model = HookableFakeModel()
        boundary = ModelCallBoundary()
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=boundary)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(model.calls, 9)
        self.assertEqual(boundary.root_call_entries, 9)
        self.assertEqual(boundary.verified_root_calls, 9)
        self.assertEqual(boundary.completed_calls, 9)
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_root_boundary_rejects_modified_batch_before_model_forward(self):
        model = HookableFakeModel()
        boundary = ModelCallBoundary()
        rows = triplets()

        def tamper_before_root(**call):
            call["call_kwargs"]["input_ids"][0, 0] = 99
            return boundary(**call)

        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=tamper_before_root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempted_forwards"], 1)
        self.assertEqual(result["completed_forwards"], 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(boundary.root_call_entries, 1)
        self.assertEqual(boundary.verified_root_calls, 0)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_later_root_pre_hook_rewrite_is_rejected_at_forward_entry(self):
        model = HookableFakeModel()
        boundary = ModelCallBoundary()
        rows = triplets()

        def rewrite_kwargs(_module, _args, kwargs):
            rewritten = dict(kwargs)
            rewritten["input_ids"] = kwargs["input_ids"].clone()
            rewritten["input_ids"][0, 0] = 99
            return (), rewritten

        model.register_forward_pre_hook(rewrite_kwargs, with_kwargs=True)
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=boundary)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempted_forwards"], 1)
        self.assertEqual(result["completed_forwards"], 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(boundary.root_call_entries, 1)
        self.assertEqual(boundary.verified_root_calls, 0)
        self.assertNotIn("forward", model.__dict__)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_forward_exception_restores_class_defined_forward(self):
        model = HookableFakeModel(fail_at=1)
        boundary = ModelCallBoundary()
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=boundary)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempted_forwards"], 1)
        self.assertEqual(result["completed_forwards"], 0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(boundary.root_call_entries, 1)
        self.assertEqual(boundary.verified_root_calls, 1)
        self.assertEqual(boundary.completed_calls, 0)
        self.assertNotIn("forward", model.__dict__)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_preexisting_instance_forward_is_restored_after_success(self):
        model = HookableFakeModel()
        original_instance_forward = model.forward
        model.forward = original_instance_forward
        boundary = ModelCallBoundary()
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=boundary)
        self.assertEqual(result["status"], "completed")
        self.assertIs(model.__dict__["forward"], original_instance_forward)
        self.assertEqual(boundary.root_call_entries, 9)
        self.assertEqual(boundary.verified_root_calls, 9)
        self.assertEqual(boundary.completed_calls, 9)

    def test_boundary_rejects_extra_model_option_before_root_entry(self):
        model = HookableFakeModel()
        boundary = ModelCallBoundary()
        rows = triplets()

        def add_extra_option(**call):
            call["call_kwargs"]["return_dict"] = True
            return boundary(**call)

        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows), torch_module=torch, device="cpu",
            model_call=add_extra_option)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempted_forwards"], 1)
        self.assertEqual(model.calls, 0)
        self.assertEqual(boundary.root_call_entries, 0)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_success_runs_exactly_nine_forwards_and_only_finite_d3(self):
        model = FakeModel()
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows),
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
        hashes = hash_binding(rows)
        rows[2]["batches"].pop("temporal_control_black")
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "record_3_condition_set_invalid"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hashes,
                torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_hash_preflight_does_not_rematerialize_cpu_batches(self):
        rows = triplets()
        hashes = hash_binding(rows)
        hashes["expected_batch_hashes"]["r3"]["temporal_control_black"] = "0" * 64
        model = FakeModel()
        original_to = FrozenBatch.to
        cpu_calls = []

        def count_cpu_materializations(self, device):
            if device == "cpu":
                cpu_calls.append(self)
            return original_to(self, device)

        from unittest.mock import patch
        with patch.object(FrozenBatch, "to", count_cpu_materializations), self.assertRaisesRegex(
                ValueError, "record_3_temporal_control_black_frozen_hash_mismatch"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hashes,
                torch_module=torch, device="cpu")
        self.assertEqual(cpu_calls, [])
        self.assertEqual(model.calls, 0)

    def test_success_only_transfers_each_batch_once_for_its_forward(self):
        rows = triplets()
        model = FakeModel()
        original_to = FrozenBatch.to
        cpu_calls = []

        def count_cpu_transfers(self, device):
            if device == "cpu":
                cpu_calls.append(self)
            return original_to(self, device)

        from unittest.mock import patch
        with patch.object(FrozenBatch, "to", count_cpu_transfers):
            result = score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hash_binding(rows), torch_module=torch, device="cpu")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(model.calls, 9)
        self.assertEqual(len(cpu_calls), 9)

    def test_non_cpu_tensor_is_rejected_without_copy_or_forward(self):
        rows = triplets()
        hashes = hash_binding(rows)
        rows[0]["batches"]["full"]["input_ids"] = DeviceTensorLike()
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "record_1_full_batch_not_cpu"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hashes, torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_forward_failure_keeps_d3_null_and_records_counts(self):
        model = FakeModel(fail_at=8)
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows),
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
        rows = triplets()
        with self.assertRaisesRegex(ValueError, "answer_label_token_contract_invalid"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 3],
                **hash_binding(rows),
                torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_arbitrary_mapping_is_rejected_before_any_forward(self):
        rows = triplets()
        hashes = hash_binding(rows)
        rows[0]["batches"]["full"] = ArbitraryBatch(
            rows[0]["batches"]["full"])
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "frozen_batch_snapshot_required"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hashes, torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 0)

    def test_frozen_batch_subclass_side_effect_is_rejected_before_to(self):
        rows = triplets()
        subclass = MutableBatchSubclass(rows[0]["batches"]["full"])
        rows[0]["batches"]["full"] = subclass
        hashes = hash_binding(rows)
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "frozen_batch_snapshot_required"):
            score_preflighted_triplets(
                model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
                **hashes, torch_module=torch, device="cpu")
        self.assertEqual(subclass.to_calls, 0)
        self.assertEqual(model.calls, 0)

    def test_nonfinite_logits_stop_and_return_no_d3(self):
        rows = triplets()
        result = score_preflighted_triplets(
            model=FakeModel(nonfinite_at=4), triplets=rows,
            token_ids=[0, 1, 2, 3, 4], **hash_binding(rows),
            torch_module=torch, device="cpu")
        self.assertEqual(result["failure_code"], "nonfinite_next_token_logits")
        self.assertEqual(result["attempted_forwards"], 4)
        self.assertEqual(result["completed_forwards"], 3)
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_resource_guard_exception_is_sanitized_and_stops(self):
        rows = triplets()
        def guard(row, condition_position, _condition):
            if (row, condition_position) == (3, 2):
                raise RuntimeError("sensitive path must not escape")

        result = score_preflighted_triplets(
            model=FakeModel(), triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows),
            torch_module=torch, device="cpu", before_forward=guard)
        self.assertEqual(result["failure_code"], "forward_or_score_failed_RuntimeError")
        self.assertEqual(result["attempted_forwards"], 7)
        self.assertNotIn("sensitive path", json.dumps(result))
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_answer_token_tensor_device_failure_is_sanitized_before_forward(self):
        model = FakeModel()
        rows = triplets()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hash_binding(rows),
            torch_module=TokenTensorFailTorch(), device="cuda")
        self.assertEqual(result["failure_code"], "answer_token_tensor_creation_failed_RuntimeError")
        self.assertEqual(result["attempted_forwards"], 0)
        self.assertNotIn("private detail", json.dumps(result))
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_mutated_later_batch_is_detected_after_transfer_before_its_forward(self):
        rows = triplets()
        hashes = hash_binding(rows)

        def mutate_later_batch(row, condition_position, _condition):
            if (row, condition_position) == (2, 1):
                rows[2]["batches"]["full"]["input_ids"][0, 0] = 99

        model = FakeModel()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hashes, torch_module=torch, device="cpu",
            before_forward=mutate_later_batch)
        self.assertEqual(model.calls, 6)
        self.assertEqual(result["attempted_forwards"], 6)
        self.assertEqual(result["failure_code"], "frozen_batch_hash_changed_after_device_transfer")
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])

    def test_current_batch_mutated_by_guard_is_detected_before_forward(self):
        rows = triplets()
        hashes = hash_binding(rows)

        def mutate_current_batch(row, condition_position, _condition):
            if (row, condition_position) == (1, 1):
                rows[0]["batches"]["full"]["input_ids"][0, 0] = 99

        model = FakeModel()
        result = score_preflighted_triplets(
            model=model, triplets=rows, token_ids=[0, 1, 2, 3, 4],
            **hashes, torch_module=torch, device="cpu",
            before_forward=mutate_current_batch)
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["attempted_forwards"], 0)
        self.assertEqual(result["failure_code"], "frozen_batch_hash_changed_after_device_transfer")
        self.assertIsNone(result["D3_control_minus_claim_mask_nats"])


if __name__ == "__main__":
    unittest.main()
