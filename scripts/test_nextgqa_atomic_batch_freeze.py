from __future__ import annotations

import unittest

from nextgqa_atomic_batch_freeze import (
    CONDITIONS, PreflightError, batch_sha256, freeze_all_batches,
)
from nextgqa_claim_mask_scoring import validate_scoring_inputs


class FakeTensor:
    def __init__(self, values, *, dtype="float32", shape=None, device="cpu"):
        self._values = values
        self.dtype = dtype
        self.shape = shape or [len(values)]
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor(self._values, dtype=self.dtype,
                          shape=self.shape, device="cpu")

    def tolist(self):
        return self._values

    def clone(self):
        return FakeTensor(list(self._values), dtype=self.dtype,
                          shape=self.shape, device=self.device)


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: (FakeTensor(value.tolist(), dtype=value.dtype,
                                           shape=value.shape, device=device)
                                if isinstance(value, FakeTensor) else value)
                          for key, value in self.items()})


def fixture():
    rows = []
    hashes = {}
    for row_pos in range(3):
        record_id = f"r{row_pos + 1}"
        batches = {
            condition: FakeBatch({
                "input_ids": FakeTensor([1, 2, row_pos], dtype="int64"),
                "pixel_values": FakeTensor([0.1, 0.2, 0.3 + row_pos],
                                            shape=[1, 3]),
                "video_metadata": [row_pos, 32],
            })
            for condition in CONDITIONS
        }
        rows.append({"record_id": record_id, "gold_index": row_pos,
                     "batches": batches})
        hashes[record_id] = {condition: batch_sha256(batch)
                             for condition, batch in batches.items()}
    return rows, hashes


class AtomicBatchFreezeTests(unittest.TestCase):
    def test_success_materializes_all_nine_before_gate_and_returns_no_auth(self):
        rows, hashes = fixture()
        events = []

        def gate():
            events.append("gate")
            return True

        result = freeze_all_batches(
            triplets=rows, token_ids=[11, 12, 13, 14, 15],
            expected_record_ids=["r1", "r2", "r3"],
            expected_batch_hashes=hashes, authenticated_gate_check=gate)
        self.assertEqual(result["status"], "all_nine_batches_frozen")
        self.assertEqual(result["requested_forwards"], 9)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(len(result["triplets"]), 3)
        validate_scoring_inputs(result["triplets"], result["token_ids"])
        self.assertEqual(events, ["gate"])

    def test_missing_condition_rejected_before_gate(self):
        rows, hashes = fixture()
        del rows[2]["batches"][CONDITIONS[-1]]
        called = []
        with self.assertRaisesRegex(PreflightError, "condition_set_invalid"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               authenticated_gate_check=lambda: called.append(1))
        self.assertEqual(called, [])

    def test_hash_mismatch_rejected_before_gate(self):
        rows, hashes = fixture()
        hashes["r2"]["claim_evidence_black"] = "0" * 64
        called = []
        with self.assertRaisesRegex(PreflightError, "exact_condition_batch_hash_mismatch"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               authenticated_gate_check=lambda: called.append(1))
        self.assertEqual(called, [])

    def test_upstream_gate_failure_never_returns_frozen_inputs(self):
        rows, hashes = fixture()
        with self.assertRaisesRegex(PreflightError, "authenticated_upstream_gate_failed"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               authenticated_gate_check=lambda: (_ for _ in ()).throw(
                                   RuntimeError("private details")))

    def test_nonfinite_input_fails_closed(self):
        rows, hashes = fixture()
        rows[0]["batches"]["full"]["pixel_values"] = FakeTensor([float("nan")])
        with self.assertRaisesRegex(PreflightError, "nonfinite"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               authenticated_gate_check=lambda: True)

    def test_duplicate_token_ids_rejected(self):
        rows, hashes = fixture()
        with self.assertRaisesRegex(PreflightError, "token_contract_invalid"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 14],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               authenticated_gate_check=lambda: True)

    def test_torch_cpu_batches_are_snapshotted_before_return(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - repository environment has torch
            self.skipTest("torch unavailable")
        rows, hashes = fixture()
        source = torch.tensor([1.0, 2.0])
        rows[0]["batches"]["full"]["torch_values"] = source
        hashes["r1"]["full"] = batch_sha256(rows[0]["batches"]["full"])
        result = freeze_all_batches(
            triplets=rows, token_ids=[11, 12, 13, 14, 15],
            expected_record_ids=["r1", "r2", "r3"],
            expected_batch_hashes=hashes,
            authenticated_gate_check=lambda: True)
        frozen = result["triplets"][0]["batches"]["full"]["torch_values"]
        source[0] = 99.0
        self.assertEqual(float(frozen[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
