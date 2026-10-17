from __future__ import annotations

import unittest
from unittest.mock import patch

from nextgqa_atomic_batch_freeze import (
    CONDITIONS, PreflightError, batch_sha256, freeze_all_batches,
)
from nextgqa_claim_mask_scoring import (
    score_preflighted_triplets, validate_scoring_inputs,
)


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
    to_calls = 0

    def to(self, device):
        type(self).to_calls += 1
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
                "pixel_values": FakeTensor([
                    0.1, 0.2, 0.3 + row_pos + CONDITIONS.index(condition)],
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
    def test_success_prechecks_then_materializes_and_finalizes_without_auth(self):
        rows, hashes = fixture()
        events = []
        FakeBatch.to_calls = 0
        import nextgqa_atomic_batch_freeze as freezer
        original_hash = freezer.batch_sha256

        def observed_hash(batch):
            events.append("batch_hash")
            return original_hash(batch)

        def gate():
            events.append("final_gate")
            self.assertEqual(FakeBatch.to_calls, 9)
            self.assertEqual(events.count("batch_hash"), 18)
            return True

        def precheck():
            events.append("precheck")
            return True

        with patch("nextgqa_atomic_batch_freeze.batch_sha256", side_effect=observed_hash):
            result = freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes, upstream_precheck=precheck,
                authenticated_gate_check=gate)
        self.assertEqual(result["status"], "all_nine_batches_frozen")
        self.assertEqual(result["requested_forwards"], 9)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(len(result["triplets"]), 3)
        validate_scoring_inputs(result["triplets"], result["token_ids"])
        self.assertEqual(events[0], "precheck")
        self.assertEqual(events[-1], "final_gate")
        self.assertEqual(events.count("batch_hash"), 18)

    def test_identical_condition_pixels_block_before_authenticated_gate(self):
        rows, hashes = fixture()
        shared = FakeTensor([0.1, 0.2, 0.3], shape=[1, 3])
        for condition in CONDITIONS:
            rows[0]["batches"][condition]["pixel_values"] = shared
            hashes["r1"][condition] = batch_sha256(rows[0]["batches"][condition])
        FakeBatch.to_calls = 0
        with self.assertRaisesRegex(PreflightError,
                                   "record_1_condition_pixel_tensors_not_distinct"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes, upstream_precheck=lambda: True,
                authenticated_gate_check=lambda: self.fail("final gate ran"))
        self.assertEqual(FakeBatch.to_calls, 3)

    def test_failed_upstream_precheck_skips_all_batch_materialization(self):
        rows, hashes = fixture()
        FakeBatch.to_calls = 0
        with self.assertRaisesRegex(PreflightError, "upstream_precheck_failed"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes,
                upstream_precheck=lambda: (_ for _ in ()).throw(RuntimeError()),
                authenticated_gate_check=lambda: self.fail("final gate ran"))
        self.assertEqual(FakeBatch.to_calls, 0)

    def test_false_upstream_precheck_skips_all_batch_materialization(self):
        rows, hashes = fixture()
        FakeBatch.to_calls = 0
        with self.assertRaisesRegex(PreflightError, "upstream_precheck_not_passed"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes, upstream_precheck=lambda: False,
                authenticated_gate_check=lambda: self.fail("final gate ran"))
        self.assertEqual(FakeBatch.to_calls, 0)

    def test_malformed_nested_batch_skips_upstream_precheck(self):
        rows, hashes = fixture()
        del rows[1]["batches"][CONDITIONS[0]]
        events = []
        with self.assertRaisesRegex(PreflightError, "condition_set_invalid"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes,
                upstream_precheck=lambda: events.append("precheck"),
                authenticated_gate_check=lambda: self.fail("final gate ran"))
        self.assertEqual(events, [])

    def test_precheck_cannot_rewrite_expected_hash_to_match_batch_mutation(self):
        rows, hashes = fixture()

        def mutating_precheck():
            batch = rows[0]["batches"]["full"]
            batch["input_ids"] = FakeTensor([99], dtype="int64")
            hashes["r1"]["full"] = batch_sha256(batch)
            return True

        with self.assertRaisesRegex(PreflightError, "exact_condition_batch_hash_mismatch"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes, upstream_precheck=mutating_precheck,
                authenticated_gate_check=lambda: self.fail("final gate ran"))

    def test_precheck_cannot_mutate_returned_token_contract(self):
        rows, hashes = fixture()
        tokens = [11, 12, 13, 14, 15]

        def mutating_precheck():
            tokens[0] = 99
            return True

        result = freeze_all_batches(
            triplets=rows, token_ids=tokens,
            expected_record_ids=["r1", "r2", "r3"],
            expected_batch_hashes=hashes, upstream_precheck=mutating_precheck,
            authenticated_gate_check=lambda: True)
        self.assertEqual(result["token_ids"], [11, 12, 13, 14, 15])

    def test_freeze_to_nine_forward_scorer_integration(self):
        """Synthetic software-path check only; D3 is not a research result."""
        try:
            import torch
        except ImportError:  # pragma: no cover - repository environment has torch
            self.skipTest("torch unavailable")
        rows, hashes = fixture()
        for row_pos, row in enumerate(rows):
            gold = row["gold_index"]
            for condition, gold_score, foil_score in (
                    ("full", 2.0, 0.0),
                    ("claim_evidence_black", 0.0, 2.0),
                    ("temporal_control_black", 1.0, 0.0)):
                scores = [0.0] * 20
                scores[11 + gold] = gold_score
                scores[11 + (gold + 1) % 5] = foil_score
                row["batches"][condition]["synthetic_logits"] = torch.tensor(scores)
                hashes[f"r{row_pos + 1}"][condition] = batch_sha256(
                    row["batches"][condition])
        frozen = freeze_all_batches(
            triplets=rows, token_ids=[11, 12, 13, 14, 15],
            expected_record_ids=["r1", "r2", "r3"],
            expected_batch_hashes=hashes,
            upstream_precheck=lambda: True,
            authenticated_gate_check=lambda: True)

        class Model:
            def __init__(self):
                self.calls = 0

            def __call__(self, synthetic_logits, **_kwargs):
                self.calls += 1
                return type("Output", (), {
                    "logits": synthetic_logits.reshape(1, 1, -1),
                })()

        model = Model()
        scored = score_preflighted_triplets(
            model=model, triplets=frozen["triplets"],
            token_ids=frozen["token_ids"], record_ids=frozen["record_ids"],
            expected_batch_hashes=frozen["batch_sha256"],
            torch_module=torch, device="cpu")
        self.assertEqual(model.calls, 9)
        self.assertEqual(scored["completed_forwards"], 9)
        self.assertAlmostEqual(scored["D3_control_minus_claim_mask_nats"], 3.0,
                               delta=1e-6)
        self.assertEqual(frozen["execution_authorization"], "not_granted")

    def test_missing_condition_rejected_before_final_gate(self):
        rows, hashes = fixture()
        del rows[2]["batches"][CONDITIONS[-1]]
        called = []
        with self.assertRaisesRegex(PreflightError, "condition_set_invalid"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               upstream_precheck=lambda: True,
                               authenticated_gate_check=lambda: called.append(1))
        self.assertEqual(called, [])

    def test_hash_mismatch_rejected_before_final_gate(self):
        rows, hashes = fixture()
        hashes["r2"]["claim_evidence_black"] = "0" * 64
        called = []
        with self.assertRaisesRegex(PreflightError, "exact_condition_batch_hash_mismatch"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               upstream_precheck=lambda: True,
                               authenticated_gate_check=lambda: called.append(1))
        self.assertEqual(called, [])

    def test_upstream_gate_failure_never_returns_frozen_inputs(self):
        rows, hashes = fixture()
        with self.assertRaisesRegex(PreflightError, "authenticated_upstream_gate_failed"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               upstream_precheck=lambda: True,
                               authenticated_gate_check=lambda: (_ for _ in ()).throw(
                                   RuntimeError("private details")))

    def test_final_gate_false_never_returns_frozen_inputs(self):
        rows, hashes = fixture()
        FakeBatch.to_calls = 0
        with self.assertRaisesRegex(PreflightError, "authenticated_upstream_gate_not_passed"):
            freeze_all_batches(
                triplets=rows, token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=["r1", "r2", "r3"],
                expected_batch_hashes=hashes, upstream_precheck=lambda: True,
                authenticated_gate_check=lambda: False)
        self.assertEqual(FakeBatch.to_calls, 9)

    def test_nonfinite_input_fails_closed(self):
        rows, hashes = fixture()
        rows[0]["batches"]["full"]["pixel_values"] = FakeTensor([float("nan")])
        with self.assertRaisesRegex(PreflightError, "nonfinite"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 15],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               upstream_precheck=lambda: True,
                               authenticated_gate_check=lambda: True)

    def test_duplicate_token_ids_rejected(self):
        rows, hashes = fixture()
        with self.assertRaisesRegex(PreflightError, "token_contract_invalid"):
            freeze_all_batches(triplets=rows, token_ids=[11, 12, 13, 14, 14],
                               expected_record_ids=["r1", "r2", "r3"],
                               expected_batch_hashes=hashes,
                               upstream_precheck=lambda: True,
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
            upstream_precheck=lambda: True,
            authenticated_gate_check=lambda: True)
        frozen = result["triplets"][0]["batches"]["full"]["torch_values"]
        source[0] = 99.0
        self.assertEqual(float(frozen[0]), 1.0)

    def test_stream_hash_preserves_tensor_bytes_dtype_shape_and_scalar_shape(self):
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch unavailable")
        values = torch.tensor([1.0, -2.5, 0.125], dtype=torch.bfloat16)
        base = batch_sha256({"pixels": values})
        self.assertEqual(base, batch_sha256({"pixels": values.clone()}))
        self.assertNotEqual(base, batch_sha256({"pixels": values.to(torch.float32)}))
        values[1] = -2.0
        self.assertNotEqual(base, batch_sha256({"pixels": values}))
        scalar = torch.tensor(3.0, dtype=torch.float32)
        self.assertEqual(64, len(batch_sha256({"scalar": scalar})))
        empty = torch.empty((0, 3), dtype=torch.bfloat16)
        self.assertEqual(batch_sha256({"empty": empty}),
                         batch_sha256({"empty": empty.clone()}))
        noncontiguous = torch.arange(12, dtype=torch.int32).reshape(3, 4)[:, ::2]
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(batch_sha256({"x": noncontiguous}),
                         batch_sha256({"x": noncontiguous.contiguous()}))
        self.assertNotEqual(batch_sha256({"x": torch.tensor([0.0])}),
                            batch_sha256({"x": torch.tensor([-0.0])}))

    def test_container_types_and_nested_mapping_order_have_stable_domains(self):
        self.assertEqual(batch_sha256({"x": 1}),
                         "0bd1c0f20db2f25be462f5362663b968df252687b74c06551320b71f7f7bc7e9")
        self.assertNotEqual(batch_sha256({"x": [1, 2]}),
                            batch_sha256({"x": (1, 2)}))
        self.assertEqual(batch_sha256({"outer": {"b": 2, "a": 1}}),
                         batch_sha256({"outer": {"a": 1, "b": 2}}))
        self.assertNotEqual(batch_sha256({"x": True}), batch_sha256({"x": 1}))
        self.assertNotEqual(batch_sha256({"x": 1}), batch_sha256({"x": 1.0}))

    def test_cpu_snapshot_preserves_tuple_container_type(self):
        rows, hashes = fixture()
        rows[0]["batches"]["full"]["nested"] = (1, [2, 3])
        hashes["r1"]["full"] = batch_sha256(rows[0]["batches"]["full"])
        result = freeze_all_batches(
            triplets=rows, token_ids=[11, 12, 13, 14, 15],
            expected_record_ids=["r1", "r2", "r3"],
            expected_batch_hashes=hashes,
            upstream_precheck=lambda: True,
            authenticated_gate_check=lambda: True)
        self.assertEqual(result["triplets"][0]["batches"]["full"]["nested"],
                         (1, [2, 3]))
        self.assertEqual(result["batch_hash_schema"], "nextgqa_typed_batch_stream_v1")

    def test_stream_hash_rejects_nonfinite_and_unsupported_tensor_layouts(self):
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch unavailable")
        with self.assertRaisesRegex(PreflightError, "nonfinite"):
            batch_sha256({"pixels": torch.tensor([float("inf")])})
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sparse = torch.sparse_coo_tensor(
                [[0]], [1.0], (1,), check_invariants=True)
        with self.assertRaisesRegex(PreflightError, "layout_unsupported"):
            batch_sha256({"pixels": sparse})


if __name__ == "__main__":
    unittest.main()
