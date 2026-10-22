from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_qta_generated_text_v2_scorer import (
    POOL_RECEIPT_NAME, SCORING_BRANCHES, SELECTOR_SMOKE_NAME,
    candidate_interventions, validate_scoring_smoke_receipt,
    validate_selector_pool, score_candidate,
)


class QtaV2ScorerGateTests(unittest.TestCase):
    def setUp(self):
        self.lock_sha = "a" * 64
        self.preflight_sha = "b" * 64
        self.selector_smoke_sha = "c" * 64
        self.pool_sha = "d" * 64
        self.sources = {"runner.py": "e" * 64}
        self.data = {"test.csv": "f" * 64}
        self.eligible = [{"lock_index": i, "status": "eligible"}
                         for i in (7, 8, 9, 10)]
        self.lock = {"status": "candidate_order_frozen_geometry_not_preflighted_not_authorizing",
            "candidates": [{} for _ in range(16)], "dataset_sha256": self.data}
        self.preflight = {"status": "geometry_preflight_passed_not_scoring_authorization",
            "candidate_order_sha256": self.lock_sha, "dataset_sha256": self.data}
        self.selector_smoke = {
            "schema_version": 1,
            "status": "selector_engineering_smoke_completed_no_scoring",
            "mode": "smoke", "candidate_lock_sha256": self.lock_sha,
            "smoke_preflight_sha256": self.preflight_sha,
            "source_sha256": self.sources, "dataset_sha256": self.data,
            "candidate_indices_attempted": [0], "cpu_geometry_attempted": 1,
            "cpu_geometry_passed": 1, "selector_forwards_attempted": 1,
            "selector_forwards_completed": 1, "eligible_count": 1,
            "generation_performed": False, "target_scoring_performed": False,
            "scoring_authorized": False, "direct_faithfulness": False,
            "faithfulness_result": False,
            "attempts": [{"lock_index": 0, "status": "engineering_smoke_eligible"}]}
        self.pool = {
            "status": "v2_selector_feasibility_passed_no_generation_or_scoring",
            "mode": "pool", "candidate_lock_sha256": self.lock_sha,
            "smoke_preflight_sha256": self.preflight_sha,
            "smoke_receipt_name": SELECTOR_SMOKE_NAME,
            "smoke_receipt_sha256": self.selector_smoke_sha,
            "source_sha256": self.sources, "dataset_sha256": self.data,
            "direct_faithfulness": False, "faithfulness_result": False,
            "generation_performed": False, "target_scoring_performed": False,
            "scoring_authorized": False,
            "candidate_indices_attempted": list(range(4, 11)),
            "eligible_count": 4, "cpu_geometry_attempted": 7,
            "cpu_geometry_passed": 7, "selector_forwards_attempted": 7,
            "selector_forwards_completed": 7,
            "exclusions_by_reason": {
                "eligibility:qta_disjoint_equal_count_control_unavailable": 2,
                "eligibility:qta_evidence_groups_overlap": 1},
            "attempts": ([{"lock_index": i, "status": "selector_ineligible"}
                          for i in (4, 5, 6)]
                + [{"lock_index": i, "status": "eligible"} for i in (7, 8, 9, 10)]),
            "unattempted_pool_indices": list(range(11, 16)),
            "eligible_rows": self.eligible}

    def test_selector_receipt_hash_chain_and_fixed_eligible_rows_are_accepted(self):
        self.assertEqual(validate_selector_pool(self.pool, pool_sha=self.pool_sha,
            lock=self.lock, lock_sha=self.lock_sha,
            preflight=self.preflight, preflight_sha=self.preflight_sha,
            selector_smoke=self.selector_smoke,
            selector_smoke_sha=self.selector_smoke_sha,
            source_hashes=self.sources, data_hashes=self.data), self.eligible)

    def test_selector_receipt_rejects_changed_source_or_partition(self):
        with self.assertRaisesRegex(ValueError, "binding_invalid"):
            validate_selector_pool({**self.pool,
                "source_sha256": {"runner.py": "0" * 64}},
                pool_sha=self.pool_sha, lock=self.lock, lock_sha=self.lock_sha,
                preflight=self.preflight, preflight_sha=self.preflight_sha,
                selector_smoke=self.selector_smoke,
                selector_smoke_sha=self.selector_smoke_sha,
                source_hashes=self.sources, data_hashes=self.data)
        with self.assertRaisesRegex(ValueError, "partition_or_counts"):
            validate_selector_pool({**self.pool,
                "candidate_indices_attempted": list(range(5, 12))},
                pool_sha=self.pool_sha, lock=self.lock, lock_sha=self.lock_sha,
                preflight=self.preflight, preflight_sha=self.preflight_sha,
                selector_smoke=self.selector_smoke,
                selector_smoke_sha=self.selector_smoke_sha,
                source_hashes=self.sources, data_hashes=self.data)

    def test_scoring_smoke_receipt_requires_all_four_branches_and_same_pool(self):
        target_ids = [4, 5]
        scores = {name: {"target_token_count": 2,
            "target_logprob_mean": -1.0, "target_token_logprobs": [-1.0, -1.0]}
            for name in SCORING_BRANCHES}
        receipt = {"status": "completed_v2_exploratory_smoke_no_faithfulness_claim",
            "mode": "smoke", "selector_pool_sha256": self.pool_sha,
            "source_sha256": self.sources, "dataset_sha256": self.data,
            "candidate_lock_indices": [7], "requested_n": 1, "effective_n": 1, "failures": [],
            "generation_performed": True, "target_scoring_performed": True,
            "direct_faithfulness": False, "faithfulness_result": False,
            "claim_truth_reviewed": False,
            "branch_counts": {name: {"requested": 1, "attempted": 1, "completed": 1}
                               for name in SCORING_BRANCHES},
            "generation_calls": {"requested": 1, "attempted": 1, "completed": 1},
            "records": [{"lock_index": 7, "target_token_ids": target_ids,
                "target_token_count": 2,
                "target_token_ids_sha256": hashlib.sha256(b"[4,5]").hexdigest(),
                "target_score_by_condition": scores,
                "primary_contrast": {
                    "baseline_annotation_selective_text_dependence": 0.0,
                    "qta_annotation_selective_text_dependence": 0.0,
                    "delta_annotation_selective_text_dependence": 0.0}}]}
        validate_scoring_smoke_receipt(receipt, pool_sha=self.pool_sha,
            source_hashes=self.sources, data_hashes=self.data)
        broken = {**receipt, "branch_counts": dict(receipt["branch_counts"])}
        broken["branch_counts"][SCORING_BRANCHES[-1]] = {
            "requested": 1, "attempted": 1, "completed": 0}
        with self.assertRaisesRegex(ValueError, "successful_hash_bound"):
            validate_scoring_smoke_receipt(broken, pool_sha=self.pool_sha,
                source_hashes=self.sources, data_hashes=self.data)

    def test_one_generated_suffix_is_reused_for_exactly_four_scoring_branches(self):
        import torch
        groups = tuple((10 + 2*i, 11 + 2*i) for i in range(6))
        input_ids = torch.tensor([list(range(30))])
        prepared = {"geometry_sha256": "g" * 64,
            "batch": {"input_ids": input_ids},
            "rendered": "fixed rendered prompt",
            "mapped": {"groups": [{"token_positions": list(g)} for g in groups],
                "evidence_group_indices": [2],
                "evidence_token_positions": list(groups[2])}}
        selector_row = {"lock_index": 7, "geometry_sha256": "g" * 64,
            "evidence_group_indices": [2], "qta_group_indices": [0],
            "control_group_indices": [1],
            "group_attention_scores": [9.0, 1.0, 0.5, 0.3, 0.2, 0.1],
            "evidence_key_count": 2, "qta_key_count": 2,
            "control_key_count": 2}
        class Model:
            def __init__(self): self.calls = 0
            def generate(self, **kwargs):
                self.calls += 1
                self.kwargs = kwargs
                return SimpleNamespace(sequences=torch.tensor(
                    [list(range(30)) + [31, 32, 33]]))
        model = Model()
        processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=33))
        branch_counts = {name: {"requested": 1, "attempted": 0, "completed": 0}
                         for name in SCORING_BRANCHES}
        generation_calls = {"requested": 1, "attempted": 0, "completed": 0}
        forwarded = []
        values = iter((0.1, 0.2, 0.3, 0.5))
        def fake_forward(_model, _batch, **kwargs):
            forwarded.append(kwargs)
            return {"target_logprob_mean": next(values)}
        with patch("run_qta_generated_text_v2_scorer.prepare_candidate_geometry",
                   return_value=prepared), \
             patch("run_qta_generated_text_v2_scorer._move_batch_to_cuda",
                   side_effect=lambda batch, _torch: batch), \
             patch("run_qta_generated_text_v2_scorer.append_target_to_unpadded_batch",
                   return_value=({}, (29, 30, 31))), \
             patch("run_qta_generated_text_v2_scorer._forward", side_effect=fake_forward), \
             patch("torch.cuda.empty_cache"):
            result = score_candidate(repo=Path("."),
                candidate={"qid": "qid-private", "video_sha256": "v" * 64},
                selector_row=selector_row, processor=processor, model=model,
                torch=torch, rows={}, gsub={}, branch_counts=branch_counts,
                generation_calls=generation_calls, stage={})
        self.assertEqual(model.calls, 1)
        self.assertEqual(generation_calls, {"requested": 1, "attempted": 1, "completed": 1})
        self.assertEqual(len(forwarded), 4)
        self.assertTrue(all(call["target_ids"] == (31, 32, 33) for call in forwarded))
        self.assertEqual(set(branch_counts), set(SCORING_BRANCHES))
        self.assertTrue(all(c == {"requested": 1, "attempted": 1, "completed": 1}
            for c in branch_counts.values()))
        self.assertAlmostEqual(result["primary_contrast"][
            "delta_annotation_selective_text_dependence"], 0.1)

    def test_frozen_evidence_qta_control_sets_are_reconstructed(self):
        groups = tuple((10 + 2*i, 11 + 2*i) for i in range(6))
        mapped = {"evidence_group_indices": [2],
            "evidence_token_positions": list(groups[2])}
        row = {"evidence_group_indices": [2], "qta_group_indices": [0],
            "control_group_indices": [1],
            "group_attention_scores": [9.0, 1.0, 0.5, 0.3, 0.2, 0.1],
            "evidence_key_count": 2, "qta_key_count": 2,
            "control_key_count": 2}
        sets = candidate_interventions(mapped=mapped,
            group_positions=groups, frozen_selector_row=row)
        self.assertEqual(sets["qta_groups"], (0,))
        self.assertEqual(sets["control_groups"], (1,))
        self.assertEqual(len(sets["evidence"]), len(sets["control"]))
        with self.assertRaisesRegex(ValueError, "changed"):
            candidate_interventions(mapped=mapped, group_positions=groups,
                frozen_selector_row={**row, "control_group_indices": [3]})


if __name__ == "__main__":
    unittest.main()
