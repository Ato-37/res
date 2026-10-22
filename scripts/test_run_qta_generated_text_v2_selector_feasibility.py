from __future__ import annotations

import unittest

from run_qta_generated_text_v2_selector_feasibility import (
    POOL_START, POOL_STOP_EXCLUSIVE, _eligible_sets, candidate_indices,
    pool_attempts_complete, validate_candidate_lock,
    selector_failure_requires_stop, validate_successful_smoke_receipt,
)


class QtaV2SelectorFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.groups = tuple((10 + i * 2, 11 + i * 2) for i in range(6))
        self.mapped = {
            "evidence_group_indices": [2],
            "evidence_token_positions": list(self.groups[2]),
        }

    def test_freezes_three_way_disjoint_equal_key_sets(self):
        result = _eligible_sets(mapped=self.mapped,
            group_positions=self.groups,
            group_scores=[9.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        self.assertEqual(result["qta_group_indices"], [0])
        self.assertEqual(result["evidence_group_indices"], [2])
        self.assertEqual(result["control_group_indices"], [1])
        self.assertEqual(result["evidence_key_count"], 2)
        self.assertEqual(result["control_key_count"], 2)
        self.assertEqual(result["qta_key_count"], 2)
        self.assertTrue(result["pairwise_disjoint_equal_key_sets"])

    def test_selector_evidence_overlap_is_ineligible(self):
        with self.assertRaisesRegex(ValueError, "qta_evidence_groups_overlap"):
            _eligible_sets(mapped=self.mapped, group_positions=self.groups,
                group_scores=[0.1, 0.2, 9.0, 0.3, 0.4, 0.5])

    def test_expanded_group_key_budget_mismatch_is_ineligible(self):
        groups = (self.groups[0], self.groups[1], self.groups[2],
                  (16,), self.groups[4], self.groups[5])
        with self.assertRaisesRegex(ValueError, "budget_mismatch"):
            _eligible_sets(mapped=self.mapped, group_positions=groups,
                group_scores=[0.1, 9.0, 0.5, 0.3, 0.2, 0.1])

    def test_candidate_lock_binds_exact_current_dataset_hashes(self):
        hashes = {"test.csv": "a" * 64, "gsub_test.json": "b" * 64}
        lock = {"schema_version": 1,
            "status": "candidate_order_frozen_geometry_not_preflighted_not_authorizing",
            "candidates": [{} for _ in range(POOL_STOP_EXCLUSIVE)],
            "dataset_sha256": hashes}
        validate_candidate_lock(lock, hashes)
        with self.assertRaisesRegex(ValueError, "candidate_lock"):
            validate_candidate_lock(lock, {**hashes, "test.csv": "c" * 64})

    def test_candidate_lock_rejects_boolean_schema_and_short_pool(self):
        hashes = {"test.csv": "a" * 64}
        lock = {"schema_version": True,
            "status": "candidate_order_frozen_geometry_not_preflighted_not_authorizing",
            "candidates": [{} for _ in range(POOL_STOP_EXCLUSIVE)],
            "dataset_sha256": hashes}
        with self.assertRaisesRegex(ValueError, "candidate_lock"):
            validate_candidate_lock(lock, hashes)
        lock["schema_version"] = 1
        lock["candidates"] = lock["candidates"][:-1]
        with self.assertRaisesRegex(ValueError, "candidate_lock"):
            validate_candidate_lock(lock, hashes)

    def test_candidate_indices_are_fixed_and_bounded(self):
        self.assertEqual(candidate_indices("smoke"), ([0], 1))
        indices, required = candidate_indices("pool")
        self.assertEqual(indices, list(range(4, 16)))
        self.assertEqual(required, 4)

    def test_pool_requires_hash_bound_successful_selector_smoke(self):
        source = {"runner.py": "a" * 64}
        data = {"test.csv": "b" * 64}
        receipt = {"schema_version": 1,
            "status": "selector_engineering_smoke_completed_no_scoring",
            "mode": "smoke", "candidate_lock_sha256": "c" * 64,
            "smoke_preflight_sha256": "d" * 64,
            "source_sha256": source, "dataset_sha256": data,
            "candidate_indices_attempted": [0], "cpu_geometry_attempted": 1,
            "cpu_geometry_passed": 1, "selector_forwards_attempted": 1,
            "selector_forwards_completed": 1, "eligible_count": 1,
            "generation_performed": False, "target_scoring_performed": False,
            "scoring_authorized": False, "direct_faithfulness": False,
            "faithfulness_result": False,
            "attempts": [{"lock_index": 0, "status": "engineering_smoke_eligible"}]}
        validate_successful_smoke_receipt(receipt, lock_sha="c" * 64,
            preflight_sha="d" * 64, source_hashes=source, data_hashes=data)
        with self.assertRaisesRegex(ValueError, "successful_hash_bound"):
            validate_successful_smoke_receipt({**receipt, "source_sha256": {}},
                lock_sha="c" * 64, preflight_sha="d" * 64,
                source_hashes=source, data_hashes=data)
        with self.assertRaisesRegex(ValueError, "successful_hash_bound"):
            validate_successful_smoke_receipt(None, lock_sha="c" * 64,
                preflight_sha="d" * 64, source_hashes=source, data_hashes=data)

    def test_pool_accounts_for_selector_failure_then_four_eligible_rows(self):
        statuses = ["selector_failure", "eligible", "eligible", "eligible", "eligible"]
        attempts = [{"lock_index": POOL_START + i, "status": status}
                    for i, status in enumerate(statuses)]
        self.assertTrue(pool_attempts_complete(attempts=attempts,
            eligible_count=4, required=4))
        self.assertFalse(pool_attempts_complete(attempts=attempts[:4],
            eligible_count=3, required=4))
        self.assertFalse(pool_attempts_complete(attempts=attempts[1:],
            eligible_count=4, required=4))

    def test_selector_memory_errors_stop_fixed_pool(self):
        self.assertTrue(selector_failure_requires_stop(
            RuntimeError("CUDA out of memory"), "selector:RuntimeError:CUDA out of memory"))
        self.assertTrue(selector_failure_requires_stop(
            RuntimeError("guard"), "selector:RuntimeError:per_selector_free_vram_guard_failed"))
        self.assertFalse(selector_failure_requires_stop(
            RuntimeError("capture shape invalid"), "selector:RuntimeError:capture_shape_invalid"))


if __name__ == "__main__":
    unittest.main()
