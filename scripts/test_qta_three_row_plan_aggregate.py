from __future__ import annotations

import unittest

from qta_nine_branch_plan import build_qta_nine_branch_plan
from qta_three_row_plan_aggregate import aggregate_qta_nine_branch_plans


def _plan(index, *, row_hash, temporal_groups=None, layers=tuple(range(5, 17)),
          bias=0.6931471805599453,
          sham_seed="fixed-test-seed"):
    return build_qta_nine_branch_plan(
        row_binding_sha256=row_hash,
        temporal_groups=(temporal_groups or ((0, 1), (2, 3), (4, 5), (6, 7))),
        target_group_indices=(index % 2,), sham_group_indices=((index + 2) % 4,),
        evidence_group_indices=(0,), control_group_indices=(2,),
        typed_batch_sha256=chr(97 + index) * 64,
        score_binding_sha256=chr(100 + index) * 64,
        intervention_layers_1_based=layers, logit_bias=bias, sham_seed=sham_seed)


def _inputs(count=3):
    expected = []
    rows = []
    for index in range(count):
        row_hash = str(index + 1) * 64
        plan = _plan(index, row_hash=row_hash)
        expected.append({
            "row_binding_sha256": row_hash,
            "typed_batch_sha256": plan["typed_batch_sha256"],
            "score_binding_sha256": plan["score_binding_sha256"],
            "video_grid_thw": [4, 1, 1],
            "evidence_group_indices": [0],
            "control_group_indices": [2],
            "temporal_group_token_positions": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "selection_spec_sha256": plan["selection_spec_sha256"],
        })
        rows.append({"record_position": index + 1,
                     "row_binding_sha256": row_hash, "plan": plan})
    return rows, expected


class QtaThreeRowPlanAggregateTests(unittest.TestCase):
    def assert_blocked_without_partial_acceptance(self, report):
        self.assertEqual(report["status"], "blocked_qta_plan_bundle")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["requested_forwards"], 27)
        self.assertEqual(report["validated_plans"], 0)
        self.assertEqual(report["planned_forwards"], 0)
        self.assertEqual(report["completed_forwards"], 0)
        self.assertEqual(report["record_plan_sha256"], [])
        self.assertIsNone(report["bundle_binding_sha256"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_valid_three_record_bundle_keeps_fixed_denominator_and_no_authority(self):
        rows, expected = _inputs()
        report = aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected)
        self.assertEqual(report["status"], "three_plans_bound_unvalidated")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["validated_plans"], 3)
        self.assertEqual(report["requested_forwards"], 27)
        self.assertEqual(report["planned_forwards"], 27)
        self.assertEqual(report["completed_forwards"], 0)
        self.assertEqual(len(report["record_plan_sha256"]), 3)
        self.assertEqual(len({row["plan_sha256"] for row in report["record_plan_sha256"]}), 3)
        self.assertTrue(report["bundle_binding_sha256"])
        self.assertEqual(report["data_provenance_status"], "not_verified_by_aggregator")
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_short_or_oversized_cohort_fails_without_denominator_shrink(self):
        rows, expected = _inputs()
        for count in (2, 4):
            with self.subTest(count=count):
                self.assert_blocked_without_partial_acceptance(
                    aggregate_qta_nine_branch_plans(
                        rows=rows[:count] if count < 3 else rows + [rows[-1]],
                        expected_rows=expected[:count] if count < 3 else expected + [expected[-1]]))

    def test_any_invalid_middle_row_blocks_all_three(self):
        rows, expected = _inputs()
        rows[1]["plan"]["branches"].pop()
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_reordered_or_duplicate_row_bindings_block_all_three(self):
        rows, expected = _inputs()
        reordered = [rows[1], rows[0], rows[2]]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=reordered, expected_rows=expected))
        rows, expected = _inputs()
        rows[2]["row_binding_sha256"] = rows[1]["row_binding_sha256"]
        expected[2]["row_binding_sha256"] = expected[1]["row_binding_sha256"]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_plans_cannot_be_swapped_between_frozen_row_bindings(self):
        rows, expected = _inputs()
        rows[0]["plan"], rows[1]["plan"] = rows[1]["plan"], rows[0]["plan"]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

        rows, expected = _inputs()
        rows[0]["plan"], rows[1]["plan"] = rows[1]["plan"], rows[0]["plan"]
        for key in ("typed_batch_sha256", "score_binding_sha256"):
            expected[0][key], expected[1][key] = expected[1][key], expected[0][key]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_wrong_expected_batch_or_score_hash_blocks_all_three(self):
        for field in ("typed_batch_sha256", "score_binding_sha256"):
            with self.subTest(field=field):
                rows, expected = _inputs()
                expected[1][field] = "f" * 64
                self.assert_blocked_without_partial_acceptance(
                    aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_processor_grid_or_evidence_control_mapping_mismatch_blocks_all_three(self):
        rows, expected = _inputs()
        expected[1]["video_grid_thw"] = [5, 1, 1]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_invalid_expected_temporal_group_geometry_blocks_all_three(self):
        for malformed in (
                [[0], [1, 2], [3], [4]],
                [[0, 1], [3, 4], [5, 6], [7, 8]]):
            with self.subTest(malformed=malformed):
                rows, expected = _inputs()
                expected[0]["temporal_group_token_positions"] = malformed
                self.assert_blocked_without_partial_acceptance(
                    aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

        rows, expected = _inputs()
        rows[0]["plan"] = _plan(
            0, row_hash=expected[0]["row_binding_sha256"],
            temporal_groups=((100, 101), (102, 103), (104, 105), (106, 107)))
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_bool_cannot_masquerade_as_plan_integer_fields(self):
        import copy
        for field in ("schema_version", "temporal_group_count", "requested_forwards"):
            with self.subTest(field=field):
                rows, expected = _inputs()
                rows[0]["plan"] = copy.deepcopy(rows[0]["plan"])
                rows[0]["plan"][field] = True
                self.assert_blocked_without_partial_acceptance(
                    aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))
        rows, expected = _inputs()
        expected[1]["evidence_group_indices"] = [1]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_treatment_configuration_must_match_across_rows(self):
        rows, expected = _inputs()
        rows[1]["plan"] = _plan(1, row_hash=expected[1]["row_binding_sha256"], layers=(5, 7))
        expected[1]["typed_batch_sha256"] = rows[1]["plan"]["typed_batch_sha256"]
        expected[1]["score_binding_sha256"] = rows[1]["plan"]["score_binding_sha256"]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_cohort_wide_consistent_nonprimary_layers_are_rejected(self):
        rows, expected = _inputs()
        for i, row in enumerate(rows):
            row["plan"] = _plan(i, row_hash=row["row_binding_sha256"], layers=(5, 7))
            expected[i]["typed_batch_sha256"] = row["plan"]["typed_batch_sha256"]
            expected[i]["score_binding_sha256"] = row["plan"]["score_binding_sha256"]
            expected[i]["selection_spec_sha256"] = row["plan"]["selection_spec_sha256"]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

    def test_treatment_layer_edges_and_one_ulp_bias_changes_are_rejected(self):
        import math
        for layers, bias in (
                (tuple(range(4, 16)), 0.6931471805599453),
                (tuple(range(6, 18)), 0.6931471805599453),
                (tuple(range(5, 17)), math.nextafter(math.log(2.0), -math.inf)),
                (tuple(range(5, 17)), math.nextafter(math.log(2.0), math.inf))):
            with self.subTest(layers=layers, bias=bias):
                rows, expected = _inputs()
                rows[0]["plan"] = _plan(
                    0, row_hash=expected[0]["row_binding_sha256"],
                    layers=layers, bias=bias)
                expected[0]["typed_batch_sha256"] = rows[0]["plan"]["typed_batch_sha256"]
                expected[0]["score_binding_sha256"] = rows[0]["plan"]["score_binding_sha256"]
                expected[0]["selection_spec_sha256"] = rows[0]["plan"]["selection_spec_sha256"]
                self.assert_blocked_without_partial_acceptance(
                    aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))

        rows, expected = _inputs()
        rows[1]["plan"] = _plan(
            1, row_hash=expected[1]["row_binding_sha256"], sham_seed="other-seed")
        expected[1]["typed_batch_sha256"] = rows[1]["plan"]["typed_batch_sha256"]
        expected[1]["score_binding_sha256"] = rows[1]["plan"]["score_binding_sha256"]
        expected[1]["selection_spec_sha256"] = rows[1]["plan"]["selection_spec_sha256"]
        self.assert_blocked_without_partial_acceptance(
            aggregate_qta_nine_branch_plans(rows=rows, expected_rows=expected))


if __name__ == "__main__":
    unittest.main()
