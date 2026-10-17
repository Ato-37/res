from __future__ import annotations

import copy
import unittest

from qta_nine_branch_plan import build_qta_nine_branch_plan
from qta_prescore_mapping_binding import audit_qta_prescore_mapping_bundle
from qta_synthetic_cohort_plan_preflight import preflight_synthetic_qta_plan_bundle
from test_qta_synthetic_cohort_preflight import synthetic_mapping_inputs


def _bundle():
    mapping = synthetic_mapping_inputs()
    mapping_report = audit_qta_prescore_mapping_bundle(**mapping)
    plans, score_pins, selection_pins = [], {}, {}
    for i, (token, row_hash) in enumerate(zip(
            mapping["expected_cohort_tokens"], mapping_report["row_binding_sha256"],
            strict=True)):
        score = (chr(100 + i) * 64)
        plan = build_qta_nine_branch_plan(
            row_binding_sha256=row_hash,
            temporal_groups=((0,), (1,), (2,), (3,), (4,)),
            target_group_indices=(1,), sham_group_indices=(3,),
            evidence_group_indices=(1, 3), control_group_indices=(0, 4),
            typed_batch_sha256=mapping["expected_processor_batch_sha256_by_token"][token],
            score_binding_sha256=score,
            intervention_layers_1_based=tuple(range(5, 17)),
            logit_bias=0.6931471805599453,
            sham_seed="qta-sham-seed-20261018-v1")
        plans.append(plan)
        score_pins[token] = score
        selection_pins[token] = plan["selection_spec_sha256"]
    return mapping, plans, score_pins, selection_pins


class QtaSyntheticCohortPlanPreflightTests(unittest.TestCase):
    def _run(self, mapping, plans, score_pins, selection_pins):
        return preflight_synthetic_qta_plan_bundle(
            mapping_inputs=mapping, plans=plans,
            expected_score_binding_sha256_by_token=score_pins,
            expected_selection_spec_sha256_by_token=selection_pins)

    def _assert_blocked(self, report):
        self.assertEqual(report["status"], "blocked_synthetic_qta_cohort_plan_preflight")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["validated_plans"], 0)
        self.assertEqual(report["requested_forwards"], 27)
        self.assertEqual(report["planned_forwards"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self.assertIsNone(report["mapping_bundle_binding_sha256"])
        self.assertIsNone(report["plan_bundle_binding_sha256"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_complete_synthetic_bundle_binds_rows_and_plans_without_authority(self):
        mapping, plans, score_pins, selection_pins = _bundle()
        report = self._run(mapping, plans, score_pins, selection_pins)
        self.assertEqual(report["status"],
                         "three_row_structural_and_plan_binding_passed_unvalidated")
        self.assertEqual(report["verified_structural_rows"], 3)
        self.assertEqual(report["validated_plans"], 3)
        self.assertEqual(report["requested_forwards"], 27)
        self.assertEqual(report["planned_forwards"], 27)
        self.assertEqual(report["completed_forwards"], 0)
        self.assertEqual(len(report["row_binding_sha256"]), 3)
        self.assertEqual(report["data_provenance_status"], "not_verified_by_this_preflight")
        self.assertFalse(report["human_review_authenticated"])
        self.assertFalse(report["prediction_link_authenticated"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_mapping_failure_blocks_every_plan(self):
        mapping, plans, score_pins, selection_pins = _bundle()
        mapping = copy.deepcopy(mapping)
        mapping["processor_rows"][1]["review_token"] = "wrong-token"
        self._assert_blocked(self._run(mapping, plans, score_pins, selection_pins))

    def test_swapped_plans_or_wrong_external_score_pin_blocks_all(self):
        mapping, plans, score_pins, selection_pins = _bundle()
        swapped = copy.deepcopy(plans)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self._assert_blocked(self._run(mapping, swapped, score_pins, selection_pins))

        wrong_score = dict(score_pins)
        wrong_score[mapping["expected_cohort_tokens"][1]] = "f" * 64
        self._assert_blocked(self._run(mapping, plans, wrong_score, selection_pins))

    def test_short_cohort_and_invalid_selection_pin_fail_closed(self):
        mapping, plans, score_pins, selection_pins = _bundle()
        self._assert_blocked(self._run(mapping, plans[:2], score_pins, selection_pins))
        wrong_selection = dict(selection_pins)
        wrong_selection[mapping["expected_cohort_tokens"][0]] = "not-a-sha"
        self._assert_blocked(self._run(mapping, plans, score_pins, wrong_selection))


if __name__ == "__main__":
    unittest.main()
