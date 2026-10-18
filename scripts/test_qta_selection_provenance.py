from __future__ import annotations

import copy
import hashlib
import math
import unittest

from qta_selection_provenance import (
    FROZEN_SELECTOR_LAYERS_1_BASED,
    FROZEN_SELECTOR_SOURCE_SHA256,
    build_qta_selection_provenance,
    selection_provenance_sha256,
    validate_qta_selection_provenance,
)
from qta_nine_branch_plan import build_qta_nine_branch_plan
from qta_nine_branch_plan import _plan_digest


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class QtaSelectionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.groups = [[2 * i, 2 * i + 1] for i in range(11)]
        # Equal maxima exercise the frozen earliest-index tie break.
        self.layer_masses = [
            [1.0 if i in (1, 7) else float(i) / 100 for i in range(11)]
            for _ in FROZEN_SELECTOR_LAYERS_1_BASED
        ]
        self.expected = {
            "record_token": _h("record token"),
            "opaque_review_token": "synthetic_record_token01",
            "row_binding_sha256": _h("row"),
            "typed_batch_sha256": _h("batch"),
            "score_binding_sha256": _h("score"),
            "baseline_pq_row_sha256": _h("pq-row"),
            "prompt_sha256": _h("prompt"),
            "input_ids_sha256": _h("ids"),
            "input_length": 36,
            "score_position": 35,
            "question_positions": list(range(30, 35)),
            "temporal_groups": self.groups,
            "selector_source_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
            "selector_layers_1_based": FROZEN_SELECTOR_LAYERS_1_BASED,
            "sham_seed": "qta-sham-seed-20261018-v1",
        }
        self.value = build_qta_selection_provenance(
            **self.expected, layer_group_masses=self.layer_masses)

    def test_selection_recomputed_and_authority_denied(self):
        report = validate_qta_selection_provenance(
            self.value, expected_bindings=self.expected)
        self.assertEqual(report["status"],
                         "selection_arithmetic_reproduced_non_authorizing")
        self.assertEqual(report["target_group_indices"], [1, 7])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertEqual(report["authorized_forwards"], 0)

    def test_canonical_float_hex_round_trip_and_stable_hash(self):
        first = selection_provenance_sha256(self.value)
        report = validate_qta_selection_provenance(
            copy.deepcopy(self.value), expected_bindings=self.expected)
        self.assertEqual(report["validated_group_count"], 11)
        self.assertEqual(selection_provenance_sha256(self.value), first)
        self.assertEqual(len(self.value["layer_group_mass_f32_be_hex"][0][0]), 8)
        self.assertEqual(len(self.value["group_scores_f64_be_hex"][0]), 16)

    def test_sham_uses_protocol_raw_review_token(self):
        token = self.expected["opaque_review_token"]
        prefix = f"qta-sham-v1\0{self.expected['sham_seed']}\0{token}\0"
        raw_ranked = sorted(range(len(self.groups)), key=lambda i: (
            hashlib.sha256(f"{prefix}{i}".encode()).digest(), i))
        self.assertEqual(self.value["sham_ranked_group_indices"], raw_ranked)
        hashed_prefix = f"qta-sham-v1\0{self.expected['sham_seed']}\0{_h(token)}\0"
        hashed_ranked = sorted(range(len(self.groups)), key=lambda i: (
            hashlib.sha256(f"{hashed_prefix}{i}".encode()).digest(), i))
        self.assertNotEqual(raw_ranked, hashed_ranked)

    def test_changed_per_layer_mass_without_recomputed_score_is_rejected(self):
        changed = copy.deepcopy(self.value)
        changed["layer_group_mass_f32_be_hex"][0][0] = "3f800000"
        with self.assertRaisesRegex(ValueError, "qta_selection_recomputation_mismatch"):
            validate_qta_selection_provenance(changed, expected_bindings=self.expected)

    def test_nonfinite_negative_or_malformed_float_encoding_is_rejected(self):
        for encoded in ("7f800000", "7fc00000", "bf800000", "xyz"):
            changed = copy.deepcopy(self.value)
            changed["layer_group_mass_f32_be_hex"][0][0] = encoded
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                validate_qta_selection_provenance(changed,
                                                  expected_bindings=self.expected)

    def test_wrong_bindings_and_group_geometry_fail_closed(self):
        wrong_batch = dict(self.expected, typed_batch_sha256=_h("other batch"))
        with self.assertRaisesRegex(ValueError, "qta_selection_binding_mismatch"):
            validate_qta_selection_provenance(self.value, expected_bindings=wrong_batch)
        wrong_positions = dict(self.expected, question_positions=[31, 35])
        with self.assertRaisesRegex(ValueError, "qta_selection_geometry_or_protocol_mismatch"):
            validate_qta_selection_provenance(self.value,
                                              expected_bindings=wrong_positions)
        wrong_source = dict(self.expected, selector_source_sha256=_h("wrong selector"))
        wrong_source_value = copy.deepcopy(self.value)
        wrong_source_value["selector_source_sha256"] = wrong_source[
            "selector_source_sha256"]
        with self.assertRaisesRegex(ValueError, "qta_selection_source_not_frozen"):
            validate_qta_selection_provenance(wrong_source_value,
                                              expected_bindings=wrong_source)

    def test_wrong_sham_inputs_or_selected_groups_fail_recomputation(self):
        wrong_seed = dict(self.expected, sham_seed="changed-seed")
        with self.assertRaisesRegex(ValueError,
                                    "qta_selection_geometry_or_protocol_mismatch"):
            validate_qta_selection_provenance(self.value, expected_bindings=wrong_seed)
        wrong_token = dict(self.expected, opaque_review_token="different_review_token01")
        with self.assertRaisesRegex(ValueError, "qta_selection_sham_token_mismatch"):
            validate_qta_selection_provenance(self.value, expected_bindings=wrong_token)
        changed = copy.deepcopy(self.value)
        changed["sham_group_indices"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "qta_selection_recomputation_mismatch"):
            validate_qta_selection_provenance(changed, expected_bindings=self.expected)
        for field, replacement in (("top_k", True),
                                   ("target_group_indices", [True, 7]),
                                   ("temporal_group_count", True)):
            changed = copy.deepcopy(self.value)
            changed[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_qta_selection_provenance(
                    changed, expected_bindings=self.expected)

    def test_plan_target_sham_and_visual_key_positions_are_bound(self):
        plan = build_qta_nine_branch_plan(
            row_binding_sha256=self.expected["row_binding_sha256"],
            temporal_groups=self.groups,
            target_group_indices=self.value["target_group_indices"],
            sham_group_indices=self.value["sham_group_indices"],
            evidence_group_indices=(0,), control_group_indices=(10,),
            typed_batch_sha256=self.expected["typed_batch_sha256"],
            score_binding_sha256=self.expected["score_binding_sha256"],
            intervention_layers_1_based=FROZEN_SELECTOR_LAYERS_1_BASED,
            logit_bias=math.log(2),
            sham_seed=self.expected["sham_seed"])
        report = validate_qta_selection_provenance(
            self.value, expected_bindings=self.expected, plan=plan)
        self.assertEqual(report["target_group_indices"],
                         plan["selected_groups"]["target"])
        altered_plan = copy.deepcopy(plan)
        altered_plan["branches"][3]["biased_visual_key_positions"] = [999]
        altered_plan["plan_sha256"] = _plan_digest(altered_plan)
        with self.assertRaisesRegex(ValueError,
                                    "qta_selection_plan_invalid"):
            validate_qta_selection_provenance(
                self.value, expected_bindings=self.expected, plan=altered_plan)

    def test_malformed_or_wrongly_bound_plan_is_rejected(self):
        valid = build_qta_nine_branch_plan(
            row_binding_sha256=self.expected["row_binding_sha256"],
            temporal_groups=self.groups,
            target_group_indices=self.value["target_group_indices"],
            sham_group_indices=self.value["sham_group_indices"],
            evidence_group_indices=(0,), control_group_indices=(10,),
            typed_batch_sha256=self.expected["typed_batch_sha256"],
            score_binding_sha256=self.expected["score_binding_sha256"],
            intervention_layers_1_based=FROZEN_SELECTOR_LAYERS_1_BASED,
            logit_bias=math.log(2),
            sham_seed=self.expected["sham_seed"])
        malformed = copy.deepcopy(valid)
        malformed["branches"] = []
        malformed["plan_sha256"] = _plan_digest(malformed)
        with self.assertRaisesRegex(ValueError, "qta_selection_plan_invalid"):
            validate_qta_selection_provenance(
                self.value, expected_bindings=self.expected, plan=malformed)
        wrong_binding = build_qta_nine_branch_plan(
            row_binding_sha256=self.expected["row_binding_sha256"],
            temporal_groups=self.groups,
            target_group_indices=self.value["target_group_indices"],
            sham_group_indices=self.value["sham_group_indices"],
            evidence_group_indices=(0,), control_group_indices=(10,),
            typed_batch_sha256=_h("other batch"),
            score_binding_sha256=self.expected["score_binding_sha256"],
            intervention_layers_1_based=FROZEN_SELECTOR_LAYERS_1_BASED,
            logit_bias=math.log(2),
            sham_seed=self.expected["sham_seed"])
        with self.assertRaisesRegex(ValueError, "qta_selection_plan_binding_mismatch"):
            validate_qta_selection_provenance(
                self.value, expected_bindings=self.expected, plan=wrong_binding)

    def test_raw_prompt_ids_and_attention_tensors_are_not_emitted(self):
        rendered = str(self.value)
        self.assertNotIn("synthetic question", rendered)
        self.assertNotIn("input_ids", rendered.replace("'input_ids_sha256'", ""))
        self.assertNotIn("attention_tensor", rendered)


if __name__ == "__main__":
    unittest.main()
