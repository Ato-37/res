"""CPU contract tests for the draft fixed-answer selective-dependence helper."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_prediction_selective_dependence import (  # noqa: E402
    fixed_prediction_intervention_contrast,
    fixed_answer_foil_margin,
    freeze_baseline_answer_and_foil,
    score_bound_six_condition_logits,
    selective_dependence_contrast,
    validate_six_condition_input_binding,
    score_six_condition_logits,
)


class FixedPredictionTests(unittest.TestCase):
    @staticmethod
    def _binding_fixture():
        one = {
            "answer_label_order": ["A", "B", "C", "D", "E"],
            "answer_token_ids": [0, 1, 2, 3, 4],
            "input_ids_sha256": "a" * 64,
            "attention_mask_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "model_revision": "d" * 40,
            "prompt_sha256": "e" * 64,
            "input_length": 584,
            "score_position": 583,
        }
        return {condition: dict(one) for condition in (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation")}

    def test_six_condition_binding_requires_same_ids_prefix_position_and_model(self):
        bindings = self._binding_fixture()
        result = validate_six_condition_input_binding(bindings)
        self.assertEqual(result["status"], "synthetic_binding_contract_only")
        self.assertEqual(result["bound_conditions"], 6)
        self.assertEqual(result["input_length"], 584)
        self.assertEqual(result["score_position"], 583)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(
            result["binding_sha256"],
            validate_six_condition_input_binding(dict(reversed(list(bindings.items()))))[
                "binding_sha256"],
        )

    def test_six_condition_binding_rejects_branch_mismatch(self):
        for key, value in (
            ("answer_token_ids", [0, 1, 2, 3, 5]),
            ("input_ids_sha256", "f" * 64),
            ("attention_mask_sha256", "f" * 64),
            ("tokenizer_sha256", "f" * 64),
            ("model_revision", "other-revision"),
            ("prompt_sha256", "f" * 64),
            ("score_position", 582),
        ):
            bindings = self._binding_fixture()
            bindings["intervention_control_ablation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_six_condition_input_binding(bindings)

    def test_six_condition_binding_rejects_invalid_schema_labels_ids_and_hashes(self):
        valid = self._binding_fixture()
        for alter in (
            lambda data: data.pop("intervention_full"),
            lambda data: data["baseline_full"].update(answer_label_order="ABCDE"),
            lambda data: data["baseline_full"].update(answer_token_ids=[10, 11, 12, 13, 13]),
            lambda data: data["baseline_full"].update(input_ids_sha256="not-a-hash"),
        ):
            bindings = {key: dict(value) for key, value in valid.items()}
            alter(bindings)
            with self.assertRaises(ValueError):
                validate_six_condition_input_binding(bindings)

    @staticmethod
    def _six_logits():
        conditions = (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation")
        rows = ([4., 3., 2., 1., 0.], [2., 3., 2., 1., 0.],
                [3., 2., 2., 1., 0.], [1., 4., 3., 2., 0.],
                [2., 3., 2., 1., 0.], [3., 2., 2., 1., 0.])
        outputs = {}
        for name, row in zip(conditions, rows):
            logits = torch.full((1, 1, 9), -10., dtype=torch.float32)
            logits[0, 0, :5] = torch.tensor(row)
            outputs[name] = {"logits": logits}
        return outputs

    def test_logits_adapter_extracts_last_row_and_keeps_baseline_labels_frozen(self):
        outputs = self._six_logits()
        # One branch exposes full-sequence logits; the others expose the
        # logits_to_keep=1 form used by the Qwen runner.
        full = torch.full((1, 4, 9), -10., dtype=torch.float32)
        full[0, :, :5] = torch.tensor([[50., 0., 0., 0., 0.],
                                       [40., 0., 0., 0., 0.],
                                       [30., 0., 0., 0., 0.],
                                       [4., 3., 2., 1., 0.]])
        outputs["baseline_full"] = type("Output", (), {"logits": full})()
        result = score_six_condition_logits(
            condition_outputs=outputs, answer_token_ids=[0, 1, 2, 3, 4],
            input_lengths={name: 4 for name in outputs}, torch_module=torch)
        self.assertEqual(result["status"], "logits_adapter_contract_only")
        self.assertEqual(result["score_position_by_condition"]["baseline_full"], 3)
        self.assertEqual(result["contrast"]["answer_index"], 0)
        self.assertEqual(result["contrast"]["foil_index"], 1)
        self.assertLess(result["choice_log_probabilities_by_condition"]["baseline_full"][0], 0)
        self.assertEqual(result["contrast"]["answer_index_frozen_from"], "baseline_full")

    def test_logits_adapter_rejects_wrong_shapes_lengths_and_nonfinite_logits(self):
        cases = []
        outputs = self._six_logits()
        wrong_batch = dict(outputs)
        wrong_batch["baseline_full"] = {"logits": torch.zeros(2, 1, 9)}
        cases.append((wrong_batch, {name: 4 for name in outputs}))
        wrong_seq = dict(outputs)
        wrong_seq["baseline_full"] = {"logits": torch.zeros(1, 2, 9)}
        cases.append((wrong_seq, {name: 4 for name in outputs}))
        wrong_length = {name: 4 for name in outputs}
        wrong_length["baseline_full"] = 0
        cases.append((outputs, wrong_length))
        mismatched_lengths = {name: 4 for name in outputs}
        mismatched_lengths["intervention_full"] = 5
        cases.append((outputs, mismatched_lengths))
        nonfinite = dict(outputs)
        bad = outputs["baseline_full"]["logits"].clone()
        bad[0, 0, 0] = float("nan")
        nonfinite["baseline_full"] = {"logits": bad}
        cases.append((nonfinite, {name: 4 for name in outputs}))
        for candidate, lengths in cases:
            with self.subTest(lengths=lengths["baseline_full"]), self.assertRaises(ValueError):
                score_six_condition_logits(
                    condition_outputs=candidate, answer_token_ids=[0, 1, 2, 3, 4],
                    input_lengths=lengths, torch_module=torch)

    def test_logits_adapter_rejects_token_ids_outside_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "out_of_range"):
            score_six_condition_logits(
                condition_outputs=self._six_logits(), answer_token_ids=[0, 1, 2, 3, 9],
                input_lengths={name: 4 for name in self._six_logits()}, torch_module=torch)

    def test_logits_adapter_accepts_all_full_sequence_and_reduced_precision_logits(self):
        conditions = (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation")
        for dtype in (torch.float16, torch.bfloat16):
            outputs = {}
            for name in conditions:
                logits = torch.zeros((1, 4, 9), dtype=dtype)
                logits[0, 0, 0] = 30
                logits[0, -1, :5] = torch.tensor([4., 3., 2., 1., 0.], dtype=dtype)
                outputs[name] = {"logits": logits}
            result = score_six_condition_logits(
                condition_outputs=outputs, answer_token_ids=[0, 1, 2, 3, 4],
                input_lengths={name: 4 for name in conditions}, torch_module=torch)
            self.assertEqual(result["contrast"]["answer_index"], 0)
            self.assertAlmostEqual(
                result["choice_log_probabilities_by_condition"]["baseline_full"][0],
                float(torch.log_softmax(torch.tensor([4., 3., 2., 1., 0., 0., 0., 0., 0.]),
                                        dim=-1)[0]), places=6)

    def test_bound_logits_adapter_derives_ids_and_lengths_from_validated_bindings(self):
        result = score_bound_six_condition_logits(
            branch_bindings=self._binding_fixture(),
            condition_outputs=self._six_logits(), torch_module=torch)
        self.assertEqual(result["status"], "bound_logits_adapter_contract_only")
        self.assertEqual(result["answer_token_ids"], [0, 1, 2, 3, 4])
        self.assertEqual(set(result["score_position_by_condition"].values()), {583})
        self.assertEqual(len(result["input_binding_sha256"]), 64)
        self.assertIn("not authenticated", result["interpretation_limit"])

    def test_bound_logits_adapter_rejects_branch_binding_mismatch_before_scoring(self):
        bindings = self._binding_fixture()
        bindings["intervention_control_ablation"]["answer_token_ids"] = [0, 1, 2, 3, 5]
        with self.assertRaisesRegex(ValueError, "six_condition_scoring_binding_mismatch"):
            score_bound_six_condition_logits(
                branch_bindings=bindings, condition_outputs=self._six_logits(),
                torch_module=torch)

    def test_six_condition_contract_freezes_baseline_full_prediction_and_foil(self):
        scores = {
            "baseline_full": [-.1, -.2, -.3, -.4, -.5],
            "baseline_evidence_ablation": [-.5, -.2, -.3, -.4, -.6],
            "baseline_control_ablation": [-.2, -.3, -.4, -.1, -.5],
            "intervention_full": [-.8, -.3, -.1, -.4, -.5],
            "intervention_evidence_ablation": [-.7, -.15, -.25, -.35, -.45],
            "intervention_control_ablation": [-.2, -.3, -.4, -.1, -.5],
        }
        result = fixed_prediction_intervention_contrast(scores)
        self.assertEqual(result["answer_index_frozen_from"], "baseline_full")
        self.assertEqual(result["foil_index_frozen_from"], "baseline_full")
        self.assertEqual((result["answer_index"], result["foil_index"]), (0, 1))
        # C becomes the intervention-full argmax, but every margin remains A vs B.
        self.assertAlmostEqual(result["margin_by_condition"]["intervention_full"], -.5)
        self.assertAlmostEqual(
            result["delta_f"],
            (result["margin_by_condition"]["intervention_control_ablation"]
             - result["margin_by_condition"]["intervention_evidence_ablation"])
            - (result["margin_by_condition"]["baseline_control_ablation"]
               - result["margin_by_condition"]["baseline_evidence_ablation"]),
        )
        self.assertEqual(result["status"], "synthetic_contract_only")

    def test_six_condition_contract_rejects_missing_extra_or_invalid_branch(self):
        scores = {name: [-.1, -.2, -.3, -.4, -.5] for name in (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation")}
        for altered in (
            {key: value for key, value in scores.items() if key != "intervention_full"},
            {**scores, "sham_full": [-.1, -.2, -.3, -.4, -.5]},
            {**scores, "intervention_evidence_ablation": [-.1, -.2]},
            {**scores, "baseline_full": [-.1, -.2, -.3, -.4, float("nan")]},
            {**scores, "baseline_full": [.001, -.2, -.3, -.4, -.5]},
        ):
            with self.subTest(branches=tuple(altered)):
                with self.assertRaises(ValueError):
                    fixed_prediction_intervention_contrast(altered)

    def test_baseline_answer_and_foil_use_stable_alphabetical_ties(self):
        self.assertEqual(freeze_baseline_answer_and_foil([-.2, -.2, -.3, -.4, -.5]),
                         (0, 1))
        self.assertEqual(freeze_baseline_answer_and_foil([-.5, -.2, -.2, -.4, -.6]),
                         (1, 2))

    def test_margin_reuses_frozen_labels_even_when_condition_argmax_changes(self):
        answer, foil = freeze_baseline_answer_and_foil([-.1, -.2, -.3, -.4, -.5])
        treated = [-.7, -.1, -.3, -.4, -.5]  # B becomes argmax, but A vs B stays fixed.
        self.assertAlmostEqual(fixed_answer_foil_margin(
            treated, answer_index=answer, foil_index=foil), -.6)

    def test_delta_f_uses_evidence_control_interaction_and_full_cancels(self):
        baseline = {"full": 1.0, "evidence_ablation": .4, "control_ablation": .7}
        treatment = {"full": 5.0, "evidence_ablation": 3.8, "control_ablation": 4.4}
        result = selective_dependence_contrast(
            baseline_margins=baseline, intervention_margins=treatment)
        self.assertAlmostEqual(result["baseline_selective_dependence"], .3)
        self.assertAlmostEqual(result["intervention_selective_dependence"], .6)
        self.assertAlmostEqual(result["delta_f"], .3)
        self.assertAlmostEqual(result["delta_f"],
                               (treatment["control_ablation"]
                                - treatment["evidence_ablation"])
                               - (baseline["control_ablation"]
                                  - baseline["evidence_ablation"]))
        offset = 1e6
        shifted = selective_dependence_contrast(
            baseline_margins={key: value + offset for key, value in baseline.items()},
            intervention_margins={key: value + offset for key, value in treatment.items()},
        )
        self.assertAlmostEqual(shifted["delta_f"], result["delta_f"])

    def test_delta_f_is_invariant_to_within_arm_offsets_and_large_full_margins(self):
        baseline = {"full": 1.0, "evidence_ablation": .4, "control_ablation": .7}
        treatment = {"full": 5.0, "evidence_ablation": 3.8, "control_ablation": 4.4}
        expected = selective_dependence_contrast(
            baseline_margins=baseline, intervention_margins=treatment)["delta_f"]
        shifted_baseline = {**baseline, "full": -1e16,
                            "evidence_ablation": baseline["evidence_ablation"] + 11,
                            "control_ablation": baseline["control_ablation"] + 11}
        shifted_treatment = {**treatment, "full": 1e16,
                             "evidence_ablation": treatment["evidence_ablation"] - 23,
                             "control_ablation": treatment["control_ablation"] - 23}
        actual = selective_dependence_contrast(
            baseline_margins=shifted_baseline,
            intervention_margins=shifted_treatment)["delta_f"]
        self.assertAlmostEqual(actual, expected)

    def test_evidence_only_margin_drop_changes_delta_f_with_expected_sign(self):
        baseline = {"full": 1.0, "evidence_ablation": .4, "control_ablation": .7}
        treatment = {"full": 5.0, "evidence_ablation": 3.8, "control_ablation": 4.4}
        original = selective_dependence_contrast(
            baseline_margins=baseline, intervention_margins=treatment)["delta_f"]
        changed_treatment = {**treatment, "evidence_ablation": 3.3}
        changed = selective_dependence_contrast(
            baseline_margins=baseline,
            intervention_margins=changed_treatment)["delta_f"]
        self.assertAlmostEqual(changed, original + .5)

    def test_rejects_nonfinite_missing_or_invalid_inputs(self):
        for values in ([0.0] * 4, [0.0, 0.0, 0.0, 0.0, float("nan")],
                       [0.0, 0.0, 0.0, 0.0, True], [0.0, 0.0, 0.0, 0.0, "0.0"],
                       [0.0, 0.0, 0.0, 0.0, 10**10000], "12345"):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    freeze_baseline_answer_and_foil(values)
        with self.assertRaisesRegex(ValueError, "indices_invalid"):
            fixed_answer_foil_margin([0.0] * 5, answer_index=1, foil_index=1)
        with self.assertRaisesRegex(ValueError, "exact_three_condition"):
            selective_dependence_contrast(
                baseline_margins={"full": 0.0},
                intervention_margins={"full": 0.0})
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            selective_dependence_contrast(
                baseline_margins={"full": 0.0, "evidence_ablation": 0.0,
                                  "control_ablation": float("inf")},
                intervention_margins={"full": 0.0, "evidence_ablation": 0.0,
                                      "control_ablation": 0.0})
        with self.assertRaisesRegex(ValueError, "condition_margin_invalid"):
            selective_dependence_contrast(
                baseline_margins={"full": 1.0, "evidence_ablation": 0.0,
                                  "control_ablation": False},
                intervention_margins={"full": 1.0, "evidence_ablation": 0.0,
                                      "control_ablation": 0.0})
        with self.assertRaisesRegex(ValueError, "arithmetic_nonfinite"):
            selective_dependence_contrast(
                baseline_margins={"full": 1e308, "evidence_ablation": -1e308,
                                  "control_ablation": 0.0},
                intervention_margins={"full": 0.0, "evidence_ablation": 0.0,
                                      "control_ablation": 0.0})
        with self.assertRaisesRegex(ValueError, "condition_margin_invalid"):
            selective_dependence_contrast(
                baseline_margins={"full": 10**10000, "evidence_ablation": 0.0,
                                  "control_ablation": 0.0},
                intervention_margins={"full": 0.0, "evidence_ablation": 0.0,
                                      "control_ablation": 0.0})


if __name__ == "__main__":
    unittest.main()
