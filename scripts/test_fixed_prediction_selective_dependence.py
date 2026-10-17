"""CPU contract tests for the draft fixed-answer selective-dependence helper."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_prediction_selective_dependence import (  # noqa: E402
    fixed_answer_foil_margin,
    freeze_baseline_answer_and_foil,
    selective_dependence_contrast,
)


class FixedPredictionTests(unittest.TestCase):
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

    def test_rejects_nonfinite_missing_or_invalid_inputs(self):
        for values in ([0.0] * 4, [0.0, 0.0, 0.0, 0.0, float("nan")],
                       [0.0, 0.0, 0.0, 0.0, True], [0.0, 0.0, 0.0, 0.0, "0.0"],
                       "12345"):
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


if __name__ == "__main__":
    unittest.main()
