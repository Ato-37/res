from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qta_fixed_prediction_key_ablation import (  # noqa: E402
    _mask_representable_bias, _resolve_inputs, compute_delta_f, select_group_control,
)


class FixedPredictionKeyAblationTests(unittest.TestCase):
    def test_resolves_only_existing_inputs_under_private_root(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            private = repo / "data/private_nextgqa"
            private.mkdir(parents=True)
            names = ("selection.json", "lock.json", "pairs.json", "packet.json", "qa.json")
            for name in names:
                (private / name).write_text("{}", encoding="utf-8")
            inputs = _resolve_inputs(repo, *(f"data/private_nextgqa/{name}" for name in names[:3]),
                                     [f"data/private_nextgqa/{names[3]}"],
                                     f"data/private_nextgqa/{names[4]}")
            self.assertEqual(inputs["selection"], "data/private_nextgqa/selection.json")
            self.assertEqual(inputs["packets"], ["data/private_nextgqa/packet.json"])

    def test_rejects_input_outside_private_root(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            private = repo / "data/private_nextgqa"
            private.mkdir(parents=True)
            public = repo / "data/public.json"
            public.parent.mkdir(parents=True, exist_ok=True)
            public.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "private_input_path_invalid"):
                _resolve_inputs(repo, "data/public.json", "data/public.json",
                    "data/public.json", ["data/public.json"], "data/public.json")

    def test_nearest_disjoint_group_control_ties_to_earlier(self):
        self.assertEqual(select_group_control((4, 5, 6, 7), 16), (0, 1, 2, 3))
        self.assertEqual(select_group_control((0, 1, 2, 3), 16), (4, 5, 6, 7))
        self.assertEqual(select_group_control((8, 9, 10, 11), 16), (4, 5, 6, 7))

    def test_rejects_bad_evidence_group_sets(self):
        for groups in ((), (1, 1), (2, 4), (-1, 0), (14, 15, 16)):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                select_group_control(groups, 16)

    def test_delta_f_uses_same_frozen_prediction_and_foil(self):
        branches = {
            "baseline_evidence_ablation": [0.1, 0.0, -1, -2, -3],
            "baseline_control_ablation": [0.3, -0.1, -1, -2, -3],
            "qta_evidence_ablation": [0.0, 0.0, -1, -2, -3],
            "qta_control_ablation": [0.5, -0.2, -1, -2, -3],
        }
        result = compute_delta_f(branches, 0, 1)
        self.assertAlmostEqual(result["f_baseline_nats"], 0.3)
        self.assertAlmostEqual(result["f_qta_nats"], 0.7)
        self.assertAlmostEqual(result["delta_f_qta_minus_baseline_nats"], 0.4)

    def test_rejects_wrong_branches_and_fixed_labels(self):
        branches = {key: [0, -1, -2, -3, -4] for key in (
            "baseline_evidence_ablation", "baseline_control_ablation",
            "qta_evidence_ablation", "qta_control_ablation")}
        with self.assertRaisesRegex(ValueError, "branch_score_set_invalid"):
            compute_delta_f({**branches, "extra": [0]*5}, 0, 1)
        with self.assertRaisesRegex(ValueError, "branch_score_set_invalid"):
            compute_delta_f(branches, 0, 0)

    def test_mask_bias_check_uses_bfloat16_representable_value(self):
        exact = 0.6931471805599453
        rounded = _mask_representable_bias(torch, torch.bfloat16, exact)
        self.assertNotEqual(rounded, exact)
        self.assertEqual(rounded, float(torch.tensor(exact, dtype=torch.bfloat16).item()))


if __name__ == "__main__":
    unittest.main()
