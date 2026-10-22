from __future__ import annotations

import json
import unittest

from run_qta_synthetic_oracle_claim_gpu_v1 import (
    _answer_margin, _canonical, _stratified_summary,
)


class Scalar:
    def __init__(self, value: float):
        self.value = value

    def item(self) -> float:
        return self.value


class SyntheticOracleGpuRunnerTests(unittest.TestCase):
    def test_answer_margin_is_signed_to_supported_label(self):
        log_probs = [Scalar(-0.25), Scalar(-1.5)]
        self.assertAlmostEqual(_answer_margin(
            log_probs, a_id=0, b_id=1, supported_label="A"), 1.25)
        self.assertAlmostEqual(_answer_margin(
            log_probs, a_id=0, b_id=1, supported_label="B"), -1.25)

    def test_summary_keys_are_json_serializable(self):
        summary = {"gate|18": {"n": 1, "primary_scene_contrast": 0.125}}
        self.assertEqual(json.loads(_canonical(summary)), summary)

    def test_stratified_summary_zero_variance_has_no_invented_interval(self):
        cells = {(family, onset): [0.25] * 4
                 for family in ("gate", "rotation", "expansion")
                 for onset in (4, 10, 18, 24)}
        result = _stratified_summary(cells)
        self.assertEqual(result["estimate"], 0.25)
        self.assertTrue(result["zero_variance"])
        self.assertIsNone(result["ci95"])
        self.assertIsNone(result["p_two_sided"])

    def test_stratified_summary_uses_all_fixed_cells(self):
        cells = {(family, onset): [0.0, 0.1, 0.2, 0.3]
                 for family in ("gate", "rotation", "expansion")
                 for onset in (4, 10, 18, 24)}
        result = _stratified_summary(cells)
        self.assertAlmostEqual(result["estimate"], 0.15)
        self.assertFalse(result["zero_variance"])
        self.assertLess(result["ci95"][0], result["estimate"])
        self.assertGreater(result["ci95"][1], result["estimate"])


if __name__ == "__main__":
    unittest.main()
