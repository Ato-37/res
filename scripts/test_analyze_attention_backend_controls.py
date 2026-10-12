import json
import tempfile
import unittest
from pathlib import Path

from analyze_attention_backend_controls import analyze

ROOT = Path(__file__).resolve().parents[1]
BF16 = ROOT / "results/attention_backend_shape_diagnostic_20261012_01.json"
ACC = ROOT / "results/attention_backend_shape_diagnostic_20261012_02.json"
FP32 = ROOT / "results/attention_backend_shape_diagnostic_fp32_20261012_01.json"


class Tests(unittest.TestCase):
    def test_fixed_controls(self):
        result = analyze(BF16, ACC, FP32)
        self.assertEqual(result["record_count"], 3)
        self.assertTrue(result["all_bf16_accumulation_zero"])
        self.assertTrue(result["all_bf16_eager_nonzero"])
        self.assertTrue(result["all_fp32_zero"])

    def test_rejects_changed_seed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            report = json.loads(BF16.read_text())
            report["seed"] = 1
            p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                analyze(p, ACC, FP32)

    def test_rejects_nonfinite_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            report = json.loads(ACC.read_text())
            report["records"][0]["raw_max_abs"] = float("nan")
            p.write_text(json.dumps(report, allow_nan=True))
            with self.assertRaises(ValueError):
                analyze(BF16, p, FP32)


if __name__ == "__main__":
    unittest.main()
