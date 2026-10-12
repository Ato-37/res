import json
import tempfile
import unittest
from pathlib import Path
from analyze_qwen_attention_control import analyze

ROOT = Path(__file__).resolve().parents[1]
EAGER = ROOT / "results/qwen_sequence_length_numerics_gpu_diagnostic_20261012_03.json"
SDPA = ROOT / "results/qwen_sequence_length_numerics_gpu_diagnostic_sdpa_20261012_01.json"

class Tests(unittest.TestCase):
    def test_fixed_control(self):
        out = analyze(EAGER, SDPA)
        self.assertTrue(out["fixed_input_equal"])
        self.assertEqual(out["comparison_count"], 19)
        self.assertEqual(out["contrasts"][5]["sdpa_raw_max_abs"], 0.0)
    def test_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"; x = json.loads(SDPA.read_text()); x["comparisons"][0]["right"] = "bad"; p.write_text(json.dumps(x))
            with self.assertRaises(ValueError): analyze(EAGER, p)
    def test_rejects_wrong_attention_label(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"; x = json.loads(SDPA.read_text()); x["attention_implementation"] = "eager"; p.write_text(json.dumps(x))
            with self.assertRaises(ValueError): analyze(EAGER, p)

if __name__ == "__main__": unittest.main()
