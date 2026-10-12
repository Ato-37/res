import json
import tempfile
import unittest
from pathlib import Path

from compare_qwen_sequence_length_diagnostics import compare


ROOT = Path(__file__).resolve().parents[1]
LEFT = ROOT / "results/qwen_sequence_length_numerics_gpu_diagnostic_20261012_01.json"
RIGHT = ROOT / "results/qwen_sequence_length_numerics_gpu_diagnostic_20261012_02.json"


class CompareTests(unittest.TestCase):
    def test_published_pair_matches(self):
        result = compare(LEFT, RIGHT)
        self.assertTrue(result["fixed_fields_equal"])
        self.assertEqual(result["right_load_witness"], "verified_immediately_before_model_load")

    def test_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            data = json.loads(LEFT.read_text())
            data["status"] = "failed"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                compare(path, RIGHT)

    def test_rejects_numeric_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            data = json.loads(RIGHT.read_text())
            data["comparisons"][0]["raw_max_abs"] += 1e-12
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                compare(LEFT, path)

    def test_ignores_provenance_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.json"
            data = json.loads(LEFT.read_text())
            data["source_commit"] = "different"
            data["elapsed_seconds"] = 999
            path.write_text(json.dumps(data))
            result = compare(path, LEFT)
            self.assertTrue(result["fixed_fields_equal"])

    def test_requires_fixed_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            data = json.loads(LEFT.read_text())
            data["effective_comparison_pairs"] = 18
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                compare(path, RIGHT)


if __name__ == "__main__":
    unittest.main()
