import json
import tempfile
import unittest
from pathlib import Path

from compare_necessity_smoke_conditions import compare


def raw(values):
    return {
        "n_requested": 3, "n_effective": 3, "failures": [],
        "model": "m", "seed": 1, "num_frames": 16, "frame_size": 1024, "max_pixels": 200704,
        "quant": "4bit", "dtype": "bfloat16", "attn_implementation": "eager",
        "samples_file_sha256": "samples",
        "records": [{"sample_id": str(i), "selective_necessity_margin_drop": v,
                      "baseline_choice_correct": i < 2} for i, v in enumerate(values)],
    }


class Tests(unittest.TestCase):
    def test_paired_delta_excludes_nonprimary(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps(raw([1.0, 3.0, 9.0])))
            b.write_text(json.dumps(raw([0.5, 2.0, 1.0])))
            out = compare(a, b, draws=100, seed=1)
            self.assertEqual(out["n_primary_paired"], 2)
            self.assertAlmostEqual(out["primary_mean_delta_nat"], 0.75)
            self.assertNotIn("records", out)

    def test_rejects_mismatched_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps(raw([1.0, 2.0, 3.0])))
            x = raw([1.0, 2.0, 3.0]); x["records"][0]["sample_id"] = "other"
            b.write_text(json.dumps(x))
            with self.assertRaises(ValueError): compare(a, b)


if __name__ == "__main__":
    unittest.main()
