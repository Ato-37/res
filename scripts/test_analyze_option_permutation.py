import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_option_permutation as aop


def rec(gold, base, shifts):
    return {"gold": gold, "baseline_choice_logprobs": base, "identity_prompt_matches": True,
            "shifts": [{"shift": i + 1, "choice_logprobs": s} for i, s in enumerate(shifts)]}


class PermTests(unittest.TestCase):
    def test_summary_on_synthetic_receipt(self):
        rng = np.random.default_rng(1)
        records = []
        for _ in range(60):
            gold = int(rng.integers(0, 5))
            wrong = np.log(np.full(5, 0.05)); wrong[(gold + 1) % 5] = np.log(0.8)
            right = np.log(np.full(5, 0.05)); right[gold] = np.log(0.8)
            records.append(rec(gold, wrong.tolist(), [right.tolist()] * 4))  # original wrong, four shifts right -> ensemble right
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps({"records": records}))
            out = Path(d) / "o.json"
            self.assertEqual(aop.main(["--receipt", str(p), "--output", str(out)]), 0)
            s = json.loads(out.read_text())
        self.assertEqual(s["analysed"], 60)
        self.assertEqual(s["original_order_accuracy"], 0.0)
        self.assertEqual(s["P5_primary"]["accuracy"], 1.0)
        self.assertTrue(s["P5_primary"]["supported"])
        self.assertEqual(s["single_shift_accuracy"], [1.0] * 4)

    def test_shift_mapping_roundtrip(self):
        # new position p shows original option (p+j)%5, so original option o sits at (o-j)%5
        for j in range(1, 5):
            for o in range(5):
                p = (o - j) % 5
                self.assertEqual((p + j) % 5, o)


if __name__ == "__main__":
    unittest.main()
