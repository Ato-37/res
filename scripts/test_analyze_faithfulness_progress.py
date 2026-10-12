import json
import tempfile
import unittest
from pathlib import Path

from analyze_faithfulness_progress import analyze

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "results/head_faithfulness_confirm_20261009.json"
BLACK = ROOT / "results/necessity_choice_margin_replication_heldout.json"
BLUR = ROOT / "results/necessity_choice_margin_replication_blur_heldout.json"
NOAA = ROOT / "results/noaa_qa_gate_20261012T053359Z.json"


class Tests(unittest.TestCase):
    def test_snapshot_statuses_and_counts(self):
        out = analyze(HEAD, BLACK, BLUR, NOAA)
        self.assertEqual(out["direct_faithfulness"]["status"], "直接評価済み")
        self.assertEqual(out["direct_faithfulness"]["n_effective"], 79)
        self.assertFalse(out["direct_faithfulness"]["threshold_supported"])
        self.assertEqual(out["supporting_diagnostics"]["status"], "補助診断のみ")
        self.assertEqual(out["unresolved_gates"]["noaa_human_attestation"]["status"], "no_go")

    def test_rejects_changed_noaa_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            x = json.loads(NOAA.read_text()); x["decision"] = "go"
            p.write_text(json.dumps(x))
            with self.assertRaises(ValueError): analyze(HEAD, BLACK, BLUR, p)

    def test_rejects_incomplete_head(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            x = json.loads(HEAD.read_text()); x["n_effective"] = 78
            p.write_text(json.dumps(x))
            with self.assertRaises(ValueError): analyze(p, BLACK, BLUR, NOAA)


if __name__ == "__main__":
    unittest.main()
