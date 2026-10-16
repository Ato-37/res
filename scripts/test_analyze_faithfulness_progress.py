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
HEAD_SMOKE = ROOT / "results/head_faithfulness_repro_smoke_20261013_01.json"
MASK_SMOKE = ROOT / "results/necessity_choice_margin_smoke_derived_20261013_01.json"
MASK_BLUR_SMOKE = ROOT / "results/necessity_choice_margin_black_blur_smoke_derived_20261013_01.json"


class Tests(unittest.TestCase):
    def test_snapshot_statuses_and_counts(self):
        out = analyze(HEAD, BLACK, BLUR, NOAA)
        self.assertEqual(out["historical_head_confirmation"]["status"], "closed_practical_threshold_not_met")
        self.assertIn("not generated-rationale", out["historical_head_confirmation"]["claim_scope"])
        self.assertEqual(out["historical_head_confirmation"]["n_effective"], 79)
        self.assertFalse(out["historical_head_confirmation"]["threshold_supported"])
        claim = out["generated_claim_faithfulness"]
        self.assertEqual(claim["status"], "not_evaluated_no_verified_direct_score_artifact")
        self.assertEqual((claim["human_reviewed_records_declared"], claim["scored_records_declared"]), (0, 0))
        self.assertEqual(claim["count_source"], "operator_declared_unverified")
        self.assertFalse(claim["verified_score_artifact_supplied"])
        self.assertIsNone(claim["estimate_nat"])
        self.assertIsNone(claim["confidence_interval_nat"])
        self.assertFalse(out["unresolved_gates"]["verified_generated_claim_score"])
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

    def test_records_smokes_as_supporting_only(self):
        out = analyze(HEAD, BLACK, BLUR, NOAA, HEAD_SMOKE, MASK_SMOKE, MASK_BLUR_SMOKE)
        self.assertEqual(out["engineering_smokes"]["head_intervention"]["status"], "補助診断のみ")
        self.assertEqual(out["engineering_smokes"]["grounding_mask"]["n_effective"], 3)
        self.assertEqual(out["engineering_smokes"]["grounding_mask_black_blur_pair"]["n_effective_each"], 3)

    def test_declared_claim_counts_never_claim_a_verified_score(self):
        out = analyze(HEAD, BLACK, BLUR, NOAA, claim_reviewed_count=3, claim_scored_count=3)
        self.assertEqual(out["generated_claim_faithfulness"]["status"], "not_evaluated_no_verified_direct_score_artifact")
        self.assertIsNone(out["generated_claim_faithfulness"]["estimate_nat"])
        with self.assertRaisesRegex(ValueError, "generated_claim_progress_counts_invalid"):
            analyze(HEAD, BLACK, BLUR, NOAA, claim_reviewed_count=1, claim_scored_count=1)

    def test_rejects_substituted_head_artifact_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "same-count-different-artifact.json"
            x = json.loads(HEAD.read_text()); x["practical_threshold_nat"] = 0.3
            p.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError, "head confirmation artifact identity mismatch"):
                analyze(p, BLACK, BLUR, NOAA)


if __name__ == "__main__":
    unittest.main()
