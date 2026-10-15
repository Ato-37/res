"""Tests for provenance-bound repeatability summary analysis."""
from pathlib import Path
import unittest
import hashlib

from analyze_qwen_temporal_position_repeatability import analyze, _verify_committed_artifact


REPO = Path(__file__).resolve().parents[1]
PREFIX = "qwen_temporal_position_synthetic_40frame_repeat_20261015_"


class RepeatabilityAnalysisTests(unittest.TestCase):
    def test_reproduces_three_published_processes(self):
        result = analyze(REPO)
        self.assertEqual(result["run_count"], 3)
        self.assertEqual([row["run_id"] for row in result["runs"]], ["01", "02", "03"])
        self.assertEqual(result["conditions"]["native_auto"]["peak_allocated_gib"]["range"], 0.0)
        self.assertAlmostEqual(
            result["conditions"]["native_auto"]["forward_seconds"]["median"],
            7.383296012, places=5,
        )
        self.assertEqual(result["effective_samples_per_run"], 1)
        self.assertEqual(
            result["analyzer_sha256"],
            hashlib.sha256((REPO / "scripts/analyze_qwen_temporal_position_repeatability.py").read_bytes()).hexdigest(),
        )

    def test_rejects_paired_summary_and_manifest_tampering_against_git(self):
        summary_path = f"results/{PREFIX}01.json"
        manifest_path = f"results/{PREFIX}01_run_manifest.json"
        summary_bytes = (REPO / summary_path).read_bytes()
        manifest_bytes = (REPO / manifest_path).read_bytes()
        self.assertNotEqual(summary_bytes + b" ", summary_bytes)
        self.assertNotEqual(manifest_bytes + b" ", manifest_bytes)
        with self.assertRaisesRegex(ValueError, "differs from committed Git blob"):
            _verify_committed_artifact(REPO, summary_path, summary_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "differs from committed Git blob"):
            _verify_committed_artifact(REPO, manifest_path, manifest_bytes + b" ")

    def test_rejects_nonprespecified_runs(self):
        with self.assertRaisesRegex(ValueError, "exactly repeat runs"):
            analyze(REPO, ("01", "02", "04"))


if __name__ == "__main__":
    unittest.main()
