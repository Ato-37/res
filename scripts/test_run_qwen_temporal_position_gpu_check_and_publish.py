import tempfile
from pathlib import Path
import unittest

from run_qwen_temporal_position_gpu_check_and_publish import (
    _output_paths,
    derive_public_summary,
)


def valid_raw():
    return {
        "fixture": "synthetic_subsecond",
        "failures": 0,
        "effective_samples": 1,
        "requested_samples": 1,
        "effective_forwards": 3,
        "requested_forwards": 3,
        "annotations_used": False,
        "audio_used": False,
        "synthetic_temporal_assertions": {
            "candidate_matches_fractional_reference": True,
            "native_matches_fractional_reference": True,
        },
        "actual_forward_position_ids_verified": True,
        "fixed_vs_legacy_v517_sensitivity": {"top1_equal": True},
        "source_commit": "a" * 40,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "b" * 40,
        "model_config_sha256": "c" * 64,
        "versions": {"transformers": "5.18.0"},
        "cuda_version": "12.8",
        "seed": 1,
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "effective_frames": 4,
        "sampled_fps": 8 / 3,
        "grid_thw": [2, 2, 2],
        "seconds_per_grid": 0.75,
        "native_temporal_anchors": [0, 1],
        "fractional_temporal_anchors": [0, 1],
        "legacy_v517_temporal_anchors": [0, 0],
        "changed_position_id_entries": 0,
        "source_sha256": {"runner.py": "d" * 64},
        "installed_model_source_sha256": "e" * 64,
        "installed_processor_source_sha256": "f" * 64,
        "installed_vision_utils_source_sha256": "0" * 64,
    }


class PublishSummaryTests(unittest.TestCase):
    def test_derives_minimal_synthetic_summary(self):
        result = derive_public_summary(valid_raw(), "1" * 64)
        self.assertEqual(result["fixture"], "deterministic synthetic pixels; no project media or annotations")
        self.assertEqual(result["sample_counts"], {"requested": 1, "effective": 1, "failures": 0})
        self.assertNotIn("prompt", result)
        self.assertNotIn("conditions", result)

    def test_refuses_real_media_fixture(self):
        raw = valid_raw()
        raw["fixture"] = "noaa"
        with self.assertRaisesRegex(ValueError, "only the synthetic"):
            derive_public_summary(raw, "1" * 64)

    def test_refuses_incomplete_or_mismatched_forward_counts(self):
        raw = valid_raw()
        raw["failures"] = 1
        with self.assertRaisesRegex(ValueError, "did not complete"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["effective_forwards"] = 2
        with self.assertRaisesRegex(ValueError, "exactly three"):
            derive_public_summary(raw, "1" * 64)

    def test_output_paths_are_scoped_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "results").mkdir()
            summary, manifest = _output_paths(repo, "results/test-run")
            self.assertEqual(summary, repo / "results/test-run.json")
            self.assertEqual(manifest, repo / "results/test-run_run_manifest.json")
            with self.assertRaisesRegex(ValueError, "directly inside results"):
                _output_paths(repo, "results/subdir/test-run")
            summary.touch()
            with self.assertRaises(FileExistsError):
                _output_paths(repo, "results/test-run")


if __name__ == "__main__":
    unittest.main()
