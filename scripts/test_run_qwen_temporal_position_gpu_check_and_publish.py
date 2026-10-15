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
        "conditions": [
            {"condition": name, "finite_logits": True, "forward_seconds": 2.0,
             "peak_allocated_gib": 8.0}
            for name in ("native_auto", "explicit_native_noop", "legacy_v517_reference")
        ],
        "gpu": "NVIDIA GeForce RTX 5070 Ti",
        "gpu_free_before_gib": 14.0,
        "gpu_total_gib": 16.0,
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
        "requested_frames": 4,
        "sampled_fps": 8 / 3,
        "grid_thw": [2, 32, 32],
        "seconds_per_grid": 0.75,
        "native_temporal_anchors": [15, 16],
        "fractional_temporal_anchors": [15, 16],
        "legacy_v517_temporal_anchors": [15, 15],
        "changed_position_id_entries": 256,
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
        self.assertEqual(result["gpu_resources"]["device"], "NVIDIA GeForce RTX 5070 Ti")
        self.assertEqual(len(result["gpu_resources"]["forwards"]), 3)

    def test_derives_32_frame_capacity_summary(self):
        raw = valid_raw()
        raw["effective_frames"] = raw["requested_frames"] = 32
        raw["grid_thw"] = [16, 32, 32]
        raw["native_temporal_anchors"] = [15 + int(i * 0.75 * 2) for i in range(16)]
        raw["fractional_temporal_anchors"] = list(raw["native_temporal_anchors"])
        raw["legacy_v517_temporal_anchors"] = [15] * 16
        raw["changed_position_id_entries"] = 15 * 16 * 16
        result = derive_public_summary(raw, "1" * 64)
        self.assertEqual(result["synthetic_video"]["frames"], 32)
        self.assertEqual(result["synthetic_video"]["grid_thw"], [16, 32, 32])

    def test_refuses_real_media_fixture(self):
        raw = valid_raw()
        raw["fixture"] = "noaa"
        with self.assertRaisesRegex(ValueError, "only the synthetic"):
            derive_public_summary(raw, "1" * 64)

    def test_refuses_frame_grid_or_anchor_mismatch(self):
        raw = valid_raw()
        raw["grid_thw"] = [8, 32, 32]
        with self.assertRaisesRegex(ValueError, "grid does not match"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["fractional_temporal_anchors"] = [0]
        with self.assertRaisesRegex(ValueError, "anchors"):
            derive_public_summary(raw, "1" * 64)

    def test_refuses_incomplete_or_mismatched_forward_counts(self):
        raw = valid_raw()
        raw["failures"] = 1
        with self.assertRaisesRegex(ValueError, "did not complete"):
            derive_public_summary(raw, "1" * 64)

    def test_refuses_invalid_resource_diagnostics(self):
        raw = valid_raw()
        raw["conditions"][1]["peak_allocated_gib"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid per-forward"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["gpu_free_before_gib"] = 17
        with self.assertRaisesRegex(ValueError, "invalid GPU preflight"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["conditions"][0]["forward_seconds"] = True
        with self.assertRaisesRegex(ValueError, "invalid per-forward"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["conditions"][0]["peak_allocated_gib"] = 17
        with self.assertRaisesRegex(ValueError, "exceeds GPU total"):
            derive_public_summary(raw, "1" * 64)
        raw = valid_raw()
        raw["effective_forwards"] = 2
        with self.assertRaisesRegex(ValueError, "exactly three"):
            derive_public_summary(raw, "1" * 64)

    def test_refuses_inconsistent_geometry_timing_anchors_and_boolean_counts(self):
        mutations = (
            ("grid_thw", [2, 999, 999]),
            ("sampled_fps", 24.0),
            ("seconds_per_grid", 9.5),
            ("native_temporal_anchors", [999, 999]),
            ("legacy_v517_temporal_anchors", [15, 16]),
            ("native_temporal_anchors", [999, 1000]),
            ("changed_position_id_entries", 0),
            ("requested_samples", True),
            ("failures", False),
            ("actual_forward_position_ids_verified", 1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                raw = valid_raw()
                raw[field] = value
                with self.assertRaises(ValueError):
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
