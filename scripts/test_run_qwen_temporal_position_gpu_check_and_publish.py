import tempfile
from pathlib import Path
import subprocess
import json
import unittest

from run_qwen_temporal_position_gpu_check_and_publish import (
    _output_paths,
    build_run_manifest,
    derive_public_summary,
    _verify_reusable_raw,
    _validate_reusable_raw_path,
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
        "changed_position_id_entries": 0,
        "source_sha256": {"runner.py": "d" * 64},
        "installed_model_source_sha256": "e" * 64,
        "installed_processor_source_sha256": "f" * 64,
        "installed_vision_utils_source_sha256": "0" * 64,
    }


class PublishSummaryTests(unittest.TestCase):
    def test_manifest_binds_fresh_run_window_and_sanitized_command(self):
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15T03:00:00+00:00"
        raw["completed_utc"] = "2026-10-15T03:00:12+00:00"
        raw["argv"] = ["--reuse-raw", "/home/path-sentinel/private.json"]
        raw["local_path"] = "/home/path-sentinel"
        public = derive_public_summary(raw, "1" * 64)
        manifest = build_run_manifest(
            public, Path("results/synthetic.json"), raw,
            summary_sha256="2" * 64, fresh_gpu_run=True,
            manifest_created_utc="2026-10-15T03:01:00Z",
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["execution"]["mode"], "fresh_synthetic_gpu_run")
        self.assertEqual(manifest["runner_started_utc"], raw["started_utc"])
        self.assertEqual(manifest["runner_completed_utc"], raw["completed_utc"])
        self.assertTrue(manifest["execution"]["raw_output_path_redacted"])
        self.assertEqual(manifest["execution"]["allowlisted_arguments"], [
            "--fixture", "synthetic_subsecond", "--synthetic-frame-count", "4",
            "--output", "<redacted>",
        ])
        self.assertNotIn("path-sentinel", json.dumps(manifest))
        self.assertEqual(manifest["artifact_sha256"], "2" * 64)

    def test_manifest_reuse_mode_does_not_claim_gpu_execution(self):
        public = derive_public_summary(valid_raw(), "1" * 64)
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15T03:00:00Z"
        raw["completed_utc"] = "2026-10-15T03:00:12Z"
        manifest = build_run_manifest(
            public, Path("results/reused.json"), raw,
            summary_sha256="2" * 64, fresh_gpu_run=False,
            manifest_created_utc="2026-10-15T03:01:00Z",
        )
        self.assertEqual(manifest["execution"]["mode"], "raw_reuse_publish_only")
        self.assertEqual(manifest["runner_started_utc"], raw["started_utc"])
        self.assertEqual(manifest["runner_completed_utc"], raw["completed_utc"])

    def test_fresh_run_manifest_requires_valid_utc_timestamps(self):
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15 03:00:00"
        raw["completed_utc"] = "2026-10-15T03:00:12Z"
        public = derive_public_summary(raw, "1" * 64)
        with self.assertRaisesRegex(ValueError, "ordered UTC runner"):
            build_run_manifest(
                public, Path("results/invalid.json"), raw,
                summary_sha256="2" * 64, fresh_gpu_run=True,
                manifest_created_utc="2026-10-15T03:01:00Z",
            )

    def test_fresh_manifest_rejects_reversed_runner_window(self):
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15T03:00:12Z"
        raw["completed_utc"] = "2026-10-15T03:00:00Z"
        public = derive_public_summary(raw, "1" * 64)
        with self.assertRaisesRegex(ValueError, "ordered UTC runner"):
            build_run_manifest(
                public, Path("results/reversed.json"), raw,
                summary_sha256="2" * 64, fresh_gpu_run=True,
                manifest_created_utc="2026-10-15T03:01:00Z",
            )

    def test_manifest_rejects_absolute_or_out_of_scope_summary_path(self):
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15T03:00:00Z"
        raw["completed_utc"] = "2026-10-15T03:00:12Z"
        public = derive_public_summary(raw, "1" * 64)
        for path in (Path("/home/path-sentinel/private.json"),
                     Path("results/nested/private.json"), Path("private.json")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative direct-child"):
                build_run_manifest(
                    public, path, raw, summary_sha256="2" * 64,
                    fresh_gpu_run=True, manifest_created_utc="2026-10-15T03:01:00Z",
                )

    def test_manifest_creation_cannot_precede_runner_completion(self):
        raw = valid_raw()
        raw["started_utc"] = "2026-10-15T03:00:00Z"
        raw["completed_utc"] = "2026-10-15T03:00:12Z"
        public = derive_public_summary(raw, "1" * 64)
        with self.assertRaisesRegex(ValueError, "precedes runner completion"):
            build_run_manifest(
                public, Path("results/too-early.json"), raw,
                summary_sha256="2" * 64, fresh_gpu_run=True,
                manifest_created_utc="2026-10-15T03:00:11Z",
            )

    def test_reusable_raw_verifies_ancestor_source_and_current_source_hash(self):
        repo = Path(__file__).resolve().parents[1]
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        import hashlib
        names = {
            "run_qwen_temporal_position_gpu_check.py",
            "qwen_temporal_geometry.py",
            "video_time_mapping.py",
            "run_dataset_video_gpu_smoke.py",
        }
        hashes = {name: hashlib.sha256((repo / "scripts" / name).read_bytes()).hexdigest()
                  for name in names}

        with tempfile.NamedTemporaryFile(
            prefix=".qwen_temporal_position_raw_test_", suffix=".json", dir=repo / "results"
        ) as handle:
            raw = {"source_commit": current, "source_sha256": hashes}
            self.assertEqual(_verify_reusable_raw(raw, Path(handle.name), repo, current), current)
            raw["source_sha256"]["run_qwen_temporal_position_gpu_check.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source files differ"):
                _verify_reusable_raw(raw, Path(handle.name), repo, current)
            raw["source_sha256"] = dict(hashes)
            del raw["source_sha256"]["video_time_mapping.py"]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _verify_reusable_raw(raw, Path(handle.name), repo, current)

    def test_reusable_raw_path_rejects_files_outside_results(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            outside = Path(temp) / ".qwen_temporal_position_raw_external.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "directly under results"):
                _validate_reusable_raw_path(outside, repo)

    def test_reusable_raw_path_rejects_symlink_alias(self):
        repo = Path(__file__).resolve().parents[1]
        results = repo / "results"
        with tempfile.NamedTemporaryFile(
            prefix=".qwen_temporal_position_raw_target_", suffix=".json", dir=results
        ) as target:
            alias = results / ".qwen_temporal_position_raw_alias_test.json"
            try:
                alias.symlink_to(Path(target.name).name)
                with self.assertRaisesRegex(ValueError, "directly under results"):
                    _validate_reusable_raw_path(alias, repo)
            finally:
                alias.unlink(missing_ok=True)

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
        raw["changed_position_id_entries"] = 0
        result = derive_public_summary(raw, "1" * 64)
        self.assertEqual(result["synthetic_video"]["frames"], 32)
        self.assertEqual(result["synthetic_video"]["grid_thw"], [16, 32, 32])

    def test_derives_40_frame_capacity_summary(self):
        raw = valid_raw()
        raw["effective_frames"] = raw["requested_frames"] = 40
        raw["grid_thw"] = [20, 32, 32]
        raw["native_temporal_anchors"] = [15 + int(i * 0.75 * 2) for i in range(20)]
        raw["fractional_temporal_anchors"] = list(raw["native_temporal_anchors"])
        raw["legacy_v517_temporal_anchors"] = [15] * 20
        raw["changed_position_id_entries"] = 0
        result = derive_public_summary(raw, "1" * 64)
        self.assertEqual(result["synthetic_video"]["frames"], 40)
        self.assertEqual(result["synthetic_video"]["grid_thw"], [20, 32, 32])

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
            ("changed_position_id_entries", 1),
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
