import unittest
import hashlib
import json
import subprocess
from pathlib import Path

from analyze_nvidia_smi_gpu_trace import (
    EXPECTED_SOURCE_FILES, EXPECTED_SOURCE_FILES_V2, read_trace, sha256_bytes,
    summarize, verify_source_commit,
)


def trace_rows():
    return "\n".join(
        f"2026/10/15 03:00:0{i}.000, {i*10}, {i*5}, {1000+i}, {20+i}.5"
        for i in range(1, 8)
    ).encode("ascii")


def source_artifacts():
    summary = {
        "schema_version": 1,
        "purpose": "synthetic_subsecond_temporal_position_regression_check",
        "fixture": "deterministic synthetic pixels; no project media or annotations",
        "source_commit": "a" * 40,
        "raw_run_sha256": "b" * 64,
        "model_revision": "c" * 40,
        "model_config_sha256": "d" * 64,
        "source_sha256": {name: "e" * 64 for name in EXPECTED_SOURCE_FILES},
        "synthetic_video": {"frames": 40},
        "sample_counts": {"requested": 1, "effective": 1, "failures": 0},
        "forward_counts": {"requested": 3, "effective": 3},
        "gpu_resources": {
            "device": "NVIDIA GeForce RTX 5070 Ti",
            "forwards": [
                {"condition": "native_auto", "started_utc": "2026-10-15T03:00:02.000+00:00",
                 "completed_utc": "2026-10-15T03:00:03.000+00:00"},
                {"condition": "explicit_native_noop", "started_utc": "2026-10-15T03:00:04.000+00:00",
                 "completed_utc": "2026-10-15T03:00:05.000+00:00"},
                {"condition": "legacy_v517_reference", "started_utc": "2026-10-15T03:00:06.000+00:00",
                 "completed_utc": "2026-10-15T03:00:06.500+00:00"},
            ],
        },
    }
    summary_raw = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema_version": 2,
        "fixture": "synthetic_subsecond",
        "source_commit": "a" * 40,
        "raw_run_sha256": "b" * 64,
        "model_revision": "c" * 40,
        "model_config_sha256": "d" * 64,
        "artifact": "results/source.json",
        "artifact_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "requested_samples": 1,
        "effective_samples": 1,
        "failures": 0,
        "runner_started_utc": "2026-10-15T03:00:02+00:00",
        "runner_completed_utc": "2026-10-15T03:00:06.500+00:00",
        "execution": {
            "mode": "fresh_synthetic_gpu_run",
            "runner": "scripts/run_qwen_temporal_position_gpu_check.py",
            "requested_forwards": 3,
            "fixture_frames": 40,
            "raw_output_path_redacted": True,
        },
    }
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    return summary, manifest, summary_raw, manifest_raw


class NvidiaSmiTraceTests(unittest.TestCase):
    def test_filters_exact_utc_runner_interval_and_nearest_rank_metrics(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        report = summarize(trace_rows(), run, manifest, summary_raw,
                           manifest_raw, "results/source.json")
        self.assertEqual(report["trace"]["total_samples"], 7)
        self.assertEqual(report["trace"]["runner_interval_samples"], 5)
        self.assertEqual(report["metrics"]["runner_interval"]["gpu_utilization_percent"], {
            "min": 20, "median": 40, "p95_nearest_rank": 60, "max": 60,
        })
        self.assertEqual(report["metrics"]["forward_union"]["memory_controller_utilization_percent"]["max"], 30)
        self.assertEqual(report["metrics"]["per_condition"]["native_auto"]["sample_count"], 1)
        self.assertEqual(report["metrics"]["per_condition"]["legacy_v517_reference"]["sample_count"], 1)

    def test_rejects_invalid_metrics_and_nonmonotonic_trace(self):
        for raw in (
            b"2026/10/15 03:00:01.000, 101, 0, 1, 1.0\n",
            b"2026/10/15 03:00:01.000, 1, 1, 0, 1.0\n",
            b"2026/10/15 03:00:02.000, 1, 1, 10, 2.0\n2026/10/15 03:00:01.000, 1, 1, 10, 2.0\n",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_trace(raw)

    def test_rejects_sparse_trace_inside_run_window(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        manifest["runner_started_utc"] = "2026-10-15T03:00:03+00:00"
        manifest["runner_completed_utc"] = "2026-10-15T03:00:04+00:00"
        with self.assertRaisesRegex(ValueError, "insufficient_samples"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")

    def test_refuses_non_synthetic_or_incomplete_source_run(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        run["purpose"] = "real_video_faithfulness"
        manifest["fixture"] = "noaa"
        with self.assertRaisesRegex(ValueError, "source_run_not_expected"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")

    def test_rejects_unbound_manifest_and_non_utc_interval(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        manifest["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_run_not_expected"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        manifest["runner_started_utc"] = "2026-10-15T12:00:02+09:00"
        manifest["runner_completed_utc"] = "2026-10-15T12:00:06+09:00"
        with self.assertRaisesRegex(ValueError, "run_interval_invalid"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")

    def test_rejects_missing_forward_sample_and_naive_forward_timestamps(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        run["gpu_resources"]["forwards"][2]["started_utc"] = "2026-10-15T03:00:05.200+00:00"
        run["gpu_resources"]["forwards"][2]["completed_utc"] = "2026-10-15T03:00:05.800+00:00"
        with self.assertRaisesRegex(ValueError, "forward_interval_has_no_samples"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        run["gpu_resources"]["forwards"][0]["started_utc"] = "2026-10-15T03:00:02.000"
        with self.assertRaisesRegex(ValueError, "forward_interval_invalid"):
            summarize(trace_rows(), run, manifest, summary_raw,
                      manifest_raw, "results/source.json")

    def test_adjacent_forward_windows_assign_shared_timestamp_once(self):
        run, manifest, summary_raw, manifest_raw = source_artifacts()
        run["gpu_resources"]["forwards"][0]["completed_utc"] = "2026-10-15T03:00:04.000+00:00"
        run["gpu_resources"]["forwards"][1]["started_utc"] = "2026-10-15T03:00:04.000+00:00"
        run["gpu_resources"]["forwards"][1]["completed_utc"] = "2026-10-15T03:00:05.500+00:00"
        report = summarize(trace_rows(), run, manifest, summary_raw,
                           manifest_raw, "results/source.json")
        per_condition = report["metrics"]["per_condition"]
        self.assertEqual(per_condition["native_auto"]["sample_count"], 2)
        self.assertEqual(per_condition["explicit_native_noop"]["sample_count"], 2)
        self.assertEqual(report["trace"]["forward_union_samples"], 5)

    def test_verifies_exact_source_set_against_git_ancestor_blobs(self):
        repo = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        hashes = {
            name: sha256_bytes(subprocess.check_output(
                ["git", "show", f"{commit}:scripts/{name}"], cwd=repo))
            for name in EXPECTED_SOURCE_FILES
        }
        verify_source_commit(repo, commit, hashes)
        with self.assertRaisesRegex(ValueError, "source_hashes_invalid"):
            verify_source_commit(repo, commit, {next(iter(hashes)): next(iter(hashes.values()))})
        changed = dict(hashes)
        changed[next(iter(changed))] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_commit_hash_mismatch"):
            verify_source_commit(repo, commit, changed)

    def test_verifies_v2_source_set_including_publisher(self):
        repo = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        hashes = {
            name: sha256_bytes(subprocess.check_output(
                ["git", "show", f"{commit}:scripts/{name}"], cwd=repo))
            for name in EXPECTED_SOURCE_FILES_V2
        }
        verify_source_commit(repo, commit, hashes, schema_version=2)
        with self.assertRaisesRegex(ValueError, "source_hashes_invalid"):
            verify_source_commit(repo, commit, hashes, schema_version=3)


if __name__ == "__main__":
    unittest.main()
