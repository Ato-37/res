#!/usr/bin/env python3
"""Summarize a UTC nvidia-smi trace over a verified synthetic-run interval."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics

EXPECTED_SOURCE_FILES = {
    "run_qwen_temporal_position_gpu_check.py",
    "qwen_temporal_geometry.py",
    "video_time_mapping.py",
    "run_dataset_video_gpu_smoke.py",
}
EXPECTED_SOURCE_FILES_V2 = EXPECTED_SOURCE_FILES | {
    "run_qwen_temporal_position_gpu_check_and_publish.py",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=timezone.utc)


def nearest_rank(values: list[float], probability: float) -> float:
    if not values or not 0 < probability <= 1:
        raise ValueError("invalid_quantile_input")
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[index]


def read_trace(raw: bytes) -> list[dict]:
    rows = []
    try:
        text = raw.decode("ascii")
        for fields in csv.reader(text.splitlines(), skipinitialspace=True):
            if len(fields) != 5:
                raise ValueError("trace_row_field_count_invalid")
            timestamp = parse_timestamp(fields[0])
            gpu_util = int(fields[1])
            memory_util = int(fields[2])
            memory_mib = int(fields[3])
            power_w = float(fields[4])
            if (not 0 <= gpu_util <= 100 or not 0 <= memory_util <= 100
                    or memory_mib <= 0 or not math.isfinite(power_w) or power_w <= 0):
                raise ValueError("trace_metric_out_of_range")
            rows.append({
                "timestamp": timestamp,
                "gpu_utilization_percent": gpu_util,
                "memory_controller_utilization_percent": memory_util,
                "memory_used_mib": memory_mib,
                "power_w": power_w,
            })
    except (UnicodeError, csv.Error, ValueError, OverflowError) as error:
        raise ValueError("nvidia_smi_trace_invalid") from error
    if not rows or any(a["timestamp"] >= b["timestamp"] for a, b in zip(rows, rows[1:])):
        raise ValueError("nvidia_smi_trace_empty_or_nonmonotonic")
    return rows


def _exact_counts(value: object, expected: dict[str, int]) -> bool:
    return (isinstance(value, dict) and set(value) == set(expected)
            and all(type(value[key]) is int and value[key] == count
                    for key, count in expected.items()))


def summarize_metrics(rows: list[dict]) -> dict:
    fields = (
        "gpu_utilization_percent", "memory_controller_utilization_percent",
        "memory_used_mib", "power_w",
    )
    metrics = {}
    for field in fields:
        values = [row[field] for row in rows]
        metrics[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "p95_nearest_rank": nearest_rank(values, 0.95),
            "max": max(values),
        }
    return metrics


def verify_source_commit(repo: Path, source_commit: str, source_hashes: dict,
                         schema_version: int = 1) -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                      cwd=repo, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode != 0:
        raise ValueError("source_commit_not_ancestor")
    expected_files = (EXPECTED_SOURCE_FILES if schema_version == 1
                      else EXPECTED_SOURCE_FILES_V2 if schema_version in (2, 3) else set())
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_files):
        raise ValueError("source_hashes_invalid")
    for name, expected in source_hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("source_hashes_invalid")
        try:
            source = subprocess.check_output(
                ["git", "show", f"{source_commit}:scripts/{name}"], cwd=repo,
                stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError("source_commit_file_missing") from error
        if sha256_bytes(source) != expected:
            raise ValueError("source_commit_hash_mismatch")


def summarize(trace: bytes, run_summary: dict, run_manifest: dict,
              summary_raw: bytes, manifest_raw: bytes, summary_path: str) -> dict:
    if not isinstance(run_summary, dict):
        raise ValueError("run_summary_invalid")
    if not isinstance(run_manifest, dict):
        raise ValueError("run_manifest_invalid")
    sample_counts = {"requested": 1, "effective": 1, "failures": 0}
    forward_counts = {"requested": 3, "effective": 3}
    source_commit = run_summary.get("source_commit")
    execution = run_manifest.get("execution")
    gpu = run_summary.get("gpu_resources")
    model_revision = run_summary.get("model_revision")
    model_config_sha256 = run_summary.get("model_config_sha256")
    if (run_summary.get("purpose") != "synthetic_subsecond_temporal_position_regression_check"
            or type(run_summary.get("schema_version")) is not int
            or run_summary.get("schema_version") not in (1, 2, 3)
            or run_summary.get("fixture") != "deterministic synthetic pixels; no project media or annotations"
            or not isinstance(run_summary.get("raw_run_sha256"), str)
            or len(run_summary["raw_run_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in run_summary["raw_run_sha256"])
            or run_manifest.get("raw_run_sha256") != run_summary.get("raw_run_sha256")
            or not isinstance(model_revision, str) or len(model_revision) != 40
            or any(c not in "0123456789abcdef" for c in model_revision)
            or run_manifest.get("model_revision") != model_revision
            or not isinstance(model_config_sha256, str) or len(model_config_sha256) != 64
            or any(c not in "0123456789abcdef" for c in model_config_sha256)
            or run_manifest.get("model_config_sha256") != model_config_sha256
            or run_manifest.get("fixture") != "synthetic_subsecond"
            or not isinstance(source_commit, str) or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)
            or not _exact_counts(run_summary.get("sample_counts"), sample_counts)
            or not _exact_counts(run_summary.get("forward_counts"), forward_counts)
            or type(run_manifest.get("schema_version")) is not int
            or run_manifest.get("schema_version") != 2
            or run_manifest.get("source_commit") != source_commit
            or run_manifest.get("artifact") != summary_path
            or run_manifest.get("artifact_sha256") != sha256_bytes(summary_raw)
            or type(run_manifest.get("requested_samples")) is not int
            or run_manifest.get("requested_samples") != 1
            or type(run_manifest.get("effective_samples")) is not int
            or run_manifest.get("effective_samples") != 1
            or type(run_manifest.get("failures")) is not int or run_manifest["failures"] != 0
            or not isinstance(execution, dict)
            or execution.get("mode") != "fresh_synthetic_gpu_run"
            or execution.get("runner") != "scripts/run_qwen_temporal_position_gpu_check.py"
            or type(execution.get("requested_forwards")) is not int
            or execution.get("requested_forwards") != 3
            or type(execution.get("fixture_frames")) is not int
            or execution.get("fixture_frames") not in (4, 16, 32, 40)
            or run_summary.get("synthetic_video", {}).get("frames") != execution.get("fixture_frames")
            or execution.get("raw_output_path_redacted") is not True
            or not isinstance(gpu, dict)
            or not isinstance(gpu.get("device"), str)
            or not gpu["device"].startswith("NVIDIA ")):
        raise ValueError("source_run_not_expected_synthetic_smoke")
    start_raw, end_raw = (run_manifest.get("runner_started_utc"),
                          run_manifest.get("runner_completed_utc"))
    try:
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("run_interval_invalid") from error
    if (start.tzinfo is None or end.tzinfo is None
            or start.utcoffset() != timezone.utc.utcoffset(start)
            or end.utcoffset() != timezone.utc.utcoffset(end) or end <= start):
        raise ValueError("run_interval_invalid")
    rows = read_trace(trace)
    selected = [row for row in rows if start <= row["timestamp"] <= end]
    if len(selected) < 5:
        raise ValueError("insufficient_samples_in_runner_interval")
    per_forward = gpu.get("forwards")
    expected_conditions = ("native_auto", "explicit_native_noop", "legacy_v517_reference")
    if (not isinstance(per_forward, list) or len(per_forward) != 3
            or any(not isinstance(row, dict) for row in per_forward)
            or tuple(row.get("condition") for row in per_forward) != expected_conditions):
        raise ValueError("forward_interval_provenance_missing")
    condition_windows = []
    for condition_index, row in enumerate(per_forward):
        try:
            forward_start = datetime.fromisoformat(row["started_utc"])
            forward_end = datetime.fromisoformat(row["completed_utc"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("forward_interval_invalid") from error
        if (forward_start.tzinfo is None or forward_end.tzinfo is None
                or forward_start.utcoffset() != timezone.utc.utcoffset(forward_start)
                or forward_end.utcoffset() != timezone.utc.utcoffset(forward_end)
                or forward_start < start or forward_end > end or forward_end <= forward_start):
            raise ValueError("forward_interval_invalid")
        if run_summary.get("schema_version") >= 3:
            try:
                snapshot = datetime.fromisoformat(row["allocator_snapshot_utc"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("allocator_snapshot_invalid") from error
            if (snapshot.tzinfo is None
                    or snapshot.utcoffset() != timezone.utc.utcoffset(snapshot)
                    or not forward_start <= snapshot <= forward_end):
                raise ValueError("allocator_snapshot_invalid")
        if condition_windows and forward_start < condition_windows[-1]["end"]:
            raise ValueError("forward_intervals_overlap_or_reordered")
        includes_final_endpoint = condition_index == len(per_forward) - 1
        condition_rows = [item for item in selected if (
            forward_start <= item["timestamp"]
            and (item["timestamp"] <= forward_end if includes_final_endpoint
                 else item["timestamp"] < forward_end))]
        if not condition_rows:
            raise ValueError("forward_interval_has_no_samples")
        condition_windows.append({
            "condition": row["condition"], "start": forward_start, "end": forward_end,
            "rows": condition_rows,
        })
    forward_rows = [row for row in selected if any(
        window["start"] <= row["timestamp"]
        and (row["timestamp"] <= window["end"] if index == len(condition_windows) - 1
             else row["timestamp"] < window["end"])
        for index, window in enumerate(condition_windows))]
    metrics = {
        "runner_interval": summarize_metrics(selected),
        "forward_union": summarize_metrics(forward_rows),
        "per_condition": {
            window["condition"]: {
                "sample_count": len(window["rows"]),
                "metrics": summarize_metrics(window["rows"]),
            } for window in condition_windows
        },
    }
    return {
        "schema_version": 2,
        "purpose": "synthetic_qwen_gpu_utilization_diagnostic",
        "source_commit": source_commit,
        "source_run": {
            "summary_path": "results/<redacted-by-allowlist>",
            "summary_sha256": sha256_bytes(summary_raw),
            "manifest_sha256": sha256_bytes(manifest_raw),
            "started_utc": start.isoformat(),
            "completed_utc": end.isoformat(),
            "sample_counts": sample_counts,
            "forward_counts": forward_counts,
        },
        "gpu": gpu["device"],
        "trace": {
            "sha256": sha256_bytes(trace),
            "total_samples": len(rows),
            "runner_interval_samples": len(selected),
            "forward_union_samples": len(forward_rows),
            "sampling_interval_ms": 250,
            "timestamp_timezone": "UTC (TZ=UTC set for nvidia-smi)",
        },
        "metrics": metrics,
        "limitations": [
            "Synthetic engineering diagnostic only; no faithfulness evidence.",
            "nvidia-smi utilization is a sampled busy percentage, not measured memory bandwidth in GB/s.",
            "The runner interval includes model loading and preprocessing as well as three forwards.",
            "Forward windows include CUDA synchronization, logits transfer, and finite-value checking.",
            "A single run has no uncertainty interval or inferential comparison.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    trace_path = args.trace.resolve()
    run_path = args.run_summary.resolve()
    run_manifest_path = args.run_manifest.resolve()
    prefix = (repo / args.output_prefix).resolve()
    if (not trace_path.is_relative_to(repo / "results")
            or not run_path.is_relative_to(repo / "results")
            or not run_manifest_path.is_relative_to(repo / "results")
            or not prefix.is_relative_to(repo / "results")
            or prefix.suffix or any(Path(str(prefix) + suffix).exists()
                                    for suffix in (".json", "_run_manifest.json"))):
        parser.error("result inputs and a new extensionless output prefix under results/ required")
    trace_raw = trace_path.read_bytes()
    run_raw = run_path.read_bytes()
    run_manifest_raw = run_manifest_path.read_bytes()
    run = json.loads(run_raw)
    run_manifest = json.loads(run_manifest_raw)
    report = summarize(trace_raw, run, run_manifest, run_raw, run_manifest_raw,
                       run_path.relative_to(repo).as_posix())
    verify_source_commit(repo, report["source_commit"], run.get("source_sha256"),
                         schema_version=run.get("schema_version", 1))
    report["source_run"]["summary_path"] = run_path.relative_to(repo).as_posix()
    output = Path(str(prefix) + ".json")
    manifest = Path(str(prefix) + "_run_manifest.json")
    output_raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_value = {
        "schema_version": 2,
        "artifact": output.relative_to(repo).as_posix(),
        "artifact_sha256": sha256_bytes(output_raw),
        "trace_path": trace_path.relative_to(repo).as_posix(),
        "trace_sha256": sha256_bytes(trace_raw),
        "analyzer_path": "scripts/analyze_nvidia_smi_gpu_trace.py",
        "analyzer_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "source_run_summary_sha256": sha256_bytes(run_raw),
        "source_run_manifest_sha256": sha256_bytes(run_manifest_raw),
        "source_hashes_verified_at_commit": True,
        "source_commit": report["source_commit"],
        "sample_count_in_runner_interval": report["trace"]["runner_interval_samples"],
    }
    manifest_raw = (json.dumps(manifest_value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    created = []
    try:
        for path, data in ((output, output_raw), (manifest, manifest_raw)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            created.append(path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(json.dumps({"summary": output.relative_to(repo).as_posix(),
                      "manifest": manifest.relative_to(repo).as_posix(),
                      "samples": report["trace"]["runner_interval_samples"],
                      "gpu_utilization_runner_interval": report["metrics"]["runner_interval"]["gpu_utilization_percent"],
                      "gpu_utilization_forward_union": report["metrics"]["forward_union"]["gpu_utilization_percent"],
                      "memory_controller_forward_union": report["metrics"]["forward_union"]["memory_controller_utilization_percent"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
