"""Run the synthetic Qwen temporal-position GPU check and publish a safe summary.

The raw run artifact stays local and is removed only after a derived summary
and provenance manifest have been validated. Publication stages exactly those
two generated files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

from run_all_layer_and_publish import git_identity, push_with_rebase, run
from run_thesis_summary_and_publish import git_output, synchronize_clean_branch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def derive_public_summary(raw: dict, raw_sha256: str) -> dict:
    """Validate synthetic-only scope and retain minimum derived diagnostics."""
    if type(raw.get("schema_version")) is not int or raw["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported raw-run schema version")
    if raw.get("fixture") != "synthetic_subsecond":
        raise ValueError("only the synthetic_subsecond fixture may be published")
    if raw.get("effective_frames") not in (4, 16, 32, 40) or raw.get("requested_frames") != raw.get("effective_frames"):
        raise ValueError("only the validated 4-, 16-, 32-, or 40-frame synthetic fixture may be published")
    frame_count = raw["effective_frames"]
    grid = raw.get("grid_thw")
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(value) is not int for value in grid)
            or grid != [frame_count // 2, 32, 32]):
        raise ValueError("temporal processor grid does not match synthetic frame count")
    if (type(raw.get("sampled_fps")) not in (int, float)
            or not math.isclose(raw["sampled_fps"], 8 / 3, rel_tol=0, abs_tol=1e-12)
            or type(raw.get("seconds_per_grid")) not in (int, float)
            or not math.isclose(raw["seconds_per_grid"], 0.75, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("synthetic sampling rate or temporal interval changed")
    native = raw.get("native_temporal_anchors")
    fractional = raw.get("fractional_temporal_anchors")
    legacy = raw.get("legacy_v517_temporal_anchors")
    if any(not isinstance(values, list) or len(values) != grid[0]
           or any(type(value) is not int for value in values)
           for values in (native, fractional, legacy)):
        raise ValueError("temporal anchors do not match processor grid")
    # This pinned prompt/processor fixture has a verified first temporal base
    # of 15; checking only relative spacing would let arbitrary row shifts pass.
    if native[0] != 15:
        raise ValueError("synthetic temporal position base differs from fixed fixture")
    expected_fractional = [15 + int(step * 0.75 * 2) for step in range(grid[0])]
    expected_legacy = [15 + step * 2 * int(0.75) for step in range(grid[0])]
    if fractional != expected_fractional or native != expected_fractional or legacy != expected_legacy:
        raise ValueError("native/fractional/legacy temporal anchors failed fixed-fixture reference")
    # This field counts native-vs-candidate entries, not native-vs-legacy.
    # The pinned 5.18 native path already matches the fractional reference.
    expected_changed = 0
    if type(raw.get("changed_position_id_entries")) is not int or raw["changed_position_id_entries"] != expected_changed:
        raise ValueError("changed temporal-position entry count does not match fixed fixture")
    if (type(raw.get("failures")) is not int or raw["failures"] != 0
            or type(raw.get("effective_samples")) is not int or raw["effective_samples"] != 1
            or type(raw.get("requested_samples")) is not int or raw["requested_samples"] != 1):
        raise ValueError("run did not complete the one requested synthetic sample")
    if (type(raw.get("effective_forwards")) is not int or raw["effective_forwards"] != 3
            or type(raw.get("requested_forwards")) is not int or raw["requested_forwards"] != 3):
        raise ValueError("expected exactly three completed synthetic forwards")
    if raw.get("annotations_used") is not False or raw.get("audio_used") is not False:
        raise ValueError("unexpected annotation or audio input")
    assertions = raw.get("synthetic_temporal_assertions", {})
    if assertions.get("candidate_matches_fractional_reference") is not True:
        raise ValueError("fractional candidate failed its synthetic reference check")
    if assertions.get("native_matches_fractional_reference") is not True:
        raise ValueError("native positions failed their synthetic reference check")
    if raw.get("actual_forward_position_ids_verified") is not True:
        raise ValueError("actual model forward position IDs were not verified")
    sensitivity = raw.get("fixed_vs_legacy_v517_sensitivity")
    if not isinstance(sensitivity, dict):
        raise ValueError("missing derived position-ID sensitivity summary")
    conditions = raw.get("conditions")
    expected_conditions = ("native_auto", "explicit_native_noop", "legacy_v517_reference")
    if not isinstance(conditions, list) or tuple(
        row.get("condition") for row in conditions if isinstance(row, dict)
    ) != expected_conditions:
        raise ValueError("missing or unexpected per-forward resource diagnostics")
    forward_diagnostics = []
    for row in conditions:
        seconds = row.get("forward_seconds")
        peak = row.get("peak_allocated_gib")
        if (row.get("finite_logits") is not True or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0
                or type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0):
            raise ValueError("invalid per-forward timing/VRAM diagnostic")
        forward = {
            "condition": row["condition"],
            "forward_seconds": seconds,
            "peak_allocated_gib": peak,
        }
        if raw.get("schema_version") >= 2:
            memory = row.get("allocator_memory")
            memory_fields = (
                "allocated_bytes", "peak_allocated_bytes", "reserved_bytes",
                "peak_reserved_bytes", "cuda_free_bytes", "cuda_total_bytes",
            )
            if (not isinstance(memory, dict) or set(memory) != set(memory_fields)
                    or any(type(memory.get(key)) is not int or memory[key] <= 0
                           for key in memory_fields if key != "cuda_free_bytes")
                    or type(memory.get("cuda_free_bytes")) is not int
                    or memory["cuda_free_bytes"] < 0
                    or memory["allocated_bytes"] > memory["reserved_bytes"]
                    or memory["peak_allocated_bytes"] < memory["allocated_bytes"]
                    or memory["peak_reserved_bytes"] < memory["reserved_bytes"]
                    or memory["peak_allocated_bytes"] > memory["peak_reserved_bytes"]
                    or memory["peak_allocated_bytes"] > memory["cuda_total_bytes"]
                    or memory["peak_reserved_bytes"] > memory["cuda_total_bytes"]
                    or memory["cuda_free_bytes"] > memory["cuda_total_bytes"]):
                raise ValueError("invalid per-forward CUDA allocator/device memory counters")
            if not math.isclose(
                peak, memory["peak_allocated_bytes"] / 1024**3,
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError("inconsistent per-forward allocated-memory summaries")
            forward["allocator_memory"] = dict(memory)
        start_raw, end_raw = row.get("started_utc"), row.get("completed_utc")
        if start_raw is not None or end_raw is not None or raw.get("schema_version") >= 3:
            try:
                start = datetime.fromisoformat(start_raw)
                end = datetime.fromisoformat(end_raw)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid per-forward UTC interval") from error
            if (start.tzinfo is None or end.tzinfo is None
                    or start.utcoffset() != timezone.utc.utcoffset(start)
                    or end.utcoffset() != timezone.utc.utcoffset(end) or end <= start):
                raise ValueError("invalid per-forward UTC interval")
            forward["started_utc"] = start.isoformat()
            forward["completed_utc"] = end.isoformat()
            if raw.get("schema_version") >= 3:
                try:
                    snapshot = datetime.fromisoformat(row.get("allocator_snapshot_utc"))
                except (TypeError, ValueError) as error:
                    raise ValueError("invalid allocator snapshot UTC timestamp") from error
                if (snapshot.tzinfo is None
                        or snapshot.utcoffset() != timezone.utc.utcoffset(snapshot)
                        or not start <= snapshot <= end):
                    raise ValueError("invalid allocator snapshot UTC timestamp")
                forward["allocator_snapshot_utc"] = snapshot.isoformat()
        forward_diagnostics.append(forward)
    gpu_total = raw.get("gpu_total_gib")
    gpu_free = raw.get("gpu_free_before_gib")
    gpu_name = raw.get("gpu")
    if (not isinstance(gpu_name, str) or not gpu_name.startswith("NVIDIA ")
            or type(gpu_total) not in (int, float) or not math.isfinite(gpu_total)
            or type(gpu_free) not in (int, float) or not math.isfinite(gpu_free)
            or gpu_free <= 0 or gpu_total < gpu_free):
        raise ValueError("invalid GPU preflight diagnostics")
    if any(row["peak_allocated_gib"] > gpu_total for row in forward_diagnostics):
        raise ValueError("per-forward allocated VRAM exceeds GPU total memory")
    host = raw.get("host_memory")
    host_fields = ("total_before_bytes", "available_before_bytes", "total_after_bytes",
                   "available_after_bytes", "process_peak_rss_bytes")
    if (not isinstance(host, dict)
            or any(type(host.get(key)) is not int or host[key] <= 0 for key in host_fields)
            or host["available_before_bytes"] > host["total_before_bytes"]
            or host["available_after_bytes"] > host["total_after_bytes"]
            or host["process_peak_rss_bytes"] > host["total_before_bytes"]
            or host["available_before_bytes"] < 8 * 1024**3):
        raise ValueError("invalid host memory preflight or process RSS diagnostics")
    return {
        "schema_version": raw.get("schema_version", 1),
        "purpose": "synthetic_subsecond_temporal_position_regression_check",
        "fixture": "deterministic synthetic pixels; no project media or annotations",
        "source_commit": raw["source_commit"],
        "raw_run_sha256": raw_sha256,
        "model": raw["model"],
        "model_revision": raw["model_revision"],
        "model_config_sha256": raw["model_config_sha256"],
        "versions": raw["versions"],
        "cuda_version": raw["cuda_version"],
        "seed": raw["seed"],
        "quantization": raw["quantization"],
        "compute_dtype": raw["compute_dtype"],
        "gpu_resources": {
            "device": gpu_name,
            "free_before_gib": gpu_free,
            "total_gib": gpu_total,
            "forwards": forward_diagnostics,
        },
        "host_memory": {
            "total_before_bytes": host["total_before_bytes"],
            "available_before_bytes": host["available_before_bytes"],
            "total_after_bytes": host["total_after_bytes"],
            "available_after_bytes": host["available_after_bytes"],
            "process_peak_rss_bytes": host["process_peak_rss_bytes"],
        },
        "sample_counts": {
            "requested": raw["requested_samples"],
            "effective": raw["effective_samples"],
            "failures": raw["failures"],
        },
        "forward_counts": {
            "requested": raw["requested_forwards"],
            "effective": raw["effective_forwards"],
        },
        "synthetic_video": {
            "frames": raw["effective_frames"],
            "sampled_fps": raw["sampled_fps"],
            "grid_thw": raw["grid_thw"],
            "seconds_per_grid": raw["seconds_per_grid"],
            "native_temporal_anchors": raw["native_temporal_anchors"],
            "fractional_temporal_anchors": raw["fractional_temporal_anchors"],
            "legacy_v517_temporal_anchors": raw["legacy_v517_temporal_anchors"],
            "changed_position_id_entries": raw["changed_position_id_entries"],
        "native_matches_fractional_reference": assertions[
                "native_matches_fractional_reference"
            ],
        },
        "fixed_vs_legacy_v517_sensitivity": sensitivity,
        "source_sha256": raw["source_sha256"],
        "installed_model_source_sha256": raw["installed_model_source_sha256"],
        "installed_processor_source_sha256": raw["installed_processor_source_sha256"],
        "installed_vision_utils_source_sha256": raw["installed_vision_utils_source_sha256"],
        "limitations": [
            "Synthetic engineering regression only; no faithfulness, QA, or semantic-validity evidence.",
            "One deterministic sample has no uncertainty interval or inferential test.",
            "No raw frames, model outputs, dataset labels, or per-item information are published.",
        ],
    }


def _output_paths(repo: Path, prefix: str) -> tuple[Path, Path]:
    base = (repo / prefix).resolve()
    if base.parent != (repo / "results").resolve():
        raise ValueError("--output-prefix must be a filename prefix directly inside results/")
    summary = base.with_suffix(".json")
    manifest = base.with_name(base.name + "_run_manifest.json")
    if summary.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite existing publication artifact")
    return summary, manifest


def _run_raw_gpu_check(repo: Path, raw_path: Path, frame_count: int) -> None:
    command = [
        str(repo / ".venv/bin/python"),
        str(repo / "scripts/run_qwen_temporal_position_gpu_check.py"),
        "--fixture", "synthetic_subsecond", "--synthetic-frame-count", str(frame_count), "--output",
        str(raw_path.relative_to(repo)),
    ]
    run(command, repo)


def _validate_reusable_raw_path(raw_path: Path, repo: Path) -> None:
    results = (repo / "results").resolve()
    if (raw_path.is_symlink() or not raw_path.is_file() or raw_path.resolve().parent != results
            or not raw_path.name.startswith(".qwen_temporal_position_raw_")):
        raise ValueError("reusable raw run must be a private runner artifact directly under results/")


def _verify_reusable_raw(raw: dict, raw_path: Path, repo: Path, current_commit: str) -> str:
    """Verify a preserved synthetic run before deriving/publishing it without rerunning GPU."""
    _validate_reusable_raw_path(raw_path, repo)
    source_commit = raw.get("source_commit")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)):
        raise ValueError("reusable raw run has invalid source commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, current_commit],
        cwd=repo, check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("reusable raw run source is not an ancestor of current HEAD")
    hashes = raw.get("source_sha256")
    expected_names = {
        "run_qwen_temporal_position_gpu_check.py",
        "qwen_temporal_geometry.py",
        "video_time_mapping.py",
        "run_dataset_video_gpu_smoke.py",
    }
    if raw.get("schema_version") >= 2:
        expected_names.add("run_qwen_temporal_position_gpu_check_and_publish.py")
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise ValueError("reusable raw run source hash set is incomplete or unexpected")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("reusable raw run source hash entry is malformed")
        try:
            source_at_run = subprocess.check_output(
                ["git", "show", f"{source_commit}:scripts/{name}"], cwd=repo,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError("reusable raw run source file is absent from source commit") from error
        if hashlib.sha256(source_at_run).hexdigest() != expected:
            raise ValueError("reusable raw run source files differ from recorded source commit")
    return source_commit


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        return None
    return parsed


def build_run_manifest(public: dict, summary_path: Path, raw: dict, *,
                       summary_sha256: str, fresh_gpu_run: bool,
                       manifest_created_utc: str) -> dict:
    """Bind a safe execution record without exposing a local raw-artifact path."""
    if (not isinstance(summary_path, Path) or summary_path.is_absolute()
            or summary_path.parent != Path("results") or summary_path.suffix != ".json"):
        raise ValueError("summary artifact must be a relative direct-child JSON path under results/")
    started, completed = raw.get("started_utc"), raw.get("completed_utc")
    parsed_start, parsed_end = _parse_utc_timestamp(started), _parse_utc_timestamp(completed)
    if fresh_gpu_run and (parsed_start is None or parsed_end is None or parsed_end < parsed_start):
        raise ValueError("fresh GPU run lacks ordered UTC runner timestamps")
    if not fresh_gpu_run:
        if (started is None) != (completed is None):
            raise ValueError("reused raw run has a partial runner timestamp pair")
        if started is not None and (parsed_start is None or parsed_end is None or parsed_end < parsed_start):
            raise ValueError("reused raw run has invalid runner timestamps")
    if _parse_utc_timestamp(manifest_created_utc) is None:
        raise ValueError("manifest creation timestamp must be UTC")
    parsed_created = _parse_utc_timestamp(manifest_created_utc)
    if parsed_end is not None and parsed_created < parsed_end:
        raise ValueError("manifest creation precedes runner completion")
    frame_count = public["synthetic_video"]["frames"]
    return {
        "schema_version": 2,
        "artifact": summary_path.as_posix(),
        "artifact_sha256": summary_sha256,
        "source_commit": public["source_commit"],
        "raw_run_sha256": public["raw_run_sha256"],
        "fixture": "synthetic_subsecond",
        "model_revision": public["model_revision"],
        "model_config_sha256": public["model_config_sha256"],
        "requested_samples": public["sample_counts"]["requested"],
        "effective_samples": public["sample_counts"]["effective"],
        "failures": public["sample_counts"]["failures"],
        "execution": {
            "mode": "fresh_synthetic_gpu_run" if fresh_gpu_run else "raw_reuse_publish_only",
            "runner": "scripts/run_qwen_temporal_position_gpu_check.py",
            "allowlisted_arguments": ["--fixture", "synthetic_subsecond",
                                      "--synthetic-frame-count", str(frame_count),
                                      "--output", "<redacted>"],
            "fixture_frames": frame_count,
            "requested_forwards": 3,
            "raw_output_path_redacted": True,
        },
        "runner_started_utc": started,
        "runner_completed_utc": completed,
        "manifest_created_utc": manifest_created_utc,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        help="unique filename prefix directly under results/ (defaults to UTC timestamp)",
    )
    parser.add_argument("--synthetic-frame-count", type=int, choices=(4, 16, 32, 40), default=4)
    parser.add_argument(
        "--reuse-raw",
        type=Path,
        help="derive/publish an already completed private runner artifact under results/ without another GPU run",
    )
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    branch = git_output(repo, "branch", "--show-current")
    if not branch:
        parser.error("detached HEAD: refusing to run/publish")
    git_identity(repo)
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.strip()
    staged = git_output(repo, "diff", "--cached", "--name-only")
    if tracked or staged:
        parser.error("tracked changes or staged files exist; commit them before running")
    synchronize_clean_branch(repo, branch)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.output_prefix or f"results/qwen_temporal_position_synthetic_{stamp}"
    summary_path, manifest_path = _output_paths(repo, prefix)
    raw_path = ((repo / args.reuse_raw) if args.reuse_raw is not None else
                repo / "results" / f".qwen_temporal_position_raw_{uuid.uuid4().hex}.json")
    current_commit = git_output(repo, "rev-parse", "HEAD")
    if args.reuse_raw is None:
        source_commit = current_commit
        print(f"Running 1 synthetic {args.synthetic_frame_count}-frame sample and 3 Qwen forwards; no real media/QA used.", flush=True)
        _run_raw_gpu_check(repo, raw_path, args.synthetic_frame_count)
    else:
        print("Reusing a verified completed synthetic runner artifact; no model/GPU execution.", flush=True)
    try:
        if args.reuse_raw is not None:
            _validate_reusable_raw_path(raw_path, repo)
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        if args.reuse_raw is None:
            if raw.get("source_commit") != source_commit:
                raise ValueError("source commit changed during the GPU run")
        else:
            source_commit = _verify_reusable_raw(raw, raw_path, repo, current_commit)
        public = derive_public_summary(raw, sha256_bytes(raw_bytes))
        summary_bytes = (json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        manifest = build_run_manifest(
            public, summary_path.relative_to(repo), raw,
            summary_sha256=sha256_bytes(summary_bytes),
            fresh_gpu_run=args.reuse_raw is None,
            manifest_created_utc=datetime.now(timezone.utc).isoformat(),
        )
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        summary_path.write_bytes(summary_bytes)
        manifest_path.write_bytes(manifest_bytes)
        summary_path.chmod(0o644)
        manifest_path.chmod(0o644)
    except Exception:
        print(f"Raw synthetic run artifact preserved locally at {raw_path}", file=sys.stderr)
        raise
    finally:
        if raw_path.exists() and summary_path.exists() and manifest_path.exists():
            raw_path.unlink()

    relative = [str(summary_path.relative_to(repo)), str(manifest_path.relative_to(repo))]
    run(["git", "add", "-f", *relative], repo)
    staged_paths = git_output(repo, "diff", "--cached", "--name-only").splitlines()
    if set(staged_paths) != set(relative):
        raise RuntimeError(f"refusing unexpected staged files: {staged_paths}")
    run(["git", "diff", "--cached", "--check", "--", *relative], repo)
    run(["git", "commit", "--only", "-m", "Publish synthetic Qwen timing regression", *relative], repo)
    push_with_rebase(repo, branch)
    run(["git", "fetch", "origin", branch], repo)
    head = git_output(repo, "rev-parse", "HEAD")
    remote = git_output(repo, "rev-parse", f"origin/{branch}")
    if head != remote:
        raise RuntimeError("published result commit was not verified on origin")
    print(f"Published and verified {relative[0]} (commit {head})", flush=True)


if __name__ == "__main__":
    main()
