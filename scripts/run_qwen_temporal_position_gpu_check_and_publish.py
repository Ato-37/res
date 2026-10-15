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
        forward_diagnostics.append({
            "condition": row["condition"],
            "forward_seconds": seconds,
            "peak_allocated_gib": peak,
        })
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
    return {
        "schema_version": 1,
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
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise ValueError("reusable raw run source hash set is incomplete or unexpected")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("reusable raw run source hash entry is malformed")
        source = repo / "scripts" / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("reusable raw run source files differ from current checkout")
    return source_commit


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
        manifest = {
            "schema_version": 1,
            "artifact": str(summary_path.relative_to(repo)),
            "artifact_sha256": sha256_bytes(summary_bytes),
            "source_commit": source_commit,
            "raw_run_sha256": public["raw_run_sha256"],
            "fixture": "synthetic_subsecond",
            "model_revision": public["model_revision"],
            "model_config_sha256": public["model_config_sha256"],
            "requested_samples": public["sample_counts"]["requested"],
            "effective_samples": public["sample_counts"]["effective"],
            "failures": public["sample_counts"]["failures"],
        }
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
