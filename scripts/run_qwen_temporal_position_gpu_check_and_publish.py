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
    if raw.get("failures") != 0 or raw.get("effective_samples") != 1:
        raise ValueError("run did not complete the one requested synthetic sample")
    if raw.get("effective_forwards") != 3 or raw.get("requested_forwards") != 3:
        raise ValueError("expected exactly three completed synthetic forwards")
    if raw.get("annotations_used") is not False or raw.get("audio_used") is not False:
        raise ValueError("unexpected annotation or audio input")
    assertions = raw.get("synthetic_temporal_assertions", {})
    if assertions.get("candidate_matches_fractional_reference") is not True:
        raise ValueError("fractional candidate failed its synthetic reference check")
    if not raw.get("actual_forward_position_ids_verified"):
        raise ValueError("actual model forward position IDs were not verified")
    sensitivity = raw.get("fixed_vs_legacy_v517_sensitivity")
    if not isinstance(sensitivity, dict):
        raise ValueError("missing derived position-ID sensitivity summary")
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


def _run_raw_gpu_check(repo: Path, raw_path: Path) -> None:
    command = [
        str(repo / ".venv/bin/python"),
        str(repo / "scripts/run_qwen_temporal_position_gpu_check.py"),
        "--fixture", "synthetic_subsecond", "--output",
        str(raw_path.relative_to(repo)),
    ]
    run(command, repo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        help="unique filename prefix directly under results/ (defaults to UTC timestamp)",
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
    raw_path = repo / "results" / f".qwen_temporal_position_raw_{uuid.uuid4().hex}.json"
    source_commit = git_output(repo, "rev-parse", "HEAD")
    print("Running 1 synthetic sample and 3 Qwen forwards; no real media/QA used.", flush=True)
    _run_raw_gpu_check(repo, raw_path)
    try:
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        if raw.get("source_commit") != source_commit:
            raise ValueError("source commit changed during the GPU run")
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
