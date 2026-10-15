"""Summarize three published synthetic Qwen GPU process repeats descriptively."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


CONDITIONS = ("native_auto", "explicit_native_noop", "legacy_v517_reference")
RUN_IDS = ("01", "02", "03")
PREFIX = "qwen_temporal_position_synthetic_40frame_repeat_20261015_"
SOURCE_NAMES = {
    "qwen_temporal_geometry.py", "run_dataset_video_gpu_smoke.py",
    "run_qwen_temporal_position_gpu_check.py", "video_time_mapping.py",
}


def _json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj, raw


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_source(repo: Path, commit: str, name: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:scripts/{name}"], cwd=repo,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return _sha(result.stdout)


def _verify_committed_artifact(repo: Path, relative: str, raw: bytes) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=repo,
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if tracked.returncode != 0:
        raise ValueError(f"input artifact is not tracked by Git: {relative}")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repo,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if committed != raw:
        raise ValueError(f"working input differs from committed Git blob: {relative}")


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def analyze(repo: Path, run_ids: tuple[str, ...] = RUN_IDS) -> dict:
    if tuple(run_ids) != RUN_IDS:
        raise ValueError("exactly repeat runs 01, 02, and 03 are required")
    rows = []
    for run_id in run_ids:
        name = f"{PREFIX}{run_id}.json"
        summary_relative = f"results/{name}"
        manifest_relative = f"results/{PREFIX}{run_id}_run_manifest.json"
        summary, raw = _json(repo / summary_relative)
        manifest, manifest_raw = _json(repo / manifest_relative)
        _verify_committed_artifact(repo, summary_relative, raw)
        _verify_committed_artifact(repo, manifest_relative, manifest_raw)
        digest = _sha(raw)
        if (manifest.get("artifact") != f"results/{name}"
                or manifest.get("artifact_sha256") != digest
                or type(manifest.get("schema_version")) is not int
                or manifest.get("schema_version") != 1
                or manifest.get("fixture") != "synthetic_subsecond"
                or any(type(manifest.get(key)) is not int for key in
                       ("requested_samples", "effective_samples", "failures"))
                or manifest.get("requested_samples") != 1
                or manifest.get("effective_samples") != 1
                or manifest.get("failures") != 0):
            raise ValueError(f"manifest or completion mismatch for run {run_id}")
        if (type(summary.get("schema_version")) is not int or summary.get("schema_version") != 1
                or summary.get("purpose") != "synthetic_subsecond_temporal_position_regression_check"
                or summary.get("fixture") != "deterministic synthetic pixels; no project media or annotations"
                or summary.get("source_commit") != manifest.get("source_commit")
                or summary.get("model_revision") != manifest.get("model_revision")
                or summary.get("model_config_sha256") != manifest.get("model_config_sha256")
                or summary.get("raw_run_sha256") != manifest.get("raw_run_sha256")
                or not isinstance(summary.get("sample_counts"), dict)
                or any(type(summary["sample_counts"].get(key)) is not int for key in
                       ("requested", "effective", "failures"))
                or summary.get("sample_counts") != {"requested": 1, "effective": 1, "failures": 0}
                or not isinstance(summary.get("forward_counts"), dict)
                or any(type(summary["forward_counts"].get(key)) is not int for key in
                       ("requested", "effective"))
                or summary.get("forward_counts") != {"requested": 3, "effective": 3}
                or summary.get("synthetic_video", {}).get("frames") != 40):
            raise ValueError(f"summary identity/count mismatch for run {run_id}")
        commit = manifest.get("source_commit")
        source_hashes = summary.get("source_sha256")
        if (not isinstance(commit, str) or len(commit) != 40
                or not isinstance(source_hashes, dict) or set(source_hashes) != SOURCE_NAMES):
            raise ValueError(f"incomplete source provenance for run {run_id}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode != 0:
            raise ValueError(f"source commit is not an ancestor of current HEAD: run {run_id}")
        for source_name, expected in source_hashes.items():
            if _git_source(repo, commit, source_name) != expected:
                raise ValueError(f"source hash does not match source commit: {run_id}/{source_name}")
        resources = summary.get("gpu_resources", {})
        if (resources.get("device") != "NVIDIA GeForce RTX 5070 Ti"
                or not _positive(resources.get("total_gib"))
                or not _positive(resources.get("free_before_gib"))
                or resources["free_before_gib"] > resources["total_gib"]):
            raise ValueError(f"invalid GPU preflight for run {run_id}")
        video = summary.get("synthetic_video", {})
        if (summary.get("model") != "Qwen/Qwen2.5-VL-7B-Instruct"
                or summary.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
                or summary.get("model_config_sha256") != "77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43"
                or summary.get("seed") != 20261012
                or summary.get("quantization") != "4bit"
                or summary.get("compute_dtype") != "bfloat16"
                or video.get("grid_thw") != [20, 32, 32]
                or video.get("sampled_fps") != 8 / 3
                or video.get("seconds_per_grid") != 0.75):
            raise ValueError(f"fixed model/configuration/fixture mismatch for run {run_id}")
        forwards = resources.get("forwards")
        if not isinstance(forwards, list) or [r.get("condition") for r in forwards] != list(CONDITIONS):
            raise ValueError(f"condition set/order mismatch for run {run_id}")
        measurements = {}
        for item in forwards:
            seconds, peak = item.get("forward_seconds"), item.get("peak_allocated_gib")
            if not _positive(seconds) or not _positive(peak) or peak > resources["total_gib"]:
                raise ValueError(f"invalid measurement for run {run_id}")
            measurements[item["condition"]] = {
                "forward_seconds": seconds, "peak_allocated_gib": peak,
            }
        identity = {
            key: summary.get(key) for key in
            ("model", "model_revision", "model_config_sha256", "versions", "cuda_version",
             "seed", "quantization", "compute_dtype", "source_sha256")
        }
        identity.update({"device": resources["device"], "total_gib": resources["total_gib"]})
        if any(value is None for value in identity.values()):
            raise ValueError(f"missing run identity for {run_id}")
        rows.append({"run_id": run_id, "artifact": f"results/{name}",
                     "artifact_sha256": digest,
                     "manifest": manifest_relative,
                     "manifest_sha256": _sha(manifest_raw), "source_commit": commit,
                     "identity": identity, "measurements": measurements})
    reference = rows[0]["identity"].copy()
    for row in rows[1:]:
        if row["identity"] != reference:
            raise ValueError("model, software, precision, seed, or runner source changed across runs")

    result = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for metric in ("forward_seconds", "peak_allocated_gib"):
            values = [row["measurements"][condition][metric] for row in rows]
            result[condition][metric] = {
                "values": values, "median": sorted(values)[1],
                "minimum": min(values), "maximum": max(values),
                "range": max(values) - min(values),
            }
    return {
        "schema_version": 1,
        "purpose": "descriptive_repeatability_summary_synthetic_gpu_engineering_check",
        "analyzer_sha256": _sha(Path(__file__).read_bytes()),
        "run_count": len(rows), "requested_samples_per_run": 1,
        "effective_samples_per_run": 1, "failures_per_run": 0,
        "forwards_per_run": 3, "fixture_frames": 40,
        "identity": reference, "runs": rows, "conditions": result,
        "limitations": [
            "Three fresh processes only; descriptive min/median/max/range, no inferential test or stable variance estimate.",
            "Fixed condition order confounds condition with warmup, runtime order, and thermal state.",
            "Synthetic engineering check only; no faithfulness, QA, or semantic-validity evidence.",
            "Peak allocated VRAM is PyTorch allocation, not total-device memory or a capacity guarantee.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.repo / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = analyze(args.repo.resolve())
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": data["run_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
