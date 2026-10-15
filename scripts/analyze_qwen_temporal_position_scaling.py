"""Reproduce descriptive scaling ratios from synthetic Qwen GPU summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = ("native_auto", "explicit_native_noop", "legacy_v517_reference")
FRAME_COUNTS = (4, 16, 32)


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite_positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_summary(data: dict) -> tuple[int, dict]:
    if (type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("purpose") != "synthetic_subsecond_temporal_position_regression_check"):
        raise ValueError("unexpected artifact type")
    counts = data.get("sample_counts")
    forwards = data.get("forward_counts")
    if (not isinstance(counts, dict) or any(type(counts.get(key)) is not int for key in ("requested", "effective", "failures"))
            or counts != {"requested": 1, "effective": 1, "failures": 0}
            or not isinstance(forwards, dict) or any(type(forwards.get(key)) is not int for key in ("requested", "effective"))
            or forwards != {"requested": 3, "effective": 3}):
        raise ValueError("incomplete sample/forward counts")
    video = data.get("synthetic_video")
    if not isinstance(video, dict):
        raise ValueError("synthetic video summary missing")
    frames = video.get("frames")
    if type(frames) is not int or frames not in FRAME_COUNTS:
        raise ValueError("unsupported frame count")
    grid = video.get("grid_thw")
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(value) is not int for value in grid)
            or grid != [frames // 2, 32, 32]
            or type(video.get("sampled_fps")) not in (int, float)
            or not math.isclose(video.get("sampled_fps", float("nan")), 8 / 3, rel_tol=0, abs_tol=1e-12)
            or type(video.get("seconds_per_grid")) not in (int, float)
            or not math.isclose(video.get("seconds_per_grid", float("nan")), 0.75, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("fixed synthetic temporal geometry mismatch")
    native = video.get("native_temporal_anchors")
    fractional = video.get("fractional_temporal_anchors")
    legacy = video.get("legacy_v517_temporal_anchors")
    expected_native = [15 + int(i * 0.75 * 2) for i in range(frames // 2)]
    expected_legacy = [15] * (frames // 2)
    if (not isinstance(native, list) or any(type(value) is not int for value in native)
            or not isinstance(fractional, list) or any(type(value) is not int for value in fractional)
            or not isinstance(legacy, list) or any(type(value) is not int for value in legacy)
            or native != expected_native or fractional != expected_native or legacy != expected_legacy
            or video.get("native_matches_fractional_reference") is not True
            or type(video.get("changed_position_id_entries")) is not int
            or video.get("changed_position_id_entries") != 0):
        raise ValueError("fixed synthetic temporal anchors mismatch")

    resources = data.get("gpu_resources")
    if not isinstance(resources, dict) or not isinstance(resources.get("device"), str):
        raise ValueError("GPU resource summary missing")
    total = resources.get("total_gib")
    free = resources.get("free_before_gib")
    if not _finite_positive(total) or not _finite_positive(free) or free > total:
        raise ValueError("invalid preflight memory values")
    rows = resources.get("forwards")
    if not isinstance(rows, list) or [r.get("condition") for r in rows if isinstance(r, dict)] != list(CONDITIONS):
        raise ValueError("forward condition set/order mismatch")
    measured = {}
    for row in rows:
        seconds, peak = row.get("forward_seconds"), row.get("peak_allocated_gib")
        # The privacy-safe published schema omits the raw finite-logit flag;
        # its publisher admits summaries only when all raw logits were finite.
        if (not _finite_positive(seconds)
                or not _finite_positive(peak) or peak > total):
            raise ValueError("invalid forward resource measurement")
        measured[row["condition"]] = {"forward_seconds": seconds, "peak_allocated_gib": peak}

    identity = {
        "model": data.get("model"), "model_revision": data.get("model_revision"),
        "model_config_sha256": data.get("model_config_sha256"),
        "device": resources["device"], "cuda_version": data.get("cuda_version"),
        "versions": data.get("versions"), "seed": data.get("seed"),
        "quantization": data.get("quantization"), "compute_dtype": data.get("compute_dtype"),
        "total_gib": total,
    }
    if any(value is None for value in identity.values()):
        raise ValueError("run identity fields missing")
    source_hashes = data.get("source_sha256")
    expected_sources = {
        "run_qwen_temporal_position_gpu_check.py", "qwen_temporal_geometry.py",
        "video_time_mapping.py", "run_dataset_video_gpu_smoke.py",
    }
    if (not isinstance(source_hashes, dict) or set(source_hashes) != expected_sources
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in source_hashes.values())):
        raise ValueError("runner source hash set is incomplete or invalid")
    source_commit = data.get("source_commit")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)):
        raise ValueError("source commit is missing or invalid")
    return frames, {"identity": identity, "video_tokens": frames * 128,
                    "free_before_gib": free, "forwards": measured,
                    "source_commit": source_commit,
                    "source_sha256": source_hashes}


def analyze(paths: list[Path]) -> dict:
    if len(paths) != len(FRAME_COUNTS):
        raise ValueError("exactly one 4-, 16-, and 32-frame artifact is required")
    rows = []
    seen = set()
    for path in paths:
        data, digest = _read(path)
        frames, parsed = _validate_summary(data)
        if frames in seen:
            raise ValueError("duplicate frame-count artifact")
        seen.add(frames)
        rows.append({"frames": frames, "video_tokens": parsed["video_tokens"],
                     "artifact_name": path.name, "artifact_sha256": digest, **parsed})
    rows.sort(key=lambda row: row["frames"])
    if tuple(row["frames"] for row in rows) != FRAME_COUNTS:
        raise ValueError("required frame-count conditions are incomplete")
    if any(row["identity"] != rows[0]["identity"] for row in rows[1:]):
        raise ValueError("model/device/run configuration differs across artifacts")
    reference_sources = rows[0]["source_sha256"]
    for row in rows[1:]:
        if any(row["source_sha256"][name] != reference_sources[name]
               for name in reference_sources if name != "run_qwen_temporal_position_gpu_check.py"):
            raise ValueError("non-runner source differs across artifacts")

    comparisons = []
    for before, after in zip(rows, rows[1:]):
        comparisons.append({
            "from_frames": before["frames"], "to_frames": after["frames"],
            "frame_count_ratio": after["frames"] / before["frames"],
            "forward_time_ratios": {
                condition: after["forwards"][condition]["forward_seconds"] /
                           before["forwards"][condition]["forward_seconds"]
                for condition in CONDITIONS
            },
            "peak_allocated_memory_ratios": {
                condition: after["forwards"][condition]["peak_allocated_gib"] /
                           before["forwards"][condition]["peak_allocated_gib"]
                for condition in CONDITIONS
            },
        })
    return {
        "schema_version": 1,
        "analysis": "descriptive_synthetic_qwen_temporal_capacity_scaling",
        "conditions": [{key: row[key] for key in
                         ("frames", "video_tokens", "artifact_name", "artifact_sha256",
                          "free_before_gib", "forwards", "source_commit")} | {
                              "runner_source_sha256": row["source_sha256"]["run_qwen_temporal_position_gpu_check.py"]
                          } for row in rows],
        "comparisons": comparisons,
        "limitations": [
            "One run per frame count/condition; time ratios are descriptive and have no uncertainty interval.",
            "Input frame content differs across sizes; warmup, order, and runtime noise are uncontrolled.",
            "Peak values are PyTorch allocated memory, not whole-device use or remaining free VRAM.",
            "Runner source hashes are reported per artifact; inspect any difference before treating ratios as directly comparable.",
            "Synthetic engineering diagnostics only; no QA, faithfulness, or semantic evidence is measured.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs=3, type=Path,
                        help="one each of the 4-, 16-, and 32-frame published summaries")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = analyze(args.artifacts)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
