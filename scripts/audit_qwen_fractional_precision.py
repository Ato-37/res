"""Compare fixed Qwen fractional temporal-ID cases using Python and Torch CPU math.

This is a numeric implementation audit, not model inference or a claim about
which dtype is globally correct. Cases and boundaries are fixed in source.
Usage::

    .venv/bin/python scripts/audit_qwen_fractional_precision.py --output results/NEW.json
    .venv/bin/python scripts/audit_qwen_fractional_precision.py --output results/NEW.json --json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

from qwen_temporal_geometry import qwen_temporal_geometry


FIXED_INTERVALS_SECONDS = (
    0.49999999,
    0.5,
    0.50000001,
    0.99999999,
    1.0,
    1.00000001,
    199.9998016357422,  # interval observed in the fixed NOAA runtime artifact
    0.75,               # fixed synthetic subsecond GPU fixture
)
TEMPORAL_STEPS = 4
TEMPORAL_PATCH_SIZE = 2
TOKENS_PER_SECOND = 2
BASE_POSITION = 15
NOAA_RESULT_RELATIVE = Path("results/noaa_temporal_position_gpu_check_20261012_01.json")
SYNTHETIC_RESULT_RELATIVE = Path("results/qwen_temporal_position_synthetic_subsecond_20261012_01.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _torch_ticks(seconds: float, steps: int, tps: int, base: int) -> dict:
    """Evaluate two source-order float32 expressions on CPU, then convert to long."""
    import torch

    interval32 = torch.tensor(seconds, dtype=torch.float32, device="cpu")
    positions32 = torch.arange(steps, dtype=torch.float32, device="cpu")
    # Current Transformers main: form (tokens_per_second * seconds) first,
    # then multiply temporal indices and quantize to integer coordinates.
    main_scaled_interval = interval32 * tps
    main_relative = (positions32 * main_scaled_interval).long()
    # Transformers 4.49 source order: index * seconds * tokens_per_second,
    # with the intermediate tensor operations remaining float32.
    v449_relative = (positions32 * interval32 * tps).long()
    return {
        "float32_represented_seconds": float(interval32.item()),
        "main_float32_scaled_interval": float(main_scaled_interval.item()),
        "transformers_main_torch_float32_ticks": [base + int(v) for v in main_relative.tolist()],
        "transformers_449_torch_float32_ticks": [base + int(v) for v in v449_relative.tolist()],
    }


def audit_interval(seconds: float, *, steps: int = TEMPORAL_STEPS) -> dict:
    """Compare the pure helper with source-order Torch float32 arithmetic."""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise ValueError("seconds must be a finite positive number")
    if not (0 < float(seconds) < float("inf")):
        raise ValueError("seconds must be a finite positive number")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")

    # An interval is two sampled frames at a constant frame rate for a
    # temporal_patch_size of two. The helper's own tolerance checks this input.
    timestamps = [index * (float(seconds) / TEMPORAL_PATCH_SIZE)
                  for index in range(steps * TEMPORAL_PATCH_SIZE)]
    geometry = qwen_temporal_geometry(
        timestamps,
        TEMPORAL_PATCH_SIZE,
        tokens_per_second=TOKENS_PER_SECOND,
        second_per_grid=float(seconds),
        video_start_position=BASE_POSITION,
    )
    helper_ticks = list(geometry["transformers_4_49_0_temporal_rope_ticks_fractional"])
    torch_values = _torch_ticks(float(seconds), steps, TOKENS_PER_SECOND, BASE_POSITION)
    main_ticks = torch_values["transformers_main_torch_float32_ticks"]
    v449_ticks = torch_values["transformers_449_torch_float32_ticks"]
    return {
        "input_seconds_per_grid_python_float": float(seconds),
        "temporal_steps": steps,
        "tokens_per_second": TOKENS_PER_SECOND,
        "video_start_position": BASE_POSITION,
        **torch_values,
        "helper_python_float_fractional_ticks": helper_ticks,
        "helper_matches_transformers_main_torch_float32": helper_ticks == main_ticks,
        "helper_matches_transformers_449_torch_float32": helper_ticks == v449_ticks,
        "helper_vs_main_mismatch_step_indices": [
            index for index, (left, right) in enumerate(zip(helper_ticks, main_ticks))
            if left != right
        ],
        "helper_vs_449_mismatch_step_indices": [
            index for index, (left, right) in enumerate(zip(helper_ticks, v449_ticks))
            if left != right
        ],
    }


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_report(repo_root: Path) -> dict:
    import torch
    import transformers
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    noaa_path = repo_root / NOAA_RESULT_RELATIVE
    synthetic_path = repo_root / SYNTHETIC_RESULT_RELATIVE
    noaa_result = json.loads(noaa_path.read_text(encoding="utf-8"))
    synthetic_result = json.loads(synthetic_path.read_text(encoding="utf-8"))
    if noaa_result.get("purpose") != "real_video_temporal_position_runtime_regression_check":
        raise ValueError("unexpected NOAA runtime input artifact")
    if synthetic_result.get("purpose") != "synthetic_subsecond_temporal_position_regression_check":
        raise ValueError("unexpected synthetic runtime input artifact")

    fixed_cases = [audit_interval(seconds) for seconds in FIXED_INTERVALS_SECONDS]
    noaa_seconds = float(noaa_result["seconds_per_grid"])
    synthetic_seconds = float(synthetic_result["seconds_per_grid"])
    noaa_steps = int(noaa_result["grid_thw"][0])
    synthetic_steps = int(synthetic_result["grid_thw"][0])
    noaa_case = audit_interval(noaa_seconds, steps=noaa_steps)
    synthetic_case = audit_interval(synthetic_seconds, steps=synthetic_steps)

    helper_path = Path(__file__).resolve().with_name("qwen_temporal_geometry.py")
    time_mapping_path = Path(__file__).resolve().with_name("video_time_mapping.py")
    model_source_path = Path(inspect.getfile(Qwen2_5_VLModel)).resolve()
    runner_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "purpose": "qwen_fractional_temporal_position_precision_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(repo_root),
        "fixed_cases": fixed_cases,
        "case_count_requested": len(FIXED_INTERVALS_SECONDS),
        "case_count_effective": len(fixed_cases),
        "failures": 0,
        "actual_runtime_fixtures": {
            "noaa": {
                "input_artifact_sha256": sha256_file(noaa_path),
                "source_media_sha256": noaa_result["media_sha256"],
                "observed_seconds_per_grid": noaa_seconds,
                "temporal_steps": noaa_steps,
                "native_ticks_from_runtime": noaa_result["native_temporal_anchors"],
                "fractional_ticks_from_runtime": noaa_result["fractional_temporal_anchors"],
                "precision_audit": noaa_case,
            },
            "synthetic_subsecond": {
                "input_artifact_sha256": sha256_file(synthetic_path),
                "synthetic_pixels_sha256": synthetic_result["synthetic_pixels_sha256"],
                "observed_seconds_per_grid": synthetic_seconds,
                "temporal_steps": synthetic_steps,
                "native_ticks_from_runtime": synthetic_result["native_temporal_anchors"],
                "fractional_ticks_from_runtime": synthetic_result["fractional_temporal_anchors"],
                "precision_audit": synthetic_case,
            },
        },
        "source_sha256": {
            "runner": sha256_file(runner_path),
            "qwen_temporal_geometry": sha256_file(helper_path),
            "video_time_mapping": sha256_file(time_mapping_path),
            "installed_qwen_modeling_source": sha256_file(model_source_path),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "source_references": {
            "transformers_449": "https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
            "transformers_517": "https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
            "transformers_main": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        },
        "limitations": [
            "CPU-only arithmetic audit; no model weights, GPU, video inference, labels, or QA.",
            "Python float and Torch float32 operation order can quantize values differently near integer boundaries.",
            "No dtype or implementation is declared universally correct; results are limited to the listed fixed inputs and source expressions.",
            "The NOAA and synthetic intervals are imported from previously recorded runtime artifacts; this audit does not rerun those model forwards.",
        ],
    }


def _write_exclusive(output: Path, repo_root: Path, report: dict) -> Path:
    output = output if output.is_absolute() else repo_root / output
    output = output.resolve()
    results_dir = (repo_root / "results").resolve()
    if output.parent != results_dir or output.suffix.lower() != ".json":
        raise ValueError("output must be a JSON file directly under repository results/")
    results_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="unique JSON output directly under repository results/")
    parser.add_argument("--json", action="store_true", help="print the full audit JSON")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        report = build_report(repo_root)
        output = _write_exclusive(args.output, repo_root, report)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"fractional precision audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(output.read_text(encoding="utf-8"), end="")
    else:
        main_mismatches = sum(
            not case["helper_matches_transformers_main_torch_float32"]
            for case in report["fixed_cases"]
        )
        v449_mismatches = sum(
            not case["helper_matches_transformers_449_torch_float32"]
            for case in report["fixed_cases"]
        )
        print(
            f"Qwen fractional precision audit: {report['case_count_effective']}/"
            f"{report['case_count_requested']} fixed cases; helper/Torch-main mismatches "
            f"{main_mismatches}, helper/Torch-4.49 mismatches {v449_mismatches}; "
            f"artifact {output.relative_to(repo_root)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
