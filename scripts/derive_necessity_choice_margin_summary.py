"""Strip a local necessity-run JSON to a privacy-minimal aggregate summary."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_VERSION = re.compile(r"^\d+(?:\.\d+){1,3}(?:\+cu\d{2,4})?$")
_FAILURE_CODES = {
    "cuda_oom", "media_io_error", "input_validation_error", "runtime_error",
}
_SUMMARY_KEYS = (
    "all", "baseline_correct_primary", "baseline_incorrect_secondary",
    "correct_minus_incorrect_interaction",
)
_STAT_KEYS = (
    "n", "mean", "sd", "ci95_pointwise", "positive_fraction",
    "signflip_p_two_sided", "mean_correct_minus_incorrect",
)
_DENIED_KEYS = {
    "video", "video_id", "qid", "question", "choices", "answer", "prompt",
    "sample_id", "sample_key", "records", "failures", "samples_file",
}


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("summary statistic has an invalid type")
    if not math.isfinite(value):
        raise ValueError("summary statistic is non-finite")
    return value


def _summary(value, require_n=True):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("summary cohort must be an object or null")
    result = {}
    for key in _STAT_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key == "ci95_pointwise":
            if item is not None:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("confidence interval must contain two bounds")
                item = [_finite_or_none(x) for x in item]
        else:
            item = _finite_or_none(item)
        result[key] = item
    if require_n and ("n" not in result or type(result["n"]) is not int or result["n"] < 0):
        raise ValueError("summary cohort must record a nonnegative integer n")
    return result


def _version(value, name):
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ValueError(f"{name} is not a safe version token")
    return value


def _revision(value, name):
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{name} must be a full immutable revision")
    return value


def derive(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 3:
        raise ValueError("expected necessity runner result schema version 3")
    records = raw.get("records")
    failures = raw.get("failures")
    requested, effective = raw.get("n_requested"), raw.get("n_effective")
    if (not isinstance(records, list) or not isinstance(failures, list)
            or type(requested) is not int or type(effective) is not int
            or requested < 1 or effective != len(records)
            or requested != effective + len(failures)):
        raise ValueError("run accounting does not reconcile")
    video_ids = []
    for row in records:
        if not isinstance(row, dict) or type(row.get("baseline_choice_correct")) is not bool:
            raise ValueError("effective records have invalid aggregate fields")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("effective record is missing its video unit")
        video_ids.append(video_id)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("records are not one independent analysis unit per video")
    source_commit = raw.get("source_commit")
    if not isinstance(source_commit, str) or not _COMMIT.fullmatch(source_commit):
        raise ValueError("source commit is missing or mutable")
    model = raw.get("model")
    if model != "Qwen/Qwen2.5-VL-7B-Instruct":
        raise ValueError("model is outside the reviewed public model allowlist")
    if type(raw.get("seed")) is not int:
        raise ValueError("run seed is missing")
    draws = raw.get("bootstrap_draws")
    if type(draws) is not int or draws < 100:
        raise ValueError("bootstrap draw count is invalid")
    expected_seeds = {
        "all": raw["seed"],
        "baseline_correct_primary": raw["seed"] + 1,
        "baseline_incorrect_secondary": raw["seed"] + 2,
        "correct_minus_incorrect_interaction": raw["seed"] + 3,
    }
    if raw.get("summary_seed_schedule") != expected_seeds:
        raise ValueError("summary seed schedule does not match the runner contract")
    summary = raw.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(_SUMMARY_KEYS):
        raise ValueError("summary cohorts are incomplete")

    failure_counts = Counter()
    for failure in failures:
        code = failure.get("failure_code") if isinstance(failure, dict) else None
        if not isinstance(code, str):
            code = None
        failure_counts[code if code in _FAILURE_CODES else "other"] += 1

    safe = {
        "schema_version": 1,
        "analysis": "necessity_choice_margin_aggregate",
        "status": "legacy_exploratory_aggregate",
        "source_commit": source_commit,
        "code_sha256": _code_hashes(),
        "model": model,
        "model_revision": _revision(raw.get("model_revision"), "model_revision"),
        "tokenizer_revision": _revision(raw.get("tokenizer_revision"), "tokenizer_revision"),
        "quant": raw.get("quant"),
        "dtype": raw.get("dtype"),
        "attn_implementation": raw.get("attn_implementation"),
        "python_version": _version(raw.get("python_version"), "python_version"),
        "python_implementation": raw.get("python_implementation"),
        "torch_version": _version(raw.get("torch_version"), "torch_version"),
        "transformers_version": _version(raw.get("transformers_version"), "transformers_version"),
        "device": raw.get("device"),
        "seed": raw.get("seed"),
        "bootstrap_draws": raw.get("bootstrap_draws"),
        "summary_seed_schedule": raw.get("summary_seed_schedule"),
        "num_frames": raw.get("num_frames"),
        "frame_size": raw.get("frame_size"),
        "max_pixels": raw.get("max_pixels"),
        "cap_pixels_per_frame": raw.get("cap_pixels_per_frame"),
        "mask_intervention": raw.get("mask_intervention"),
        "primary_estimand": "mean[(full margin - evidence-mask margin) - "
                             "(full margin - matched-control-mask margin)] among baseline-correct samples",
        "control_protocol": "same number of black-masked frames; non-grounding indices chosen "
                            "to minimize temporal rank-distance from evidence indices",
        "n_requested": requested,
        "n_effective": effective,
        "n_failures": len(failures),
        "failure_summary_status": ("no_successful_samples" if effective == 0 else "available"),
        "n_baseline_correct": sum(row["baseline_choice_correct"] for row in records),
        "n_baseline_incorrect": sum(not row["baseline_choice_correct"] for row in records),
        "failure_counts_by_code": dict(sorted(failure_counts.items())),
        "summary": {
            key: _summary(value, require_n=key != "correct_minus_incorrect_interaction")
            for key, value in summary.items()
        },
        "multiplicity": "One predeclared primary cohort (baseline-correct); all-sample, "
                        "baseline-incorrect, and interaction results are secondary and "
                        "not multiplicity-corrected.",
        "signflip_test": "Two-sided sample-level random sign flips of the mean contrast; "
                         "assumes symmetric paired null effects.",
        "limitations": [
            "Black-frame masking is an out-of-distribution perturbation; equal-dose control reduces but does not remove this concern.",
            "The gold-vs-best-foil letter margin is prompt-dependent and is not official free-generation accuracy.",
            "Conditioning on baseline correctness is part of the prespecified primary cohort; this does not establish general XAI faithfulness or head causality.",
        ],
    }
    if safe["quant"] not in {"4bit", "8bit", "none"}:
        raise ValueError("quantization mode is outside the runner contract")
    if safe["dtype"] != "bfloat16" or safe["attn_implementation"] != "eager":
        raise ValueError("precision/runtime mode is outside the runner contract")
    if safe["python_implementation"] not in {"CPython", "PyPy"}:
        raise ValueError("Python implementation is not a safe known token")
    if safe["device"] not in {"cpu", "cuda"}:
        raise ValueError("device type is outside the runner contract")
    safe["runtime_accelerator"] = "CUDA GPU" if safe["device"] == "cuda" else "CPU"
    if type(safe["seed"]) is not int or type(safe["num_frames"]) is not int:
        raise ValueError("seed or frame count has an invalid type")
    safe["frame_size"] = raw.get("frame_size")
    safe["max_pixels"] = raw.get("max_pixels")
    safe["cap_pixels_per_frame"] = raw.get("cap_pixels_per_frame")
    if (type(safe["frame_size"]) is not int or type(safe["max_pixels"]) is not int
            or safe["frame_size"] < 1 or safe["max_pixels"] < 1):
        raise ValueError("frame dimensions must be integers")
    if type(safe["cap_pixels_per_frame"]) is not bool:
        raise ValueError("pixel cap flag must be boolean")
    if (not isinstance(safe["mask_intervention"], dict)
            or set(safe["mask_intervention"]) != {"fill", "blur_radius"}
            or safe["mask_intervention"]["fill"] not in {"black", "blur"}):
        raise ValueError("mask intervention is outside the runner contract")
    blur = safe["mask_intervention"]["blur_radius"]
    if safe["mask_intervention"]["fill"] == "black" and blur is not None:
        raise ValueError("black mask must not publish a blur radius")
    if blur is not None and (isinstance(blur, bool) or not isinstance(blur, (int, float))
                             or not math.isfinite(blur) or blur < 0):
        raise ValueError("mask blur radius is invalid")
    interaction = safe["summary"]["correct_minus_incorrect_interaction"]
    if interaction is not None:
        interaction.update({
            "n_baseline_correct": safe["n_baseline_correct"],
            "n_baseline_incorrect": safe["n_baseline_incorrect"],
        })
    if effective:
        if (safe["summary"]["all"]["n"] != effective
                or safe["summary"]["baseline_correct_primary"]["n"]
                   != safe["n_baseline_correct"]
                or safe["summary"]["baseline_incorrect_secondary"]["n"]
                   != safe["n_baseline_incorrect"]):
            raise ValueError("summary cohort counts do not match the per-video records")
    _reject_private_content(safe)
    return safe


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/derive_necessity_choice_margin_summary.py",
        "scripts/necessity_choice_margin_test.py",
        "scripts/necessity_choice_margin_utils.py",
        "scripts/run_necessity_choice_margin_test_and_publish.py",
        "scripts/activation_patching_sweep.py",
        "scripts/necessity_mask_test.py",
        "scripts/video_time_mapping.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def publish_manifest(summary: dict, artifact_bytes: bytes) -> dict:
    """Create a path- and cohort-free provenance record for a derived result."""
    return {
        "schema_version": 1,
        "artifact": "necessity_choice_margin_aggregate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": summary["source_commit"],
        "derived_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "code_sha256": summary["code_sha256"],
        "n_requested": summary["n_requested"],
        "n_effective": summary["n_effective"],
        "n_failures": summary["n_failures"],
        "privacy_note": "Raw sample manifest and per-item result remain local.",
    }


def _reject_private_content(value, path="summary"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _DENIED_KEYS or "path" in lowered:
                raise ValueError(f"private field is not publishable: {path}.{key}")
            _reject_private_content(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_content(child, f"{path}[{index}]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    derived = derive(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(derived, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")
    args.output.chmod(0o644)


if __name__ == "__main__":
    main()
