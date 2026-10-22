#!/usr/bin/env python3
"""Recompute an aggregate-only report from a private QA-key-bias receipt.

The output contains no row, video, question, answer, or claim identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
from typing import Any

BRANCHES = (
    "baseline_qa_pixel_mask", "baseline_control_pixel_mask",
    "qa_bias_qa_pixel_mask", "qa_bias_control_pixel_mask",
    "sham_bias_qa_pixel_mask", "sham_bias_control_pixel_mask",
)
T975_DF2 = 4.302652729911275


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if len(values) != 3 or not all(_finite(x) for x in values):
        raise ValueError("summary_requires_three_finite_rows")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T975_DF2 * sd / math.sqrt(3)
    return {"rows_nat_per_token": values, "mean_nat_per_token": mean,
            "sd_nat_per_token": sd,
            "descriptive_95pct_t_interval_nat_per_token": [mean-half, mean+half],
            "positive_rows": sum(x > 0 for x in values), "n": 3}


def summarize(payload: dict[str, Any], artifact_sha256: str) -> dict[str, Any]:
    if (payload.get("status") != "completed_exploratory_claim_text_qa_interval_key_bias_cohort"
            or payload.get("mode") != "full"
            or payload.get("direct_faithfulness") is not False
            or payload.get("faithfulness_result") is not False
            or payload.get("requested_records") != 3
            or payload.get("effective_records") != 3
            or payload.get("requested_forwards") != 18
            or payload.get("attempted_forwards") != 18
            or payload.get("completed_forwards") != 18
            or payload.get("failures") not in (None, [])):
        raise ValueError("full_receipt_contract_invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("full_receipt_record_count_invalid")
    primary_values: list[float] = []
    specificity_values: list[float] = []
    observed_rows = []
    seen_tokens: set[str] = set()
    for record in records:
        row_token = record.get("row_token_sha256")
        if not isinstance(row_token, str) or len(row_token) != 64 or row_token in seen_tokens:
            raise ValueError("row_identity_contract_invalid")
        seen_tokens.add(row_token)
        token_scores = record.get("branch_token_logprobs_nats")
        branch_means = record.get("branch_mean_logprob_nats_per_token")
        if not isinstance(token_scores, dict) or set(token_scores) != set(BRANCHES):
            raise ValueError("branch_token_scores_invalid")
        if not isinstance(branch_means, dict) or set(branch_means) != set(BRANCHES):
            raise ValueError("branch_means_invalid")
        token_count = record.get("target_token_count")
        if type(token_count) is not int or token_count < 1:
            raise ValueError("target_token_count_invalid")
        recomputed: dict[str, float] = {}
        for branch in BRANCHES:
            scores = token_scores[branch]
            if (not isinstance(scores, list) or len(scores) != token_count
                    or not all(_finite(x) for x in scores)
                    or not _finite(branch_means[branch])):
                raise ValueError("branch_token_vector_invalid")
            value = statistics.mean(float(x) for x in scores)
            if not _close(value, float(branch_means[branch])):
                raise ValueError("branch_mean_arithmetic_mismatch")
            recomputed[branch] = value
        f_base = recomputed["baseline_control_pixel_mask"] - recomputed["baseline_qa_pixel_mask"]
        f_qa = recomputed["qa_bias_control_pixel_mask"] - recomputed["qa_bias_qa_pixel_mask"]
        f_sham = recomputed["sham_bias_control_pixel_mask"] - recomputed["sham_bias_qa_pixel_mask"]
        primary = f_qa - f_base
        specificity = f_qa - f_sham
        saved = record.get("primary")
        checks = {"f_baseline_nats": f_base, "f_qa_bias_nats": f_qa,
                  "f_sham_bias_nats": f_sham,
                  "delta_qa_bias_minus_baseline_nats": primary,
                  "specificity_qa_minus_sham_bias_nats": specificity}
        if not isinstance(saved, dict) or any(
                not _finite(saved.get(key)) or not _close(float(saved[key]), value)
                for key, value in checks.items()):
            raise ValueError("primary_arithmetic_mismatch")
        if record.get("forward_count") != {"attempted": 6, "completed": 6}:
            raise ValueError("row_forward_count_invalid")
        primary_values.append(primary)
        specificity_values.append(specificity)
        observed_rows.append({"F_baseline": f_base, "F_QA_bias": f_qa,
                              "F_sham": f_sham, "primary": primary,
                              "specificity": specificity})
    return {
        "schema_version": 1,
        "analysis": "aggregate_only_claim_QA_interval_internal_bias_summary",
        "source_receipt_sha256": artifact_sha256,
        "n_rows": 3,
        "requested_effective_attempted_completed_forwards": [18, 18, 18, 18],
        "frame_count": payload.get("frame_count"),
        "sham_protocol": payload.get("key_bias_windows"),
        "primary_QA_bias_minus_baseline": _mean_ci(primary_values),
        "specificity_QA_bias_minus_matched_sham": _mean_ci(specificity_values),
        "row_level_aggregate_scores": observed_rows,
        "direct_faithfulness": False,
        "interpretation": "small sequential exploratory proxy; QA intervals are not claim-reviewed support; no direct semantic faithfulness or population-level inference",
    }


def _load_private(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private_receipt_permissions_invalid")
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("private_receipt_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_sources(repo: Path, payload: dict[str, Any]) -> None:
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_hash_manifest_missing")
    for relative, expected in sources.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or not isinstance(expected, str)
                or len(expected) != 64):
            raise ValueError("source_hash_manifest_invalid")
        digest = hashlib.sha256((repo / path).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("source_hash_mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="owner-only receipt under data/private_nextgqa")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    private = (repo / "data/private_nextgqa").resolve(strict=True)
    path = args.input.resolve(strict=True)
    if not path.is_relative_to(private):
        parser.error("input_must_be_inside_private_data")
    try:
        payload, digest = _load_private(path)
        _verify_sources(repo, payload)
        print(json.dumps(summarize(payload, digest), sort_keys=True,
                         ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
