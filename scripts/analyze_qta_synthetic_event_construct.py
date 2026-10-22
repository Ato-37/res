#!/usr/bin/env python3
"""Analyze a complete local synthetic QTA construct receipt by frozen protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
from typing import Any

from scipy.stats import t as student_t
from qta_synthetic_event_construct_v1 import assignment_manifest, sha256_canonical
from run_qta_synthetic_event_construct_gpu import (
    MODEL_REVISION, _checkpoint_digest, _preflight_expected_source_subset,
    source_hashes, validate_preflight_receipt, validate_smoke_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data/private_nextgqa"
PROTOCOL = ROOT / "experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md"
ANALYZER = Path(__file__).resolve()
STRATA = (4, 10, 18, 24)
EXPECTED_LIMITATIONS = [
    "synthetic construct validation only; no explanation or claim was evaluated",
    "approximate model-based stratified t inference; no finite-population correction",
    "does not establish natural-video evidence dependence or explanation faithfulness",
]


class AnalysisError(ValueError):
    """Input is not a complete, private, protocol-compatible full cohort receipt."""


def _read_private_json(path: Path, private_root: Path) -> tuple[dict[str, Any], str]:
    root = Path(os.path.abspath(private_root))
    absolute = Path(os.path.abspath(path))
    if absolute.parent != root:
        raise AnalysisError("input_must_be_direct_child_of_private_directory")
    root_info = root.lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700):
        raise AnalysisError("private_directory_permissions_invalid")
    try:
        fd = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise AnalysisError("input_open_failed") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 1_000_000):
            raise AnalysisError("input_permissions_or_size_invalid")
        raw = b""
        while len(raw) <= 1_000_000:
            block = os.read(fd, min(65536, 1_000_001 - len(raw)))
            if not block:
                break
            raw += block
        if len(raw) > 1_000_000:
            raise AnalysisError("input_too_large")
    finally:
        os.close(fd)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("input_json_invalid") from exc
    if not isinstance(value, dict):
        raise AnalysisError("input_schema_invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_full_receipt(receipt: dict[str, Any], *,
                           preflight: dict[str, Any], preflight_sha256: str,
                           smoke: dict[str, Any], smoke_sha256: str
                           ) -> dict[str, dict[str, float]]:
    if (type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "passed_full_synthetic_construct_cohort"
            or receipt.get("mode") != "full"
            or receipt.get("purpose") != "synthetic_qta_construct_check_not_faithfulness"
            or receipt.get("requested_pairs") != 40
            or receipt.get("effective_pairs") != 40
            or receipt.get("requested_clips") != 80
            or receipt.get("effective_clips") != 80
            or receipt.get("requested_forwards") != 720
            or receipt.get("attempted_forwards") != 720
            or receipt.get("completed_forwards") != 720
            or receipt.get("failure_count") != 0
            or receipt.get("cohort_included") is not True
            or receipt.get("faithfulness_result") is not False
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("precision") != "4bit_bf16_compute_bf16_eager"
            or receipt.get("protocol_sha256") != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()):
        raise AnalysisError("full_cohort_receipt_incomplete_or_protocol_mismatch")
    expected_sources = source_hashes(ROOT)
    if receipt.get("source_sha256") != expected_sources:
        raise AnalysisError("full_cohort_source_hash_mismatch")
    for field in ("assignment_manifest_sha256", "external_smoke_seed_sha256",
                  "preflight_receipt_sha256", "smoke_receipt_sha256",
                  "checkpoint_sha256", "model_config_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise AnalysisError("full_cohort_provenance_digest_invalid")
    checkpoint_shards = receipt.get("checkpoint_shards")
    if (not isinstance(checkpoint_shards, dict) or len(checkpoint_shards) != 5
            or any(not isinstance(row, dict) or set(row) != {"size_bytes", "sha256"}
                   or type(row["size_bytes"]) is not int or row["size_bytes"] < 1
                   or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
                   or any(char not in "0123456789abcdef" for char in row["sha256"])
                   for row in checkpoint_shards.values())
            or _checkpoint_digest(checkpoint_shards) != receipt["checkpoint_sha256"]):
        raise AnalysisError("checkpoint_manifest_invalid")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("python"), str):
        raise AnalysisError("runtime_manifest_invalid")
    try:
        expected_assignment_sha = sha256_canonical(
            assignment_manifest(python_version=runtime["python"]))
    except Exception as exc:
        raise AnalysisError("assignment_manifest_invalid") from exc
    if receipt.get("assignment_manifest_sha256") != expected_assignment_sha:
        raise AnalysisError("assignment_manifest_hash_mismatch")
    if (not isinstance(receipt.get("source_commit"), str)
            or len(receipt["source_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in receipt["source_commit"])):
        raise AnalysisError("source_commit_invalid")
    if (receipt.get("preflight_receipt_sha256") != preflight_sha256
            or receipt.get("smoke_receipt_sha256") != smoke_sha256):
        raise AnalysisError("upstream_receipt_digest_mismatch")
    try:
        validate_preflight_receipt(
            preflight, expected_protocol_sha256=receipt["protocol_sha256"],
            expected_source_sha256=_preflight_expected_source_subset(expected_sources),
            expected_assignment_sha256=receipt["assignment_manifest_sha256"], mode="full")
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("preflight_receipt_binding_invalid") from exc
    try:
        validate_smoke_receipt(
            smoke, expected_protocol_sha256=receipt["protocol_sha256"],
            expected_sources=expected_sources,
            expected_assignment_sha256=receipt["assignment_manifest_sha256"],
            expected_checkpoint_sha256=receipt["checkpoint_sha256"],
            expected_model_config_sha256=receipt["model_config_sha256"],
            expected_smoke_seed_sha256=receipt["external_smoke_seed_sha256"],
            expected_preflight_receipt_sha256=preflight_sha256,
            expected_source_commit=receipt["source_commit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("smoke_receipt_binding_invalid") from exc
    if (type(preflight.get("schema_version")) is not int
            or preflight.get("schema_version") != 1
            or preflight.get("status") != "passed_cpu_processor_preflight_no_model_forward"
            or preflight.get("purpose") != "qta_synthetic_event_construct_processor_preflight"
            or preflight.get("requested_pairs") != 40 or preflight.get("effective_pairs") != 40
            or preflight.get("requested_clips") != 80 or preflight.get("effective_clips") != 80
            or preflight.get("model_forwards") != 0
            or preflight.get("model_weights_loaded") is not False
            or preflight.get("gpu_used") is not False
            or preflight.get("protocol_sha256") != receipt.get("protocol_sha256")
            or preflight.get("source_sha256") != _preflight_expected_source_subset(expected_sources)
            or preflight.get("assignment_manifest_sha256") != receipt.get("assignment_manifest_sha256")
            or preflight.get("processor_sha256") != receipt.get("processor_sha256")
            or preflight.get("tokenizer_sha256") != receipt.get("tokenizer_sha256")):
        raise AnalysisError("preflight_receipt_binding_invalid")
    if (type(smoke.get("schema_version")) is not int
            or smoke.get("schema_version") != 1
            or smoke.get("status") != "passed_external_engineering_smoke_only"
            or smoke.get("mode") != "smoke"
            or smoke.get("requested_pairs") != 1 or smoke.get("effective_pairs") != 1
            or smoke.get("requested_clips") != 2 or smoke.get("effective_clips") != 2
            or smoke.get("requested_forwards") != 18
            or smoke.get("attempted_forwards") != 18 or smoke.get("completed_forwards") != 18
            or smoke.get("failure_count") != 0 or smoke.get("cohort_included") is not False
            or smoke.get("faithfulness_result") is not False
            or smoke.get("model_weights_loaded") is not True
            or smoke.get("gpu_used") is not True
            or smoke.get("model_revision") != receipt.get("model_revision")
            or smoke.get("source_sha256") != expected_sources
            or smoke.get("protocol_sha256") != receipt.get("protocol_sha256")
            or smoke.get("assignment_manifest_sha256") != receipt.get("assignment_manifest_sha256")
            or smoke.get("external_smoke_seed_sha256") != receipt.get("external_smoke_seed_sha256")
            or smoke.get("preflight_receipt_sha256") != preflight_sha256
            or smoke.get("source_commit") != receipt.get("source_commit")
            or smoke.get("checkpoint_shards") != checkpoint_shards
            or smoke.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
            or smoke.get("model_config_sha256") != receipt.get("model_config_sha256")
            or smoke.get("processor_sha256") != receipt.get("processor_sha256")
            or smoke.get("tokenizer_sha256") != receipt.get("tokenizer_sha256")):
        raise AnalysisError("smoke_receipt_binding_invalid")
    groups = receipt.get("analysis_sufficient_statistics_by_onset")
    if not isinstance(groups, dict) or set(groups) != {str(onset) for onset in STRATA}:
        raise AnalysisError("stratum_sufficient_statistics_invalid")
    result: dict[str, dict[str, float]] = {}
    for onset in STRATA:
        key = str(onset)
        item = groups[key]
        if (not isinstance(item, dict) or item.get("n_pairs") != 10
                or set(item) != {"n_pairs", "qta_delta_sum", "qta_delta_sum_squares",
                                 "sham_delta_sum", "sham_delta_sum_squares"}):
            raise AnalysisError("stratum_denominator_or_fields_invalid")
        row = {}
        for field in ("qta_delta_sum", "qta_delta_sum_squares", "sham_delta_sum",
                      "sham_delta_sum_squares"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AnalysisError("stratum_statistic_nonfinite_or_invalid")
            row[field] = float(value)
        for prefix in ("qta", "sham"):
            total = row[f"{prefix}_delta_sum"]
            total_sq = row[f"{prefix}_delta_sum_squares"]
            centered = total * total / 10
            numerator = total_sq - centered
            if not math.isfinite(centered) or not math.isfinite(numerator):
                raise AnalysisError("stratum_moment_arithmetic_overflow")
            tolerance = 1e-12 * max(1.0, abs(total_sq), abs(total * total / 10))
            if numerator < -tolerance:
                raise AnalysisError("stratum_sum_squares_inconsistent")
        result[key] = row
    return result


def _estimate(groups: dict[str, dict[str, float]], prefix: str) -> dict[str, Any]:
    means = []
    variances = []
    for onset in STRATA:
        row = groups[str(onset)]
        total = row[f"{prefix}_delta_sum"]
        total_sq = row[f"{prefix}_delta_sum_squares"]
        means.append(total / 10)
        centered = total * total / 10
        numerator = total_sq - centered
        if not math.isfinite(centered) or not math.isfinite(numerator):
            raise AnalysisError("stratum_moment_arithmetic_overflow")
        tolerance = 1e-12 * max(1.0, abs(total_sq), abs(centered))
        variances.append(max(0.0, numerator) / 9 if numerator >= -tolerance else math.nan)
    weights = [0.25] * 4
    theta = sum(weight * mean for weight, mean in zip(weights, means, strict=True))
    components = [weight**2 * variance / 10
                  for weight, variance in zip(weights, variances, strict=True)]
    variance = sum(components)
    if variance == 0:
        return {"estimate": theta, "standard_error": 0.0, "df": None,
                "ci95": None, "two_sided_p": None, "stratum_means": means,
                "stratum_sample_variances": variances,
                "inference_status": "undefined_zero_estimated_variance"}
    df_denominator = sum(component**2 / 9 for component in components)
    if df_denominator <= 0 or not math.isfinite(df_denominator):
        raise AnalysisError("welch_satterthwaite_df_invalid")
    df = variance**2 / df_denominator
    se = math.sqrt(variance)
    critical = float(student_t.ppf(0.975, df))
    p_value = float(2 * student_t.sf(abs(theta / se), df))
    return {"estimate": theta, "standard_error": se, "df": df,
            "ci95": [theta - critical * se, theta + critical * se],
            "two_sided_p": p_value, "stratum_means": means,
            "stratum_sample_variances": variances,
            "inference_status": "approximate_stratified_welch_satterthwaite_t"}


def analyze(receipt: dict[str, Any], receipt_sha256: str, *,
            preflight: dict[str, Any], preflight_sha256: str,
            smoke: dict[str, Any], smoke_sha256: str) -> dict[str, Any]:
    groups = _validate_full_receipt(
        receipt, preflight=preflight, preflight_sha256=preflight_sha256,
        smoke=smoke, smoke_sha256=smoke_sha256)
    return {
        "schema_version": 1,
        "purpose": "analysis_of_synthetic_qta_construct_check_not_faithfulness",
        "status": "analyzed_full_synthetic_cohort",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_receipt_sha256": receipt_sha256,
        "preflight_receipt_sha256": preflight_sha256,
        "smoke_receipt_sha256": smoke_sha256,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(ANALYZER.read_bytes()).hexdigest(),
        "primary_endpoint": "equally_weighted_onset_stratum_mean_scene_pair_DeltaF_QTA",
        "primary": _estimate(groups, "qta"),
        "secondary_sham_descriptive": {
            "equally_weighted_mean": sum(groups[str(onset)]["sham_delta_sum"] / 10
                                          for onset in STRATA) / 4,
            "stratum_means": [groups[str(onset)]["sham_delta_sum"] / 10
                              for onset in STRATA],
            "inference": "not_performed_by_frozen_protocol",
        },
        "n_scene_pairs": 40,
        "n_pairs_per_onset": 10,
        "onset_strata": list(STRATA),
        "statistical_test": "two-sided one-sample t test of theta=0 using stratified Welch-Satterthwaite df",
        "multiplicity_adjustment": "none; one preregistered primary endpoint",
        "limitations": EXPECTED_LIMITATIONS,
        "faithfulness_result": False,
    }


def _write_new_private_json(path: Path, payload: dict[str, Any], private_root: Path) -> Path:
    root = Path(os.path.abspath(private_root))
    output = Path(os.path.abspath(path))
    if output.parent != root or output.suffix.lower() != ".json":
        raise AnalysisError("output_must_be_json_direct_child_of_private_directory")
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise AnalysisError("private_directory_permissions_invalid")
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    temp_name = f".{output.name}.{secrets.token_hex(8)}.partial"
    fd = None
    linked = False
    try:
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=root_fd)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError("short_write")
            written += count
        os.fsync(fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != len(raw)):
            raise AnalysisError("output_write_verification_failed")
        os.close(fd)
        fd = None
        try:
            os.link(temp_name, output.name, src_dir_fd=root_fd, dst_dir_fd=root_fd,
                    follow_symlinks=False)
        except FileExistsError as exc:
            raise AnalysisError("refusing_to_overwrite_output") from exc
        linked = True
        os.unlink(temp_name, dir_fd=root_fd)
        os.fsync(root_fd)
    except Exception:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        if linked:
            try:
                os.unlink(output.name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(root_fd)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="full GPU receipt under data/private_nextgqa")
    parser.add_argument("--preflight-receipt", type=Path, required=True,
                        help="matching CPU preflight receipt under data/private_nextgqa")
    parser.add_argument("--smoke-receipt", type=Path, required=True,
                        help="matching reserved-seed GPU smoke receipt under data/private_nextgqa")
    parser.add_argument("--output", type=Path, required=True, help="new owner-only JSON under data/private_nextgqa")
    args = parser.parse_args(argv)
    try:
        receipt, digest = _read_private_json(args.receipt, PRIVATE_ROOT)
        preflight, preflight_digest = _read_private_json(args.preflight_receipt, PRIVATE_ROOT)
        smoke, smoke_digest = _read_private_json(args.smoke_receipt, PRIVATE_ROOT)
        result = analyze(receipt, digest, preflight=preflight,
                         preflight_sha256=preflight_digest, smoke=smoke,
                         smoke_sha256=smoke_digest)
        output = _write_new_private_json(args.output, result, PRIVATE_ROOT)
    except (OSError, AnalysisError, ValueError) as exc:
        parser.error(str(exc) if str(exc).isascii() else "synthetic_analysis_failed")
    primary = result["primary"]
    print(json.dumps({"status": result["status"], "n_scene_pairs": result["n_scene_pairs"],
                      "estimate": primary["estimate"], "ci95": primary["ci95"],
                      "two_sided_p": primary["two_sided_p"],
                      "private_output_name": output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
