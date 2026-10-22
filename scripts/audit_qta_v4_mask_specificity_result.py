#!/usr/bin/env python3
"""Independent structural/arithmetic audit for a private QTA-v4 specificity receipt."""
from __future__ import annotations

import json
import math
import argparse
import hashlib
import os
from pathlib import Path
import statistics
import stat
from typing import Any

import run_nextgqa_qta_v4_mask_specificity as protocol
import run_nextgqa_qta_v4_union_scorer as v4
from run_nextgqa_qta_exploratory_proxy_v2 import PRIVATE_REL, sha256_file


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(code)
    return float(value)


def independent_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("audit_summary_nonfinite_or_empty")
    return {"n": len(values), "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values), "maximum": max(values),
        "positive": sum(x > 0 for x in values),
        "negative": sum(x < 0 for x in values), "zero": sum(x == 0 for x in values)}


def _same_number(actual: Any, expected: float, label: str) -> None:
    value = _finite(actual, label + "_nonfinite")
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(label + "_arithmetic_mismatch")


def validate_completed_metadata(payload: dict, *, eligible_n: int) -> None:
    if (payload.get("status") != "qta_v4_mask_specificity_exploratory_complete"
            or type(payload.get("requested_rows")) is not int
            or payload.get("requested_rows") != protocol.EXPECTED_N
            or type(payload.get("effective_rows")) is not int
            or payload.get("effective_rows") != protocol.EXPECTED_N
            or type(payload.get("eligible_specificity_rows")) is not int
            or payload.get("eligible_specificity_rows") != eligible_n
            or type(payload.get("structurally_unavailable_rows")) is not int
            or payload.get("structurally_unavailable_rows") != protocol.EXPECTED_N - eligible_n
            or not isinstance(payload.get("summary"), dict)
            or payload.get("rights_status") != "unverified_local_only"
            or payload.get("faithfulness_result") is not False
            or payload.get("inferential_status")
               != "finite cohort descriptive only; no p-value or population interval"):
        raise ValueError("audit_complete_metadata_invalid")


def read_private_score(path: Path, private_root: Path) -> tuple[dict, str, str]:
    """Read, hash, and inspect one owner-only file descriptor snapshot."""
    real_root = private_root.resolve(strict=True)
    if path.resolve(strict=True).parent != real_root or stat.S_IMODE(real_root.stat().st_mode) != 0o700:
        raise ValueError("private_input_root_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()
                or stat.S_IMODE(st.st_mode) != 0o600):
            raise ValueError("private_input_mode_invalid")
        blocks = []
        while block := os.read(fd, 1024 * 1024):
            blocks.append(block)
        raw = b"".join(blocks)
    finally:
        os.close(fd)
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), oct(stat.S_IMODE(st.st_mode))


def audit_records(records: list[dict], *, preflight_rows: list[dict], old_rows: list[dict],
                  branch_counts: dict, requested: int, attempted: int, completed: int,
                  published_summary: dict | None = None) -> dict:
    n = protocol.EXPECTED_N
    eligible_n = sum(row.get("eligible") is True for row in preflight_rows)
    expected_forwards = 4*n + 2*eligible_n
    expected_counts = {name: {"requested": n if name in protocol.V4_BRANCHES else eligible_n,
                              "attempted": n if name in protocol.V4_BRANCHES else eligible_n,
                              "completed": n if name in protocol.V4_BRANCHES else eligible_n}
                       for name in protocol.ALL_BRANCHES}
    branch_schema = (isinstance(branch_counts, dict)
        and set(branch_counts) == set(protocol.ALL_BRANCHES)
        and all(isinstance(branch_counts[name], dict)
            and set(branch_counts[name]) == {"requested", "attempted", "completed"}
            and all(type(branch_counts[name][field]) is int
                    for field in ("requested", "attempted", "completed"))
            for name in protocol.ALL_BRANCHES))
    if (len(records) != n or len(preflight_rows) != n or len(old_rows) != n
            or type(requested) is not int or type(attempted) is not int or type(completed) is not int
            or requested != expected_forwards or attempted != expected_forwards
            or completed != expected_forwards or not branch_schema
            or branch_counts != expected_counts):
        raise ValueError("audit_full_denominator_or_branch_counts_invalid")
    d_annotation, d_placebo, specificity = [], [], []
    eligible_count = 0
    for result, geometry, old in zip(records, preflight_rows, old_rows, strict=True):
        binding = geometry.get("row_binding_sha256")
        if (result.get("row_binding_sha256") != binding
                or result.get("video_id") != geometry.get("video_id")
                or result.get("qid") != geometry.get("qid")
                or result.get("geometry_sha256") != geometry.get("geometry_sha256")
                or old.get("row_binding_sha256") != binding
                or type(result.get("answer_index")) is not int
                or type(result.get("foil_index")) is not int
                or result.get("answer_index") != old.get("answer_index")
                or result.get("foil_index") != old.get("foil_index")
                or result.get("specificity_eligible") is not geometry.get("eligible")):
            raise ValueError("audit_row_identity_or_frozen_prediction_mismatch")
        if result.get("historical_answer_foil_match") is not True:
            raise ValueError("audit_historical_prediction_flag_invalid")
        margin_error = _finite(result.get("historical_baseline_full_margin_abs_error"),
                               "audit_margin_error_invalid")
        if not 0 <= margin_error <= 1e-5:
            raise ValueError("audit_historical_margin_tolerance_exceeded")
        margins = result.get("margin_by_condition")
        if not isinstance(margins, dict) or set(margins) != set(protocol.V4_BRANCHES):
            raise ValueError("audit_annotation_margin_conditions_invalid")
        margins = {name: _finite(margins[name], "audit_margin_nonfinite")
                   for name in protocol.V4_BRANCHES}
        d_base = margins["baseline_full"] - margins["baseline_union_mask"]
        d_qta = margins["intervention_full"] - margins["intervention_union_mask"]
        annotation = d_qta - d_base
        _same_number(result.get("D_annotation_union"), annotation, "audit_D_annotation")
        d_annotation.append(annotation)
        if geometry["eligible"]:
            eligible_count += 1
            placebo_base = _finite(result.get("baseline_placebo_margin"), "audit_placebo_base_invalid")
            placebo_qta = _finite(result.get("intervention_placebo_margin"), "audit_placebo_qta_invalid")
            placebo = (margins["intervention_full"] - placebo_qta) - (
                margins["baseline_full"] - placebo_base)
            spec = annotation - placebo
            _same_number(result.get("D_placebo_translation"), placebo, "audit_D_placebo")
            _same_number(result.get("specificity_difference"), spec, "audit_specificity")
            d_placebo.append(placebo)
            specificity.append(spec)
        elif any(key in result for key in ("D_placebo_translation", "specificity_difference",
                                           "baseline_placebo_margin", "intervention_placebo_margin")):
            raise ValueError("audit_unavailable_row_has_placebo_score")
    if eligible_count != eligible_n or not eligible_n:
        raise ValueError("audit_eligible_denominator_invalid")
    summary = {
        "all_147_D_annotation_union": independent_summary(d_annotation),
        "eligible_D_annotation_union": independent_summary([x for x, row in zip(
            d_annotation, preflight_rows, strict=True) if row["eligible"]]),
        "eligible_D_placebo_translation": independent_summary(d_placebo),
        "eligible_specificity_difference": independent_summary(specificity),
    }
    if published_summary is not None:
        if (set(published_summary) != set(summary) | {"inferential_status"}
                or published_summary.get("inferential_status")
                   != "finite cohort descriptive only; no p-value or population interval"):
            raise ValueError("audit_published_summary_keys_invalid")
        for key, expected in summary.items():
            actual = published_summary[key]
            if set(actual) != set(expected):
                raise ValueError("audit_published_summary_fields_invalid")
            for field, value in expected.items():
                other = actual[field]
                if value is None:
                    if other is not None:
                        raise ValueError("audit_published_summary_null_mismatch")
                elif (type(other) not in (int, float) or isinstance(other, bool)
                      or not math.isclose(float(other), float(value), rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError("audit_published_summary_value_mismatch")
    return {"summary": summary, "eligible_rows": eligible_n,
            "unavailable_rows": n - eligible_n, "requested_forwards": expected_forwards}


def audit_private_result(repo: Path, score_name: str) -> dict:
    private = repo / PRIVATE_REL
    score_path = private / score_name
    if Path(score_name).name != score_name:
        raise ValueError("audit_private_score_receipt_missing")
    payload, score_sha256, score_mode = read_private_score(score_path, private)
    base, gate, plan, census = v4.load_v4_plan(repo)
    items = v4.selected_items(repo, base, plan)
    old_receipt = v4.private_json(private / protocol.V4_SCORE_NAME, private)
    old_rows, old_sha = protocol.validate_old_score_receipt(old_receipt, plan=plan,
        census=census, private=private)
    preflight = v4.private_json(private / protocol.PREFLIGHT_NAME, private)
    smoke = v4.private_json(private / protocol.SMOKE_NAME, private)
    source_hashes = {name: sha256_file(repo / name) for name in protocol.SOURCE_FILES}
    data_hashes = {name: sha256_file(repo / name) for name in protocol.V4_DATA_FILES}
    expected_header = protocol.build_preflight_header(
        processor_manifest=preflight["processor_manifest"], repo=repo,
        source_hashes=source_hashes, data_hashes=data_hashes, v4_score_sha256=old_sha)
    protocol.validate_preflight(preflight, expected_header)
    protocol.validate_preflight_bindings(preflight["rows"], items, old_rows)
    protocol.validate_smoke_receipt(smoke,
        preflight_sha256=sha256_file(private / protocol.PREFLIGHT_NAME), old_sha=old_sha,
        source_hashes=source_hashes, data_hashes=data_hashes,
        eligible_row=next(row for row in preflight["rows"] if row["eligible"]),
        old_row={row["row_binding_sha256"]: row for row in old_rows}[
            next(row for row in preflight["rows"] if row["eligible"])["row_binding_sha256"]])
    eligible_n = preflight["eligible_specificity_rows"]
    validate_completed_metadata(payload, eligible_n=eligible_n)
    if (payload.get("provenance", {}).get("source_sha256") != source_hashes
            or payload.get("provenance", {}).get("data_sha256") != data_hashes
            or payload.get("provenance", {}).get("v4_score_sha256") != old_sha
            or payload.get("provenance", {}).get("specificity_preflight_sha256")
               != sha256_file(private / protocol.PREFLIGHT_NAME)
            or payload.get("provenance", {}).get("specificity_smoke_sha256")
               != sha256_file(private / protocol.SMOKE_NAME)):
        raise ValueError("audit_full_provenance_invalid")
    result = audit_records(payload.get("records"), preflight_rows=preflight["rows"],
        old_rows=old_rows, branch_counts=payload.get("branch_counts"),
        requested=payload.get("requested_forwards"), attempted=payload.get("attempted_forwards"),
        completed=payload.get("completed_forwards"), published_summary=payload.get("summary"))
    if (payload.get("eligible_specificity_rows") != result["eligible_rows"]
            or payload.get("structurally_unavailable_rows") != result["unavailable_rows"]):
        raise ValueError("audit_reported_eligible_denominator_mismatch")
    result.update({"receipt_sha256": score_sha256,
        "receipt_mode": score_mode,
        "status": payload["status"], "effective_rows": payload.get("effective_rows"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "hardware": payload.get("hardware"), "rights_status": payload.get("rights_status"),
        "faithfulness_result": payload.get("faithfulness_result"),
        "inferential_status": payload.get("inferential_status")})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--score-name", required=True)
    args = parser.parse_args(argv)
    result = audit_private_result(args.repo.resolve(), args.score_name)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
