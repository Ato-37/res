"""Non-authorizing structural join for QTA review-to-processor mappings.

This CPU-only helper checks that independently produced row artifacts agree
on token, video hash, first-stage receipt, interval list, calibration, mapped
PTS intervals, processor batch, and evidence/control groups. It does not
authenticate any input, establish human truth, load a model, or authorize a
forward. All three requested rows remain the denominator on any failure.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Sequence

from annotation_clock_calibration import map_annotation_intervals_to_pts


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{field}_invalid")
    return value


def _fraction(value: Any, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError(f"{field}_invalid")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field}_invalid")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"{field}_invalid") from None


def interval_list_sha256(intervals: Any) -> str:
    """Hash the canonical UTF-8 JSON encoding of ordered review intervals."""
    if (not isinstance(intervals, list) or not intervals
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in intervals)):
        raise ValueError("reviewed_intervals_invalid")
    previous_end = Fraction(-1)
    seen_intervals: set[tuple[Fraction, Fraction]] = set()
    normalized = []
    for pair in intervals:
        start, end = (_fraction(pair[0], "reviewed_interval"),
                      _fraction(pair[1], "reviewed_interval"))
        current = (start, end)
        duplicate = current in seen_intervals
        if (start < 0 or end <= start
                or any((value * 1000).denominator != 1 for value in (start, end))
                or (not duplicate and start <= previous_end)):
            raise ValueError("reviewed_intervals_invalid")
        seen_intervals.add(current)
        previous_end = max(previous_end, end)
        normalized.append([str(start), str(end)])
    try:
        # Reduced rational-string arrays avoid JSON int/float rendering
        # differences and preserve exact millisecond endpoints.
        raw = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    except (UnicodeError, TypeError, ValueError, OverflowError):
        raise ValueError("reviewed_intervals_invalid") from None
    return hashlib.sha256(raw).hexdigest()


def audit_qta_prescore_mapping_bundle(
    *, expected_cohort_tokens: Sequence[str],
    expected_video_review_receipt_sha256: str,
    expected_review_payload_sha256: str,
    expected_video_sha256_by_token: dict[str, str],
    expected_processor_batch_sha256_by_token: dict[str, str],
    expected_sampling_metadata_sha256_by_token: dict[str, str],
    expected_processor_groups_by_token: dict[str, dict[str, list[int]]],
    expected_calibration_sha256_by_token: dict[str, str],
    expected_baseline_source_sha256_by_token: dict[str, dict[str, str]],
    review_rows: Sequence[dict], clock_rows: Sequence[dict],
    processor_rows: Sequence[dict], baseline_rows: Sequence[dict],
) -> dict:
    """Check a three-row QTA evidence-to-processor structural join.

    The caller supplies outputs from the existing review, clock, batch, and
    baseline auditors plus independent source/batch digest pins. Successful
    output means only that these supplied records agree structurally. The
    helper emits no user text or raw intervals and always denies execution.
    """
    if (not isinstance(expected_cohort_tokens, Sequence)
            or isinstance(expected_cohort_tokens, (str, bytes))):
        raise ValueError("expected_three_unique_cohort_tokens_required")
    tokens = tuple(expected_cohort_tokens)
    if (len(tokens) != 3 or any(not isinstance(token, str) or not token for token in tokens)
            or len(set(tokens)) != 3):
        raise ValueError("expected_three_unique_cohort_tokens_required")
    review_receipt_sha = _digest(expected_video_review_receipt_sha256,
                                 "video_review_receipt_sha256")
    review_payload_sha = _digest(expected_review_payload_sha256,
                                 "review_payload_sha256")
    if (not isinstance(expected_video_sha256_by_token, dict)
            or set(expected_video_sha256_by_token) != set(tokens)
            or not isinstance(expected_processor_batch_sha256_by_token, dict)
            or set(expected_processor_batch_sha256_by_token) != set(tokens)
            or not isinstance(expected_sampling_metadata_sha256_by_token, dict)
            or set(expected_sampling_metadata_sha256_by_token) != set(tokens)
            or not isinstance(expected_processor_groups_by_token, dict)
            or set(expected_processor_groups_by_token) != set(tokens)
            or not isinstance(expected_calibration_sha256_by_token, dict)
            or set(expected_calibration_sha256_by_token) != set(tokens)
            or not isinstance(expected_baseline_source_sha256_by_token, dict)
            or set(expected_baseline_source_sha256_by_token) != set(tokens)):
        raise ValueError("expected_binding_inventory_invalid")
    expected_video_hashes = {token: _digest(expected_video_sha256_by_token[token],
                                             "expected_video_sha256") for token in tokens}
    expected_batch_hashes = {token: _digest(expected_processor_batch_sha256_by_token[token],
                                             "expected_processor_batch_sha256") for token in tokens}
    expected_sampling_hashes = {token: _digest(expected_sampling_metadata_sha256_by_token[token],
                                                "expected_sampling_metadata_sha256") for token in tokens}
    expected_groups = {}
    for token in tokens:
        expected = expected_processor_groups_by_token[token]
        if (not isinstance(expected, dict)
                or set(expected) != {"evidence_group_indices", "control_group_indices",
                                     "video_grid_thw"}):
            raise ValueError("expected_processor_groups_invalid")
        expected_evidence = _group_indices(expected["evidence_group_indices"],
                                           "expected_evidence_groups")
        expected_control = _group_indices(expected["control_group_indices"],
                                          "expected_control_groups")
        expected_grid = expected["video_grid_thw"]
        if (not isinstance(expected_grid, list) or len(expected_grid) != 3
                or any(type(value) is not int or value <= 0 for value in expected_grid)):
            raise ValueError("expected_processor_groups_invalid")
        expected_groups[token] = (expected_evidence, expected_control, expected_grid)
    expected_calibration_hashes = {token: _digest(expected_calibration_sha256_by_token[token],
                                                   "expected_calibration_sha256") for token in tokens}
    source_fields = ("question_sha256", "options_sha256", "claim_sha256")
    expected_source_hashes = {}
    for token in tokens:
        pins = expected_baseline_source_sha256_by_token[token]
        if not isinstance(pins, dict) or set(pins) != set(source_fields):
            raise ValueError("expected_baseline_source_binding_invalid")
        expected_source_hashes[token] = {
            field: _digest(pins[field], f"expected_{field}") for field in source_fields
        }
    collections = (review_rows, clock_rows, processor_rows, baseline_rows)
    if any(not isinstance(items, Sequence) or isinstance(items, (str, bytes))
           or len(items) != 3 for items in collections):
        return _blocked("cohort_row_count_mismatch")

    row_digests = []
    for index, (review, clock, processor, baseline) in enumerate(
            zip(*collections, strict=True)):
        token = tokens[index]
        try:
            if any(not isinstance(row, dict) for row in (review, clock, processor, baseline)):
                raise ValueError("row_artifact_invalid")
            if any(row.get("review_token") != token
                   for row in (review, clock, processor, baseline)):
                raise ValueError("review_token_binding_mismatch")
            video_hash = expected_video_hashes[token]
            if any(_digest(row.get("source_video_sha256"), "source_video_sha256") != video_hash
                   for row in (review, clock, processor, baseline)):
                raise ValueError("source_video_binding_mismatch")
            if _digest(review.get("review_payload_sha256"), "review_payload_sha256") != review_payload_sha:
                raise ValueError("review_payload_binding_mismatch")
            if review.get("support_decision") != "supports_claim":
                raise ValueError("video_claim_not_supported")
            if (baseline.get("video_claim_review_receipt_sha256") != review_receipt_sha
                    or clock.get("video_claim_review_receipt_sha256") != review_receipt_sha):
                raise ValueError("video_review_receipt_binding_mismatch")

            intervals = review.get("evidence_intervals_seconds")
            interval_sha = interval_list_sha256(intervals)
            if (baseline.get("video_support_decision") != "supports_claim"
                    or _digest(baseline.get("reviewed_intervals_sha256"),
                               "reviewed_intervals_sha256") != interval_sha):
                raise ValueError("reviewed_interval_digest_mismatch")

            calibration_sha = _digest(clock.get("calibration_sha256"),
                                      "calibration_sha256")
            calibration_payload = clock.get("calibration_binding_payload")
            calibration_fields = {
                "source_video_sha256", "slope", "intercept",
                "anchor_pts_min_seconds", "anchor_pts_max_seconds",
                "guard_pts_seconds",
            }
            if (not isinstance(calibration_payload, dict)
                    or set(calibration_payload) != calibration_fields
                    or calibration_payload.get("source_video_sha256") != video_hash):
                raise ValueError("calibration_payload_invalid")
            for field in ("slope", "intercept", "anchor_pts_min_seconds",
                          "anchor_pts_max_seconds", "guard_pts_seconds"):
                value = calibration_payload[field]
                if (not isinstance(value, str)
                        or str(_fraction(value, f"calibration_{field}")) != value):
                    raise ValueError("calibration_payload_invalid")
            recomputed_calibration_sha = hashlib.sha256(json.dumps(
                calibration_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
            if recomputed_calibration_sha != calibration_sha:
                raise ValueError("calibration_payload_digest_mismatch")
            if calibration_sha != expected_calibration_hashes[token]:
                raise ValueError("calibration_binding_mismatch")
            slope, intercept = clock.get("slope"), clock.get("intercept")
            if (_fraction(calibration_payload["slope"], "calibration_slope")
                    != _fraction(slope, "clock_slope")
                    or _fraction(calibration_payload["intercept"], "calibration_intercept")
                    != _fraction(intercept, "clock_intercept")):
                raise ValueError("calibration_clock_value_mismatch")
            mapped = map_annotation_intervals_to_pts(intervals, slope, intercept)
            anchor_min = _fraction(clock.get("anchor_pts_min_seconds"), "anchor_pts_min")
            anchor_max = _fraction(clock.get("anchor_pts_max_seconds"), "anchor_pts_max")
            guard = _fraction(clock.get("guard_pts_seconds"), "guard_pts_seconds")
            if (_fraction(calibration_payload["anchor_pts_min_seconds"], "calibration_anchor_min")
                    != anchor_min
                    or _fraction(calibration_payload["anchor_pts_max_seconds"], "calibration_anchor_max")
                    != anchor_max
                    or _fraction(calibration_payload["guard_pts_seconds"], "calibration_guard")
                    != guard):
                raise ValueError("calibration_clock_value_mismatch")
            if anchor_min < 0 or anchor_max <= anchor_min or guard < 0:
                raise ValueError("clock_span_or_guard_invalid")
            if any(Fraction(start) < anchor_min or Fraction(end) > anchor_max
                   for start, end in mapped):
                raise ValueError("review_interval_outside_anchor_span")

            processor_interval = processor.get("reviewed_intervals_rational_seconds")
            if processor_interval != mapped:
                raise ValueError("processor_interval_mapping_mismatch")
            if _fraction(processor.get("interval_boundary_guard_rational_seconds"),
                         "processor_guard") != guard:
                raise ValueError("processor_clock_guard_mismatch")
            batch_sha = _digest(processor.get("typed_batch_sha256"), "typed_batch_sha256")
            if batch_sha != expected_batch_hashes[token]:
                raise ValueError("processor_batch_binding_mismatch")
            sampling_sha = _digest(processor.get("sampling_metadata_sha256"),
                                   "sampling_metadata_sha256")
            if sampling_sha != expected_sampling_hashes[token]:
                raise ValueError("sampling_metadata_binding_mismatch")
            evidence = _group_indices(processor.get("evidence_group_indices"), "evidence_groups")
            control = _group_indices(processor.get("control_group_indices"), "control_groups")
            grid = processor.get("video_grid_thw")
            if (not isinstance(grid, list) or len(grid) != 3
                    or any(type(value) is not int or value <= 0 for value in grid)):
                raise ValueError("processor_video_grid_invalid")
            temporal_group_count = grid[0]
            if any(index >= temporal_group_count for index in (*evidence, *control)):
                raise ValueError("processor_group_index_out_of_range")
            if len(evidence) != len(control) or set(evidence) & set(control):
                raise ValueError("evidence_control_group_contract_invalid")
            if (evidence, control, grid) != expected_groups[token]:
                raise ValueError("processor_group_mapping_mismatch")
            payload = {
                "review_token": token,
                "source_video_sha256": video_hash,
                "video_review_receipt_sha256": review_receipt_sha,
                "review_payload_sha256": review_payload_sha,
                "reviewed_intervals_sha256": interval_sha,
                "calibration_sha256": calibration_sha,
                "mapped_pts_intervals": mapped,
                "guard_pts_seconds": str(guard),
                "typed_batch_sha256": batch_sha,
                "sampling_metadata_sha256": sampling_sha,
                "video_grid_thw": grid,
                "evidence_group_indices": evidence,
                "control_group_indices": control,
            }
            for field in source_fields:
                observed = _digest(baseline.get(field), field)
                if observed != expected_source_hashes[token][field]:
                    raise ValueError("baseline_source_binding_mismatch")
                payload[f"baseline_{field}"] = observed
            row_digests.append(hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest())
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            return _blocked(str(exc), failed_row=index + 1,
                            verified_rows=len(row_digests))
    aggregate = hashlib.sha256(json.dumps(
        row_digests, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    return {
        "status": "structurally_bound_pending_independent_score_freeze",
        "requested_records": 3,
        "verified_structural_rows": 3,
        "row_binding_sha256": row_digests,
        "bundle_binding_sha256": aggregate,
        "model_load_permitted": False,
        "branch_forwards": 0,
        "execution_authorization": "not_granted",
        "reviewer_identity_verified": False,
        "semantic_review_authenticated": False,
    }


def _group_indices(value: Any, field: str) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(type(index) is not int or index < 0 for index in value)
            or value != sorted(set(value))):
        raise ValueError(f"{field}_invalid")
    return value


def _blocked(code: str, *, failed_row: int | None = None,
             verified_rows: int = 0) -> dict:
    return {
        "status": "blocked",
        "failure_code": code if code.isascii() and code.replace("_", "").isalnum()
        else "row_binding_invalid",
        "failed_row_ordinal": failed_row,
        "requested_records": 3,
        "verified_structural_rows": verified_rows,
        "model_load_permitted": False,
        "branch_forwards": 0,
        "execution_authorization": "not_granted",
    }
