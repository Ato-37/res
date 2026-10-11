"""Structurally validate a proposed video faithfulness protocol (CPU only).

This validator checks declared fields and referenced hashes. It does not verify
that cited artifacts exist, that human judgments are genuine, that annotations
semantically match claims, that a control is scientifically adequate, or that a
runtime/cache policy is correct. It never grants execution authorization.

Usage::

    .venv/bin/python scripts/validate_faithfulness_protocol.py protocol.json

The input is private by default: stdout contains only validity, failure codes,
and the SHA-256 of the input bytes. No names, IDs, claims, or paths are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class DuplicateJsonKey(ValueError):
    """Raised when an object repeats a key instead of being unambiguously parsed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_exact_keys(value: Any, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _valid_hash_fields(value: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(_is_sha(value.get(field)) for field in fields)


def validate_protocol(value: Any, *, input_sha256: str | None = None) -> dict[str, Any]:
    """Validate schema v1 and return a privacy-safe structural decision only."""
    failures: set[str] = set()
    if not isinstance(value, dict):
        failures.add("protocol_not_object")
        return _summary(failures, input_sha256)

    required = {
        "schema_version", "protocol_stage", "model_revision", "tokenizer_revision",
        "cohort", "annotation", "sampling", "estimand", "intervention", "cache",
    }
    if set(value) != required:
        failures.add("protocol_schema_keys_invalid")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        failures.add("schema_version_invalid")
    stage = value.get("protocol_stage")
    if not isinstance(stage, str) or stage not in {"exploratory", "confirmatory"}:
        failures.add("protocol_stage_invalid")
    if not _is_nonempty_text(value.get("model_revision")):
        failures.add("model_revision_missing")
    if not _is_nonempty_text(value.get("tokenizer_revision")):
        failures.add("tokenizer_revision_missing")

    cohort = value.get("cohort")
    cohort_keys = {
        "primary_unit", "video_count", "record_count", "analysis_unit_count",
        "records_aggregated_within_video", "split_manifest_sha256",
        "overlap_audit_sha256", "split_overlap_count",
    }
    if not _has_exact_keys(cohort, cohort_keys):
        failures.add("cohort_schema_invalid")
    else:
        if cohort.get("primary_unit") != "video_id":
            failures.add("primary_unit_must_be_video")
        counts = (cohort["video_count"], cohort["record_count"],
                  cohort["analysis_unit_count"], cohort["split_overlap_count"])
        if not all(_is_count(count) for count in counts):
            failures.add("cohort_counts_invalid")
        elif (
            cohort["video_count"] == 0
            or cohort["record_count"] < cohort["video_count"]
            or cohort["analysis_unit_count"] != cohort["video_count"]
        ):
            failures.add("analysis_units_not_video_level")
        if type(cohort.get("records_aggregated_within_video")) is not bool:
            failures.add("within_video_aggregation_invalid")
        elif (
            _is_count(cohort.get("record_count"))
            and _is_count(cohort.get("video_count"))
            and cohort["record_count"] > cohort["video_count"]
            and cohort["records_aggregated_within_video"] is not True
        ):
            failures.add("repeated_video_records_not_aggregated")
        if not _valid_hash_fields(cohort, ("split_manifest_sha256", "overlap_audit_sha256")):
            failures.add("cohort_provenance_hash_invalid")
        if stage == "confirmatory" and cohort.get("split_overlap_count") != 0:
            failures.add("confirmatory_video_overlap")

    annotation = value.get("annotation")
    annotation_keys = {
        "locator_kind", "source_manifest_sha256", "time_unit", "time_origin",
        "evaluation_metric", "semantic_review_record_sha256", "point_conversion",
    }
    if not _has_exact_keys(annotation, annotation_keys):
        failures.add("annotation_schema_invalid")
    else:
        locator = annotation.get("locator_kind")
        if not isinstance(locator, str) or locator not in {"point", "interval"}:
            failures.add("annotation_locator_kind_invalid")
        if not _is_sha(annotation.get("source_manifest_sha256")):
            failures.add("annotation_source_hash_invalid")
        origin = annotation.get("time_origin")
        if annotation.get("time_unit") != "seconds" or not isinstance(origin, str) or origin not in {
            "video_start", "absolute_utc"
        }:
            failures.add("annotation_time_basis_invalid")
        metric = annotation.get("evaluation_metric")
        if not isinstance(metric, str) or metric not in {
            "point_absolute_error_seconds", "interval_iou", "interval_distance_seconds"
        }:
            failures.add("annotation_metric_invalid")
        review_hash = annotation.get("semantic_review_record_sha256")
        if review_hash is not None and not _is_sha(review_hash):
            failures.add("semantic_review_reference_invalid")
        conversion = annotation.get("point_conversion")
        if locator == "point" and isinstance(metric, str):
            if metric == "point_absolute_error_seconds":
                if conversion is not None:
                    failures.add("point_conversion_unneeded_or_ambiguous")
            elif metric in {"interval_iou", "interval_distance_seconds"}:
                conversion_keys = {
                    "rule_sha256", "derived_interval_artifact_sha256", "review_record_sha256"
                }
                if not _has_exact_keys(conversion, conversion_keys) or not _valid_hash_fields(
                    conversion, ("rule_sha256", "derived_interval_artifact_sha256", "review_record_sha256")
                ):
                    failures.add("point_used_as_interval_without_review_refs")
        elif locator == "interval" and isinstance(metric, str):
            if metric == "point_absolute_error_seconds":
                failures.add("interval_used_as_point")
            if conversion is not None:
                failures.add("interval_conversion_unexpected")
        if isinstance(metric, str) and metric in {"interval_iou", "interval_distance_seconds"} and not _is_sha(review_hash):
            failures.add("interval_semantic_review_reference_missing")

    sampling = value.get("sampling")
    sampling_keys = {"policy", "config_sha256", "uses_annotation_for_selection"}
    if not _has_exact_keys(sampling, sampling_keys):
        failures.add("sampling_schema_invalid")
    else:
        policy = sampling.get("policy")
        if not isinstance(policy, str) or policy not in {"whole_video_fixed", "annotation_conditioned", "model_selected"}:
            failures.add("sampling_policy_invalid")
        if not _is_sha(sampling.get("config_sha256")):
            failures.add("sampling_config_hash_invalid")
        if type(sampling.get("uses_annotation_for_selection")) is not bool:
            failures.add("sampling_annotation_flag_invalid")
        elif (
            stage == "confirmatory"
            and sampling["uses_annotation_for_selection"] is True
        ):
            failures.add("confirmatory_sampling_uses_annotation")
        if policy == "annotation_conditioned" and sampling.get("uses_annotation_for_selection") is not True:
            failures.add("conditioned_sampling_not_declared")

    estimand = value.get("estimand")
    estimand_keys = {
        "primary_estimand", "primary_score_target", "score_kind", "target_text_sha256",
        "gold_role", "claim_intervention_mapping_sha256",
    }
    if not _has_exact_keys(estimand, estimand_keys):
        failures.add("estimand_schema_invalid")
    else:
        target = estimand.get("primary_score_target")
        score = estimand.get("score_kind")
        kind = estimand.get("primary_estimand")
        valid_pairs = {
            ("claim_selective_dependence", "claim", "teacher_forced_logprob"),
            ("answer_dependence", "answer", "teacher_forced_logprob"),
            ("grounding_accuracy", "timestamp", "temporal_metric"),
            ("qa_accuracy", "answer", "accuracy"),
        }
        if not all(isinstance(item, str) for item in (kind, target, score)):
            failures.add("score_target_estimand_mismatch")
        elif (kind, target, score) not in valid_pairs:
            failures.add("score_target_estimand_mismatch")
        target_hash = estimand.get("target_text_sha256")
        if isinstance(target, str) and target in {"claim", "answer"} and not _is_sha(target_hash):
            failures.add("target_text_hash_invalid")
        if target == "timestamp" and target_hash is not None:
            failures.add("timestamp_target_text_hash_unexpected")
        if estimand.get("gold_role") != "evaluation_only":
            failures.add("gold_role_must_be_evaluation_only")
        mapping_hash = estimand.get("claim_intervention_mapping_sha256")
        if kind == "claim_selective_dependence":
            if not _is_sha(mapping_hash):
                failures.add("claim_intervention_mapping_reference_missing")
        elif mapping_hash is not None:
            failures.add("claim_intervention_mapping_reference_unexpected")

    intervention = value.get("intervention")
    intervention_keys = {
        "kind", "selector_sha256", "selector_gold_access", "control_protocol_sha256",
        "control_match_dimensions",
    }
    if not _has_exact_keys(intervention, intervention_keys):
        failures.add("intervention_schema_invalid")
    else:
        kind = intervention.get("kind")
        if not isinstance(kind, str) or kind != "internal_activation":
            failures.add("intervention_kind_invalid")
        if not _valid_hash_fields(intervention, ("selector_sha256", "control_protocol_sha256")):
            failures.add("intervention_provenance_hash_invalid")
        if type(intervention.get("selector_gold_access")) is not bool:
            failures.add("selector_gold_access_invalid")
        elif intervention["selector_gold_access"] is True:
            failures.add("selector_uses_gold")
        dimensions = intervention.get("control_match_dimensions")
        allowed_dimensions = {"layer", "token_count", "magnitude", "temporal_position", "question_independence"}
        if (
            not isinstance(dimensions, list)
            or any(not isinstance(item, str) or item not in allowed_dimensions for item in dimensions)
            or len(set(dimensions)) != len(dimensions)
        ):
            failures.add("control_match_dimensions_invalid")
        elif (
            isinstance(estimand, dict)
            and estimand.get("primary_estimand") == "claim_selective_dependence"
        ):
            if not {"layer", "token_count", "magnitude"}.issubset(set(dimensions)):
                failures.add("claim_control_matching_incomplete")

    cache = value.get("cache")
    cache_keys = {"policy", "use_cache", "decode_position_policy", "runtime_evidence_sha256", "runtime_source_version"}
    if not _has_exact_keys(cache, cache_keys):
        failures.add("cache_schema_invalid")
    else:
        policy = cache.get("policy")
        if not isinstance(policy, str):
            failures.add("cache_policy_invalid")
        if type(cache.get("use_cache")) is not bool:
            failures.add("cache_flag_invalid")
        if policy == "no_cache_full_sequence":
            if cache.get("use_cache") is not False or cache.get("decode_position_policy") is not None:
                failures.add("no_cache_policy_inconsistent")
            if cache.get("runtime_evidence_sha256") is not None or cache.get("runtime_source_version") is not None:
                failures.add("no_cache_runtime_reference_unexpected")
        elif policy == "cached_prefill_decode":
            if cache.get("use_cache") is not True:
                failures.add("cached_policy_inconsistent")
            decode_policy = cache.get("decode_position_policy")
            if not isinstance(decode_policy, str) or decode_policy not in {
                "explicit_native", "explicit_fixed", "runtime_default"
            }:
                failures.add("cached_decode_position_policy_missing")
            if not _is_sha(cache.get("runtime_evidence_sha256")) or not _is_nonempty_text(
                cache.get("runtime_source_version")
            ):
                failures.add("cached_runtime_evidence_missing")
        else:
            failures.add("cache_policy_invalid")

    return _summary(failures, input_sha256)


def _summary(failures: set[str], input_sha256: str | None) -> dict[str, Any]:
    if input_sha256 is not None and not _is_sha(input_sha256):
        failures.add("input_hash_invalid")
        input_sha256 = None
    result = {
        "structural_valid": not failures,
        "failure_codes": sorted(failures),
        "semantic_validation": "not_machine_verifiable",
        "execution_authorization": "not_granted",
    }
    if input_sha256 is not None:
        result["protocol_sha256"] = input_sha256
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path, help="Private protocol JSON input")
    args = parser.parse_args(argv)
    try:
        raw = args.protocol.read_bytes()
        input_sha = hashlib.sha256(raw).hexdigest()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKey, ValueError):
        result = _summary({"protocol_unreadable_or_invalid_json"}, None)
        print(json.dumps(result, sort_keys=True))
        return 2
    result = validate_protocol(value, input_sha256=input_sha)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["structural_valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
