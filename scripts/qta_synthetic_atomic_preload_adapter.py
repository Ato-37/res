"""Atomic CPU-only binding of caller-supplied QTA source/review bytes.

This adapter composes existing leaf validators for a fixed three-row fixture
and emits only hashes and aggregate status. Synthetic fixtures are used in
tests; caller input provenance is not authenticated. It is not a production
preflight: no reviewer identity, semantic validity, clock/frame correspondence,
or execution authority is authenticated.
It never reads files, loads a model, touches CUDA, forwards, or scores.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from qta_actual_prebaseline_diagnostic import (
    REVIEW_KEYS, SOURCE_KEYS, diagnose_qta_prebaseline,
)
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_prescore_mapping_binding import interval_list_sha256
from qta_prescore_review_row_adapter import adapt_review_bundle_to_qta_rows


ROW_COUNT = 3
MAX_INPUT_BYTES = 24_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DENIAL = {
    "model_load_permitted": False,
    "score_ready": False,
    "execution_authorization": "not_granted",
    "forward_count": 0,
    "score_calls": 0,
}
_FAILURE_CODES = frozenset({
    "source_input_schema_invalid", "source_input_type_invalid",
    "source_input_too_large", "review_input_schema_invalid",
    "review_input_type_invalid", "review_input_too_large",
    "atomic_leaf_validation_incomplete", "atomic_leaf_output_invalid",
    "atomic_join_denominator_invalid", "atomic_join_row_schema_invalid",
    "atomic_three_row_order_or_source_mismatch",
    "atomic_row_digest_input_invalid", "atomic_three_row_uniqueness_invalid",
    "atomic_binding_failed",
})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda _value: (_ for _ in ()).throw(
                          ValueError("nonfinite_json_value")))


def _blocked(code: str) -> dict[str, Any]:
    safe = code if isinstance(code, str) and code in _FAILURE_CODES else "atomic_binding_failed"
    return {
        "schema_version": 1,
        "status": "blocked_atomic_three_row_binding",
        "scope": "human_free_non_authorizing_adapter_only",
        "requested_rows": ROW_COUNT,
        "validated_rows": 0,
        "failure_code": safe,
        "source_input_sha256": None,
        "review_input_sha256": None,
        "ordered_row_binding_sha256": [],
        "cohort_binding_sha256": None,
        "human_identity_authenticated": False,
        "semantic_review_authenticated": False,
        "clock_frame_correspondence_authenticated": False,
        **_DENIAL,
    }


def _snapshot_inputs(inputs: Mapping[str, Any] | None, expected_keys: set[str],
                     label: str) -> dict[str, Any]:
    if not isinstance(inputs, Mapping) or set(inputs) != expected_keys:
        raise ValueError(label + "_input_schema_invalid")
    snapshot: dict[str, Any] = {}
    total = 0
    for key in sorted(expected_keys):
        value = inputs[key]
        if type(value) is bytes:
            total += len(value)
            snapshot[key] = value
        elif key.startswith("expected_") and type(value) is str:
            snapshot[key] = value
        else:
            raise ValueError(label + "_input_type_invalid")
    if total > MAX_INPUT_BYTES:
        raise ValueError(label + "_input_too_large")
    return snapshot


def _input_digest(snapshot: dict[str, Any]) -> str:
    entries = []
    for key in sorted(snapshot):
        value = snapshot[key]
        if type(value) is bytes:
            entries.append([key, "bytes", _sha(value)])
        else:
            entries.append([key, "pin", value])
    return _sha(_canonical(entries))


def bind_synthetic_atomic_inputs(*, actual_source_inputs: Mapping[str, Any] | None,
                                 review_bundle_inputs: Mapping[str, Any] | None
                                 ) -> dict[str, Any]:
    """Validate exact caller-supplied source/review bytes as one ordered 3-row unit.

    The leaf validators are rerun from the original byte inputs. Any missing,
    malformed, reordered, mismatched, or oversized input blocks all rows and
    omits all partial hashes. Successful hashes are pseudonymous and should
    remain local. The function cannot determine whether caller bytes are
    synthetic; it remains non-authorizing even if misused with other bytes.
    Synthetic fixtures are used in tests. This function cannot determine
    whether caller bytes are synthetic; it is non-authorizing regardless.
    """
    try:
        # Read caller-controlled mappings exactly once. All subsequent hashing,
        # validators, parsing, and row binding use these immutable byte/string
        # snapshots, never the original mappings.
        source_snapshot = _snapshot_inputs(actual_source_inputs, SOURCE_KEYS, "source")
        review_snapshot = _snapshot_inputs(review_bundle_inputs, REVIEW_KEYS, "review")
        source_digest = _input_digest(source_snapshot)
        review_digest = _input_digest(review_snapshot)

        diagnostic = diagnose_qta_prebaseline(
            actual_source_inputs=source_snapshot,
            review_bundle_inputs=review_snapshot)
        gates = diagnostic.get("gates") if isinstance(diagnostic, dict) else None
        required = ("actual_qa_source_binding", "review_bundle_byte_structure",
                    "source_to_review_ordered_join")
        if (not isinstance(gates, dict)
                or any(not isinstance(gates.get(name), dict)
                       or gates[name].get("status") != "passed"
                       or type(gates[name].get("validated_rows")) is not int
                       or gates[name]["validated_rows"] != ROW_COUNT for name in required)
                or diagnostic.get("requested_rows") != ROW_COUNT
                or diagnostic.get("status") != "diagnostic_incomplete"):
            return _blocked("atomic_leaf_validation_incomplete")

        # Re-run the same leaf APIs to obtain their private, in-memory row
        # objects. These are never returned or serialized by this adapter.
        source_result = bind_qta_actual_qa_source(**source_snapshot)
        review_rows = adapt_review_bundle_to_qta_rows(**review_snapshot)
        source_summary = (source_result.get("summary")
                          if isinstance(source_result, dict) else None)
        private_bundle = source_result.get("private_bundle")
        if (not isinstance(source_summary, dict)
                or source_summary.get("status") != "bound_unvalidated"
                or type(source_summary.get("validated_rows")) is not int
                or source_summary["validated_rows"] != ROW_COUNT
                or source_summary.get("model_load_permitted") is not False
                or source_summary.get("scoring_authorized") is not False
                or source_summary.get("score_calls") != 0
                or not isinstance(private_bundle, dict)
                or private_bundle.get("classification")
                   != "private_local_only_unvalidated_qa_source"
                or not isinstance(private_bundle.get("raw_json"), bytes)
                or private_bundle.get("sha256") != _sha(private_bundle["raw_json"])
                or len(review_rows) != ROW_COUNT):
            return _blocked("atomic_leaf_output_invalid")
        private = _strict_json(private_bundle["raw_json"])
        source_claims = _strict_json(source_snapshot["claim_manifest_raw"])
        selections = _strict_json(source_snapshot["selection_raw"])
        source_lineage = _strict_json(source_snapshot["lineage_manifest_raw"])
        worksheet = _strict_json(review_snapshot["worksheet_raw"])
        review = _strict_json(review_snapshot["review_raw"])
        collections = (
            private.get("review_source_crosswalk"), source_claims.get("records"),
            selections.get("records"), source_lineage.get("records"),
            worksheet.get("records"), review.get("records"), review_rows,
        )
        if any(not isinstance(rows, list) or len(rows) != ROW_COUNT
               for rows in collections):
            return _blocked("atomic_join_denominator_invalid")

        bindings = []
        seen_tokens = set()
        for ordinal, rows in enumerate(zip(*collections, strict=True), start=1):
            cross, claim, selection, lineage, worksheet_row, review_row, adapted = rows
            if any(not isinstance(row, dict) for row in rows):
                return _blocked("atomic_join_row_schema_invalid")
            token = adapted.get("review_token")
            claim_text_sha = claim.get("claim_text_sha256")
            video_sha = lineage.get("video_sha256")
            if (not isinstance(token, str) or not token or token in seen_tokens
                    or token != selection.get("review_token")
                    or token != worksheet_row.get("review_token")
                    or token != review_row.get("review_token")
                    or token != cross.get("review_token")
                    or selection.get("video_id") != cross.get("video_id")
                    or selection.get("claim_id") != cross.get("claim_id")
                    or selection.get("dataset_video_id") != cross.get("dataset_video_id")
                    or selection.get("dataset_question_id") != cross.get("dataset_question_id")
                    or claim.get("video_id") != cross.get("video_id")
                    or claim.get("claim_id") != cross.get("claim_id")
                    or worksheet_row.get("video_id") != cross.get("video_id")
                    or worksheet_row.get("claim_id") != cross.get("claim_id")
                    or worksheet_row.get("claim_text_sha256") != claim_text_sha
                    or not isinstance(video_sha, str) or not _SHA256.fullmatch(video_sha)
                    or worksheet_row.get("video_sha256") != video_sha
                    or adapted.get("source_video_sha256") != video_sha
                    or review_row.get("review", {}).get("support_decision")
                       != adapted.get("support_decision")):
                return _blocked("atomic_three_row_order_or_source_mismatch")
            seen_tokens.add(token)
            intervals_sha = interval_list_sha256(adapted.get("evidence_intervals_seconds"))
            descriptor = {
                "ordinal": ordinal,
                "review_token_sha256": _sha(token.encode("utf-8")),
                "video_sha256": video_sha,
                "claim_text_sha256": claim_text_sha,
                "claim_id_sha256": _sha(str(claim.get("claim_id")).encode("utf-8")),
                "dataset_video_id_sha256": _sha(
                    str(lineage.get("dataset_video_id")).encode("utf-8")),
                "dataset_question_id_sha256": _sha(
                    str(lineage.get("dataset_question_id")).encode("utf-8")),
                "review_payload_sha256": adapted.get("review_payload_sha256"),
                "reviewed_intervals_sha256": intervals_sha,
                "support_decision": adapted.get("support_decision"),
            }
            if (not isinstance(descriptor["review_payload_sha256"], str)
                    or not _SHA256.fullmatch(descriptor["review_payload_sha256"])
                    or not isinstance(claim_text_sha, str)
                    or not _SHA256.fullmatch(claim_text_sha)):
                return _blocked("atomic_row_digest_input_invalid")
            bindings.append(_sha(_canonical(descriptor)))

        if len(seen_tokens) != ROW_COUNT or len(set(bindings)) != ROW_COUNT:
            return _blocked("atomic_three_row_uniqueness_invalid")
        cohort_sha = _sha(_canonical(bindings))
        return {
            "schema_version": 1,
            "status": "three_row_structural_binding_passed_unvalidated",
            "scope": "human_free_non_authorizing_adapter_only",
            "requested_rows": ROW_COUNT,
            "validated_rows": ROW_COUNT,
            "failure_code": None,
            "source_input_sha256": source_digest,
            "review_input_sha256": review_digest,
            "ordered_row_binding_sha256": bindings,
            "cohort_binding_sha256": cohort_sha,
            "human_identity_authenticated": False,
            "semantic_review_authenticated": False,
            "clock_frame_correspondence_authenticated": False,
            "input_classification": "caller_supplied_fixture_not_independently_authenticated",
            "rights_status": "unverified_local_only",
            **_DENIAL,
        }
    except Exception:
        # Never serialize exception text from arbitrary input bytes/mappings.
        return _blocked("atomic_binding_failed")
