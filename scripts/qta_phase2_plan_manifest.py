"""Non-authorizing schema for the signed phase-2 QTA plan artifact.

The manifest structurally joins the existing nine-branch plans to the exact
three record tokens and processor-batch hashes in both signed phase receipts.
It does not authenticate the meaning or provenance of those plans, human
judgments, model inputs, or outcomes, and it never authorizes inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from qta_nine_branch_plan import BRANCH_ORDER, validate_qta_nine_branch_plan
from qta_phase_freeze_receipts import verify_phase_freeze_receipt
from qta_three_row_plan_aggregate import (
    FROZEN_QTA_LAYERS_1_BASED, FROZEN_QTA_LOGIT_BIAS, FROZEN_QTA_SHAM_SEED,
)
from qta_selection_provenance import (
    FROZEN_SELECTOR_SOURCE_SHA256, validate_qta_selection_provenance,
)
from qta_attention_capture_sidecars import validate_qta_capture_sidecar_index


ROW_COUNT = 3
PHASE2_FORWARDS = ROW_COUNT * (len(BRANCH_ORDER) - 1)
MAX_MANIFEST_BYTES = 32_000_000
MAX_UPSTREAM_BYTES = 24_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
_MANIFEST_KEYS = {
    "schema_version", "purpose", "phase", "phase1_receipt_sha256",
    "baseline_pq_manifest_sha256", "expected_attention_source_dtype",
    "requested_records", "planned_forwards", "completed_forwards",
    "selection_arithmetic_reproduced_records",
    "forward_source_authenticated", "attention_capture_authenticated",
    "records", "execution_authorization", "model_load_permitted",
    "scoring_authorized", "authorized_forwards",
}
_ROW_KEYS = {
    "record_position", "record_token", "processor_batch_sha256",
    "row_binding_sha256", "baseline_pq_row_sha256", "baseline_p_index",
    "baseline_q_index", "score_binding_sha256", "plan",
}
_BASELINE_KEYS = {
    "schema_version", "purpose", "video_claim_review_receipt_sha256", "records",
}
_BASELINE_ROW_KEYS = {
    "review_token", "question", "options", "claim", "option_log_probabilities",
    "p_index", "q_index", "video_support_decision", "reviewed_intervals_sha256",
    "score_binding", "score_binding_sha256",
}
_SCORE_BINDING_KEYS = {
    "schema_version", "review_token", "answer_label_order", "answer_token_ids",
    "choice_token_binding_sha256", "question_sha256", "options_sha256",
    "claim_sha256", "source_video_sha256", "reviewed_intervals_sha256",
    "video_support_decision",
    "prompt_sha256", "input_ids_sha256",
    "input_length", "score_position", "score_input_token_id",
    "video_token_id", "question_positions", "question_positions_sha256",
    "question_position_mapper_sha256",
    "processor_batch_sha256", "option_log_probabilities", "p_index", "q_index",
    "run_digests",
}
_RUN_DIGEST_KEYS = {
    "protocol_sha256", "model_sha256", "tokenizer_sha256",
    "scorer_sha256", "processor_sha256",
}
_STAGE1_SOURCE_PIN_KEYS = {
    "schema_version", "purpose", "classification", "status",
    "requested_records", "effective_records", "source_rows",
    "expected_baseline_source_sha256_by_token",
    "cohort_review_source_pins_by_token",
    "expected_baseline_candidate_source_bindings_by_token",
    "expected_cohort_tokens", "expected_video_claim_review_receipt_sha256",
    "expected_review_payload_sha256", "stage1_audit_status",
    "actual_qa_source_status", "stage1_source_bytes_sha256",
    "private_qa_bundle_sha256", "reviewer_identity_verified",
    "semantic_review_authenticated", "clock_frame_correspondence_verified",
    "scoring_authorized", "limitations", "model_load_permitted",
    "execution_authorization",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _canonical_utf8(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes) -> Any:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda _value: (_ for _ in ()).throw(
                          ValueError("nonfinite_json_value")))


def _record_rows(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != ROW_COUNT:
        raise ValueError("phase_record_count_invalid")
    output = []
    seen = set()
    for row in value:
        if (not isinstance(row, Mapping)
                or set(row) != {"record_token", "processor_batch_sha256"}):
            raise ValueError("phase_record_binding_invalid")
        token = row["record_token"]
        batch = row["processor_batch_sha256"]
        if not isinstance(token, str) or not _TOKEN.fullmatch(token) or token in seen:
            raise ValueError("phase_record_token_invalid_or_duplicate")
        if not isinstance(batch, str) or not _SHA256.fullmatch(batch):
            raise ValueError("phase_processor_batch_digest_invalid")
        seen.add(token)
        output.append({"record_token": token, "processor_batch_sha256": batch})
    return output


def _phase1_preflight_row_hashes(raw: bytes) -> list[str]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_UPSTREAM_BYTES:
        raise ValueError("phase1_preflight_bytes_invalid")
    report = _strict_json(raw)
    keys = {
        "status", "scope", "data_provenance_status", "requested_records",
        "verified_structural_rows", "failure_code", "row_binding_sha256",
        "bundle_binding_sha256", "clock_joint_freeze_authenticated",
        "prediction_link_authentication", "model_load_permitted", "branch_forwards",
        "scoring_authorization", "execution_authorization",
    }
    if not isinstance(report, dict) or set(report) != keys:
        raise ValueError("phase1_preflight_report_schema_invalid")
    row_hashes = report["row_binding_sha256"]
    if (report["status"] != "structural_row_binding_passed_unvalidated"
            or report["scope"] != "structural_row_binding_integration_only"
            or report["data_provenance_status"] != "not_verified_by_this_preflight"
            or type(report["requested_records"]) is not int or report["requested_records"] != ROW_COUNT
            or type(report["verified_structural_rows"]) is not int
            or report["verified_structural_rows"] != ROW_COUNT
            or report["failure_code"] is not None
            or not isinstance(row_hashes, list) or len(row_hashes) != ROW_COUNT
            or any(not isinstance(value, str) or not _SHA256.fullmatch(value)
                   for value in row_hashes)
            or report["clock_joint_freeze_authenticated"] is not False
            or report["prediction_link_authentication"] != "self_asserted_unauthenticated"
            or report["model_load_permitted"] is not False
            or type(report["branch_forwards"]) is not int or report["branch_forwards"] != 0
            or report["scoring_authorization"] != "not_granted"
            or report["execution_authorization"] != "not_granted"):
        raise ValueError("phase1_preflight_report_not_complete_non_authorizing")
    expected_bundle = _sha(json.dumps(
        row_hashes, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii"))
    if report["bundle_binding_sha256"] != expected_bundle:
        raise ValueError("phase1_preflight_bundle_digest_mismatch")
    return list(row_hashes)


def _phase1_stage1_source_pins(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_UPSTREAM_BYTES:
        raise ValueError("phase1_stage1_source_pin_bytes_invalid")
    artifact = _strict_json(raw)
    if (not isinstance(artifact, dict)
            or set(artifact) != _STAGE1_SOURCE_PIN_KEYS
            or type(artifact.get("schema_version")) is not int
            or artifact["schema_version"] != 1
            or artifact.get("purpose") !=
                "qta_stage1_source_pins_structurally_derived_non_authorizing"
            or artifact.get("status") != "three_row_source_pins_structurally_derived"
            or type(artifact.get("requested_records")) is not int
            or artifact["requested_records"] != ROW_COUNT
            or type(artifact.get("effective_records")) is not int
            or artifact["effective_records"] != ROW_COUNT
            or artifact.get("reviewer_identity_verified") is not False
            or artifact.get("semantic_review_authenticated") is not False
            or artifact.get("clock_frame_correspondence_verified") is not False
            or artifact.get("scoring_authorized") is not False
            or artifact.get("model_load_permitted") is not False
            or artifact.get("execution_authorization") != "not_granted"):
        raise ValueError("phase1_stage1_source_pin_artifact_invalid")
    receipt_sha = artifact.get("expected_video_claim_review_receipt_sha256")
    tokens = artifact.get("expected_cohort_tokens")
    rows = artifact.get("source_rows")
    if (not isinstance(receipt_sha, str) or not _SHA256.fullmatch(receipt_sha)
            or not isinstance(tokens, list) or len(tokens) != ROW_COUNT
            or any(not isinstance(token, str) or not _REVIEW_TOKEN.fullmatch(token)
                   for token in tokens)
            or len(set(tokens)) != ROW_COUNT
            or not isinstance(rows, list) or len(rows) != ROW_COUNT):
        raise ValueError("phase1_stage1_source_pin_rows_invalid")
    row_keys = {
        "review_token", "question_sha256", "options_sha256", "claim_sha256",
        "source_video_sha256", "video_support_decision", "reviewed_intervals_sha256",
    }
    for token, row in zip(tokens, rows, strict=True):
        if (not isinstance(row, dict) or set(row) != row_keys
                or row.get("review_token") != token
                or row.get("video_support_decision") != "supports_claim"
                or any(not isinstance(row.get(field), str)
                       or not _SHA256.fullmatch(row[field])
                       for field in row_keys - {"review_token", "video_support_decision"})):
            raise ValueError("phase1_stage1_source_pin_row_invalid")
    return {"video_claim_review_receipt_sha256": receipt_sha,
            "records": rows}


def _baseline_rows(raw: bytes) -> list[dict[str, Any]]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("baseline_pq_manifest_bytes_invalid")
    manifest = _strict_json(raw)
    if (not isinstance(manifest, dict) or set(manifest) != _BASELINE_KEYS
            or _canonical_utf8(manifest) != raw
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 3
            or manifest["purpose"] != "qta_baseline_prediction_link_manifest_v3"
            or not isinstance(manifest["video_claim_review_receipt_sha256"], str)
            or not _SHA256.fullmatch(manifest["video_claim_review_receipt_sha256"])):
        raise ValueError("baseline_pq_manifest_schema_invalid")
    rows = manifest["records"]
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ValueError("baseline_pq_record_count_invalid")
    checked = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _BASELINE_ROW_KEYS:
            raise ValueError("baseline_pq_record_schema_invalid")
        token = row["review_token"]
        scores = row["option_log_probabilities"]
        if not isinstance(token, str) or not _REVIEW_TOKEN.fullmatch(token) or token in seen:
            raise ValueError("baseline_pq_record_token_invalid_or_duplicate")
        if (not isinstance(scores, list) or len(scores) != 5
                or any(isinstance(score, bool) or not isinstance(score, float)
                       or not math.isfinite(score) or score > 0 for score in scores)
                or math.fsum(math.exp(score) for score in scores) > 1.0 + 1e-12):
            raise ValueError("baseline_pq_scores_invalid")
        p_index = min(range(5), key=lambda index: (-scores[index], index))
        q_index = min((index for index in range(5) if index != p_index),
                      key=lambda index: (-scores[index], index))
        binding = row["score_binding"]
        if (not isinstance(binding, dict) or set(binding) != _SCORE_BINDING_KEYS
                or type(binding["schema_version"]) is not int
                or binding["schema_version"] != 2
                or binding["review_token"] != token
                or binding["answer_label_order"] != ["A", "B", "C", "D", "E"]
                or not isinstance(binding["answer_token_ids"], list)
                or len(binding["answer_token_ids"]) != 5
                or any(type(value) is not int or value < 0
                       for value in binding["answer_token_ids"])
                or len(set(binding["answer_token_ids"])) != 5
                or not isinstance(binding["choice_token_binding_sha256"], str)
                or not _SHA256.fullmatch(binding["choice_token_binding_sha256"])
                or not isinstance(binding["run_digests"], dict)
                or set(binding["run_digests"]) != _RUN_DIGEST_KEYS
                or any(not isinstance(value, str) or not _SHA256.fullmatch(value)
                       for value in binding["run_digests"].values())):
            raise ValueError("baseline_pq_score_binding_schema_invalid")
        for field in ("prompt_sha256", "input_ids_sha256", "processor_batch_sha256"):
            if not isinstance(binding[field], str) or not _SHA256.fullmatch(binding[field]):
                raise ValueError("baseline_pq_score_binding_digest_invalid")
        for field in ("question_sha256", "options_sha256", "claim_sha256",
                      "source_video_sha256", "reviewed_intervals_sha256"):
            if not isinstance(binding[field], str) or not _SHA256.fullmatch(binding[field]):
                raise ValueError("baseline_pq_score_binding_source_digest_invalid")
        actual_source_digests = {
            "question_sha256": _sha(row["question"].encode("utf-8")),
            "options_sha256": _sha(_canonical_utf8(row["options"])),
            "claim_sha256": _sha(row["claim"].encode("utf-8")),
            "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
        }
        if (any(binding[field] != digest for field, digest
                in actual_source_digests.items())
                or binding["source_video_sha256"] == ""):
            raise ValueError("baseline_pq_score_binding_source_mismatch")
        if (type(binding["input_length"]) is not int or binding["input_length"] < 1
                or type(binding["score_position"]) is not int
                or binding["score_position"] != binding["input_length"] - 1
                or type(binding["score_input_token_id"]) is not int
                or binding["score_input_token_id"] < 0
                or binding["option_log_probabilities"] != scores
                or type(binding["p_index"]) is not int or binding["p_index"] != p_index
                or type(binding["q_index"]) is not int or binding["q_index"] != q_index
                or binding["video_support_decision"] != "supports_claim"):
            raise ValueError("baseline_pq_score_binding_content_mismatch")
        positions = binding["question_positions"]
        if (type(binding["video_token_id"]) is not int
                or binding["video_token_id"] < 0
                or not isinstance(positions, list) or not positions
                or any(type(position) is not int or position < 0 for position in positions)
                or positions != list(range(positions[0], positions[-1] + 1))
                or positions[-1] >= binding["score_position"]
                or binding["question_positions_sha256"]
                != _sha(_canonical_utf8(positions))
                or binding["question_position_mapper_sha256"]
                != FROZEN_SELECTOR_SOURCE_SHA256):
            raise ValueError("baseline_pq_question_position_binding_invalid")
        binding_sha256 = _sha(_canonical_utf8(binding))
        if row["score_binding_sha256"] != binding_sha256:
            raise ValueError("baseline_pq_score_binding_hash_mismatch")
        if (type(row["p_index"]) is not int or row["p_index"] != p_index
                or type(row["q_index"]) is not int or row["q_index"] != q_index
                or row["video_support_decision"] != "supports_claim"
                or not isinstance(row["reviewed_intervals_sha256"], str)
                or not _SHA256.fullmatch(row["reviewed_intervals_sha256"])):
            raise ValueError("baseline_pq_prediction_or_source_binding_invalid")
        if (not isinstance(row["question"], str) or not row["question"]
                or not isinstance(row["claim"], str) or not row["claim"]
                or not isinstance(row["options"], list) or len(row["options"]) != 5
                or any(not isinstance(option, str) or not option for option in row["options"])):
            raise ValueError("baseline_pq_private_text_schema_invalid")
        seen.add(token)
        checked.append({
            "review_token": token,
            "review_token_sha256": _sha(token.encode("ascii")),
            "p_index": p_index,
            "q_index": q_index,
            "score_binding_sha256": binding_sha256,
            "processor_batch_sha256": binding["processor_batch_sha256"],
            "score_binding": binding,
            "row_sha256": _sha(_canonical_utf8(row)),
        })
    return checked


def _core_manifest(*, phase1_receipt_sha256: str,
                   phase1_records: Any, phase2_records: Any,
                   rows: Any, phase1_row_hashes: list[str],
                   baseline_pq_manifest_sha256: str,
                   baseline_rows: list[dict[str, Any]],
                   phase1_source_pins: dict[str, Any],
                   expected_attention_source_dtype: str) -> dict[str, Any]:
    if not isinstance(phase1_receipt_sha256, str) or not _SHA256.fullmatch(
            phase1_receipt_sha256):
        raise ValueError("phase1_receipt_hash_invalid")
    first = _record_rows(phase1_records)
    second = _record_rows(phase2_records)
    if first != second:
        raise ValueError("phase2_records_differ_from_phase1")
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ValueError("phase2_plan_row_count_invalid")
    if (len(phase1_row_hashes) != ROW_COUNT or len(baseline_rows) != ROW_COUNT
            or not _SHA256.fullmatch(baseline_pq_manifest_sha256)):
        raise ValueError("phase2_upstream_binding_count_invalid")
    if (not isinstance(expected_attention_source_dtype, str)
            or expected_attention_source_dtype not in {"float32", "bfloat16"}):
        raise ValueError("phase2_attention_source_dtype_invalid")

    clean = []
    row_hashes = set()
    common_treatment = None
    source_rows = phase1_source_pins["records"]
    for offset, (receipt_row, expected_row_hash, baseline, source, row) in enumerate(
            zip(first, phase1_row_hashes, baseline_rows, source_rows, rows, strict=True), start=1):
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise ValueError("phase2_plan_row_schema_invalid")
        if (type(row["record_position"]) is not int
                or row["record_position"] != offset
                or row["record_token"] != receipt_row["record_token"]
                or row["processor_batch_sha256"] != receipt_row["processor_batch_sha256"]
                or row["record_token"] != baseline["review_token_sha256"]
                or row["record_token"] != _sha(source["review_token"].encode("ascii"))):
            raise ValueError("phase2_plan_record_or_batch_mismatch")
        if any(baseline["score_binding"][field] != source[field]
               for field in ("question_sha256", "options_sha256", "claim_sha256",
                             "source_video_sha256", "reviewed_intervals_sha256",
                             "video_support_decision")):
            raise ValueError("phase2_plan_stage1_source_pin_mismatch")
        row_hash = row["row_binding_sha256"]
        if (not isinstance(row_hash, str) or not _SHA256.fullmatch(row_hash)
                or row_hash in row_hashes or row_hash != expected_row_hash):
            raise ValueError("phase2_plan_row_binding_invalid_or_duplicate")
        if (row["baseline_pq_row_sha256"] != baseline["row_sha256"]
                or type(row["baseline_p_index"]) is not int
                or row["baseline_p_index"] != baseline["p_index"]
                or type(row["baseline_q_index"]) is not int
                or row["baseline_q_index"] != baseline["q_index"]):
            raise ValueError("phase2_plan_baseline_pq_binding_mismatch")
        plan = row["plan"]
        validate_qta_nine_branch_plan(plan)
        if (plan["row_binding_sha256"] != row_hash
                or plan["typed_batch_sha256"] != receipt_row["processor_batch_sha256"]):
            raise ValueError("phase2_plan_binding_mismatch")
        if (plan["score_binding_sha256"] != baseline["score_binding_sha256"]
                or baseline["processor_batch_sha256"]
                != receipt_row["processor_batch_sha256"]):
            raise ValueError("phase2_plan_score_binding_mismatch")
        if (plan["intervention_layers_1_based"] != FROZEN_QTA_LAYERS_1_BASED
                or plan["selection_spec"]["selector_layers_1_based"]
                != FROZEN_QTA_LAYERS_1_BASED
                or plan["logit_bias"] != FROZEN_QTA_LOGIT_BIAS
                or plan["selection_spec"]["sham_seed"] != FROZEN_QTA_SHAM_SEED):
            raise ValueError("phase2_plan_differs_from_frozen_qta_protocol")
        score_binding = baseline["score_binding"]
        selection = plan["selection_provenance"]
        if not isinstance(selection, Mapping):
            raise ValueError("phase2_selection_provenance_missing")
        validate_qta_selection_provenance(
            selection,
            expected_bindings={
                "record_token": receipt_row["record_token"],
                "opaque_review_token": baseline["review_token"],
                "row_binding_sha256": row_hash,
                "typed_batch_sha256": receipt_row["processor_batch_sha256"],
                "score_binding_sha256": baseline["score_binding_sha256"],
                "baseline_pq_row_sha256": baseline["row_sha256"],
                "prompt_sha256": score_binding["prompt_sha256"],
                "input_ids_sha256": score_binding["input_ids_sha256"],
                "input_length": score_binding["input_length"],
                "score_position": score_binding["score_position"],
                "question_positions": score_binding["question_positions"],
                "temporal_groups": plan["temporal_groups"],
                "selector_source_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
                "selector_layers_1_based": FROZEN_QTA_LAYERS_1_BASED,
                "sham_seed": FROZEN_QTA_SHAM_SEED,
            },
            plan=plan,
        )
        treatment = (tuple(plan["intervention_layers_1_based"]),
                     float(plan["logit_bias"]), plan["selection_spec_sha256"])
        if common_treatment is None:
            common_treatment = treatment
        elif treatment != common_treatment:
            raise ValueError("phase2_cohort_treatment_mismatch")
        row_hashes.add(row_hash)
        clean.append({
            "record_position": offset,
            "record_token": receipt_row["record_token"],
            "processor_batch_sha256": receipt_row["processor_batch_sha256"],
            "row_binding_sha256": row_hash,
            "baseline_pq_row_sha256": baseline["row_sha256"],
            "baseline_p_index": baseline["p_index"],
            "baseline_q_index": baseline["q_index"],
            "score_binding_sha256": baseline["score_binding_sha256"],
            "plan": dict(plan),
        })
    return {
        "schema_version": 2,
        "purpose": "qta_phase2_nine_branch_plan_manifest_v2",
        "phase": "post_baseline",
        "phase1_receipt_sha256": phase1_receipt_sha256,
        "baseline_pq_manifest_sha256": baseline_pq_manifest_sha256,
        "expected_attention_source_dtype": expected_attention_source_dtype,
        "requested_records": ROW_COUNT,
        "planned_forwards": PHASE2_FORWARDS,
        "completed_forwards": 0,
        "selection_arithmetic_reproduced_records": ROW_COUNT,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "records": clean,
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorized": False,
        "authorized_forwards": 0,
    }

def create_qta_phase2_plan_manifest_candidate(*, phase1_receipt_raw: bytes,
                                               phase1_record_bindings: Any,
                                               phase1_preflight_raw: bytes,
                                               phase1_stage1_source_pins_raw: bytes,
                                               baseline_pq_manifest_raw: bytes,
                                               plans: Any,
                                               # Synthetic callers default to
                                               # float32; model runs pin this.
                                               expected_attention_source_dtype: str = "float32") -> bytes:
    """Serialize plans plus expected capture dtype; grants no authority."""
    if not isinstance(phase1_receipt_raw, bytes) or not phase1_receipt_raw:
        raise ValueError("phase1_receipt_bytes_required")
    if len(phase1_receipt_raw) > MAX_UPSTREAM_BYTES:
        raise ValueError("phase1_receipt_bytes_invalid")
    if (not isinstance(baseline_pq_manifest_raw, bytes)
            or not baseline_pq_manifest_raw
            or len(baseline_pq_manifest_raw) > MAX_UPSTREAM_BYTES):
        raise ValueError("baseline_pq_manifest_bytes_invalid")
    phase1_rows = _record_rows(phase1_record_bindings)
    phase1_hashes = _phase1_preflight_row_hashes(phase1_preflight_raw)
    stage1_pins = _phase1_stage1_source_pins(phase1_stage1_source_pins_raw)
    baseline_rows = _baseline_rows(baseline_pq_manifest_raw)
    baseline_obj = _strict_json(baseline_pq_manifest_raw)
    if (stage1_pins["video_claim_review_receipt_sha256"]
            != baseline_obj["video_claim_review_receipt_sha256"]):
        raise ValueError("phase1_baseline_video_review_receipt_mismatch")
    if not isinstance(plans, list) or len(plans) != ROW_COUNT:
        raise ValueError("phase2_plan_row_count_invalid")
    rows = []
    for index, (record, plan) in enumerate(zip(phase1_rows, plans, strict=True), start=1):
        if not isinstance(plan, Mapping):
            raise ValueError("phase2_plan_invalid")
        rows.append({
            "record_position": index,
            "record_token": record["record_token"],
            "processor_batch_sha256": record["processor_batch_sha256"],
            "row_binding_sha256": plan.get("row_binding_sha256"),
            "baseline_pq_row_sha256": baseline_rows[index - 1]["row_sha256"],
            "baseline_p_index": baseline_rows[index - 1]["p_index"],
            "baseline_q_index": baseline_rows[index - 1]["q_index"],
            "score_binding_sha256": baseline_rows[index - 1]["score_binding_sha256"],
            "plan": dict(plan),
        })
    manifest = _core_manifest(
        phase1_receipt_sha256=_sha(phase1_receipt_raw),
        phase1_records=phase1_rows, phase2_records=phase1_rows, rows=rows,
        phase1_row_hashes=phase1_hashes,
        baseline_pq_manifest_sha256=_sha(baseline_pq_manifest_raw),
        baseline_rows=baseline_rows, phase1_source_pins=stage1_pins,
        expected_attention_source_dtype=expected_attention_source_dtype)
    return _canonical(manifest) + b"\n"


def _validate_manifest_bytes(*, manifest_raw: bytes,
                              phase1_receipt_sha256: str,
                              phase1_records: Any,
                              phase2_records: Any,
                              phase1_preflight_raw: bytes,
                              phase1_stage1_source_pins_raw: bytes,
                              baseline_pq_manifest_raw: bytes) -> dict[str, Any]:
    if (not isinstance(manifest_raw, bytes) or not manifest_raw
            or len(manifest_raw) > MAX_MANIFEST_BYTES):
        raise ValueError("phase2_manifest_bytes_invalid")
    manifest = _strict_json(manifest_raw)
    if (not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS
            or _canonical(manifest) + b"\n" != manifest_raw):
        raise ValueError("phase2_manifest_schema_or_canonical_encoding_invalid")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2
            or manifest["purpose"] != "qta_phase2_nine_branch_plan_manifest_v2"
            or manifest["phase"] != "post_baseline"
            or type(manifest["requested_records"]) is not int
            or manifest["requested_records"] != ROW_COUNT
            or type(manifest["planned_forwards"]) is not int
            or manifest["planned_forwards"] != PHASE2_FORWARDS
            or type(manifest["completed_forwards"]) is not int
            or manifest["completed_forwards"] != 0
            or type(manifest["selection_arithmetic_reproduced_records"]) is not int
            or manifest["selection_arithmetic_reproduced_records"] != ROW_COUNT
            or not isinstance(manifest["expected_attention_source_dtype"], str)
            or manifest["expected_attention_source_dtype"] not in {"float32", "bfloat16"}
            or manifest["forward_source_authenticated"] is not False
            or manifest["attention_capture_authenticated"] is not False
            or manifest["execution_authorization"] != "not_granted"
            or manifest["model_load_permitted"] is not False
            or manifest["scoring_authorized"] is not False
            or type(manifest["authorized_forwards"]) is not int
            or manifest["authorized_forwards"] != 0):
        raise ValueError("phase2_manifest_contract_invalid")
    if manifest["phase1_receipt_sha256"] != phase1_receipt_sha256:
        raise ValueError("phase2_manifest_phase1_receipt_mismatch")
    if manifest["baseline_pq_manifest_sha256"] != _sha(baseline_pq_manifest_raw):
        raise ValueError("phase2_manifest_baseline_pq_manifest_mismatch")
    stage1_pins = _phase1_stage1_source_pins(phase1_stage1_source_pins_raw)
    if (stage1_pins["video_claim_review_receipt_sha256"]
            != _strict_json(baseline_pq_manifest_raw)[
                "video_claim_review_receipt_sha256"]):
        raise ValueError("phase1_baseline_video_review_receipt_mismatch")
    _core_manifest(
        phase1_receipt_sha256=phase1_receipt_sha256,
        phase1_records=phase1_records, phase2_records=phase2_records,
        rows=manifest["records"],
        phase1_row_hashes=_phase1_preflight_row_hashes(phase1_preflight_raw),
        baseline_pq_manifest_sha256=_sha(baseline_pq_manifest_raw),
        baseline_rows=_baseline_rows(baseline_pq_manifest_raw),
        phase1_source_pins=stage1_pins,
        expected_attention_source_dtype=manifest[
            "expected_attention_source_dtype"])
    return {
        "status": "phase2_plan_manifest_bound_unvalidated",
        "requested_records": ROW_COUNT,
        "validated_plan_rows": ROW_COUNT,
        "selection_arithmetic_reproduced_records":
            manifest["selection_arithmetic_reproduced_records"],
        "forward_source_authenticated": manifest["forward_source_authenticated"],
        "attention_capture_authenticated": manifest["attention_capture_authenticated"],
        "scheduled_phase2_forwards": PHASE2_FORWARDS,
        "total_phase_forwards_including_phase1": ROW_COUNT + PHASE2_FORWARDS,
        "manifest_sha256": _sha(manifest_raw),
        "plan_semantics_authenticated": False,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorization": "not_granted",
        "phase_authorized": False,
        "authorized_forwards": 0,
    }


def verify_qta_phase2_plan_candidate(*, phase1_receipt_raw: bytes,
                                     phase1_artifact_bytes: Mapping[str, bytes],
                                     phase2_receipt_raw: bytes,
                                     phase2_artifact_bytes: Mapping[str, bytes],
                                     attention_capture_sidecars: Mapping[str, bytes],
                                     integrity_roster_raw: bytes,
                                     release_roster_raw: bytes,
                                     root_public_key_pem: bytes,
                                     expected_root_public_der_sha256: str) -> dict[str, Any]:
    """Verify both signed byte envelopes, then validate their plan-row join.

    The report confirms signatures and structure only. It never authorizes
    model load, scoring, or phase execution.
    """
    receipt_report = verify_phase_freeze_receipt(
        receipt_raw=phase2_receipt_raw, phase="post_baseline",
        artifact_bytes=phase2_artifact_bytes,
        integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256,
        prior_phase1_receipt_raw=phase1_receipt_raw,
        prior_phase1_artifact_bytes=phase1_artifact_bytes)
    phase1_obj = _strict_json(phase1_receipt_raw)
    phase2_obj = _strict_json(phase2_receipt_raw)
    if phase1_obj.get("freeze_id") != phase2_obj.get("freeze_id"):
        raise ValueError("phase2_freeze_id_differs_from_phase1")
    manifest_raw = phase2_artifact_bytes.get("nine_branch_plan_manifest")
    phase1_preflight_raw = phase1_artifact_bytes.get("atomic_cohort_preflight")
    baseline_pq_manifest_raw = phase2_artifact_bytes.get("baseline_pq_manifest")
    stage1_source_pins_raw = phase1_artifact_bytes.get("stage1_review_and_lineage")
    report = _validate_manifest_bytes(
        manifest_raw=manifest_raw,
        phase1_receipt_sha256=_sha(phase1_receipt_raw),
        phase1_records=phase1_obj.get("record_bindings"),
        phase2_records=phase2_obj.get("record_bindings"),
        phase1_preflight_raw=phase1_preflight_raw,
        phase1_stage1_source_pins_raw=stage1_source_pins_raw,
        baseline_pq_manifest_raw=baseline_pq_manifest_raw)
    manifest = _strict_json(manifest_raw)
    baselines = _baseline_rows(baseline_pq_manifest_raw)
    expected_capture_rows = []
    for row, baseline in zip(manifest["records"], baselines, strict=True):
        plan = row["plan"]
        selection = plan["selection_provenance"]
        score_binding = baseline["score_binding"]
        run_digests = score_binding["run_digests"]
        expected_capture_binding = {
            "record_token": row["record_token"],
            "row_binding_sha256": row["row_binding_sha256"],
            "typed_batch_sha256": row["processor_batch_sha256"],
            "baseline_pq_row_sha256": row["baseline_pq_row_sha256"],
            "prompt_sha256": score_binding["prompt_sha256"],
            "input_ids_sha256": score_binding["input_ids_sha256"],
            "model_sha256": run_digests["model_sha256"],
            "processor_sha256": run_digests["processor_sha256"],
            "tokenizer_sha256": run_digests["tokenizer_sha256"],
            "question_positions": score_binding["question_positions"],
            "temporal_groups": plan["temporal_groups"],
        }
        expected_selection_bindings = {
            "record_token": row["record_token"],
            "opaque_review_token": baseline["review_token"],
            "row_binding_sha256": row["row_binding_sha256"],
            "typed_batch_sha256": row["processor_batch_sha256"],
            "score_binding_sha256": baseline["score_binding_sha256"],
            "baseline_pq_row_sha256": row["baseline_pq_row_sha256"],
            "prompt_sha256": score_binding["prompt_sha256"],
            "input_ids_sha256": score_binding["input_ids_sha256"],
            "input_length": score_binding["input_length"],
            "score_position": score_binding["score_position"],
            "question_positions": score_binding["question_positions"],
            "temporal_groups": plan["temporal_groups"],
            "selector_source_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
            "selector_layers_1_based": FROZEN_QTA_LAYERS_1_BASED,
            "sham_seed": FROZEN_QTA_SHAM_SEED,
        }
        expected_capture_rows.append({
            "expected_capture_binding": expected_capture_binding,
            "input_length": score_binding["input_length"],
            "expected_attention_source_dtype": manifest[
                "expected_attention_source_dtype"],
            "selection_provenance": selection,
            "expected_selection_bindings": expected_selection_bindings,
            "plan": plan,
        })
    capture_report = validate_qta_capture_sidecar_index(
        index_raw=phase2_artifact_bytes["attention_capture_index"],
        sidecars=attention_capture_sidecars, expected_rows=expected_capture_rows)
    return {
        **report,
        "capture_sidecar_index_sha256": _sha(
            phase2_artifact_bytes["attention_capture_index"]),
        "capture_sidecar_index_valid": True,
        "capture_sidecar_records_validated": capture_report["record_count"],
        "capture_sidecar_total_bytes": capture_report["total_sidecar_bytes"],
        "expected_attention_source_dtype": manifest[
            "expected_attention_source_dtype"],
        "capture_sidecars_joined_to_selection_masses": True,
        "attention_capture_authenticated": False,
        "forward_source_authenticated": False,
        "phase1_receipt_signature_valid": True,
        "phase2_receipt_signature_valid": receipt_report["receipt_signature_valid"],
        "phase2_artifact_bytes_match": receipt_report["artifact_bytes_match"],
        "phase1_receipt_recursively_verified": receipt_report["prior_phase1_receipt_valid"],
        "plan_manifest_is_signed_phase2_artifact": True,
        "same_freeze_id": True,
    }
