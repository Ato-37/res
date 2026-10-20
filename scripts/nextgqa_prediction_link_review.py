"""Validate a post-baseline human link review for QTA prediction claims.

This structural validator binds exact displayed question/options/claim/p/q
fields to a frozen baseline manifest. It cannot authenticate a reviewer or
prove that review occurred, and it never grants score execution authority.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Sequence

from qta_phase2_plan_manifest import validate_baseline_pq_manifest


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JUDGMENTS = {
    "supports_p_over_q",
    "compatible_not_discriminating",
    "unrelated",
    "contradicts_p",
    "ambiguous",
}
RECEIPT_KEYS = {
    "schema_version", "purpose", "status", "baseline_manifest_sha256",
    "protocol_sha256", "prior_video_review_receipt_sha256",
    "reviewer_role", "reviewer_private_id",
    "independence_attestation", "gold_and_branch_outcomes_hidden",
    "reviewed_at_utc", "records", "execution_authorization",
}
RECORD_KEYS = {
    "review_token", "displayed_fields_sha256", "judgment", "rationale",
}
BASELINE_MANIFEST_KEYS = {
    "schema_version", "purpose", "video_claim_review_receipt_sha256", "records",
}
BASELINE_RECORD_KEYS = {
    "review_token", "question", "options", "claim", "option_log_probabilities",
    "p_index", "q_index", "video_support_decision", "reviewed_intervals_sha256",
}
BASELINE_SCORE_BINDING_RECORD_KEYS = BASELINE_RECORD_KEYS | {
    "score_binding", "score_binding_sha256",
}
MAX_BASELINE_MANIFEST_BYTES = 8 * 1024 * 1024


class PredictionLinkReviewError(ValueError):
    """Stable, content-free validation error."""


def _fail(code: str):
    raise PredictionLinkReviewError(code)


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        _fail(code)
    return value


def _is_finite_score(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def displayed_fields_sha256(row: dict[str, Any], *,
                            video_claim_review_receipt_sha256: str) -> str:
    """Digest all text and prior video-review provenance shown to the reviewer."""
    if not isinstance(row, dict):
        _fail("baseline_row_invalid")
    token = row.get("review_token")
    question, options, claim = row.get("question"), row.get("options"), row.get("claim")
    p_index, q_index = row.get("p_index"), row.get("q_index")
    log_probs = row.get("option_log_probabilities")
    if (not isinstance(token, str) or not token or len(token) > 128
            or not isinstance(question, str) or not question.strip()
            or not isinstance(claim, str) or not claim.strip()
            or row.get("video_support_decision") != "supports_claim"
            or not isinstance(row.get("reviewed_intervals_sha256"), str)
            or not SHA256_RE.fullmatch(row["reviewed_intervals_sha256"])
            or not isinstance(video_claim_review_receipt_sha256, str)
            or not SHA256_RE.fullmatch(video_claim_review_receipt_sha256)
            or not isinstance(options, list) or len(options) != 5
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or type(p_index) is not int or type(q_index) is not int
            or p_index not in range(5) or q_index not in range(5)
            or p_index == q_index or not isinstance(log_probs, list)
            or len(log_probs) != 5
            or any(not _is_finite_score(score) for score in log_probs)):
        _fail("baseline_row_fields_invalid")
    try:
        scores = [float(score) for score in log_probs]
        displayed = {
            "question": question,
            "options": options,
            "claim": claim,
            "prior_video_review": {
                "receipt_sha256": video_claim_review_receipt_sha256,
                "support_decision": row["video_support_decision"],
                "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
            },
            "p": {"label": "ABCDE"[p_index], "text": options[p_index]},
            "q": {"label": "ABCDE"[q_index], "text": options[q_index]},
        }
        encoded = json.dumps(displayed, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
    except (UnicodeError, OverflowError, TypeError, ValueError):
        _fail("baseline_row_fields_invalid")
    expected_p = min(range(5), key=lambda index: (-scores[index], index))
    expected_q = min((index for index in range(5) if index != expected_p),
                     key=lambda index: (-scores[index], index))
    if (p_index, q_index) != (expected_p, expected_q):
        _fail("baseline_prediction_rule_mismatch")
    return hashlib.sha256(encoded).hexdigest()


def validate_prediction_link_receipt(
    *, receipt: Any, baseline_manifest_raw: bytes,
    expected_baseline_manifest_sha256: str,
    expected_protocol_sha256: str,
    expected_video_review_receipt_sha256: str,
    expected_cohort_tokens: Sequence[str],
    expected_source_field_sha256_by_token: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return cohort eligibility without implying human authentication/authority."""
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        _fail("prediction_link_receipt_schema_invalid")
    if (type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or receipt.get("purpose") != "qta_prediction_link_adjudication"
            or receipt.get("status") != "prediction_link_review_recorded_unvalidated"
            or receipt.get("execution_authorization") != "not_granted"):
        _fail("prediction_link_receipt_status_invalid")
    expected_baseline = _sha(expected_baseline_manifest_sha256,
                             "expected_baseline_manifest_hash_invalid")
    if (not isinstance(baseline_manifest_raw, bytes)
            or len(baseline_manifest_raw) > MAX_BASELINE_MANIFEST_BYTES
            or hashlib.sha256(baseline_manifest_raw).hexdigest() != expected_baseline):
        _fail("baseline_manifest_bytes_hash_mismatch")
    try:
        manifest = json.loads(baseline_manifest_raw.decode("utf-8"),
                              object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        _fail("baseline_manifest_json_invalid")
    if (not isinstance(manifest, dict) or set(manifest) != BASELINE_MANIFEST_KEYS
            or type(manifest.get("schema_version")) is not int
            or (manifest.get("schema_version"), manifest.get("purpose")) !=
                (3, "qta_baseline_prediction_link_manifest_v3")):
        _fail("baseline_manifest_schema_invalid")
    try:
        validate_baseline_pq_manifest(baseline_manifest_raw)
    except (TypeError, ValueError, KeyError, UnicodeError,
            OverflowError, RecursionError):
        _fail("baseline_manifest_score_binding_invalid")
    expected_protocol = _sha(expected_protocol_sha256,
                             "expected_protocol_hash_invalid")
    expected_video_review = _sha(expected_video_review_receipt_sha256,
                                 "expected_video_review_receipt_hash_invalid")
    if _sha(manifest.get("video_claim_review_receipt_sha256"),
            "baseline_video_review_receipt_hash_invalid") != expected_video_review:
        _fail("baseline_video_review_receipt_binding_mismatch")
    if (_sha(receipt.get("baseline_manifest_sha256"),
             "baseline_manifest_hash_invalid") != expected_baseline
            or _sha(receipt.get("protocol_sha256"), "protocol_hash_invalid")
            != expected_protocol
            or _sha(receipt.get("prior_video_review_receipt_sha256"),
                    "prior_video_review_receipt_hash_invalid") != expected_video_review):
        _fail("prediction_link_manifest_binding_mismatch")
    reviewer_id = receipt.get("reviewer_private_id")
    if (receipt.get("reviewer_role") != "independent_prediction_link_reviewer"
            or not isinstance(reviewer_id, str) or not reviewer_id.strip()
            or len(reviewer_id) > 128
            or receipt.get("independence_attestation")
            != "independent_from_generation_scoring_custody_and_video_review"
            or receipt.get("gold_and_branch_outcomes_hidden") is not True):
        _fail("prediction_link_reviewer_attestation_invalid")
    try:
        reviewed_at = datetime.fromisoformat(receipt.get("reviewed_at_utc", "").replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        _fail("prediction_link_review_time_invalid")
    if reviewed_at.tzinfo is None:
        _fail("prediction_link_review_time_invalid")
    if reviewed_at.utcoffset() != timedelta(0):
        _fail("prediction_link_review_time_invalid")
    if (not isinstance(expected_cohort_tokens, Sequence)
            or isinstance(expected_cohort_tokens, (str, bytes))
            or not expected_cohort_tokens
            or any(not isinstance(token, str) or not token for token in expected_cohort_tokens)
            or len(set(expected_cohort_tokens)) != len(expected_cohort_tokens)):
        _fail("expected_cohort_tokens_invalid")
    baseline_rows = manifest.get("records")
    if (not isinstance(baseline_rows, list)
            or len(baseline_rows) != len(expected_cohort_tokens)):
        _fail("baseline_manifest_cohort_count_mismatch")
    if (not isinstance(expected_source_field_sha256_by_token, dict)
            or set(expected_source_field_sha256_by_token) != set(expected_cohort_tokens)):
        _fail("expected_source_digest_inventory_invalid")
    expected_tokens = []
    expected_digests = {}
    for row in baseline_rows:
        if (not isinstance(row, dict)
                or set(row) != BASELINE_SCORE_BINDING_RECORD_KEYS):
            _fail("baseline_manifest_record_schema_invalid")
        digest = displayed_fields_sha256(
            row, video_claim_review_receipt_sha256=expected_video_review)
        token = row["review_token"]
        if token in expected_digests:
            _fail("baseline_review_token_duplicate")
        if token not in expected_cohort_tokens:
            _fail("baseline_manifest_cohort_binding_mismatch")
        expected_tokens.append(token)
        expected_digests[token] = digest
        expected_fields = expected_source_field_sha256_by_token[token]
        if (not isinstance(expected_fields, dict)
                or set(expected_fields) != {"question_sha256", "options_sha256", "claim_sha256",
                                            "video_support_decision",
                                            "reviewed_intervals_sha256"}):
            _fail("expected_source_digest_fields_invalid")
        actual_fields = {
            "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "options_sha256": hashlib.sha256(json.dumps(
                row["options"], ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            "claim_sha256": hashlib.sha256(row["claim"].encode("utf-8")).hexdigest(),
        }
        if (row["video_support_decision"] != "supports_claim"
                or expected_fields.get("video_support_decision") != row["video_support_decision"]
                or _sha(expected_fields.get("reviewed_intervals_sha256"),
                        "expected_reviewed_intervals_digest_invalid")
                != _sha(row["reviewed_intervals_sha256"],
                        "baseline_reviewed_intervals_digest_invalid")):
            _fail("baseline_video_review_binding_mismatch")
        if any(_sha(expected_fields.get(key), "expected_source_digest_invalid") != value
               for key, value in actual_fields.items()):
            _fail("baseline_source_field_binding_mismatch")
    if expected_tokens != list(expected_cohort_tokens):
        _fail("baseline_manifest_cohort_binding_mismatch")
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != len(expected_tokens):
        _fail("prediction_link_record_count_mismatch")
    seen = []
    eligible = True
    for item in records:
        if not isinstance(item, dict) or set(item) != RECORD_KEYS:
            _fail("prediction_link_record_schema_invalid")
        token = item.get("review_token")
        if not isinstance(token, str) or token in seen or token not in expected_digests:
            _fail("prediction_link_record_token_invalid")
        seen.append(token)
        if _sha(item.get("displayed_fields_sha256"), "displayed_fields_hash_invalid") != expected_digests[token]:
            _fail("prediction_link_display_binding_mismatch")
        judgment = item.get("judgment")
        rationale = item.get("rationale")
        if not isinstance(judgment, str) or judgment not in JUDGMENTS:
            _fail("prediction_link_judgment_invalid")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4000:
            _fail("prediction_link_rationale_invalid")
        if judgment != "supports_p_over_q":
            eligible = False
    if seen != expected_tokens:
        _fail("prediction_link_record_order_mismatch")
    return {
        "status": "prediction_link_review_structurally_valid_unvalidated",
        "requested_records": len(expected_tokens),
        "support_judgment_count": len(expected_tokens) if eligible else sum(
            item["judgment"] == "supports_p_over_q" for item in records),
        "all_expected_rows_have_supporting_self_reported_judgment": eligible,
        "reviewer_identity_verified": False,
        "semantic_judgment_authenticated": False,
        "execution_authorization": "not_granted",
        "limitations": [
            "self-reported review and judgment only",
            "reviewer identity and actual viewing are not authenticated",
            "expected cohort/source hashes must come from independently verified frozen inputs",
            "structural validity is not semantic correctness or score authorization",
        ],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("baseline_manifest_duplicate_json_key")
        result[key] = value
    return result
