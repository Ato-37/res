"""Build a non-authorizing QTA baseline p/q review-manifest candidate.

This is a structural contract only. It checks a fixed three-row input bundle,
caller-pinned source and run digests, processor/prompt/token bindings, and
recomputes p/q from each baseline-full A-E log-probability vector. It does not
authenticate any pin, execute a model, verify a human review, or authorize any
forward. A successful result is only candidate manifest bytes plus hashes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Sequence


SHA256 = re.compile(r"^[0-9a-f]{64}$")
TOKENS = ("A", "B", "C", "D", "E")
RUN_DIGEST_FIELDS = (
    "protocol_sha256", "model_sha256", "tokenizer_sha256",
    "scorer_sha256", "processor_sha256",
)
INPUT_KEYS = {"schema_version", "purpose", "video_claim_review_receipt_sha256",
              "run_digests", "records"}
ROW_KEYS = {
    "review_token", "question", "options", "claim", "video_support_decision",
    "reviewed_intervals_sha256", "source_video_sha256", "option_log_probabilities",
    "answer_token_ids", "input_ids", "input_ids_sha256", "prompt_sha256",
    "input_length", "score_position", "score_input_token_id",
    "processor_batch_sha256", "scored_option_token_ids",
}


class CandidateError(ValueError):
    """Stable, non-content-bearing structural validation failure."""


def _fail(code: str) -> None:
    raise CandidateError(code)


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        _fail(f"{field}_invalid")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (UnicodeError, TypeError, ValueError, OverflowError):
        _fail("candidate_json_invalid")


def _hash_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field}_invalid")
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError:
        _fail(f"{field}_invalid")


def _source_hashes(row: dict[str, Any]) -> dict[str, str]:
    question_sha = _hash_text(row.get("question"), "question")
    claim_sha = _hash_text(row.get("claim"), "claim")
    options = row.get("options")
    if (not isinstance(options, list) or len(options) != 5
            or any(not isinstance(option, str) or not option.strip() for option in options)):
        _fail("options_invalid")
    options_sha = hashlib.sha256(_canonical_json(options)).hexdigest()
    return {"question_sha256": question_sha, "options_sha256": options_sha,
            "claim_sha256": claim_sha}


def _valid_log_probability(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, float):
        return False
    return math.isfinite(value) and value <= 0.0


def _validate_row(row: Any, *, token: str, expected_source: dict[str, str],
                  answer_token_ids: list[int]) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        _fail("baseline_input_record_schema_invalid")
    if row["review_token"] != token:
        _fail("baseline_input_token_order_mismatch")
    for key in ("source_video_sha256", "reviewed_intervals_sha256",
                "processor_batch_sha256", "input_ids_sha256", "prompt_sha256"):
        _digest(row.get(key), key)
    if row["video_support_decision"] != "supports_claim":
        _fail("video_claim_not_supported")
    actual_sources = _source_hashes(row)
    if not isinstance(expected_source, dict) or set(expected_source) != {
            *actual_sources, "source_video_sha256", "reviewed_intervals_sha256",
            "video_support_decision"}:
        _fail("expected_source_binding_invalid")
    for key, value in actual_sources.items():
        if _digest(expected_source.get(key), f"expected_{key}") != value:
            _fail("baseline_source_field_binding_mismatch")
    if (_digest(expected_source.get("source_video_sha256"), "expected_source_video_sha256")
            != row["source_video_sha256"]
            or _digest(expected_source.get("reviewed_intervals_sha256"),
                       "expected_reviewed_intervals_sha256")
            != row["reviewed_intervals_sha256"]):
        _fail("baseline_video_review_binding_mismatch")
    if (expected_source["video_support_decision"] != "supports_claim"
            or row["video_support_decision"] != expected_source["video_support_decision"]):
        _fail("baseline_video_support_decision_binding_mismatch")

    scores = row["option_log_probabilities"]
    if (not isinstance(scores, list) or len(scores) != 5
            or any(not _valid_log_probability(x) for x in scores)
            or math.fsum(math.exp(x) for x in scores) > 1.0 + 1e-12):
        _fail("baseline_full_ae_log_probabilities_invalid")
    p_index = min(range(5), key=lambda i: (-scores[i], i))
    q_index = min((i for i in range(5) if i != p_index),
                  key=lambda i: (-scores[i], i))

    ids = row["input_ids"]
    if (not isinstance(ids, list) or not ids
            or any(type(item) is not int or item < 0 for item in ids)):
        _fail("processor_input_ids_invalid")
    if hashlib.sha256(_canonical_json(ids)).hexdigest() != row["input_ids_sha256"]:
        _fail("processor_input_ids_digest_mismatch")
    if (type(row["input_length"]) is not int or row["input_length"] != len(ids)
            or type(row["score_position"]) is not int
            or row["score_position"] != len(ids) - 1
            or row["score_input_token_id"] != ids[-1]):
        _fail("baseline_score_position_binding_invalid")
    if (not isinstance(row["answer_token_ids"], list)
            or any(type(x) is not int or x < 0 for x in row["answer_token_ids"])
            or row["answer_token_ids"] != answer_token_ids
            or not isinstance(row["scored_option_token_ids"], list)
            or any(type(x) is not int or x < 0 for x in row["scored_option_token_ids"])
            or row["scored_option_token_ids"] != answer_token_ids):
        _fail("baseline_option_token_binding_invalid")
    if type(row["score_input_token_id"]) is not int or row["score_input_token_id"] < 0:
        _fail("baseline_score_position_binding_invalid")
    return {
        "review_token": token,
        "question": row["question"], "options": row["options"],
        "claim": row["claim"],
        "option_log_probabilities": scores,
        "p_index": p_index, "q_index": q_index,
        "video_support_decision": row["video_support_decision"],
        "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
    }


def build_baseline_pq_candidate(*, input_raw: bytes,
                                expected_cohort_tokens: Sequence[str],
                                expected_run_digests: dict[str, str],
                                expected_video_claim_review_receipt_sha256: str,
                                expected_source_bindings_by_token: dict[str, dict[str, str]],
                                expected_answer_token_ids: Sequence[int]) -> dict[str, Any]:
    """Validate three baseline rows and return review-manifest candidate bytes.

    Expected digests/cohort must be supplied by a separate frozen-input process;
    this function does not authenticate that process. Its output is never an
    authorization, receipt, identity proof, or semantic proof.
    """
    if (not isinstance(input_raw, bytes) or len(input_raw) > 8 * 1024 * 1024):
        _fail("baseline_input_bytes_invalid")
    if (not isinstance(expected_cohort_tokens, Sequence)
            or isinstance(expected_cohort_tokens, (str, bytes))
            or len(expected_cohort_tokens) != 3
            or any(not isinstance(token, str) or not token or len(token) > 128
                   for token in expected_cohort_tokens)
            or len(set(expected_cohort_tokens)) != 3):
        _fail("expected_three_unique_cohort_tokens_required")
    if not isinstance(expected_run_digests, dict) or set(expected_run_digests) != set(RUN_DIGEST_FIELDS):
        _fail("expected_run_digest_inventory_invalid")
    run_digests = {key: _digest(expected_run_digests[key], f"expected_{key}")
                    for key in RUN_DIGEST_FIELDS}
    expected_review_sha = _digest(expected_video_claim_review_receipt_sha256,
                                  "expected_video_claim_review_receipt_sha256")
    if (not isinstance(expected_source_bindings_by_token, dict)
            or set(expected_source_bindings_by_token) != set(expected_cohort_tokens)):
        _fail("expected_source_binding_inventory_invalid")
    if (not isinstance(expected_answer_token_ids, Sequence)
            or isinstance(expected_answer_token_ids, (str, bytes))
            or len(expected_answer_token_ids) != 5
            or any(type(x) is not int or x < 0 for x in expected_answer_token_ids)
            or len(set(expected_answer_token_ids)) != 5):
        _fail("expected_answer_token_ids_invalid")
    answer_ids = list(expected_answer_token_ids)
    try:
        source = json.loads(input_raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=lambda _: _fail("nonfinite_json"))
    except CandidateError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        _fail("baseline_input_json_invalid")
    if (not isinstance(source, dict) or set(source) != INPUT_KEYS
            or type(source.get("schema_version")) is not int
            or source["schema_version"] != 1
            or source.get("purpose") != "qta_baseline_full_forward_candidate_input"):
        _fail("baseline_input_schema_invalid")
    video_review_sha = _digest(source.get("video_claim_review_receipt_sha256"),
                               "video_claim_review_receipt_sha256")
    if video_review_sha != expected_review_sha:
        _fail("baseline_video_claim_review_receipt_binding_mismatch")
    if source.get("run_digests") != run_digests:
        _fail("baseline_run_digest_binding_mismatch")
    rows = source.get("records")
    if not isinstance(rows, list) or len(rows) != 3:
        _fail("baseline_input_fixed_three_rows_required")
    manifest_rows = [
        _validate_row(row, token=token,
                      expected_source=expected_source_bindings_by_token[token],
                      answer_token_ids=answer_ids)
        for token, row in zip(expected_cohort_tokens, rows, strict=True)
    ]
    manifest = {
        "schema_version": 1,
        "purpose": "qta_baseline_prediction_link_manifest",
        "video_claim_review_receipt_sha256": video_review_sha,
        "records": manifest_rows,
    }
    manifest_raw = _canonical_json(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    binding_commitment = {
        "input_bytes_sha256": hashlib.sha256(input_raw).hexdigest(),
        "run_digests": run_digests,
        "answer_token_ids_sha256": hashlib.sha256(_canonical_json(answer_ids)).hexdigest(),
        "processor_batch_sha256_by_token": {
            row["review_token"]: row["processor_batch_sha256"] for row in rows},
        "source_video_sha256_by_token": {
            row["review_token"]: row["source_video_sha256"] for row in rows},
        "reviewed_intervals_sha256_by_token": {
            row["review_token"]: row["reviewed_intervals_sha256"] for row in rows},
        "input_ids_sha256_by_token": {
            row["review_token"]: row["input_ids_sha256"] for row in rows},
        "prompt_sha256_by_token": {
            row["review_token"]: row["prompt_sha256"] for row in rows},
        "score_position_by_token": {
            row["review_token"]: row["score_position"] for row in rows},
        "score_input_token_id_by_token": {
            row["review_token"]: row["score_input_token_id"] for row in rows},
    }
    return {
        "status": "baseline_pq_review_manifest_candidate_structurally_valid",
        "requested_records": 3, "effective_records": 3,
        "baseline_manifest_raw": manifest_raw,
        "baseline_manifest_sha256": manifest_sha,
        "binding_commitment_sha256": hashlib.sha256(
            _canonical_json(binding_commitment)).hexdigest(),
        "binding_commitment": binding_commitment,
        "p_indices": [row["p_index"] for row in manifest_rows],
        "q_indices": [row["q_index"] for row in manifest_rows],
        "model_load_permitted": False,
        "branch_forwards": 0,
        "score_forwards": 0,
        "execution_authorization": "not_granted",
        "identity_proof": False,
        "semantic_proof": False,
        "limitations": [
            "structural contract only; caller pins and candidate input are unauthenticated",
            "does not verify the trusted release/root/witness gate or stage-1 human review",
            "per-record review decisions are caller-pinned; stage-1 receipt bytes are not revalidated here",
            "does not verify model output provenance or p/q semantics beyond recomputation",
            "candidate inventory and any caller success flag are not trust roots",
            "candidate manifest bytes do not grant baseline, branch, or score execution",
        ],
    }
