"""Validate a frozen, hash-only QA target manifest for future claim scoring.

Structural contract only: this does not load media, authenticate upstream
artifacts, tokenize labels, or authorize scoring. Label token-piece counts are
assertions from a future tokenizer audit and are not trusted evidence here.
"""
from __future__ import annotations

import hashlib
import json
import re

SHA256 = re.compile(r"^[a-f0-9]{64}$")
LABELS = ["A", "B", "C", "D", "E"]
TOP_KEYS = {
    "schema_version", "purpose", "status", "execution_authorization",
    "claim_input_manifest_sha256", "scoring_protocol_sha256",
    "dataset_source_manifest_sha256",
    "model_checkpoint_sha256", "model_config_sha256", "tokenizer_sha256",
    "processor_config_sha256", "prompt_builder_sha256", "records",
}
ROW_KEYS = {
    "video_id", "claim_id", "dataset_qid_sha256", "dataset_source_row_sha256", "question_sha256",
    "ordered_choice_sha256", "gold_index", "answer_label_order",
    "answer_label_token_ids", "answer_label_token_piece_counts",
    "rendered_scoring_prefix_sha256",
}
BINDING_KEYS = TOP_KEYS - {"schema_version", "purpose", "status",
                           "execution_authorization", "records"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _blocked(code):
    return {
        "status": "blocked", "scope": "unvalidated_structural_contract_only",
        "failure_codes": [code], "requested_records": 3,
        "validated_records": 0, "score_calls": 0,
        "execution_authorization": "not_granted",
    }


def validate_target_manifest(raw: bytes, expected_video_ids: list[str],
                             expected_claim_ids: list[str],
                             expected_binding_hashes: dict[str, str]) -> dict:
    """Check exact three-row target/config binding without returning identities."""
    if (not isinstance(expected_video_ids, list) or len(expected_video_ids) != 3
            or not isinstance(expected_claim_ids, list) or len(expected_claim_ids) != 3
            or any(not isinstance(x, str) or not x.strip() for x in expected_video_ids + expected_claim_ids)
            or len(set(expected_video_ids)) != 3 or len(set(expected_claim_ids)) != 3):
        return _blocked("expected_cohort_invalid")
    if (not isinstance(expected_binding_hashes, dict)
            or set(expected_binding_hashes) != BINDING_KEYS
            or any(not isinstance(value, str) or SHA256.fullmatch(value) is None
                   for value in expected_binding_hashes.values())):
        return _blocked("expected_bindings_invalid")
    if not isinstance(raw, bytes) or len(raw) > 1_000_000:
        return _blocked("manifest_bytes_invalid")
    try:
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _blocked("manifest_json_invalid")
    if (not isinstance(manifest, dict) or set(manifest) != TOP_KEYS
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
            or manifest.get("purpose") != "private_claim_linked_answer_target_contract"
            or manifest.get("status") != "frozen_unvalidated_target_manifest"
            or manifest.get("execution_authorization") != "not_granted"):
        return _blocked("manifest_schema_invalid")
    if any(not isinstance(manifest.get(key), str)
           or SHA256.fullmatch(manifest[key]) is None
           or manifest[key] != expected_binding_hashes[key]
           for key in BINDING_KEYS):
        return _blocked("manifest_binding_mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 3:
        return _blocked("manifest_record_count_invalid")
    by_video = {}
    seen_qid = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            return _blocked("target_row_schema_invalid")
        if (not isinstance(row.get("video_id"), str) or not row["video_id"].strip()
                or not isinstance(row.get("claim_id"), str) or not row["claim_id"].strip()
                or row["video_id"] in by_video):
            return _blocked("target_row_identity_invalid")
        for key in ("dataset_qid_sha256", "dataset_source_row_sha256", "question_sha256",
                    "rendered_scoring_prefix_sha256"):
            if not isinstance(row.get(key), str) or SHA256.fullmatch(row[key]) is None:
                return _blocked("target_row_hash_invalid")
        qid_hash = row["dataset_qid_sha256"]
        if qid_hash in seen_qid:
            return _blocked("target_qid_duplicate")
        seen_qid.add(qid_hash)
        choices = row.get("ordered_choice_sha256")
        if (not isinstance(choices, list) or len(choices) != 5
                or any(not isinstance(x, str) or SHA256.fullmatch(x) is None for x in choices)
                or len(set(choices)) != 5):
            return _blocked("choice_order_binding_invalid")
        if (type(row.get("gold_index")) is not int or not 0 <= row["gold_index"] < 5
                or row.get("answer_label_order") != LABELS):
            return _blocked("gold_or_label_order_invalid")
        token_ids = row.get("answer_label_token_ids")
        piece_counts = row.get("answer_label_token_piece_counts")
        if (not isinstance(token_ids, list) or len(token_ids) != 5
                or any(type(value) is not int or value < 0 for value in token_ids)
                or len(set(token_ids)) != 5
                or not isinstance(piece_counts, list) or len(piece_counts) != 5
                or any(type(value) is not int or value != 1 for value in piece_counts)):
            return _blocked("single_token_label_assertion_invalid")
        by_video[row["video_id"]] = row["claim_id"]
    if set(by_video) != set(expected_video_ids):
        return _blocked("target_video_cohort_mismatch")
    if list(by_video) != expected_video_ids:
        return _blocked("target_video_order_mismatch")
    if any(by_video[video_id] != claim_id
           for video_id, claim_id in zip(expected_video_ids, expected_claim_ids, strict=True)):
        return _blocked("target_claim_binding_mismatch")
    return {
        "status": "target_contract_structurally_valid_unvalidated",
        "scope": "unvalidated_structural_contract_only", "failure_codes": [],
        "requested_records": 3, "validated_records": 3, "score_calls": 0,
        "execution_authorization": "not_granted",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
