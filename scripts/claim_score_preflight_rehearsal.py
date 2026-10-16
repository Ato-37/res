"""Synthetic-only rehearsal of assignment + QA-target atomic preflight.

This does not authenticate caller-supplied audit/calibration/source evidence,
load media or model weights, or authorize scoring. It exists to test interface
and fail-closed behavior before independent production adapters are written.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from claim_answer_target_contract import BINDING_KEYS, validate_target_manifest
from claim_review_assignment_plan import plan_claim_review_assignments
from audit_nextgqa_claim_packet_lineage import (
    DATASET_KEYS, LINEAGE_KEYS, LINEAGE_ROW_KEYS, SHA256,
    validate_for_worksheet,
)


def _blocked(stage):
    return {
        "status": "blocked", "scope": "synthetic_preflight_rehearsal_only",
        "failure_codes": [stage], "denominator": 3,
        "assignments": [], "model_forwards_planned": 0,
        "score_calls": 0, "execution_authorization": "not_granted",
    }


def _norm_answer(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def rehearse_claim_score_preflight(*, assignment_inputs: dict,
                                   target_manifest_raw: bytes,
                                   expected_target_binding_hashes: dict,
                                   lineage_manifest_raw: bytes,
                                   qa_source_rows_raw: bytes) -> dict:
    """Join both structural planners atomically; success is still unvalidated."""
    if not isinstance(assignment_inputs, dict):
        return _blocked("assignment_inputs_invalid")
    try:
        expected_videos = assignment_inputs["expected_video_ids"]
        claim_manifest_raw = assignment_inputs["claim_input_manifest_raw"]
        claim_manifest_sha = assignment_inputs["claim_input_manifest_sha256"]
        worksheet = assignment_inputs["worksheet_records"]
        worksheet_by_video = {row["video_id"]: row for row in worksheet}
        report = assignment_inputs["review_audit_report"]
        if (not isinstance(claim_manifest_raw, bytes)
                or hashlib.sha256(claim_manifest_raw).hexdigest() != claim_manifest_sha):
            return _blocked("claim_input_manifest_binding_invalid")
        if (not isinstance(lineage_manifest_raw, bytes) or len(lineage_manifest_raw) > 1_000_000
                or hashlib.sha256(lineage_manifest_raw).hexdigest()
                != report.get("lineage_manifest_sha256")):
            return _blocked("lineage_manifest_binding_invalid")

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_json_key")
                result[key] = value
            return result

        claim_manifest = json.loads(claim_manifest_raw.decode("utf-8"),
                                    object_pairs_hook=unique_pairs,
                                    parse_constant=lambda _: (_ for _ in ()).throw(
                                        ValueError("nonfinite_json")))
        if (not isinstance(claim_manifest, dict)
                or not isinstance(claim_manifest.get("claims"), list)
                or len(claim_manifest["claims"]) != 3):
            return _blocked("claim_input_manifest_invalid")
        claim_by_video = {}
        for row in claim_manifest["claims"]:
            if (not isinstance(row, dict)
                    or not isinstance(row.get("video_id"), str)
                    or not isinstance(row.get("claim_id"), str)
                    or row["video_id"] in claim_by_video):
                return _blocked("claim_input_manifest_invalid")
            claim_by_video[row["video_id"]] = row["claim_id"]
        if set(claim_by_video) != set(expected_videos):
            return _blocked("claim_input_cohort_mismatch")
        expected_claims = [claim_by_video[video_id] for video_id in expected_videos]
        lineage = json.loads(lineage_manifest_raw.decode("utf-8"),
                             object_pairs_hook=unique_pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(
                                 ValueError("nonfinite_json")))
        if (not isinstance(lineage, dict) or set(lineage) != LINEAGE_KEYS
                or type(lineage.get("schema_version")) is not int
                or lineage.get("schema_version") != 1
                or lineage.get("purpose") != "private_source_lineage_crosscheck_for_claim_review"
                or lineage.get("status") != "metadata_lineage_crosschecked"
                or lineage.get("requested_records") != 3
                or lineage.get("effective_records") != 3
                or not isinstance(lineage.get("records"), list)
                or len(lineage["records"]) != 3):
            return _blocked("lineage_manifest_schema_invalid")
        source_hashes = lineage.get("dataset_inputs_sha256")
        if (not isinstance(source_hashes, dict) or set(source_hashes) != set(DATASET_KEYS)
                or any(not isinstance(value, str) or SHA256.fullmatch(value) is None
                       for value in source_hashes.values())):
            return _blocked("lineage_dataset_inputs_invalid")
        worksheet_doc = json.loads(assignment_inputs["worksheet_raw"].decode("utf-8"),
                                   object_pairs_hook=unique_pairs,
                                   parse_constant=lambda _: (_ for _ in ()).throw(
                                       ValueError("nonfinite_json")))
        try:
            validate_for_worksheet(worksheet_doc, lineage, lineage_manifest_raw)
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError):
            return _blocked("lineage_worksheet_binding_invalid")

        if not isinstance(qa_source_rows_raw, bytes) or len(qa_source_rows_raw) > 1_000_000:
            return _blocked("qa_source_rows_invalid")
        qa_source = json.loads(qa_source_rows_raw.decode("utf-8"),
                               object_pairs_hook=unique_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(
                                   ValueError("nonfinite_json")))
        qa_keys = {"schema_version", "purpose", "dataset_inputs_sha256", "records"}
        qa_row_keys = {"video_id", "dataset_video_id", "dataset_question_id",
                       "question", "choices", "answer"}
        if (not isinstance(qa_source, dict) or set(qa_source) != qa_keys
                or type(qa_source.get("schema_version")) is not int
                or qa_source.get("schema_version") != 1
                or qa_source.get("purpose") != "synthetic_private_qa_source_rows"
                or qa_source.get("dataset_inputs_sha256") != source_hashes
                or not isinstance(qa_source.get("records"), list)
                or len(qa_source["records"]) != 3):
            return _blocked("qa_source_rows_schema_invalid")
        qa_by_video = {}
        for row in qa_source["records"]:
            if (not isinstance(row, dict) or set(row) != qa_row_keys
                    or not isinstance(row.get("video_id"), str)
                    or row["video_id"] in qa_by_video
                    or not isinstance(row.get("dataset_video_id"), str)
                    or not isinstance(row.get("dataset_question_id"), str)
                    or not isinstance(row.get("question"), str)
                    or not isinstance(row.get("answer"), str)
                    or not isinstance(row.get("choices"), list)
                    or len(row["choices"]) != 5
                    or any(not isinstance(choice, str) or not choice.strip()
                           for choice in row["choices"])):
                return _blocked("qa_source_row_invalid")
            matches = [i for i, choice in enumerate(row["choices"])
                       if _norm_answer(choice) == _norm_answer(row["answer"])]
            if len(matches) != 1:
                return _blocked("qa_source_gold_binding_invalid")
            qa_by_video[row["video_id"]] = row
        if set(qa_by_video) != set(expected_videos):
            return _blocked("qa_source_cohort_mismatch")
        lineage_by_video = {}
        for row in lineage["records"]:
            if not isinstance(row, dict) or set(row) != LINEAGE_ROW_KEYS:
                return _blocked("lineage_row_schema_invalid")
            if not isinstance(row.get("video_id"), str) or row["video_id"] in lineage_by_video:
                return _blocked("lineage_row_identity_invalid")
            lineage_by_video[row["video_id"]] = row
        if set(lineage_by_video) != set(expected_videos):
            return _blocked("lineage_cohort_mismatch")

        target_doc = json.loads(target_manifest_raw.decode("utf-8"),
                                object_pairs_hook=unique_pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(
                                    ValueError("nonfinite_json")))
        target_rows = target_doc.get("records") if isinstance(target_doc, dict) else None
        if not isinstance(target_rows, list) or len(target_rows) != 3:
            return _blocked("target_manifest_rows_invalid")
        target_by_video = {row.get("video_id"): row for row in target_rows
                           if isinstance(row, dict)}
        if len(target_by_video) != 3 or set(target_by_video) != set(expected_videos):
            return _blocked("target_video_cohort_mismatch")
        canonical_source_inputs = json.dumps(
            lineage.get("dataset_inputs_sha256"), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        source_manifest_sha = hashlib.sha256(canonical_source_inputs).hexdigest()
        if expected_target_binding_hashes.get("dataset_source_manifest_sha256") != source_manifest_sha:
            return _blocked("target_source_manifest_mismatch")
        for video_id in expected_videos:
            source_row = lineage_by_video[video_id]
            sheet_row = worksheet_by_video.get(video_id)
            target_row = target_by_video[video_id]
            qa_row = qa_by_video[video_id]
            qa_source_row_hash = hashlib.sha256(json.dumps(
                qa_row, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            source_choice_hashes = [hashlib.sha256(choice.encode("utf-8")).hexdigest()
                                    for choice in qa_row["choices"]]
            source_gold_index = next(i for i, choice in enumerate(qa_row["choices"])
                                     if _norm_answer(choice) == _norm_answer(qa_row["answer"]))
            if (sheet_row is None
                    or source_row.get("claim_id") != claim_by_video[video_id]
                    or sheet_row.get("claim_id") != source_row.get("claim_id")
                    or sheet_row.get("video_sha256") != source_row.get("video_sha256")
                    or sheet_row.get("claim_text_sha256") != source_row.get("claim_text_sha256")
                    or not isinstance(sheet_row.get("question"), str)
                    or hashlib.sha256(sheet_row["question"].encode("utf-8")).hexdigest()
                    != source_row.get("question_sha256")
                    or qa_row.get("dataset_video_id") != source_row.get("dataset_video_id")
                    or qa_row.get("dataset_question_id") != source_row.get("dataset_question_id")
                    or qa_row.get("question") != sheet_row.get("question")
                    or target_row.get("claim_id") != source_row.get("claim_id")
                    or target_row.get("dataset_qid_sha256") != hashlib.sha256(
                        source_row["dataset_question_id"].encode("utf-8")).hexdigest()
                    or target_row.get("dataset_source_row_sha256") != qa_source_row_hash
                    or target_row.get("question_sha256") != source_row.get("question_sha256")):
                return _blocked("qa_target_source_row_mismatch")
            if (target_row.get("ordered_choice_sha256") != source_choice_hashes
                    or target_row.get("gold_index") != source_gold_index):
                return _blocked("qa_target_choices_or_gold_mismatch")
    except (KeyError, TypeError, AttributeError, IndexError, ValueError,
            UnicodeError, json.JSONDecodeError, OverflowError, RecursionError):
        return _blocked("claim_input_manifest_invalid")

    try:
        assignment = plan_claim_review_assignments(**assignment_inputs)
    except (TypeError, ValueError):
        return _blocked("assignment_plan_invalid")
    if assignment.get("status") != "assignment_plan_structurally_passed_unvalidated":
        return _blocked("assignment_plan_blocked")
    if (not isinstance(expected_target_binding_hashes, dict)
            or set(expected_target_binding_hashes) != BINDING_KEYS
            or expected_target_binding_hashes.get("claim_input_manifest_sha256") != claim_manifest_sha):
        return _blocked("target_expected_bindings_invalid")
    target = validate_target_manifest(
        target_manifest_raw, expected_videos, expected_claims,
        expected_target_binding_hashes)
    if target.get("status") != "target_contract_structurally_valid_unvalidated":
        return _blocked("qa_target_contract_blocked")

    return {
        "status": "combined_preflight_structurally_passed_unvalidated",
        "scope": "synthetic_preflight_rehearsal_only", "failure_codes": [],
        "denominator": 3,
        "assignments": assignment["assignments"],
        "assignment_plan_status": assignment["status"],
        "target_contract_status": target["status"],
        "target_manifest_sha256": target["manifest_sha256"],
        "model_forwards_planned": 9,
        "score_calls": 0, "execution_authorization": "not_granted",
    }
