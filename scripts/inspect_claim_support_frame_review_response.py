#!/usr/bin/env python3
"""Check structure and exact artifact binding for a local frame-review JSON."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

from build_test_split_claim_support_frame_review import validate_decision

FINGERPRINT_RE = re.compile(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">')
ITEMS_RE = re.compile(rb'const items=(\[.*?\]);const sourceProvenance=', re.S)
PROVENANCE_RE = re.compile(rb'const sourceProvenance=(\{.*?\});let nickname=', re.S)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def inspect(response: object, artifact_raw: bytes) -> dict:
    match = FINGERPRINT_RE.search(artifact_raw)
    if not match:
        return {"structurally_valid": False, "failure": "artifact_fingerprint_missing"}
    fingerprint = match.group(1).decode("ascii")
    canonical = artifact_raw.replace(match.group(1), b"__FINGERPRINT__", 1)
    if hashlib.sha256(canonical).hexdigest() != fingerprint:
        return {"structurally_valid": False, "failure": "artifact_fingerprint_invalid"}
    items_match = ITEMS_RE.search(artifact_raw)
    try:
        artifact_items = json.loads(items_match.group(1)) if items_match else None
    except (UnicodeError, json.JSONDecodeError):
        artifact_items = None
    expected_ids = [x.get("item_id") if isinstance(x, dict) else None
                    for x in artifact_items] if isinstance(artifact_items, list) else []
    if not 1 <= len(expected_ids) <= 3 or len(set(expected_ids)) != len(expected_ids):
        return {"structurally_valid": False, "failure": "artifact_item_inventory_invalid"}
    if any(not isinstance(item, dict)
           or not isinstance(item.get("question"), str) or not item["question"].strip()
           or not isinstance(item.get("claim"), str) or not item["claim"].strip()
           or not isinstance(item.get("frames"), list) or len(item["frames"]) != 16
           or any(not isinstance(frame, dict)
                  or not isinstance(frame.get("data"), str)
                  or not frame["data"].startswith("data:image/jpeg;base64,")
                  or type(frame.get("time")) not in (int, float)
                  for frame in item["frames"])
           for item in artifact_items):
        return {"structurally_valid": False, "failure": "artifact_item_shape_invalid"}
    provenance_match = PROVENANCE_RE.search(artifact_raw)
    try:
        provenance = json.loads(provenance_match.group(1)) if provenance_match else None
    except (UnicodeError, json.JSONDecodeError):
        provenance = None
    if (not isinstance(provenance, dict)
            or type(provenance.get("schema_version")) is not int
            or provenance.get("schema_version") != 1):
        return {"structurally_valid": False, "failure": "artifact_source_provenance_invalid"}
    source_items = provenance.get("items")
    if not isinstance(source_items, list) or [x.get("item_id") if isinstance(x, dict) else None
                                               for x in source_items] != expected_ids:
        return {"structurally_valid": False, "failure": "artifact_source_item_mapping_invalid"}
    legacy_keys = {"schema_version", "generator_sha256", "locked_assignment_sha256",
                   "pair_filter_sha256", "translation_sha256", "decoder_sha256", "items"}
    generated_keys = {"schema_version", "purpose", "packet_sha256", "lock_sha256",
                      "pair_filter_sha256", "exclusion_manifest_sha256", "runner_sha256",
                      "model_revision", "generation_prompt_version", "requested_rows",
                      "structured_rows", "unreviewable_attrition_rows", "items"}
    source_hash_keys = {"generator_sha256", "locked_assignment_sha256", "pair_filter_sha256",
                        "translation_sha256", "decoder_sha256"}
    generated_hash_keys = {"packet_sha256", "lock_sha256", "pair_filter_sha256",
                           "exclusion_manifest_sha256", "runner_sha256"}
    if set(provenance) == legacy_keys:
        if any(not isinstance(provenance.get(k), str) or not SHA256_RE.fullmatch(provenance[k])
               for k in source_hash_keys):
            return {"structurally_valid": False, "failure": "artifact_source_provenance_invalid"}
        legacy_item_hash_keys = {"packet_sha256", "video_sha256", "baseline_sha256",
                                 "interaction_sha256", "row_token_sha256"}
        for source in source_items:
            if (not isinstance(source, dict) or set(source) != legacy_item_hash_keys | {"item_id"}
                    or any(not isinstance(source.get(k), str) or not SHA256_RE.fullmatch(source[k])
                           for k in legacy_item_hash_keys)):
                return {"structurally_valid": False, "failure": "artifact_source_digest_invalid"}
        attrition_rows = 0
    elif (set(provenance) == generated_keys
          and provenance.get("purpose") == "generated_claim_support_frame_review_source_manifest"):
        if (any(not isinstance(provenance.get(k), str) or not SHA256_RE.fullmatch(provenance[k])
                for k in generated_hash_keys)
                or not isinstance(provenance.get("model_revision"), str)
                or not isinstance(provenance.get("generation_prompt_version"), str)
                or type(provenance.get("requested_rows")) is not int
                or type(provenance.get("structured_rows")) is not int
                or type(provenance.get("unreviewable_attrition_rows")) is not int
                or provenance["requested_rows"] != provenance["structured_rows"] + provenance["unreviewable_attrition_rows"]
                or provenance["structured_rows"] != len(expected_ids)
                or provenance["unreviewable_attrition_rows"] < 0
                or provenance["requested_rows"] > 3):
            return {"structurally_valid": False, "failure": "artifact_source_provenance_invalid"}
        generated_item_keys = {"record_index", "item_id", "video_sha256", "qid_sha256",
                               "candidate_pair_sha256"}
        for source in source_items:
            if (not isinstance(source, dict) or set(source) != generated_item_keys
                    or type(source.get("record_index")) is not int
                    or source["record_index"] < 0
                    or source["record_index"] >= provenance["requested_rows"]
                    or any(not isinstance(source.get(k), str) or not SHA256_RE.fullmatch(source[k])
                           for k in ("video_sha256", "qid_sha256", "candidate_pair_sha256"))):
                return {"structurally_valid": False, "failure": "artifact_source_digest_invalid"}
        indices = [source["record_index"] for source in source_items]
        if len(indices) != len(set(indices)):
            return {"structurally_valid": False, "failure": "artifact_source_digest_invalid"}
        attrition_rows = provenance["unreviewable_attrition_rows"]
    else:
        return {"structurally_valid": False, "failure": "artifact_source_provenance_invalid"}
    provenance_sha = hashlib.sha256(json.dumps(provenance, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if (not isinstance(response, dict)
            or set(response) != {"schema_version", "purpose", "status", "artifact_fingerprint",
                                 "created_at_utc", "reviewers", "limitations"}
            or type(response.get("schema_version")) is not int
            or response.get("schema_version") != 1
            or response.get("purpose") != "local_claim_support_frame_review"
            or response.get("status") != "human_responses_self_reported_unvalidated"
            or response.get("artifact_fingerprint") != fingerprint):
        return {"structurally_valid": False, "failure": "response_header_or_artifact_binding_invalid"}
    try:
        timestamp = response.get("created_at_utc")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except (ValueError, TypeError):
        return {"structurally_valid": False, "failure": "response_timestamp_invalid"}
    reviewers = response.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers:
        return {"structurally_valid": False, "failure": "reviewers_missing"}
    for reviewer in reviewers:
        if not isinstance(reviewer, dict) or set(reviewer) != {"nickname", "answers"}:
            return {"structurally_valid": False, "failure": "reviewer_schema_invalid"}
        nickname = reviewer.get("nickname")
        if not isinstance(nickname, str) or not nickname.strip() or nickname != nickname.strip() or len(nickname) > 40:
            return {"structurally_valid": False, "failure": "reviewer_nickname_invalid"}
        answers = reviewer.get("answers")
        if not isinstance(answers, list) or len(answers) != len(expected_ids):
            return {"structurally_valid": False, "failure": "reviewer_answer_count_invalid"}
        if [a.get("item_id") if isinstance(a, dict) else None for a in answers] != expected_ids:
            return {"structurally_valid": False, "failure": "review_item_order_or_binding_invalid"}
        for answer in answers:
            if (not isinstance(answer, dict) or set(answer) != {"item_id", "decision", "frame_indices"}
                    or not validate_decision(answer.get("decision"), answer.get("frame_indices"))):
                return {"structurally_valid": False, "failure": "answer_decision_or_frames_invalid"}
    limitations = response.get("limitations")
    if (not isinstance(limitations, list) or len(limitations) < 3
            or any(not isinstance(x, str) or not x.strip() for x in limitations)):
        return {"structurally_valid": False, "failure": "limitations_disclosure_invalid"}
    return {"structurally_valid": True, "reviewer_count": len(reviewers),
            "complete_reviewers": len(reviewers), "items_per_reviewer": len(expected_ids),
            "unreviewable_attrition_rows": attrition_rows,
            "artifact_fingerprint": fingerprint,
            "source_manifest_sha256": provenance_sha,
            "source_manifest_is_signed": False,
            "semantic_support_validated": False, "reviewer_identity_or_independence_verified": False,
            "player_to_model_frame_mapping": "direct model-input frame index; no player clock conversion"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True,
                        help="local downloaded JSON from reviewers")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="exact private HTML artifact used for review")
    args = parser.parse_args(argv)
    try:
        response = json.loads(args.response.read_text(encoding="utf-8"))
        result = inspect(response, args.artifact.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        result = {"structurally_valid": False, "failure": "local_review_files_unreadable"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("structurally_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
