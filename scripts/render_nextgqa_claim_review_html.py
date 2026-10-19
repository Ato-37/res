#!/usr/bin/env python3
"""Render an offline, private human-review page from a frozen 3-row worksheet.

The page contains protected question/claim text. It performs no network calls,
does not embed video bytes, and cannot authorize scoring. Browser downloads may
inherit permissive modes; move the exported JSON to a private directory and set
mode 0600 before use. Optional --claim-input-manifest and --claim-input-sidecar
must be supplied together; when present, their hashes and each displayed claim
are checked against the draft worksheet, then only the two artifact digests are
embedded in non-visible HTML metadata. This remains unreviewed and does not
authorize scoring.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
from typing import Any, Mapping

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from audit_nextgqa_claim_packet_lineage import validate_for_worksheet
from claim_text_quality import claim_text_quality_failure
from private_output_paths import validate_private_output
from build_nextgqa_v2_claim_input_bundle import V2_COHORT_SHA256

MAX_WORKSHEET_BYTES = 1_000_000
MAX_MEDIA_BYTES = 64_000_000
MAX_EMBEDDED_MEDIA_TOTAL_BYTES = 64_000_000
PRIVATE_MEDIA_ROOT = (Path(__file__).resolve().parents[1] / "data" / "private_nextgqa")


@lru_cache(maxsize=1)
def _unicode_casefold_exceptions() -> dict[str, str]:
    """Map lowercased scalars whose Unicode full casefold differs from lowercase."""
    result: dict[str, str] = {}
    for scalar in range(0x110000):
        char = chr(scalar)
        lowered, folded = char.lower(), char.casefold()
        if lowered != folded:
            result[lowered] = folded
    return result
TOP_KEYS = {
    "schema_version", "purpose", "status", "created_at_utc", "source_packet_sha256",
    "cohort_sha256", "model_revision", "source_commit", "claim_packet_runner_sha256",
    "cohort_selector_runner_sha256", "dataset_inputs_sha256", "occurrence_policy",
    "review_procedure_status", "requested_records", "effective_records",
    "claim_records_available", "review_completed", "execution_authorization", "records",
    "claim_review_policy_id", "claim_review_policy_sha256",
}
RECORD_KEYS = {"video_path", "video_sha256", "video_id", "claim_id", "review_token",
               "question", "claim_text", "claim_text_sha256", "review"}
REVIEW_KEYS = {
    "source_identity_verified", "full_source_video_reviewed",
    "question_and_claim_only_context_shown", "gold_and_seed_hidden_until_mapping_frozen",
    "outcome_scores_not_accessed", "occurrence_policy_applied", "support_decision",
    "evidence_intervals_seconds", "rationale", "reviewer_role",
    "review_record_reference", "first_played_at_utc", "reviewed_at_utc", "review_status",
}
DATASET_KEYS = {"test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json"}
CLAIM_MANIFEST_KEYS = {"schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
                       "claim_policy_sha256", "occurrence_policy", "claims"}
CLAIM_ROW_KEYS = {"video_id", "claim_id", "claim_text", "claim_text_sha256",
                  "model_revision", "generation_run_sha256"}
BUNDLE_RECORD_KEYS = {"video_id", "dataset_video_id", "dataset_question_id", "video_sha256",
                      "question_sha256", "claim_id", "claim_text_sha256"}
SIDECAR_KEYS = {
    "schema_version", "purpose", "status", "scope", "review_completed", "scoring_ready",
    "execution_authorization", "eligible_dataset_id", "cohort_sha256", "cohort_digest_check",
    "occurrence_policy", "policy_id", "policy_sha256", "model", "model_revision",
    "v1_generation_run_sha256_semantics", "worksheet_binding_check", "input_sha256",
    "claim_input_manifest_sha256", "records",
}
SIDECAR_INPUT_HASH_KEYS = {
    "eligible_manifest", "generation_packet", "lineage_manifest", "policy", "test_csv",
    "video_map", "worksheet",
}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def _validate_claim_bundle(payload: dict, worksheet_raw: bytes,
                           lineage: dict, lineage_raw: bytes,
                           manifest: Any, manifest_raw: bytes,
                           sidecar: Any, sidecar_raw: bytes) -> tuple[str, str]:
    """Bind displayed worksheet bytes/rows to a candidate v2 claim manifest."""
    try:
        parsed_manifest = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=_unique_object)
        parsed_sidecar = json.loads(sidecar_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (AttributeError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("claim_bundle_json_invalid") from None
    if parsed_manifest != manifest or parsed_sidecar != sidecar:
        raise ValueError("claim_bundle_raw_binding_invalid")
    if (not isinstance(manifest, dict) or set(manifest) != CLAIM_MANIFEST_KEYS
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or not isinstance(manifest.get("eligible_dataset_id"), str)
            or not manifest["eligible_dataset_id"]
            or manifest["eligible_dataset_id"] != manifest["eligible_dataset_id"].strip()
            or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("eligible_manifest_sha256", "")))
            or manifest.get("claim_policy_sha256") != FROZEN_POLICY_SHA256
            or manifest.get("occurrence_policy") != "all_relevant_occurrences"
            or not isinstance(manifest.get("claims"), list) or len(manifest["claims"]) != 3):
        raise ValueError("claim_manifest_schema_or_policy_invalid")
    if (not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS
            or type(sidecar.get("schema_version")) is not int
            or sidecar.get("schema_version") != 1
            or sidecar.get("purpose") != "private_nextgqa_v2_claim_input_bundle_provenance"
            or sidecar.get("status") != "candidate_unreviewed"
            or sidecar.get("scope") != "structural_only"
            or sidecar.get("review_completed") is not False
            or sidecar.get("scoring_ready") is not False
            or sidecar.get("execution_authorization") != "not_granted"
            or not isinstance(sidecar.get("eligible_dataset_id"), str)
            or not sidecar["eligible_dataset_id"]
            or sidecar["eligible_dataset_id"] != sidecar["eligible_dataset_id"].strip()
            or sidecar.get("cohort_sha256") != V2_COHORT_SHA256
            or sidecar.get("cohort_sha256") != payload.get("cohort_sha256")
            or sidecar.get("cohort_digest_check") !=
                "pinned_value_and_input_declarations_match_not_recomputed_from_full_planner_inputs"
            or sidecar.get("occurrence_policy") != payload.get("occurrence_policy")
            or sidecar.get("policy_id") != payload.get("claim_review_policy_id")
            or sidecar.get("policy_sha256") != FROZEN_POLICY_SHA256
            or sidecar.get("model") != "Qwen/Qwen2.5-VL-7B-Instruct"
            or sidecar.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
            or sidecar.get("model_revision") != payload.get("model_revision")
            or sidecar.get("v1_generation_run_sha256_semantics") !=
                "generation_packet_container_byte_digest_not_unique_run_id"
            or sidecar.get("worksheet_binding_check") !=
                "claim_and_identity_hashes_matched_only_not_review_or_custody_validation"):
        raise ValueError("claim_bundle_sidecar_status_or_protocol_invalid")
    inputs = sidecar.get("input_sha256")
    if (not isinstance(inputs, dict) or set(inputs) != SIDECAR_INPUT_HASH_KEYS
            or any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                   for digest in inputs.values())
            or inputs.get("worksheet") != hashlib.sha256(worksheet_raw).hexdigest()
            or inputs.get("lineage_manifest") != hashlib.sha256(lineage_raw).hexdigest()
            or inputs.get("policy") != FROZEN_POLICY_SHA256
            or not isinstance(inputs.get("generation_packet"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", inputs["generation_packet"])
            or not re.fullmatch(r"[a-f0-9]{64}", str(sidecar.get("claim_input_manifest_sha256", "")))
            or sidecar.get("claim_input_manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()):
        raise ValueError("claim_bundle_artifact_hash_mismatch")
    if (manifest.get("eligible_dataset_id") != sidecar.get("eligible_dataset_id")
            or manifest.get("eligible_manifest_sha256") != inputs.get("eligible_manifest")
            or manifest.get("claim_policy_sha256") != sidecar.get("policy_sha256")):
        raise ValueError("claim_bundle_manifest_binding_invalid")
    if not isinstance(sidecar.get("records"), list) or len(sidecar["records"]) != 3:
        raise ValueError("claim_bundle_records_invalid")
    worksheet_by_video = {row["video_id"]: row for row in payload["records"]}
    lineage_by_video = {row["video_id"]: row for row in lineage["records"]}
    sidecar_by_video = {}
    for row in sidecar["records"]:
        if not isinstance(row, dict) or set(row) != BUNDLE_RECORD_KEYS:
            raise ValueError("claim_bundle_record_schema_invalid")
        video_id = row.get("video_id")
        if video_id in sidecar_by_video:
            raise ValueError("claim_bundle_video_duplicate")
        sidecar_by_video[video_id] = row
    manifest_videos = set()
    for claim in manifest["claims"]:
        if not isinstance(claim, dict) or set(claim) != CLAIM_ROW_KEYS:
            raise ValueError("claim_manifest_row_schema_invalid")
        video_id = claim.get("video_id")
        if video_id in manifest_videos or video_id not in worksheet_by_video:
            raise ValueError("claim_manifest_cohort_mismatch")
        manifest_videos.add(video_id)
        worksheet_row = worksheet_by_video[video_id]
        lineage_row = lineage_by_video.get(video_id)
        bundle_row = sidecar_by_video.get(video_id)
        if lineage_row is None or bundle_row is None:
            raise ValueError("claim_bundle_lineage_join_missing")
        try:
            text_sha = hashlib.sha256(claim["claim_text"].encode("utf-8")).hexdigest()
        except (AttributeError, UnicodeEncodeError):
            raise ValueError("claim_manifest_text_invalid") from None
        if (not isinstance(claim.get("generation_run_sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", claim["generation_run_sha256"])
                or claim["generation_run_sha256"] != inputs["generation_packet"]
                or claim.get("model_revision") != sidecar.get("model") + "@" + payload["model_revision"]
                or worksheet_row.get("claim_text") != claim.get("claim_text")
                or worksheet_row.get("claim_text_sha256") != text_sha
                or claim.get("claim_text_sha256") != text_sha
                or worksheet_row.get("claim_id") != claim.get("claim_id")
                or worksheet_row.get("video_sha256") != lineage_row.get("video_sha256")
                or bundle_row.get("claim_id") != claim.get("claim_id")
                or bundle_row.get("claim_text_sha256") != text_sha
                or bundle_row.get("video_sha256") != lineage_row.get("video_sha256")
                or bundle_row.get("dataset_video_id") != lineage_row.get("dataset_video_id")
                or bundle_row.get("dataset_question_id") != lineage_row.get("dataset_question_id")
                or bundle_row.get("question_sha256") != lineage_row.get("question_sha256")):
            raise ValueError("claim_manifest_worksheet_row_mismatch")
    if manifest_videos != set(worksheet_by_video) or manifest_videos != set(sidecar_by_video):
        raise ValueError("claim_bundle_cohort_mismatch")
    return hashlib.sha256(manifest_raw).hexdigest(), hashlib.sha256(sidecar_raw).hexdigest()


def _read_private_media(path: Path) -> bytes:
    """Read one owner-only regular video file without following symlinks."""
    fd = -1
    try:
        before = path.lstat()
        if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= MAX_MEDIA_BYTES):
            raise ValueError("embedded_media_file_permissions_or_size_invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_size != before.st_size):
            raise ValueError("embedded_media_file_changed_or_invalid")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, MAX_MEDIA_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MEDIA_BYTES:
                raise ValueError("embedded_media_file_too_large")
        after = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
             opened.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns) or size != after.st_size):
            raise ValueError("embedded_media_file_changed_during_read")
        return b"".join(chunks)
    except OSError:
        raise ValueError("embedded_media_file_unreadable") from None
    finally:
        if fd >= 0:
            os.close(fd)


def load_private_media_directory(media_dir: Path, payload: dict,
                                 custody: dict) -> dict[str, bytes]:
    """Load exactly the three custodian-token files from private local storage."""
    try:
        root_info = PRIVATE_MEDIA_ROOT.lstat()
        root = PRIVATE_MEDIA_ROOT.resolve(strict=True)
        supplied = Path(os.path.abspath(media_dir))
        supplied_info = supplied.lstat()
        path = supplied.resolve(strict=True)
        if (stat.S_ISLNK(root_info.st_mode) or root_info.st_uid != os.geteuid()
                or stat.S_IMODE(root_info.st_mode) != 0o700
                or stat.S_ISLNK(supplied_info.st_mode) or supplied != path):
            raise ValueError("embedded_media_directory_invalid")
        path.relative_to(root)
        info = path.lstat()
        if (path == root or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("embedded_media_directory_invalid")
        entries = custody["records"]
        expected_names = {entry["media_file"] for entry in entries}
        actual_names = {child.name for child in path.iterdir()}
        if actual_names != expected_names:
            raise ValueError("embedded_media_directory_contents_invalid")
        result = {}
        for row, entry in zip(payload["records"], entries, strict=True):
            result[row["review_token"]] = _read_private_media(path / entry["media_file"])
        return result
    except OSError:
        raise ValueError("embedded_media_directory_unavailable") from None


def render(payload: Any, raw: bytes, lineage: Any, lineage_raw: bytes,
           expected_lineage_sha256: str, custody: Any,
           custody_raw: bytes,
           expected_custody_sha256: str,
           claim_input_manifest: Any = None, claim_input_manifest_raw: bytes | None = None,
           bundle_sidecar: Any = None, bundle_sidecar_raw: bytes | None = None,
           media_files: Mapping[str, bytes] | None = None) -> bytes:
    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if (len(raw) > MAX_WORKSHEET_BYTES or not isinstance(payload, dict) or parsed != payload
            or set(payload) != TOP_KEYS
            or payload.get("status") != "draft_not_reviewed"
            or payload.get("review_completed") is not False
            or payload.get("execution_authorization") != "not_granted"
            or payload.get("effective_records") != 3
            or payload.get("requested_records") != 3
            or payload.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or payload.get("review_procedure_status") != "frozen_pre_review"
            or payload.get("occurrence_policy") != "all_relevant_occurrences"
            or not isinstance(payload.get("records"), list)
            or len(payload["records"]) != 3):
        raise ValueError("worksheet_not_review_ready")
    if (not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("cohort_sha256", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("source_packet_sha256", "")))
            or not isinstance(payload.get("model_revision"), str)
            or not payload["model_revision"].strip()
            or payload.get("claim_review_policy_id") != "nextgqa_claim_review_v6"
            or not isinstance(payload.get("dataset_inputs_sha256"), dict)
            or set(payload["dataset_inputs_sha256"]) != DATASET_KEYS
            or any(not re.fullmatch(r"[a-f0-9]{64}", str(x))
                   for x in payload["dataset_inputs_sha256"].values())):
        raise ValueError("worksheet_provenance_invalid")
    seen_videos, seen_claims, seen_tokens = set(), set(), set()
    for row in payload["records"]:
        if (not isinstance(row, dict) or set(row) != RECORD_KEYS
                or not isinstance(row.get("review"), dict)
                or set(row["review"]) != REVIEW_KEYS
                or any(row["review"].get(k) is not None for k in REVIEW_KEYS - {"review_status"})
                or not isinstance(row.get("question"), str)
                or not isinstance(row.get("claim_text"), str)
                or row.get("review", {}).get("review_status") != "not_started"):
            raise ValueError("worksheet_record_invalid")
        video_hash = row.get("video_sha256")
        claim_hash = row.get("claim_text_sha256")
        try:
            actual_claim_hash = hashlib.sha256(row["claim_text"].encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            raise ValueError("worksheet_record_invalid") from None
        if (not isinstance(video_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", video_hash)
                or not isinstance(claim_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", claim_hash)
                or claim_hash != actual_claim_hash
                or not row["question"].strip() or not row["claim_text"].strip()
                or claim_text_quality_failure(row["claim_text"]) is not None
                or len(row["question"]) > 20_000 or len(row["claim_text"]) > 20_000
                or not isinstance(row.get("video_id"), str)
                or row["video_id"] != "video-" + video_hash
                or not isinstance(row.get("claim_id"), str)
                or not re.fullmatch(r"claim-[a-f0-9]{64}", row["claim_id"])
                or not isinstance(row.get("review_token"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{24}", row["review_token"])
                or row.get("video_id") in seen_videos or row.get("claim_id") in seen_claims
                or row.get("review_token") in seen_tokens
                or row.get("review")["review_status"] != "not_started"):
            raise ValueError("worksheet_record_invalid")
        seen_videos.add(row.get("video_id")); seen_claims.add(row.get("claim_id"))
        seen_tokens.add(row.get("review_token"))
    lineage_sha = validate_for_worksheet(payload, lineage, lineage_raw)
    if (not re.fullmatch(r"[a-f0-9]{64}", expected_lineage_sha256)
            or lineage_sha != expected_lineage_sha256):
        raise ValueError("lineage_manifest_sha256_mismatch")
    if (custody is None or custody_raw is None
            or not re.fullmatch(r"[a-f0-9]{64}", str(expected_custody_sha256))):
        raise ValueError("custody_manifest_required")
    if custody is not None or custody_raw is not None or expected_custody_sha256 is not None:
        custody_keys = {"schema_version", "purpose", "created_at_utc", "worksheet_sha256",
                        "source_lineage_manifest_sha256", "claim_review_policy_sha256", "record_count", "records"}
        try:
            custody_parsed = json.loads(custody_raw.decode("utf-8"), object_pairs_hook=_unique_object)
        except (AttributeError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("custody_manifest_invalid") from None
        if (custody_parsed != custody or not isinstance(custody, dict) or set(custody) != custody_keys
                or hashlib.sha256(custody_raw).hexdigest() != expected_custody_sha256
                or custody.get("schema_version") != 1 or custody.get("record_count") != 3
                or custody.get("purpose") != "private_custodian_token_to_verified_media_manifest"
                or custody.get("worksheet_sha256") != hashlib.sha256(raw).hexdigest()
                or custody.get("source_lineage_manifest_sha256") != lineage_sha
                or custody.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
                or not isinstance(custody.get("records"), list) or len(custody["records"]) != 3):
            raise ValueError("custody_manifest_binding_invalid")
        for row, entry in zip(payload["records"], custody["records"], strict=True):
            if (not isinstance(entry, dict)
                    or set(entry) != {"review_token", "row_ordinal", "media_file", "source_media_sha256",
                                      "staged_media_sha256", "size_bytes", "media_check_salt", "media_check_sha256"}
                    or entry.get("row_ordinal") != payload["records"].index(row) + 1
                    or entry.get("source_media_sha256") != row["video_sha256"]
                    or entry.get("staged_media_sha256") != row["video_sha256"]
                    or not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0
                    or entry["size_bytes"] > 64_000_000
                    or entry.get("review_token") != row["review_token"]
                    or entry.get("media_file") != row["review_token"] + ".mp4"
                    or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("media_check_salt", "")))
                    or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("media_check_sha256", "")))):
                raise ValueError("custody_media_token_binding_invalid")
    supplied_bundle = (claim_input_manifest is not None, claim_input_manifest_raw is not None,
                       bundle_sidecar is not None, bundle_sidecar_raw is not None)
    bundle_hashes = None
    if any(supplied_bundle):
        if not all(supplied_bundle):
            raise ValueError("claim_bundle_pair_required")
        bundle_hashes = _validate_claim_bundle(
            payload, raw, lineage, lineage_raw, claim_input_manifest,
            claim_input_manifest_raw, bundle_sidecar, bundle_sidecar_raw)
    # HTML script data is escaped so claims cannot terminate the data element.
    page_payload = {
        "schema_version": 1,
        "purpose": "anonymous_local_claim_video_review",
        "execution_authorization": "not_granted",
        "reviewer_id_casefold_exceptions": _unicode_casefold_exceptions(),
        "reviewer_id_strip_codepoints": [scalar for scalar in range(0x110000)
                                          if chr(scalar).isspace()],
        "embedded_media": media_files is not None,
        "claim_review_policy_id": payload["claim_review_policy_id"],
        "claim_review_policy_sha256": payload["claim_review_policy_sha256"],
        "occurrence_policy": payload["occurrence_policy"],
        "records": [],
    }
    page_payload["records"] = []
    if media_files is not None:
        expected_tokens = {row["review_token"] for row in payload["records"]}
        if (not isinstance(media_files, Mapping) or set(media_files) != expected_tokens
                or any(not isinstance(value, bytes) for value in media_files.values())):
            raise ValueError("embedded_media_token_set_invalid")
        if sum(len(value) for value in media_files.values()) > MAX_EMBEDDED_MEDIA_TOTAL_BYTES:
            raise ValueError("embedded_media_total_size_exceeded")
    for row in payload["records"]:
        item = {key: row[key] for key in ("review_token", "question", "claim_text")}
        custody_entry = custody["records"][len(page_payload["records"])]
        item["media_check_salt"] = custody_entry["media_check_salt"]
        item["media_check_digest"] = custody_entry["media_check_sha256"]
        if media_files is not None:
            video_bytes = media_files[row["review_token"]]
            salt = bytes.fromhex(custody_entry["media_check_salt"])
            if (not video_bytes or len(video_bytes) > MAX_MEDIA_BYTES
                    or len(video_bytes) != custody_entry["size_bytes"]
                    or hashlib.sha256(video_bytes).hexdigest() != row["video_sha256"]
                    or hashlib.sha256(salt + video_bytes).hexdigest() != custody_entry["media_check_sha256"]):
                raise ValueError("embedded_media_custody_digest_mismatch")
            item["video_base64"] = base64.b64encode(video_bytes).decode("ascii")
        page_payload["records"].append(item)
    data = json.dumps(page_payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = r'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src blob:; connect-src 'none'; img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
PROVENANCE
<title>Private offline video review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#18202a}
.notice{background:#fff2cc;padding:1rem;border:1px solid #d6a900} article{border:1px solid #aaa;padding:1rem;margin:1rem 0}
 video{display:block;max-width:100%;max-height:65vh;background:#111;margin:1rem 0} label{display:block;margin:.65rem 0}
textarea{width:100%;min-height:4rem} input[type=number]{width:8rem} button{margin:.25rem;padding:.45rem}
.claim{white-space:pre-wrap;background:#eef5ff;padding:.7rem}.bad{color:#a00}.ok{color:#075}
</style>
<h1>動画主張の人手レビュー</h1>
<p class="notice">この画面はローカル専用です。質問とモデル生成主張だけを表示し、QA正解・選択肢・gold区間・モデルの選択フレーム・スコアは表示しません。MEDIA_NOTICE 公開データセットの索引や回答を調べず、見た内容だけで判定してください。回答はレビュアーごとに別のJSON・index・任意のclock記録を作り、最後に1つのJSON envelopeへ入れます。採点許可にはなりません。</p>
<p class="notice">各動画の初回再生時刻と、必須レビュー入力が揃った時点を行ごとに記録します。いずれもブラウザの自己申告ログであり、実際の視聴や時系列を独立に証明しません。</p>
<p class="notice">レビューJSONには、上位項目のschema_version、purpose、status、review_completed、execution_authorization、claim_review_policy_id、claim_review_policy_sha256、human_reviewer_role、human_reviewer_private_id、review_environment（browser_user_agent、offline_static_page、network_fetch_available）、independence_attestation、export_disclosure_acknowledged、recordsが含まれます。各record wrapperはrecords:[{review_token,review}]で、reviewにはfull_source_video_reviewed、question_and_claim_only_context_shown、gold_and_seed_hidden_until_mapping_frozen、outcome_scores_not_accessed、media_bytes_verified、occurrence_policy_applied、support_decision、evidence_intervals_seconds、interval_rationales、media_duration_seconds、reviewer_role、first_played_at_utc、reviewed_at_utc、review_status、display_player、claim_review_policy_sha256が含まれます。interval_rationalesは非公開の自由記述です。これらの記録は自己申告で、独立に検証された事実ではありません。保存前にJSONをcustodianへ渡すことと、根拠メモに動画内容が記録されることを確認してください。</p>
<p class="notice">追加の出力もあります。完全性indexにはschema_version、purpose、review_payload_sha256、claim_review_policy_sha256、created_at_utc、review_status、execution_authorizationが含まれます。任意の時計観測JSONにはschema_version、purpose、status、capture_policy_id、capture_thresholds（max_callback_age_ms、max_request_delay_ms、max_click_player_time_delta_seconds、max_callback_player_time_delta_seconds、callback_wait_timeout_ms）、player_clock、presented_frame_metadata、decoder_frame_identity、created_at_utc、browser_user_agent、anchor_id_validation、predeclared_plan_binding、fit_heldout_split、decoder_pts_pairing、execution_authorization、recordsが含まれ、clock record wrapperはrecords:[{review_token,events}]で、各eventはanchor_id、sequence、requested_player_time_seconds、request_started_performance_ms、html5_current_time_seconds、request_video_frame_callback_metadata（media_time_seconds、presented_frames、presentation_time_ms、expected_display_time_ms、width、height、callback_now_ms）、frame_match_status、performance_now_ms、media_duration_seconds、playback_rate、muted、volume、text_tracks_disabled、paused、seeking、ready_state、current_src_is_staged_media、visibility_state、captured_at_utcを含みます。capture_policy_idはrvfc_click_bound_v1で、callback-to-capture ageとclick-to-callback delayは各250 ms以下、クリック時刻とcallback時刻のmediaTime差は各0.1秒以下、callback待ち時間は最大1000 msです。時計JSONの別indexにはschema_version、purpose、payload_sha256、created_at_utc、status、execution_authorizationが含まれます。これらは全て自己申告/ブラウザ観測値であり、身元・視聴・decoder PTSとの対応を証明しません。</p>
<p class="notice">v6方針は上記のレビューJSON、完全性index、任意の時計観測JSON/indexの出力項目を定めています。担当者欄には本名ではなく非公開のレビュアー識別子を入力してください。区間rationaleには動画内容が含まれるため、書き出した全ファイルとメモは非公開のcustody領域だけに保存してください。各情報は自己申告/ブラウザ観測であり、身元・視聴・decoder frame対応を独立に証明しません。export_disclosure_acknowledgedは、全出力項目とローカル保存条件を読んだという自己申告です。納得できない場合は再生・入力を始めず、ページを閉じてください。</p>
<p class="notice">音声と字幕トラックは無効化します。映像に焼き付いた字幕・キャプション・テロップは残ることがありますが、根拠として使わないでください。文字 overlay だけが主張を支える場合、その主張は不支持として判定します。</p>
<p>方針 <code>nextgqa_claim_review_v6</code> / occurrence: <code>all_relevant_occurrences</code> / policy SHA-256: <code>POLICY</code></p>
<div id="rows"></div>
<section id="reviewer-controls">
<h2>レビュー担当者</h2>
<label>非公開のレビュアー識別子（本名は避けてください） <input id="reviewer" autocomplete="off"></label>
<label>役割 <select id="role"><option value="">選択</option><option value="independent_human">claim生成・採点から独立した人間レビュアー</option></select></label>
<label><input type="checkbox" id="attest"> 私は人間であり、claim生成・採点・custodyから独立しています。照合manifestを見ておらず、QA正解やgold区間にもアクセスせず、方針に従い全3動画を自分でレビューしました。</label>
<label><input type="checkbox" id="export-disclosure"> v6方針とレビューJSON・完全性index・任意の時計記録・まとめたenvelopeを確認し、非公開のcustody場所だけに保管します。</label>
<p id="snapshot-notice" class="notice" hidden>前の担当者までの回答を一時保存できます。次の担当者の入力はまだ含まれません。途中で閉じる前に保存してください。<button id="snapshot-download" type="button">ここまでの回答をJSONで保存</button></p><button id="export" disabled>この担当者の回答を確定</button><span id="global"></span>
</section>
<section id="session-handoff" hidden>
<p id="handoff-status" class="notice" role="status" aria-live="polite">この担当者の記録を確定しました。</p>
<p id="reviewer-count"></p><p class="notice">同じ画面を引き継ぐ操作は、レビュアー間の独立性や会話を防げません。氏名・役割・独立性は自己申告のままで、署名・立会い・custodianの検証が別途必要です。</p>
<button id="next-reviewer">次の担当者へ</button><button id="download-envelope">全員分を1つのJSONで保存</button><button id="retry-envelope" disabled>envelopeを再ダウンロード</button>
</section>
<p>保存後のJSONは private な場所へ移し、ファイル権限を 0600 にしてください。人手記録はモデル入力・動画の真正性・権利・PTS時計校正・SESOI・採点許可を証明しません。</p>
<p>HTML5時計観測ログはclaim根拠区間と別に保存します。事前定義アンカー付近を通常速度で表示し、クリック時のplayer時刻と単調時計を記録します。ブラウザがクリック後1秒以内に新しいrequestVideoFrameCallbackを返した場合でも、click-to-callback delayとcallback-to-capture ageが各250 ms以下、rVFC mediaTimeがクリック時およびcallback時のplayer時刻から各0.1秒以内の場合だけ記録します。表示元・可視性・seek状態も検証し、条件を満たさない記録は保存せず再試行を求めます。この機械的な鮮度/近接チェックもブラウザ観測を認証せず、表示フレームとdecoder PTSの同一性を証明しません。フレーム対応・fit/held-out分割・事前固定済みの根拠・不確かさ評価が揃うまで校正値として使えません。</p>
<script id="data" type="application/json">DATA</script>
<script>
'use strict';
const base=JSON.parse(document.getElementById('data').textContent);
const MAX_MEDIA_BYTES=64000000;
const MAX_ENVELOPE_MEMBER_BYTES=2097152,MAX_ENVELOPE_BYTES=67108864;
const rows=document.getElementById('rows'), state=base.records.map(()=>({tokenOK:false,full:false,context:false,seedHidden:false,outcomeHidden:false,decision:'',intervals:[],clockEvents:[],duration:null,mime:'',firstPlayedAt:null,reviewedAt:null}));
const rowResetters=[],completedSessions=[];
const decisions=[['supports_claim','主張を支持'],['partial','一部のみ支持'],['contradicted','動画と矛盾'],['unsupported','支持なし'],['ambiguous','曖昧'],['unlocalizable','時刻特定不能']];
function el(tag,attrs={},text=''){const e=document.createElement(tag);for(const [k,v] of Object.entries(attrs)){if(k==='className')e.className=v;else e.setAttribute(k,v)}if(text)e.textContent=text;return e}
let exportInProgress=false,pendingEnvelope=null;
function reviewerKey(value){const exceptions=base.reviewer_id_casefold_exceptions;return [...value.normalize('NFKC').toLowerCase()].map(ch=>exceptions[ch]||ch).join('')}
function validReviewer(value){const chars=[...value],spaces=new Set(base.reviewer_id_strip_codepoints);return chars.length>0&&chars.length<=256&&!spaces.has(chars[0].codePointAt(0))&&!spaces.has(chars.at(-1).codePointAt(0))&&!/[\u0000-\u001f\u007f]/.test(value)}
function checkReady(){for(const s of state){const done=s.tokenOK&&Number.isFinite(s.duration)&&s.duration>0&&s.firstPlayedAt&&s.full&&s.context&&s.seedHidden&&s.outcomeHidden&&s.decision&&(s.decision!=='supports_claim'||s.intervals.length>0);if(done&&!s.reviewedAt)s.reviewedAt=new Date().toISOString();if(!done)s.reviewedAt=null}const reviewer=document.getElementById('reviewer').value,duplicate=completedSessions.some(x=>x.reviewerKey===reviewerKey(reviewer));const ready=completedSessions.length<16&&validReviewer(reviewer)&&state.every(s=>s.tokenOK&&Number.isFinite(s.duration)&&s.duration>0&&s.firstPlayedAt&&s.reviewedAt&&s.full&&s.context&&s.seedHidden&&s.outcomeHidden&&s.decision)&&reviewer&&document.getElementById('role').value==='independent_human'&&document.getElementById('attest').checked&&document.getElementById('export-disclosure').checked&&!duplicate;document.getElementById('export').disabled=!ready||exportInProgress;document.getElementById('global').textContent=completedSessions.length>=16?'最大16人分が確定済みです。envelopeを保存してください。':!validReviewer(reviewer)?'識別子は空白を除き1〜256文字で、前後空白や制御文字は使えません。':duplicate?'同じ識別子は再使用できません。':ready?' 全項目の入力がそろいました。':' 3件すべてのレビュー記録が必要です。'}
base.records.forEach((r,i)=>{
 const s=state[i],box=el('article');box.append(el('h2',{},`匿名動画 ${i+1} — ${r.review_token}.mp4`));
 box.append(el('h3',{},'質問'));box.append(el('p',{className:'claim'},r.question));box.append(el('h3',{},'固定されたモデル生成主張'));box.append(el('p',{className:'claim'},r.claim_text));
 const file=base.embedded_media?null:el('input',{type:'file',accept:'video/mp4'});if(file){box.append(el('label',{},'表示されたtoken名のcustodian動画ファイルを選択'));box.lastChild.append(file)}
 const status=el('p',{},'動画未選択');box.append(status);const video=el('video',{playsinline:''});video.muted=true;video.volume=0;video.addEventListener('volumechange',()=>{if(!video.muted||video.volume!==0){video.muted=true;video.volume=0}});video.textTracks?.addEventListener?.('addtrack',e=>{e.track.mode='disabled'});let objectUrl='',selection=0,metadataSelection=null;video.addEventListener('loadedmetadata',()=>{if(metadataSelection===selection&&s.tokenOK&&objectUrl&&video.currentSrc===objectUrl){s.duration=video.duration;checkReady()}});box.append(video);
 const play=el('button',{},'再生 / 一時停止');play.type='button';play.onclick=()=>video.paused?video.play():video.pause();video.addEventListener('play',()=>{if(s.tokenOK&&s.firstPlayedAt===null)s.firstPlayedAt=new Date().toISOString();checkReady()});const seek=el('input',{type:'range',min:'0',max:'1000',value:'0'});seek.addEventListener('input',()=>{if(video.duration>0)video.currentTime=video.duration*(Number(seek.value)/1000)});video.addEventListener('timeupdate',()=>{if(video.duration>0)seek.value=String(Math.round(video.currentTime/video.duration*1000))});box.append(play,seek,el('p',{},'音声は強制ミュート。字幕トラックは無効。再生速度は通常速度に固定。'));
 const flagInputs=[];
 if(file)file.addEventListener('change',async()=>{const current=++selection;metadataSelection=null;clockCapturePending=false;clockCaptureGeneration++;s.tokenOK=false;s.duration=null;s.mime='';s.full=false;s.context=false;s.seedHidden=false;s.outcomeHidden=false;s.decision='';s.intervals=[];s.clockEvents=[];s.firstPlayedAt=null;s.reviewedAt=null;for(const cb of flagInputs)cb.checked=false;if(typeof select!=='undefined')select.value='';start.value='';end.value='';reason.value='';anchorId.value='';redraw();refreshClockButton();status.textContent='匿名ファイル名とcustodian照合値を確認中…';video.pause();if(objectUrl){URL.revokeObjectURL(objectUrl);objectUrl=''}video.removeAttribute('src');video.load();checkReady();try{const f=file.files[0];if(!f||f.size>MAX_MEDIA_BYTES||f.name!==r.review_token+'.mp4')throw Error('opaque_token_filename_mismatch');const payload=new Uint8Array(await f.arrayBuffer()),salt=Uint8Array.from(r.media_check_salt.match(/../g),x=>parseInt(x,16)),joined=new Uint8Array(salt.length+payload.length);joined.set(salt);joined.set(payload,salt.length);const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',joined))).map(x=>x.toString(16).padStart(2,'0')).join('');if(current!==selection)return;if(actual!==r.media_check_digest)throw Error('custody_media_digest_mismatch');objectUrl=URL.createObjectURL(f);metadataSelection=current;video.src=objectUrl;video.load();video.playbackRate=1;video.muted=true;s.mime=f.type;s.tokenOK=true;status.textContent='匿名tokenとcustodian照合値が一致しました。';status.className='ok'}catch(e){if(current!==selection)return;video.removeAttribute('src');video.load();s.tokenOK=false;status.textContent='ファイル名・内容照合に失敗、または動画が64 MBを超過しました。選択し直してください。';status.className='bad'}checkReady()});
 if(!file){status.textContent='埋め込み動画の匿名token・容量・照合値を確認中…';(async()=>{try{const binary=atob(r.video_base64);if(binary.length>MAX_MEDIA_BYTES)throw Error('embedded_media_too_large');const payload=new Uint8Array(binary.length);for(let j=0;j<binary.length;j++)payload[j]=binary.charCodeAt(j);const salt=Uint8Array.from(r.media_check_salt.match(/../g),x=>parseInt(x,16)),joined=new Uint8Array(salt.length+payload.length);joined.set(salt);joined.set(payload,salt.length);const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',joined))).map(x=>x.toString(16).padStart(2,'0')).join('');if(actual!==r.media_check_digest)throw Error('custody_media_digest_mismatch');objectUrl=URL.createObjectURL(new Blob([payload],{type:'video/mp4'}));metadataSelection=selection;video.src=objectUrl;video.load();video.playbackRate=1;video.muted=true;s.mime='video/mp4';s.tokenOK=true;status.textContent='埋め込み動画の照合値が一致しました。';status.className='ok'}catch(_error){s.tokenOK=false;status.textContent='埋め込み動画の容量または照合値を確認できません。';status.className='bad'}checkReady()})()}
 const flags=[['full','動画全体を通常速度で視聴した'],['context','表示された質問と主張だけを回答コンテキストとして用いた'],['seedHidden','gold回答とQA seed区間をマッピング凍結まで見なかった'],['outcomeHidden','採点結果を見ていない']];
 for(const [key,label] of flags){const cb=el('input',{type:'checkbox'});flagInputs.push(cb);cb.addEventListener('change',()=>{s[key]=cb.checked;s.reviewedAt=null;checkReady()});box.append(el('label'));box.lastChild.append(cb,document.createTextNode(' '+label))}
 const select=el('select');select.append(el('option',{value:''},'判定を選択'));for(const [v,t] of decisions)select.append(el('option',{value:v},t));select.addEventListener('change',()=>{s.decision=select.value;s.reviewedAt=null;checkReady()});box.append(el('label',{},'主張全体の視覚的支持判定'));box.lastChild.append(select);
 box.append(el('p',{},'supports_claimの場合、全ての重要な事実要素が直接見える時間を記録します。反復する根拠は全て別区間で記録します。入力値は0.001秒単位ですが、映像上の根拠の時間分解能はフレーム間隔（この3件は約33–40 ms）に制約されるため、1 ms精度の測定を意味しません。'));const intervals=el('div');box.append(intervals);
 function redraw(){intervals.replaceChildren();s.intervals.forEach((it,j)=>{const row=el('div');row.append(document.createTextNode(`区間${j+1}: ${it.start.toFixed(3)}–${it.end.toFixed(3)}秒 — ${it.rationale} `));const del=el('button',{},'削除');del.type='button';del.onclick=()=>{s.intervals.splice(j,1);s.reviewedAt=null;redraw();checkReady()};row.append(del);intervals.append(row)})}
 const start=el('input',{type:'number',min:'0',step:'0.001'}),end=el('input',{type:'number',min:'0',step:'0.001'}),reason=el('textarea');
 const anchorId=el('input',{type:'text',maxLength:'40',autocomplete:'off',class:'clock-control'}),clockButton=el('button',{class:'clock-control'},'HTML5時計アンカー観測を追加');clockButton.type='button';let clockCapturePending=false,clockCaptureGeneration=0;const MAX_RVFC_AGE_MS=250,MAX_RVFC_REQUEST_DELAY_MS=250,MAX_PLAYER_TIME_DELTA_SECONDS=0.1,MAX_RVFC_WAIT_MS=1000;clockButton.addEventListener('click',()=>{const id=anchorId.value.trim();if(!s.tokenOK||!Number.isFinite(video.duration)||video.currentSrc!==objectUrl||video.playbackRate!==1||video.seeking||video.readyState<2||document.visibilityState!=='visible'||typeof video.requestVideoFrameCallback!=='function'||!(/^[A-Za-z0-9_-]{1,40}$/).test(id)||s.clockEvents.some(x=>x.anchor_id===id)||clockCapturePending){alert('照合済み動画を表示し、seek完了・通常速度・可視状態にして、未使用アンカーIDを指定してください。対応ブラウザの新しい提示フレームを取得できない場合は記録されません。');return}const requestedPlayerTime=video.currentTime,requestStarted=performance.now();clockCapturePending=true;refreshClockButton();const requestedSelection=selection,requestedSrc=objectUrl,requestGeneration=++clockCaptureGeneration;const timeout=setTimeout(()=>{if(requestedSelection!==selection||requestGeneration!==clockCaptureGeneration)return;clockCaptureGeneration++;clockCapturePending=false;refreshClockButton();alert('新しい提示フレームを1秒以内に取得できませんでした。アンカー付近を再表示して再試行してください。')},MAX_RVFC_WAIT_MS);video.requestVideoFrameCallback((callbackNow,meta)=>{clearTimeout(timeout);if(requestedSelection!==selection||requestGeneration!==clockCaptureGeneration||!clockCapturePending||requestedSrc!==objectUrl||video.currentSrc!==requestedSrc)return;const capturedAt=performance.now(),age=capturedAt-callbackNow,requestDelay=callbackNow-requestStarted,mediaTime=meta.mediaTime,currentTime=video.currentTime;if(!Number.isFinite(age)||age<0||age>MAX_RVFC_AGE_MS||!Number.isFinite(requestDelay)||requestDelay<0||requestDelay>MAX_RVFC_REQUEST_DELAY_MS||!Number.isFinite(mediaTime)||!Number.isFinite(currentTime)||Math.abs(mediaTime-currentTime)>MAX_PLAYER_TIME_DELTA_SECONDS||Math.abs(mediaTime-requestedPlayerTime)>MAX_PLAYER_TIME_DELTA_SECONDS||video.seeking||video.readyState<2||document.visibilityState!=='visible'){clockCaptureGeneration++;clockCapturePending=false;refreshClockButton();alert('新しい提示フレームがクリック時刻・鮮度・再生状態の条件を満たしません。アンカー付近を再表示して再試行してください。');return}const frame={media_time_seconds:mediaTime,presented_frames:meta.presentedFrames,presentation_time_ms:meta.presentationTime,expected_display_time_ms:meta.expectedDisplayTime,width:meta.width,height:meta.height,callback_now_ms:callbackNow};s.clockEvents.push({anchor_id:id,sequence:s.clockEvents.length+1,requested_player_time_seconds:requestedPlayerTime,request_started_performance_ms:requestStarted,html5_current_time_seconds:currentTime,request_video_frame_callback_metadata:frame,frame_match_status:'fresh_click_bound_callback_unverified_decoder',performance_now_ms:capturedAt,media_duration_seconds:video.duration,playback_rate:video.playbackRate,muted:video.muted,volume:video.volume,text_tracks_disabled:Array.from(video.textTracks||[]).every(t=>t.mode==='disabled'),paused:video.paused,seeking:video.seeking,ready_state:video.readyState,current_src_is_staged_media:video.currentSrc===objectUrl,visibility_state:document.visibilityState,captured_at_utc:new Date().toISOString()});clockCaptureGeneration++;clockCapturePending=false;anchorId.value='';refreshClockButton()})});box.append(el('label',{},'事前定義した時計アンカーID'));box.lastChild.append(anchorId,clockButton);const anchorStatus=el('p',{},'時計観測 0 件（根拠区間とは別記録。アンカー付近で再生し、クリック時の再生時刻に対応する新しい提示フレームを捕捉します）');box.append(anchorStatus);function refreshClockButton(){clockButton.disabled=!s.tokenOK||clockCapturePending;anchorStatus.textContent=`時計観測 ${s.clockEvents.length} 件（根拠区間とは別記録）`}refreshClockButton();
 const capture=el('button',{},'再生時刻を開始へ');capture.type='button';capture.onclick=()=>{start.value=video.currentTime.toFixed(3)};const captureEnd=el('button',{},'再生時刻を終了へ');captureEnd.type='button';captureEnd.onclick=()=>{end.value=video.currentTime.toFixed(3)};
 const add=el('button',{},'区間を追加');add.type='button';add.onclick=()=>{const a=Number(start.value),b=Number(end.value),why=reason.value.trim(),duration=video.duration;const ms=x=>Math.abs(x*1000-Math.round(x*1000))<1e-6;if(!Number.isFinite(a)||!Number.isFinite(b)||!Number.isFinite(duration)||a<0||b<=a||b>duration||!ms(a)||!ms(b)||!why||s.intervals.some(x=>a===x.start&&b===x.end)){alert('動画長内の正の区間（ミリ秒精度）と根拠メモを入力してください。同一区間の重複登録はできません。別の根拠出現は重なる場合も個別に記録してください。');return}s.intervals.push({start:a,end:b,rationale:why});s.intervals.sort((x,y)=>x.start-y.start);s.reviewedAt=null;redraw();checkReady();start.value='';end.value='';reason.value=''};
 box.append(el('label',{},'開始秒（inclusive・0.001秒精度）'));box.lastChild.append(start,capture);box.append(el('label',{},'終了秒（inclusive・0.001秒精度）'));box.lastChild.append(end,captureEnd);box.append(el('label',{},'この区間がどの主張要素を支持するか（非公開メモ）'));box.lastChild.append(reason);box.append(add);rows.append(box);rowResetters.push(()=>{clockCaptureGeneration++;clockCapturePending=false;video.pause();try{video.currentTime=0}catch(_){}s.full=s.context=s.seedHidden=s.outcomeHidden=false;s.decision='';s.intervals=[];s.clockEvents=[];s.firstPlayedAt=s.reviewedAt=null;for(const cb of flagInputs)cb.checked=false;select.value='';start.value=end.value=reason.value=anchorId.value='';if(file){file.value='';s.tokenOK=false;s.duration=null;s.mime='';if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl='';video.removeAttribute('src');video.load();status.textContent='動画未選択'}redraw();refreshClockButton()})
});
for(const id of ['reviewer','role','attest','export-disclosure'])document.getElementById(id).addEventListener('input',checkReady);for(const id of ['role','attest','export-disclosure'])document.getElementById(id).addEventListener('change',checkReady);
function download(text,name,type='application/json'){const blob=new Blob([text],{type}),a=el('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function hex(bytes){return Array.from(bytes,x=>x.toString(16).padStart(2,'0')).join('')}
function b64(bytes){let binary='';for(let i=0;i<bytes.length;i+=0x8000)binary+=String.fromCharCode(...bytes.subarray(i,i+0x8000));return btoa(binary)}
async function member(role,bytes){if(!bytes.length||bytes.length>MAX_ENVELOPE_MEMBER_BYTES)throw Error('member_size_invalid');return {role,size_bytes:bytes.length,sha256:hex(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))),content_base64:b64(bytes)}}
function downloadEnvelope(){if(!pendingEnvelope)return;download(pendingEnvelope,'nextgqa_combined_review_handoff.json')}
function setReviewVisible(visible){document.getElementById('reviewer-controls').hidden=!visible;rows.hidden=!visible;document.getElementById('session-handoff').hidden=visible}
document.getElementById('export').onclick=async()=>{
 const exportButton=document.getElementById('export');if(exportButton.disabled||exportInProgress)return;exportInProgress=true;exportButton.disabled=true;
 try{
  const now=new Date().toISOString(),policySha=base.claim_review_policy_sha256,reviewer=document.getElementById('reviewer').value,reviewerIdentity=reviewerKey(reviewer);
  if(completedSessions.length>=16)throw Error('session_limit');if(!validReviewer(reviewer))throw Error('reviewer_id_invalid');
  if(completedSessions.some(x=>x.reviewerKey===reviewerIdentity))throw Error('duplicate_reviewer');
  const out={schema_version:3,purpose:'anonymous_local_claim_video_human_review',status:'human_review_recorded_unvalidated',review_completed:true,execution_authorization:'not_granted',claim_review_policy_id:base.claim_review_policy_id,claim_review_policy_sha256:policySha,human_reviewer_role:'independent_human',human_reviewer_private_id:reviewer,review_environment:{browser_user_agent:navigator.userAgent,offline_static_page:true,network_fetch_available:false},independence_attestation:'reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk',export_disclosure_acknowledged:document.getElementById('export-disclosure').checked,records:base.records.map((r,i)=>{const s=state[i];return {review_token:r.review_token,review:{full_source_video_reviewed:s.full,question_and_claim_only_context_shown:s.context,gold_and_seed_hidden_until_mapping_frozen:s.seedHidden,outcome_scores_not_accessed:s.outcomeHidden,media_bytes_verified:s.tokenOK,occurrence_policy_applied:base.occurrence_policy,support_decision:s.decision,evidence_intervals_seconds:s.intervals.map(x=>[x.start,x.end]),interval_rationales:s.intervals.map(x=>x.rationale),media_duration_seconds:s.duration,reviewer_role:'independent_human',first_played_at_utc:s.firstPlayedAt,reviewed_at_utc:s.reviewedAt,review_status:'completed',display_player:'HTML5 video; audio forced muted; text/caption tracks disabled; playback rate 1.0; UA='+navigator.userAgent,claim_review_policy_sha256:policySha}}})};
  const reviewRaw=new TextEncoder().encode(JSON.stringify(out,null,2)+'\n'),reviewDigest=hex(new Uint8Array(await crypto.subtle.digest('SHA-256',reviewRaw))),index={schema_version:1,purpose:'anonymous_unvalidated_human_review_handoff',review_payload_sha256:reviewDigest,claim_review_policy_sha256:policySha,created_at_utc:now,review_status:'unvalidated_requires_custodian_audit',execution_authorization:'not_granted'},indexRaw=new TextEncoder().encode(JSON.stringify(index,null,2)+'\n'),members=[await member('review_payload',reviewRaw),await member('review_index',indexRaw)];
  if(state.some(x=>x.clockEvents.length)){
   const clock={schema_version:3,purpose:'local_unvalidated_html5_player_clock_observations',status:'raw_player_clock_observations_only',capture_policy_id:'rvfc_click_bound_v1',capture_thresholds:{max_callback_age_ms:250,max_request_delay_ms:250,max_click_player_time_delta_seconds:0.1,max_callback_player_time_delta_seconds:0.1,callback_wait_timeout_ms:1000},player_clock:'html5_video_currentTime_seconds',presented_frame_metadata:'requestVideoFrameCallback_when_available_not_decoder_matched',decoder_frame_identity:'not_verified',created_at_utc:now,browser_user_agent:navigator.userAgent,anchor_id_validation:'format_and_per_video_uniqueness_only_unverified',predeclared_plan_binding:'not_checked',fit_heldout_split:'not_assigned',decoder_pts_pairing:'not_provided',execution_authorization:'not_granted',records:base.records.map((r,i)=>({review_token:r.review_token,events:state[i].clockEvents}))},clockRaw=new TextEncoder().encode(JSON.stringify(clock,null,2)+'\n'),clockDigest=hex(new Uint8Array(await crypto.subtle.digest('SHA-256',clockRaw))),clockIndex={schema_version:1,purpose:'unvalidated_html5_clock_observation_integrity_index',payload_sha256:clockDigest,created_at_utc:now,status:'raw_observations_only',execution_authorization:'not_granted'};members.push(await member('clock_observations',clockRaw),await member('clock_index',new TextEncoder().encode(JSON.stringify(clockIndex,null,2)+'\n')))
  }
  const session={sequence:completedSessions.length+1,members},sessions=[...completedSessions.map(x=>x.session),session],envelope={schema_version:1,purpose:'combined_nextgqa_formal_review_handoff_envelope',status:'unvalidated_combined_review_handoff',execution_authorization:'not_granted',sessions},candidateEnvelope=JSON.stringify(envelope,null,2)+'\n';if(new TextEncoder().encode(candidateEnvelope).length>MAX_ENVELOPE_BYTES)throw Error('envelope_too_large');
  completedSessions.push({reviewerKey:reviewerIdentity,session});pendingEnvelope=candidateEnvelope;document.getElementById('reviewer-count').textContent=`確定済み: ${completedSessions.length} 人`;document.getElementById('handoff-status').textContent='この担当者の記録をメモリ上に確定しました。まだダウンロードされていません。次の担当者へ進むか、最後に全員分を1つのJSONで保存してください。';setReviewVisible(false);document.getElementById('retry-envelope').disabled=false;document.getElementById('snapshot-notice').hidden=true;document.getElementById('next-reviewer').disabled=completedSessions.length>=16;exportInProgress=false
 }catch(_error){exportInProgress=false;checkReady();document.getElementById('global').textContent='出力を作成できませんでした。入力を保持しています。容量または識別子を確認して再試行してください。'}
};
document.getElementById('next-reviewer').onclick=()=>{if(!completedSessions.length||completedSessions.length>=16)return;document.getElementById('snapshot-notice').hidden=false;for(const reset of rowResetters)reset();document.getElementById('reviewer').value='';document.getElementById('role').value='';document.getElementById('attest').checked=false;document.getElementById('export-disclosure').checked=false;for(const id of ['reviewer','role','attest','export-disclosure'])document.getElementById(id).disabled=false;setReviewVisible(true);checkReady()};
document.getElementById('download-envelope').onclick=downloadEnvelope;document.getElementById('retry-envelope').onclick=downloadEnvelope;document.getElementById('snapshot-download').onclick=downloadEnvelope;
checkReady();
</script></html>'''
    provenance = ""
    if bundle_hashes is not None:
        provenance = (
            f'<meta name="claim-input-manifest-sha256" content="{bundle_hashes[0]}">\n'
            f'<meta name="claim-input-bundle-sidecar-sha256" content="{bundle_hashes[1]}">'
        )
    html_doc = (template.replace("POLICY", payload["claim_review_policy_sha256"])
                .replace("PROVENANCE", provenance)
                .replace("DATA", data).replace("MEDIA_NOTICE", "各行でcustodian動画ファイルを選びます。" if media_files is None else "動画をこのHTMLに埋め込んでいます。ファイルには保護対象素材が含まれるため、非公開で保管してください。" )).encode("utf-8")
    return html_doc


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True,
                        help="independently recorded digest of the audited lineage manifest")
    parser.add_argument("--custody-manifest", required=True)
    parser.add_argument("--expected-custody-sha256", required=True,
                        help="independently recorded digest of the private custody manifest")
    parser.add_argument("--claim-input-manifest",
                        help="optional candidate v2 claim manifest; must be paired with its sidecar")
    parser.add_argument("--claim-input-sidecar",
                        help="optional candidate v2 claim-input provenance sidecar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embed-videos-from", help="private mode-0700 token directory under data/private_nextgqa")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    source = Path(args.input).resolve()
    lineage_source = Path(args.lineage_manifest).resolve()
    custody_source = Path(args.custody_manifest).resolve()
    claim_manifest_source = Path(args.claim_input_manifest).resolve() if args.claim_input_manifest else None
    claim_sidecar_source = Path(args.claim_input_sidecar).resolve() if args.claim_input_sidecar else None
    if bool(claim_manifest_source) != bool(claim_sidecar_source):
        parser.error("claim_bundle_pair_required")
    repo_root = Path(__file__).resolve().parents[1]
    try:
        target = validate_private_output(args.output, repo_root)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "private_output_path_invalid")
    private_inputs = [source, lineage_source, custody_source]
    if claim_manifest_source is not None:
        private_inputs.extend([claim_manifest_source, claim_sidecar_source])
    if any((path.stat().st_mode & 0o777) != 0o600 for path in private_inputs):
        parser.error("private_input_mode_invalid")
    try:
        raw = source.read_bytes(); payload = json.loads(raw.decode("utf-8"))
        lineage_raw = lineage_source.read_bytes()
        lineage = json.loads(lineage_raw.decode("utf-8"))
        custody_raw = custody_source.read_bytes()
        custody = json.loads(custody_raw.decode("utf-8"))
        claim_manifest = claim_manifest_raw = claim_sidecar = claim_sidecar_raw = None
        if claim_manifest_source is not None:
            claim_manifest_raw = claim_manifest_source.read_bytes()
            claim_manifest = json.loads(claim_manifest_raw.decode("utf-8"))
            claim_sidecar_raw = claim_sidecar_source.read_bytes()
            claim_sidecar = json.loads(claim_sidecar_raw.decode("utf-8"))
        media_files = load_private_media_directory(Path(args.embed_videos_from), payload, custody) if args.embed_videos_from else None
        result = render(payload, raw, lineage, lineage_raw, args.expected_lineage_sha256,
                        custody, custody_raw, args.expected_custody_sha256,
                        claim_manifest, claim_manifest_raw, claim_sidecar, claim_sidecar_raw,
                        media_files)
        write_exclusive(target, result)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except OSError:
        parser.error("input_or_output_io_error")
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "worksheet_invalid")
    print("Private offline reviewer page created; protected content remains local; scoring authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
