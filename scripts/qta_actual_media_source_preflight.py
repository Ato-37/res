#!/usr/bin/env python3
"""Join a local three-video sampler receipt to actual QA source by video SHA.

CPU-only, local-only and non-authorizing. This checks byte-identity joins, not
sampler provenance, human review, semantic support, or permission to score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping

from build_nextgqa_pre_review_byte_inventory import CODE_FILES
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from run_qta_actual_qa_source_binding_smoke import (
    _build_claim_and_selection, _target_manifest,
)
from run_qta_real_video_sampler_preflight import (
    PRIVATE_ROOT, load_candidate_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROW_COUNT = 3
MAX_SAMPLER_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 24 * 1024 * 1024
DENIAL = {
    "model_load_permitted": False,
    "execution_authorized": False,
    "scoring_authorized": False,
    "faithfulness_scored": False,
    "baseline_forwards": 0,
    "branch_forwards": 0,
    "score_calls": 0,
}

SOURCE_KEYS = {
    "csv_raw", "claim_input_manifest_raw", "worksheet_raw",
    "target_candidate_raw", "lineage_manifest_raw", "qta_protocol_raw",
    "qta_config_raw",
}
SOURCE_INVENTORY_ROLES = {
    "csv_raw": "source_test_csv",
    "claim_input_manifest_raw": "claim_input_manifest",
    "worksheet_raw": "claim_mapping_worksheet",
    "target_candidate_raw": "diagnostic_target_candidate",
    "lineage_manifest_raw": "claim_lineage",
    "qta_protocol_raw": "qta_protocol_draft",
    "qta_config_raw": "qta_treatment_selection",
}
REPO_SOURCE_PATHS = {
    "csv_raw": "NExT-GQA/datasets/nextgqa/test.csv",
    "qta_protocol_raw": "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
    "qta_config_raw": "experiments/2026-10-18_qta_treatment_selection.md",
}
DATASET_CSV_MODES = {0o400, 0o444, 0o600, 0o644, 0o755}
PINNED_SOURCE_MODES = {0o400, 0o444, 0o600, 0o644, 0o555, 0o755}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, code: str) -> Any:
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ValueError(code) from None


def _code_closure(inventory_raw: bytes, *, repo_root: Path,
                  code_files: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    inventory = _strict_json(inventory_raw, "candidate_inventory_invalid")
    if (not isinstance(inventory, dict)
            or inventory.get("purpose") != "pre_review_candidate_byte_inventory"
            or inventory.get("status") != "candidate_only"
            or inventory.get("execution_authorization") != "not_granted"
            or inventory.get("joint_freeze") is not False
            or inventory.get("score_ready") is not False
            or inventory.get("missing_required_slots") != []):
        raise ValueError("candidate_inventory_not_current_candidate")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate_inventory_artifacts_invalid")
    by_role = {}
    for item in artifacts:
        if (not isinstance(item, dict) or not isinstance(item.get("role"), str)
                or item["role"] in by_role):
            raise ValueError("candidate_inventory_artifacts_invalid")
        by_role[item["role"]] = item
    for role, relpath in code_files.items():
        item = by_role.get(role)
        if not isinstance(item, dict):
            raise ValueError("candidate_inventory_code_role_missing")
        path_parts = Path(relpath).parts
        if (Path(relpath).is_absolute()
                or any(part in {"", ".", ".."} for part in path_parts)):
            raise ValueError("candidate_inventory_code_path_unsafe")
        root_fd = os.open(repo_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                          | getattr(os, "O_NOFOLLOW", 0))
        current_fd = root_fd
        file_fd = None
        try:
            for part in path_parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                  | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
                info = os.fstat(next_fd)
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                    os.close(next_fd)
                    raise ValueError("candidate_inventory_code_directory_invalid")
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = next_fd
            file_fd = os.open(path_parts[-1], os.O_RDONLY
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
        except OSError:
            raise ValueError("candidate_inventory_code_path_unavailable_or_symlink") from None
        finally:
            if current_fd != root_fd:
                os.close(current_fd)
            os.close(root_fd)
        try:
            before = os.fstat(file_fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) not in {0o644, 0o755}
                    or before.st_size < 1 or before.st_size > 4 * 1024 * 1024):
                raise ValueError("candidate_inventory_code_file_invalid")
            chunks, size = [], 0
            while True:
                part = os.read(file_fd, min(1024 * 1024, 4 * 1024 * 1024 + 1 - size))
                if not part:
                    break
                chunks.append(part)
                size += len(part)
                if size > 4 * 1024 * 1024:
                    raise ValueError("candidate_inventory_code_file_too_large")
            after = os.fstat(file_fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                     after.st_ctime_ns) or size != before.st_size):
                raise ValueError("candidate_inventory_code_changed_during_read")
            raw = b"".join(chunks)
        finally:
            os.close(file_fd)
        if (item.get("sha256") != _sha(raw)
                or item.get("size_bytes") != len(raw)):
            raise ValueError("candidate_inventory_code_closure_stale")
    return by_role


def _sampler_rows(raw: bytes, *, expected_inventory_sha256: str,
                  expected_media_sha256: set[str]) -> dict[str, dict[str, Any]]:
    report = _strict_json(raw, "sampler_receipt_invalid")
    if (not isinstance(report, dict)
            or report.get("purpose") != "local_qta_real_video_sampler_eligibility_preflight"
            or report.get("status") != "passed"
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("requested_records") != ROW_COUNT
            or report.get("effective_records") != ROW_COUNT
            or report.get("sampler") != "endpoint_inclusive_uniform_decoded_frame_index_v1"
            or report.get("frames_per_video") != 32
            or report.get("candidate_inventory_sha256") != expected_inventory_sha256
            or report.get("gpu_used") is not False
            or report.get("model_loaded") is not False
            or report.get("faithfulness_scored") is not False
            or report.get("execution_authorization") != "not_granted"):
        raise ValueError("sampler_receipt_schema_or_inventory_binding_invalid")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ValueError("sampler_receipt_row_count_invalid")
    by_sha = {}
    indexes = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sampler_receipt_row_invalid")
        index, digest = row.get("record_index"), row.get("source_sha256")
        if (type(index) is not int or index not in {1, 2, 3} or index in indexes
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)
                or digest in by_sha
                or row.get("sampled_frame_count") != 32
                or not isinstance(row.get("sampled_pts"), list)
                or len(row["sampled_pts"]) != 32
                or not isinstance(row.get("sampled_rgb_sha256"), list)
                or len(row["sampled_rgb_sha256"]) != 32
                or any(not isinstance(x, str) or not SHA256.fullmatch(x)
                       for x in row["sampled_rgb_sha256"])):
            raise ValueError("sampler_receipt_row_invalid")
        indexes.add(index)
        by_sha[digest] = row
    if set(by_sha) != expected_media_sha256:
        raise ValueError("sampler_receipt_media_inventory_mismatch")
    return by_sha


def preflight_actual_media_source_join(*, sampler_receipt_raw: bytes,
                                       candidate_inventory_raw: bytes,
                                       media_sha256_by_role: Mapping[str, str],
                                       actual_source_inputs: Mapping[str, Any],
                                       repo_root: Path = REPO_ROOT,
                                       code_files: Mapping[str, str] = CODE_FILES) -> dict[str, Any]:
    """Revalidate actual QA bytes, then join sampler hashes to lineage hashes.

    The inventory's source-code closure is checked against this checkout. All
    input provenance remains caller-pinned/unverified; success never authorizes
    baseline or treatment forwards.
    """
    result = {
        "schema_version": 1,
        "purpose": "qta_actual_media_source_hash_preflight_non_authorizing",
        "status": "blocked",
        "requested_rows": ROW_COUNT,
        "sampler_rows": 0,
        "actual_qa_rows": 0,
        "joined_rows": 0,
        "failure_code": "preflight_blocked",
        "provenance": "local_bytes_structurally_checked_but_not_authenticated",
        "sampler_provenance_authenticated": False,
        "actual_source_provenance_authenticated": False,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "later_gates": ["stage1_release_custody", "human_claim_review",
                        "clock_frame_correspondence", "clock_calibration",
                        "processor_batch_binding", "prescore_mapping",
                        "baseline_full_video_pq_freeze",
                        "stage2_prediction_link_review"],
        **DENIAL,
    }
    try:
        if (not isinstance(candidate_inventory_raw, bytes)
                or len(candidate_inventory_raw) > MAX_SAMPLER_BYTES
                or not isinstance(sampler_receipt_raw, bytes)
                or len(sampler_receipt_raw) > MAX_SAMPLER_BYTES):
            raise ValueError("preflight_input_bytes_invalid")
        inventory_roles = _code_closure(candidate_inventory_raw, repo_root=repo_root,
                                        code_files=code_files)
        inventory_sha = _sha(candidate_inventory_raw)
        media_roles = {"staged_media_01", "staged_media_02", "staged_media_03"}
        if (not isinstance(media_sha256_by_role, Mapping)
                or set(media_sha256_by_role) != media_roles
                or any(not isinstance(value, str) or not SHA256.fullmatch(value)
                       for value in media_sha256_by_role.values())
                or len(set(media_sha256_by_role.values())) != ROW_COUNT):
            raise ValueError("candidate_inventory_media_hashes_invalid")
        for role, digest in media_sha256_by_role.items():
            item = inventory_roles.get(role)
            if (not isinstance(item, dict) or item.get("sha256") != digest
                    or type(item.get("size_bytes")) is not int
                    or item["size_bytes"] < 1):
                raise ValueError("candidate_inventory_media_role_mismatch")
        sampler_by_sha = _sampler_rows(
            sampler_receipt_raw, expected_inventory_sha256=inventory_sha,
            expected_media_sha256=set(media_sha256_by_role.values()))
        result["sampler_rows"] = ROW_COUNT
        if (not isinstance(actual_source_inputs, Mapping)
                or set(actual_source_inputs) != SOURCE_KEYS
                or any(not isinstance(actual_source_inputs[key], bytes)
                       for key in SOURCE_KEYS)
                or any(len(value) > MAX_SOURCE_BYTES
                       for value in actual_source_inputs.values())):
            raise ValueError("actual_source_input_schema_invalid")
        for input_key, role in SOURCE_INVENTORY_ROLES.items():
            artifact = inventory_roles.get(role)
            raw = actual_source_inputs[input_key]
            if (not isinstance(artifact, dict) or artifact.get("sha256") != _sha(raw)
                    or type(artifact.get("size_bytes")) is not int
                    or artifact.get("size_bytes") != len(raw)):
                raise ValueError(f"candidate_inventory_source_role_mismatch_{role}")
        lineage_input = _strict_json(actual_source_inputs["lineage_manifest_raw"],
                                     "actual_qa_lineage_invalid")
        lineage_records = lineage_input.get("records") if isinstance(lineage_input, dict) else None
        if not isinstance(lineage_records, list) or len(lineage_records) != ROW_COUNT:
            raise ValueError("actual_qa_lineage_row_count_invalid")
        lineage_ids = [row.get("video_id") if isinstance(row, dict) else None
                       for row in lineage_records]
        if (any(not isinstance(value, str) or not value for value in lineage_ids)
                or len(set(lineage_ids)) != ROW_COUNT):
            raise ValueError("actual_qa_lineage_video_identity_invalid")
        claim_manifest_raw, selection_raw, _worksheet, _lineage = _build_claim_and_selection(
            claim_raw=actual_source_inputs["claim_input_manifest_raw"],
            worksheet_raw=actual_source_inputs["worksheet_raw"],
            lineage_raw=actual_source_inputs["lineage_manifest_raw"])
        target_manifest_raw = _target_manifest(actual_source_inputs["target_candidate_raw"])
        protocol_pin = inventory_roles["qta_protocol_draft"]["sha256"]
        config_pin = inventory_roles["qta_treatment_selection"]["sha256"]
        bound = bind_qta_actual_qa_source(
            csv_raw=actual_source_inputs["csv_raw"],
            claim_manifest_raw=claim_manifest_raw,
            selection_raw=selection_raw,
            target_manifest_raw=target_manifest_raw,
            lineage_manifest_raw=actual_source_inputs["lineage_manifest_raw"],
            qta_protocol_raw=actual_source_inputs["qta_protocol_raw"],
            expected_qta_protocol_sha256=protocol_pin,
            qta_config_raw=actual_source_inputs["qta_config_raw"],
            expected_qta_config_sha256=config_pin)
        summary = bound.get("summary") if isinstance(bound, dict) else None
        private_bundle = bound.get("private_bundle") if isinstance(bound, dict) else None
        if (not isinstance(summary, dict) or summary.get("status") != "bound_unvalidated"
                or summary.get("requested_rows") != ROW_COUNT
                or summary.get("validated_rows") != ROW_COUNT
                or summary.get("model_load_permitted") is not False
                or summary.get("branch_forwards") != 0
                or not isinstance(private_bundle, dict)
                or private_bundle.get("classification")
                != "private_local_only_unvalidated_qa_source"):
            raise ValueError("actual_qa_source_binding_blocked")
        private = _strict_json(private_bundle.get("raw_json", b""),
                               "actual_qa_private_bundle_invalid")
        crosswalk = private.get("review_source_crosswalk")
        lineage = _strict_json(actual_source_inputs["lineage_manifest_raw"],
                               "actual_qa_lineage_invalid")
        lineage_rows = lineage.get("records") if isinstance(lineage, dict) else None
        if (not isinstance(crosswalk, list) or len(crosswalk) != ROW_COUNT
                or not isinstance(lineage_rows, list) or len(lineage_rows) != ROW_COUNT):
            raise ValueError("actual_qa_crosswalk_row_count_invalid")
        crosswalk_by_sha = {}
        lineage_by_video = {}
        for line in lineage_rows:
            video_id = line.get("video_id") if isinstance(line, dict) else None
            if not isinstance(video_id, str) or not video_id or video_id in lineage_by_video:
                raise ValueError("actual_qa_lineage_video_identity_invalid")
            lineage_by_video[video_id] = line
        seen_tokens = set()
        seen_video_ids = set()
        for row in crosswalk:
            if not isinstance(row, dict):
                raise ValueError("actual_qa_crosswalk_row_invalid")
            video_id, token = row.get("video_id"), row.get("review_token")
            if (not isinstance(video_id, str) or not video_id
                    or not isinstance(token, str) or not token
                    or video_id in seen_video_ids or token in seen_tokens):
                raise ValueError("actual_qa_crosswalk_identity_invalid")
            line = lineage_by_video.get(video_id)
            digest = line.get("video_sha256") if line else None
            if (not isinstance(digest, str) or not SHA256.fullmatch(digest)
                    or digest in crosswalk_by_sha):
                raise ValueError("actual_qa_lineage_video_digest_invalid")
            seen_video_ids.add(video_id)
            seen_tokens.add(token)
            crosswalk_by_sha[digest] = (video_id, token)
        if set(crosswalk_by_sha) != set(sampler_by_sha):
            raise ValueError("sampler_to_actual_source_video_hash_mismatch")
        # Access by digest, never by sampler row order or record_index.
        for digest in sampler_by_sha:
            if digest not in crosswalk_by_sha:
                raise ValueError("sampler_to_actual_source_video_hash_mismatch")
        result.update({"status": "source_video_hash_joined_unvalidated",
                       "actual_qa_rows": ROW_COUNT, "joined_rows": ROW_COUNT,
                       "failure_code": None})
    except Exception as exc:
        code = str(exc)
        result["failure_code"] = (code if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", code)
                                  else "preflight_blocked")
    result.update(DENIAL)
    if result["status"] != "source_video_hash_joined_unvalidated":
        result["sampler_rows"] = 0
        result["actual_qa_rows"] = 0
        result["joined_rows"] = 0
    return result


def _private_root() -> Path:
    if PRIVATE_ROOT.is_symlink():
        raise ValueError("private_root_symlink_rejected")
    root = PRIVATE_ROOT.resolve(strict=True)
    info = root.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_root_not_private")
    return root


def _read_private_file(root: Path, name: str, limit: int, *,
                       allow_inventory_path: bool = False) -> bytes:
    if not isinstance(name, str):
        raise ValueError("input_filename_invalid")
    parts = Path(name).parts
    if (Path(name).is_absolute() or any(part in {"", ".", ".."} for part in parts)
            or (len(parts) != 1 and not allow_inventory_path)
            or (allow_inventory_path and not re.fullmatch(
                r"inventory-[A-Za-z0-9._-]{1,96}/pre_review_candidate_byte_inventory\.json",
                name))):
        raise ValueError("input_filename_invalid")
    fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_fd = fd
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            info = os.fstat(next_fd)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                os.close(next_fd)
                raise ValueError("inventory_directory_owner_mode_or_type_invalid")
            if parent_fd != fd:
                os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                          dir_fd=parent_fd)
    except OSError:
        raise ValueError("private_input_path_unavailable_or_symlink") from None
    finally:
        if 'parent_fd' in locals() and parent_fd != fd:
            os.close(parent_fd)
        os.close(fd)
    try:
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > limit):
            raise ValueError("input_file_permissions_or_size_invalid")
        chunks, size = [], 0
        while True:
            part = os.read(file_fd, min(1024 * 1024, limit + 1 - size))
            if not part:
                break
            chunks.append(part)
            size += len(part)
            if size > limit:
                raise ValueError("input_file_too_large")
        after = os.fstat(file_fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns) or size != before.st_size):
            raise ValueError("input_file_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _read_repo_source(root: Path, relative: str, *, role: str,
                      limit: int = MAX_SOURCE_BYTES) -> bytes:
    """Read a fixed repository source path without following any symlink."""
    expected = REPO_SOURCE_PATHS.get(role)
    if relative != expected:
        raise ValueError(f"{role}_path_not_allowlisted")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                      | getattr(os, "O_NOFOLLOW", 0))
    current_fd = root_fd
    try:
        parts = Path(expected).parts
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                os.close(next_fd)
                raise ValueError(f"{role}_directory_owner_or_type_invalid")
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                          dir_fd=current_fd)
    except OSError:
        raise ValueError(f"{role}_path_unavailable_or_symlink") from None
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    try:
        before = os.fstat(file_fd)
        allowed_modes = DATASET_CSV_MODES if role == "csv_raw" else PINNED_SOURCE_MODES
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) not in allowed_modes
                or before.st_size < 1 or before.st_size > limit):
            raise ValueError(f"{role}_owner_mode_type_or_size_invalid")
        chunks, size = [], 0
        while True:
            part = os.read(file_fd, min(1024 * 1024, limit + 1 - size))
            if not part:
                break
            chunks.append(part)
            size += len(part)
            if size > limit:
                raise ValueError(f"{role}_file_too_large")
        after = os.fstat(file_fd)
        before_fields = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
        after_fields = (after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns)
        if before_fields != after_fields or size != before.st_size:
            raise ValueError(f"{role}_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", required=True,
                        help="inventory-NAME/pre_review_candidate_byte_inventory.json "
                             "under data/private_nextgqa (directory/file modes 0700/0600)")
    for name in ("sampler-receipt", "claim-input-manifest", "worksheet",
                 "target-candidate", "lineage-manifest"):
        parser.add_argument(f"--{name}", required=True,
                            help="0600 file basename directly under data/private_nextgqa")
    parser.epilog = ("CSV, QTA protocol, and treatment config are read from fixed, "
                     "symlink-rejected repository paths: "
                     + ", ".join(REPO_SOURCE_PATHS.values()))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = _private_root()
        inventory_raw = _read_private_file(root, args.candidate_inventory,
                                           MAX_SAMPLER_BYTES,
                                           allow_inventory_path=True)
        # Reuse the sampler's secure candidate-inventory validator. It returns
        # only the frozen staged-media SHA map and exact inventory digest.
        media_sha, inventory_sha = load_candidate_inventory(root / args.candidate_inventory)
        if _sha(inventory_raw) != inventory_sha:
            raise ValueError("candidate_inventory_changed_during_read")
        sampler_raw = _read_private_file(root, args.sampler_receipt, MAX_SAMPLER_BYTES)
        source_inputs = {
            "csv_raw": _read_repo_source(REPO_ROOT, REPO_SOURCE_PATHS["csv_raw"],
                                          role="csv_raw"),
            "claim_input_manifest_raw": _read_private_file(
                root, args.claim_input_manifest, MAX_SOURCE_BYTES),
            "worksheet_raw": _read_private_file(root, args.worksheet, MAX_SOURCE_BYTES),
            "target_candidate_raw": _read_private_file(
                root, args.target_candidate, MAX_SOURCE_BYTES),
            "lineage_manifest_raw": _read_private_file(root, args.lineage_manifest, MAX_SOURCE_BYTES),
            "qta_protocol_raw": _read_repo_source(
                REPO_ROOT, REPO_SOURCE_PATHS["qta_protocol_raw"], role="qta_protocol_raw"),
            "qta_config_raw": _read_repo_source(
                REPO_ROOT, REPO_SOURCE_PATHS["qta_config_raw"], role="qta_config_raw"),
        }
        report = preflight_actual_media_source_join(
            sampler_receipt_raw=sampler_raw, candidate_inventory_raw=inventory_raw,
            media_sha256_by_role=media_sha, actual_source_inputs=source_inputs)
    except Exception as exc:
        error = str(exc)
        code = error if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", error) else "preflight_blocked"
        report = {
            "schema_version": 1,
            "purpose": "qta_actual_media_source_hash_preflight_non_authorizing",
            "status": "blocked", "requested_rows": ROW_COUNT,
            "sampler_rows": 0, "actual_qa_rows": 0, "joined_rows": 0,
            "failure_code": code,
            "provenance": "local_bytes_structurally_checked_but_not_authenticated",
            "sampler_provenance_authenticated": False,
            "actual_source_provenance_authenticated": False,
            "human_identity_verified": False, "semantic_review_authenticated": False,
            "later_gates": ["stage1_release_custody", "human_claim_review",
                            "clock_frame_correspondence", "clock_calibration",
                            "processor_batch_binding", "prescore_mapping",
                            "baseline_full_video_pq_freeze", "stage2_prediction_link_review"],
            **DENIAL,
        }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "source_video_hash_joined_unvalidated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
