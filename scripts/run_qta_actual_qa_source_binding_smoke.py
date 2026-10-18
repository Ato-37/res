#!/usr/bin/env python3
"""Bind one fixed three-row QTA cohort to local QA source bytes, aggregate-only.

The CLI prints only an allow-listed summary. It never writes or prints the
private bundle returned by the underlying binder and never loads a model or
scores. Protocol/config/code bytes come from exact Git blobs at current HEAD.
An explicit caller-pinned candidate inventory binds this fixed cohort's source
and code bytes; the local inventory pin is unverified and is not a trust root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any

from qta_actual_qa_source_contract import bind_qta_actual_qa_source

MAX_FILE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_INVENTORY_BYTES = 8 * 1024 * 1024
ROW_COUNT = 3
COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_JSON_MODES = {0o600}
# The NExT-GQA checkout may preserve archive-created executable bits on the
# public benchmark CSV. It is read as inert bytes only; do not execute it.
DATASET_CSV_MODES = {0o400, 0o444, 0o600, 0o644, 0o755}
PINNED_SOURCE_MODES = {0o400, 0o444, 0o600, 0o644, 0o555, 0o755}
INPUT_PATHS = {
    "csv": "NExT-GQA/datasets/nextgqa/test.csv",
    "claim_manifest": "data/private_nextgqa/claim_input_manifest_v6_20261017_0635.json",
    "worksheet": "data/private_nextgqa/claim_mapping_sentence_v6_20261017_0635.json",
    "lineage": "data/private_nextgqa/claim_lineage_sentence_exploratory_20261016_01.json",
    "target_candidate": "data/private_nextgqa/diagnostic_target_v6_current_20261017_01.json",
    "protocol": "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
    "config": "experiments/2026-10-18_qta_treatment_selection.md",
}
PINNED_CODE_PATHS = (
    "scripts/run_qta_actual_qa_source_binding_smoke.py",
    "scripts/qta_actual_qa_source_contract.py",
    "scripts/nextgqa_actual_qa_source_adapter.py",
)
PRIVATE_DIRECTORY = "data/private_nextgqa"
INVENTORY_RE = re.compile(
    r"^data/private_nextgqa/inventory-[A-Za-z0-9._-]{1,64}/"
    r"pre_review_candidate_byte_inventory\.json$"
)
INVENTORY_BINDINGS = {
    "source_test_csv": "csv",
    "claim_input_manifest": "claim_manifest",
    "claim_mapping_worksheet": "worksheet",
    "claim_lineage": "lineage",
    "diagnostic_target_candidate": "target_candidate",
    "qta_protocol_draft": "protocol",
    "qta_treatment_selection": "config",
    "qta_actual_qa_source_binding_cli": PINNED_CODE_PATHS[0],
    "qta_actual_qa_source_contract": PINNED_CODE_PATHS[1],
    "actual_qa_source_adapter": PINNED_CODE_PATHS[2],
}
TARGET_SOURCE_FIELDS = (
    "video_id", "claim_id", "claim_text_sha256", "dataset_qid_sha256",
    "dataset_source_row_sha256", "question_sha256", "ordered_choice_sha256",
    "gold_index",
)
CLAIM_MANIFEST_KEYS = {
    "schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
    "claim_policy_sha256", "occurrence_policy", "claims",
}
CLAIM_ROW_KEYS = {
    "video_id", "claim_id", "claim_text", "claim_text_sha256",
    "model_revision", "generation_run_sha256",
}
LINEAGE_ROW_KEYS = {
    "video_id", "claim_id", "dataset_question_id", "dataset_video_id",
    "video_key", "relative_video_path", "video_sha256", "question_sha256",
    "claim_text_sha256",
}


class BindingInputError(ValueError):
    """Stable content-free CLI failure code."""


def _fail(code: str) -> None:
    raise BindingInputError(code)


def _strict_json(raw: bytes, code: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: _fail("nonfinite_json"))
    except BindingInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _open_dir_chain(root_fd: int, relative: str, *, private: bool = False) -> int:
    current = os.dup(root_fd)
    walked = []
    try:
        for part in PurePosixPath(relative).parts:
            if part in {"", ".", ".."}:
                _fail("input_path_invalid")
            try:
                next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                  | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
            except OSError:
                _fail("input_directory_unavailable_or_symlink")
            info = os.fstat(next_fd)
            walked.append(part)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                os.close(next_fd)
                _fail("input_directory_owner_or_type_invalid")
            if (private and "/".join(walked) == PRIVATE_DIRECTORY
                    and stat.S_IMODE(info.st_mode) != 0o700):
                os.close(next_fd)
                _fail("private_data_directory_mode_invalid")
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _read_owned_file(root: Path, raw_path: str, *, role: str,
                     allowed_modes: set[int], max_bytes: int = MAX_FILE_BYTES) -> tuple[bytes, str]:
    expected_path = INPUT_PATHS.get(role)
    if not expected_path or raw_path != expected_path:
        _fail(f"{role}_path_not_allowlisted")
    relative = PurePosixPath(expected_path)
    private = role in {"claim_manifest", "worksheet", "lineage", "target_candidate"}
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                       | getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_fd = _open_dir_chain(root_fd, relative.parent.as_posix(), private=private)
    finally:
        os.close(root_fd)
    try:
        fd = os.open(relative.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    except OSError:
        os.close(parent_fd)
        _fail(f"{role}_open_failed")
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) not in allowed_modes
                or before.st_size < 1 or before.st_size > max_bytes):
            _fail(f"{role}_owner_mode_type_or_size_invalid")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"{role}_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _fail(f"{role}_file_grew_during_read")
        after = os.fstat(fd)
        fields_before = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
        fields_after = (after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns)
        if fields_before != fields_after:
            _fail(f"{role}_changed_during_read")
        return b"".join(chunks), expected_path
    finally:
        os.close(fd)
        os.close(parent_fd)


def _read_candidate_inventory(root: Path, raw_path: str) -> bytes:
    if not isinstance(raw_path, str) or not INVENTORY_RE.fullmatch(raw_path):
        _fail("inventory_path_not_allowlisted_pattern")
    relative = PurePosixPath(raw_path)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                      | getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_fd = _open_dir_chain(root_fd, relative.parent.as_posix(), private=True)
    finally:
        os.close(root_fd)
    try:
        parent = os.fstat(parent_fd)
        if stat.S_IMODE(parent.st_mode) != 0o700:
            _fail("inventory_directory_mode_invalid")
        try:
            fd = os.open(relative.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                         dir_fd=parent_fd)
        except OSError:
            _fail("inventory_open_failed")
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_size < 1 or before.st_size > MAX_INVENTORY_BYTES):
                _fail("inventory_owner_mode_type_or_size_invalid")
            chunks, remaining = [], before.st_size
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    _fail("inventory_short_read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                _fail("inventory_file_grew_during_read")
            after = os.fstat(fd)
            before_fields = (before.st_dev, before.st_ino, before.st_size,
                             before.st_mtime_ns, before.st_ctime_ns)
            after_fields = (after.st_dev, after.st_ino, after.st_size,
                            after.st_mtime_ns, after.st_ctime_ns)
            if before_fields != after_fields:
                _fail("inventory_changed_during_read")
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _validate_inventory(raw: bytes, expected_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        _fail("expected_inventory_sha256_invalid")
    if _sha(raw) != expected_sha256:
        _fail("inventory_expected_sha256_mismatch")
    inventory = _strict_json(raw, "inventory_json_invalid")
    expected_keys = {
        "schema_version", "purpose", "status", "review_started", "review_completed",
        "joint_freeze", "score_ready", "execution_authorization", "rights_status",
        "artifacts", "missing_required_slots",
    }
    if (not isinstance(inventory, dict) or set(inventory) != expected_keys
            or type(inventory.get("schema_version")) is not int
            or inventory["schema_version"] != 1
            or inventory.get("purpose") != "pre_review_candidate_byte_inventory"
            or inventory.get("status") != "candidate_only"
            or inventory.get("review_started") is not False
            or inventory.get("review_completed") is not False
            or inventory.get("joint_freeze") is not False
            or inventory.get("score_ready") is not False
            or inventory.get("execution_authorization") != "not_granted"
            or inventory.get("rights_status") != "unverified_local_only"
            or inventory.get("missing_required_slots") != []
            or not isinstance(inventory.get("artifacts"), list)):
        _fail("inventory_candidate_schema_or_status_invalid")
    by_role = {}
    for artifact in inventory["artifacts"]:
        if (not isinstance(artifact, dict)
                or set(artifact) != {"role", "size_bytes", "sha256"}
                or not isinstance(artifact.get("role"), str)
                or type(artifact.get("size_bytes")) is not int
                or artifact["size_bytes"] <= 0
                or not isinstance(artifact.get("sha256"), str)
                or not SHA256.fullmatch(artifact["sha256"])):
            _fail("inventory_artifact_schema_invalid")
        role = artifact["role"]
        if role in by_role:
            _fail("inventory_artifact_role_duplicate")
        by_role[role] = artifact
    if set(INVENTORY_BINDINGS) - set(by_role):
        _fail("inventory_required_source_roles_missing")
    return {"raw_sha256_verified": True, "by_role": by_role}


def _verify_inventory_bindings(inventory: dict[str, Any], raw_inputs: dict[str, bytes],
                               pinned_code: dict[str, bytes]) -> None:
    for inventory_role, source in INVENTORY_BINDINGS.items():
        if isinstance(source, str) and source in raw_inputs:
            raw = raw_inputs[source]
        else:
            raw = pinned_code.get(source)
        item = inventory["by_role"].get(inventory_role)
        if (raw is None or item is None or item["size_bytes"] != len(raw)
                or item["sha256"] != _sha(raw)):
            _fail("inventory_artifact_bytes_mismatch")


def _git_blob(root: Path, commit: str, relpath: str, role: str) -> bytes:
    if not COMMIT_RE.fullmatch(commit):
        _fail("pin_commit_must_be_full_object_id")
    spec = f"{commit}:{relpath}"
    try:
        commit_type = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-t", commit],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10).stdout.strip()
        if commit_type != b"commit":
            _fail("pin_object_is_not_commit")
        tree_mode = subprocess.run(
            ["git", "-C", str(root), "ls-tree", commit, "--", relpath],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        _fail(f"{role}_git_blob_unavailable")
    fields = tree_mode.decode("ascii", errors="ignore").strip().split()
    if len(fields) < 3 or fields[1] != "blob":
        _fail(f"{role}_not_tracked_blob_at_pin")
    try:
        actual_mode = int(fields[0], 8)
    except ValueError:
        _fail(f"{role}_git_mode_invalid")
    if actual_mode not in {0o100644, 0o100755}:
        _fail(f"{role}_git_mode_invalid")
    try:
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", spec],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        _fail(f"{role}_git_blob_unavailable")
    return blob


def _verify_current_head(root: Path, commit: str) -> None:
    if not COMMIT_RE.fullmatch(commit):
        _fail("pin_commit_must_be_full_object_id")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10).stdout.decode("ascii").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        _fail("current_head_unavailable")
    if head != commit:
        _fail("pin_commit_must_equal_current_head")


def _verify_pinned_code(root: Path, commit: str) -> dict[str, bytes]:
    verified = {}
    for index, relpath in enumerate(PINNED_CODE_PATHS):
        pinned = _git_blob(root, commit, relpath, f"code_{index}")
        if not pinned or len(pinned) > MAX_FILE_BYTES:
            _fail("pinned_code_blob_size_invalid")
        path = PurePosixPath(relpath)
        root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                          | getattr(os, "O_NOFOLLOW", 0))
        try:
            parent_fd = _open_dir_chain(root_fd, path.parent.as_posix())
        finally:
            os.close(root_fd)
        try:
            fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        except OSError:
            os.close(parent_fd)
            _fail("pinned_code_worktree_unavailable")
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) not in {0o644, 0o755}
                    or info.st_size < 1 or info.st_size > MAX_FILE_BYTES):
                _fail("pinned_code_owner_mode_type_or_size_invalid")
            chunks, remaining = [], info.st_size
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    _fail("pinned_code_short_read")
                chunks.append(chunk)
                remaining -= len(chunk)
            worktree = b"".join(chunks)
            if os.read(fd, 1) or worktree != pinned:
                _fail("pinned_code_worktree_blob_mismatch")
            verified[relpath] = pinned
        finally:
            os.close(fd)
            os.close(parent_fd)
    return verified


def _build_claim_and_selection(*, claim_raw: bytes, worksheet_raw: bytes,
                               lineage_raw: bytes) -> tuple[bytes, bytes, dict, dict]:
    claim_doc = _strict_json(claim_raw, "claim_manifest_json_invalid")
    worksheet = _strict_json(worksheet_raw, "worksheet_json_invalid")
    lineage = _strict_json(lineage_raw, "lineage_json_invalid")
    if (not isinstance(claim_doc, dict) or set(claim_doc) != CLAIM_MANIFEST_KEYS
            or type(claim_doc.get("schema_version")) is not int
            or claim_doc["schema_version"] != 1
            or not isinstance(claim_doc.get("claims"), list)
            or len(claim_doc["claims"]) != ROW_COUNT):
        _fail("claim_manifest_schema_invalid")
    if (not isinstance(worksheet, dict) or not isinstance(worksheet.get("records"), list)
            or len(worksheet["records"]) != ROW_COUNT):
        _fail("worksheet_cohort_invalid")
    if (not isinstance(lineage, dict) or not isinstance(lineage.get("records"), list)
            or len(lineage["records"]) != ROW_COUNT):
        _fail("lineage_cohort_invalid")
    claims, selections = [], []
    seen_video, seen_claim, seen_token = set(), set(), set()
    for claim, sheet, line in zip(claim_doc["claims"], worksheet["records"],
                                  lineage["records"], strict=True):
        if (not isinstance(claim, dict) or set(claim) != CLAIM_ROW_KEYS
                or not isinstance(sheet, dict) or not isinstance(line, dict)
                or set(line) != LINEAGE_ROW_KEYS):
            _fail("claim_lineage_row_schema_invalid")
        text = claim.get("claim_text")
        video_id, claim_id = claim.get("video_id"), claim.get("claim_id")
        claim_sha = _sha(text.encode("utf-8")) if isinstance(text, str) else None
        token = sheet.get("review_token")
        if (not isinstance(text, str) or not text.strip()
                or claim.get("claim_text_sha256") != claim_sha
                or video_id != sheet.get("video_id") or video_id != line.get("video_id")
                or claim_id != sheet.get("claim_id") or claim_id != line.get("claim_id")
                or claim_sha != line.get("claim_text_sha256")
                or not isinstance(token, str) or not token.strip()
                or token != token.strip() or token in seen_token):
            _fail("claim_worksheet_lineage_binding_invalid")
        if video_id in seen_video or claim_id in seen_claim:
            _fail("claim_cohort_identity_duplicate")
        seen_video.add(video_id)
        seen_claim.add(claim_id)
        seen_token.add(token)
        claims.append({key: claim[key] for key in (
            "video_id", "claim_id", "claim_text", "claim_text_sha256")})
        selections.append({
            "review_token": token,
            "video_id": video_id,
            "claim_id": claim_id,
            "dataset_video_id": line.get("dataset_video_id"),
            "dataset_question_id": line.get("dataset_question_id"),
        })
    return (
        json.dumps({"records": claims}, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8"),
        json.dumps({"records": selections}, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8"),
        worksheet, lineage,
    )


def _target_manifest(target_raw: bytes) -> bytes:
    candidate = _strict_json(target_raw, "target_candidate_json_invalid")
    if (not isinstance(candidate, dict)
            or candidate.get("purpose") != "private_nextgqa_diagnostic_target_candidate_unvalidated"
            or candidate.get("status") != "diagnostic_candidate_unvalidated"
            or candidate.get("requested_records") != ROW_COUNT
            or candidate.get("effective_records") != ROW_COUNT
            or not isinstance(candidate.get("records"), list)
            or len(candidate["records"]) != ROW_COUNT):
        _fail("target_candidate_schema_invalid")
    rows = []
    for row in candidate["records"]:
        if not isinstance(row, dict) or any(key not in row for key in TARGET_SOURCE_FIELDS):
            _fail("target_candidate_source_fields_missing")
        rows.append({key: row[key] for key in TARGET_SOURCE_FIELDS})
    return json.dumps({"records": rows}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def run_binding(*, args: argparse.Namespace, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    if not root.is_dir():
        _fail("repo_root_invalid")
    _verify_current_head(root, args.pin_commit)
    pinned_code = _verify_pinned_code(root, args.pin_commit)
    raw = {}
    private_roles = {
        "csv": (args.csv, DATASET_CSV_MODES),
        "claim_manifest": (args.claim_manifest, PRIVATE_JSON_MODES),
        "worksheet": (args.worksheet, PRIVATE_JSON_MODES),
        "lineage": (args.lineage, PRIVATE_JSON_MODES),
        "target_candidate": (args.target_candidate, PRIVATE_JSON_MODES),
    }
    total = 0
    for role, (path_arg, modes) in private_roles.items():
        raw[role], _ = _read_owned_file(
            root, path_arg, role=role, allowed_modes=modes)
        total += len(raw[role])
        if total > MAX_TOTAL_BYTES:
            _fail("input_total_size_exceeds_limit")
    protocol_raw, protocol_rel = _read_owned_file(
        root, args.protocol, role="protocol", allowed_modes=PINNED_SOURCE_MODES)
    config_raw, config_rel = _read_owned_file(
        root, args.config, role="config", allowed_modes=PINNED_SOURCE_MODES)
    total += len(protocol_raw) + len(config_raw)
    if total > MAX_TOTAL_BYTES:
        _fail("input_total_size_exceeds_limit")
    inventory_raw = _read_candidate_inventory(root, args.inventory)
    total += len(inventory_raw)
    if total > MAX_TOTAL_BYTES:
        _fail("input_total_size_exceeds_limit")
    inventory = _validate_inventory(inventory_raw, args.expected_inventory_sha256)
    protocol_blob = _git_blob(root, args.pin_commit, protocol_rel, "protocol")
    config_blob = _git_blob(root, args.pin_commit, config_rel, "config")
    if protocol_raw != protocol_blob:
        _fail("protocol_worktree_blob_mismatch")
    if config_raw != config_blob:
        _fail("config_worktree_blob_mismatch")
    _verify_inventory_bindings(inventory, {**raw, "protocol": protocol_raw,
                                  "config": config_raw}, pinned_code)
    _claim_records, selection_raw, worksheet, lineage = _build_claim_and_selection(
        claim_raw=raw["claim_manifest"], worksheet_raw=raw["worksheet"],
        lineage_raw=raw["lineage"])
    target_raw = _target_manifest(raw["target_candidate"])
    # Recheck worksheet and lineage themselves have fixed three-row identity
    # and order, without trusting their own status fields as human evidence.
    del worksheet
    _ = lineage
    result = bind_qta_actual_qa_source(
        csv_raw=raw["csv"], claim_manifest_raw=_claim_records,
        selection_raw=selection_raw, target_manifest_raw=target_raw,
        lineage_manifest_raw=raw["lineage"], qta_protocol_raw=protocol_raw,
        expected_qta_protocol_sha256=_sha(protocol_blob),
        qta_config_raw=config_raw, expected_qta_config_sha256=_sha(config_blob))
    summary = result.get("summary") if isinstance(result, dict) else None
    if not isinstance(summary, dict):
        _fail("binder_summary_invalid")
    if result.get("private_bundle") is not None:
        # Drop references promptly. Never write, print, hash into public output,
        # or return this object from the CLI helper.
        result["private_bundle"] = None
    success = (summary.get("status") == "bound_unvalidated"
               and summary.get("requested_rows") == ROW_COUNT
               and summary.get("validated_rows") == ROW_COUNT)
    failure = summary.get("failure_code")
    if not success and (not isinstance(failure, str)
                        or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", failure)):
        failure = "source_binding_blocked"
    return {
        "schema_version": 1,
        "purpose": "qta_actual_qa_source_binding_smoke_aggregate_only",
        "status": "source_binding_passed_unvalidated" if success else "blocked",
        "requested_rows": ROW_COUNT,
        "validated_rows": ROW_COUNT if success else 0,
        "pin_commit": args.pin_commit,
        "pin_provenance": "caller_supplied_local_git_unverified",
        "inventory_pin_provenance": "caller_supplied_local_unverified",
        "inventory_is_trust_root": False,
        "inventory_content_match": True,
        "protocol_blob_match": True,
        "config_blob_match": True,
        "failure_code": None if success else failure,
        "rights_status": "unverified_local_only",
        "faithfulness_scored": False,
        "model_load_permitted": False,
        "execution_authorized": False,
        "scoring_authorized": False,
        "forwards": 0,
        "score_calls": 0,
        "private_bundle_retained": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="repo-relative NExT-GQA CSV path")
    parser.add_argument("--claim-manifest", required=True,
                        help="repo-relative private claim-input manifest path")
    parser.add_argument("--worksheet", required=True,
                        help="repo-relative frozen review worksheet path")
    parser.add_argument("--lineage", required=True,
                        help="repo-relative source lineage manifest path")
    parser.add_argument("--target-candidate", required=True,
                        help="repo-relative three-row diagnostic target candidate path")
    parser.add_argument("--protocol", required=True,
                        help="repo-relative QTA protocol source file path")
    parser.add_argument("--config", required=True,
                        help="repo-relative QTA config source file path")
    parser.add_argument("--inventory", required=True,
                        help="local candidate inventory path under data/private_nextgqa/inventory-*/")
    parser.add_argument("--expected-inventory-sha256", required=True,
                        help="explicit caller pin for the candidate inventory bytes; local and unverified")
    parser.add_argument("--pin-commit", required=True,
                        help="full current HEAD commit object ID; local caller pin is unverified")
    return parser


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_binding(args=args, repo_root=repo_root)
        exit_code = 0 if report["status"] == "source_binding_passed_unvalidated" else 2
    except (BindingInputError, OSError, ValueError, TypeError, KeyError,
            RecursionError, OverflowError):
        value = "input_or_binding_invalid"
        # Stable detail is safe only when it is an internal code, never path or
        # source content. Unknown exception text is deliberately discarded.
        if isinstance(sys.exc_info()[1], BindingInputError):
            candidate = str(sys.exc_info()[1])
            if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", candidate):
                value = candidate
        report = {
            "schema_version": 1,
            "purpose": "qta_actual_qa_source_binding_smoke_aggregate_only",
            "status": "blocked", "requested_rows": ROW_COUNT,
            "validated_rows": 0, "failure_code": value,
            "pin_provenance": "caller_supplied_local_git_unverified",
            "inventory_pin_provenance": "caller_supplied_local_unverified",
            "inventory_is_trust_root": False,
            "inventory_content_match": False,
            "rights_status": "unverified_local_only", "faithfulness_scored": False,
            "model_load_permitted": False, "execution_authorized": False,
            "scoring_authorized": False, "forwards": 0, "score_calls": 0,
            "private_bundle_retained": False,
        }
        exit_code = 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
