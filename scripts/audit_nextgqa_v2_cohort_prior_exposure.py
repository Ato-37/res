#!/usr/bin/env python3
"""Aggregate-only exact-byte prior-exposure audit for the frozen v2 cohort.

The audit hashes the 147 media files already bound by the private v2 plan and
compares their SHA-256 values, through the existing local HMAC key, with the
explicitly partial prior-exposure index. It emits counts and provenance
fingerprints only. It does not certify unmatched media as fresh, select or
exclude rows, decode videos, load a model, or measure faithfulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

try:
    from build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _loads, _open_path_no_follow,
        _stat_identity, _token,
    )
except ImportError:  # package import under unittest discovery
    from scripts.build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _loads, _open_path_no_follow,
        _stat_identity, _token,
    )

PLAN_NAME = "nextgqa_tlvs_exploratory_split_v2_20261018_05.json"
PREFLIGHT_NAME = "nextgqa_tlvs_exploratory_qta_preflight_receipt_v2_20261018_06.json"
INPUT_FREEZE_NAME = "nextgqa_tlvs_qta_v2_run_input_freeze_local_20261018_17.json"
INDEX_NAME = "prior_exposure_partial_20261016_03.json"
KEY_NAME = "prior_exposure_hmac_key_20261016_01.bin"
DATA_FILES = {
    "test_csv": "NExT-GQA/datasets/nextgqa/test.csv",
    "gsub_test_json": "NExT-GQA/datasets/nextgqa/gsub_test.json",
    "frame2time_test_json": "NExT-GQA/datasets/nextgqa/frame2time_test.json",
    "map_vid_vidorID_json": "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json",
}
EXPECTED_N = 147
MAX_PLAN_BYTES = 128 * 1024 * 1024
MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 8 * 1024 * 1024 * 1024


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_private(path: Path, private_root: Path, *, max_bytes: int) -> bytes:
    root = private_root.resolve(strict=True)
    info = root.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700 or path.parent.resolve(strict=True) != root):
        raise ValueError("private_input_root_invalid")
    try:
        fd = _open_path_no_follow(path)
    except OSError as exc:
        raise ValueError("private_input_unavailable_or_symlink") from exc
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 0 or before.st_size > max_bytes):
            raise ValueError("private_input_mode_owner_or_size_invalid")
        pieces = []
        total = 0
        while True:
            block = stream.read(min(1024 * 1024, max_bytes + 1 - total))
            if not block:
                break
            pieces.append(block)
            total += len(block)
            if total > max_bytes:
                raise ValueError("private_input_size_limit_exceeded")
        after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after) or total != after.st_size:
            raise ValueError("private_input_changed_during_read")
        return b"".join(pieces)


def _hash_file_stable(path: Path, *, max_bytes: int) -> tuple[str, int]:
    try:
        fd = _open_path_no_follow(path)
    except OSError as exc:
        raise ValueError("input_unavailable_or_symlink") from exc
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_size < 0
                or before.st_size > max_bytes):
            raise ValueError("input_size_or_type_invalid")
        total = 0
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("input_size_limit_exceeded")
            digest.update(block)
        after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after) or total != after.st_size:
            raise ValueError("input_changed_during_read")
    return digest.hexdigest(), total


def _load_index(raw: bytes) -> tuple[set[str], str]:
    value: Any = _loads(raw)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("purpose") != "outcome_blind_prior_exposure_index"
            or value.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(value.get("coverage"), dict)
            or value["coverage"].get("historical_completeness_authenticated") is not False
            or not isinstance(value.get("sources"), list)
            or type(value["coverage"].get("source_count")) is not int
            or value["coverage"].get("source_count") != len(value["sources"])
            or not isinstance(value.get("identities"), dict)):
        raise ValueError("prior_index_schema_invalid")
    tokens = value["identities"].get("media_sha256_hmac_sha256")
    if (not isinstance(tokens, list)
            or any(not isinstance(item, str) or len(item) != 64
                   or any(c not in "0123456789abcdef" for c in item) for item in tokens)
            or len(tokens) != len(set(tokens))):
        raise ValueError("prior_index_media_identity_invalid")
    roster_raw = json.dumps(value["sources"], sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
    return set(tokens), _sha256(roster_raw)


def _validate_frozen_inputs(*, plan_raw: bytes, preflight_raw: bytes,
                            input_freeze_raw: bytes,
                            dataset_hashes: dict[str, str],
                            plan_sha256: str) -> tuple[dict, dict]:
    plan, preflight, input_freeze = (_loads(plan_raw), _loads(preflight_raw),
                                     _loads(input_freeze_raw))
    if (not isinstance(plan, dict) or plan.get("status") !=
            "test_associated_exploratory_qid_valid_plan_not_evaluation"
            or not isinstance(preflight, dict)
            or preflight.get("status") != "preflight_receipt_only_no_model_scoring"
            or not isinstance(input_freeze, dict)
            or type(input_freeze.get("schema_version")) is not int
            or input_freeze.get("schema_version") != 1
            or input_freeze.get("status") !=
            "local_v2_inputs_frozen_integrity_only_not_authorization"):
        raise ValueError("v2_plan_or_preflight_status_invalid")
    gate_counts = input_freeze.get("counts")
    gate_data = input_freeze.get("data_sha256")
    expected_gate_data = {DATA_FILES[key]: value for key, value in dataset_hashes.items()}
    if (input_freeze.get("plan_sha256") != plan_sha256
            or input_freeze.get("preflight_receipt_sha256") != _sha256(preflight_raw)
            or not isinstance(gate_counts, dict)
            or gate_counts.get("evaluation_rows_bound") != EXPECTED_N
            or gate_counts.get("evaluation_videos") != EXPECTED_N
            or not isinstance(gate_data, dict) or gate_data != expected_gate_data):
        raise ValueError("v2_local_input_freeze_binding_mismatch")
    evaluation, calibration = plan.get("evaluation"), plan.get("calibration")
    counts = preflight.get("counts")
    if (not isinstance(evaluation, list) or len(evaluation) != EXPECTED_N
            or not isinstance(calibration, list) or len(calibration) != 343
            or not isinstance(counts, dict)
            or any(type(counts.get(key)) is not int for key in
                   ("evaluation_rows_bound", "invalid_annotation_intervals", "join_failures"))
            or counts.get("evaluation_rows_bound") != EXPECTED_N
            or counts.get("invalid_annotation_intervals") != 0
            or counts.get("join_failures") != 0):
        raise ValueError("v2_frozen_denominator_invalid")
    provenance = plan.get("provenance")
    receipt_hashes = preflight.get("input_sha256")
    if not isinstance(provenance, dict) or not isinstance(receipt_hashes, dict):
        raise ValueError("v2_provenance_missing")
    if (not isinstance(plan_sha256, str) or len(plan_sha256) != 64
            or any(c not in "0123456789abcdef" for c in plan_sha256)
            or receipt_hashes.get("split_plan") != plan_sha256):
        raise ValueError("v2_preflight_split_plan_hash_mismatch")
    prov_keys = {
        "test_csv": "test_csv_sha256",
        "gsub_test_json": "gsub_test_sha256",
        "frame2time_test_json": "frame2time_test_sha256",
        "map_vid_vidorID_json": "map_vid_vidorID_sha256",
    }
    for logical, plan_key in prov_keys.items():
        current = dataset_hashes[logical]
        if (provenance.get(plan_key) != current
                or receipt_hashes.get(logical) != current):
            raise ValueError("current_dataset_hash_mismatch")
    receipt_eval = preflight.get("evaluation_row_bindings")
    if not isinstance(receipt_eval, list) or len(receipt_eval) != EXPECTED_N:
        raise ValueError("v2_preflight_row_binding_count_invalid")
    if any(not isinstance(row, dict) for row in evaluation + calibration + receipt_eval):
        raise ValueError("v2_plan_or_preflight_row_schema_invalid")
    plan_media = [row.get("media_sha256") for row in evaluation]
    if (len(plan_media) != EXPECTED_N
            or any(not isinstance(x, str) or len(x) != 64 for x in plan_media)
            or len(set(plan_media)) != EXPECTED_N):
        raise ValueError("v2_plan_media_binding_invalid")
    plan_bindings = [row.get("selected_row_binding_sha256") for row in evaluation]
    receipt_bindings = [row.get("row_binding_sha256") for row in receipt_eval]
    if (any(not isinstance(x, str) or len(x) != 64
            or any(c not in "0123456789abcdef" for c in x) for x in plan_bindings)
            or plan_bindings != receipt_bindings):
        raise ValueError("v2_plan_preflight_row_binding_mismatch")
    video_ids = [row.get("video_id") for row in evaluation]
    pairs = [(row.get("video_id"), row.get("qid")) for row in evaluation]
    if (len(video_ids) != EXPECTED_N or any(not isinstance(x, str) or not x for x in video_ids)
            or len(set(video_ids)) != EXPECTED_N
            or any(not isinstance(v, str) or not v or not isinstance(q, str) or not q
                   for v, q in pairs)
            or len(set(pairs)) != EXPECTED_N):
        raise ValueError("v2_evaluation_identity_denominator_invalid")
    return plan, preflight


def audit_media_entries(*, entries: list[dict[str, Any]], index_tokens: set[str], key: bytes,
                        private_root: Path, expected_count: int = EXPECTED_N) -> dict[str, Any]:
    """Hash plan-bound media files and return aggregate counts only."""
    if len(entries) != expected_count:
        raise ValueError("evaluation_denominator_invalid")
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("hmac_key_invalid")
    private_abs = Path(os.path.abspath(private_root))
    seen_paths: set[Path] = set()
    seen_media_hashes: set[str] = set()
    matched_unique = 0
    total_bytes = 0
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("evaluation_entry_invalid")
        expected_hash, raw_path = row.get("media_sha256"), row.get("media_path")
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(c not in "0123456789abcdef" for c in expected_hash)
                or not isinstance(raw_path, str)):
            raise ValueError("evaluation_media_binding_invalid")
        media_path = Path(os.path.abspath(raw_path))
        try:
            media_path.relative_to(private_abs)
        except ValueError as exc:
            raise ValueError("evaluation_media_outside_private_root") from exc
        if media_path in seen_paths:
            raise ValueError("evaluation_media_path_duplicate")
        seen_paths.add(media_path)
        remaining = MAX_TOTAL_MEDIA_BYTES - total_bytes
        if remaining <= 0:
            raise ValueError("evaluation_total_media_size_limit_exceeded")
        actual_hash, byte_count = _hash_file_stable(
            media_path, max_bytes=min(MAX_MEDIA_BYTES, remaining))
        if actual_hash != expected_hash:
            raise ValueError("evaluation_media_hash_mismatch")
        total_bytes += byte_count
        if actual_hash in seen_media_hashes:
            raise ValueError("evaluation_media_duplicate")
        seen_media_hashes.add(actual_hash)
        candidate_token = _token(key, "media-sha256", (actual_hash,))
        if candidate_token in index_tokens:
            matched_unique += 1
    return {
        "evaluation_rows_requested": expected_count,
        "evaluation_media_files_hashed": len(seen_paths),
        "evaluation_unique_media_count": len(seen_media_hashes),
        "evaluation_media_bytes_hashed": total_bytes,
        "exact_byte_matches_to_supplied_partial_index": matched_unique,
        "unique_media_unmatched_to_supplied_index": len(seen_media_hashes) - matched_unique,
    }


def build_receipt(*, plan: dict, plan_sha256: str, preflight_sha256: str,
                  input_freeze_sha256: str,
                  auditor_script_sha256: str,
                  dataset_hashes: dict[str, str], index_raw: bytes, key: bytes,
                  private_root: Path, expected_count: int = EXPECTED_N) -> dict[str, Any]:
    index_tokens, roster_sha = _load_index(index_raw)
    counts = audit_media_entries(entries=plan["evaluation"], index_tokens=index_tokens,
                                  key=key, private_root=private_root,
                                  expected_count=expected_count)
    return {
        "schema_version": 1,
        "purpose": "nextgqa_v2_cohort_prior_exposure_audit_aggregate_only",
        "status": "completed",
        "provenance": {
            "input_freeze_sha256": input_freeze_sha256,
            "auditor_script_sha256": auditor_script_sha256,
            "v2_plan_sha256": plan_sha256,
            "v2_preflight_sha256": preflight_sha256,
            "dataset_sha256": dataset_hashes,
            "prior_index_sha256": _sha256(index_raw),
            "prior_source_roster_sha256": roster_sha,
            "prior_index_completeness": "known_partial_unauthenticated",
            "identity_match_policy": "exact_media_sha256_hmac_only_same_key_and_canonicalization",
        },
        "counts": counts,
        "interpretation": {
            "unmatched_means_fresh": False,
            "matches_excluded_or_resampled": False,
            "source_event_independence_assessed": False,
            "rights_status": "unverified_local_only",
            "faithfulness_scored": False,
            "model_forwards": 0,
        },
    }


def write_private_exclusive(path: Path, payload: dict[str, Any], *, private_root: Path) -> None:
    root = private_root.resolve(strict=True)
    if (path.parent.resolve(strict=True) != root or stat.S_IMODE(root.stat().st_mode) != 0o700
            or root.stat().st_uid != os.getuid()):
        raise ValueError("private_output_root_invalid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def run(repo: Path, output_name: str) -> dict[str, Any]:
    if Path(output_name).name != output_name:
        raise ValueError("output_name_must_be_basename")
    private = repo / "data/private_nextgqa"
    plan_path, preflight_path = private / PLAN_NAME, private / PREFLIGHT_NAME
    plan_raw = _read_private(plan_path, private, max_bytes=MAX_PLAN_BYTES)
    preflight_raw = _read_private(preflight_path, private, max_bytes=MAX_PLAN_BYTES)
    input_freeze_raw = _read_private(private / INPUT_FREEZE_NAME, private,
                                     max_bytes=MAX_PLAN_BYTES)
    dataset_hashes = {}
    for logical, rel in DATA_FILES.items():
        digest, _ = _hash_file_stable(repo / rel, max_bytes=MAX_SOURCE_BYTES)
        dataset_hashes[logical] = digest
    plan, preflight = _validate_frozen_inputs(plan_raw=plan_raw,
        preflight_raw=preflight_raw, dataset_hashes=dataset_hashes,
        input_freeze_raw=input_freeze_raw,
        plan_sha256=_sha256(plan_raw))
    index_path, key_path = private / INDEX_NAME, private / KEY_NAME
    index_raw = _read_private(index_path, private, max_bytes=MAX_SOURCE_BYTES)
    key = _read_private(key_path, private, max_bytes=4096)
    if len(key) < 32:
        raise ValueError("hmac_key_invalid")
    receipt = build_receipt(plan=plan, plan_sha256=_sha256(plan_raw),
        preflight_sha256=_sha256(preflight_raw), dataset_hashes=dataset_hashes,
        input_freeze_sha256=_sha256(input_freeze_raw),
        auditor_script_sha256=_hash_file_stable(Path(__file__), max_bytes=MAX_PLAN_BYTES)[0],
        index_raw=index_raw, key=key, private_root=private)
    write_private_exclusive(private / output_name, receipt, private_root=private)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", required=True,
                        help="new basename in data/private_nextgqa; existing files are refused")
    args = parser.parse_args(argv)
    try:
        receipt = run(args.repo.resolve(), args.output_name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"audit_failed:{str(exc).split(':', 1)[0]}")
    print(json.dumps({"status": receipt["status"], "counts": receipt["counts"],
                      "privacy": "aggregate counts only; private receipt"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
