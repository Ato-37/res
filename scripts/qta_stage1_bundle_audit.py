#!/usr/bin/env python3
"""CPU-only, non-authorizing bundle audit for the three-row QTA stage 1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from audit_nextgqa_review_handoff import audit as audit_review, audit_custody
from audit_nextgqa_review_release_order import verify_review_release_order
from nextgqa_completed_review_attestation import verify_attestation
from render_nextgqa_claim_review_html import render as render_review_html


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _json(raw: bytes):
    if not isinstance(raw, bytes):
        raise ValueError("input_bytes_required")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def _failure(exc: Exception) -> str:
    # Existing validators expose stable ValueError codes; never echo source bytes.
    value = str(exc)
    if value and len(value) <= 120 and re.fullmatch(r"[a-z][a-z0-9_]*", value):
        return value
    return "component_validation_failed"


def _read_exact_source(value: bytes | str | Path, role: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, (str, Path)):
        raise ValueError("artifact_source_type_invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(Path(value), flags)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600):
            raise ValueError("artifact_file_invalid")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (after.st_size != before.st_size or after.st_ino != before.st_ino
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise ValueError("artifact_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _open_private_media_dir(path: Path, expected_basenames: set[str]) -> tuple[int, tuple[int, int, int, int]]:
    """Open and validate the private directory before any media path access."""
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{24}\.mp4", name)
           for name in expected_basenames):
        raise ValueError("media_names_invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("media_directory_invalid")
        if set(os.listdir(fd)) != expected_basenames:
            raise ValueError("media_directory_contents_invalid")
        identity = (info.st_dev, info.st_ino, info.st_mode, info.st_uid)
        return fd, identity
    except Exception:
        os.close(fd)
        raise


def _read_private_media_at(dir_fd: int, basename: str) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{24}\.mp4", basename):
        raise ValueError("media_name_invalid")
    fd = os.open(basename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600):
            raise ValueError("media_file_invalid")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (after.st_size != before.st_size or after.st_ino != before.st_ino
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise ValueError("media_changed_during_read")
        identity = (before.st_dev, before.st_ino, before.st_mode, before.st_uid,
                    before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        return b"".join(chunks), identity
    finally:
        os.close(fd)


def _bind_artifact_role(artifact_sources, role: str, expected: bytes,
                        failures: list[str], details: dict[str, Any]) -> None:
    codes = []
    try:
        if role not in artifact_sources:
            raise ValueError("artifact_role_missing")
        actual = _read_exact_source(artifact_sources[role], role)
        if actual != expected:
            raise ValueError("artifact_role_byte_mismatch")
    except Exception as exc:
        codes.append(_failure(exc))
    details[role] = {"status": "byte_match" if not codes else "invalid",
                     "failure_codes": codes, "passed": not codes}
    failures.extend(f"artifact_binding:{role}:{code}" for code in codes)


def audit_qta_stage1_bundle(*, worksheet_raw: bytes, lineage_raw: bytes,
                            review_raw: bytes, review_index_raw: bytes,
                            custody_raw: bytes, pre_review_receipt_raw: bytes,
                            integrity_roster_raw: bytes, release_roster_raw: bytes,
                            inventory_raw: bytes, release_witness_raw: bytes,
                            reviewer_attestation_raw: bytes, media_dir: Path,
                            expected_lineage_sha256: str,
                            expected_custody_sha256: str,
                            root_public_key_pem: bytes,
                            expected_root_public_der_sha256: str,
                            expected_ceremony_id: str,
                            artifact_sources: Mapping[str, bytes | str | Path]) -> dict[str, Any]:
    """Re-run the stage-1 validators from exact supplied bytes.

    This is an aggregation/preflight report only. It does not evaluate the
    separate clock/frame/calibration gates, authorize model loading, or score.
    """
    components: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def record(name: str, operation):
        try:
            result = operation()
            if isinstance(result, dict):
                codes = list(result.get("failure_codes", []))
                status = result.get("status", "invalid")
                passed = (status not in {"invalid", "blocked", ""}
                          and not codes)
            elif isinstance(result, list):
                codes = list(result)
                status, passed = ("valid" if not codes else "invalid"), not codes
            else:
                raise ValueError("validator_result_shape_invalid")
            components[name] = {"status": status, "failure_codes": codes,
                                "passed": passed}
            failures.extend(f"{name}:{code}" for code in codes)
            if not passed and not codes:
                failures.append(f"{name}:not_valid")
            return result if passed else None
        except Exception as exc:  # validators can raise several stable error types
            code = _failure(exc)
            components[name] = {"status": "invalid", "failure_codes": [code],
                                "passed": False}
            failures.append(f"{name}:{code}")
            return None

    parsed: dict[str, Any] = {}
    for name, raw in (("worksheet", worksheet_raw), ("lineage", lineage_raw),
                      ("review", review_raw), ("review_index", review_index_raw),
                      ("custody", custody_raw)):
        try:
            parsed[name] = _json(raw)
            components[f"{name}_bytes"] = {"status": "parsed", "failure_codes": [],
                                           "passed": True}
        except Exception as exc:
            code = _failure(exc)
            components[f"{name}_bytes"] = {"status": "invalid", "failure_codes": [code],
                                           "passed": False}
            failures.append(f"{name}_bytes:{code}")

    # Snapshot all join-critical artifact roles and media before validation.
    # Verifiers and joins below consume these same byte objects; no relevant
    # path is reopened after the snapshot boundary.
    artifact_snapshot: dict[str, bytes | str | Path] = {}
    snapshot_failures: list[str] = []
    snapshot_roles = {
        "claim_mapping_worksheet", "claim_lineage", "review_custody",
        "claim_input_manifest", "claim_input_sidecar", "standalone_review_html",
        "review_package_html",
        *(f"review_package_media_{i:02d}" for i in range(1, 4)),
    }
    try:
        artifact_snapshot = dict(artifact_sources)
    except Exception:
        snapshot_failures.append("artifact_sources_invalid")
    for role in sorted(snapshot_roles & set(artifact_snapshot)):
        try:
            artifact_snapshot[role] = _read_exact_source(artifact_snapshot[role], role)
        except Exception as exc:
            artifact_snapshot.pop(role, None)
            snapshot_failures.append(f"artifact_snapshot:{role}:{_failure(exc)}")

    media_snapshot: dict[str, bytes] = {}
    media_snapshot_identity: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    media_tokens = []
    worksheet_rows = parsed.get("worksheet", {}).get("records", []) if isinstance(
        parsed.get("worksheet"), dict) else []
    if isinstance(worksheet_rows, list) and len(worksheet_rows) == 3:
        media_tokens = [row.get("review_token") for row in worksheet_rows
                        if isinstance(row, dict) and isinstance(row.get("review_token"), str)]
    tokens_valid = (len(media_tokens) == 3 and len(set(media_tokens)) == 3
                    and all(re.fullmatch(r"[A-Za-z0-9_-]{24}", token)
                            for token in media_tokens))
    basenames = sorted(f"{token}.mp4" for token in media_tokens) if tokens_valid else []
    media_dir_fd = None
    media_dir_identity = None
    if not tokens_valid:
        snapshot_failures.append("media_snapshot:review_token_set_invalid")
    else:
        try:
            media_dir_fd, media_dir_identity = _open_private_media_dir(Path(media_dir), set(basenames))
        except Exception as exc:
            snapshot_failures.append(f"media_snapshot:directory:{_failure(exc)}")
    if media_dir_fd is not None:
        for ordinal, basename in enumerate(basenames, 1):
            try:
                media_snapshot[basename], media_snapshot_identity[basename] = _read_private_media_at(
                    media_dir_fd, basename)
            except Exception as exc:
                snapshot_failures.append(f"media_snapshot:record_{ordinal:02d}:{_failure(exc)}")
    for ordinal, basename in enumerate(sorted(media_snapshot), 1):
        role = f"staged_media_{ordinal:02d}"
        # The staged-media inventory role is populated from the exact single
        # media snapshot, rather than reopened from a separate path/value.
        if role in artifact_sources:
            artifact_snapshot[role] = media_snapshot[basename]
    components["artifact_snapshot"] = {
        "status": "captured" if not snapshot_failures else "invalid",
        "failure_codes": sorted(snapshot_failures), "passed": not snapshot_failures,
    }
    failures.extend(snapshot_failures)

    review_report = None
    if all(k in parsed for k in ("worksheet", "lineage", "review", "review_index")):
        review_report = record("review_handoff", lambda: audit_review(
            parsed["worksheet"], worksheet_raw, parsed["lineage"], lineage_raw,
            parsed["review"], review_raw, parsed["review_index"], review_index_raw,
            expected_lineage_sha256))
        if review_report is not None and (
                review_report.get("status") != "unvalidated_handoff_structurally_valid"
                or review_report.get("execution_authorization") != "not_granted"
                or review_report.get("semantic_review_authenticated") is not False):
            components["review_handoff"]["passed"] = False
            components["review_handoff"]["failure_codes"].append("review_handoff_status_invalid")
            failures.append("review_handoff:review_handoff_status_invalid")
        if review_report is not None and review_report.get("reviewed_records") != 3:
            components["review_handoff"]["passed"] = False
            components["review_handoff"]["failure_codes"].append("reviewed_record_count_not_three")
            failures.append("review_handoff:reviewed_record_count_not_three")
    else:
        components["review_handoff"] = {"status": "blocked", "failure_codes": ["source_json_invalid"],
                                         "passed": False}

    if all(k in parsed for k in ("worksheet", "lineage", "custody")):
        record("custody", lambda: audit_custody(
            parsed["worksheet"], worksheet_raw, parsed["lineage"], lineage_raw,
            parsed["custody"], custody_raw, Path(media_dir), expected_custody_sha256,
            media_bytes_by_name=media_snapshot))
    else:
        components["custody"] = {"status": "blocked", "failure_codes": ["source_json_invalid"],
                                  "passed": False}

    release_report = record("release_witness", lambda: verify_review_release_order(
        receipt_raw=pre_review_receipt_raw,
        integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw,
        inventory_raw=inventory_raw, artifact_sources=artifact_snapshot,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256,
        attestation_raw=release_witness_raw,
        expected_ceremony_id=expected_ceremony_id))
    if release_report is not None and (
            release_report.get("status") != "review_release_order_witness_attestation_valid"
            or release_report.get("model_load_permitted") is not False
            or release_report.get("execution_authorization") != "not_granted"):
        components["release_witness"]["passed"] = False
        components["release_witness"]["failure_codes"].append("release_status_invalid")
        failures.append("release_witness:release_status_invalid")

    artifact_binding_details: dict[str, Any] = {}
    if release_report is not None and all(k in parsed for k in
                                           ("worksheet", "lineage", "custody")):
        for role, raw in (("claim_mapping_worksheet", worksheet_raw),
                          ("claim_lineage", lineage_raw),
                          ("review_custody", custody_raw)):
            _bind_artifact_role(artifact_snapshot, role, raw, failures,
                                artifact_binding_details)

        # The custody audit enforces exactly worksheet-token.mp4 filenames.
        # Inventory roles are sorted by basename, so compare every indexed role
        # against both that exact media file and its package counterpart.
        for index, basename in enumerate(basenames, start=1):
            staged = media_snapshot.get(basename)
            if staged is not None:
                staged_role = f"staged_media_{index:02d}"
                package_role = f"review_package_media_{index:02d}"
                _bind_artifact_role(artifact_snapshot, staged_role, staged, failures,
                                    artifact_binding_details)
                _bind_artifact_role(artifact_snapshot, package_role, staged, failures,
                                    artifact_binding_details)
            else:
                code = "media_snapshot_missing"
                artifact_binding_details[f"media_record_{index:02d}"] = {
                    "status": "invalid", "failure_codes": [code], "passed": False}
                failures.append(f"artifact_binding:media_record_{index:02d}:{code}")

        manifest_value = sidecar_value = None
        manifest_raw = sidecar_raw = None
        has_manifest = "claim_input_manifest" in artifact_snapshot
        has_sidecar = "claim_input_sidecar" in artifact_snapshot
        try:
            if has_manifest != has_sidecar:
                raise ValueError("claim_input_bundle_pair_incomplete")
            if has_manifest:
                manifest_raw = artifact_snapshot["claim_input_manifest"]
                sidecar_raw = artifact_snapshot["claim_input_sidecar"]
                manifest_value, sidecar_value = _json(manifest_raw), _json(sidecar_raw)
        except Exception as exc:
            code = _failure(exc)
            artifact_binding_details["claim_input_bundle"] = {
                "status": "invalid", "failure_codes": [code], "passed": False}
            failures.append(f"artifact_binding:claim_input_bundle:{code}")

        try:
            expected_html = render_review_html(
                parsed["worksheet"], worksheet_raw, parsed["lineage"], lineage_raw,
                expected_lineage_sha256, parsed["custody"], custody_raw,
                expected_custody_sha256,
                manifest_value, manifest_raw, sidecar_value, sidecar_raw)
            _bind_artifact_role(artifact_snapshot, "review_package_html", expected_html,
                                failures, artifact_binding_details)
            _bind_artifact_role(artifact_snapshot, "standalone_review_html", expected_html,
                                failures, artifact_binding_details)
        except Exception as exc:
            code = _failure(exc)
            artifact_binding_details["review_package_html"] = {
                "status": "invalid", "failure_codes": [code], "passed": False}
            failures.append(f"artifact_binding:review_package_html:{code}")
    else:
        artifact_binding_details["status"] = {
            "status": "blocked", "failure_codes": ["release_or_source_gate_failed"],
            "passed": False}
        failures.append("artifact_binding:release_or_source_gate_failed")

    # Verify the reviewer adoption over these exact export bytes and the hash of
    # the exact pre-review receipt bytes that the witness just validated.
    reviewer_report = None
    if all(k in parsed for k in ("worksheet", "lineage", "review", "review_index")):
        reviewer_report = record("reviewer_attestation", lambda: verify_attestation(
            attestation_raw=reviewer_attestation_raw,
            review_raw=review_raw, worksheet=parsed["worksheet"],
            worksheet_raw=worksheet_raw, lineage=parsed["lineage"],
            lineage_raw=lineage_raw, review=parsed["review"],
            review_index=parsed["review_index"], review_index_raw=review_index_raw,
            expected_lineage_sha256=expected_lineage_sha256,
            pre_review_receipt_sha256=hashlib.sha256(pre_review_receipt_raw).hexdigest(),
            integrity_roster_raw=integrity_roster_raw,
            root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256))
    else:
        components["reviewer_attestation"] = {
            "status": "blocked", "failure_codes": ["release_or_review_gate_failed"],
            "passed": False}
        failures.append("reviewer_attestation:release_or_review_gate_failed")

    # Explicit structural binding check: reviewer signature must attest the
    # hash passed above, while release witness report is tied to receipt bytes.
    if reviewer_report is not None and reviewer_report.get("pre_review_receipt_verified") is not False:
        components["reviewer_attestation"]["passed"] = False
        components["reviewer_attestation"]["failure_codes"].append("receipt_binding_not_explicit")
        failures.append("reviewer_attestation:receipt_binding_not_explicit")
    if reviewer_report is not None and (
            reviewer_report.get("status") != "reviewer_key_attested_unvalidated"
            or reviewer_report.get("execution_authorization") != "not_granted"
            or reviewer_report.get("key_holder_adoption_attested") is not True):
        components["reviewer_attestation"]["passed"] = False
        components["reviewer_attestation"]["failure_codes"].append("reviewer_attestation_status_invalid")
        failures.append("reviewer_attestation:reviewer_attestation_status_invalid")

    for gate, code in (("clock_frame", "clock_frame_audit_not_supplied"),
                       ("clock_calibration", "clock_calibration_binding_not_supplied")):
        components[gate] = {"status": "missing_separate_gate", "failure_codes": [code],
                            "passed": False}

    # Do not reread media bytes. Check only filesystem identity metadata so a
    # path replacement or in-place edit after the byte snapshot cannot be
    # mistaken for a still-current staged input by a later caller.
    media_still_current = True
    for ordinal, basename in enumerate(basenames, 1):
        expected = media_snapshot_identity.get(basename)
        try:
            if media_dir_fd is None:
                raise ValueError("media_directory_changed_after_snapshot")
            directory_now = os.fstat(media_dir_fd)
            if (directory_now.st_dev, directory_now.st_ino, directory_now.st_mode,
                    directory_now.st_uid) != media_dir_identity:
                raise ValueError("media_directory_changed_after_snapshot")
            directory_path_now = os.stat(Path(media_dir), follow_symlinks=False)
            if (directory_path_now.st_dev, directory_path_now.st_ino,
                    directory_path_now.st_mode, directory_path_now.st_uid) != media_dir_identity:
                raise ValueError("media_directory_path_changed_after_snapshot")
            current = os.stat(basename, dir_fd=media_dir_fd, follow_symlinks=False)
            observed = (current.st_dev, current.st_ino, current.st_mode, current.st_uid,
                        current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            if (expected is None or not stat.S_ISREG(current.st_mode)
                    or stat.S_IMODE(current.st_mode) != 0o600 or observed != expected):
                raise ValueError("media_path_changed_after_snapshot")
        except Exception as exc:
            media_still_current = False
            code = _failure(exc)
            key = f"media_record_{ordinal:02d}"
            artifact_binding_details[key] = {
                "status": "invalid", "failure_codes": [code], "passed": False}
            failures.append(f"artifact_binding:{key}:{code}")
    components["media_snapshot_stability"] = {
        "status": "unchanged" if media_still_current else "invalid",
        "failure_codes": [] if media_still_current else ["media_path_changed_after_snapshot"],
        "passed": media_still_current,
    }
    if media_dir_fd is not None:
        os.close(media_dir_fd)

    # Recompute from component states (rather than trusting any supplied report).
    all_stage1 = all(components.get(name, {}).get("passed") is True for name in
                     ("worksheet_bytes", "lineage_bytes", "review_bytes",
                      "review_index_bytes", "custody_bytes", "review_handoff",
                      "custody", "release_witness", "reviewer_attestation",
                      "artifact_snapshot", "media_snapshot_stability")) \
        and bool(artifact_binding_details) \
        and all(item.get("passed") is True for item in artifact_binding_details.values())
    if not all_stage1 and not failures:
        failures.append("stage1:incomplete_component_set")
    return {
        "status": "stage1_structurally_valid_non_authorizing" if all_stage1 else "blocked",
        "requested_records": 3,
        "effective_records": 3,
        "structurally_valid_records": 3 if all_stage1 else 0,
        "failure_count": len(failures),
        "failure_codes": sorted(set(failures)),
        "components": components,
        "artifact_role_bindings": artifact_binding_details,
        "clock_frame_status": "not_audited_separate_gate_required",
        "clock_calibration_status": "not_audited_separate_gate_required",
        "model_load_permitted": False,
        "score_ready": False,
        "reviewer_identity_verified": False,
        "semantic_review_authenticated": False,
        "frame_correspondence_verified": False,
        "execution_authorization": "not_granted",
    }
