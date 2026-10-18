#!/usr/bin/env python3
"""Read-only, non-authorizing verifier for a frozen two-phase QTA bundle.

Both trust anchors are required from the caller. This CLI never derives either
value from the candidate bundle and never authorizes model loading or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from qta_phase_freeze_receipts import PHASE_ARTIFACT_ROLES
from qta_phase2_plan_manifest import verify_qta_phase2_plan_candidate


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema_version", "purpose", "phase1_receipt", "phase1_artifacts",
    "phase2_receipt", "phase2_artifacts", "integrity_roster",
    "release_roster", "root_public_key", "attention_capture_sidecar_dir",
}
_MAX_MANIFEST_BYTES = 64_000
_MAX_FILE_BYTES = 24_000_000
_MAX_SIDECAR_BYTES = 384 * 1024 * 1024
_MAX_PHASE_ARTIFACT_BYTES = 96_000_000


class BundleError(ValueError):
    """Raised when a local receipt bundle is malformed or unsafe to read."""


def _read_private_file(root: Path, relative: Any, *, max_bytes: int) -> bytes:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise BundleError("bundle_relative_path_invalid")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BundleError("bundle_file_missing") from exc
    if root not in resolved.parents:
        raise BundleError("bundle_path_escapes_root")
    current = candidate
    while current != root:
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise BundleError("bundle_file_missing") from exc
        if stat.S_ISLNK(mode):
            raise BundleError("bundle_symlink_rejected")
        current = current.parent
    try:
        info = resolved.stat()
        if (not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size > max_bytes):
            raise BundleError("bundle_file_permissions_or_size_invalid")
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        try:
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
                    or stat.S_IMODE(opened.st_mode) & 0o077
                    or opened.st_size > max_bytes):
                raise BundleError("bundle_file_changed_or_unsafe")
            chunks = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1_048_576, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise BundleError("bundle_file_too_large")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BundleError("bundle_file_read_failed") from exc


def _load_manifest(root: Path) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError("bundle_manifest_duplicate_key")
            result[key] = value
        return result

    try:
        mode = stat.S_IMODE(root.stat().st_mode)
        if not root.is_dir() or mode & 0o077:
            raise BundleError("bundle_root_permissions_invalid")
        raw = _read_private_file(root, "bundle.json", max_bytes=_MAX_MANIFEST_BYTES)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=lambda _value: (_ for _ in ()).throw(
                               BundleError("bundle_manifest_nonfinite_value")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError("bundle_manifest_json_invalid") from exc
    if (not isinstance(value, dict) or set(value) != _MANIFEST_KEYS
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("purpose") != "qta_phase_receipt_bundle_non_authorizing_v1"):
        raise BundleError("bundle_manifest_schema_invalid")
    for key, phase in (("phase1_artifacts", "pre_baseline"),
                       ("phase2_artifacts", "post_baseline")):
        roles = value.get(key)
        if (not isinstance(roles, dict)
                or set(roles) != set(PHASE_ARTIFACT_ROLES[phase])):
            raise BundleError("bundle_artifact_role_set_invalid")
    return value


def _artifact_map(root: Path, value: Any) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise BundleError("bundle_artifact_map_invalid")
    result = {role: _read_private_file(root, relative, max_bytes=_MAX_FILE_BYTES)
              for role, relative in value.items()}
    if sum(map(len, result.values())) > _MAX_PHASE_ARTIFACT_BYTES:
        raise BundleError("bundle_phase_artifacts_too_large")
    return result


def _load_sidecars(root: Path, relative: Any, index_raw: bytes) -> dict[str, bytes]:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise BundleError("bundle_sidecar_dir_invalid")
    directory = root / relative
    current = root
    for part in relative.split("/"):
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise BundleError("bundle_symlink_rejected")
        except OSError as exc:
            raise BundleError("bundle_sidecar_dir_missing") from exc
    try:
        resolved = directory.resolve(strict=True)
        mode = stat.S_IMODE(directory.lstat().st_mode)
    except OSError as exc:
        raise BundleError("bundle_sidecar_dir_missing") from exc
    if (root not in resolved.parents or not directory.is_dir()
            or stat.S_ISLNK(directory.lstat().st_mode) or mode & 0o077):
        raise BundleError("bundle_sidecar_dir_permissions_invalid")
    try:
        index = json.loads(index_raw.decode("ascii"))
        rows = index["records"]
        digests = {row["attention_bytes_sha256"] for row in rows}
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BundleError("bundle_capture_index_invalid") from exc
    if any(not isinstance(digest, str) or not _SHA256.fullmatch(digest)
           for digest in digests):
        raise BundleError("bundle_sidecar_digest_invalid")
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise BundleError("bundle_sidecar_dir_read_failed") from exc
    if {entry.name for entry in entries} != {f"{digest}.bin" for digest in digests}:
        raise BundleError("bundle_sidecar_file_set_invalid")
    sidecars: dict[str, bytes] = {}
    total = 0
    for digest in digests:
        raw = _read_private_file(root, f"{relative}/{digest}.bin",
                                 max_bytes=_MAX_SIDECAR_BYTES)
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise BundleError("bundle_sidecar_digest_mismatch")
        if total > _MAX_SIDECAR_BYTES:
            raise BundleError("bundle_sidecars_too_large")
        sidecars[digest] = raw
    return sidecars


def verify_bundle(*, bundle_dir: Path,
                  expected_external_cohort_pin_sha256: str,
                  expected_root_public_der_sha256: str) -> dict[str, Any]:
    """Validate exact local bundle bytes while keeping execution unauthorized."""
    if (not isinstance(expected_external_cohort_pin_sha256, str)
            or not _SHA256.fullmatch(expected_external_cohort_pin_sha256)):
        raise BundleError("external_cohort_pin_cli_hash_invalid")
    if (not isinstance(expected_root_public_der_sha256, str)
            or not _SHA256.fullmatch(expected_root_public_der_sha256)):
        raise BundleError("trust_root_cli_hash_invalid")
    try:
        provided_root = Path(os.path.abspath(bundle_dir))
        if stat.S_ISLNK(provided_root.lstat().st_mode):
            raise BundleError("bundle_root_symlink_rejected")
        root = provided_root.resolve(strict=True)
    except OSError as exc:
        raise BundleError("bundle_root_missing") from exc
    manifest = _load_manifest(root)
    p1 = _read_private_file(root, manifest["phase1_receipt"],
                            max_bytes=_MAX_FILE_BYTES)
    p2 = _read_private_file(root, manifest["phase2_receipt"],
                            max_bytes=_MAX_FILE_BYTES)
    p1_artifacts = _artifact_map(root, manifest["phase1_artifacts"])
    p2_artifacts = _artifact_map(root, manifest["phase2_artifacts"])
    if p2_artifacts.get("phase1_receipt") != p1:
        raise BundleError("bundle_phase1_receipt_copy_mismatch")
    sidecars = _load_sidecars(root, manifest["attention_capture_sidecar_dir"],
                              p2_artifacts["attention_capture_index"])
    report = verify_qta_phase2_plan_candidate(
        phase1_receipt_raw=p1,
        phase1_artifact_bytes=p1_artifacts,
        phase2_receipt_raw=p2,
        phase2_artifact_bytes=p2_artifacts,
        attention_capture_sidecars=sidecars,
        integrity_roster_raw=_read_private_file(
            root, manifest["integrity_roster"], max_bytes=_MAX_FILE_BYTES),
        release_roster_raw=_read_private_file(
            root, manifest["release_roster"], max_bytes=_MAX_FILE_BYTES),
        root_public_key_pem=_read_private_file(
            root, manifest["root_public_key"], max_bytes=_MAX_FILE_BYTES),
        expected_root_public_der_sha256=expected_root_public_der_sha256,
        expected_external_cohort_pin_sha256=expected_external_cohort_pin_sha256,
    )
    # Preserve the verifier's non-authorizing contract and expose no row identifiers.
    safe = {
        key: report[key] for key in (
            "status", "validated_plan_rows", "phase1_receipt_signature_valid",
            "phase2_receipt_signature_valid", "external_cohort_pin_matches_caller_pin",
            "external_cohort_pin_origin_authenticated", "phase1_receipt_recursively_verified",
            "forward_source_authenticated", "attention_capture_authenticated",
            "capture_sidecars_joined_to_selection_masses", "capture_sidecar_records_validated",
            "capture_call_receipts_joined_to_sidecar_payloads",
            "capture_call_receipts_validated", "execution_authorization",
            "model_load_permitted", "scoring_authorization", "phase_authorized",
            "authorized_forwards",
        )
    }
    if (safe["execution_authorization"] != "not_granted"
            or safe["forward_source_authenticated"] is not False
            or safe["attention_capture_authenticated"] is not False
            or safe["model_load_permitted"] is not False
            or safe["scoring_authorization"] != "not_granted"
            or safe["phase_authorized"] is not False
            or safe["authorized_forwards"] != 0):
        raise BundleError("verifier_authority_contract_violation")
    safe["expected_external_cohort_pin_sha256"] = expected_external_cohort_pin_sha256
    safe["external_cohort_pin_value_origin_authenticated"] = False
    safe["trust_root_value_origin_authenticated"] = False
    return safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="mode-0700 local directory containing bundle.json and receipts")
    parser.add_argument("--expected-external-cohort-pin-sha256", required=True,
                        help="64 lowercase hex characters obtained outside the bundle")
    parser.add_argument("--expected-trust-root-der-sha256", required=True,
                        help="64 lowercase hex characters obtained outside the bundle")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = verify_bundle(
            bundle_dir=args.bundle_dir,
            expected_external_cohort_pin_sha256=args.expected_external_cohort_pin_sha256,
            expected_root_public_der_sha256=args.expected_trust_root_der_sha256,
        )
    except (BundleError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
