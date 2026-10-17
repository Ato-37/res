"""Build and verify a non-authorizing, signed pre-review integrity receipt.

This module verifies an externally pinned Ed25519 trust root, its signed
three-role roster, a freeze-owner signature, and the exact bytes described by
the existing pre-review candidate inventory. A successful result is only
``signature_valid_order_unverified``. It never permits opening the package,
model loading, or scoring: a trusted/witnessed release-order statement has not
been defined. ``created_at_utc`` is a local-clock assertion only.

The API accepts bytes for synthetic tests and paths for local operation. It
does not create keys or identities. Keep real receipts and artifacts local.
"""
from __future__ import annotations

import base64
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

from build_nextgqa_pre_review_byte_inventory import (
    CODE_FILES,
    PRIVATE_FILES,
    PUBLIC_OR_SOURCE_FILES,
)


ROLES = ("freeze_owner", "claim_reviewer", "frame_reviewer")
PACKAGE_ROLES = {"review_package_html"}
PACKAGE_ROLES |= {f"review_package_media_{index:02d}" for index in range(1, 4)}
PACKAGE_ROLES |= {f"staged_media_{index:02d}" for index in range(1, 4)}
REQUIRED_ARTIFACT_ROLES = (
    set(PRIVATE_FILES) | set(PUBLIC_OR_SOURCE_FILES) | set(CODE_FILES) | PACKAGE_ROLES
)
PRIVATE_ARTIFACT_ROLES = set(PRIVATE_FILES) | PACKAGE_ROLES
_SHA = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MAX_JSON = 8 * 1024 * 1024
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")
_ROSTER_DOMAIN = b"nextgqa_pre_review_roster_v1\0"
_RECEIPT_DOMAIN = b"nextgqa_pre_review_integrity_receipt_v1\0"


class ReceiptError(ValueError):
    """Stable, content-free failure from the receipt boundary."""


def _fail(code: str) -> None:
    raise ReceiptError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("json_not_canonicalizable")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _parse(raw: bytes, code: str) -> Any:
    if not isinstance(raw, bytes) or len(raw) > _MAX_JSON:
        _fail(code)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: _fail("json_nonfinite_value"))
    except ReceiptError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, RecursionError):
        _fail(code)


def _utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (parsed.tzinfo is not None and parsed.utcoffset() is not None
            and parsed.utcoffset() == timezone.utc.utcoffset(parsed))


def _memfd(name: str, raw: bytes) -> int:
    if not hasattr(os, "memfd_create"):
        _fail("anonymous_fd_unavailable")
    fd = None
    try:
        fd = os.memfd_create(name, 0)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short_write")
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except OSError:
        if fd is not None:
            os.close(fd)
        _fail("openssl_input_fd_failed")


def _public_der(public_pem: bytes) -> bytes:
    if not isinstance(public_pem, bytes) or not public_pem or len(public_pem) > 64_000:
        _fail("public_key_invalid")
    fd = _memfd("nextgqa-ed25519-public", public_pem)
    try:
        try:
            result = subprocess.run(
                ["openssl", "pkey", "-pubin", "-in", f"/proc/self/fd/{fd}",
                 "-outform", "DER"], pass_fds=(fd,), capture_output=True,
                timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            _fail("openssl_public_key_parse_failed")
    finally:
        os.close(fd)
    der = result.stdout
    if (result.returncode != 0 or len(der) != 44
            or not der.startswith(_ED25519_SPKI_PREFIX)):
        _fail("public_key_not_ed25519")
    return der


def _verify_signature(public_pem: bytes, signature: bytes, payload: bytes) -> bool:
    key_fd = _memfd("nextgqa-ed25519-verify-key", public_pem)
    payload_fd = _memfd("nextgqa-ed25519-verify-payload", payload)
    signature_fd = _memfd("nextgqa-ed25519-signature", signature)
    try:
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 f"/proc/self/fd/{key_fd}", "-rawin", "-in",
                 f"/proc/self/fd/{payload_fd}", "-sigfile",
                 f"/proc/self/fd/{signature_fd}"],
                pass_fds=(key_fd, payload_fd, signature_fd), capture_output=True,
                timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            _fail("openssl_signature_verify_failed")
    finally:
        os.close(key_fd)
        os.close(payload_fd)
        os.close(signature_fd)
    return result.returncode == 0


def _sign(private_pem: bytes, public_pem: bytes, payload: bytes) -> bytes:
    private_fd = _memfd("nextgqa-ed25519-private", private_pem)
    payload_fd = _memfd("nextgqa-ed25519-sign-payload", payload)
    try:
        try:
            derived = subprocess.run(
                ["openssl", "pkey", "-in", f"/proc/self/fd/{private_fd}",
                 "-passin", "pass:", "-pubout", "-outform", "DER"],
                pass_fds=(private_fd,), capture_output=True, timeout=10, check=False)
            expected = _public_der(public_pem)
            if derived.returncode != 0 or derived.stdout != expected:
                _fail("signing_key_does_not_match_roster")
            signed = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey",
                 f"/proc/self/fd/{private_fd}", "-passin", "pass:", "-rawin",
                 "-in", f"/proc/self/fd/{payload_fd}"],
                pass_fds=(private_fd, payload_fd), capture_output=True,
                timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            _fail("openssl_signature_create_failed")
    finally:
        os.close(private_fd)
        os.close(payload_fd)
    if signed.returncode != 0 or len(signed.stdout) != 64:
        _fail("openssl_signature_create_failed")
    return signed.stdout


def _decode_signature(value: object, code: str) -> bytes:
    if not isinstance(value, str):
        _fail(code)
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        _fail(code)
    if len(raw) != 64:
        _fail(code)
    return raw


def _validate_roster(roster_raw: bytes, root_public_pem: bytes,
                     expected_root_public_der_sha256: str) -> tuple[dict, dict[str, bytes]]:
    root_der = _public_der(root_public_pem)
    if (not isinstance(expected_root_public_der_sha256, str)
            or not _SHA.fullmatch(expected_root_public_der_sha256)
            or _sha(root_der) != expected_root_public_der_sha256):
        _fail("external_trust_root_pin_mismatch")
    roster = _parse(roster_raw, "roster_json_invalid")
    keys = {"schema_version", "purpose", "roster_id", "keys", "signature_ed25519_base64"}
    if (not isinstance(roster, dict) or set(roster) != keys
            or type(roster.get("schema_version")) is not int
            or roster.get("schema_version") != 1
            or roster.get("purpose") != "nextgqa_pre_review_trust_roster_v1"
            or not isinstance(roster.get("roster_id"), str)
            or not _KEY_ID.fullmatch(roster["roster_id"])):
        _fail("roster_schema_invalid")
    signature = _decode_signature(roster["signature_ed25519_base64"], "roster_signature_invalid")
    body = {key: value for key, value in roster.items() if key != "signature_ed25519_base64"}
    if not _verify_signature(root_public_pem, signature,
                             _ROSTER_DOMAIN + _canonical(body)):
        _fail("root_signed_roster_signature_invalid")
    entries = roster.get("keys")
    if not isinstance(entries, list) or len(entries) != len(ROLES):
        _fail("roster_role_set_invalid")
    public_by_role: dict[str, bytes] = {}
    seen_ids = set()
    seen_key_fingerprints = set()
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {"role", "key_id", "public_key_pem_base64", "public_key_der_sha256"}):
            _fail("roster_key_entry_invalid")
        role, key_id = entry.get("role"), entry.get("key_id")
        if (role not in ROLES or role in public_by_role
                or not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id)
                or key_id in seen_ids):
            _fail("roster_role_set_invalid")
        try:
            public_pem = base64.b64decode(entry["public_key_pem_base64"], validate=True)
        except (ValueError, TypeError):
            _fail("roster_public_key_encoding_invalid")
        der = _public_der(public_pem)
        if entry.get("public_key_der_sha256") != _sha(der):
            _fail("roster_public_key_hash_mismatch")
        if _sha(der) in seen_key_fingerprints:
            _fail("roster_role_keys_must_be_distinct")
        public_by_role[role] = public_pem
        seen_ids.add(key_id)
        seen_key_fingerprints.add(_sha(der))
    if set(public_by_role) != set(ROLES):
        _fail("roster_role_set_invalid")
    return roster, public_by_role


def _inventory_expected_roles() -> set[str]:
    return set(REQUIRED_ARTIFACT_ROLES)


def _validate_inventory(inventory_raw: bytes) -> tuple[dict, list[dict[str, Any]]]:
    inventory = _parse(inventory_raw, "inventory_json_invalid")
    expected_keys = {
        "schema_version", "purpose", "status", "review_started", "review_completed",
        "joint_freeze", "score_ready", "execution_authorization", "rights_status",
        "artifacts", "missing_required_slots",
    }
    if (not isinstance(inventory, dict) or set(inventory) != expected_keys
            or type(inventory.get("schema_version")) is not int
            or inventory.get("schema_version") != 1
            or inventory.get("purpose") != "pre_review_candidate_byte_inventory"
            or inventory.get("status") != "candidate_only"
            or inventory.get("review_started") is not False
            or inventory.get("review_completed") is not False
            or inventory.get("joint_freeze") is not False
            or inventory.get("score_ready") is not False
            or inventory.get("execution_authorization") != "not_granted"
            or inventory.get("rights_status") != "unverified_local_only"
            or inventory.get("missing_required_slots") != []):
        _fail("inventory_not_complete_candidate")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("inventory_artifacts_invalid")
    by_role = {}
    for item in artifacts:
        if (not isinstance(item, dict) or set(item) != {"role", "size_bytes", "sha256"}
                or not isinstance(item.get("role"), str)
                or type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not _SHA.fullmatch(item["sha256"])):
            _fail("inventory_artifact_entry_invalid")
        role = item["role"]
        if role in by_role:
            _fail("inventory_artifact_role_duplicate")
        by_role[role] = item
    if set(by_role) != _inventory_expected_roles():
        _fail("inventory_artifact_role_set_mismatch")
    return inventory, sorted(by_role.values(), key=lambda item: item["role"])


def _hash_source(value: bytes | str | Path, role: str) -> tuple[int, str]:
    if isinstance(value, bytes):
        return len(value), _sha(value)
    if not isinstance(value, (str, Path)):
        _fail("artifact_source_type_invalid")
    path = Path(value)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError):
        _fail("artifact_open_failed")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
            _fail("artifact_file_invalid")
        if role in PRIVATE_ARTIFACT_ROLES and stat.S_IMODE(before.st_mode) != 0o600:
            _fail("private_artifact_mode_invalid")
        digest = hashlib.sha256()
        size = 0
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except OSError:
                _fail("artifact_read_failed")
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (size != before.st_size or after.st_size != before.st_size
                or after.st_ino != before.st_ino or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            _fail("artifact_changed_during_hash")
        return size, digest.hexdigest()
    finally:
        os.close(fd)


def _validate_artifact_sources(bindings: list[dict[str, Any]],
                               artifact_sources: Mapping[str, bytes | str | Path]) -> None:
    expected = {item["role"] for item in bindings}
    if not isinstance(artifact_sources, Mapping) or set(artifact_sources) != expected:
        _fail("artifact_source_role_set_mismatch")
    for item in bindings:
        size, digest = _hash_source(artifact_sources[item["role"]], item["role"])
        if size != item["size_bytes"] or digest != item["sha256"]:
            _fail("artifact_bytes_hash_mismatch")


def _receipt_body(*, roster_raw: bytes, inventory_raw: bytes,
                  bindings: list[dict[str, Any]], owner_key_id: str,
                  owner_public_key_pem: bytes, created_at_utc: str) -> dict[str, Any]:
    if not _utc(created_at_utc):
        _fail("local_timestamp_invalid")
    if not isinstance(owner_key_id, str) or not _KEY_ID.fullmatch(owner_key_id):
        _fail("freeze_owner_key_id_invalid")
    return {
        "schema_version": 1,
        "purpose": "nextgqa_pre_review_handoff_integrity_only_v1",
        "status": "integrity_receipt_candidate",
        "scope": "review_handoff_integrity_only",
        "protocol_v2_status": "design_draft_not_frozen",
        "roster_sha256": _sha(roster_raw),
        "inventory_sha256": _sha(inventory_raw),
        "artifact_bindings": bindings,
        "signer_role": "freeze_owner",
        "signer_key_id": owner_key_id,
        "signer_public_key_der_sha256": _sha(_public_der(owner_public_key_pem)),
        "created_at_utc": created_at_utc,
        "timestamp_semantics": "local_assertion_only_non_authoritative",
        "chronology_status": "unverified_no_release_order_evidence",
        "freeze_owner_attests_review_not_started": True,
        "review_state_semantics": "signed_assertion_only",
        "review_completed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
        "package_opening_permitted": False,
        "unresolved_gate": "trusted_or_witnessed_release_order_evidence_absent",
    }


def build_integrity_receipt(*, roster_raw: bytes, inventory_raw: bytes,
                            artifact_sources: Mapping[str, bytes | str | Path],
                            root_public_key_pem: bytes,
                            expected_root_public_der_sha256: str,
                            owner_key_id: str, owner_private_key_pem: bytes,
                            owner_public_key_pem: bytes,
                            created_at_utc: str) -> bytes:
    """Build a signed integrity-only receipt; never grants package release."""
    _, role_keys = _validate_roster(roster_raw, root_public_key_pem,
                                    expected_root_public_der_sha256)
    owner_pem = role_keys["freeze_owner"]
    if owner_key_id not in {  # key ID is role-bound by the root-signed roster
            entry["key_id"] for entry in _parse(roster_raw, "roster_json_invalid")["keys"]
            if entry["role"] == "freeze_owner"}:
        _fail("freeze_owner_key_id_not_rostered")
    if _public_der(owner_pem) != _public_der(owner_public_key_pem):
        _fail("freeze_owner_public_key_mismatch")
    _, bindings = _validate_inventory(inventory_raw)
    _validate_artifact_sources(bindings, artifact_sources)
    body = _receipt_body(roster_raw=roster_raw, inventory_raw=inventory_raw,
                         bindings=bindings, owner_key_id=owner_key_id,
                         owner_public_key_pem=owner_pem,
                         created_at_utc=created_at_utc)
    signature = _sign(owner_private_key_pem, owner_pem,
                      _RECEIPT_DOMAIN + _canonical(body))
    return _canonical({**body, "signature_ed25519_base64": base64.b64encode(signature).decode("ascii")}) + b"\n"


def verify_integrity_receipt(*, receipt_raw: bytes, roster_raw: bytes,
                             inventory_raw: bytes,
                             artifact_sources: Mapping[str, bytes | str | Path],
                             root_public_key_pem: bytes,
                             expected_root_public_der_sha256: str) -> dict[str, Any]:
    """Verify all byte/signature bindings; result is explicitly non-authorizing."""
    roster, role_keys = _validate_roster(roster_raw, root_public_key_pem,
                                         expected_root_public_der_sha256)
    _, expected_bindings = _validate_inventory(inventory_raw)
    receipt = _parse(receipt_raw, "receipt_json_invalid")
    signature_key = "signature_ed25519_base64"
    expected_keys = set(_receipt_body(
        roster_raw=roster_raw, inventory_raw=inventory_raw,
        bindings=expected_bindings, owner_key_id="owner", owner_public_key_pem=role_keys["freeze_owner"],
        created_at_utc="2000-01-01T00:00:00Z").keys()) | {signature_key}
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        _fail("receipt_schema_invalid")
    body = {key: value for key, value in receipt.items() if key != signature_key}
    if body.get("roster_sha256") != _sha(roster_raw):
        _fail("receipt_roster_hash_mismatch")
    if body.get("inventory_sha256") != _sha(inventory_raw):
        _fail("receipt_inventory_hash_mismatch")
    if body.get("artifact_bindings") != expected_bindings:
        _fail("receipt_artifact_bindings_mismatch")
    by_role = {entry["role"]: entry for entry in roster["keys"]}
    owner_entry = by_role["freeze_owner"]
    if (body.get("signer_role") != "freeze_owner"
            or body.get("signer_key_id") != owner_entry["key_id"]
            or body.get("signer_public_key_der_sha256") != owner_entry["public_key_der_sha256"]):
        _fail("receipt_signer_not_rostered_freeze_owner")
    # All safety status fields are exact constants. In particular a local
    # timestamp can never stand in for chronology or package-release evidence.
    fixed_fields = {
        "schema_version": 1,
        "purpose": "nextgqa_pre_review_handoff_integrity_only_v1",
        "status": "integrity_receipt_candidate",
        "scope": "review_handoff_integrity_only",
        "protocol_v2_status": "design_draft_not_frozen",
        "timestamp_semantics": "local_assertion_only_non_authoritative",
        "chronology_status": "unverified_no_release_order_evidence",
        "freeze_owner_attests_review_not_started": True,
        "review_state_semantics": "signed_assertion_only",
        "review_completed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
        "package_opening_permitted": False,
        "unresolved_gate": "trusted_or_witnessed_release_order_evidence_absent",
    }
    if type(body.get("schema_version")) is not int:
        _fail("receipt_safety_status_invalid")
    for key, value in fixed_fields.items():
        if body.get(key) != value or (type(value) is bool and type(body.get(key)) is not bool):
            _fail("receipt_safety_status_invalid")
    if not _utc(body.get("created_at_utc")):
        _fail("local_timestamp_invalid")
    signature = _decode_signature(receipt.get(signature_key), "receipt_signature_invalid")
    if not _verify_signature(role_keys["freeze_owner"], signature,
                             _RECEIPT_DOMAIN + _canonical(body)):
        _fail("freeze_owner_receipt_signature_invalid")
    _validate_artifact_sources(expected_bindings, artifact_sources)
    return {
        "status": "signature_valid_order_unverified",
        "scope": "review_handoff_integrity_only",
        "verified_artifact_count": len(expected_bindings),
        "trust_root_pin_matched": True,
        "root_signed_role_roster_valid": True,
        "freeze_owner_signature_valid": True,
        "human_identity_verified": False,
        "reviewer_independence_verified": False,
        "freeze_owner_review_state_attestation_only": True,
        "local_timestamp_authoritative": False,
        "chronology_verified": False,
        "package_opening_permitted": False,
        "model_load_permitted": False,
        "score_ready": False,
        "execution_authorization": "not_granted",
        "unresolved_gate": "trusted_or_witnessed_release_order_evidence_absent",
    }


def _read_file(path: str | Path, *, private: bool, max_bytes: int | None = None,
               failure_code: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError):
        _fail(failure_code)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            _fail(failure_code)
        if private and stat.S_IMODE(info.st_mode) != 0o600:
            _fail(failure_code)
        if max_bytes is not None and info.st_size > max_bytes:
            _fail(failure_code)
        chunks = []
        size = 0
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except OSError:
                _fail(failure_code)
            if not chunk:
                break
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                _fail(failure_code)
            chunks.append(chunk)
        after = os.fstat(fd)
        if (size != info.st_size or after.st_size != info.st_size
                or after.st_ino != info.st_ino or after.st_mtime_ns != info.st_mtime_ns
                or after.st_ctime_ns != info.st_ctime_ns):
            _fail(failure_code)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _open_private_directory(path: Path) -> int:
    """Open a directory without following any path component symlinks."""
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(absolute.anchor, flags)
    except (OSError, ValueError):
        _fail("output_parent_invalid")
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                _fail("output_parent_invalid")
            try:
                next_fd = os.open(component, flags, dir_fd=fd)
            except (OSError, ValueError):
                _fail("output_parent_invalid")
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            _fail("output_parent_must_be_private_directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _write_private_output(path: Path, raw: bytes) -> None:
    parent_fd = _open_private_directory(path.parent)
    try:
        name = path.name
        if name in {"", ".", ".."} or "/" in name:
            _fail("output_path_invalid")
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        except FileExistsError:
            _fail("output_already_exists")
        except (OSError, ValueError):
            _fail("output_create_failed")
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    _fail("output_write_failed")
                view = view[written:]
            os.fsync(fd)
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                _fail("output_mode_invalid")
        except BaseException:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        os.close(fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _parse_artifact_arguments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            _fail("artifact_argument_invalid")
        role, path = value.split("=", 1)
        if role not in REQUIRED_ARTIFACT_ROLES or not path or role in result:
            _fail("artifact_argument_role_invalid_or_duplicate")
        result[role] = Path(path)
    if set(result) != REQUIRED_ARTIFACT_ROLES:
        _fail("artifact_argument_role_set_mismatch")
    return result


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "") + "\nA successful check never permits package opening.")
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--root-public-key", required=True, type=Path,
                        help="externally pinned Ed25519 root key PEM (0600)")
    parser.add_argument("--root-public-der-sha256", required=True,
                        help="independently obtained SHA-256 pin of root SPKI DER")
    parser.add_argument("--roster", required=True, type=Path,
                        help="root-signed role roster JSON (0600)")
    parser.add_argument("--inventory", required=True, type=Path,
                        help="candidate byte inventory JSON (0600)")
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH",
                        help="one exact artifact file; repeat for every required role")
    parser.add_argument("--receipt", type=Path,
                        help="signed receipt JSON for verify mode (0600)")
    parser.add_argument("--owner-private-key", type=Path,
                        help="freeze-owner Ed25519 private key for build mode (0600)")
    parser.add_argument("--created-at-utc",
                        help="local clock assertion; never treated as trusted chronology")
    parser.add_argument("--output", type=Path,
                        help="new receipt path under an existing private mode-0700 directory")
    args = parser.parse_args(argv)
    try:
        root_public = _read_file(args.root_public_key, private=True,
                                 max_bytes=64_000, failure_code="trust_root_key_file_invalid")
        roster_raw = _read_file(args.roster, private=True, max_bytes=_MAX_JSON,
                                failure_code="roster_file_invalid")
        inventory_raw = _read_file(args.inventory, private=True, max_bytes=_MAX_JSON,
                                   failure_code="inventory_file_invalid")
        artifact_paths = _parse_artifact_arguments(args.artifact)
        if args.mode == "build":
            if args.receipt is not None or args.owner_private_key is None or args.output is None or args.created_at_utc is None:
                _fail("build_arguments_invalid")
            owner_private = _read_file(args.owner_private_key, private=True,
                                       max_bytes=64_000, failure_code="owner_private_key_file_invalid")
            roster, role_keys = _validate_roster(
                roster_raw, root_public, args.root_public_der_sha256)
            owner_entry = next(item for item in roster["keys"] if item["role"] == "freeze_owner")
            receipt_raw = build_integrity_receipt(
                roster_raw=roster_raw, inventory_raw=inventory_raw,
                artifact_sources=artifact_paths, root_public_key_pem=root_public,
                expected_root_public_der_sha256=args.root_public_der_sha256,
                owner_key_id=owner_entry["key_id"], owner_private_key_pem=owner_private,
                owner_public_key_pem=role_keys["freeze_owner"],
                created_at_utc=args.created_at_utc)
            _write_private_output(args.output, receipt_raw)
            print(json.dumps({
                "status": "integrity_receipt_created_candidate_only",
                "verified_artifact_count": len(REQUIRED_ARTIFACT_ROLES),
                "package_opening_permitted": False,
                "model_load_permitted": False,
                "score_ready": False,
                "execution_authorization": "not_granted",
                "chronology_verified": False,
                "output_mode": "0600",
            }, sort_keys=True))
            return 0
        if args.receipt is None or args.output is not None or args.owner_private_key is not None or args.created_at_utc is not None:
            _fail("verify_arguments_invalid")
        receipt_raw = _read_file(args.receipt, private=True, max_bytes=_MAX_JSON,
                                 failure_code="receipt_file_invalid")
        result = verify_integrity_receipt(
            receipt_raw=receipt_raw, roster_raw=roster_raw,
            inventory_raw=inventory_raw, artifact_sources=artifact_paths,
            root_public_key_pem=root_public,
            expected_root_public_der_sha256=args.root_public_der_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0
    except ReceiptError as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc),
                          "package_opening_permitted": False,
                          "model_load_permitted": False,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "blocked", "failure_code": "internal_error",
                          "package_opening_permitted": False,
                          "model_load_permitted": False,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Successful verification is never a release gate."""
    return _cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
