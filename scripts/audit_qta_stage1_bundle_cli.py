#!/usr/bin/env python3
"""Read-only, non-authorizing CLI for one private QTA Stage-1 bundle directory.

This reduces manual path wiring only. All trust pins remain explicit caller
inputs, the existing Stage-1 audit is rerun unchanged, and scoring stays denied.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from qta_stage1_bundle_audit import MAX_PRIVATE_ARTIFACT_BYTES, audit_qta_stage1_bundle


PURPOSE = "qta_stage1_audit_bundle_non_authorizing_v1"
MAX_MANIFEST_BYTES = 64_000
MAX_TOTAL_BYTES = 1_000_000_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ROLES = {
    "worksheet", "lineage", "review", "review_index", "custody",
    "pre_review_receipt", "integrity_roster", "release_roster", "inventory",
    "release_witness", "reviewer_attestation", "root_public_key",
}
MANIFEST_KEYS = {"schema_version", "purpose", "sources", "media_dir", "artifact_sources"}


class BundleError(ValueError):
    """Stable, content-free bundle validation error."""


def _strict_json(raw: bytes) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise BundleError("bundle_duplicate_json_key")
            value[key] = item
        return value

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              BundleError("bundle_nonfinite_json")))
    except BundleError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BundleError("bundle_json_invalid") from exc


def _relative_path(value: Any, code: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or Path(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise BundleError(code)
    return value


def _private_read(root_fd: int, relative: Any, *, max_bytes: int,
                  total_bytes: list[int]) -> bytes:
    relative = _relative_path(relative, "bundle_relative_path_invalid")
    parts = relative.split("/")
    current_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                  | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise BundleError("bundle_symlink_rejected") from exc
                raise BundleError("bundle_directory_open_failed") from exc
            info = os.fstat(next_fd)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                os.close(next_fd)
                raise BundleError("bundle_directory_permissions_invalid")
            os.close(current_fd)
            current_fd = next_fd
        try:
            fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), dir_fd=current_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise BundleError("bundle_symlink_rejected") from exc
            raise BundleError("bundle_file_open_failed") from exc
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_size < 1 or before.st_size > max_bytes):
                raise BundleError("bundle_file_permissions_or_size_invalid")
            if total_bytes[0] + before.st_size > MAX_TOTAL_BYTES:
                raise BundleError("bundle_total_bytes_exceeded")
            # Reserve from the cap before allocating or reading the body.
            total_bytes[0] += before.st_size
            blocks = []
            total = 0
            while True:
                chunk = os.read(fd, min(1_048_576, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes or total > before.st_size:
                    raise BundleError("bundle_file_too_large_or_changed")
                blocks.append(chunk)
            after = os.fstat(fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                     after.st_ctime_ns) or total != before.st_size):
                raise BundleError("bundle_file_changed_during_read")
            return b"".join(blocks)
        finally:
            os.close(fd)
    finally:
        os.close(current_fd)


def _load_bundle_dir(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, bytes], Path,
                                                dict[str, bytes]]:
    supplied = Path(os.path.abspath(bundle_dir))
    root_fd = None
    try:
        root_info = supplied.lstat()
        if stat.S_ISLNK(root_info.st_mode):
            raise BundleError("bundle_root_symlink_rejected")
        root_fd = os.open(supplied, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                          | getattr(os, "O_NOFOLLOW", 0))
        opened_root = os.fstat(root_fd)
        root = supplied.resolve(strict=True)
        root_info = root.stat()
    except BundleError:
        if root_fd is not None:
            os.close(root_fd)
        raise
    except OSError as exc:
        if root_fd is not None:
            os.close(root_fd)
        raise BundleError("bundle_root_missing") from exc
    try:
        if (not stat.S_ISDIR(opened_root.st_mode) or opened_root.st_uid != os.geteuid()
                or stat.S_IMODE(opened_root.st_mode) != 0o700
                or (opened_root.st_dev, opened_root.st_ino) !=
                (root_info.st_dev, root_info.st_ino)):
            raise BundleError("bundle_root_permissions_invalid")
        total_bytes = [0]
        manifest_raw = _private_read(root_fd, "bundle.json", max_bytes=MAX_MANIFEST_BYTES,
                                     total_bytes=total_bytes)
        manifest = _strict_json(manifest_raw)
        if (not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS
                or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 1 or manifest.get("purpose") != PURPOSE
                or not isinstance(manifest.get("sources"), dict)
                or set(manifest["sources"]) != SOURCE_ROLES
                or not isinstance(manifest.get("artifact_sources"), dict)):
            raise BundleError("bundle_manifest_schema_invalid")

        all_paths = ["bundle.json", *manifest["sources"].values(),
                     *manifest["artifact_sources"].values()]
        canonical_paths = [_relative_path(path, "bundle_relative_path_invalid")
                           for path in all_paths]
        if len(set(canonical_paths)) != len(canonical_paths):
            raise BundleError("bundle_path_alias_rejected")

        sources = {name: _private_read(root_fd, path, max_bytes=MAX_PRIVATE_ARTIFACT_BYTES,
                                       total_bytes=total_bytes)
                   for name, path in manifest["sources"].items()}
        inventory = _strict_json(sources["inventory"])
        inventory_rows = inventory.get("artifacts") if isinstance(inventory, dict) else None
        if not isinstance(inventory_rows, list):
            raise BundleError("bundle_inventory_schema_invalid")
        expected_roles = []
        for row in inventory_rows:
            if not isinstance(row, dict) or not isinstance(row.get("role"), str) or not row["role"]:
                raise BundleError("bundle_inventory_role_invalid")
            expected_roles.append(row["role"])
        if len(set(expected_roles)) != len(expected_roles):
            raise BundleError("bundle_inventory_role_duplicate")
        if set(manifest["artifact_sources"]) != set(expected_roles):
            raise BundleError("bundle_artifact_role_set_mismatch")
        artifacts = {role: _private_read(root_fd, path,
                                         max_bytes=MAX_PRIVATE_ARTIFACT_BYTES,
                                         total_bytes=total_bytes)
                     for role, path in manifest["artifact_sources"].items()}

        media_relative = _relative_path(manifest["media_dir"], "bundle_media_dir_invalid")
        if any(path == media_relative or path.startswith(media_relative + "/")
               for path in canonical_paths):
            raise BundleError("bundle_media_dir_path_conflict")
        worksheet = _strict_json(sources["worksheet"])
        rows = worksheet.get("records") if isinstance(worksheet, dict) else None
        tokens = [row.get("review_token") for row in rows
                  if isinstance(row, dict)] if isinstance(rows, list) else []
        if (len(tokens) != 3 or len(set(tokens)) != 3
                or any(not isinstance(token, str)
                       or not re.fullmatch(r"[A-Za-z0-9_-]{24}", token)
                       for token in tokens)):
            raise BundleError("bundle_media_token_set_invalid")
        expected_media = {f"{token}.mp4" for token in tokens}
        media_parts = media_relative.split("/")
        media_fd = os.dup(root_fd)
        try:
            for part in media_parts:
                try:
                    next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                      | getattr(os, "O_NOFOLLOW", 0), dir_fd=media_fd)
                except OSError as exc:
                    if exc.errno == errno.ELOOP:
                        raise BundleError("bundle_symlink_rejected") from exc
                    raise BundleError("bundle_media_dir_open_failed") from exc
                info = os.fstat(next_fd)
                if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                        or stat.S_IMODE(info.st_mode) != 0o700):
                    os.close(next_fd)
                    raise BundleError("bundle_media_dir_permissions_invalid")
                os.close(media_fd)
                media_fd = next_fd
            try:
                if set(os.listdir(media_fd)) != expected_media:
                    raise BundleError("bundle_media_file_set_invalid")
                for name in expected_media:
                    info = os.stat(name, dir_fd=media_fd, follow_symlinks=False)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                            or stat.S_IMODE(info.st_mode) != 0o600
                            or info.st_size < 1 or info.st_size > MAX_PRIVATE_ARTIFACT_BYTES):
                        raise BundleError("bundle_media_file_permissions_or_size_invalid")
                # The downstream audit reopens media by path. Reserve the
                # maximum three-file payload to keep the byte cap valid even
                # if an owner changes media between this preflight and its read.
                media_reservation = 3 * MAX_PRIVATE_ARTIFACT_BYTES
                if total_bytes[0] + media_reservation > MAX_TOTAL_BYTES:
                    raise BundleError("bundle_total_bytes_exceeded")
            except BundleError:
                raise
            except OSError as exc:
                raise BundleError("bundle_media_file_stat_failed") from exc
        finally:
            os.close(media_fd)
        return manifest, sources, root / media_relative, artifacts
    finally:
        if root_fd is not None:
            os.close(root_fd)


def verify_bundle(*, bundle_dir: Path, expected_lineage_sha256: str,
                  expected_custody_sha256: str,
                  expected_root_public_der_sha256: str,
                  expected_ceremony_id: str) -> dict[str, Any]:
    for name, value in (("lineage", expected_lineage_sha256),
                        ("custody", expected_custody_sha256),
                        ("root", expected_root_public_der_sha256)):
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise BundleError(f"expected_{name}_sha256_invalid")
    if (not isinstance(expected_ceremony_id, str) or not expected_ceremony_id
            or len(expected_ceremony_id) > 200 or "\n" in expected_ceremony_id):
        raise BundleError("expected_ceremony_id_invalid")
    _, sources, media_dir, artifacts = _load_bundle_dir(bundle_dir)
    report = audit_qta_stage1_bundle(
        worksheet_raw=sources["worksheet"], lineage_raw=sources["lineage"],
        review_raw=sources["review"], review_index_raw=sources["review_index"],
        custody_raw=sources["custody"],
        pre_review_receipt_raw=sources["pre_review_receipt"],
        integrity_roster_raw=sources["integrity_roster"],
        release_roster_raw=sources["release_roster"], inventory_raw=sources["inventory"],
        release_witness_raw=sources["release_witness"],
        reviewer_attestation_raw=sources["reviewer_attestation"], media_dir=media_dir,
        expected_lineage_sha256=expected_lineage_sha256,
        expected_custody_sha256=expected_custody_sha256,
        root_public_key_pem=sources["root_public_key"],
        expected_root_public_der_sha256=expected_root_public_der_sha256,
        expected_ceremony_id=expected_ceremony_id, artifact_sources=artifacts)
    if (not isinstance(report, dict)
            or report.get("status") != "stage1_structurally_valid_non_authorizing"
            or type(report.get("requested_records")) is not int
            or report.get("requested_records") != 3
            or type(report.get("effective_records")) is not int
            or report.get("effective_records") != 3
            or type(report.get("structurally_valid_records")) is not int
            or report.get("structurally_valid_records") != 3
            or type(report.get("failure_count")) is not int
            or report.get("failure_count") != 0 or report.get("failure_codes") != []
            or report.get("model_load_permitted") is not False
            or report.get("score_ready") is not False
            or report.get("reviewer_identity_verified") is not False
            or report.get("semantic_review_authenticated") is not False
            or report.get("frame_correspondence_verified") is not False
            or report.get("execution_authorization") != "not_granted"):
        raise BundleError("stage1_audit_contract_invalid")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--expected-custody-sha256", required=True)
    parser.add_argument("--expected-root-public-der-sha256", required=True)
    parser.add_argument("--expected-ceremony-id", required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_bundle(
            bundle_dir=args.bundle_dir,
            expected_lineage_sha256=args.expected_lineage_sha256,
            expected_custody_sha256=args.expected_custody_sha256,
            expected_root_public_der_sha256=args.expected_root_public_der_sha256,
            expected_ceremony_id=args.expected_ceremony_id)
    except BundleError as exc:
        report = {"status": "blocked", "failure_codes": [str(exc)],
                  "model_load_permitted": False, "score_ready": False,
                  "execution_authorization": "not_granted"}
    except Exception:
        report = {"status": "blocked", "failure_codes": ["stage1_audit_failed"],
                  "model_load_permitted": False, "score_ready": False,
                  "execution_authorization": "not_granted"}
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("status") == "stage1_structurally_valid_non_authorizing" else 2


if __name__ == "__main__":
    sys.exit(main())
