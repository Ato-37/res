#!/usr/bin/env python3
"""Stage browser review exports as private, byte-identical local copies.

This verifies byte digests and declared JSON purpose/status fields only. It does
not authenticate human review or browser telemetry, validate review semantics,
or authorize scoring.
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
from typing import Any

from private_output_paths import validate_private_output

MAX_INPUT_BYTES = 2 * 1024 * 1024
PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class StageError(ValueError):
    """A sanitized error code safe for CLI output."""


def _fail(code: str) -> None:
    raise StageError(code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _json_object(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda _: _fail("json_constant_invalid"))
    except StageError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _read_private_input(path_text: str) -> bytes:
    path = Path(path_text)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        before = path.lstat()
        if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid() or before.st_size > MAX_INPUT_BYTES):
            _fail("input_file_invalid")
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid()
                or opened.st_size > MAX_INPUT_BYTES
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            _fail("input_file_changed_or_invalid")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, MAX_INPUT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_INPUT_BYTES:
                _fail("input_file_too_large")
        after = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or total != after.st_size):
            _fail("input_file_changed_during_read")
        return b"".join(chunks)
    except StageError:
        raise
    except OSError:
        _fail("input_file_unreadable")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                _fail("input_file_unreadable")


def _validate_exports(review_raw: bytes, review_index_raw: bytes,
                      clock_raw: bytes | None, clock_index_raw: bytes | None) -> list[dict[str, str]]:
    review = _json_object(review_raw, "review_payload_json_invalid")
    review_index = _json_object(review_index_raw, "review_index_json_invalid")
    if (type(review.get("schema_version")) is not int or review["schema_version"] != 3
            or review.get("purpose") != "anonymous_local_claim_video_human_review"
            or review.get("status") != "human_review_recorded_unvalidated"
            or review.get("execution_authorization") != "not_granted"):
        _fail("review_payload_declaration_invalid")
    if (type(review_index.get("schema_version")) is not int
            or review_index["schema_version"] != 1
            or review_index.get("purpose") != "anonymous_unvalidated_human_review_handoff"
            or review_index.get("review_status") != "unvalidated_requires_custodian_audit"
            or review_index.get("execution_authorization") != "not_granted"):
        _fail("review_index_declaration_invalid")
    review_sha = hashlib.sha256(review_raw).hexdigest()
    if review_index.get("review_payload_sha256") != review_sha:
        _fail("review_payload_hash_mismatch")

    entries = [
        {"role": "review_payload", "sha256": hashlib.sha256(review_raw).hexdigest(),
         "status": "staged_unvalidated"},
        {"role": "review_index", "sha256": hashlib.sha256(review_index_raw).hexdigest(),
         "status": "staged_unvalidated"},
    ]
    if clock_raw is None:
        return entries
    clock = _json_object(clock_raw, "clock_observations_json_invalid")
    clock_index = _json_object(clock_index_raw, "clock_index_json_invalid")
    if (type(clock.get("schema_version")) is not int or clock["schema_version"] != 2
            or clock.get("purpose") != "local_unvalidated_html5_player_clock_observations"
            or clock.get("status") != "raw_player_clock_observations_only"
            or clock.get("execution_authorization") != "not_granted"):
        _fail("clock_observations_declaration_invalid")
    if (type(clock_index.get("schema_version")) is not int
            or clock_index["schema_version"] != 1
            or clock_index.get("purpose") != "unvalidated_html5_clock_observation_integrity_index"
            or clock_index.get("status") != "raw_observations_only"
            or clock_index.get("execution_authorization") != "not_granted"):
        _fail("clock_index_declaration_invalid")
    if clock_index.get("payload_sha256") != hashlib.sha256(clock_raw).hexdigest():
        _fail("clock_observations_hash_mismatch")
    entries.extend([
        {"role": "clock_observations", "sha256": hashlib.sha256(clock_raw).hexdigest(),
         "status": "staged_unvalidated"},
        {"role": "clock_index", "sha256": hashlib.sha256(clock_index_raw).hexdigest(),
         "status": "staged_unvalidated"},
    ])
    return entries


def _write_exclusive(path: Path, raw: bytes, created: list[Path]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags, 0o600)
        created.append(path)
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                _fail("output_write_failed")
            offset += written
        os.fsync(fd)
    except StageError:
        raise
    except FileExistsError:
        _fail("output_already_exists")
    except OSError:
        _fail("output_write_failed")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                _fail("output_write_failed")


def stage_exports(*, review_payload_path: str, review_index_path: str,
                  output_prefix: str, clock_observations_path: str | None = None,
                  clock_index_path: str | None = None,
                  repo_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(output_prefix, str) or PREFIX.fullmatch(output_prefix) is None:
        _fail("output_prefix_invalid")
    if (clock_observations_path is None) != (clock_index_path is None):
        _fail("clock_export_pair_incomplete")
    repo = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)

    review_raw = _read_private_input(review_payload_path)
    review_index_raw = _read_private_input(review_index_path)
    clock_raw = (_read_private_input(clock_observations_path)
                 if clock_observations_path is not None else None)
    clock_index_raw = (_read_private_input(clock_index_path)
                       if clock_index_path is not None else None)
    entries = _validate_exports(review_raw, review_index_raw, clock_raw, clock_index_raw)

    payloads = {
        "review_payload": review_raw,
        "review_index": review_index_raw,
    }
    if clock_raw is not None:
        payloads["clock_observations"] = clock_raw
        payloads["clock_index"] = clock_index_raw
    for entry in entries:
        entry["basename"] = f"{output_prefix}_{entry['role']}.json"
    manifest_name = f"{output_prefix}_manifest.json"
    try:
        output_paths = {
            entry["role"]: validate_private_output(
                str(repo / "data" / "private_nextgqa" / entry["basename"]), repo)
            for entry in entries
        }
        manifest_path = validate_private_output(
            str(repo / "data" / "private_nextgqa" / manifest_name), repo)
    except FileExistsError:
        _fail("output_already_exists")
    except (OSError, ValueError):
        _fail("output_path_invalid")
    manifest = {
        "schema_version": 1,
        "purpose": "private_staged_nextgqa_review_exports_unvalidated",
        "status": "staged_unvalidated",
        "artifacts": [{key: entry[key] for key in ("role", "basename", "sha256", "status")}
                      for entry in entries],
        "provenance_status": "bytes_and_declared_fields_only_unauthenticated",
        "execution_authorization": "not_granted",
    }
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    created: list[Path] = []
    completed = False
    try:
        for role, raw in payloads.items():
            path = output_paths[role]
            _write_exclusive(path, raw, created)
        _write_exclusive(manifest_path, manifest_raw, created)
        result = {
            "status": "staged_unvalidated",
            "artifact_count": len(entries),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "output_mode": "0600",
            "provenance_status": manifest["provenance_status"],
            "execution_authorization": "not_granted",
        }
        completed = True
        return result
    finally:
        if not completed:
            for path in reversed(created):
                try:
                    path.unlink()
                except OSError:
                    pass


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-payload", required=True)
    parser.add_argument("--review-index", required=True)
    parser.add_argument("--clock-observations")
    parser.add_argument("--clock-index")
    parser.add_argument("--output-prefix", required=True,
                        help="safe basename prefix for private staged files")
    args = parser.parse_args(argv)
    try:
        result = stage_exports(
            review_payload_path=args.review_payload,
            review_index_path=args.review_index,
            clock_observations_path=args.clock_observations,
            clock_index_path=args.clock_index,
            output_prefix=args.output_prefix,
            repo_root=repo_root,
        )
    except StageError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
