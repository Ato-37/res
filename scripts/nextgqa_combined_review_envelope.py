#!/usr/bin/env python3
"""Strict byte-preserving envelope for multiple unvalidated NExT-GQA reviews.

This prototype packages existing review/index documents without changing their
bytes. It validates declared schemas and hashes only; it does not authenticate
reviewers, validate semantic judgments, or authorize scoring.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import unicodedata
from typing import Any, Mapping, Sequence

from stage_nextgqa_review_exports import StageError, _validate_exports

SCHEMA_VERSION = 1
PURPOSE = "combined_nextgqa_formal_review_handoff_envelope"
STATUS = "unvalidated_combined_review_handoff"
AUTHORIZATION = "not_granted"
ROOT_KEYS = {"schema_version", "purpose", "status", "execution_authorization", "sessions"}
SESSION_KEYS = {"sequence", "members"}
MEMBER_KEYS = {"role", "size_bytes", "sha256", "content_base64"}
REQUIRED_ROLES = ("review_payload", "review_index")
OPTIONAL_ROLES = ("clock_observations", "clock_index")
ALL_ROLES = set(REQUIRED_ROLES + OPTIONAL_ROLES)
ROLE_FILENAMES = {
    "review_payload": "review_payload.json",
    "review_index": "review_index.json",
    "clock_observations": "clock_observations.json",
    "clock_index": "clock_index.json",
}
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_SESSIONS = 16
MAX_ENVELOPE_BYTES = 64 * 1024 * 1024
MAX_REVIEWER_ID_LENGTH = 256
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class EnvelopeError(ValueError):
    """Sanitized validation or local-output failure."""


def _fail(code: str) -> None:
    raise EnvelopeError(code)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _strict_json(raw: bytes, code: str) -> Any:
    def finite_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            _fail("nonfinite_json_value")
        return value

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates,
            parse_float=finite_float,
            parse_constant=lambda _value: _fail("nonfinite_json_value"),
        )
    except EnvelopeError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail(code)


def _reviewer_id(review_raw: bytes) -> str:
    review = _strict_json(review_raw, "review_payload_json_invalid")
    if not isinstance(review, dict):
        _fail("review_payload_json_invalid")
    reviewer_id = review.get("human_reviewer_private_id")
    if (not isinstance(reviewer_id, str) or not reviewer_id.strip()
            or reviewer_id != reviewer_id.strip()
            or len(reviewer_id) > MAX_REVIEWER_ID_LENGTH
            or any(ord(char) < 0x20 or ord(char) == 0x7f for char in reviewer_id)):
        _fail("reviewer_private_id_invalid")
    return unicodedata.normalize("NFKC", reviewer_id).casefold()


def _validate_session_members(members: Mapping[str, bytes]) -> str:
    if not isinstance(members, Mapping):
        _fail("session_members_invalid")
    roles = set(members)
    if roles not in (set(REQUIRED_ROLES), set(REQUIRED_ROLES + OPTIONAL_ROLES)):
        _fail("session_member_role_set_invalid")
    if any(not isinstance(raw, bytes) or not raw or len(raw) > MAX_MEMBER_BYTES
           for raw in members.values()):
        _fail("member_bytes_invalid_or_too_large")
    try:
        _validate_exports(
            members["review_payload"], members["review_index"],
            members.get("clock_observations"), members.get("clock_index"),
        )
    except (StageError, ValueError) as exc:
        code = str(exc).split(":", 1)[0]
        if not re.fullmatch(r"[a-z0-9_]+", code):
            code = "inner_export_validation_failed"
        _fail("inner_" + code)
    return _reviewer_id(members["review_payload"])


def _encode_member(role: str, raw: bytes) -> dict[str, Any]:
    return {
        "role": role,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_base64": base64.b64encode(raw).decode("ascii"),
    }


def build_envelope(sessions: Sequence[Mapping[str, bytes]]) -> bytes:
    """Build canonical envelope bytes from 2+ reviewer sessions.

    Each session maps the exact existing role names to their original bytes.
    Session order is the supplied order and is frozen as sequence 1..N.
    """
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        _fail("sessions_invalid")
    if not 1 <= len(sessions) <= MAX_SESSIONS:
        _fail("session_count_out_of_range")
    seen_reviewers: set[str] = set()
    prepared_sessions = []
    for sequence, members in enumerate(sessions, start=1):
        reviewer_id = _validate_session_members(members)
        if reviewer_id in seen_reviewers:
            _fail("duplicate_reviewer_private_id")
        seen_reviewers.add(reviewer_id)
        roles = REQUIRED_ROLES if set(members) == set(REQUIRED_ROLES) else REQUIRED_ROLES + OPTIONAL_ROLES
        prepared_sessions.append({
            "sequence": sequence,
            "members": [
                {"role": role, "size_bytes": len(members[role]),
                 "sha256": hashlib.sha256(members[role]).hexdigest(), "content_base64": ""}
                for role in roles
            ],
        })

    # Bound the aggregate before allocating Base64 copies of any member bytes.
    skeleton = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "status": STATUS,
        "execution_authorization": AUTHORIZATION,
        "sessions": prepared_sessions,
    }
    skeleton_length = len((json.dumps(
        skeleton, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False) + "\n").encode("ascii"))
    encoded_total = sum(4 * ((len(members[role]) + 2) // 3)
                        for members in sessions for role in members)
    if skeleton_length + encoded_total > MAX_ENVELOPE_BYTES:
        _fail("envelope_too_large")
    for session_doc, members in zip(prepared_sessions, sessions, strict=True):
        for member_doc in session_doc["members"]:
            role = member_doc["role"]
            member_doc["content_base64"] = base64.b64encode(members[role]).decode("ascii")
    return (json.dumps(skeleton, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def parse_envelope(raw: bytes) -> list[dict[str, bytes]]:
    """Validate an envelope and return exact inner member bytes by session."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_ENVELOPE_BYTES:
        _fail("envelope_bytes_invalid_or_too_large")
    document = _strict_json(raw, "envelope_json_invalid")
    if (not isinstance(document, dict) or set(document) != ROOT_KEYS
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != SCHEMA_VERSION
            or document.get("purpose") != PURPOSE
            or document.get("status") != STATUS
            or document.get("execution_authorization") != AUTHORIZATION):
        _fail("envelope_root_schema_or_status_invalid")
    sessions = document.get("sessions")
    if not isinstance(sessions, list) or not 1 <= len(sessions) <= MAX_SESSIONS:
        _fail("session_count_out_of_range")

    result: list[dict[str, bytes]] = []
    reviewer_ids: set[str] = set()
    for expected_sequence, session in enumerate(sessions, start=1):
        if (not isinstance(session, dict) or set(session) != SESSION_KEYS
                or type(session.get("sequence")) is not int
                or session["sequence"] != expected_sequence):
            _fail("session_schema_or_sequence_invalid")
        members = session.get("members")
        if not isinstance(members, list) or len(members) not in (2, 4):
            _fail("session_member_count_invalid")
        roles = REQUIRED_ROLES if len(members) == 2 else REQUIRED_ROLES + OPTIONAL_ROLES
        recovered: dict[str, bytes] = {}
        for member, expected_role in zip(members, roles, strict=True):
            if not isinstance(member, dict) or set(member) != MEMBER_KEYS:
                _fail("member_schema_invalid")
            role = member.get("role")
            size = member.get("size_bytes")
            digest = member.get("sha256")
            encoded = member.get("content_base64")
            if (role != expected_role or role not in ALL_ROLES
                    or type(size) is not int or not 1 <= size <= MAX_MEMBER_BYTES
                    or not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None
                    or not isinstance(encoded, str)):
                _fail("member_metadata_invalid")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                _fail("member_base64_invalid")
            if (base64.b64encode(decoded).decode("ascii") != encoded
                    or len(decoded) != size):
                _fail("member_base64_or_size_mismatch")
            if hashlib.sha256(decoded).hexdigest() != digest:
                _fail("member_sha256_mismatch")
            recovered[role] = decoded
        reviewer_id = _validate_session_members(recovered)
        if reviewer_id in reviewer_ids:
            _fail("duplicate_reviewer_private_id")
        reviewer_ids.add(reviewer_id)
        result.append(recovered)
    return result


def _check_private_root(root: Path) -> Path:
    try:
        absolute = Path(os.path.abspath(root))
        info = absolute.lstat()
        if (absolute.resolve(strict=True) != absolute or stat.S_ISLNK(info.st_mode)
                or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            _fail("private_root_invalid")
        return absolute
    except OSError:
        _fail("private_root_invalid")


def _rename_noreplace(dir_fd: int, source: str, destination: str) -> None:
    """Publish a completed staging directory atomically without replacement."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        _fail("atomic_noreplace_publish_unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(dir_fd, os.fsencode(source), dir_fd,
                       os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            _fail("output_already_exists")
        if error in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            _fail("atomic_noreplace_publish_unavailable")
        _fail("atomic_publish_failed")


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
            _fail("private_write_failed")
        offset += written


def split_envelope(raw: bytes, output_dir: str | Path, *,
                   private_root: str | Path) -> dict[str, Any]:
    """Split exact members from one to 16 sessions under a new private child.

    Files are created with mode 0600, directories with mode 0700, and existing
    paths are never overwritten. Any failure removes only this call's outputs.
    """
    sessions = parse_envelope(raw)
    root = _check_private_root(Path(private_root))
    target = Path(os.path.abspath(output_dir))
    if (target.parent != root or OUTPUT_NAME_RE.fullmatch(target.name) is None
            or target.exists() or target.is_symlink()):
        _fail("output_must_be_new_private_root_child")
    root_fd = -1
    staging_fd = -1
    staging_name = f".{target.name}.staging-{secrets.token_hex(8)}"
    staging_created = False
    created_subdirs: list[str] = []
    created_files: list[tuple[str, str]] = []
    completed = False
    try:
        root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                          | getattr(os, "O_NOFOLLOW", 0))
        root_opened = os.fstat(root_fd)
        root_current = root.lstat()
        if ((root_opened.st_dev, root_opened.st_ino, root_opened.st_uid,
             stat.S_IMODE(root_opened.st_mode)) !=
                (root_current.st_dev, root_current.st_ino, root_current.st_uid,
                 stat.S_IMODE(root_current.st_mode))
                or root_opened.st_uid != os.geteuid()
                or stat.S_IMODE(root_opened.st_mode) != 0o700):
            _fail("private_root_changed_or_invalid")
        os.mkdir(staging_name, 0o700, dir_fd=root_fd)
        staging_created = True
        staging_fd = os.open(staging_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                             | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        stage_stat = os.fstat(staging_fd)
        if (not stat.S_ISDIR(stage_stat.st_mode) or stage_stat.st_uid != os.geteuid()
                or stat.S_IMODE(stage_stat.st_mode) != 0o700):
            _fail("staging_directory_invalid")
        output_sessions = []
        for sequence, members in enumerate(sessions, start=1):
            dirname = f"session_{sequence:03d}"
            os.mkdir(dirname, 0o700, dir_fd=staging_fd)
            created_subdirs.append(dirname)
            session_fd = os.open(dirname, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                 | getattr(os, "O_NOFOLLOW", 0), dir_fd=staging_fd)
            try:
                os.fchmod(session_fd, 0o700)
                written = []
                for role in REQUIRED_ROLES + OPTIONAL_ROLES:
                    if role not in members:
                        continue
                    filename = ROLE_FILENAMES[role]
                    fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=session_fd)
                    created_files.append((dirname, filename))
                    try:
                        os.fchmod(fd, 0o600)
                        _write_all(fd, members[role])
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    written.append({
                        "role": role, "basename": filename,
                        "size_bytes": len(members[role]),
                        "sha256": hashlib.sha256(members[role]).hexdigest(),
                    })
                os.fsync(session_fd)
                output_sessions.append({"sequence": sequence, "directory": dirname,
                                        "members": written})
            finally:
                os.close(session_fd)
        os.fsync(staging_fd)
        os.close(staging_fd)
        staging_fd = -1
        _rename_noreplace(root_fd, staging_name, target.name)
        staging_created = False
        completed = True
        try:
            os.fsync(root_fd)
        except OSError:
            return {"status": "published_durability_unconfirmed",
                    "execution_authorization": AUTHORIZATION,
                    "session_count": len(sessions), "output_dir": str(target),
                    "sessions": output_sessions}
        return {"status": "split_unvalidated", "execution_authorization": AUTHORIZATION,
                "session_count": len(sessions), "output_dir": str(target),
                "sessions": output_sessions}
    except EnvelopeError:
        raise
    except FileExistsError:
        _fail("output_already_exists")
    except OSError:
        _fail("private_output_creation_failed")
    finally:
        if not completed:
            if staging_created and staging_fd < 0 and root_fd >= 0:
                try:
                    staging_fd = os.open(
                        staging_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
                except OSError:
                    staging_fd = -1
            for dirname, filename in reversed(created_files):
                if staging_fd >= 0:
                    try:
                        session_fd = os.open(dirname, os.O_RDONLY
                                             | getattr(os, "O_DIRECTORY", 0)
                                             | getattr(os, "O_NOFOLLOW", 0), dir_fd=staging_fd)
                        try:
                            os.unlink(filename, dir_fd=session_fd)
                        finally:
                            os.close(session_fd)
                    except OSError:
                        pass
            for dirname in reversed(created_subdirs):
                if staging_fd >= 0:
                    try:
                        os.rmdir(dirname, dir_fd=staging_fd)
                    except OSError:
                        pass
        if staging_fd >= 0:
            os.close(staging_fd)
        if root_fd >= 0:
            if not completed and staging_created:
                try:
                    os.rmdir(staging_name, dir_fd=root_fd)
                except OSError:
                    pass
            os.close(root_fd)


def _read_envelope_file(path: Path) -> bytes:
    """Read a bounded, stable regular file without following a symlink."""
    fd = -1
    try:
        before = path.lstat()
        if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600):
            _fail("input_file_invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o600 or opened.st_size < 1
                or opened.st_size > MAX_ENVELOPE_BYTES
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            _fail("input_file_invalid_or_too_large")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, MAX_ENVELOPE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_ENVELOPE_BYTES:
                _fail("input_file_too_large")
        after = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
             opened.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns) or size != after.st_size):
            _fail("input_file_changed_during_read")
        return b"".join(chunks)
    except EnvelopeError:
        raise
    except OSError:
        _fail("input_file_unreadable")
    finally:
        if fd >= 0:
            os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = _read_envelope_file(args.input)
    result = split_envelope(raw, args.output_dir, private_root=args.private_root)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "output_dir"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
