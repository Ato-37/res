#!/usr/bin/env python3
"""Aggregate-only exact-byte reconciliation of a private MP4 pool to prior exposure.

This CPU-only diagnostic emits counts only. It establishes matches only to the
caller-supplied, explicitly incomplete HMAC index; it cannot certify an
unmatched file as unexposed or source/event-independent.
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
        MAX_SOURCE_BYTES, _loads as _json, _open_path_no_follow, _read_secret,
        _read_stable_regular_file, _stat_identity, _token,
    )
except ImportError:  # package import under unittest discovery
    from scripts.build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _loads as _json, _open_path_no_follow, _read_secret,
        _read_stable_regular_file, _stat_identity, _token,
    )

MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_MEDIA_FILES = 10000
MAX_TOTAL_MEDIA_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_INDEX_BYTES = 256 * 1024 * 1024


def _tree_snapshot(root_fd: int) -> tuple[dict[str, tuple[int, ...]], list[str]]:
    root_info = os.fstat(root_fd)
    snapshot: dict[str, tuple[int, ...]] = {".": _stat_identity(root_info)}
    videos: list[str] = []

    def visit(directory_fd: int, relative: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise ValueError("candidate_tree_unreadable") from exc
        for name in names:
            child_rel = f"{relative}/{name}" if relative != "." else name
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError("candidate_tree_changed") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("candidate_tree_symlink_rejected")
            snapshot[child_rel] = _stat_identity(info)
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError("candidate_tree_changed") from exc
                try:
                    if _stat_identity(os.fstat(child_fd)) != _stat_identity(info):
                        raise ValueError("candidate_tree_changed")
                    visit(child_fd, child_rel)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                if name.lower().endswith(".mp4"):
                    videos.append(child_rel)
            else:
                raise ValueError("candidate_tree_special_file_rejected")
    visit(root_fd, ".")
    if len(videos) > MAX_MEDIA_FILES:
        raise ValueError("candidate_media_count_limit_exceeded")
    return snapshot, videos


def _open_candidate_root(path: Path) -> tuple[int, tuple[int, ...]]:
    try:
        root_fd = _open_path_no_follow(path)
    except OSError as exc:
        raise ValueError("candidate_root_unavailable") from exc
    root_info = os.fstat(root_fd)
    if (not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.getuid() or stat.S_IMODE(root_info.st_mode) != 0o700):
        os.close(root_fd)
        raise ValueError("candidate_root_must_be_owned_private_directory")
    return root_fd, _stat_identity(root_info)


def _hash_stable_file(root_fd: int, relative: str,
                      expected_tree: dict[str, tuple[int, ...]], *,
                      max_bytes: int) -> tuple[str, int]:
    parts = relative.split("/")
    current_fd = root_fd
    owned_fds: list[int] = []
    try:
        for component in parts[:-1]:
            current_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=current_fd)
            owned_fds.append(current_fd)
            prefix = "/".join(parts[:len(owned_fds)])
            if _stat_identity(os.fstat(current_fd)) != expected_tree[prefix]:
                raise ValueError("candidate_tree_changed_during_audit")
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=current_fd)
    except (OSError, ValueError) as exc:
        for opened_fd in reversed(owned_fds):
            os.close(opened_fd)
        if isinstance(exc, ValueError):
            raise
        raise ValueError("candidate_media_unavailable") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_size < 0
                    or before.st_size > min(MAX_MEDIA_BYTES, max_bytes)):
                raise ValueError("candidate_media_size_or_type_invalid")
            if _stat_identity(before) != expected_tree[relative]:
                raise ValueError("candidate_tree_changed_during_audit")
            total = 0
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > min(MAX_MEDIA_BYTES, max_bytes):
                    raise ValueError("candidate_media_size_or_type_invalid")
                digest.update(block)
            after = os.fstat(stream.fileno())
            if _stat_identity(before) != _stat_identity(after) or total != after.st_size:
                raise ValueError("candidate_media_changed_during_read")
        return digest.hexdigest(), total
    finally:
        for opened_fd in reversed(owned_fds):
            os.close(opened_fd)


def _load_index(raw: bytes) -> set[str]:
    value: Any = _json(raw)
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or type(value.get("schema_version")) is not int
            or value.get("purpose") != "outcome_blind_prior_exposure_index"
            or value.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(value.get("coverage"), dict)
            or value["coverage"].get("historical_completeness_authenticated") is not False
            or not isinstance(value.get("identities"), dict)):
        raise ValueError("prior_index_schema_invalid")
    identities = value["identities"].get("media_sha256_hmac_sha256")
    if (not isinstance(identities, list)
            or any(not isinstance(item, str) or len(item) != 64
                   or any(char not in "0123456789abcdef" for char in item)
                   for item in identities)
            or len(identities) != len(set(identities))):
        raise ValueError("prior_index_media_identity_invalid")
    return set(identities)


def reconcile(*, pool_root: Path, index_raws: list[bytes], secret: bytes,
              expected_video_files: int | None = None) -> dict[str, Any]:
    if not index_raws or len(index_raws) > 64:
        raise ValueError("prior_index_count_invalid")
    prior_media_tokens: set[str] = set()
    total_index_bytes = 0
    for index_raw in index_raws:
        total_index_bytes += len(index_raw)
        if len(index_raw) > MAX_SOURCE_BYTES or total_index_bytes > MAX_TOTAL_INDEX_BYTES:
            raise ValueError("prior_index_size_limit_exceeded")
        prior_media_tokens.update(_load_index(index_raw))
    root_fd, root_identity = _open_candidate_root(pool_root)
    try:
        before, videos = _tree_snapshot(root_fd)
        candidate_digests: set[str] = set()
        total_media_bytes = 0
        for relative in videos:
            remaining = MAX_TOTAL_MEDIA_BYTES - total_media_bytes
            if remaining <= 0:
                raise ValueError("candidate_total_size_limit_exceeded")
            digest, byte_count = _hash_stable_file(
                root_fd, relative, before, max_bytes=remaining)
            total_media_bytes += byte_count
            if total_media_bytes > MAX_TOTAL_MEDIA_BYTES:
                raise ValueError("candidate_total_size_limit_exceeded")
            candidate_digests.add(digest)
        after, _ = _tree_snapshot(root_fd)
        if before != after or _stat_identity(os.fstat(root_fd)) != root_identity:
            raise ValueError("candidate_tree_changed_during_audit")
    finally:
        os.close(root_fd)
    candidate_tokens = {_token(secret, "media-sha256", (digest,)) for digest in candidate_digests}
    matched = len(candidate_tokens & prior_media_tokens)
    return {
        "schema_version": 1,
        "purpose": "prior_exposure_candidate_pool_reconciliation_aggregate_only",
        "status": "completed" if expected_video_files is None or len(videos) == expected_video_files
                 else "candidate_file_count_mismatch",
        "candidate_video_file_count": len(videos),
        "candidate_unique_media_count": len(candidate_digests),
        "candidate_media_bytes_hashed": total_media_bytes,
        "prior_index_manifest_count": len(index_raws),
        "prior_index_media_identity_count": len(prior_media_tokens),
        "exact_media_byte_matches": matched,
        "unique_media_unmatched_to_supplied_index": len(candidate_digests) - matched,
        "expected_video_file_count": expected_video_files,
        "historical_completeness_authenticated": False,
        "source_event_independence_assessed": False,
        "rights_status": "unverified_local_only",
        "faithfulness_scored": False,
        "model_forwards": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True, type=Path,
                        help="owner-only candidate directory containing MP4s")
    parser.add_argument("--prior-index", required=True, action="append", type=Path,
                        help="private HMAC prior-exposure index; repeat to union known histories")
    parser.add_argument("--hmac-key", required=True, type=Path,
                        help="private mode-0600 HMAC key")
    parser.add_argument("--expected-video-files", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        secret = _read_secret(args.hmac_key)
        if len(args.prior_index) > 64:
            raise ValueError("prior_index_count_invalid")
        index_raws = []
        index_bytes = 0
        for path in args.prior_index:
            remaining = MAX_TOTAL_INDEX_BYTES - index_bytes
            raw = _read_stable_regular_file(path,
                                           max_bytes=min(MAX_SOURCE_BYTES, remaining),
                                           required_mode=0o600)
            index_raws.append(raw)
            index_bytes += len(raw)
        result = reconcile(pool_root=args.pool_root, index_raws=index_raws, secret=secret,
                           expected_video_files=args.expected_video_files)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "reconciliation_failed")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
