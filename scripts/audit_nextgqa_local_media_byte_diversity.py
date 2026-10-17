#!/usr/bin/env python3
"""Hash the locally available NExT-GQA test media and report aggregate identity.

This CPU-only audit reads mapped video bytes once (streaming in 1 MiB blocks)
to detect exact duplicates and optionally compare them with the existing
private HMAC prior-exposure index. It emits counts only; it does not decode,
view, copy, or publish videos and cannot verify source/event independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from collections import Counter
from typing import Any


MAX_MEDIA_FILES = 10_000
MAX_SINGLE_MEDIA_BYTES = 128 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 16 * 1024 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_stable_fd(fd: int, *, code: str, max_bytes: int,
                   private: bool = False) -> bytes:
    before = os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes
            or (private and (before.st_uid != os.geteuid()
                             or stat.S_IMODE(before.st_mode) != 0o600))):
        raise ValueError(code)
    chunks = []
    remaining = before.st_size
    while remaining:
        block = os.read(fd, min(1024 * 1024, remaining))
        if not block:
            raise ValueError(code)
        chunks.append(block)
        remaining -= len(block)
    if os.read(fd, 1):
        raise ValueError(code)
    after = os.fstat(fd)
    if (after.st_dev != before.st_dev or after.st_ino != before.st_ino
            or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns):
        raise ValueError(code)
    return b"".join(chunks)


def _strict_json(path: Path, code: str, *, private: bool = False) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(code)
            result[key] = value
        return result
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            raw = _read_stable_fd(fd, code=code, max_bytes=16 * 1024 * 1024,
                                  private=private)
        finally:
            os.close(fd)
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError(code)))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        raise ValueError(code) from None


def _load_video_ids(test_csv: Path) -> set[str]:
    ids = set()
    try:
        with test_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"video_id", "qid"}.issubset(reader.fieldnames):
                raise ValueError("test_csv_required_columns_missing")
            for row in reader:
                video_id, qid = row.get("video_id"), row.get("qid")
                if (not isinstance(video_id, str) or not video_id.strip()
                        or not isinstance(qid, str) or not qid.strip()):
                    raise ValueError("test_csv_identifier_invalid")
                ids.add(video_id)
                del row
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("test_csv_invalid") from None
    if not ids:
        raise ValueError("test_csv_empty")
    if len(ids) > MAX_MEDIA_FILES:
        raise ValueError("media_file_count_limit_exceeded")
    return ids


def _read_prior_index(path: Path, secret_path: Path) -> tuple[set[str], bytes]:
    index = _strict_json(path, "prior_index_json_invalid", private=True)
    if (not isinstance(index, dict) or type(index.get("schema_version")) is not int
            or index.get("schema_version") != 1
            or index.get("purpose") != "outcome_blind_prior_exposure_index"
            or index.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(index.get("identities"), dict)):
        raise ValueError("prior_index_schema_invalid")
    exposed = index["identities"].get("media_sha256_hmac_sha256")
    if (not isinstance(exposed, list)
            or any(not isinstance(token, str) or not SHA256.fullmatch(token) for token in exposed)):
        raise ValueError("prior_index_media_tokens_invalid")
    try:
        fd = os.open(secret_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            raw_key = _read_stable_fd(fd, code="hmac_key_file_invalid", max_bytes=4096,
                                      private=True)
        finally:
            os.close(fd)
    except OSError:
        raise ValueError("hmac_key_file_invalid") from None
    if len(raw_key) < 32:
        raise ValueError("hmac_key_too_short")
    return set(exposed), raw_key


def _media_token(secret: bytes, media_sha256: str) -> str:
    payload = json.dumps(["nextgqa-exposure-v1", "media-sha256", media_sha256],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _open_media_beneath_root(root: Path, parts: tuple[str, ...]) -> int:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
        return fd
    except OSError:
        raise ValueError("media_file_open_failed") from None
    finally:
        os.close(current_fd)


def _hash_stable_file(path: Path, root: Path, parts: tuple[str, ...], expected_size: int) -> str:
    fd = _open_media_beneath_root(root, parts)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size != expected_size
                or before.st_size < 1 or before.st_size > MAX_SINGLE_MEDIA_BYTES):
            raise ValueError("media_file_invalid")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("media_short_read")
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(fd)
        if (after.st_size != before.st_size or after.st_ino != before.st_ino
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise ValueError("media_changed_during_hash")
        return digest.hexdigest()
    finally:
        os.close(fd)


def audit_local_media(*, test_csv: Path, video_map_json: Path, video_root: Path,
                      prior_index: Path | None = None,
                      hmac_key: Path | None = None) -> dict:
    if (prior_index is None) != (hmac_key is None):
        raise ValueError("prior_index_and_hmac_key_must_be_paired")
    root = video_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("video_root_invalid")
    ids = _load_video_ids(test_csv)
    mapping = _strict_json(video_map_json, "video_map_json_invalid")
    if not isinstance(mapping, dict):
        raise ValueError("video_map_schema_invalid")
    prior_tokens: set[str] = set()
    secret: bytes | None = None
    if prior_index is not None and hmac_key is not None:
        prior_tokens, secret = _read_prior_index(prior_index, hmac_key)

    hashes = []
    total_bytes = found = missing = 0
    sizes = []
    for video_id in sorted(ids):
        relative = mapping.get(video_id)
        if not isinstance(relative, str) or not relative.strip():
            missing += 1
            continue
        raw_parts = relative.replace("\\", "/").split("/")
        normalized = PurePosixPath(relative.replace("\\", "/"))
        if ("\x00" in relative or normalized.is_absolute()
                or any(part in {"", ".", ".."} for part in raw_parts)
                or normalized.suffix):
            raise ValueError("video_map_path_invalid")
        media_path = root.joinpath(*normalized.parts).with_suffix(".mp4")
        try:
            info = media_path.lstat()
        except OSError:
            missing += 1
            continue
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                or not media_path.resolve(strict=True).is_relative_to(root)):
            raise ValueError("media_file_path_invalid")
        if info.st_size < 1 or info.st_size > MAX_SINGLE_MEDIA_BYTES:
            raise ValueError("media_file_size_invalid")
        total_bytes += info.st_size
        if total_bytes > MAX_TOTAL_MEDIA_BYTES:
            raise ValueError("media_total_size_limit_exceeded")
        hashes.append(_hash_stable_file(media_path, root,
                                        tuple((*normalized.parts[:-1], normalized.name + ".mp4")),
                                        info.st_size))
        sizes.append(info.st_size)
        found += 1
    counts = Counter(hashes)
    unique_hashes = set(hashes)
    matched_unique = (sum(_media_token(secret, digest) in prior_tokens for digest in unique_hashes)
                      if secret is not None else None)
    matched_files = (sum(count for digest, count in counts.items()
                         if _media_token(secret, digest) in prior_tokens)
                     if secret is not None else None)
    ordered_sizes = sorted(sizes)
    return {
        "schema_version": 1,
        "purpose": "nextgqa_local_test_media_byte_diversity_audit",
        "scope": "local_streaming_sha256_counts_only_no_decode_or_media_emission",
        "dataset_video_id_count": len(ids),
        "mapped_video_files_found": found,
        "mapped_video_files_missing": missing,
        "source_bytes_hashed": total_bytes,
        "unique_media_byte_hashes": len(unique_hashes),
        "exact_duplicate_extra_files": found - len(unique_hashes),
        "exact_duplicate_hash_groups": sum(value > 1 for value in counts.values()),
        "largest_exact_duplicate_group": max(counts.values(), default=0),
        "median_file_bytes": ((ordered_sizes[(len(ordered_sizes) - 1) // 2]
                               + ordered_sizes[len(ordered_sizes) // 2]) / 2
                              if ordered_sizes else 0),
        "prior_exposure_match_status": "partial_hmac_index_compared" if secret is not None
                                       else "not_compared",
        "current_files_matching_prior_media_tokens": matched_files,
        "distinct_current_hashes_matching_prior_media_tokens": matched_unique,
        "prior_history_completeness_authenticated": False,
        "source_cluster_independence_verified": False,
        "faithfulness_scored": False,
        "gpu_used": False,
        "rights_status": "unverified_local_only",
        "limitations": [
            "exact byte equality detects only byte-identical files; transcodes or shared source events are not detected",
            "prior exposure matches cover only the supplied partial HMAC index",
            "video IDs and media byte hashes do not establish independent source/event clusters",
            "missing local files are unavailable, not evidence of dataset absence or ineligibility",
            "this is an availability/exclusion diagnostic, not a reviewed cohort or faithfulness result",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--video-map-json", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--prior-index", type=Path)
    parser.add_argument("--hmac-key", type=Path)
    args = parser.parse_args()
    try:
        report = audit_local_media(test_csv=args.test_csv,
                                   video_map_json=args.video_map_json,
                                   video_root=args.video_root,
                                   prior_index=args.prior_index,
                                   hmac_key=args.hmac_key)
    except ValueError as exc:
        print(f"audit_failed: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("audit_failed: input_unavailable_or_changed", file=sys.stderr)
        return 2
    except Exception:
        print("audit_failed: unexpected_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
