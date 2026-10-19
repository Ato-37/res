#!/usr/bin/env python3
"""Count mapped video bytes and known prior-exposure matches inside a local ZIP.

This is an outcome-blind availability audit. It emits aggregate counts only;
it does not extract, decode, display, or publish media or dataset identifiers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

from audit_nextgqa_local_media_byte_diversity import (
    MAX_MEDIA_FILES,
    MAX_SINGLE_MEDIA_BYTES,
    MAX_TOTAL_MEDIA_BYTES,
    _media_token,
    _read_prior_index,
    _strict_json,
)

MAX_ARCHIVE_FILE_BYTES = 32 * 1024**3
MAX_ARCHIVE_MEMBERS = 20_000
MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024**2


def _preflight_zip_directory(stream, before: os.stat_result) -> int:
    if (not stat.S_ISREG(before.st_mode) or before.st_size <= 0
            or before.st_size > MAX_ARCHIVE_FILE_BYTES):
        raise ValueError("zip_file_invalid")
    end_record = zipfile._EndRecData(stream)
    if end_record is None:
        raise ValueError("zip_input_invalid")
    entry_count = int(end_record[zipfile._ECD_ENTRIES_TOTAL])
    directory_size = int(end_record[zipfile._ECD_SIZE])
    if entry_count < 0 or entry_count > MAX_ARCHIVE_MEMBERS:
        raise ValueError("zip_member_count_limit_exceeded")
    if directory_size < 0 or directory_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("zip_central_directory_limit_exceeded")
    return entry_count


def _open_checked_zip(zip_path: Path):
    try:
        fd = os.open(zip_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        stream = os.fdopen(fd, "rb")
    except OSError:
        raise ValueError("zip_input_unavailable") from None
    try:
        before = os.fstat(stream.fileno())
        entry_count = _preflight_zip_directory(stream, before)
        return stream, before, entry_count
    except BaseException:
        stream.close()
        raise


def _video_ids(csv_path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "video_id" not in reader.fieldnames:
                raise ValueError("split_csv_video_id_missing")
            for row in reader:
                value = row.get("video_id")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("split_csv_video_id_invalid")
                ids.add(value)
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("split_csv_invalid") from None
    if not ids:
        raise ValueError("split_csv_empty")
    if len(ids) > MAX_MEDIA_FILES:
        raise ValueError("split_video_id_limit_exceeded")
    return ids


def _map_member_path(value: str) -> str:
    raw = value.replace("\\", "/")
    raw_parts = raw.split("/")
    normalized = PurePosixPath(raw)
    parts = normalized.parts
    if (not value or "\x00" in value or normalized.is_absolute()
            or not parts or any(part in {"", ".", ".."} for part in raw_parts)
            or normalized.suffix):
        raise ValueError("video_map_path_invalid")
    return str(normalized.with_suffix(".mp4"))


def audit_zip_split(*, split_csv: Path, video_map_json: Path, zip_path: Path,
                    prior_indices: list[Path], hmac_key: Path) -> dict:
    if not prior_indices:
        raise ValueError("prior_index_required")
    ids = _video_ids(split_csv)
    mapping = _strict_json(video_map_json, "video_map_json_invalid")
    if not isinstance(mapping, dict):
        raise ValueError("video_map_schema_invalid")
    loaded = [_read_prior_index(path, hmac_key) for path in prior_indices]
    keys = {entry[1] for entry in loaded}
    if len(keys) != 1:
        raise ValueError("prior_index_key_mismatch")
    secret = loaded[0][1]
    prior_tokens = [entry[0] for entry in loaded]

    found = missing = ambiguous = 0
    total_bytes = 0
    unique_hashes: set[str] = set()
    matches = [set() for _ in prior_indices]
    duplicate_hash_files = 0
    try:
        zip_stream, archive_stat, declared_member_count = _open_checked_zip(zip_path)
        try:
          with zipfile.ZipFile(zip_stream) as archive:
            all_entries = archive.infolist()
            if len(all_entries) != declared_member_count:
                raise ValueError("zip_member_count_changed")
            entries = [info for info in all_entries
                       if not info.is_dir() and info.filename.lower().endswith(".mp4")]
            by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in entries:
                by_name.setdefault(info.filename, []).append(info)
            selected_member_names: set[str] = set()
            for video_id in sorted(ids):
                relative = mapping.get(video_id)
                if not isinstance(relative, str) or not relative.strip():
                    missing += 1
                    continue
                target = _map_member_path(relative)
                candidates = [info for name, infos in by_name.items()
                              if name == target or name.endswith("/" + target)
                              for info in infos]
                if not candidates:
                    missing += 1
                    continue
                if len(candidates) != 1:
                    ambiguous += 1
                    continue
                info = candidates[0]
                member_path = PurePosixPath(info.filename)
                raw_member_parts = info.filename.split("/")
                mode = (info.external_attr >> 16) & 0o170000
                file_type = stat.S_IFMT(mode)
                if (member_path.is_absolute()
                        or "\\" in info.filename
                        or any(part in {"", ".", ".."} for part in raw_member_parts)
                        or file_type not in (0, stat.S_IFREG)
                        or info.file_size < 1
                        or info.file_size > MAX_SINGLE_MEDIA_BYTES):
                    raise ValueError("zip_member_invalid")
                if info.filename in selected_member_names:
                    raise ValueError("video_map_member_alias")
                selected_member_names.add(info.filename)
                total_bytes += info.file_size
                if total_bytes > MAX_TOTAL_MEDIA_BYTES:
                    raise ValueError("zip_total_size_limit_exceeded")
                digest = hashlib.sha256()
                size = 0
                try:
                    with archive.open(info, "r") as stream:
                        while True:
                            block = stream.read(1024 * 1024)
                            if not block:
                                break
                            digest.update(block)
                            size += len(block)
                except (OSError, RuntimeError, zipfile.BadZipFile, EOFError):
                    raise ValueError("zip_member_read_failed") from None
                if size != info.file_size:
                    raise ValueError("zip_member_size_mismatch")
                media_hash = digest.hexdigest()
                if media_hash in unique_hashes:
                    duplicate_hash_files += 1
                unique_hashes.add(media_hash)
                token = _media_token(secret, media_hash)
                for index, token_set in enumerate(prior_tokens):
                    if token in token_set:
                        matches[index].add(media_hash)
                found += 1
          after = os.fstat(zip_stream.fileno())
          if (after.st_dev != archive_stat.st_dev or after.st_ino != archive_stat.st_ino
                  or after.st_size != archive_stat.st_size
                  or after.st_mtime_ns != archive_stat.st_mtime_ns
                  or after.st_ctime_ns != archive_stat.st_ctime_ns):
                raise ValueError("zip_changed_during_audit")
        finally:
            zip_stream.close()
    except zipfile.BadZipFile:
        raise ValueError("zip_input_invalid") from None
    except OSError:
        raise ValueError("zip_input_unavailable") from None

    union_matches = set().union(*matches)
    return {
        "schema_version": 1,
        "purpose": "nextgqa_zip_split_media_exposure_availability_audit",
        "scope": "mapped_zip_media_stream_hash_counts_only_no_extract_or_decode",
        "split_video_id_count": len(ids),
        "zip_mp4_member_count": len(entries),
        "mapped_video_members_found": found,
        "mapped_video_members_missing": missing,
        "mapped_video_members_ambiguous": ambiguous,
        "mapped_bytes_hashed": total_bytes,
        "unique_media_byte_hashes": len(unique_hashes),
        "duplicate_hash_extra_files": duplicate_hash_files,
        "distinct_hashes_matching_each_prior_index": [len(value) for value in matches],
        "distinct_hashes_matching_prior_index_union": len(union_matches),
        "prior_history_completeness_authenticated": False,
        "source_cluster_independence_verified": False,
        "media_extracted_or_emitted": False,
        "faithfulness_scored": False,
        "gpu_used": False,
        "rights_status": "unverified_local_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--video-map-json", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--prior-index", required=True, action="append", type=Path)
    parser.add_argument("--hmac-key", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_zip_split(split_csv=args.split_csv,
                                 video_map_json=args.video_map_json,
                                 zip_path=args.zip_path,
                                 prior_indices=args.prior_index,
                                 hmac_key=args.hmac_key)
    except ValueError as exc:
        print(f"audit_failed: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("audit_failed: unexpected_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
