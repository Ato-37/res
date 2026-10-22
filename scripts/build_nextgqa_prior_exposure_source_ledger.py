#!/usr/bin/env python3
"""Build a private source-attributed HMAC ledger from an explicit roster.

This is an outcome-blind CPU audit aid. It keeps canonical video-ID, media-byte,
and video/question identities in separate HMAC domains and retains a per-source
token list so a later split audit can identify which listed sources contain a
match. It never emits raw IDs, QIDs, media hashes, questions, answers, paths, or
source filenames. The roster is a partial local history, never a freshness
certificate or execution authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

try:
    from build_nextgqa_prior_exposure_manifest import (
        FAMILIES, MAX_SOURCE_BYTES, MAX_TOTAL_SOURCE_BYTES, NAMESPACE, _loads, _read_secret,
        _read_stable_regular_file, _token, build,
    )
except ImportError:
    from scripts.build_nextgqa_prior_exposure_manifest import (
        FAMILIES, MAX_SOURCE_BYTES, MAX_TOTAL_SOURCE_BYTES, NAMESPACE, _loads, _read_secret,
        _read_stable_regular_file, _token, build,
    )

ROSTER_PURPOSE = "explicit_prior_exposure_index_inputs_with_media_and_custody_adapters"
MAX_SOURCES = 64
LEDGER_DOMAIN = b"nextgqa-source-ledger-v1\x00"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("roster_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("roster_path_invalid")
    return Path(*path.parts)


def build_source_ledger(*, roster_raw: bytes, source_raws: list[bytes],
                        secret: bytes) -> tuple[dict[str, Any], bytes]:
    if not isinstance(roster_raw, bytes) or len(roster_raw) > MAX_SOURCE_BYTES:
        raise ValueError("roster_size_invalid")
    roster = _loads(roster_raw)
    if (not isinstance(roster, dict)
            or set(roster) != {"schema_version", "purpose", "inputs", "omitted_duplicate_content"}
            or type(roster.get("schema_version")) is not int
            or roster.get("schema_version") != 1
            or roster.get("purpose") != ROSTER_PURPOSE
            or not isinstance(roster.get("inputs"), list)
            or not isinstance(roster.get("omitted_duplicate_content"), list)):
        raise ValueError("roster_schema_invalid")
    for omitted in roster["omitted_duplicate_content"]:
        if (not isinstance(omitted, dict)
                or set(omitted) != {"family", "namespace", "path", "same_bytes_as"}
                or omitted.get("family") not in FAMILIES
                or not isinstance(omitted.get("namespace"), str)
                or NAMESPACE.fullmatch(omitted["namespace"]) is None):
            raise ValueError("roster_omitted_duplicate_invalid")
        _safe_relative_path(omitted.get("path"))
        _safe_relative_path(omitted.get("same_bytes_as"))
    rows = roster["inputs"]
    if not 1 <= len(rows) <= MAX_SOURCES or len(source_raws) != len(rows):
        raise ValueError("source_count_invalid")
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("secret_invalid")

    seen_paths: set[Path] = set()
    seen_content: set[str] = set()
    total_size = 0
    source_records = []
    unions = {"dataset_video_hmac_sha256": set(),
              "media_sha256_hmac_sha256": set(),
              "video_question_pair_hmac_sha256": set()}
    for ordinal, (entry, raw) in enumerate(zip(rows, source_raws, strict=True), start=1):
        if (not isinstance(entry, dict)
                or set(entry) != {"family", "namespace", "relative_path"}
                or entry.get("family") not in FAMILIES
                or not isinstance(entry.get("namespace"), str)
                or NAMESPACE.fullmatch(entry["namespace"]) is None):
            raise ValueError("roster_entry_invalid")
        relative = _safe_relative_path(entry["relative_path"])
        if relative in seen_paths:
            raise ValueError("roster_duplicate_path")
        seen_paths.add(relative)
        if not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("source_size_invalid")
        total_size += len(raw)
        if total_size > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("source_total_size_invalid")
        content_sha = hashlib.sha256(raw).hexdigest()
        if content_sha in seen_content:
            raise ValueError("duplicate_source_content")
        seen_content.add(content_sha)
        manifest, _ = build([(entry["family"], entry["namespace"],
                              f"source-{ordinal}", raw)], secret)
        source = manifest["sources"][0]
        identities = manifest["identities"]
        for name in unions:
            unions[name].update(identities[name])
        source_records.append({
            "source_ordinal": ordinal,
            "source_id_hmac_sha256": source["source_fingerprint_hmac_sha256"],
            "source_class": source["source_class"],
            "identity_namespace": source["identity_namespace"],
            "status": source["status"],
            "rows_total": source["rows_total"],
            "rows_with_any_identity": source["rows_with_any_identity"],
            "rows_with_video_id": source["rows_with_video_id"],
            "rows_with_media_sha256": source["rows_with_media_sha256"],
            "rows_with_video_question_pair": source["rows_with_video_question_pair"],
            "video_identity_hmac_sha256": identities["dataset_video_hmac_sha256"],
            "media_identity_hmac_sha256": identities["media_sha256_hmac_sha256"],
            "video_question_pair_hmac_sha256": identities["video_question_pair_hmac_sha256"],
            "unresolved_reason_counts": source["unresolved_reason_counts"],
        })

    result = {
        "schema_version": 1,
        "purpose": "source_attributed_prior_exposure_hmac_ledger",
        "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
        "source_roster_hmac_sha256": _token(
            secret, "source-file-sha256", (hashlib.sha256(roster_raw).hexdigest(),)),
        "sources": source_records,
        "known_identity_unions": {name: sorted(tokens) for name, tokens in unions.items()},
        "coverage": {
            "source_count": len(source_records),
            "sources_with_identity_rows": sum(
                (row["rows_with_any_identity"] or 0) > 0 for row in source_records),
            "historical_completeness_authenticated": False,
            "hmac_key_to_source_binding_authenticated": False,
            "freshness_conclusion": "not_made",
            "execution_authorization": "not_granted",
        },
        "limitations": [
            "only explicitly listed roster sources were processed",
            "unlisted, deleted, aggregate-only, or identity-redacted history is not recovered",
            "a non-match is not evidence of freshness",
            "dataset video ID, media bytes, and video/question pair use separate identity domains",
            "namespace correctness and source provenance remain unauthenticated",
        ],
    }
    result["ledger_integrity_hmac_sha256"] = hmac.new(
        secret, LEDGER_DOMAIN + _canonical(result), hashlib.sha256).hexdigest()
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    return result, encoded


def _write_private_exclusive(path: Path, raw: bytes) -> None:
    parent = path.parent
    info = parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid()):
        raise ValueError("output_parent_not_owner_only")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(path.name, 0o600, dir_fd=parent_fd, follow_symlinks=False)
        except Exception:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--secret-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve(strict=True)
    private_root = (repo / "data/private_nextgqa").resolve(strict=True)
    try:
        if args.roster.is_absolute() or args.secret_file.is_absolute() or args.output.is_absolute():
            raise ValueError("paths_must_be_repository_relative")
        roster_path = repo / args.roster
        output_path = repo / args.output
        if output_path.parent.resolve(strict=True) != private_root:
            raise ValueError("output_must_be_in_private_root")
        roster_raw = _read_stable_regular_file(roster_path, max_bytes=MAX_SOURCE_BYTES,
                                               required_mode=0o600,
                                               required_uid=os.getuid())
        secret = _read_secret(repo / args.secret_file, required_uid=os.getuid())
        roster = _loads(roster_raw)
        if not isinstance(roster, dict) or not isinstance(roster.get("inputs"), list):
            raise ValueError("roster_schema_invalid")
        source_raws = []
        for entry in roster["inputs"]:
            if not isinstance(entry, dict):
                raise ValueError("roster_entry_invalid")
            relative = _safe_relative_path(entry.get("relative_path"))
            source_raws.append(_read_stable_regular_file(repo / relative,
                                                         max_bytes=MAX_SOURCE_BYTES,
                                                         required_uid=os.getuid()))
        result, encoded = build_source_ledger(roster_raw=roster_raw,
                                              source_raws=source_raws, secret=secret)
        _write_private_exclusive(output_path, encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "ledger_build_failed")
    print(json.dumps({"status": "completed_local_only",
                      "source_count": result["coverage"]["source_count"],
                      "sources_with_identity_rows": result["coverage"]["sources_with_identity_rows"],
                      "execution_authorization": "not_granted"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
