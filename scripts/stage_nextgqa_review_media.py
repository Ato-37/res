#!/usr/bin/env python3
"""Stage opaque, hash-verified local media copies for an independent review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Any

from private_output_paths import validate_private_output
from render_nextgqa_claim_review_html import render

MAX_MEDIA_BYTES = 64_000_000


def _write_exclusive(path: Path, raw: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def stage(worksheet: Any, worksheet_raw: bytes, lineage: Any, lineage_raw: bytes,
          expected_lineage_sha256: str, video_root: Path, media_dir: Path,
          manifest_path: Path, repo_root: Path) -> dict:
    """Copy exact source bytes to random-token filenames and return a private manifest."""
    private_root = repo_root.resolve() / "data" / "private_nextgqa"
    if media_dir.parent != private_root or manifest_path.parent != private_root:
        raise ValueError("custody_outputs_must_be_private_direct_children")
    if os.path.lexists(media_dir) or os.path.lexists(manifest_path):
        raise FileExistsError("custody_output_exists")
    media_dir.mkdir(mode=0o700)
    os.chmod(media_dir, 0o700)
    info = media_dir.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("custody_media_directory_invalid")

    try:
        lineage_by_claim = {row["claim_id"]: row for row in lineage["records"]}
        custody_rows = []
        for ordinal, row in enumerate(worksheet["records"], start=1):
            lineage_row = lineage_by_claim.get(row["claim_id"])
            if not isinstance(lineage_row, dict):
                raise ValueError("custody_lineage_record_missing")
            relative = Path(lineage_row["relative_video_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("custody_source_path_invalid")
            source = (video_root / relative).resolve(strict=True)
            root = video_root.resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("custody_source_path_invalid")
            raw = source.read_bytes()
            if len(raw) > MAX_MEDIA_BYTES:
                raise ValueError("custody_source_media_exceeds_review_limit")
            source_sha256 = hashlib.sha256(raw).hexdigest()
            if source_sha256 != row["video_sha256"]:
                raise ValueError("custody_source_media_hash_mismatch")
            filename = row["review_token"] + ".mp4"
            destination = media_dir / filename
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            staged_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
            if staged_sha256 != source_sha256:
                raise ValueError("custody_staged_media_hash_mismatch")
            salt = secrets.token_bytes(32)
            media_check_sha256 = hashlib.sha256(salt + destination.read_bytes()).hexdigest()
            custody_rows.append({
                "review_token": row["review_token"], "row_ordinal": ordinal,
                "media_file": filename, "source_media_sha256": source_sha256,
                "staged_media_sha256": staged_sha256, "size_bytes": len(raw),
                "media_check_salt": salt.hex(), "media_check_sha256": media_check_sha256,
            })

        if len(custody_rows) != 3 or len({r["review_token"] for r in custody_rows}) != 3:
            raise ValueError("custody_record_coverage_invalid")
        manifest = {
            "schema_version": 1,
            "purpose": "private_custodian_token_to_verified_media_manifest",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
            "source_lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
            "record_count": len(custody_rows), "records": custody_rows,
        }
        encoded = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        render(worksheet, worksheet_raw, lineage, lineage_raw, expected_lineage_sha256,
               manifest, encoded, hashlib.sha256(encoded).hexdigest())
        _write_exclusive(manifest_path, encoded)
    except BaseException:
        for child in media_dir.iterdir():
            child.unlink()
        media_dir.rmdir()
        raise
    return {"staged_records": len(custody_rows), "media_directory_mode": 0o700,
            "media_file_mode": 0o600, "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "execution_authorization": "not_granted"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--custody-manifest", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    root = Path(__file__).resolve().parents[1]
    try:
        media_dir = Path(os.path.abspath(args.media_dir))
        manifest_path = validate_private_output(args.custody_manifest, root)
        worksheet_path = Path(args.worksheet)
        lineage_path = Path(args.lineage_manifest)
        if any(stat.S_IMODE(p.stat().st_mode) != 0o600 for p in (worksheet_path, lineage_path)):
            raise ValueError("private_input_mode_invalid")
        worksheet_raw, lineage_raw = worksheet_path.read_bytes(), lineage_path.read_bytes()
        worksheet, lineage = json.loads(worksheet_raw), json.loads(lineage_raw)
        result = stage(worksheet, worksheet_raw, lineage, lineage_raw,
                       args.expected_lineage_sha256, Path(args.video_root),
                       media_dir, manifest_path, root)
    except FileExistsError:
        parser.error("refusing_to_overwrite_custody_output")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "custody_stage_invalid")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
