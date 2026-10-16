#!/usr/bin/env python3
"""Create an isolated reviewer directory containing only HTML and opaque media."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from audit_nextgqa_review_handoff import audit_custody
from render_nextgqa_claim_review_html import render


def _write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _package_review_after_audit(html_path: Path, worksheet: Any, custody: Any, media_dir: Path,
                   output_dir: Path, private_root: Path) -> dict:
    """Internal copier; caller must first run custody audit and bind page to inputs."""
    private_root = private_root.resolve(strict=True)
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir.parent != private_root or os.path.lexists(output_dir):
        raise ValueError("review_package_path_invalid_or_exists")
    info = private_root.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("review_package_private_root_invalid")
    html_info = html_path.lstat()
    if (not stat.S_ISREG(html_info.st_mode) or stat.S_ISLNK(html_info.st_mode)
            or html_info.st_uid != os.geteuid() or stat.S_IMODE(html_info.st_mode) != 0o600):
        raise ValueError("review_page_private_file_invalid")
    html_raw = html_path.read_bytes()
    text = html_raw.decode("utf-8")
    for row in worksheet["records"]:
        for key in ("video_sha256", "video_id", "claim_id", "claim_text_sha256", "video_path"):
            value = row.get(key)
            if isinstance(value, str) and value and value in text:
                raise ValueError("review_page_contains_stable_source_value")

    output_dir.mkdir(mode=0o700)
    try:
        os.chmod(output_dir, 0o700)
        _write_new(output_dir / "index.html", html_raw)
        for row in worksheet["records"]:
            filename = row["review_token"] + ".mp4"
            src = media_dir / filename
            src_info = src.lstat()
            if (not stat.S_ISREG(src_info.st_mode) or stat.S_ISLNK(src_info.st_mode)
                    or src_info.st_uid != os.geteuid() or stat.S_IMODE(src_info.st_mode) != 0o600):
                raise ValueError("review_media_private_file_invalid")
            media = src.read_bytes()
            entry = next((r for r in custody["records"] if r.get("review_token") == row["review_token"]), None)
            if (entry is None or hashlib.sha256(media).hexdigest() != row["video_sha256"]
                    or hashlib.sha256(bytes.fromhex(entry["media_check_salt"]) + media).hexdigest()
                    != entry["media_check_sha256"]):
                raise ValueError("review_media_crosscheck_failed")
            _write_new(output_dir / filename, media)
        names = {p.name for p in output_dir.iterdir()}
        expected = {"index.html"} | {r["review_token"] + ".mp4" for r in worksheet["records"]}
        if names != expected or any(stat.S_IMODE(p.stat().st_mode) != 0o600 for p in output_dir.iterdir()):
            raise ValueError("review_package_contents_invalid")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return {"status": "private_isolated_reviewer_package_created", "files": 4,
            "directory_mode": 0o700, "file_mode": 0o600,
            "contains_custody_manifest": False, "execution_authorization": "not_granted"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worksheet", required=True)
    p.add_argument("--lineage-manifest", required=True)
    p.add_argument("--expected-lineage-sha256", required=True)
    p.add_argument("--custody-manifest", required=True)
    p.add_argument("--expected-custody-sha256", required=True)
    p.add_argument("--media-dir", required=True)
    p.add_argument("--review-page", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--user-authorized-local-only", action="store_true")
    a = p.parse_args(argv)
    if not a.user_authorized_local_only:
        p.error("explicit_local_only_authorization_required")
    root = Path(__file__).resolve().parents[1]
    private = root / "data" / "private_nextgqa"
    paths = [Path(a.worksheet), Path(a.lineage_manifest), Path(a.custody_manifest), Path(a.review_page)]
    try:
        if any(x.stat().st_uid != os.geteuid() or stat.S_IMODE(x.stat().st_mode) != 0o600 for x in paths):
            raise ValueError("private_input_mode_invalid")
        raw = [x.read_bytes() for x in paths[:3]]
        values = [json.loads(x) for x in raw]
        failures = audit_custody(values[0], raw[0], values[1], raw[1], values[2], raw[2],
                                 Path(a.media_dir), a.expected_custody_sha256)
        if failures:
            raise ValueError("custody_audit_failed")
        if hashlib.sha256(raw[1]).hexdigest() != a.expected_lineage_sha256:
            raise ValueError("lineage_manifest_hash_mismatch")
        expected_page = render(values[0], raw[0], values[1], raw[1], a.expected_lineage_sha256,
                               values[2], raw[2], a.expected_custody_sha256)
        if paths[3].read_bytes() != expected_page:
            raise ValueError("review_page_not_bound_to_frozen_inputs")
        result = _package_review_after_audit(paths[3], values[0], values[2], Path(a.media_dir), Path(a.output_dir), private)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        p.error(str(exc).split(":", 1)[0] or "review_package_failed")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
