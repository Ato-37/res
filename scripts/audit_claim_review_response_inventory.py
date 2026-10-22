#!/usr/bin/env python3
"""Count exact-artifact-bound local claim-review exports without exposing content.

This inventory checks response structure only. It does not authenticate a
reviewer, validate a semantic judgment, or satisfy a signed Stage-1 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable

from inspect_claim_support_frame_review_response import inspect

FINGERPRINT_RE = re.compile(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">')
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _read_regular(path: Path, *, require_private: bool = False,
                  max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_size > max_bytes
                or require_private and stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("review_artifact_permissions_invalid" if require_private
                             else "response_file_permissions_invalid")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("response_file_truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _private_artifact(path: Path) -> bytes:
    raw = _read_regular(path, require_private=True)
    match = FINGERPRINT_RE.search(raw)
    if not match:
        raise ValueError("review_artifact_fingerprint_missing")
    canonical = raw.replace(match.group(1), b"__FINGERPRINT__", 1)
    if hashlib.sha256(canonical).hexdigest().encode("ascii") != match.group(1):
        raise ValueError("review_artifact_fingerprint_invalid")
    return raw


def _json_paths(roots: Iterable[Path]):
    for root in roots:
        try:
            root_info = root.lstat()
        except OSError:
            continue
        if (not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode)
                or root_info.st_uid != os.getuid()):
            continue
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(directory)
            dirnames[:] = [name for name in dirnames
                           if not (base / name).is_symlink()]
            for name in filenames:
                path = base / name
                if path.suffix.lower() != ".json" or path.is_symlink():
                    continue
                try:
                    info = path.lstat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode) and info.st_size <= MAX_RESPONSE_BYTES:
                    yield path


def inventory(artifacts: list[Path], roots: list[Path]) -> dict:
    artifact_rows = []
    raw_by_fingerprint = {}
    for path in artifacts:
        raw = _private_artifact(path)
        fingerprint = FINGERPRINT_RE.search(raw).group(1).decode("ascii")
        if fingerprint in raw_by_fingerprint:
            if raw_by_fingerprint[fingerprint] != raw:
                raise ValueError("review_artifact_fingerprint_collision")
            continue
        raw_by_fingerprint[fingerprint] = raw
        artifact_rows.append({
            "artifact_fingerprint": fingerprint,
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "bound_response_files": 0,
            "structurally_valid_response_files": 0,
            "self_reported_reviewer_entries": 0,
            "self_reported_item_decisions": 0,
        })

    by_fingerprint = {row["artifact_fingerprint"]: row for row in artifact_rows}
    seen_responses = set()
    candidate_json_files = 0
    bound_response_files = 0
    invalid_bound_response_files = 0
    for path in _json_paths(roots):
        candidate_json_files += 1
        try:
            response_raw = _read_regular(path)
            response = json.loads(response_raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
            continue
        if (not isinstance(response, dict)
                or response.get("purpose") != "local_claim_support_frame_review"):
            continue
        fingerprint = response.get("artifact_fingerprint")
        row = by_fingerprint.get(fingerprint)
        if row is None:
            continue
        response_sha = hashlib.sha256(response_raw).hexdigest()
        if response_sha in seen_responses:
            continue
        seen_responses.add(response_sha)
        bound_response_files += 1
        row["bound_response_files"] += 1
        checked = inspect(response, raw_by_fingerprint[fingerprint])
        if not checked.get("structurally_valid"):
            invalid_bound_response_files += 1
            continue
        row["structurally_valid_response_files"] += 1
        reviewer_count = checked["reviewer_count"]
        row["self_reported_reviewer_entries"] += reviewer_count
        row["self_reported_item_decisions"] += reviewer_count * checked["items_per_reviewer"]

    return {
        "schema_version": 1,
        "purpose": "privacy_safe_claim_review_response_inventory",
        "status": "complete_structural_inventory_only",
        "candidate_json_files_scanned": candidate_json_files,
        "artifact_count": len(artifact_rows),
        "bound_response_files_total": bound_response_files,
        "invalid_bound_response_files_total": invalid_bound_response_files,
        "artifacts": artifact_rows,
        "reviewer_identity_or_independence_verified": False,
        "semantic_support_validated": False,
        "signed_stage1_or_direct_faithfulness_gate_satisfied": False,
        "direct_intervention_score_inventory": "not_scanned",
        "limitations": [
            "counts are self-reported local exports structurally bound to exact HTML fingerprints",
            "nicknames do not authenticate reviewer identity or independent review",
            "this scanner does not inspect signed custody/release bundles or validate semantics",
            "a structurally valid response is not a direct faithfulness score",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="private HTML artifact; may be supplied more than once")
    parser.add_argument("--response-root", type=Path, action="append", required=True,
                        help="local directory scanned recursively for JSON exports")
    parser.add_argument("--output", type=Path,
                        help="optional private output JSON (no overwrite; chmod 0600)")
    args = parser.parse_args(argv)
    try:
        result = inventory(args.artifact, args.response_root)
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("inventory_output_exists")
            if not args.output.parent.is_dir():
                raise ValueError("inventory_output_parent_missing")
            fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            summary = {k: result[k] for k in (
                "status", "candidate_json_files_scanned", "artifact_count",
                "bound_response_files_total", "invalid_bound_response_files_total")}
            print(json.dumps(summary, sort_keys=True))
        else:
            print(serialized, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
