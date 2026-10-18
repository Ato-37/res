#!/usr/bin/env python3
"""Freeze a private exact-path/content manifest for a local MP4 candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

try:
    from reconcile_nextgqa_prior_exposure_pool import (
        MAX_TOTAL_MEDIA_BYTES, _hash_stable_file, _open_candidate_root,
        _stat_identity, _tree_snapshot,
    )
except ImportError:
    from scripts.reconcile_nextgqa_prior_exposure_pool import (
        MAX_TOTAL_MEDIA_BYTES, _hash_stable_file, _open_candidate_root,
        _stat_identity, _tree_snapshot,
    )


def freeze(pool_root: Path, output: Path, expected_videos: int = 490) -> dict:
    root_fd, root_identity = _open_candidate_root(pool_root)
    try:
        snapshot, paths = _tree_snapshot(root_fd)
        if expected_videos != 490 or len(paths) != expected_videos:
            raise ValueError("candidate_video_count_mismatch")
        total_bytes = sum(snapshot[path][4] for path in paths)
        if total_bytes > MAX_TOTAL_MEDIA_BYTES:
            raise ValueError("candidate_total_size_limit_exceeded")
        entries = []
        for path in paths:
            digest, _size = _hash_stable_file(root_fd, path, snapshot, max_bytes=512 * 1024**2)
            entries.append({"path": path, "sha256": digest})
        after, _ = _tree_snapshot(root_fd)
        if after != snapshot or _stat_identity(os.fstat(root_fd)) != root_identity:
            raise ValueError("candidate_tree_changed_during_freeze")
    finally:
        os.close(root_fd)
    payload = (json.dumps({"schema_version": 1, "purpose": "private_candidate_pool_freeze",
                          "expected_videos": expected_videos, "entries": entries},
                         sort_keys=True, separators=(",", ":")) + "\n").encode()
    parent = output.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode)
            or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700):
        raise ValueError("manifest_parent_must_be_owned_mode_0700_directory")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return {"status": "frozen", "video_count": len(entries),
            "private_inventory_fingerprint_recorded": True,
            "rights_status": "unverified_local_only"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-videos", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = freeze(args.pool_root, args.output, args.expected_videos)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "pool_freeze_failed")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
