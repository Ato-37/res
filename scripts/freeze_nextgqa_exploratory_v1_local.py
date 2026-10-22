"""Independently reproduce and locally freeze the v1 exploratory split.

The mode-0600 receipt is an owner-local integrity anchor, not a signature or
research attestation. It records no IDs or paths and is never published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from build_nextgqa_exploratory_video_split import build_split, sha256_file

RECEIPT_NAME = "nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"
V1_PLAN_NAME = "nextgqa_tlvs_exploratory_split_20261018_01.json"
V1_SOURCE = "scripts/build_nextgqa_exploratory_video_split.py"
TEST_CSV = "NExT-GQA/datasets/nextgqa/test.csv"
VIDEO_ROOT = "data/private_nextgqa/zip_gold_candidates_20261017_01"


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def normalize_paths(plan: dict, repo: Path) -> dict:
    normalized = json.loads(json.dumps(plan))
    for part in ("calibration", "evaluation"):
        for row in normalized[part]:
            row["media_path"] = str((repo / row["media_path"]).resolve())
    return normalized


def reproduce(repo: Path) -> tuple[dict, dict]:
    base = json.loads((repo / "data/private_nextgqa" / V1_PLAN_NAME).read_text())
    prior_manifests = sorted((repo / "data").glob("nextgqa_samples*.json"))
    rebuilt = build_split(test_csv=repo / TEST_CSV, video_root=repo / VIDEO_ROOT,
        prior_manifests=prior_manifests,
        prior_video_roots=[repo / "data/private_nextgqa"], calibration_fraction=0.70)
    if normalize_paths(base, repo) != normalize_paths(rebuilt, repo):
        raise ValueError("v1_plan_reproduction_mismatch")
    report = {"schema_version": 1,
        "status": "local_v1_plan_reproduced_freeze_not_signature",
        "plan_sha256": sha256_file(repo / "data/private_nextgqa" / V1_PLAN_NAME),
        "normalized_plan_sha256": canonical(normalize_paths(rebuilt, repo)),
        "freezer_source_sha256": sha256_file(Path(__file__).resolve()),
        "v1_source_sha256": sha256_file(repo / V1_SOURCE),
        "test_csv_sha256": sha256_file(repo / TEST_CSV),
        "candidate_media_count": rebuilt["provenance"]["eligible_candidate_video_count_after_exclusions"],
        "calibration_count": len(rebuilt["calibration"]),
        "evaluation_count": len(rebuilt["evaluation"]),
        "interpretation": "local integrity anchor only; not signed human review or rights attestation"}
    return report, rebuilt


def read_and_verify_freeze(repo: Path, *, freeze_path: Path | None = None) -> dict:
    private = repo / "data/private_nextgqa"
    freeze_path = freeze_path or private / RECEIPT_NAME
    root = private.resolve(strict=True)
    if freeze_path.resolve(strict=True).parent != root or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("local_freeze_path_or_root_invalid")
    fd = os.open(freeze_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("local_freeze_mode_invalid")
        chunks = []
        while block := os.read(fd, 1024 * 1024): chunks.append(block)
    finally:
        os.close(fd)
    value = json.loads(b"".join(chunks).decode("utf-8"))
    plan_path, test_csv, source = (repo / "data/private_nextgqa" / V1_PLAN_NAME,
        repo / TEST_CSV, repo / V1_SOURCE)
    if (value.get("status") != "local_v1_plan_reproduced_freeze_not_signature"
            or value.get("plan_sha256") != sha256_file(plan_path)
            or value.get("v1_source_sha256") != sha256_file(source)
            or value.get("test_csv_sha256") != sha256_file(test_csv)
            or value.get("freezer_source_sha256") != sha256_file(Path(__file__).resolve())):
        raise ValueError("local_freeze_byte_binding_mismatch")
    regenerated, _ = reproduce(repo)
    if regenerated != value:
        raise ValueError("local_freeze_reproduction_mismatch")
    return value


def write_exclusive(path: Path, payload: dict, private: Path) -> None:
    root = private.resolve(strict=True)
    if (path.parent.resolve(strict=True) != root or stat.S_IMODE(root.stat().st_mode) != 0o700
            or root.stat().st_uid != os.getuid()):
        raise ValueError("private_root_invalid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default=RECEIPT_NAME)
    args = parser.parse_args(argv)
    repo = args.repo.resolve(); private = repo / "data/private_nextgqa"
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    try:
        receipt, _ = reproduce(repo)
        write_exclusive(private / args.output_name, receipt, private)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"freeze_failed:{type(exc).__name__}:{exc}")
    print(json.dumps({"status": receipt["status"],
        "calibration_videos": receipt["calibration_count"],
        "evaluation_videos": receipt["evaluation_count"],
        "privacy": "local-only receipt; no identifiers or hashes on stdout"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
