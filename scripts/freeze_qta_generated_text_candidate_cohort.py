#!/usr/bin/env python3
"""Freeze an outcome-blind candidate order excluding known local media hashes.

This creates only a private hash-pinned candidate lock. It does not decode,
load a model, establish historical freshness, or authorize scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SHA256 = re.compile(r"^[0-9a-f]{64}$")
MEDIA_SHA_KEY = re.compile(r"(?:video|media).{0,24}sha256|sha256.{0,24}(?:video|media)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_media_hashes(value, *, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and SHA256.fullmatch(child) and MEDIA_SHA_KEY.search(str(key)):
                found.add(child)
            _walk_media_hashes(child, found=found)
    elif isinstance(value, list):
        for child in value:
            _walk_media_hashes(child, found=found)


def collect_known_local_media_hashes(private_root: Path) -> tuple[set[str], list[dict]]:
    private_root = private_root.resolve(strict=True)
    if not private_root.is_dir() or stat.S_IMODE(private_root.stat().st_mode) != 0o700:
        raise ValueError("private_root_mode_invalid")
    hashes: set[str] = set()
    files = []
    for path in sorted(private_root.rglob("*.json")):
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or path.is_symlink()):
            raise ValueError("private_inventory_file_mode_or_type_invalid")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("private_inventory_json_invalid") from None
        # A candidate lock freezes an outcome-blind order but exposes no video
        # to a human/model. Its proposed candidates must not become "prior
        # exposure" merely because a later freeze inventories this lock.
        if (isinstance(payload, dict)
                and payload.get("status") == "candidate_order_frozen_geometry_not_preflighted_not_authorizing"
                and isinstance(payload.get("purpose"), str)
                and payload["purpose"].startswith(
                    "exploratory_generated_text_annotation_dependence_")
                and isinstance(payload.get("candidates"), list)):
            payload = {key: value for key, value in payload.items() if key != "candidates"}
        _walk_media_hashes(payload, found=hashes)
        files.append({"relative_path": path.relative_to(private_root).as_posix(),
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    return hashes, files


def _local_video_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def build_candidate_lock(*, dataset_root: Path, video_root: Path,
                         private_root: Path, split: str = "test") -> dict:
    dataset_root = dataset_root.resolve(strict=True)
    video_root = video_root.resolve(strict=True)
    if split not in {"test", "val"}:
        raise ValueError("candidate_split_invalid")
    excluded, prior_files = collect_known_local_media_hashes(private_root)
    data_names = (f"{split}.csv", f"gsub_{split}.json", "map_vid_vidorID.json",
                  f"frame2time_{split}.json")
    data_hashes = {name: sha256_file(dataset_root / name) for name in data_names}
    with (dataset_root / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    gsub = json.loads((dataset_root / f"gsub_{split}.json").read_text(encoding="utf-8"))
    video_map = json.loads((dataset_root / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    paths: dict[str, list[Path]] = {}
    for path in sorted(video_root.rglob("*.mp4")):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(video_root):
            raise ValueError("candidate_media_path_invalid")
        paths.setdefault(_local_video_key(path.relative_to(video_root).as_posix()), []).append(path)
    by_video = {}
    for row in rows:
        video_id, qid = str(row.get("video_id", "")), str(row.get("qid", ""))
        if not video_id or not qid or video_id not in video_map:
            continue
        annotation = gsub.get(video_id)
        locations = annotation.get("location") if isinstance(annotation, dict) else None
        if not isinstance(locations, dict) or not isinstance(locations.get(qid), list) or not locations[qid]:
            continue
        key = _local_video_key(str(video_map[video_id]))
        candidates = paths.get(key, [])
        if len(candidates) != 1:
            continue
        by_video.setdefault(video_id, []).append((qid, candidates[0]))

    ordered = []
    seen_media = set()
    excluded_count = 0
    duplicate_media_count = 0
    for video_id, qrows in by_video.items():
        qid, path = min(qrows, key=lambda pair: hashlib.sha256(
            f"{video_id}:{pair[0]}".encode("utf-8")).digest())
        media_sha = sha256_file(path)
        if media_sha in excluded:
            excluded_count += 1
            continue
        if media_sha in seen_media:
            duplicate_media_count += 1
            continue
        seen_media.add(media_sha)
        ordered.append({"video_id": video_id, "qid": qid,
                        "video_relative_path": path.relative_to(video_root).as_posix(),
                        "video_sha256": media_sha,
                        "pair_sha256": hashlib.sha256(f"{video_id}\0{qid}".encode()).hexdigest(),
                        "outcome_blind_video_order_sha256": hashlib.sha256(
                            video_id.encode("utf-8")).hexdigest()})
    ordered.sort(key=lambda row: (row["outcome_blind_video_order_sha256"], row["pair_sha256"]))
    return {"schema_version": 1,
            "status": "candidate_order_frozen_geometry_not_preflighted_not_authorizing",
            "purpose": f"exploratory_generated_text_annotation_dependence_{split}_candidate_cohort",
            "split": split,
            "candidate_count": len(ordered), "excluded_known_media_hash_count": excluded_count,
            "duplicate_media_hash_count": duplicate_media_count,
            "known_prior_media_hashes": sorted(excluded),
            "prior_inventory_files": prior_files,
            "prior_inventory_sha256": hashlib.sha256(json.dumps(
                prior_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "dataset_sha256": data_hashes,
            "candidate_order_rule": "SHA256(video_id), then SHA256(video_id NUL qid); one qid per video; no outcomes",
            "video_root_identity_sha256": hashlib.sha256(str(video_root).encode()).hexdigest(),
            "candidates": ordered,
            "history_limit": "Excludes media hashes present in current local mode-0600 JSON only; not exhaustive of prior human/model exposure."}


def write_private_exclusive(path: Path, payload: dict, *, private_root: Path) -> None:
    private_root = private_root.resolve(strict=True)
    path = path.absolute()
    if path.parent.resolve(strict=True) != private_root or path.exists() or path.is_symlink():
        raise ValueError("private_output_path_invalid_or_exists")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True, help="new basename under data/private_nextgqa")
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="NExT-GQA split; selection order is outcome-blind within the split")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    private_root = repo / "data/private_nextgqa"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", args.output_name):
        parser.error("output_basename_invalid")
    try:
        payload = build_candidate_lock(dataset_root=repo / "NExT-GQA/datasets/nextgqa",
            video_root=Path.home() / "nextgqa_sample/NExTVideo", private_root=private_root,
            split=args.split)
        write_private_exclusive(private_root / args.output_name, payload, private_root=private_root)
    except (OSError, ValueError) as exc:
        parser.error(f"candidate_lock_failed:{type(exc).__name__}")
    print(json.dumps({"status": payload["status"], "split": payload["split"],
        "candidate_count": payload["candidate_count"],
        "known_media_exclusions": payload["excluded_known_media_hash_count"],
        "duplicate_media_hashes": payload["duplicate_media_hash_count"],
        "private_output": args.output_name, "local_only": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
