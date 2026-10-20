#!/usr/bin/env python3
"""Freeze a small, outcome-blind claim temporal-order cohort locally.

The source pool contains only candidate-pair hashes and local videos. This
freezer selects distinct eligible videos before generation/scoring, records
only media/QID hashes, and refuses overlap with prior scored claim packets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from private_output_paths import validate_private_output
import run_nextgqa_gold_interval_margin_pilot as pilot


def _write_exclusive(path: Path, payload: dict) -> None:
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _scored_video_digests(private_root: Path) -> set[str]:
    """Read only explicit video digests from model-output artifact families."""
    prefixes = ("claim_generation_", "claim_review_", "saved_claim_",
                "benchmark_interval_claim_cohort_gpu_", "gold_interval_cohort_")
    output = set()
    for path in private_root.rglob("*.json"):
        if not path.name.startswith(prefixes):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if (key in ("video_sha256", "source_video_sha256")
                            and isinstance(item, str) and len(item) == 64):
                        output.add(item)
                    elif isinstance(item, (dict, list)):
                        stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
    return output


def freeze(*, repo: Path, pool_rel: Path, video_root: Path,
           lock_rel: Path, pair_filter_rel: Path, n: int = 3) -> dict:
    if n != 3:
        raise ValueError("frozen_cohort_requires_three_rows")
    private_root = (repo / "data" / "private_nextgqa").resolve(strict=True)
    pool = (repo / pool_rel).resolve(strict=True)
    if not pool.is_relative_to(private_root):
        raise ValueError("candidate_pool_outside_private_root")
    summary_path = pool / "summary.json"
    pairs_path = pool / "eligible_pairs.json"
    exclusion_path = pool / "prior_media_sha256_exclusions.json"
    for path in (summary_path, pairs_path, exclusion_path):
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("candidate_pool_private_file_permissions_invalid")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pair_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    if (summary.get("purpose") != "outcome_blind_local_candidate_pool"
            or summary.get("dataset_split") != "test"
            or summary.get("extracted_unique_media_count") != 20):
        raise ValueError("candidate_pool_contract_invalid")
    pair_hashes = pilot.validate_candidate_pair_filter(pair_payload)
    exclusions = pilot.validate_private_media_exclusion(exclusion_payload)
    pilot.MAX_VIDEOS = n
    selected, counts, meta, cohort_hash = pilot.freeze_cohort(
        repo, video_root, exclusions, 16,
        allowed_pair_sha256=pair_hashes, dataset_split="test")
    if len(selected) != n or len({x["decode"]["source_sha256"] for x in selected}) != n:
        raise ValueError("frozen_cohort_not_three_unique_eligible_videos")
    scored = _scored_video_digests(private_root)
    media = {x["decode"]["source_sha256"] for x in selected}
    overlap = media & scored
    if overlap:
        raise ValueError("frozen_cohort_overlaps_local_scored_claim_artifacts")
    assignments = pilot.build_cohort_lock_assignments(selected)
    selected_pairs = [pilot.candidate_pair_sha256(x["video_id"], x["qid"])
                      for x in selected]
    lock = {
        "schema_version": 1,
        "status": "frozen_before_claim_generation_and_scoring",
        "purpose": "exploratory_saved_claim_temporal_order_bias_interaction",
        "dataset_split": "test",
        "requested_videos": n,
        "effective_videos": len(selected),
        "selection_rule": "first three eligible unique media in frozen candidate loader order; no model outcomes used",
        "candidate_pool_summary_sha256": pilot.sha256_file(summary_path),
        "candidate_pair_filter_sha256": pilot.sha256_file(pairs_path),
        "prior_media_exclusion_sha256": pilot.sha256_file(exclusion_path),
        "candidate_pairs_available": len(pair_hashes),
        "prior_media_exclusion_count": len(exclusions),
        "prior_scored_claim_digest_count": len(scored),
        "local_scored_claim_overlap_count": 0,
        "candidate_videos_scanned": counts["candidate_scan_count"],
        "cohort_sha256": cohort_hash,
        "assignments": assignments,
        "source_sha256": {
            "scripts/freeze_temporal_order_claim_cohort.py": pilot.sha256_file(Path(__file__).resolve()),
            "scripts/run_nextgqa_gold_interval_margin_pilot.py": pilot.sha256_file(repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py"),
        },
        "limitations": [
            "candidate media appear in prior split/candidate metadata; complete exposure history is unavailable",
            "no matching video digest was found in current local claim-generation/scoring output families",
            "selection is exploratory and does not certify a fresh confirmatory cohort",
        ],
    }
    pair_filter = {"schema_version": 1, "pair_sha256": selected_pairs}
    lock_path = validate_private_output(str(lock_rel), repo)
    filter_path = validate_private_output(str(pair_filter_rel), repo)
    if lock_path == filter_path:
        raise ValueError("private_output_paths_must_differ")
    _write_exclusive(lock_path, lock)
    try:
        _write_exclusive(filter_path, pair_filter)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return {"effective_videos": len(selected), "cohort_sha256": cohort_hash,
            "local_scored_claim_overlap_count": 0,
            "candidate_pairs_available": len(pair_hashes),
            "candidate_videos_scanned": counts["candidate_scan_count"],
            "lock_name": lock_path.name, "pair_filter_name": filter_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True,
        help="pool directory relative to data/private_nextgqa/")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--lock-output", required=True)
    parser.add_argument("--pair-filter-output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pool_rel = Path("data/private_nextgqa") / args.pool
    result = freeze(repo=repo, pool_rel=pool_rel, video_root=args.video_root,
        lock_rel=Path(args.lock_output), pair_filter_rel=Path(args.pair_filter_output))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
