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
    """Conservatively find explicit video digests in local scoring outputs."""
    prefixes = ("claim_generation_", "claim_review_", "saved_claim_",
                "benchmark_interval_claim_cohort_gpu_", "gold_interval_cohort_")
    output = set()

    def carries_score_marker(value) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "completed_forwards" and type(item) is int and item > 0):
                    return True
                if key in {"scoring_performed", "target_scoring_performed",
                           "faithfulness_scored"} and item is True:
                    return True
                if isinstance(item, (dict, list)) and carries_score_marker(item):
                    return True
        elif isinstance(value, list):
            return any(carries_score_marker(item) for item in value)
        return False

    for path in private_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        marker = carries_score_marker(payload)
        scan_output_family = path.name.startswith(prefixes)
        purpose = (payload.get("purpose", "")
                   if isinstance(payload, dict) else "")
        # Generation packets are prior model exposure even before answer
        # scoring. Include them by their schema purpose, not only by their
        # filename: several valid packets use task-specific prefixes such as
        # ``temporal_order_claim_generation_*``.
        claim_generation = (isinstance(purpose, str)
                            and ("claim_generation" in purpose
                                 or "generated_claim" in purpose
                                 or "claim_review_pilot" in purpose))
        if not marker and not scan_output_family and not claim_generation:
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


def _selection_exclusions(source_exclusions, scored_digests: set[str]) -> set[str]:
    return set(source_exclusions) | set(scored_digests)


def _read_exposure_manifest(repo: Path, relative: Path) -> tuple[set[str], str, dict]:
    """Verify a frozen local generation/scoring union and its source hashes."""
    private_root = (repo / "data" / "private_nextgqa").resolve(strict=True)
    if (relative.is_absolute() or not relative.is_relative_to(Path("data/private_nextgqa"))):
        raise ValueError("claim_exposure_manifest_path_invalid")
    path = (repo / relative).resolve(strict=True)
    if not path.is_relative_to(private_root):
        raise ValueError("claim_exposure_manifest_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("claim_exposure_manifest_permissions_invalid")
        blocks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
    finally:
        os.close(fd)
    manifest = json.loads(raw.decode("utf-8"))
    sources = manifest.get("sources")
    media = manifest.get("media_sha256")
    if (manifest.get("schema_version") != 1
            or manifest.get("purpose") != "local_claim_exposure_union_manifest"
            or not isinstance(sources, list) or not isinstance(media, list)
            or manifest.get("source_file_count") != len(sources)
            or manifest.get("unique_media_count") != len(media)
            or media != sorted(set(media))
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value) for value in media)):
        raise ValueError("claim_exposure_manifest_contract_invalid")
    inventory_raw = json.dumps(sources, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
    inventory_hash = hashlib.sha256(inventory_raw).hexdigest()
    if manifest.get("source_inventory_sha256") != inventory_hash:
        raise ValueError("claim_exposure_manifest_inventory_hash_mismatch")
    for source in sources:
        if (not isinstance(source, dict)
                or set(source) != {"relative_filename", "source_sha256", "role"}
                or not isinstance(source.get("relative_filename"), str)
                or not isinstance(source.get("source_sha256"), str)
                or len(source["source_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["source_sha256"])
                or source.get("role") not in {"claim_generation", "scoring_receipt"}):
            raise ValueError("claim_exposure_manifest_source_invalid")
        source_rel = Path(source["relative_filename"])
        if (source_rel.is_absolute() or not source_rel.is_relative_to(private_root.relative_to(repo))):
            raise ValueError("claim_exposure_manifest_source_path_invalid")
        source_entry = repo / source_rel
        source_entry_info = source_entry.lstat()
        if stat.S_ISLNK(source_entry_info.st_mode):
            raise ValueError("claim_exposure_manifest_source_path_invalid")
        source_path = source_entry.resolve(strict=True)
        if not source_path.is_relative_to(private_root):
            raise ValueError("claim_exposure_manifest_source_path_invalid")
        source_info = source_path.stat()
        if (not stat.S_ISREG(source_info.st_mode) or source_info.st_uid != os.getuid()
                or stat.S_IMODE(source_info.st_mode) != 0o600
                or pilot.sha256_file(source_path) != source.get("source_sha256")):
            raise ValueError("claim_exposure_manifest_source_hash_mismatch")
    return set(media), hashlib.sha256(raw).hexdigest(), manifest


def _private_json_sha256(payload: dict) -> str:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze(*, repo: Path, pool_rel: Path | None, video_root: Path,
           lock_rel: Path, pair_filter_rel: Path, n: int = 3,
           dataset_split: str = "test",
           purpose: str = "exploratory_saved_claim_temporal_order_bias_interaction",
           exposure_manifest_rel: Path | None = None) -> dict:
    if n != 3:
        raise ValueError("frozen_cohort_requires_three_rows")
    if dataset_split not in ("val", "test"):
        raise ValueError("candidate_pool_split_invalid")
    if purpose not in {
            "exploratory_saved_claim_temporal_order_bias_interaction",
            "exploratory_saved_claim_video_content_frame_mean_split_check",
            "exploratory_claim_likelihood_selected_window_prediction_dependence"}:
        raise ValueError("cohort_purpose_invalid")
    private_root = (repo / "data" / "private_nextgqa").resolve(strict=True)
    pool_summary_hash = pair_filter_hash = prior_media_exclusion_hash = None
    exclusions: set[str] = set()
    pair_hashes = None
    if pool_rel is not None:
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
                or summary.get("dataset_split") != dataset_split
                or not isinstance(summary.get("extracted_unique_media_count"), int)
                or summary.get("extracted_unique_media_count") < n):
            raise ValueError("candidate_pool_contract_invalid")
        pair_hashes = pilot.validate_candidate_pair_filter(pair_payload)
        exclusions = set(pilot.validate_private_media_exclusion(exclusion_payload))
        pool_summary_hash = pilot.sha256_file(summary_path)
        pair_filter_hash = pilot.sha256_file(pairs_path)
        prior_media_exclusion_hash = pilot.sha256_file(exclusion_path)
    # Extend the source-provided exclusion set with every locally scored claim
    # artifact before selection. This makes successive exploratory cohorts
    # actually disjoint instead of selecting the original first rows and only
    # discovering overlap afterward.
    scored = _scored_video_digests(private_root)
    exposure_manifest_sha256 = exposure_inventory_sha256 = None
    exposure_source_count = exposure_media_count = None
    if exposure_manifest_rel is not None:
        exposure_media, exposure_manifest_sha256, exposure_manifest = \
            _read_exposure_manifest(repo, exposure_manifest_rel)
        scored.update(exposure_media)
        exposure_inventory_sha256 = exposure_manifest["source_inventory_sha256"]
        exposure_source_count = exposure_manifest["source_file_count"]
        exposure_media_count = exposure_manifest["unique_media_count"]
    selection_exclusions = _selection_exclusions(exclusions, scored)
    pilot.MAX_VIDEOS = n
    selected, counts, meta, cohort_hash = pilot.freeze_cohort(
        repo, video_root, selection_exclusions, 16,
        allowed_pair_sha256=pair_hashes, dataset_split=dataset_split)
    if len(selected) != n or len({x["decode"]["source_sha256"] for x in selected}) != n:
        raise ValueError("frozen_cohort_not_three_unique_eligible_videos")
    media = {x["decode"]["source_sha256"] for x in selected}
    overlap = media & scored
    if overlap:
        raise ValueError("frozen_cohort_overlaps_local_scored_claim_artifacts")
    assignments = pilot.build_cohort_lock_assignments(selected)
    selected_pairs = [pilot.candidate_pair_sha256(x["video_id"], x["qid"])
                      for x in selected]
    pair_filter = {"schema_version": 1, "pair_sha256": selected_pairs}
    lock = {
        "schema_version": 1,
        "status": "frozen_before_claim_generation_and_scoring",
        "purpose": purpose,
        "dataset_split": dataset_split,
        "requested_videos": n,
        "effective_videos": len(selected),
        "selection_rule": "first three eligible unique media in frozen candidate loader order after unioning source exclusions with all locally scored claim media; no new-cohort outcomes used",
        "candidate_pool_summary_sha256": pool_summary_hash,
        "candidate_pair_filter_sha256": pair_filter_hash,
        "selected_pair_filter_sha256": _private_json_sha256(pair_filter),
        "prior_media_exclusion_sha256": prior_media_exclusion_hash,
        "candidate_pairs_available": len(pair_hashes) if pair_hashes is not None else None,
        "full_split_candidate_video_count": meta.get("candidate_video_count"),
        "dataset_input_sha256": {name: pilot.sha256_file(repo / "NExT-GQA" / "datasets" / "nextgqa" / name)
            for name in (f"{dataset_split}.csv", f"gsub_{dataset_split}.json", "map_vid_vidorID.json")},
        "local_sample_exclusion_manifest_count": meta.get("private_exclusion_manifest_count"),
        "local_sample_exclusion_manifest_digest": meta.get("private_exclusion_manifest_digest"),
        "prior_media_exclusion_count": len(exclusions),
        "prior_scored_claim_digest_count": len(scored),
        "prior_scored_claim_digest_set_sha256": hashlib.sha256(json.dumps(
            sorted(scored), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "local_claim_exposure_manifest_sha256": exposure_manifest_sha256,
        "local_claim_exposure_source_inventory_sha256": exposure_inventory_sha256,
        "local_claim_exposure_source_file_count": exposure_source_count,
        "local_claim_exposure_media_hash_count": exposure_media_count,
        "local_scored_claim_overlap_count": 0,
        "candidate_videos_scanned": counts["candidate_scan_count"],
        "cohort_sha256": cohort_hash,
        "assignments": assignments,
        "source_sha256": {
            "scripts/freeze_temporal_order_claim_cohort.py": pilot.sha256_file(Path(__file__).resolve()),
            "scripts/run_nextgqa_gold_interval_margin_pilot.py": pilot.sha256_file(repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py"),
            "scripts/build_claim_exposure_union_manifest.py": pilot.sha256_file(repo / "scripts/build_claim_exposure_union_manifest.py"),
        },
        "limitations": [
            "candidate media appear in prior split/candidate metadata; complete exposure history is unavailable",
            "no matching video digest was found in current local claim-generation/scoring output families",
            "selection is exploratory and does not certify a fresh confirmatory cohort",
        ],
    }
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
            "selected_pair_filter_sha256": lock["selected_pair_filter_sha256"],
            "candidate_pairs_available": len(pair_hashes) if pair_hashes is not None else None,
            "candidate_videos_scanned": counts["candidate_scan_count"],
            "lock_name": lock_path.name, "pair_filter_name": filter_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=None,
        help="optional pool directory relative to data/private_nextgqa; omit to scan the full local split")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--lock-output", required=True)
    parser.add_argument("--pair-filter-output", required=True)
    parser.add_argument("--claim-exposure-manifest", type=Path, default=None,
        help="optional private generation/scoring union manifest, relative to repository root")
    parser.add_argument("--dataset-split", choices=("val", "test"), default="test")
    parser.add_argument("--purpose", choices=(
        "exploratory_saved_claim_temporal_order_bias_interaction",
        "exploratory_saved_claim_video_content_frame_mean_split_check",
        "exploratory_claim_likelihood_selected_window_prediction_dependence"),
        default="exploratory_saved_claim_temporal_order_bias_interaction")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pool_rel = Path("data/private_nextgqa") / args.pool if args.pool is not None else None
    result = freeze(repo=repo, pool_rel=pool_rel, video_root=args.video_root,
        lock_rel=Path(args.lock_output), pair_filter_rel=Path(args.pair_filter_output),
        dataset_split=args.dataset_split, purpose=args.purpose,
        exposure_manifest_rel=args.claim_exposure_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
