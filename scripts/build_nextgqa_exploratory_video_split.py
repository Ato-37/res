"""Build a private, deterministic video-disjoint calibration/evaluation plan.

This is an exploratory split inside the NExT-GQA test-associated media pool,
not an official train/test split or confirmatory evaluation. It reads metadata
and hashes local media; it does not decode videos, use or emit QA text, or run a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = "nextgqa_tlvs_inspired_exploratory_video_split_v1"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_exclusions(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    video_ids: set[str] = set()
    video_stems: set[str] = set()
    fingerprints = []
    for path in sorted(paths):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("prior_manifest_root_invalid")
        rows = 0
        for row in value:
            if not isinstance(row, dict):
                continue
            rows += 1
            if row.get("video_id") is not None:
                video_ids.add(str(row["video_id"]))
            video = row.get("video")
            if isinstance(video, str):
                video_stems.add(Path(video).stem)
        fingerprints.append({"sha256": sha256_file(path), "row_count": rows})
    return video_ids, video_stems, fingerprints


def build_split(
    *, test_csv: Path, video_root: Path, prior_manifests: list[Path],
    prior_video_roots: list[Path] | None = None,
    calibration_fraction: float = 0.70,
) -> dict[str, Any]:
    if not 0.5 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction_out_of_range")
    if not test_csv.is_file() or not video_root.is_dir():
        raise ValueError("input_path_unavailable")

    rows_by_video: dict[str, list[str]] = defaultdict(list)
    test_csv_row_count = 0
    with test_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "qid"}.issubset(reader.fieldnames or ()):
            raise ValueError("test_csv_schema_invalid")
        for row in reader:
            video_id, qid = row.get("video_id"), row.get("qid")
            if not video_id or not qid:
                raise ValueError("test_csv_identifier_missing")
            test_csv_row_count += 1
            rows_by_video[str(video_id)].append(str(qid))

    media_by_id: dict[str, Path] = {}
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(video_root.resolve()):
            raise ValueError("video_path_escapes_root")
        video_id = path.stem
        if video_id in media_by_id:
            raise ValueError("duplicate_media_for_video_id")
        media_by_id[video_id] = path

    prior_ids, prior_stems, prior_hashes = _manifest_exclusions(prior_manifests)
    prior_media_hashes: set[str] = set()
    prior_media_file_count = 0
    for prior_root in prior_video_roots or []:
        if not prior_root.is_dir():
            continue
        for path in sorted(prior_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            resolved = path.resolve()
            # The candidate tree is scanned separately and must not double as
            # its own prior-media exclusion set.
            if resolved.is_relative_to(video_root.resolve()):
                continue
            if not resolved.is_relative_to(prior_root.resolve()):
                raise ValueError("prior_video_path_escapes_root")
            prior_media_file_count += 1
            prior_media_hashes.add(sha256_file(path))
    candidates = []
    eligible_qid_count = 0
    excluded_prior_id = 0
    excluded_prior_stem = 0
    excluded_prior_exact_media = 0
    missing_annotation = 0
    for video_id, path in sorted(media_by_id.items()):
        if video_id in prior_ids:
            excluded_prior_id += 1
            continue
        if path.stem in prior_stems:
            excluded_prior_stem += 1
            continue
        if video_id not in rows_by_video:
            missing_annotation += 1
            continue
        media_sha = sha256_file(path)
        if media_sha in prior_media_hashes:
            excluded_prior_exact_media += 1
            continue
        qids = sorted(set(rows_by_video[video_id]), key=lambda qid: hashlib.sha256(
            f"{PROTOCOL}\0qid\0{video_id}\0{qid}".encode("utf-8")).hexdigest())
        eligible_qid_count += len(qids)
        candidates.append({
            "video_id": video_id,
            "media_path": str(path),
            "media_sha256": media_sha,
            "qid": qids[0],
            "available_test_qids": len(set(rows_by_video[video_id])),
        })
    if len(candidates) < 2:
        raise ValueError("insufficient_disjoint_candidates")

    # Group by exact media bytes before assignment so copied files cannot land
    # in both pools. Hash-order assignment is stable for a frozen inventory.
    media_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        media_groups[row["media_sha256"]].append(row)
    ordered_groups = sorted(media_groups.items(), key=lambda item: hashlib.sha256(
        f"{PROTOCOL}\0group\0{item[0]}".encode("ascii")).hexdigest())
    calibration_n = int(len(ordered_groups) * calibration_fraction)
    if calibration_n <= 0 or calibration_n >= len(ordered_groups):
        raise ValueError("split_group_count_invalid")
    calibration_hashes = {digest for digest, _ in ordered_groups[:calibration_n]}
    calibration, evaluation = [], []
    for row in candidates:
        (calibration if row["media_sha256"] in calibration_hashes else evaluation).append(row)
    calibration.sort(key=lambda row: row["video_id"])
    evaluation.sort(key=lambda row: row["video_id"])
    if ({row["media_sha256"] for row in calibration}
            & {row["media_sha256"] for row in evaluation}):
        raise ValueError("exact_media_overlap_after_split")
    if ({row["video_id"] for row in calibration}
            & {row["video_id"] for row in evaluation}):
        raise ValueError("video_id_overlap_after_split")

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "exploratory_split_plan_not_evaluation",
        "cohort": "NExT-GQA official test-associated local media; not confirmatory",
        "split": {
            "algorithm": "sort by SHA256(protocol_tag || NUL || 'group' || NUL || exact_media_sha256)",
            "calibration_fraction": calibration_fraction,
            "requested_calibration_groups": calibration_n,
            "effective_calibration_videos": len(calibration),
            "effective_evaluation_videos": len(evaluation),
            "unit": "one video with one deterministically selected test qid",
            "selected_qid_policy": "per video, minimum SHA256(protocol_tag || NUL || 'qid' || NUL || video_id || NUL || qid)",
            "eligible_qid_count": eligible_qid_count,
            "selected_qid_count": len(candidates),
            "unselected_qid_count": eligible_qid_count - len(candidates),
            "video_id_overlap": 0,
            "exact_media_sha256_overlap": 0,
        },
        "provenance": {
            "test_csv_sha256": sha256_file(test_csv),
            "test_csv_row_count": test_csv_row_count,
            "test_csv_unique_video_id_count": len(rows_by_video),
            "candidate_media_file_count_before_exclusions": len(media_by_id),
            "eligible_candidate_video_count_after_exclusions": len(candidates),
            "prior_manifest_count": len(prior_hashes),
            "prior_manifest_row_count": sum(item["row_count"] for item in prior_hashes),
            "prior_manifest_unique_video_id_count": len(prior_ids),
            "prior_manifest_unique_video_path_stem_count": len(prior_stems),
            "prior_media_file_count": prior_media_file_count,
            "prior_media_unique_sha256_count": len(prior_media_hashes),
            "excluded_by_prior_video_id": excluded_prior_id,
            "excluded_by_prior_path_stem": excluded_prior_stem,
            "excluded_by_prior_exact_media_hash": excluded_prior_exact_media,
            "excluded_without_test_annotation": missing_annotation,
            "prior_manifest_sha256_and_rows": prior_hashes,
            "question_text_or_answer_text_included": False,
            "media_content_opened_or_decoded": False,
            "source_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "calibration": calibration,
        "evaluation": evaluation,
        "limitations": [
            "Calibration and evaluation are disjoint by video and exact media hash, but both come from the official test split.",
            "This exploratory allocation is not a confirmatory test or evidence of faithfulness.",
            "Exact hashes do not detect re-encodes or near-duplicate scenes.",
            "Raw identifiers, local paths, and media hashes are private and must remain local.",
        ],
    }


def write_private_exclusive(
    path: Path, value: dict[str, Any], *, private_root: Path,
) -> None:
    if not path.resolve().is_relative_to(private_root.resolve()):
        raise ValueError("output_must_remain_in_private_root")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".split_plan_tmp_", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path,
                        default=Path("NExT-GQA/datasets/nextgqa/test.csv"))
    parser.add_argument("--video-root", type=Path,
                        default=Path("data/private_nextgqa/zip_gold_candidates_20261017_01"))
    parser.add_argument("--manifest-glob", default="data/nextgqa_samples*.json")
    parser.add_argument("--prior-video-root", type=Path, action="append",
                        default=[Path("data/private_nextgqa")])
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    parser.add_argument("--out", type=Path,
                        default=Path("data/private_nextgqa/nextgqa_tlvs_exploratory_split_v1.json"))
    args = parser.parse_args()
    manifests = [Path(value) for value in sorted(Path(".").glob(args.manifest_glob))]
    plan = build_split(test_csv=args.test_csv, video_root=args.video_root,
                       prior_manifests=manifests,
                       prior_video_roots=args.prior_video_root,
                       calibration_fraction=args.calibration_fraction)
    write_private_exclusive(args.out, plan, private_root=Path("data/private_nextgqa"))
    print(json.dumps({
        "status": plan["status"],
        "candidate_videos": plan["provenance"]["eligible_candidate_video_count_after_exclusions"],
        "calibration_videos": plan["split"]["effective_calibration_videos"],
        "evaluation_videos": plan["split"]["effective_evaluation_videos"],
        "prior_manifest_id_overlap_exclusions": plan["provenance"]["excluded_by_prior_video_id"],
        "prior_exact_media_exclusions": plan["provenance"]["excluded_by_prior_exact_media_hash"],
        "exact_media_overlap": plan["split"]["exact_media_sha256_overlap"],
        "output": str(args.out),
        "privacy": "local-only; contains identifiers, paths, and media hashes",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
