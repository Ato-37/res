#!/usr/bin/env python3
"""Freeze a local-only, outcome-blind QTA question-swap cohort.

Only video_id, qid, and question fields are accessed from test.csv. Answer,
choice, and temporal-annotation fields are not consumed. The resulting private
manifest contains identities/questions needed by the runner and must not be
published. This freezes a finite diagnostic cohort; it is not authorization or
evidence of freshness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

PROTOCOL = "qta-question-swap-repeat-controlled-v1"
N_VIDEOS = 80
VIDEO_ROOT_DEFAULT = Path.home() / "nextgqa_sample" / "NExTVideo"
PRIOR_REPEATABILITY_MANIFESTS_SHA256 = (
    "1ca111de2db7421f2f057baeeec1a43f8e6b7d6c8cfb04a17b9f246f4749f167"
)
PRIOR_REPEATABILITY_RESULT = (
    "qta_repeatability_20261018_2210/full_process_replication.json"
)


def video_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior_media_overlap(repo: Path, selected_media_hashes: set[str]) -> dict[str, Any]:
    private = repo / "data/private_nextgqa"
    specs = {
        "prior_v2_evaluation_147": private / "nextgqa_tlvs_exploratory_split_v2_20261018_05.json",
        "prior_v4_evaluation_147": private / "nextgqa_tlvs_exploratory_split_v4_union_20261018_03.json",
        "prior_visual_overlap_pool_490": private / "qta_visual_overlap_pool_freeze_20261018.json",
    }
    result: dict[str, Any] = {}
    for label, path in specs.items():
        raw = path.read_bytes()
        payload = json.loads(raw)
        if label.endswith("_147"):
            rows = payload.get("evaluation")
            if not isinstance(rows, list) or len(rows) != 147:
                raise ValueError("prior_147_evaluation_cohort_invalid")
            prior_hashes = {row.get("media_sha256") for row in rows
                            if isinstance(row, dict)}
            if len(prior_hashes) != 147 or any(not isinstance(value, str)
                                               or len(value) != 64
                                               for value in prior_hashes):
                raise ValueError("prior_147_media_identity_invalid")
        else:
            entries = payload.get("entries")
            if payload.get("expected_videos") != 490 or not isinstance(entries, list) or len(entries) != 490:
                raise ValueError("prior_490_pool_manifest_invalid")
            prior_hashes = {row.get("sha256") for row in entries
                            if isinstance(row, dict)}
            if len(prior_hashes) != 490 or any(not isinstance(value, str)
                                               or len(value) != 64
                                               for value in prior_hashes):
                raise ValueError("prior_490_media_identity_invalid")
        result[label] = {"prior_count": len(prior_hashes),
                         "overlap_count": len(selected_media_hashes & prior_hashes),
                         "source_sha256": sha256_bytes(raw)}
    return result


def csv_question_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read only the three declared columns, while hashing their canonical rows."""
    rows: list[dict[str, str]] = []
    digest = hashlib.sha256(b"qta-question-swap-selected-csv-columns-v1\0")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("test_csv_empty") from exc
        if len(header) != len(set(header)):
            raise ValueError("test_csv_duplicate_header")
        required = ("video_id", "qid", "question")
        if any(name not in header for name in required):
            raise ValueError("test_csv_required_question_columns_missing")
        indices = {name: header.index(name) for name in required}
        digest.update(json.dumps(required, separators=(",", ":")).encode("ascii"))
        for row in reader:
            if len(row) != len(header):
                raise ValueError("test_csv_row_width_invalid")
            selected = {name: row[indices[name]] for name in required}
            video_id, qid, question = (selected[name] for name in required)
            if not video_id or not qid or not question.strip():
                continue
            rows.append({"video_id": video_id, "qid": qid,
                         "question": question.strip()})
            encoded = json.dumps(selected, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return rows, digest.hexdigest()


def _read_manifests(repo: Path) -> tuple[set[str], list[dict[str, str]]]:
    excluded: set[str] = set()
    fingerprints = []
    for path in sorted((repo / "data").glob("nextgqa_samples_*.json")):
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("sample_exclusion_manifest_invalid") from exc
        if not isinstance(value, list):
            raise ValueError("sample_exclusion_manifest_schema_invalid")
        for row in value:
            if isinstance(row, dict) and isinstance(row.get("video"), str):
                excluded.add(video_key(row["video"]))
        fingerprints.append({"name": path.name, "sha256": sha256_bytes(raw)})
    return excluded, fingerprints


def prior_repeatability_exclusions(repo: Path, dataset: Path,
                                   video_root: Path) -> tuple[set[str], dict[str, Any]]:
    """Recreate the exact prior three-video selector cohort for exclusion.

    This historical-only reconstruction calls the previous candidate rule,
    which used annotation validity to identify its cohort. Those annotations
    are used solely to exclude known old inputs, never for new eligibility,
    question pairing, scoring, or interpretation.
    """
    current_manifest_digest = hashlib.sha256(json.dumps(
        [(path.name, sha256_file(path)) for path in sorted(
            (repo / "data").glob("nextgqa_samples_*.json"))],
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if current_manifest_digest != PRIOR_REPEATABILITY_MANIFESTS_SHA256:
        raise ValueError("prior_repeatability_manifest_set_changed")
    if video_root.resolve(strict=True) != VIDEO_ROOT_DEFAULT.resolve(strict=True):
        raise ValueError("prior_repeatability_video_root_changed")
    import sys
    scripts = str(repo / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_nextgqa_real_video_gpu_smoke import choose_videos
    candidates = choose_videos(repo, dataset)
    if len(candidates) < 3:
        raise ValueError("prior_repeatability_candidates_unavailable")
    candidate_ranks = [hashlib.sha256(path.name.encode()).hexdigest()
                       for path in candidates]
    if len(candidate_ranks) != len(set(candidate_ranks)):
        raise ValueError("historical_candidate_hash_order_tie")
    prior_path = repo / "data/private_nextgqa" / PRIOR_REPEATABILITY_RESULT
    prior_raw = prior_path.read_bytes()
    prior = json.loads(prior_raw)
    if (prior.get("status") != "passed"
            or prior.get("candidate_exclusion_manifests_sha256") != current_manifest_digest
            or prior.get("source_sha256", {}).get("candidate_selector") != sha256_file(
                repo / "scripts/run_nextgqa_real_video_gpu_smoke.py")):
        raise ValueError("historical_repeatability_source_binding_mismatch")
    expected_dataset_hashes = prior.get("dataset_input_sha256")
    current_dataset_hashes = {
        "test.csv": sha256_file(dataset / "test.csv"),
        "gsub_test.json": sha256_file(dataset / "gsub_test.json"),
        "map_vid_vidorID.json": sha256_file(dataset / "map_vid_vidorID.json"),
    }
    if expected_dataset_hashes != current_dataset_hashes:
        raise ValueError("historical_repeatability_dataset_binding_mismatch")
    previous_video_digest = hashlib.sha256("".join(sorted(
        sha256_file(path) for path in candidates[:3])).encode("ascii")).hexdigest()
    if previous_video_digest != prior.get("input_hash_aggregates", {}).get(
            "source_video_sha256"):
        raise ValueError("historical_repeatability_media_binding_mismatch")
    keys = {video_key(path.relative_to(video_root).as_posix())
            for path in candidates[:3]}
    if len(keys) != 3:
        raise ValueError("prior_repeatability_candidate_identity_invalid")
    return keys, {
        "method": "reconstructed the exact first three candidates of the prior deterministic QTA repeatability/wiring selector; historical annotation validity used only to exclude those inputs",
        "historical_candidate_selector_sha256": sha256_file(
            repo / "scripts/run_nextgqa_real_video_gpu_smoke.py"),
        "historical_exclusion_manifest_set_sha256": current_manifest_digest,
        "historical_candidate_count": len(candidates),
        "excluded_prior_video_count": len(keys),
        "historical_result_sha256": sha256_bytes(prior_raw),
        "historical_dataset_source_bytes_verified": True,
        "historical_media_content_aggregate_verified": True,
        "historical_candidate_order_tie_count": 0,
    }


def _pair_rank(video_id: str, left: dict[str, str], right: dict[str, str]) -> tuple[str, str, str]:
    ordered = sorted((left, right), key=lambda row: (row["qid"], row["question"]))
    payload = [PROTOCOL, "question-pair-v1", video_id,
               ordered[0]["qid"], ordered[1]["qid"]]
    rank = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return rank, ordered[0]["qid"], ordered[1]["qid"]


def _video_rank(video_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}\0video\0{video_id}".encode("utf-8")).hexdigest()


def build_plan(*, repo: Path, dataset: Path, video_root: Path,
               requested_videos: int = N_VIDEOS,
               exclude_keys: set[str] | None = None,
               prior_exclusion_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    if type(requested_videos) is not int or requested_videos < 1:
        raise ValueError("requested_video_count_invalid")
    video_root = video_root.resolve(strict=True)
    if not video_root.is_dir():
        raise ValueError("video_root_invalid")
    rows, selected_csv_sha = csv_question_rows(dataset / "test.csv")
    mapping_raw = (dataset / "map_vid_vidorID.json").read_bytes()
    mapping = json.loads(mapping_raw)
    if not isinstance(mapping, dict):
        raise ValueError("video_mapping_schema_invalid")
    path_to_id = {}
    for video_id, relative in mapping.items():
        if isinstance(video_id, str) and isinstance(relative, str):
            key = video_key(relative)
            if key in path_to_id:
                raise ValueError("video_mapping_key_ambiguous")
            path_to_id[key] = video_id
    local_by_key: dict[str, list[Path]] = {}
    for path in video_root.rglob("*.mp4"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("video_tree_contains_symlink_or_nonfile")
        key = video_key(path.relative_to(video_root).as_posix())
        local_by_key.setdefault(key, []).append(path)
    excluded, manifest_fingerprints = _read_manifests(repo)
    explicit_exclusions = set(exclude_keys or ())
    excluded |= explicit_exclusions

    by_video: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)
    candidates = []
    for key, video_id in path_to_id.items():
        files = local_by_key.get(key, [])
        if len(files) != 1 or key in excluded:
            continue
        questions = by_video.get(video_id, [])
        unique_questions: dict[str, dict[str, str]] = {}
        for row in questions:
            old = unique_questions.get(row["question"])
            if old is None or row["qid"] < old["qid"]:
                unique_questions[row["question"]] = row
        distinct = list(unique_questions.values())
        if len(distinct) < 2:
            continue
        ranked_pairs = []
        for i, left in enumerate(distinct):
            for right in distinct[i + 1:]:
                ranked_pairs.append((_pair_rank(video_id, left, right), left, right))
        ranked_pairs.sort(key=lambda item: item[0])
        _, question_a, question_b = ranked_pairs[0]
        candidates.append(((_video_rank(video_id), video_id), files[0],
                           question_a, question_b, len(distinct), len(ranked_pairs)))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < requested_videos:
        raise ValueError("eligible_video_count_below_requested_no_replacement")
    selected = candidates[:requested_videos]
    records = []
    selected_media_hashes = set()
    for ordinal, (_, path, question_a, question_b, question_count, pair_count) in enumerate(selected, 1):
        video_id = question_a["video_id"]
        media_hash = sha256_file(path)
        if media_hash in selected_media_hashes:
            raise ValueError("selected_video_media_bytes_not_unique")
        selected_media_hashes.add(media_hash)
        records.append({
            "fixed_video_ordinal": ordinal,
            "video_id": video_id,
            "video_relative_path": path.relative_to(video_root).as_posix(),
            "media_sha256": media_hash,
            "question_a": question_a,
            "question_b": question_b,
            "distinct_question_count": question_count,
            "distinct_pair_count": pair_count,
        })
    selected_hash_set = {row["media_sha256"] for row in records}
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "frozen_local_question_swap_selector_diagnostic_not_faithfulness",
        "authorization": "non_authorizing_local_finite_cohort_diagnostic",
        "dataset_split": "NExT-GQA test; test-associated and previously studied population",
        "selection": "hash-rank eligible local videos; exclude registered sample manifests and explicit prior-QTA media; within each video hash-rank pairs of distinct exact question strings using video_id/qid only; no replacement",
        "new_question_eligibility_and_pair_selection_use_only_question_columns": True,
        "historical_repeatability_exclusion_used_annotation_validity":
            prior_exclusion_provenance is not None,
        "requested_videos": requested_videos,
        "eligible_videos_before_sample": len(candidates),
        "effective_videos": len(records),
        "source_fingerprints": {
            "test_csv_selected_columns_sha256": selected_csv_sha,
            "video_mapping_sha256": sha256_bytes(mapping_raw),
            "sample_manifest_count": len(manifest_fingerprints),
            "sample_manifest_fingerprint_sha256": sha256_bytes(json.dumps(
                manifest_fingerprints, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")),
            "explicit_prior_capture_exclusion_count": len(explicit_exclusions),
            "explicit_prior_capture_exclusion_sha256": sha256_bytes(json.dumps(
                sorted(explicit_exclusions), separators=(",", ":")
            ).encode("utf-8")),
        },
        "prior_capture_exclusion_provenance": prior_exclusion_provenance,
        "video_root": str(video_root),
        "explicit_prior_capture_excluded_video_keys": sorted(explicit_exclusions),
        "fixed_records": records,
        "prior_media_overlap_audit": _prior_media_overlap(repo, selected_hash_set),
        "limitations": [
            "not an independent/fresh cohort; prior exposure and source/event independence are unresolved",
            "question text variation includes wording, length, tokenization, and semantic differences",
            "selector change does not establish relevance, grounding, answer dependence, or faithfulness",
            "all identities, questions, paths, and media hashes are protected local-only inputs",
        ],
    }


def write_private_exclusive(path: Path, private_root: Path, value: dict[str, Any]) -> None:
    root = private_root.resolve(strict=True)
    if (path.parent.resolve(strict=True) != root or not stat.S_ISDIR(root.stat().st_mode)
            or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) != 0o700):
        raise ValueError("private_output_root_invalid")
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT_DEFAULT)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--count", type=int, default=N_VIDEOS)
    parser.add_argument("--exclude-video-key", action="append", default=[],
                        help="private relative video key excluded from known prior QTA captures; repeat as needed")
    args = parser.parse_args(argv)
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    repo = args.repo.resolve()
    private = repo / "data/private_nextgqa"
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    try:
        explicit_exclusions, prior_provenance = prior_repeatability_exclusions(
            repo, dataset, args.video_root)
        plan = build_plan(repo=repo, dataset=dataset, video_root=args.video_root,
                          requested_videos=args.count,
                          exclude_keys=explicit_exclusions | set(args.exclude_video_key),
                          prior_exclusion_provenance=prior_provenance)
        write_private_exclusive(private / args.output_name, private, plan)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "failure_code": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({key: plan[key] for key in
        ("status", "requested_videos", "eligible_videos_before_sample", "effective_videos",
         "new_question_eligibility_and_pair_selection_use_only_question_columns",
         "historical_repeatability_exclusion_used_annotation_validity")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
