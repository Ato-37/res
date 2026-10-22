#!/usr/bin/env python3
"""Post-hoc exploratory overlap of NOAA QTA temporal groups with listed evidence times.

Uses local sample manifests, annotation-audit evidence timestamps, and a
completed QTA distribution receipt. It does not open gold.json, model weights,
or media and does not use question text, answer choices, or correctness fields.
The result is a coarse QA-time alignment proxy, not claim support or explanation
faithfulness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat

ROOT = Path("datasets/noaa_qa_pilot_20261011")
RUN_REL = Path("private/noaa_qta_distribution_full_20261022_01.json")
INPUTS = (
    ("private/samples.json", "private/annotation_audit_v2.json",
     "two_annotator_passes_plus_blind_visual_auditor"),
    ("dive04/private/samples.json", "dive04/private/annotation_audit_v2.json",
     "independent_AI_blind_answer_review"),
    ("dive09/private/samples.json", "dive09/private/annotation_audit.json",
     "independent_AI_blind_answer_review"),
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_private_json(repo: Path, relative: Path) -> tuple[object, str]:
    root = (repo / ROOT).resolve(strict=True)
    path = root / relative
    if (not path.resolve(strict=True).is_relative_to(root)
            or "private" not in relative.parts):
        raise ValueError("input_path_invalid")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("private_input_file_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("private_input_file_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    return json.loads(raw.decode("utf-8")), sha256(raw)


def nearest_temporal_group(timestamp: float, sampled_pts: list[float],
                           temporal_group_count: int) -> int:
    """Map an evidence point to nearest sampled frame, then its adjacent-frame group."""
    if (not math.isfinite(timestamp) or timestamp < 0
            or not isinstance(sampled_pts, list) or len(sampled_pts) < 2
            or any(not math.isfinite(x) for x in sampled_pts)
            or sampled_pts != sorted(sampled_pts)
            or type(temporal_group_count) is not int or temporal_group_count < 1
            or len(sampled_pts) % temporal_group_count):
        raise ValueError("temporal_mapping_inputs_invalid")
    frames_per_group = len(sampled_pts) // temporal_group_count
    nearest = min(range(len(sampled_pts)), key=lambda i: (abs(sampled_pts[i] - timestamp), i))
    return nearest // frames_per_group


def validate_manifest_hashes(receipt_hashes: object, actual_hashes: dict[str, str],
                             expected_paths: set[str]) -> None:
    if (not isinstance(receipt_hashes, dict) or set(receipt_hashes) != expected_paths
            or any(receipt_hashes.get(path) != digest
                   for path, digest in actual_hashes.items())):
        raise ValueError("sample_manifest_receipt_hash_mismatch")


def validate_run_rows(run_rows: object, sample_by_key: dict,
                      expected_videos: set[str]) -> list[dict]:
    if (not isinstance(run_rows, list) or len(run_rows) != 6
            or len(sample_by_key) != 6 or len(expected_videos) != 3
            or any(not isinstance(row, dict) for row in run_rows)):
        raise ValueError("fixed_six_row_three_video_contract_failed")
    seen_keys, seen_tokens, seen_videos = set(), set(), set()
    for row in run_rows:
        key = (row.get("source_video_sha256"), row.get("qid_sha256"))
        if (key not in sample_by_key or key in seen_keys
                or row.get("row_token_sha256") != sample_by_key[key][2]
                or row.get("row_token_sha256") in seen_tokens):
            raise ValueError("qta_receipt_annotation_join_invalid")
        seen_keys.add(key)
        seen_tokens.add(row["row_token_sha256"])
        seen_videos.add(key[0])
    if seen_keys != set(sample_by_key) or seen_videos != expected_videos:
        raise ValueError("fixed_six_row_three_video_contract_failed")
    return run_rows


def validate_group_indices(qta_raw: object, sham_raw: object,
                           temporal_groups: int) -> tuple[set[int], set[int]]:
    if (not isinstance(qta_raw, list) or not isinstance(sham_raw, list)
            or any(type(i) is not int for i in qta_raw + sham_raw)):
        raise ValueError("qta_temporal_group_contract_invalid")
    qta, sham = set(qta_raw), set(sham_raw)
    if (len(qta) != len(qta_raw) or len(sham) != len(sham_raw)
            or len(qta) != 1 or len(sham) != 1 or qta & sham
            or any(i < 0 or i >= temporal_groups for i in qta | sham)):
        raise ValueError("qta_temporal_group_contract_invalid")
    return qta, sham


def summarize(repo: Path, run_path: Path) -> dict:
    root = (repo / ROOT).resolve(strict=True)
    if not run_path.is_absolute():
        run_path = repo / run_path
    if (run_path.is_symlink() or not run_path.parent.resolve(strict=True).is_relative_to(root / "private")
            or run_path.name in ("", ".", "..")):
        raise ValueError("run_receipt_must_be_noaa_private")
    fd = os.open(run_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("run_receipt_file_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        run_raw = b"".join(chunks)
    finally:
        os.close(fd)
    run = json.loads(run_raw.decode("utf-8"))
    if (run.get("status") != "completed_exploratory" or run.get("run_mode") != "full"
            or run.get("requested_rows") != 6 or run.get("effective_rows") != 6
            or run.get("attempted_forwards") != 18 or run.get("completed_forwards") != 18
            or run.get("gold_used") is not False or run.get("direct_faithfulness") is not False):
        raise ValueError("qta_receipt_contract_invalid")

    sample_by_key = {}
    input_hashes = {RUN_REL.as_posix(): sha256(run_raw)}
    manifest_hashes = run.get("sample_manifest_sha256")
    method_by_video = {}
    for sample_rel, audit_rel, method in INPUTS:
        sample, sample_hash = read_private_json(repo, Path(sample_rel))
        audit, audit_hash = read_private_json(repo, Path(audit_rel))
        input_hashes[sample_rel] = sample_hash
        input_hashes[audit_rel] = audit_hash
        if (not isinstance(sample, list) or not isinstance(audit, dict)
                or not sample
                or audit.get("media_sha256") != sample[0].get("video_sha256")
                or not isinstance(audit.get("items"), list)):
            raise ValueError("annotation_source_binding_invalid")
        annotations = {item.get("qid"): item for item in audit["items"]
                       if isinstance(item, dict) and isinstance(item.get("qid"), str)}
        if len(annotations) != len(audit["items"]):
            raise ValueError("annotation_qid_inventory_invalid")
        video_digest = audit["media_sha256"]
        method_by_video[video_digest] = method
        for row in sample:
            if (not isinstance(row, dict) or row.get("video_sha256") != video_digest
                    or row.get("qid") not in annotations):
                raise ValueError("sample_annotation_join_invalid")
            item = annotations[row["qid"]]
            evidence = item.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("annotation_evidence_missing")
            points = [entry.get("timestamp_sec") for entry in evidence
                      if isinstance(entry, dict)
                      and isinstance(entry.get("timestamp_sec"), (int, float))
                      and not isinstance(entry.get("timestamp_sec"), bool)]
            if not points:
                raise ValueError("annotation_evidence_points_missing")
            qid_hash = sha256(row["qid"].encode("utf-8"))
            key = (video_digest, qid_hash)
            if key in sample_by_key:
                raise ValueError("duplicate_sample_annotation_join")
            row_token = sha256((video_digest + "\0" + row["qid"]).encode("utf-8"))
            sample_by_key[key] = (points, method, row_token)

    actual_manifests = {sample_rel: input_hashes[sample_rel]
                        for sample_rel, _, _ in INPUTS}
    validate_manifest_hashes(manifest_hashes, actual_manifests,
        {sample_rel for sample_rel, _, _ in INPUTS})
    run_rows = validate_run_rows(run.get("rows"), sample_by_key,
        set(method_by_video))

    qta_hits = sham_hits = expected_random = 0.0
    paired = {"qta_only": 0, "sham_only": 0, "both": 0, "neither": 0}
    strata = {}
    for row in run_rows:
        key = (row["source_video_sha256"], row["qid_sha256"])
        points, method, _expected_row_token = sample_by_key[key]
        pts = row["sampled_pts_seconds"]
        grid = row.get("grid_thw")
        if not isinstance(grid, list) or len(grid) != 3 or any(type(x) is not int or x < 1 for x in grid):
            raise ValueError("qta_temporal_group_contract_invalid")
        temporal_groups = grid[0]
        qta_raw = row.get("qta_group_indices")
        sham_raw = row.get("equal_budget_hash_sham_group_indices")
        qta, sham = validate_group_indices(qta_raw, sham_raw, grid[0])
        if (row.get("frame_count") != 8 or len(pts) != 8 or type(temporal_groups) is not int
                or temporal_groups < 2 or len(pts) % temporal_groups):
            raise ValueError("qta_temporal_group_contract_invalid")
        evidence_groups = {nearest_temporal_group(float(t), pts, temporal_groups)
                           for t in points}
        q_hit, s_hit = bool(qta & evidence_groups), bool(sham & evidence_groups)
        qta_hits += int(q_hit)
        sham_hits += int(s_hit)
        expected_random += len(evidence_groups) / temporal_groups
        paired["qta_only" if q_hit and not s_hit else
               "sham_only" if s_hit and not q_hit else
               "both" if q_hit and s_hit else "neither"] += 1
        row_stratum = strata.setdefault(method, {"rows": 0, "qta_hits": 0,
            "sham_hits": 0, "uniform_random_expected_hits": 0.0})
        row_stratum["rows"] += 1
        row_stratum["qta_hits"] += int(q_hit)
        row_stratum["sham_hits"] += int(s_hit)
        row_stratum["uniform_random_expected_hits"] += len(evidence_groups) / temporal_groups

    if sum(x["rows"] for x in strata.values()) != 6:
        raise ValueError("stratified_row_count_invalid")
    return {
        "schema_version": 1,
        "purpose": "posthoc_noaa_qta_group_annotation_time_alignment_proxy",
        "status": "completed_exploratory_posthoc",
        "method": "map each listed evidence timestamp to the nearest of eight sampled frames, then to the adjacent-frame temporal token group; count a hit if the single selected group intersects any listed evidence group",
        "requested_effective_rows": [6, 6],
        "unique_video_clusters": 3,
        "temporal_groups_per_row": 4,
        "qta_annotation_group_hits": int(qta_hits),
        "sham_annotation_group_hits": int(sham_hits),
        "uniform_random_group_expected_hits": round(expected_random, 6),
        "paired_outcomes": paired,
        "annotation_method_strata": strata,
        "human_review_for_current_explanation_claims_performed": False,
        "direct_faithfulness": False,
        "gold_json_opened": False,
        "correctness_fields_used": False,
        "rights_status": "unverified; local-only under user authorization",
        "limits": [
            "post-hoc analysis of the existing six-row GPU receipt; not confirmatory",
            "timestamp annotations are question-evidence points, not reviewer-verified support spans for generated explanation claims",
            "source annotation methods differ across three videos: two-annotator visual passes for one video, independent AI blind review for two videos",
            "nearest-frame mapping and two-frame temporal grouping are coarse; no frame-level image correspondence re-review was performed",
            "six rows in three video clusters do not support a confidence interval, hypothesis test, or population claim",
            "does not measure explanation truth, answer accuracy, or direct faithfulness",
        ],
        "source_sha256": input_hashes,
        "analysis_script_sha256": sha256(Path(__file__).read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_exclusive(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return sha256(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
        help="new output file under datasets/noaa_qa_pilot_20261011/private/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = args.output
    if output.is_absolute() or ".." in output.parts:
        parser.error("output_path_invalid")
    output = (repo / output).resolve()
    if not output.is_relative_to((repo / ROOT / "private").resolve()):
        parser.error("output_must_remain_private")
    try:
        payload = summarize(repo, repo / ROOT / RUN_REL)
        digest = write_exclusive(output, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0])
    print(json.dumps({"status": payload["status"],
        "rows": payload["requested_effective_rows"][1],
        "qta_hits": payload["qta_annotation_group_hits"],
        "sham_hits": payload["sham_annotation_group_hits"],
        "output_sha256": digest,
        "output_relative_path": output.relative_to(repo).as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
