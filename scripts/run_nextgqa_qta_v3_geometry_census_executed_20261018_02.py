#!/usr/bin/env python3
"""Outcome-blind CPU census of feasible QTA annotation QIDs for v3.

This uses the frozen v1/v2 video membership, considers every interval-valid
test.csv QID for each evaluation video, and selects no outcomes. A private v3
plan is written only if all frozen evaluation videos have at least one QID
whose processor geometry supports one evidence anchor and one disjoint
uncovered control. No model weights are loaded and CUDA is never called.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any

import run_nextgqa_validation_qta_annotation_proxy as kernel
from build_nextgqa_exploratory_video_split_v2 import valid_spans
from run_nextgqa_qta_exploratory_proxy_v2 import (
    DATA_REL, PRIVATE_REL, POLICY, build_evaluation_items, choice_gold_from_row,
    load_verified_plan_and_receipt, sha256_file, validate_preflight_plans,
)

PROTOCOL = "nextgqa_test_associated_qta_geometry_feasible_qid_v3"
PLAN_NAME = "nextgqa_tlvs_exploratory_split_v3_geometry_feasible_20261018_01.json"
SOURCE_FILES = (
    "scripts/run_nextgqa_qta_v3_geometry_census.py",
    "scripts/run_nextgqa_qta_exploratory_proxy_v2.py",
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/build_nextgqa_exploratory_video_split_v2.py",
    "scripts/build_nextgqa_exploratory_qta_receipt.py",
    "scripts/freeze_nextgqa_exploratory_v1_local.py",
    "scripts/freeze_nextgqa_qta_v2_inputs_local.py",
    "experiments/2026-10-18_qta_treatment_selection.md",
)
DATA_FILES = tuple(str(DATA_REL / name) for name in (
    "test.csv", "gsub_test.json", "frame2time_test.json", "map_vid_vidorID.json"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def min_hash_qid(video_id: str, qid: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}\0qid\0{video_id}\0{qid}".encode()).hexdigest()


def safe_error_code(exc: Exception) -> str:
    value = str(exc)
    return value if re.fullmatch(r"[a-zA-Z0-9_]{1,96}", value) else type(exc).__name__


def validate_candidate_geometry(mapped: dict, groups: list) -> tuple[int, int]:
    evidence = mapped.get("evidence_group_indices")
    control = mapped.get("control_group_indices")
    covered = mapped.get("annotation_covered_group_indices")
    if (not isinstance(evidence, list) or len(evidence) != 1
            or not isinstance(control, list) or len(control) != 1
            or not isinstance(covered, list)
            or set(evidence) & set(control)
            or control[0] in covered
            or evidence[0] not in covered
            or any(type(i) is not int or i < 0 or i >= len(groups)
                   for i in (*evidence, *control, *covered))):
        raise ValueError("single_anchor_geometry_contract_invalid")
    return evidence[0], control[0]


def validate_census_structure(census: dict, counters: dict, requested_n: int) -> None:
    videos = census.get("videos")
    if (requested_n not in (1, 147) or not isinstance(videos, list)
            or len(videos) != requested_n
            or counters.get("videos_attempted") != requested_n
            or counters.get("videos_completed") != requested_n
            or counters.get("candidate_qids_attempted") != counters.get("candidate_qids_completed")):
        raise ValueError("census_all_or_none_denominator_invalid")
    seen_videos = set()
    observed_eligible = 0
    observed_candidates = 0
    for video in videos:
        if not isinstance(video, dict) or not isinstance(video.get("video_id"), str):
            raise ValueError("census_video_record_invalid")
        vid = video["video_id"]
        results, eligible = video.get("candidate_results"), video.get("eligible_qids")
        if (vid in seen_videos or not isinstance(results, list) or not isinstance(eligible, list)
                or video.get("valid_qid_count") != len(results)):
            raise ValueError("census_candidate_denominator_invalid")
        seen_videos.add(vid)
        result_ids = [x.get("qid") for x in results if isinstance(x, dict)]
        eligible_ids = [x.get("qid") for x in eligible if isinstance(x, dict)]
        if (len(result_ids) != len(results) or len(set(result_ids)) != len(results)
                or len(set(eligible_ids)) != len(eligible)
                or set(eligible_ids) != {x["qid"] for x in results if x.get("status") == "eligible"}
                or any(x.get("status") not in {"eligible", "ineligible"} for x in results)):
            raise ValueError("census_candidate_status_mismatch")
        observed_candidates += len(results)
        observed_eligible += len(eligible)
    if (observed_candidates != counters.get("candidate_qids_completed")
            or observed_eligible != counters.get("eligible_qids")):
        raise ValueError("census_aggregate_count_mismatch")


def select_v3_qids(census: dict) -> list[dict]:
    rows = census.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("census_rows_invalid")
    selected = []
    for row in rows:
        candidates = row.get("eligible_qids") if isinstance(row, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("v3_all_or_none_no_feasible_qid")
        chosen = min(candidates, key=lambda item: min_hash_qid(row["video_id"], item["qid"]))
        selected.append({"video_id": row["video_id"], "qid": chosen["qid"],
            "selection_sha256": min_hash_qid(row["video_id"], chosen["qid"]),
            "valid_qid_count": row["valid_qid_count"],
            "eligible_qid_count": len(candidates),
            "selected_row_binding_sha256": chosen["row_binding_sha256"],
            "geometry_sha256": chosen["geometry_sha256"]})
    if len({r["video_id"] for r in selected}) != len(selected):
        raise ValueError("v3_video_duplicate")
    return selected


def write_private(repo: Path, name: str, payload: dict) -> None:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("output_name_must_be_basename")
    private = repo / PRIVATE_REL
    root = private.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or root.stat().st_uid != os.getuid():
        raise ValueError("private_output_root_invalid")
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as f:
            json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
    finally:
        os.close(fd)


def run_census(repo: Path, *, processor, torch, plan: dict,
               valid_rows: dict[str, list[dict[str, str]]], gsub: dict,
               base_items: list[dict], counters: dict) -> dict:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    from run_nextgqa_qta_exploratory_proxy_v2 import DATA_REL, POLICY

    base_items = {item["video_id"]: item for item in base_items}
    videos = []
    for ordinal, base in enumerate(plan["evaluation"], 1):
        vid = base["video_id"]
        item0 = base_items[vid]
        frames, decode = decode_sampled_video(item0["video_path"], num_frames=kernel.FRAME_COUNT,
            uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        eligible, candidate_results = [], []
        candidate_rows = valid_rows.get(vid, [])
        for row in candidate_rows:
            qid = str(row["qid"])
            spans = gsub[vid]["location"][qid]
            item = {**item0, "qid": qid, "question": str(row["question"]).strip(),
                    "spans": spans, "duration": gsub[vid]["duration"]}
            counters["candidate_qids_attempted"] += 1
            try:
                choices, gold_index = choice_gold_from_row(row)
                item["choices"] = choices
                batch, mapped, groups, _qpos, _tokens, fingerprint = kernel.prepare_geometry(
                    processor, item, frames, decode, torch, POLICY)
                evidence_index, control_index = validate_candidate_geometry(mapped, groups)
                eligible.append({"qid": qid, "geometry_sha256": fingerprint,
                    "evidence_group_index": evidence_index, "control_group_index": control_index,
                    "row_binding_sha256": canonical_sha256({"row": row,
                        "spans": spans, "duration": item["duration"],
                        "choices_sha256": canonical_sha256(choices), "gold_index": gold_index})})
                candidate_results.append({"qid": qid, "status": "eligible",
                    "geometry_sha256": fingerprint, "evidence_group_index": evidence_index,
                    "control_group_index": control_index})
                counters["eligible_qids"] += 1
                del batch
            except (ValueError, TypeError, IndexError) as exc:
                # Candidate-level structural/geometry ineligibility is recorded,
                # never silently removed from the census denominator.
                candidate_results.append({"qid": qid, "status": "ineligible",
                    "failure_code": safe_error_code(exc)})
            counters["candidate_qids_completed"] += 1
        videos.append({"video_id": vid, "valid_qid_count": len(candidate_rows),
            "candidate_results": sorted(candidate_results, key=lambda item: item["qid"]),
            "eligible_qids": sorted(eligible, key=lambda item: item["qid"])})
        counters["videos_completed"] += 1
        del frames, decode
    return {"videos": videos}


def make_private_plan(*, plan_v2: dict, census: dict, source_sha256: dict,
                      data_sha256: dict, v2_plan_sha256: str,
                      preflight_receipt_sha256: str) -> dict:
    selected = select_v3_qids(census)
    selected_by_video = {row["video_id"]: row for row in selected}
    if set(selected_by_video) != {row["video_id"] for row in plan_v2["evaluation"]}:
        raise ValueError("v3_eval_video_partition_mismatch")
    evaluation = []
    for old in plan_v2["evaluation"]:
        picked = selected_by_video[old["video_id"]]
        evaluation.append({**old, **picked,
            "selection_condition": "min_hash among all valid test qids with exactly one geometry evidence/control anchor"})
    return {"schema_version": 3, "protocol": PROTOCOL,
        "status": "test_associated_exploratory_geometry_selected_not_confirmatory",
        "scope": "selection-conditioned QTA annotation dependence proxy; not faithfulness or fresh/confirmatory",
        "selection": "preserve frozen v2/v1 evaluation video membership; min-hash QID among exhaustive interval-valid candidates passing one-evidence/one-disjoint-control processor geometry census before outcomes",
        "provenance": {"v2_plan_sha256": v2_plan_sha256,
            "v2_preflight_receipt_sha256": preflight_receipt_sha256,
            "source_sha256": source_sha256, "data_sha256": data_sha256,
            "geometry_census_sha256": canonical_sha256(census)},
        "counts": {"evaluation_videos": len(evaluation),
            "all_videos_have_eligible_qid": True,
            "candidate_qids": sum(x["valid_qid_count"] for x in census["videos"]),
            "eligible_qids": sum(len(x["eligible_qids"]) for x in census["videos"]),
            "eligibility_conditioned_selection": True},
        "calibration": plan_v2["calibration"], "evaluation": evaluation,
        "limitations": ["All media are from the official test-associated benchmark pool.",
            "Selection conditions the evaluation QID on outcome-blind processor geometry feasibility.",
            "The private census and plan contain identifiers and hashes; retain local-only.",
            "Geometry feasibility does not establish semantic annotation correctness, timing calibration, or faithfulness."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(1, 147), default=147,
        help="1 performs a one-video smoke; 147 performs the complete frozen evaluation census")
    parser.add_argument("--census-output-name", required=True)
    parser.add_argument("--plan-output-name", default=None,
        help="write a selected v3 plan only when every requested video has an eligible QID; full run only")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    if args.n == 1 and args.plan_output_name:
        parser.error("smoke_cannot_write_v3_plan")
    if args.plan_output_name == args.census_output_name:
        parser.error("census_and_plan_outputs_must_be_distinct")
    if (Path(args.census_output_name).name != args.census_output_name
            or (args.plan_output_name and Path(args.plan_output_name).name != args.plan_output_name)):
        parser.error("output_names_must_be_basenames")
    repo = Path(__file__).resolve().parents[1]
    private, data_dir = repo / PRIVATE_REL, repo / DATA_REL
    started = time.monotonic()
    counters = {"videos_attempted": 0, "videos_completed": 0,
        "candidate_qids_attempted": 0, "candidate_qids_completed": 0, "eligible_qids": 0}
    try:
        plan_v2, _receipt_v2, _v1, gate = load_verified_plan_and_receipt(repo)
        full_base_items = build_evaluation_items(repo, plan_v2)
        if len(full_base_items) != 147:
            raise ValueError("frozen_v2_eval_item_denominator_invalid")
        if args.n == 1:
            plan_v2 = {**plan_v2, "evaluation": plan_v2["evaluation"][:1]}
        with (data_dir / "test.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        gsub = json.loads((data_dir / "gsub_test.json").read_text(encoding="utf-8"))
        valid_rows: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            vid, qid = str(row.get("video_id", "")), str(row.get("qid", ""))
            if vid in {entry["video_id"] for entry in plan_v2["evaluation"]} and valid_spans(gsub.get(vid), qid):
                valid_rows.setdefault(vid, []).append(row)
        if any(not valid_rows.get(entry["video_id"]) for entry in plan_v2["evaluation"]):
            raise ValueError("frozen_eval_video_without_valid_qid")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / kernel.MODEL_REVISION)
        processor = __import__("transformers").AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, min_pixels=256*28*28, max_pixels=256*28*28)
        import torch
        counters["videos_attempted"] = len(plan_v2["evaluation"])
        census = run_census(repo, processor=processor, torch=torch, plan=plan_v2,
            valid_rows=valid_rows, gsub=gsub, base_items=full_base_items,
            counters=counters)
        validate_census_structure(census, counters, args.n)
        eval_source_names = list(SOURCE_FILES)
        source_hashes = {name: sha256_file(repo / name) for name in eval_source_names}
        data_hashes = {name: sha256_file(repo / name) for name in DATA_FILES}
        full = args.n == 147
        private_census = {"schema_version": 1, "protocol": PROTOCOL,
            "status": "complete_outcome_blind_geometry_census" if full and counters["videos_completed"] == 147 else "smoke_only_not_census",
            "requested_videos": args.n, "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "candidate_qids_completed": counters["candidate_qids_completed"],
            "eligible_qids": counters["eligible_qids"],
            "partition_video_set_sha256": canonical_sha256([x["video_id"] for x in plan_v2["evaluation"]]),
            "v2_plan_sha256": gate["plan_sha256"],
            "v2_preflight_receipt_sha256": gate["preflight_receipt_sha256"],
            "source_sha256": source_hashes, "data_sha256": data_hashes,
            "elapsed_seconds": time.monotonic()-started, "videos": census["videos"],
            "scope": "geometry eligibility only; no model outputs; not authorization or faithfulness"}
        if args.plan_output_name and (not full or counters["videos_completed"] != 147):
            raise ValueError("v3_plan_requires_complete_147_video_census")
        if args.plan_output_name and sum(bool(v["eligible_qids"]) for v in census["videos"]) == 147:
            selected_plan = make_private_plan(plan_v2=plan_v2, census=census,
                source_sha256=source_hashes, data_sha256=data_hashes,
                v2_plan_sha256=gate["plan_sha256"],
                preflight_receipt_sha256=gate["preflight_receipt_sha256"])
        else:
            selected_plan = None
        if selected_plan is not None:
            # The exact private census bytes are bound into the selected plan.
            # Write the census first so its digest is stable and independently checkable.
            write_private(repo, args.census_output_name, private_census)
            selected_plan["provenance"]["geometry_census_sha256"] = sha256_file(
                private / args.census_output_name)
            write_private(repo, args.plan_output_name, selected_plan)
        else:
            if args.plan_output_name:
                private_census["status"] = "complete_census_no_plan_some_videos_infeasible"
            write_private(repo, args.census_output_name, private_census)
        stdout = {"status": private_census["status"], "requested_videos": args.n,
            "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "eligible_qids": counters["eligible_qids"],
            "feasible_videos": sum(bool(v["eligible_qids"]) for v in census["videos"]),
            "failure_count": 0, "elapsed_seconds": private_census["elapsed_seconds"],
            "private_census_name": args.census_output_name,
            "private_plan_name": args.plan_output_name}
    except Exception as exc:
        payload = {"schema_version": 1, "protocol": PROTOCOL,
            "status": "failed_incomplete_no_v3_plan", "requested_videos": args.n,
            "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "candidate_qids_completed": counters["candidate_qids_completed"],
            "eligible_qids": counters["eligible_qids"], "failure_code": safe_error_code(exc),
            "elapsed_seconds": time.monotonic()-started,
            "scope": "outcome-blind geometry only; no model outputs"}
        try:
            write_private(repo, args.census_output_name, payload)
        except Exception as write_exc:
            parser.error(f"private_census_write_failed:{type(write_exc).__name__}")
        print(json.dumps({"status": payload["status"], "requested_videos": args.n,
            "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "eligible_qids": counters["eligible_qids"], "failure_count": 1,
            "failure_code": payload["failure_code"],
            "private_census_name": args.census_output_name}, sort_keys=True))
        return 2
    print(json.dumps(stdout, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
