#!/usr/bin/env python3
"""Outcome-blind CPU census for one-sided annotation-union QTA v4.

The treatment contrast is a one-sided, four-condition estimand: baseline full,
baseline all-annotation-union mask, QTA full, and QTA all-annotation-union
mask. This census uses every interval-valid QID in the frozen 147-video
partition and selects min-hash only among QIDs whose exact annotation-union
groups match the processor's actual-PTS and scalar-FPS temporal geometry.
No model weights are loaded; no CUDA call or score is produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

import run_nextgqa_validation_qta_annotation_proxy as kernel
from build_nextgqa_exploratory_video_split_v2 import valid_spans
from run_nextgqa_qta_exploratory_proxy_v2 import (
    DATA_REL, PRIVATE_REL, build_evaluation_items, choice_gold_from_row,
    load_verified_plan_and_receipt, sha256_file,
)
from run_nextgqa_qta_v3_geometry_census import (
    canonical_sha256, ensure_outputs_absent, safe_error_code, write_private,
)

PROTOCOL = "nextgqa_test_associated_qta_annotation_union_v4"
POLICY = "annotation_union_only"
PLAN_NAME = "nextgqa_tlvs_exploratory_split_v4_union_20261018_01.json"
SOURCE_FILES = (
    "scripts/run_nextgqa_qta_v4_union_geometry_census.py",
    "scripts/run_nextgqa_qta_v4_union_scorer.py",
    "scripts/fixed_annotation_union_dependence.py",
    "scripts/run_nextgqa_qta_exploratory_proxy_v2.py",
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/build_nextgqa_exploratory_video_split_v2.py",
    "scripts/build_nextgqa_exploratory_qta_receipt.py",
    "scripts/freeze_nextgqa_exploratory_v1_local.py",
    "scripts/freeze_nextgqa_qta_v2_inputs_local.py",
    "scripts/run_nextgqa_qta_v3_geometry_census.py",
    "experiments/2026-10-18_qta_treatment_selection.md",
    "experiments/2026-10-18_qta_v4_union_treatment.md",
)
DATA_FILES = tuple(str(DATA_REL / name) for name in (
    "test.csv", "gsub_test.json", "frame2time_test.json", "map_vid_vidorID.json"))


def validate_union_geometry(mapped: dict, groups: list) -> list[int]:
    evidence = mapped.get("evidence_group_indices")
    covered = mapped.get("annotation_covered_group_indices")
    control = mapped.get("control_group_indices")
    if (not isinstance(evidence, list) or not evidence
            or not isinstance(covered, list) or not covered
            or evidence != covered or control != []
            or len(set(evidence)) != len(evidence)
            or len(set(covered)) != len(covered)
            or evidence != sorted(evidence)
            or covered != sorted(covered)
            or any(type(i) is not int or i < 0 or i >= len(groups) for i in evidence)):
        raise ValueError("annotation_union_geometry_contract_invalid")
    return evidence


def processor_geometry_provenance(processor, model_path: Path) -> dict:
    """Bind the processor's separate video component with stable JSON config."""
    base = kernel.processor_provenance(processor, model_path)
    video = getattr(processor, "video_processor", None)
    if video is None:
        raise ValueError("video_processor_missing")
    to_dict = getattr(video, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("video_processor_config_not_serializable")
    config = to_dict()
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("video_processor_config_not_canonical_json") from exc
    config_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return {**base, "video_processor_class": type(video).__name__,
            "video_processor_config_sha256": config_sha,
            "processor_fingerprint_sha256": canonical_sha256({
                "base_fingerprint": base["processor_fingerprint_sha256"],
                "video_processor_class": type(video).__name__,
                "video_processor_config_sha256": config_sha}),
            "fingerprint_scope": "tokenizer serialization + image processor config + video processor config + video token ID + model config bytes"}


def validate_census(census: dict, counters: dict, expected_n: int) -> None:
    videos = census.get("videos")
    if (expected_n not in (1, 147) or not isinstance(videos, list) or len(videos) != expected_n
            or counters.get("videos_attempted") != expected_n
            or counters.get("videos_completed") != expected_n
            or counters.get("candidate_qids_attempted") != counters.get("candidate_qids_completed")):
        raise ValueError("v4_census_all_or_none_denominator_invalid")
    vids = set(); n_candidates = n_eligible = 0
    for video in videos:
        results, eligible = video.get("candidate_results"), video.get("eligible_qids")
        if (video.get("video_id") in vids or not isinstance(results, list)
                or not isinstance(eligible, list) or len(results) != video.get("valid_qid_count")):
            raise ValueError("v4_census_video_denominator_invalid")
        vids.add(video["video_id"])
        result_ids = [r.get("qid") for r in results if isinstance(r, dict)]
        eligible_ids = [r.get("qid") for r in eligible if isinstance(r, dict)]
        if (len(result_ids) != len(results) or len(set(result_ids)) != len(results)
                or len(set(eligible_ids)) != len(eligible)
                or set(eligible_ids) != {r["qid"] for r in results if r.get("status") == "eligible"}
                or any(r.get("status") not in {"eligible", "ineligible"} for r in results)):
            raise ValueError("v4_census_candidate_status_invalid")
        n_candidates += len(results); n_eligible += len(eligible)
    if (n_candidates != counters.get("candidate_qids_completed")
            or n_eligible != counters.get("eligible_qids")):
        raise ValueError("v4_census_aggregate_mismatch")


def selection_hash(video_id: str, qid: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}\0feasible_qid\0{video_id}\0{qid}".encode()).hexdigest()


def build_v4_plan(*, base_plan: dict, census: dict, source_hashes: dict,
                  data_hashes: dict, input_gate: dict, processor_manifest: dict,
                  library_versions: dict, expected_n: int = 147) -> dict:
    videos = census["videos"]
    if (len(videos) != expected_n or any(not row["eligible_qids"] for row in videos)):
        raise ValueError("v4_all_or_none_no_feasible_qid")
    by_video = {row["video_id"]: row for row in videos}
    if set(by_video) != {row["video_id"] for row in base_plan["evaluation"]}:
        raise ValueError("v4_video_partition_mismatch")
    evaluation = []
    for original in base_plan["evaluation"]:
        row = by_video[original["video_id"]]
        selected = min(row["eligible_qids"], key=lambda item: selection_hash(row["video_id"], item["qid"]))
        total, covered = selected.get("temporal_group_count"), selected.get("covered_group_count")
        fraction, full = selected.get("covered_fraction"), selected.get("full_coverage")
        if (type(total) is not int or total <= 0 or type(covered) is not int
                or not 0 < covered <= total or type(fraction) not in (int, float)
                or fraction != covered / total or type(full) is not bool
                or full != (covered == total)):
            raise ValueError("v4_selected_mask_coverage_contract_invalid")
        evaluation.append({**original, "qid": selected["qid"],
            "selection_sha256": selection_hash(row["video_id"], selected["qid"]),
            "geometry_sha256": selected["geometry_sha256"],
            "selected_row_binding_sha256": selected["row_binding_sha256"],
            "temporal_group_count": selected["temporal_group_count"],
            "annotation_union_covered_group_count": selected["covered_group_count"],
            "annotation_union_covered_fraction": selected["covered_fraction"],
            "annotation_union_full_coverage": selected["full_coverage"],
            "eligible_qid_count": len(row["eligible_qids"]),
            "valid_qid_count": row["valid_qid_count"]})
    return {"schema_version": 4, "protocol": PROTOCOL,
        "status": "test_associated_exploratory_union_mask_plan_not_confirmatory",
        "scope": "selection-conditioned one-sided annotation-union dependence; not faithfulness, fresh, or confirmatory",
        "estimand": {"conditions": ["baseline_full", "baseline_union_mask", "intervention_full", "intervention_union_mask"],
            "answer_foil": "freeze top A-E answer and best foil from baseline_full and use them in all four branches",
            "D_t": "margin_t(full)-margin_t(union_mask)",
            "primary": "delta_union_dependence=D_intervention-D_baseline",
            "positive_sign": "QTA increases sensitivity of the fixed answer/foil margin to masking the full annotation-covered union",
            "uncertainty": "finite cohort descriptive only; no p-value or population CI"},
        "selection": "preserve frozen v2 evaluation video membership; exhaustively consider interval-valid test QIDs, require actual-PTS/scalar-FPS geometry agreement, and choose minimum SHA256 among geometry-feasible candidates before outcomes",
        "provenance": {"v2_plan_sha256": input_gate["plan_sha256"],
            "v2_preflight_receipt_sha256": input_gate["preflight_receipt_sha256"],
            "source_sha256": source_hashes, "data_sha256": data_hashes,
            "processor_manifest": processor_manifest, "library_versions": library_versions,
            "geometry_census_sha256": canonical_sha256(census)},
        "counts": {"evaluation_videos": len(evaluation),
            "candidate_qids": sum(row["valid_qid_count"] for row in videos),
            "eligible_qids": sum(len(row["eligible_qids"]) for row in videos),
            "all_videos_have_eligible_qid": True},
        "calibration": base_plan["calibration"], "evaluation": evaluation,
        "limitations": ["Both calibration and evaluation are official test-associated media.",
            "QID selection is conditioned on outcome-blind processor geometry feasibility.",
            "The intervention is an all-annotation-union video-token ablation and can create out-of-distribution inputs.",
            "The measure is prediction-margin dependence on benchmark annotation masks, not generated-claim faithfulness.",
            "All records and receipts are protected local data; rights remain unverified."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(1, 147), default=147)
    parser.add_argument("--census-output-name", required=True)
    parser.add_argument("--plan-output-name", default=None)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    if args.n == 1 and args.plan_output_name:
        parser.error("smoke_cannot_write_v4_plan")
    names = [args.census_output_name, args.plan_output_name or ""]
    if (args.plan_output_name == args.census_output_name
            or any(name and Path(name).name != name for name in names)):
        parser.error("output_names_invalid")
    repo = Path(__file__).resolve().parents[1]
    private, dataset = repo / PRIVATE_REL, repo / DATA_REL
    try:
        ensure_outputs_absent(private, names)
    except (OSError, ValueError) as exc:
        parser.error(f"output_collision_or_private_root_invalid:{type(exc).__name__}")
    started = time.monotonic()
    counters = {"videos_attempted": 0, "videos_completed": 0,
        "candidate_qids_attempted": 0, "candidate_qids_completed": 0, "eligible_qids": 0}
    try:
        base_plan, _receipt, _v1, gate = load_verified_plan_and_receipt(repo)
        items = build_evaluation_items(repo, base_plan)
        if len(items) != 147:
            raise ValueError("v4_eval_item_denominator_invalid")
        full_item_by_video = {item["video_id"]: item for item in items}
        if args.n == 1:
            base_plan = {**base_plan, "evaluation": base_plan["evaluation"][:1]}
        with (dataset / "test.csv").open("r", encoding="utf-8-sig", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
        frame2time = json.loads((dataset / "frame2time_test.json").read_text(encoding="utf-8"))
        mapping = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
        eval_ids = {row["video_id"] for row in base_plan["evaluation"]}
        candidates: dict[str, list[dict]] = {}
        for row in csv_rows:
            vid, qid = str(row.get("video_id", "")), str(row.get("qid", ""))
            if vid in eval_ids and valid_spans(gsub.get(vid), qid):
                candidates.setdefault(vid, []).append(row)
        if any(not candidates.get(vid) for vid in eval_ids):
            raise ValueError("v4_eval_video_without_interval_valid_qid")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / kernel.MODEL_REVISION)
        processor = __import__("transformers").AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, min_pixels=256*28*28, max_pixels=256*28*28)
        import torch
        processor_manifest = processor_geometry_provenance(processor, model_path)
        library_versions = {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "av", "qwen-vl-utils")}
        sources = {name: sha256_file(repo / name) for name in SOURCE_FILES}
        data_hashes = dict(gate["data_sha256"])
        from claim_mask_video_decoder import decode_sampled_video
        from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
        item_by_video = full_item_by_video
        videos = []
        counters["videos_attempted"] = args.n
        for base in base_plan["evaluation"]:
            vid = base["video_id"]
            frames, decode = decode_sampled_video(item_by_video[vid]["video_path"],
                num_frames=kernel.FRAME_COUNT, uniform_tolerance_seconds=UNIFORM_TOLERANCE)
            eligible, result_rows = [], []
            for row in candidates[vid]:
                qid = str(row["qid"]); spans = gsub[vid]["location"][qid]
                times = frame2time.get(vid); map_target = mapping.get(vid)
                if (not isinstance(times, list) or len(times) != 32
                        or not isinstance(map_target, str) or Path(map_target).stem != vid
                        or Path(base["media_path"]).stem != vid):
                    raise ValueError("v4_candidate_source_join_invalid")
                item = {**item_by_video[vid], "qid": qid,
                    "question": str(row["question"]).strip(),
                    "choices": choice_gold_from_row(row)[0],
                    "spans": spans, "duration": gsub[vid]["duration"]}
                counters["candidate_qids_attempted"] += 1
                try:
                    batch, mapped, groups, _qpos, _tokens, fingerprint = kernel.prepare_geometry(
                        processor, item, frames, decode, torch, POLICY)
                    covered = validate_union_geometry(mapped, groups)
                    evidence_positions = mapped["evidence_token_positions"]
                    if not evidence_positions:
                        raise ValueError("annotation_union_token_mask_empty")
                    row_binding = canonical_sha256({"csv_row": row, "spans": spans,
                        "duration": gsub[vid]["duration"], "frame2time": times,
                        "map_target": map_target, "media_sha256": base["media_sha256"]})
                    eligible.append({"qid": qid, "geometry_sha256": fingerprint,
                        "covered_group_count": len(covered),
                        "temporal_group_count": len(groups),
                        "covered_fraction": len(covered) / len(groups),
                        "full_coverage": len(covered) == len(groups),
                        "row_binding_sha256": row_binding})
                    result_rows.append({"qid": qid, "status": "eligible",
                        "geometry_sha256": fingerprint, "covered_group_count": len(covered),
                        "temporal_group_count": len(groups),
                        "covered_fraction": len(covered) / len(groups),
                        "full_coverage": len(covered) == len(groups)})
                    counters["eligible_qids"] += 1
                    del batch
                except (ValueError, TypeError, IndexError) as exc:
                    result_rows.append({"qid": qid, "status": "ineligible",
                        "failure_code": safe_error_code(exc)})
                counters["candidate_qids_completed"] += 1
            videos.append({"video_id": vid, "valid_qid_count": len(candidates[vid]),
                "candidate_results": sorted(result_rows, key=lambda x:x["qid"]),
                "eligible_qids": sorted(eligible, key=lambda x:x["qid"])})
            counters["videos_completed"] += 1
            del frames, decode
        census = {"videos": videos}
        validate_census(census, counters, args.n)
        if ({name: sha256_file(repo / name) for name in SOURCE_FILES} != sources
                or {name: sha256_file(repo / name) for name in data_hashes} != data_hashes
                or processor_geometry_provenance(processor, model_path) != processor_manifest
                or {name: importlib.metadata.version(name) for name in library_versions} != library_versions):
            raise ValueError("v4_source_processor_or_library_changed_during_census")
        full_census = args.n == 147
        complete = {"schema_version": 1, "protocol": PROTOCOL,
            "status": "complete_outcome_blind_union_geometry_census" if full_census else "smoke_only_not_census",
            "requested_videos": args.n, "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "candidate_qids_completed": counters["candidate_qids_completed"],
            "eligible_qids": counters["eligible_qids"],
            "partition_video_set_sha256": canonical_sha256([x["video_id"] for x in base_plan["evaluation"]]),
            "v2_plan_sha256": gate["plan_sha256"],
            "v2_preflight_receipt_sha256": gate["preflight_receipt_sha256"],
            "source_sha256": sources, "data_sha256": data_hashes,
            "processor_manifest": processor_manifest, "library_versions": library_versions,
            "elapsed_seconds": time.monotonic()-started, "videos": videos,
            "scope": "outcome-blind geometry eligibility only; not score/faithfulness"}
        feasible_videos = sum(bool(row["eligible_qids"]) for row in videos)
        selected_plan = None
        if args.plan_output_name and full_census and feasible_videos == 147:
            selected_plan = build_v4_plan(base_plan=base_plan, census=census,
                source_hashes=sources, data_hashes=data_hashes, input_gate=gate,
                processor_manifest=processor_manifest, library_versions=library_versions)
        elif args.plan_output_name:
            complete["status"] = "complete_census_no_plan_some_videos_infeasible"
        write_private(repo, args.census_output_name, complete)
        if selected_plan:
            selected_plan["provenance"]["geometry_census_sha256"] = sha256_file(private/args.census_output_name)
            write_private(repo, args.plan_output_name, selected_plan)
        print(json.dumps({"status": complete["status"],
            "requested_videos": args.n, "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "eligible_qids": counters["eligible_qids"], "feasible_videos": feasible_videos,
            "failure_count": 0, "elapsed_seconds": complete["elapsed_seconds"],
            "private_census_name": args.census_output_name,
            "private_plan_name": args.plan_output_name if selected_plan else None}, sort_keys=True))
        return 0 if selected_plan or not args.plan_output_name else 3
    except Exception as exc:
        fail = {"schema_version": 1, "protocol": PROTOCOL,
            "status": "failed_incomplete_no_v4_plan", "requested_videos": args.n,
            "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "candidate_qids_completed": counters["candidate_qids_completed"],
            "eligible_qids": counters["eligible_qids"], "failure_code": safe_error_code(exc),
            "elapsed_seconds": time.monotonic()-started}
        try:
            write_private(repo, args.census_output_name, fail)
        except FileExistsError:
            parser.error("private_census_output_created_concurrently;existing_file_preserved")
        print(json.dumps({"status": fail["status"], "completed_videos": counters["videos_completed"],
            "candidate_qids_attempted": counters["candidate_qids_attempted"],
            "eligible_qids": counters["eligible_qids"], "failure_count": 1,
            "failure_code": fail["failure_code"], "private_census_name": args.census_output_name}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
