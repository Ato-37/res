#!/usr/bin/env python3
"""Private four-condition scorer for the frozen exploratory QTA v4 cohort.

Modes are CPU-only full-cohort geometry preflight, one-row GPU engineering
smoke, or full 147-row scoring. This is an annotation-mask dependence proxy,
not generated-claim faithfulness or confirmatory evidence.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import stat
import time

import torch
import run_nextgqa_validation_qta_annotation_proxy as kernel
from fixed_annotation_union_dependence import score_four_condition_logits
from run_nextgqa_qta_v4_union_geometry_census import (
    CENSUS_NAME, DATA_FILES, PLAN_NAME, PROTOCOL, SOURCE_FILES as CENSUS_SOURCES,
    canonical_sha256, processor_geometry_provenance, validate_census,
    validate_union_geometry,
)
from run_nextgqa_qta_exploratory_proxy_v2 import (
    DATA_REL, PRIVATE_REL, build_evaluation_items, choice_gold_from_row,
    load_verified_plan_and_receipt, sha256_file,
)
from run_nextgqa_qta_v3_geometry_census import safe_error_code, write_private

PREFLIGHT_NAME = "nextgqa_tlvs_qta_v4_cpu_geometry_preflight_local_20261018_03.json"
SMOKE_NAME = "nextgqa_tlvs_qta_v4_gpu_smoke_local_20261018_03.json"
OUTPUT_PREFIX = "nextgqa_tlvs_qta_v4_score_local_20261018_03_"
EXPECTED_N = 147
EXPECTED_FORWARDS = 588
SOURCE_FILES = tuple(dict.fromkeys((*CENSUS_SOURCES,
    "scripts/run_nextgqa_qta_v4_union_scorer.py")))
CHOICE_NAMES = ("baseline_full", "baseline_union_mask",
                "intervention_full", "intervention_union_mask")


def private_json(path: Path, root: Path) -> dict:
    real_root = root.resolve(strict=True)
    if path.resolve(strict=True).parent != real_root or stat.S_IMODE(real_root.stat().st_mode) != 0o700:
        raise ValueError("private_input_root_invalid")
    fd = __import__("os").open(path, __import__("os").O_RDONLY | getattr(__import__("os"), "O_NOFOLLOW", 0))
    try:
        st = __import__("os").fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != __import__("os").getuid() or stat.S_IMODE(st.st_mode) != 0o600:
            raise ValueError("private_input_mode_invalid")
        parts = []
        while block := __import__("os").read(fd, 1024 * 1024): parts.append(block)
        raw = b"".join(parts)
    finally:
        __import__("os").close(fd)
    return json.loads(raw)


def load_v4_plan(repo: Path):
    private = repo / PRIVATE_REL
    census_path = private / CENSUS_NAME
    plan_path = private / PLAN_NAME
    census = private_json(census_path, private)
    plan = private_json(plan_path, private)
    if (census.get("status") != "complete_outcome_blind_union_geometry_census"
            or census.get("requested_videos") != EXPECTED_N
            or census.get("completed_videos") != EXPECTED_N
            or plan.get("status") != "test_associated_exploratory_union_mask_plan_not_confirmatory"
            or plan.get("protocol") != PROTOCOL or len(plan.get("evaluation", [])) != EXPECTED_N
            or plan.get("provenance", {}).get("geometry_census_sha256") != sha256_file(census_path)):
        raise ValueError("v4_plan_or_census_contract_invalid")
    counters = {"videos_attempted": census.get("requested_videos"),
        "videos_completed": census.get("completed_videos"),
        "candidate_qids_attempted": census.get("candidate_qids_attempted"),
        "candidate_qids_completed": census.get("candidate_qids_completed"),
        "eligible_qids": census.get("eligible_qids")}
    validate_census({"videos": census.get("videos")}, counters, EXPECTED_N)
    base, _receipt, _v1, gate = load_verified_plan_and_receipt(repo)
    base_ids = [x["video_id"] for x in base["evaluation"]]
    rows = plan["evaluation"]
    if ([x["video_id"] for x in rows] != base_ids
            or plan["provenance"].get("v2_plan_sha256") != gate["plan_sha256"]
            or plan["provenance"].get("v2_preflight_receipt_sha256") != gate["preflight_receipt_sha256"]):
        raise ValueError("v4_frozen_partition_binding_invalid")
    # Reconstruct eligible min-hash selection directly from the private census.
    by_video = {v["video_id"]: v for v in census["videos"]}
    if set(by_video) != set(base_ids) or any(not v["eligible_qids"] for v in by_video.values()):
        raise ValueError("v4_census_all_or_none_invalid")
    from run_nextgqa_qta_v4_union_geometry_census import selection_hash
    for row in rows:
        candidates = by_video[row["video_id"]]["eligible_qids"]
        chosen = min(candidates, key=lambda x: selection_hash(row["video_id"], x["qid"]))
        if (row.get("qid") != chosen["qid"]
                or row.get("geometry_sha256") != chosen["geometry_sha256"]
                or row.get("selected_row_binding_sha256") != chosen["row_binding_sha256"]):
            raise ValueError("v4_plan_selection_reconstruction_failed")
    return base, gate, plan, census


def selected_items(repo: Path, base_plan: dict, v4_plan: dict) -> list[dict]:
    base_items = {x["video_id"]: x for x in build_evaluation_items(repo, base_plan)}
    dataset = repo / DATA_REL
    with (dataset / "test.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    by_pair = {}
    for row in csv_rows:
        pair = (str(row.get("video_id", "")), str(row.get("qid", "")))
        if pair in by_pair: raise ValueError("v4_csv_pair_duplicate")
        by_pair[pair] = row
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    mapping = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    result = []
    for planned in v4_plan["evaluation"]:
        vid, qid = planned["video_id"], planned["qid"]
        base = base_items[vid]
        media_sha256 = frozen_media_sha256(base_plan, vid)
        row = by_pair.get((vid, qid))
        annotation = gsub.get(vid)
        if (row is None or not isinstance(annotation, dict)
                or qid not in annotation.get("location", {})
                or not isinstance(mapping.get(vid), str)
                or Path(mapping[vid]).stem != vid
                or Path(base["video_path"]).stem != vid
                or sha256_file(base["video_path"]) != media_sha256):
            raise ValueError("v4_selected_source_join_invalid")
        choices, gold = choice_gold_from_row(row)
        spans = annotation["location"][qid]
        binding = canonical_sha256({"csv_row": row, "spans": spans,
            "duration": annotation["duration"], "frame2time": json.loads(
                (dataset / "frame2time_test.json").read_text(encoding="utf-8")).get(vid),
            "map_target": mapping[vid], "media_sha256": media_sha256})
        if binding != planned["selected_row_binding_sha256"]:
            raise ValueError("v4_selected_row_binding_changed")
        result.append({**base, "media_sha256": media_sha256,
            "qid": qid, "question": str(row["question"]).strip(),
            "choices": choices, "gold": gold, "spans": spans,
            "duration": annotation["duration"], "planned": planned})
    if len(result) != EXPECTED_N or len({x["video_id"] for x in result}) != EXPECTED_N:
        raise ValueError("v4_selected_item_denominator_invalid")
    return result


def frozen_media_sha256(base_plan: dict, video_id: str) -> str:
    """Read the media pin from the frozen plan, not the derived item object."""
    matches = [row.get("media_sha256") for row in base_plan.get("evaluation", [])
               if row.get("video_id") == video_id]
    if (len(matches) != 1 or not isinstance(matches[0], str)
            or len(matches[0]) != 64 or any(c not in "0123456789abcdef" for c in matches[0])):
        raise ValueError("v4_frozen_media_hash_missing_or_duplicate")
    return matches[0]


def geometry_preflight(processor, items: list[dict], expected_manifest: dict) -> list[dict]:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    plans = []
    for item in items:
        frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
            uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        batch, mapped, groups, qpos, token_ids, fingerprint = kernel.prepare_geometry(
            processor, item, frames, decode, torch, "annotation_union_only")
        evidence = validate_union_geometry(mapped, groups)
        planned = item["planned"]
        coverage = len(evidence) / len(groups)
        if (fingerprint != planned["geometry_sha256"] or len(groups) != planned["temporal_group_count"]
                or len(evidence) != planned["annotation_union_covered_group_count"]
                or coverage != planned["annotation_union_covered_fraction"]
                or (len(evidence) == len(groups)) != planned["annotation_union_full_coverage"]
                or not qpos or len(token_ids) != 5):
            raise ValueError("v4_geometry_changed_since_census")
        plans.append({"video_id": item["video_id"], "qid": item["qid"],
            "geometry_sha256": fingerprint, "group_count": len(groups),
            "covered_count": len(evidence), "covered_fraction": coverage,
            "full_coverage": len(evidence) == len(groups),
            "row_binding_sha256": planned["selected_row_binding_sha256"]})
        del batch, frames, decode
        gc.collect()
    if len(plans) != len(items) or len({x["video_id"] for x in plans}) != len(items):
        raise ValueError("v4_geometry_preflight_denominator_invalid")
    return plans


def validate_preflight(saved: dict, expected: dict) -> None:
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("v4_cpu_preflight_receipt_mismatch")
    rows = saved.get("geometry")
    if (not isinstance(rows, list) or len(rows) != EXPECTED_N
            or len({row.get("video_id") for row in rows if isinstance(row, dict)}) != EXPECTED_N):
        raise ValueError("v4_cpu_preflight_geometry_denominator_invalid")


def validate_smoke_receipt(smoke: dict, *, plan_sha256: str, sources: dict,
                           data_hashes: dict, processor_manifest: dict,
                           checkpoint_byte_audit: dict, full_preflight: dict,
                           first_plan_row: dict) -> None:
    if not isinstance(smoke, dict):
        raise ValueError("v4_gpu_smoke_receipt_missing_or_mismatched")
    full_first = full_preflight["geometry"][0]
    expected_smoke_preflight = {key: value for key, value in full_preflight.items()
                                if key != "geometry"}
    expected_smoke_preflight.update({"requested_n": 1, "completed_n": 1,
                                     "geometry": [full_first]})
    rows = smoke.get("records")
    integer_fields = ("requested_n", "effective_n", "requested_forwards",
                      "attempted_forwards", "completed_forwards",
                      "preflight_requested", "preflight_completed")
    counts_are_exact_ints = all(type(smoke.get(key)) is int for key in integer_fields)
    expected_branches = {name: {"requested": 1, "attempted": 1, "completed": 1}
                         for name in CHOICE_NAMES}
    branches = smoke.get("branch_counts")
    branch_counts_are_exact_ints = (
        isinstance(branches, dict) and set(branches) == set(CHOICE_NAMES)
        and all(isinstance(branches[name], dict)
                and set(branches[name]) == {"requested", "attempted", "completed"}
                and all(type(branches[name][field]) is int
                        for field in ("requested", "attempted", "completed"))
                for name in CHOICE_NAMES))
    if (smoke.get("status") != "v4_one_row_four_forward_gpu_smoke_complete"
            or not counts_are_exact_ints
            or smoke.get("requested_n") != 1 or smoke.get("effective_n") != 1
            or smoke.get("requested_forwards") != 4
            or smoke.get("attempted_forwards") != 4
            or smoke.get("completed_forwards") != 4
            or smoke.get("preflight_requested") != 1
            or smoke.get("preflight_completed") != 1
            or not branch_counts_are_exact_ints
            or branches != expected_branches
            or smoke.get("plan_sha256") != plan_sha256
            or smoke.get("census_sha256") != full_preflight.get("census_sha256")
            or smoke.get("source_sha256") != sources
            or smoke.get("data_sha256") != data_hashes
            or smoke.get("processor_manifest") != processor_manifest
            or smoke.get("checkpoint_byte_audit") != checkpoint_byte_audit
            or smoke.get("model_revision") != kernel.MODEL_REVISION
            or smoke.get("precision") != "4bit_bf16_compute_bf16_eager"
            or smoke.get("seed") != 20261018
            or smoke.get("completed_forwards") != 4
            or not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get("video_id") != first_plan_row["video_id"]
            or rows[0].get("qid") != first_plan_row["qid"]
            or rows[0].get("geometry_sha256") != first_plan_row["geometry_sha256"]
            or rows[0].get("row_binding_sha256") != first_plan_row["selected_row_binding_sha256"]
            or smoke.get("preflight_sha256") != canonical_sha256(expected_smoke_preflight)):
        raise ValueError("v4_gpu_smoke_receipt_missing_or_mismatched")


def score_row(model, processor, item: dict, expected: dict, branch_counts: dict) -> dict:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    from qwen_question_temporal_attention_selector import (
        aggregate_temporal_group_attention, select_top_temporal_groups)
    frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
        uniform_tolerance_seconds=UNIFORM_TOLERANCE)
    batch, mapped, groups, qpos, token_ids, fingerprint = kernel.prepare_geometry(
        processor, item, frames, decode, torch, "annotation_union_only")
    covered = validate_union_geometry(mapped, groups)
    if fingerprint != expected["geometry_sha256"] or len(groups) != expected["group_count"]:
        raise ValueError("v4_score_geometry_mismatch")
    capture = {}
    def forward(name, **kwargs):
        branch_counts[name]["attempted"] += 1
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 2 * 1024**3:
            raise RuntimeError("v4_per_forward_free_vram_guard_failed")
        output = kernel._forward_score(model, batch, token_ids, torch, **kwargs)
        branch_counts[name]["completed"] += 1
        torch.cuda.empty_cache()
        return output
    outputs = {"baseline_full": forward("baseline_full", question_positions=qpos,
        groups=groups, capture=capture)}
    expected_layers = tuple(i - 1 for i in kernel.LAYERS_1_BASED)
    if tuple(sorted(capture)) != expected_layers:
        raise ValueError("v4_baseline_attention_capture_incomplete")
    scores = [capture[layer] for layer in sorted(capture)]
    group_scores = [sum(layer[g] for layer in scores)/len(scores) for g in range(len(groups))]
    selected_groups = select_top_temporal_groups(group_scores,
        top_k=max(1, math.ceil(len(groups) * 0.10)))
    bias_positions = tuple(sorted(pos for idx in selected_groups for pos in groups[idx]))
    mask_positions = mapped["evidence_token_positions"]
    outputs["baseline_union_mask"] = forward("baseline_union_mask", ablation_positions=mask_positions)
    outputs["intervention_full"] = forward("intervention_full", bias_positions=bias_positions)
    outputs["intervention_union_mask"] = forward("intervention_union_mask",
        bias_positions=bias_positions, ablation_positions=mask_positions)
    scored = score_four_condition_logits(condition_outputs=outputs, answer_token_ids=token_ids,
        input_lengths={name: int(batch["input_ids"].shape[1]) for name in CHOICE_NAMES},
        torch_module=torch)
    contrast = scored["contrast"]
    result = {"video_id": item["video_id"], "qid": item["qid"],
        "geometry_sha256": fingerprint,
        "delta_union_dependence": contrast["delta_union_dependence"],
        "margin_by_condition": contrast["margin_by_condition"],
        "baseline_union_dependence": contrast["baseline_union_dependence"],
        "intervention_union_dependence": contrast["intervention_union_dependence"],
        "baseline_full_correct": int(contrast["answer_index"] == item["gold"]),
        "answer_index": contrast["answer_index"], "foil_index": contrast["foil_index"],
        "temporal_group_count": len(groups),
        "annotation_union_covered_group_count": len(covered),
        "annotation_union_covered_fraction": len(covered)/len(groups),
        "annotation_union_full_coverage": len(covered) == len(groups),
        "qta_selected_group_count": len(selected_groups),
        "row_binding_sha256": item["planned"]["selected_row_binding_sha256"]}
    del outputs, batch, frames, decode
    gc.collect()
    return result


def assert_forward_denominator(branch_counts: dict, n: int) -> None:
    if (set(branch_counts) != set(CHOICE_NAMES)
            or any(x["requested"] != n or x["attempted"] != n or x["completed"] != n
                   for x in branch_counts.values())
            or sum(x["completed"] for x in branch_counts.values()) != 4*n):
        raise ValueError("v4_exact_four_forward_denominator_failed")


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only: parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]; private = repo / PRIVATE_REL
    if Path(args.output_name).name != args.output_name: parser.error("output_name_must_be_basename")
    if args.mode == "full" and (not args.output_name.startswith(OUTPUT_PREFIX)
                                  or args.output_name in {PREFLIGHT_NAME, SMOKE_NAME}):
        parser.error("full_output_name_must_be_new_versioned_private_name")
    if args.mode == "preflight" and args.output_name != PREFLIGHT_NAME:
        parser.error("preflight_requires_exact_versioned_receipt_name")
    if args.mode == "smoke" and args.output_name != SMOKE_NAME:
        parser.error("smoke_requires_exact_versioned_receipt_name")
    try:
        import os
        if os.path.lexists(private / args.output_name): raise ValueError("output_collision")
    except OSError as exc:
        parser.error(f"private_output_check_failed:{type(exc).__name__}")
    counters = {name: {"requested": 1 if args.mode == "smoke" else EXPECTED_N,
                       "attempted": 0, "completed": 0} for name in CHOICE_NAMES}
    preflight_attempted = preflight_completed = 0
    stage = "load_frozen_plan"; started = time.monotonic()
    try:
        base, gate, plan, census = load_v4_plan(repo)
        items = selected_items(repo, base, plan)
        n = 1 if args.mode == "smoke" else EXPECTED_N
        selected = items[:n]
        if len(selected) != n: raise ValueError("v4_no_substitution_cohort_failure")
        sources = {name: sha256_file(repo / name) for name in SOURCE_FILES}
        data_hashes = {name: sha256_file(repo / name) for name in DATA_FILES}
        if (sources != plan["provenance"]["source_sha256"]
                or data_hashes != plan["provenance"]["data_sha256"]):
            raise ValueError("v4_plan_source_or_data_hash_changed")
        model_path = (Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            /"snapshots"/kernel.MODEL_REVISION)
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256*28*28, max_pixels=256*28*28)
        manifest = processor_geometry_provenance(processor, model_path)
        if manifest != plan["provenance"]["processor_manifest"]:
            raise ValueError("v4_processor_provenance_changed")
        stage = "complete_cpu_geometry_preflight"
        preflight_header = {"status": "v4_full_cpu_geometry_preflight_complete_non_authorizing",
            "mode": "cohort_geometry", "requested_n": n, "completed_n": n,
            "plan_sha256": sha256_file(private/PLAN_NAME),
            "census_sha256": sha256_file(private/CENSUS_NAME),
            "source_sha256": sources, "data_sha256": data_hashes,
            "processor_manifest": manifest, "library_versions": {
                name: importlib.metadata.version(name) for name in ("torch","transformers","av","qwen-vl-utils")},
            "scope": "geometry only; no model weights, CUDA, or scores"}
        if args.mode == "preflight":
            geometries = geometry_preflight(processor, selected, manifest)
            preflight_attempted = preflight_completed = len(geometries)
            preflight = {**preflight_header, "geometry": geometries}
            write_private(repo, args.output_name, preflight)
            print(json.dumps({"status": preflight["status"], "requested_n": n,
                "completed_n": len(geometries), "private_output_name": args.output_name}, sort_keys=True))
            return 0
        if args.mode == "full":
            saved = private_json(private/PREFLIGHT_NAME, private)
            validate_preflight(saved, preflight_header)
            geometries = saved["geometry"]
            expected_rows = [{"video_id": item["video_id"], "qid": item["qid"],
                "geometry_sha256": item["planned"]["geometry_sha256"],
                "row_binding_sha256": item["planned"]["selected_row_binding_sha256"]}
                for item in selected]
            actual_rows = [{key: row.get(key) for key in
                ("video_id", "qid", "geometry_sha256", "row_binding_sha256")} for row in geometries]
            if actual_rows != expected_rows:
                raise ValueError("v4_cpu_preflight_rows_do_not_match_frozen_plan")
            preflight = {**preflight_header, "geometry": geometries}
            stage = "re_audit_pinned_checkpoint_before_smoke_acceptance"
            checkpoint_audit = kernel.audit_pinned_checkpoint(repo, model_path)
            smoke = private_json(private/SMOKE_NAME, private)
            validate_smoke_receipt(smoke, plan_sha256=preflight["plan_sha256"],
                sources=sources, data_hashes=data_hashes, processor_manifest=manifest,
                checkpoint_byte_audit=checkpoint_audit,
                full_preflight=preflight, first_plan_row=selected[0]["planned"])
        else:
            geometries = geometry_preflight(processor, selected, manifest)
            preflight_attempted = preflight_completed = len(geometries)
            preflight = {**preflight_header, "geometry": geometries}
        if args.mode == "preflight":
            raise AssertionError("unreachable_preflight_mode")
        stage = "cuda_bf16_vram_checkpoint_gate"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA_BF16_required_no_CPU_fallback")
        free, total = torch.cuda.mem_get_info()
        if free < 11*1024**3: raise ValueError("insufficient_free_vram_before_model_load")
        if args.mode != "full":
            checkpoint_audit = kernel.audit_pinned_checkpoint(repo, model_path)
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        stage = "load_pinned_bf16_compute_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(kernel.LAYERS_1_BASED)):
            raise ValueError("v4_model_processor_layer_contract_invalid")
        torch.manual_seed(20261018); torch.cuda.manual_seed_all(20261018)
        torch.cuda.reset_peak_memory_stats()
        from claim_mask_video_decoder import decode_sampled_video
        from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
        results=[]; stage="four_branch_scoring"
        for item, expected in zip(selected, geometries, strict=True):
            row=score_row(model, processor, item, expected, counters); results.append(row)
        torch.cuda.synchronize(); assert_forward_denominator(counters, n)
        if len(results) != n: raise ValueError("v4_result_denominator_invalid")
        del model; gc.collect(); torch.cuda.empty_cache()
        values=[r["delta_union_dependence"] for r in results]
        payload={"status":"v4_one_row_four_forward_gpu_smoke_complete" if args.mode=="smoke" else "v4_completed_test_associated_exploratory_finite_cohort",
            "protocol":PROTOCOL,"scope":"annotation-mask prediction-margin dependence; not generated-claim faithfulness, clean/fresh, or confirmatory",
            "rights_status":"unverified; local-only protected data","requested_n":n,"effective_n":len(results),
            "requested_forwards":4*n,"attempted_forwards":sum(x["attempted"] for x in counters.values()),
            "completed_forwards":sum(x["completed"] for x in counters.values()),"branch_counts":counters,
            "preflight_requested": n, "preflight_completed": len(geometries),
            "records":results,"delta_union_dependence_summary":{"mean":sum(values)/n,
                "sd_sample":(sum((v-sum(values)/n)**2 for v in values)/(n-1))**0.5 if n>1 else None,
                "min":min(values),"max":max(values),"sign_counts":{"positive":sum(v>0 for v in values),
                    "negative":sum(v<0 for v in values),"zero":sum(v==0 for v in values)}},
            "baseline_full_choice_correct":sum(r["baseline_full_correct"] for r in results),
            "inferential_status":"finite cohort descriptive only; no p-value or population CI",
            "plan_sha256":sha256_file(private/PLAN_NAME),"census_sha256":preflight["census_sha256"],
            "preflight_sha256":sha256_file(private/PREFLIGHT_NAME) if args.mode=="full" else canonical_sha256(preflight),
            "gpu_smoke_sha256":sha256_file(private/SMOKE_NAME) if args.mode=="full" else None,
            "source_sha256":sources,"data_sha256":data_hashes,"processor_manifest":manifest,
            "checkpoint_byte_audit":checkpoint_audit,"model_revision":kernel.MODEL_REVISION,
            "precision":"4bit_bf16_compute_bf16_eager","seed":20261018,
            "qta_layers_1_based":list(kernel.LAYERS_1_BASED),"qta_logit_bias":kernel.LOGIT_BIAS,
            "hardware":{"device":torch.cuda.get_device_name(),"cuda":torch.version.cuda,
                "free_vram_before_load_gib":free/1024**3,"total_vram_gib":total/1024**3,
                "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3},
            "elapsed_seconds":time.monotonic()-started,
            "limitations":["test-associated exploratory cohort, exposure history unknown/contaminated",
                "annotation union may mask every temporal group; full coverage is reported, never excluded",
                "key ablation may produce out-of-distribution inputs"]}
        if args.mode=="smoke":
            payload["status"]="v4_one_row_four_forward_gpu_smoke_complete"
            payload["completed_forwards"]=4
        else: assert_forward_denominator(counters,EXPECTED_N)
        write_private(repo,args.output_name,payload)
        print(json.dumps({"status":payload["status"],"requested_n":n,"effective_n":len(results),
            "completed_forwards":payload["completed_forwards"],"private_output_name":args.output_name},sort_keys=True))
        return 0
    except Exception as exc:
        failure={"status":"failed_incomplete_no_v4_estimate","protocol":PROTOCOL,
            "requested_mode":args.mode,"requested_n":1 if args.mode=="smoke" else EXPECTED_N,
            "preflight_attempted":preflight_attempted,"preflight_completed":preflight_completed,
            "completed_forwards":sum(x["completed"] for x in counters.values()),
            "branch_counts":counters,"failure_code":safe_error_code(exc),"failure_stage":stage,
            "scope":"exploratory only; no faithfulness claim"}
        try:
            write_private(repo,args.output_name,failure)
        except FileExistsError:
            print(json.dumps({"status":"failed_output_name_race_no_overwrite",
                "requested_mode":args.mode,"private_output_name":args.output_name},sort_keys=True))
            return 2
        print(json.dumps({"status":failure["status"],"requested_mode":args.mode,
            "completed_forwards":failure["completed_forwards"],"failure_code":failure["failure_code"],
            "private_output_name":args.output_name},sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
