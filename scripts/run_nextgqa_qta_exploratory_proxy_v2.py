#!/usr/bin/env python3
"""Run the frozen v2 test-associated NExT-GQA QTA annotation proxy.

The proxy measures model-score dependence on gsub annotation-anchor video
groups; it is exploratory benchmark-annotation dependence, not generated-claim
faithfulness or confirmatory evidence. All media, scores, and receipts stay
local under the private-data directory. This CLI requires explicit local-only
authorization and writes exclusively to a new mode-0600 receipt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any

import run_nextgqa_validation_qta_annotation_proxy as kernel
from build_nextgqa_exploratory_qta_receipt import build_receipt

PLAN_NAME = "nextgqa_tlvs_exploratory_split_v2_20261018_05.json"
RECEIPT_NAME = "nextgqa_tlvs_exploratory_qta_preflight_receipt_v2_20261018_06.json"
INPUT_FREEZE_NAME = "nextgqa_tlvs_qta_v2_run_input_freeze_local_20261018_17.json"
PREFLIGHT_ONLY_NAME = "nextgqa_tlvs_qta_v2_geometry_preflight_local_20261018_01.json"
V1_PLAN_NAME = "nextgqa_tlvs_exploratory_split_20261018_01.json"
V1_FREEZE_NAME = "nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"
DATA_REL = Path("NExT-GQA/datasets/nextgqa")
PRIVATE_REL = Path("data/private_nextgqa")
POLICY = "single_overlap_anchor"
REQUESTED_FULL_N = 147
BRANCH_NAMES = kernel.BRANCH_NAMES
SOURCE_FILES = (
    "scripts/run_nextgqa_qta_exploratory_proxy_v2.py",
    "scripts/run_nextgqa_qta_v4_union_geometry_census.py",
    "scripts/run_nextgqa_qta_v4_union_scorer.py",
    "scripts/run_nextgqa_qta_v3_geometry_census.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/audit_qwen_checkpoint_bytes_cpu.py",
    "scripts/build_nextgqa_exploratory_video_split.py",
    "scripts/fixed_prediction_selective_dependence.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/nextgqa_atomic_batch_freeze.py",
    "scripts/presentation_time_utils.py",
    "scripts/qwen_internal_attention_ablation.py",
    "scripts/qwen_visual_attention_boost.py",
    "experiments/2026-10-18_qta_v4_union_treatment.md",
    "scripts/fixed_annotation_union_dependence.py",
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/qwen_visual_attention_boost.py",
    "scripts/qwen_internal_attention_ablation.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/fixed_prediction_selective_dependence.py",
    "scripts/audit_qwen_checkpoint_bytes_cpu.py",
    "scripts/build_nextgqa_exploratory_video_split.py",
    "scripts/build_nextgqa_exploratory_video_split_v2.py",
    "scripts/build_nextgqa_exploratory_qta_receipt.py",
    "scripts/freeze_nextgqa_exploratory_v1_local.py",
    "scripts/freeze_nextgqa_qta_v2_inputs_local.py",
    "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
    "experiments/2026-10-18_qta_treatment_selection.md",
    "experiments/2026-10-18_qta_exploratory_qid_valid_preflight_v2.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_private_exact(path: Path, expected_sha256: str | None, *, private_root: Path) -> bytes:
    root = private_root.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or path.resolve(strict=True).parent != root:
        raise ValueError("private_source_root_or_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_source_file_mode_invalid")
        parts = []
        while data := os.read(fd, 1024 * 1024):
            parts.append(data)
        raw = b"".join(parts)
    finally:
        os.close(fd)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("private_source_hash_mismatch")
    return raw


def load_verified_plan_and_receipt(repo: Path) -> tuple[dict, dict, dict, dict]:
    private = repo / PRIVATE_REL
    from freeze_nextgqa_exploratory_v1_local import read_and_verify_freeze
    freeze = read_and_verify_freeze(repo)
    gate = json.loads(read_private_exact(private / INPUT_FREEZE_NAME, None,
                                         private_root=private).decode("utf-8"))
    if gate.get("status") != "local_v2_inputs_frozen_integrity_only_not_authorization":
        raise ValueError("local_input_freeze_status_invalid")
    plan_hash, receipt_hash = gate.get("plan_sha256"), gate.get("preflight_receipt_sha256")
    freeze_file_sha = sha256_file(private / V1_FREEZE_NAME)
    if (not isinstance(plan_hash, str) or len(plan_hash) != 64
            or not isinstance(receipt_hash, str) or len(receipt_hash) != 64
            or gate.get("v1_plan_sha256") != freeze.get("plan_sha256")
            or gate.get("v1_freeze_receipt_sha256") != freeze_file_sha):
        raise ValueError("local_input_freeze_hash_contract_invalid")
    plan_raw = read_private_exact(private / PLAN_NAME, plan_hash, private_root=private)
    receipt_raw = read_private_exact(private / RECEIPT_NAME, receipt_hash, private_root=private)
    v1_raw = read_private_exact(private / V1_PLAN_NAME, freeze["plan_sha256"], private_root=private)
    plan, receipt, v1_plan = (json.loads(raw.decode("utf-8"))
                               for raw in (plan_raw, receipt_raw, v1_raw))
    sources = gate.get("source_sha256", {})
    data_hashes = gate.get("data_sha256", {})
    if (set(sources) != set(SOURCE_FILES)
            or set(data_hashes) != {"NExT-GQA/datasets/nextgqa/test.csv",
                "NExT-GQA/datasets/nextgqa/gsub_test.json",
                "NExT-GQA/datasets/nextgqa/frame2time_test.json",
                "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"}
            or {name: sha256_file(repo / name) for name in sources} != sources
            or {name: sha256_file(repo / name) for name in data_hashes} != data_hashes):
        raise ValueError("local_input_freeze_source_or_data_changed")
    if (plan.get("status") != "test_associated_exploratory_qid_valid_plan_not_evaluation"
            or receipt.get("status") != "preflight_receipt_only_no_model_scoring"
            or plan.get("provenance", {}).get("v1_plan_sha256") != freeze["plan_sha256"]
            or plan.get("provenance", {}).get("v1_freeze_receipt_sha256") != freeze_file_sha
            or plan.get("provenance", {}).get("v1_splitter_sha256") != sources.get(
                "scripts/build_nextgqa_exploratory_video_split.py")
            or plan.get("provenance", {}).get("v2_builder_sha256") != sources.get(
                "scripts/build_nextgqa_exploratory_video_split_v2.py")
            or receipt.get("counts", {}).get("evaluation_rows_bound") != REQUESTED_FULL_N
            or receipt.get("counts", {}).get("invalid_annotation_intervals") != 0
            or receipt.get("counts", {}).get("join_failures") != 0):
        raise ValueError("frozen_preflight_receipt_contract_invalid")
    if (receipt.get("input_sha256", {}).get("split_plan") != plan_hash
            or receipt.get("input_sha256", {}).get("v1_plan") != freeze["plan_sha256"]
            or receipt.get("input_sha256", {}).get("v1_freeze_receipt") != freeze_file_sha):
        raise ValueError("receipt_source_hash_binding_mismatch")
    paths = {Path(name).name: repo / name for name in data_hashes}
    rebuilt = build_receipt(test_csv=paths["test.csv"], gsub_json=paths["gsub_test.json"],
        frame2time_json=paths["frame2time_test.json"], mapping_json=paths["map_vid_vidorID.json"],
        split_plan=private / PLAN_NAME,
        splitter_source=repo / "scripts/build_nextgqa_exploratory_video_split_v2.py",
        base_plan=private / V1_PLAN_NAME,
        v1_splitter=repo / "scripts/build_nextgqa_exploratory_video_split.py",
        expected_v1_plan_sha256=freeze["plan_sha256"],
        v1_freeze_receipt_sha256=freeze_file_sha)
    if rebuilt != receipt:
        raise ValueError("preflight_receipt_reconstruction_mismatch")
    if (plan.get("counts", {}).get("calibration_videos") != 343
            or plan.get("counts", {}).get("evaluation_videos") != REQUESTED_FULL_N
            or len(plan.get("calibration", [])) != 343
            or len(plan.get("evaluation", [])) != REQUESTED_FULL_N):
        raise ValueError("frozen_plan_denominator_invalid")
    return plan, receipt, v1_plan, gate


def build_evaluation_items(repo: Path, plan: dict) -> list[dict[str, Any]]:
    dataset = repo / DATA_REL
    with (dataset / "test.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_pair = {}
    for row in rows:
        key = (str(row.get("video_id", "")), str(row.get("qid", "")))
        if not all(key) or key in by_pair:
            raise ValueError("test_csv_pair_missing_or_duplicate")
        by_pair[key] = row
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    selected, seen_video, seen_pair, seen_media = [], set(), set(), set()
    for entry in plan["evaluation"]:
        vid, qid, media_hash = entry["video_id"], entry["qid"], entry["media_sha256"]
        if vid in seen_video or (vid, qid) in seen_pair or media_hash in seen_media:
            raise ValueError("evaluation_identity_or_media_duplicate")
        seen_video.add(vid); seen_pair.add((vid, qid)); seen_media.add(media_hash)
        row = by_pair.get((vid, qid))
        if row is None:
            raise ValueError("evaluation_csv_join_missing")
        choices, gold_index = choice_gold_from_row(row)
        annotation = gsub.get(vid)
        if not isinstance(annotation, dict) or qid not in annotation.get("location", {}):
            raise ValueError("evaluation_gsub_join_missing")
        spans = annotation["location"][qid]
        media_path = Path(entry["media_path"])
        if media_path.stem != vid or sha256_file(media_path) != media_hash:
            raise ValueError("evaluation_media_binding_invalid")
        selected.append({"video_id": vid, "qid": qid,
            "question": str(row["question"]).strip(), "choices": choices,
            "gold": gold_index, "spans": spans,
            "duration": annotation["duration"], "video_path": media_path})
    if len(selected) != REQUESTED_FULL_N:
        raise ValueError("evaluation_full_denominator_invalid")
    return selected


def choice_gold_from_row(row: dict[str, Any]) -> tuple[list[str], int]:
    choices = [str(row.get(f"a{i}", "")).strip() for i in range(5)]
    answer = str(row.get("answer", "")).strip()
    if any(not x for x in choices) or len(set(choices)) != 5 or not answer:
        raise ValueError("evaluation_AE_choices_invalid")
    matches = [i for i, value in enumerate(choices) if value == answer]
    if len(matches) != 1:
        raise ValueError("evaluation_gold_choice_mapping_invalid")
    return choices, matches[0]


def baseline_accuracy(rows: list[dict], requested_n: int) -> dict:
    if (len(rows) != requested_n or any(type(row.get("baseline_full_correct")) is not int
            or row["baseline_full_correct"] not in (0, 1) for row in rows)):
        raise ValueError("baseline_accuracy_denominator_invalid")
    n_correct = sum(row["baseline_full_correct"] for row in rows)
    return {"n": requested_n, "correct": n_correct,
            "accuracy": n_correct / requested_n,
            "interpretation": "descriptive A-E benchmark accuracy, not faithfulness"}


def finite_cohort_delta_f_summary(rows: list[dict]) -> dict:
    values = [row.get("delta_f") for row in rows]
    if (not values or any(type(v) not in (int, float) or not math.isfinite(float(v))
                          for v in values)):
        raise ValueError("finite_cohort_delta_f_invalid")
    values = [float(v) for v in values]
    return {**kernel.summarize(values), "sign_counts": {
        "positive": sum(v > 0 for v in values),
        "negative": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values)}}


def geometry_preflight_receipt(*, input_gate: dict, plans: list[dict], requested_n: int) -> dict:
    validate_preflight_plans(plans, requested_n)
    return {"status": "cpu_geometry_preflight_complete_not_authorization_or_model_result",
        "scope": "test-associated exploratory QTA proxy only",
        "requested_n": requested_n, "effective_n": len(plans),
        "plan_sha256": input_gate["plan_sha256"],
        "preflight_receipt_sha256": input_gate["preflight_receipt_sha256"],
        "v1_plan_sha256": input_gate["v1_plan_sha256"],
        "v1_freeze_receipt_sha256": input_gate["v1_freeze_receipt_sha256"],
        "source_sha256": input_gate["source_sha256"],
        "data_sha256": input_gate["data_sha256"],
        "geometry_plans": plans,
        "interpretation": "CPU decoding/tokenization geometry gate only; no model loaded and no score produced"}


def validate_geometry_preflight_receipt(saved: dict, expected: dict) -> None:
    if saved != expected:
        raise ValueError("geometry_preflight_receipt_mismatch")


def preflight_geometry(torch, processor, items: list[dict], *, requested_n: int,
                       counters: dict) -> list[dict]:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    plans = []
    for item in items[:requested_n]:
        counters["attempted"] += 1
        frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
            uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        batch, mapped, groups, _qpos, _tokens, fingerprint = kernel.prepare_geometry(
            processor, item, frames, decode, torch, POLICY)
        evidence, control = mapped["evidence_group_indices"], mapped["control_group_indices"]
        if (len(evidence) != 1 or len(control) != 1 or set(evidence) & set(control)
                or any(idx >= len(groups) for idx in (*evidence, *control))):
            raise ValueError("single_anchor_preflight_contract_invalid")
        plans.append({"geometry_sha256": fingerprint, "group_count": len(groups),
            "evidence_count": 1, "control_count": 1,
            "representative_overlap_fraction": mapped["representative_annotation_overlap_duration_fraction"],
            "representative_position_fraction": mapped["representative_annotation_group_index"] / max(1, len(groups)-1)})
        counters["completed"] += 1
        del batch, frames, decode
        gc.collect()
    validate_preflight_plans(plans, requested_n)
    if counters["completed"] != requested_n:
        raise ValueError("single_anchor_preflight_denominator_invalid")
    return plans


def validate_preflight_plans(plans: list[dict], requested_n: int) -> None:
    if (type(requested_n) is not int or requested_n not in (1, REQUESTED_FULL_N)
            or len(plans) != requested_n
            or any(not isinstance(row, dict)
                or not isinstance(row.get("geometry_sha256"), str)
                or len(row["geometry_sha256"]) != 64
                or row.get("group_count", 0) <= 0
                or row.get("evidence_count") != 1
                or row.get("control_count") != 1 for row in plans)):
        raise ValueError("single_anchor_preflight_denominator_invalid")


def validate_scored_denominator(outputs: list[dict], branch_counts: dict,
                                requested_n: int) -> None:
    if (len(outputs) != requested_n or set(branch_counts) != set(BRANCH_NAMES)
            or any(branch_counts[name].get("attempted") != requested_n
                   or branch_counts[name].get("completed") != requested_n
                   for name in BRANCH_NAMES)
            or any(not isinstance(row.get("delta_f"), (int, float))
                   or not math.isfinite(float(row["delta_f"])) for row in outputs)):
        raise ValueError("all_or_none_six_branch_denominator_failed")


def write_private_output(repo: Path, name: str, payload: dict) -> None:
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


def source_manifest(repo: Path) -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return commit, {relative: sha256_file(repo / relative) for relative in SOURCE_FILES}


def aggregate_summary(payload: dict, receipt_name: str) -> dict:
    return {"status": payload["status"], "requested_n": payload["requested_n"],
        "effective_n": payload.get("effective_n", 0), "failures": payload.get("failures", 1),
        "preflight_completed": payload.get("preflight_completed", 0),
        "completed_forwards": payload.get("completed_forwards", 0),
        "private_output_name": receipt_name}


def failure_payload(*, requested_n: int, stage: str, exc: Exception,
                    preflight_counts: dict, branch_counts: dict, sources: dict,
                    plan_sha256: str | None, receipt_sha256: str | None) -> dict:
    return {"status": "failed_incomplete_no_cohort_estimate",
        "scope": "NExT-GQA test-associated gsub annotation dependence proxy; not faithfulness",
        "prior_exposure_status": "known-exposure-contaminated/unknown-history exploratory; unmatched is not certified fresh",
        "requested_n": requested_n, "effective_n": 0, "failures": 1,
        "failure_stage": stage, "failure_type": type(exc).__name__,
        "preflight_attempted": preflight_counts["attempted"],
        "preflight_completed": preflight_counts["completed"],
        "branch_counts": {name: {**value, "effective": 0}
                          for name, value in branch_counts.items()},
        "records": [], "estimate": None, "rights_status": "unverified; local-only",
        "faithfulness_status": "not_measured", "source_sha256": sources,
        "plan_sha256": plan_sha256, "preflight_receipt_sha256": receipt_sha256}


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(1, 147), required=True,
                        help="1 is fixed-row smoke; 147 is the full frozen evaluation partition")
    parser.add_argument("--output-name", required=True,
                        help="new private filename directly under data/private_nextgqa")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run only CPU decoding/token geometry checks and write a private non-authorizing receipt")
    parser.add_argument("--preflight-receipt-name", default=None,
                        help="required for a full --n 147 scoring run; exact private CPU preflight receipt basename")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    smoke = args.n == 1
    if args.preflight_only and args.n != REQUESTED_FULL_N:
        parser.error("preflight_only_requires_full_frozen_partition_n147")
    if args.preflight_only and args.output_name != PREFLIGHT_ONLY_NAME:
        parser.error("preflight_only_requires_versioned_receipt_output_name")
    if not args.preflight_only and args.n == REQUESTED_FULL_N:
        if args.preflight_receipt_name != PREFLIGHT_ONLY_NAME:
            parser.error("full_run_requires_exact_preflight_receipt_name")
    if args.preflight_only and args.preflight_receipt_name is not None:
        parser.error("preflight_only_uses_output_name_for_receipt")
    branch_counts = {name: {"requested": args.n, "attempted": 0, "completed": 0}
                     for name in BRANCH_NAMES}
    preflight_counts = {"attempted": 0, "completed": 0}
    stage, outputs = "source_and_receipt_preflight", []
    sources = {}
    plan_sha256 = receipt_sha256 = None
    started, start = datetime.now(timezone.utc).isoformat(), time.monotonic()
    try:
        commit, sources = source_manifest(repo)
        plan, preflight_receipt, _v1, input_gate = load_verified_plan_and_receipt(repo)
        plan_sha256, receipt_sha256 = input_gate["plan_sha256"], input_gate["preflight_receipt_sha256"]
        items_all = build_evaluation_items(repo, plan)
        selected = items_all[:args.n]
        if len(selected) != args.n:
            raise ValueError("requested_cohort_unavailable_no_substitution")
        stage = "pinned_checkpoint_presence_check"
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / kernel.MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        import torch
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        processor_pins = kernel.processor_provenance(processor, model_path)
        stage = "complete_requested_cpu_geometry_preflight"
        plans = preflight_geometry(torch, processor, selected, requested_n=args.n,
                                   counters=preflight_counts)
        expected_geometry_receipt = geometry_preflight_receipt(
            input_gate=input_gate, plans=plans, requested_n=args.n)
        if args.preflight_only:
            write_private_output(repo, args.output_name, expected_geometry_receipt)
            print(json.dumps({"status": expected_geometry_receipt["status"],
                "requested_n": args.n, "effective_n": len(plans),
                "preflight_completed": preflight_counts["completed"],
                "private_output_name": args.output_name}, sort_keys=True))
            return 0
        if args.n == REQUESTED_FULL_N:
            saved_geometry_receipt = json.loads(read_private_exact(
                repo / PRIVATE_REL / PREFLIGHT_ONLY_NAME, None,
                private_root=repo / PRIVATE_REL).decode("utf-8"))
            validate_geometry_preflight_receipt(saved_geometry_receipt, expected_geometry_receipt)
        stage = "cuda_capability_check"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA_BF16_required_no_CPU_fallback")
        free, total = torch.cuda.mem_get_info()
        if free < 11 * 1024**3:
            raise ValueError("insufficient_free_vram_before_model_load")
        stage = "pinned_checkpoint_byte_audit"
        checkpoint_audit = kernel.audit_pinned_checkpoint(repo, model_path)
        stage = "model_load"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(kernel.LAYERS_1_BASED)):
            raise ValueError("model_processor_layer_contract_invalid")
        torch.manual_seed(20261018); torch.cuda.manual_seed_all(20261018)
        torch.cuda.reset_peak_memory_stats()
        from claim_mask_video_decoder import decode_sampled_video
        from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
        stage = "six_branch_scoring"
        for ordinal, (item, expected) in enumerate(zip(selected, plans, strict=True), 1):
            frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
                uniform_tolerance_seconds=UNIFORM_TOLERANCE)
            row = kernel.score_one(model, processor, torch, item, frames=frames, decode=decode,
                expected_plan=expected, branch_counts=branch_counts,
                evidence_selection_policy=POLICY)
            row["ordinal"] = ordinal
            outputs.append(row)
            del frames, decode
            gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize()
        validate_scored_denominator(outputs, branch_counts, args.n)
        delta_values = [float(r["delta_f"]) for r in outputs]
        payload = {
            "status": "engineering_smoke_only" if smoke else "completed_exploratory_finite_cohort",
            "scope": "NExT-GQA test-associated gsub annotation dependence proxy; not generated-claim faithfulness or confirmatory evaluation",
            "prior_exposure_status": "known-exposure-contaminated/unknown-history exploratory; supplied incomplete indices are not an exhaustive freshness certificate; no filtering or resampling by index match",
            "rights_status": "unverified; local-only protected data",
            "requested_n": args.n, "effective_n": len(outputs), "failures": 0,
            "cohort_partition": "frozen v2 evaluation partition; smoke uses first deterministic row only",
            "full_partition_n": REQUESTED_FULL_N,
            "preflight_receipt_sha256": input_gate["preflight_receipt_sha256"],
            "plan_sha256": input_gate["plan_sha256"],
            "source_input_sha256": {"data": input_gate["data_sha256"],
                "sources": input_gate["source_sha256"],
                "v1_plan": input_gate["v1_plan_sha256"],
                "v1_freeze_receipt": input_gate["v1_freeze_receipt_sha256"]},
            "preflight_requested": args.n, "preflight_attempted": preflight_counts["attempted"],
            "preflight_completed": preflight_counts["completed"],
            "branch_counts": {name: {**value, "effective": len(outputs)}
                              for name, value in branch_counts.items()},
            "requested_forwards": 6 * args.n,
            "attempted_forwards": sum(x["attempted"] for x in branch_counts.values()),
            "completed_forwards": sum(x["completed"] for x in branch_counts.values()),
            "primary": "delta_f=(QTA control-anchor margin - QTA evidence-anchor margin) - (baseline control-anchor margin - baseline evidence-anchor margin); fixed p/q from baseline full",
            "records": outputs,
            "finite_cohort_delta_f": None if smoke else finite_cohort_delta_f_summary(outputs),
            "baseline_full_restricted_choice_accuracy": baseline_accuracy(outputs, args.n),
            "inference": "engineering_only" if smoke else "descriptive_only_no_p_value_or_population_CI",
            "intervention": "QTA question-token attention boost, layers 5-16, fixed ln(2) logit bias, top max(1,ceil(0.10*T)) temporal groups; six forwards with equal single-anchor evidence/control attention-key ablations",
            "annotation_policy": POLICY,
            "limitations": ["Both calibration and evaluation are within official NExT-GQA test-associated media.",
                "Prior-exposure indices are incomplete and unauthenticated; some rows are known matches, unmatched rows are not certified fresh, and no rows were removed or resampled.",
                "gsub is benchmark annotation, not generated explanation claim support.",
                "frame2time is exactly bound metadata but not decoder/player clock calibration.",
                "No independent human claim/frame/prediction review; no direct faithfulness measure.",
                "Key ablation may induce distribution shift; finite-cohort exploratory only."],
            "started_utc": started, "elapsed_seconds": time.monotonic() - start,
            "hardware": {"device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                         "free_vram_before_load_gib": free / 1024**3,
                         "total_vram_gib": total / 1024**3,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "model_revision": kernel.MODEL_REVISION,
            "precision": "4bit_bf16_compute_bf16_eager", "processor_provenance": processor_pins,
            "checkpoint_byte_audit": checkpoint_audit,
            "seed": 20261018, "frame_count": kernel.FRAME_COUNT,
            "qta_layers_1_based": list(kernel.LAYERS_1_BASED), "qta_logit_bias": kernel.LOGIT_BIAS,
            "source_commit": commit, "source_sha256": sources,
            "environment": {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "bitsandbytes", "av", "qwen-vl-utils")},
            "faithfulness_status": "not_measured",
            "preflight_receipt_scope": preflight_receipt["scope"],
        }
    except Exception as exc:
        payload = failure_payload(requested_n=args.n, stage=stage, exc=exc,
            preflight_counts=preflight_counts, branch_counts=branch_counts, sources=sources,
            plan_sha256=plan_sha256, receipt_sha256=receipt_sha256)
        try:
            write_private_output(repo, args.output_name, payload)
        except Exception as write_exc:
            parser.error(f"private_failure_receipt_failed:{type(write_exc).__name__}")
        print(json.dumps(aggregate_summary(payload, args.output_name), sort_keys=True))
        return 2
    try:
        write_private_output(repo, args.output_name, payload)
    except Exception as exc:
        parser.error(f"private_output_failed:{type(exc).__name__}")
    print(json.dumps(aggregate_summary(payload, args.output_name), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
