#!/usr/bin/env python3
"""Exploratory GPU sensitivity to NOAA question-evidence timestamp groups.

This is not a claim-support/faithfulness evaluation. It compares the frozen
baseline top-option margin after equal-budget visual-key bias to timestamp-
mapped groups versus position-matched non-evidence groups.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import analyze_noaa_qta_annotation_time_alignment as alignment
import run_noaa_qta_distribution_pilot as base


def source_hashes(repo: Path) -> dict[str, str]:
    hashes = base.source_hashes(repo)
    hashes["scripts/run_noaa_annotation_matched_group_margin.py"] = base.sha256_file(
        repo / "scripts/run_noaa_annotation_matched_group_margin.py")
    return hashes


def matched_control_groups(evidence_groups: tuple[int, ...], group_count: int,
                           row_token: str) -> tuple[int, ...]:
    if (type(group_count) is not int or group_count < 2 or not evidence_groups
            or tuple(sorted(set(evidence_groups))) != evidence_groups
            or any(type(i) is not int or i < 0 or i >= group_count for i in evidence_groups)
            or len(evidence_groups) > group_count // 2
            or not isinstance(row_token, str) or not row_token):
        raise ValueError("evidence_group_contract_invalid")
    candidates = [candidate for candidate in itertools.combinations(
        (i for i in range(group_count) if i not in evidence_groups), len(evidence_groups))]
    if not candidates:
        raise ValueError("matched_control_unavailable")
    min_distance = min(sum(abs(e - c) for e, c in zip(evidence_groups, candidate, strict=True))
                       for candidate in candidates)
    nearest = [candidate for candidate in candidates
               if sum(abs(e - c) for e, c in zip(evidence_groups, candidate, strict=True)) == min_distance]
    prefix = f"noaa-annotation-position-match-v1\0{row_token}\0"
    return min(nearest, key=lambda candidate: hashlib.sha256(
        (prefix + ",".join(map(str, candidate))).encode()).digest())


def calculate_fixed_margin(scores: list[float], top_index: int, foil_index: int) -> float:
    if (not isinstance(scores, list) or len(scores) != 5 or top_index == foil_index
            or not 0 <= top_index < 5 or not 0 <= foil_index < 5
            or any(not math.isfinite(float(x)) for x in scores)):
        raise ValueError("fixed_margin_inputs_invalid")
    return float(scores[top_index]) - float(scores[foil_index])


def load_annotation_points(repo: Path) -> tuple[dict[tuple[str, str], tuple[list[float], str]], dict[str, str]]:
    data_root = repo / "datasets/noaa_qa_pilot_20261011"
    result, hashes = {}, {}
    for sample_rel, audit_rel, method in alignment.INPUTS:
        sample, sample_hash = alignment.read_private_json(repo, Path(sample_rel))
        audit, audit_hash = alignment.read_private_json(repo, Path(audit_rel))
        hashes[sample_rel], hashes[audit_rel] = sample_hash, audit_hash
        if (not isinstance(sample, list) or not sample or not isinstance(audit, dict)
                or not isinstance(audit.get("items"), list)
                or audit.get("media_sha256") != sample[0].get("video_sha256")):
            raise ValueError("annotation_source_binding_invalid")
        by_qid = {item.get("qid"): item for item in audit["items"]
                  if isinstance(item, dict) and isinstance(item.get("qid"), str)}
        if len(by_qid) != len(audit["items"]):
            raise ValueError("annotation_qid_inventory_invalid")
        video_sha = audit["media_sha256"]
        for row in sample:
            if (not isinstance(row, dict) or row.get("video_sha256") != video_sha
                    or row.get("qid") not in by_qid):
                raise ValueError("sample_annotation_join_invalid")
            evidence = by_qid[row["qid"]].get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("annotation_evidence_missing")
            points = [float(x["timestamp_sec"]) for x in evidence
                      if isinstance(x, dict)
                      and isinstance(x.get("timestamp_sec"), (int, float))
                      and not isinstance(x.get("timestamp_sec"), bool)]
            if not points:
                raise ValueError("annotation_evidence_points_missing")
            key = (video_sha, base.sha256_bytes(row["qid"].encode()))
            if key in result:
                raise ValueError("duplicate_annotation_row")
            result[key] = (points, method)
    if len(result) != 6:
        raise ValueError("fixed_six_annotation_rows_required")
    return result, hashes


def run(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "datasets/noaa_qa_pilot_20261011"
    output = Path(args.output)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("output_path_invalid")
    output = repo / output
    if not output.parent.resolve().is_relative_to((data_root / "private").resolve()):
        raise ValueError("output_must_remain_private")
    if args.mode == "full":
        if not args.smoke_receipt:
            raise ValueError("full_run_requires_smoke_receipt")
        smoke_path = repo / args.smoke_receipt
        if (smoke_path.is_symlink() or not smoke_path.resolve(strict=True).is_relative_to(data_root / "private")):
            raise ValueError("smoke_path_invalid")
        smoke = json.loads(smoke_path.read_text())
        if smoke.get("status") != "completed_smoke" or smoke.get("completed_forwards") != 3:
            raise ValueError("matched_group_smoke_not_passed")
    started = datetime.now(timezone.utc).isoformat()
    monotonic = time.monotonic()
    counters = {"attempted": 0, "completed": 0}
    stage = "load_and_bind_local_data"
    try:
        records, sample_hashes = base.load_samples(repo)
        annotations, annotation_hashes = load_annotation_points(repo)
        if set(sample_hashes) != {x[0] for x in alignment.INPUTS}:
            raise ValueError("sample_manifest_contract_invalid")
        if args.mode == "full":
            smoke = json.loads(smoke_path.read_text())
            if (smoke.get("source_sha256") != source_hashes(repo)
                    or smoke.get("sample_manifest_sha256") != sample_hashes
                    or smoke.get("protocol_sha256") != base.sha256_bytes(b"annotation_matched_group_margin_v1")
                    or smoke.get("annotation_sha256") != annotation_hashes
                    or smoke.get("model_revision") != base.MODEL_REVISION):
                raise ValueError("smoke_protocol_or_input_mismatch")
        selected = records[:1] if args.mode == "smoke" else records
        import torch
        import psutil
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_available = int(psutil.virtual_memory().available)
        if free < base.MIN_FREE_VRAM or host_available < base.MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / base.MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        import claim_mask_video_decoder as decoder
        stage = "cpu_preflight_selected_rows"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        decoded_cache = {}
        prepared = [base._prepare_record(row, processor, process_vision_info,
            torch, decoder, decoded_cache) for row in selected]
        for record, inputs in zip(selected, prepared, strict=True):
            key = (record["video_sha256"], record["qid_sha256"])
            if key not in annotations:
                raise ValueError("annotation_run_join_missing")
            points, method = annotations[key]
            groups = len(inputs["groups"])
            pts = [float(__import__("fractions").Fraction(*x["relative_presentation_time"]))
                   for x in inputs["decoded"]["sampled_frames"]]
            evidence = tuple(sorted({alignment.nearest_temporal_group(t, pts, groups) for t in points}))
            controls = matched_control_groups(evidence, groups, record["row_token"])
            if len(evidence) != len(controls):
                raise ValueError("matched_group_budget_mismatch")
            inputs["evidence_groups"] = evidence
            inputs["control_groups"] = controls
            inputs["annotation_method"] = method
            inputs["evidence_points_sha256"] = base.sha256_bytes(
                json.dumps(sorted(points), separators=(",", ":")).encode())
        if args.mode == "full" and selected[0]["row_token"] != smoke.get("smoke_row_token"):
            raise ValueError("smoke_row_binding_mismatch")
        stage = "load_pinned_qwen"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for record, inputs in zip(selected, prepared, strict=True):
            batch = inputs["batch"].to("cuda")
            if base._tensor_digest(batch, torch) != inputs["input_sha256"]:
                raise ValueError("frozen_processor_batch_changed")
            seq_len = int(batch["input_ids"].shape[1])
            counters["attempted"] += 1
            with torch.inference_mode():
                baseline = model(**batch, use_cache=False, logits_to_keep=1)
            counters["completed"] += 1
            base_scores = base._score(baseline.logits, inputs["answer_ids"], torch)
            top = max(range(5), key=lambda i: (base_scores[i], -i))
            foil = max((i for i in range(5) if i != top), key=lambda i: (base_scores[i], -i))
            del baseline; torch.cuda.empty_cache()
            evidence_keys = tuple(sorted(p for i in inputs["evidence_groups"] for p in inputs["groups"][i]))
            control_keys = tuple(sorted(p for i in inputs["control_groups"] for p in inputs["groups"][i]))
            evidence_scores, ev_audit = base._intervention_forward(model, batch,
                key_positions=evidence_keys, seq_len=seq_len, answer_ids=inputs["answer_ids"],
                torch=torch, counters=counters, protocol_name="annotation_evidence_groups")
            control_scores, ctrl_audit = base._intervention_forward(model, batch,
                key_positions=control_keys, seq_len=seq_len, answer_ids=inputs["answer_ids"],
                torch=torch, counters=counters, protocol_name="position_matched_non_evidence_control")
            b = calculate_fixed_margin(base_scores, top, foil)
            e = calculate_fixed_margin(evidence_scores, top, foil)
            c = calculate_fixed_margin(control_scores, top, foil)
            rows.append({"row_token_sha256": record["row_token"],
                "source_video_sha256": record["video_sha256"], "qid_sha256": record["qid_sha256"],
                "annotation_method": inputs["annotation_method"],
                "annotation_points_sha256": inputs["evidence_points_sha256"],
                "frame_count": len(inputs["frames"]), "grid_thw": inputs["grid"],
                "sampled_pts_seconds": [float(__import__("fractions").Fraction(*x["relative_presentation_time"])) for x in inputs["decoded"]["sampled_frames"]],
                "evidence_group_indices": list(inputs["evidence_groups"]),
                "matched_control_group_indices": list(inputs["control_groups"]),
                "frozen_top_choice_index": top, "frozen_foil_choice_index": foil,
                "baseline_fixed_margin": b, "evidence_bias_fixed_margin": e,
                "matched_control_bias_fixed_margin": c,
                "evidence_minus_control_margin": e - c,
                "intervention_audits": {"evidence": ev_audit, "control": ctrl_audit}})
            del batch; torch.cuda.empty_cache()
        expected = len(selected) * 3
        if counters["attempted"] != expected or counters["completed"] != expected:
            raise RuntimeError("incomplete_forward_counts")
        payload = {"schema_version": 1,
            "status": "completed_smoke" if args.mode == "smoke" else "completed_exploratory_posthoc",
            "purpose": "exploratory_question_prediction_sensitivity_to_annotated_time_groups",
            "dataset": "local NOAA EX2503 pilot; local-only under user authorization",
            "run_mode": args.mode, "started_utc": started,
            "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "seed": base.SEED, "layers_1_based": list(base.QTA_LAYERS),
            "logit_bias": base.LOGIT_BIAS,
            "protocol_sha256": base.sha256_bytes(b"annotation_matched_group_margin_v1"),
            "source_sha256": source_hashes(repo),
            "sample_manifest_sha256": sample_hashes, "annotation_sha256": annotation_hashes,
            "smoke_row_token": selected[0]["row_token"],
            "requested_rows": 1 if args.mode == "smoke" else 6,
            "effective_rows": len(rows), "attempted_forwards": counters["attempted"],
            "completed_forwards": counters["completed"], "rows": rows,
            "summary": {"mean_evidence_minus_control_margin": sum(x["evidence_minus_control_margin"] for x in rows) / len(rows),
                "positive_rows": sum(x["evidence_minus_control_margin"] > 0 for x in rows),
                "negative_rows": sum(x["evidence_minus_control_margin"] < 0 for x in rows),
                "inference": "posthoc six questions in three videos; descriptive only; no CI, test, or population inference"},
            "gpu": {"name": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                "free_vram_before_load_bytes": int(free), "total_vram_bytes": int(total),
                "host_available_before_load_bytes": host_available,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated())},
            "elapsed_seconds": round(time.monotonic() - monotonic, 3),
            "human_review_performed": False, "gold_used": False,
            "direct_faithfulness": False,
            "rights_status": "unverified; local-only under user authorization",
            "limits": ["annotation timestamps are QA evidence points, not reviewed explanation-claim support spans",
                "same six outcome-exposed rows as prior pilot; exploratory/post-hoc only",
                "coarse frame-to-group mapping; prompt may reveal the question time",
                "position matching does not control visual salience or semantic content",
                "restricted-choice fixed top-vs-foil margin is not answer accuracy or direct faithfulness"]}
        output_sha = base._write_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "rows": len(rows),
            "completed_forwards": counters["completed"], "output_sha256": output_sha,
            "output_relative_path": output.relative_to(repo).as_posix(),
            "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"status": "failed_incomplete_no_effect_estimate", "stage": stage,
            "failure_type": type(exc).__name__, "run_mode": args.mode,
            "attempted_forwards": counters["attempted"], "completed_forwards": counters["completed"],
            "direct_faithfulness": False}
        if not output.exists():
            base._write_exclusive(output, failure)
        print(json.dumps({"status": "failed", "stage": stage,
            "failure_type": type(exc).__name__}), flush=True)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-receipt")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
