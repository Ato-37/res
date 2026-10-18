#!/usr/bin/env python3
"""Local-only CPU/GPU runner for the exploratory v4 mask-specificity diagnostic.

The 147-row cohort and four annotation-union branches are re-run together.
Eligible rows additionally receive same-cardinality translated-mask branches.
This is a test-associated prediction-dependence diagnostic, not faithfulness.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

import run_nextgqa_validation_qta_annotation_proxy as kernel
import run_nextgqa_qta_v4_union_scorer as v4
from fixed_annotation_union_dependence import score_four_condition_logits
from fixed_prediction_selective_dependence import fixed_answer_foil_margin
from qta_v4_mask_specificity_utils import (
    FIXED_SEED, PROTOCOL_ID, choose_translation, eligible_translations,
    finite_cohort_summary, specificity_descriptives,
)
from run_nextgqa_qta_v4_union_geometry_census import (
    DATA_FILES as V4_DATA_FILES, PLAN_NAME as V4_PLAN_NAME,
    CENSUS_NAME as V4_CENSUS_NAME, SOURCE_FILES as V4_CENSUS_SOURCES,
    canonical_sha256, processor_geometry_provenance, validate_union_geometry,
)
from run_nextgqa_qta_v4_union_scorer import (
    PREFLIGHT_NAME as V4_PREFLIGHT_NAME, SMOKE_NAME as V4_SMOKE_NAME,
    load_v4_plan, selected_items, frozen_media_sha256,
)
from run_nextgqa_qta_v3_geometry_census import safe_error_code
from run_nextgqa_qta_exploratory_proxy_v2 import PRIVATE_REL, sha256_file

PROTOCOL = PROTOCOL_ID
SEED = FIXED_SEED
EXPECTED_N = 147
V4_SCORE_NAME = "nextgqa_tlvs_qta_v4_score_local_20261018_03_01.json"
V4_SCORE_SHA256 = "60f3b5a1847f042406ea727932d3960c8a64be11a2b0ecb8e81808f6289b41cc"
PREFLIGHT_NAME = "nextgqa_tlvs_qta_v4_mask_specificity_preflight_local_20261018_02.json"
SMOKE_NAME = "nextgqa_tlvs_qta_v4_mask_specificity_smoke_local_20261018_02.json"
OUTPUT_PREFIX = "nextgqa_tlvs_qta_v4_mask_specificity_score_local_20261018_"
V4_BRANCHES = ("baseline_full", "baseline_union_mask",
               "intervention_full", "intervention_union_mask")
PLACEBO_BRANCHES = ("baseline_placebo_mask", "intervention_placebo_mask")
ALL_BRANCHES = V4_BRANCHES + PLACEBO_BRANCHES
SOURCE_FILES = tuple(dict.fromkeys((*V4_CENSUS_SOURCES,
    "scripts/run_nextgqa_qta_v4_union_scorer.py",
    "scripts/run_nextgqa_qta_v4_mask_specificity.py",
    "scripts/qta_v4_mask_specificity_utils.py",
    "experiments/2026-10-18_qta_v4_mask_specificity_design.md")))


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(code)
    return float(value)


def validate_old_score_receipt(receipt: dict, *, plan: dict, census: dict,
                               private: Path) -> tuple[list[dict], str]:
    path = private / V4_SCORE_NAME
    actual_sha = sha256_file(path)
    if actual_sha != V4_SCORE_SHA256:
        raise ValueError("v4_score_receipt_hash_mismatch")
    if (receipt.get("status") != "v4_completed_test_associated_exploratory_finite_cohort"
            or receipt.get("requested_n") != EXPECTED_N
            or receipt.get("effective_n") != EXPECTED_N
            or receipt.get("requested_forwards") != 4 * EXPECTED_N
            or receipt.get("attempted_forwards") != 4 * EXPECTED_N
            or receipt.get("completed_forwards") != 4 * EXPECTED_N
            or receipt.get("plan_sha256") != sha256_file(private / V4_PLAN_NAME)
            or receipt.get("census_sha256") != sha256_file(private / V4_CENSUS_NAME)
            or receipt.get("preflight_sha256") != sha256_file(private / V4_PREFLIGHT_NAME)
            or receipt.get("gpu_smoke_sha256") != sha256_file(private / V4_SMOKE_NAME)
            or receipt.get("source_sha256") != plan["provenance"].get("source_sha256")
            or receipt.get("data_sha256") != plan["provenance"].get("data_sha256")
            or receipt.get("model_revision") != kernel.MODEL_REVISION
            or receipt.get("precision") != "4bit_bf16_compute_bf16_eager"
            or receipt.get("seed") != 20261018
            or not isinstance(receipt.get("records"), list)
            or len(receipt["records"]) != EXPECTED_N):
        raise ValueError("v4_score_receipt_contract_invalid")
    branch_counts = receipt.get("branch_counts")
    expected_branch_counts = {name: {"requested": EXPECTED_N, "attempted": EXPECTED_N,
                                     "completed": EXPECTED_N} for name in V4_BRANCHES}
    if branch_counts != expected_branch_counts:
        raise ValueError("v4_score_branch_counts_invalid")
    rows = receipt["records"]
    expected_rows = plan["evaluation"]
    seen = set()
    for old, planned in zip(rows, expected_rows, strict=True):
        identity = (planned.get("video_id"), planned.get("qid"))
        margins = old.get("margin_by_condition")
        if (identity in seen or old.get("video_id") != identity[0]
                or old.get("qid") != identity[1]
                or old.get("geometry_sha256") != planned.get("geometry_sha256")
                or old.get("row_binding_sha256") != planned.get("selected_row_binding_sha256")
                or type(old.get("answer_index")) is not int
                or type(old.get("foil_index")) is not int
                or not 0 <= old["answer_index"] < 5 or not 0 <= old["foil_index"] < 5
                or old["answer_index"] == old["foil_index"]
                or not isinstance(margins, dict) or set(margins) != set(V4_BRANCHES)):
            raise ValueError("v4_score_row_binding_invalid")
        seen.add(identity)
        margins = {name: _finite(margins[name], "v4_score_margin_nonfinite") for name in V4_BRANCHES}
        d_base = margins["baseline_full"] - margins["baseline_union_mask"]
        d_qta = margins["intervention_full"] - margins["intervention_union_mask"]
        delta = d_qta - d_base
        if not math.isclose(delta, _finite(old.get("delta_union_dependence"),
                                           "v4_score_delta_nonfinite"),
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v4_score_row_arithmetic_mismatch")
    if len(seen) != EXPECTED_N:
        raise ValueError("v4_score_unique_row_count_invalid")
    return rows, actual_sha


def prepare_mask_row(*, item: dict, frames: list, decode: dict, processor, torch_module,
                     old: dict) -> dict:
    batch, mapped, groups, qpos, token_ids, geometry_sha = kernel.prepare_geometry(
        processor, item, frames, decode, torch_module, "annotation_union_only")
    evidence = validate_union_geometry(mapped, groups)
    counts = tuple(len(group) for group in groups)
    if (geometry_sha != item["planned"]["geometry_sha256"]
            or len(groups) != item["planned"]["temporal_group_count"]
            or len(evidence) != item["planned"]["annotation_union_covered_group_count"]
            or mapped.get("evidence_group_indices") != evidence
            or mapped.get("annotation_covered_group_indices") != evidence
            or old.get("geometry_sha256") != geometry_sha
            or old.get("row_binding_sha256") != item["planned"]["selected_row_binding_sha256"]):
        raise ValueError("specificity_geometry_binding_mismatch")
    annotation_positions = tuple(sorted(pos for group_idx in evidence for pos in groups[group_idx]))
    if tuple(mapped["evidence_token_positions"]) != annotation_positions:
        raise ValueError("specificity_annotation_position_binding_mismatch")
    offsets = eligible_translations(evidence, counts)
    offset = choose_translation(evidence, counts, video_id=item["video_id"], qid=item["qid"])
    if (offset is not None and offset not in offsets):
        raise ValueError("specificity_offset_not_eligible")
    translated_groups = (tuple(idx + offset for idx in evidence) if offset is not None else ())
    placebo_positions = tuple(sorted(pos for idx in translated_groups for pos in groups[idx]))
    if (offset is None and placebo_positions
            or offset is not None and (len(placebo_positions) != len(annotation_positions)
                or len(set(placebo_positions)) != len(placebo_positions)
                or set(placebo_positions) == set(annotation_positions))):
        raise ValueError("specificity_placebo_mask_contract_invalid")
    row = {
        "video_id": item["video_id"], "qid": item["qid"],
        "geometry_sha256": geometry_sha,
        "row_binding_sha256": item["planned"]["selected_row_binding_sha256"],
        "group_count": len(groups), "annotation_group_indices": evidence,
        "group_token_counts": counts, "annotation_position_count": len(annotation_positions),
        "annotation_position_sha256": hashlib.sha256(
            json.dumps(annotation_positions, separators=(",", ":")).encode()).hexdigest(),
        "eligible_offsets": offsets, "chosen_offset": offset,
        "placebo_group_indices": translated_groups,
        "placebo_position_count": len(placebo_positions),
        "placebo_position_sha256": hashlib.sha256(
            json.dumps(placebo_positions, separators=(",", ":")).encode()).hexdigest(),
        "eligible": offset is not None,
        "batch": batch, "groups": groups, "qpos": qpos, "token_ids": token_ids,
        "annotation_positions": annotation_positions,
        "placebo_positions": placebo_positions,
    }
    return row


def build_preflight_header(*, processor_manifest: dict, repo: Path,
                           source_hashes: dict, data_hashes: dict,
                           v4_score_sha256: str) -> dict:
    private = repo / PRIVATE_REL
    return {
        "schema_version": 1,
        "status": "qta_v4_mask_specificity_geometry_preflight_complete_non_authorizing",
        "protocol": PROTOCOL, "seed": SEED,
        "scope": "same-run annotation-union versus geometry-matched translated-mask prediction dependence; exploratory, not faithfulness",
        "requested_rows": EXPECTED_N,
        "selection": "all nonzero nonwrapping translations preserving group count, run shape, and exact processor visual-token position count; compact-JSON UTF-8 SHA-256 minimum",
        "source_sha256": source_hashes, "data_sha256": data_hashes,
        "v4_plan_sha256": sha256_file(private / V4_PLAN_NAME),
        "v4_census_sha256": sha256_file(private / V4_CENSUS_NAME),
        "v4_geometry_preflight_sha256": sha256_file(private / V4_PREFLIGHT_NAME),
        "v4_gpu_smoke_sha256": sha256_file(private / V4_SMOKE_NAME),
        "v4_score_sha256": v4_score_sha256,
        "processor_manifest": processor_manifest,
        "library_versions": {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "av", "qwen-vl-utils")},
    }


def geometry_preflight(*, items: list[dict], old_rows: list[dict], processor,
                       processor_manifest: dict, repo: Path, source_hashes: dict,
                       data_hashes: dict, v4_score_sha256: str) -> dict:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    rows = []
    preflight_started = time.monotonic()
    print(json.dumps({"stage": "specificity_geometry_preflight_started",
        "requested_rows": EXPECTED_N}, sort_keys=True), flush=True)
    for row_number, (item, old) in enumerate(zip(items, old_rows, strict=True), start=1):
        frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
            uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        mapped = prepare_mask_row(item=item, frames=frames, decode=decode, processor=processor,
                                  torch_module=torch, old=old)
        rows.append({key: value for key, value in mapped.items()
                     if key not in {"batch", "groups", "qpos", "token_ids",
                                    "annotation_positions", "placebo_positions"}})
        del mapped, frames, decode
        gc.collect()
        if row_number % 10 == 0 or row_number == EXPECTED_N:
            print(json.dumps({"stage": "specificity_geometry_preflight_progress",
                "completed_rows": row_number, "requested_rows": EXPECTED_N,
                "eligible_rows_so_far": sum(row["eligible"] for row in rows),
                "elapsed_seconds": round(time.monotonic() - preflight_started, 1)}, sort_keys=True),
                flush=True)
    if len(rows) != EXPECTED_N or len({(r["video_id"], r["qid"]) for r in rows}) != EXPECTED_N:
        raise ValueError("specificity_preflight_denominator_invalid")
    eligible = sum(row["eligible"] for row in rows)
    if ({name: sha256_file(repo / name) for name in SOURCE_FILES} != source_hashes
            or {name: sha256_file(repo / name) for name in V4_DATA_FILES} != data_hashes):
        raise ValueError("specificity_source_or_data_changed_during_preflight")
    return {**build_preflight_header(processor_manifest=processor_manifest, repo=repo,
                                     source_hashes=source_hashes, data_hashes=data_hashes,
                                     v4_score_sha256=v4_score_sha256),
        "completed_rows": len(rows),
        "eligible_specificity_rows": eligible,
        "structurally_unavailable_rows": EXPECTED_N - eligible,
        "run_allowed": eligible > 0,
        "rows": rows,
        "limitations": ["known test-associated cohort with unknown prior-exposure history",
            "translated masks preserve geometry/cardinality, not visual content or all distribution shift",
            "benchmark annotations are not human-reviewed claims", "rights remain unverified; local-only"],
    }


def validate_preflight(saved: dict, expected_header: dict) -> None:
    compare = expected_header
    if any(saved.get(key) != value for key, value in compare.items()):
        raise ValueError("specificity_preflight_header_mismatch")
    rows = saved.get("rows")
    if (not isinstance(rows, list) or len(rows) != EXPECTED_N
            or saved.get("completed_rows") != EXPECTED_N
            or any(not isinstance(row, dict) for row in rows)
            or len({(row.get("video_id"), row.get("qid")) for row in rows}) != EXPECTED_N
            or type(saved.get("eligible_specificity_rows")) is not int
            or saved.get("eligible_specificity_rows") != sum(row.get("eligible") is True for row in rows)
            or saved.get("structurally_unavailable_rows") != EXPECTED_N - saved.get("eligible_specificity_rows")
            or saved.get("run_allowed") is not (saved.get("eligible_specificity_rows", 0) > 0)):
        raise ValueError("specificity_preflight_rows_mismatch")
    required_hashes = ("geometry_sha256", "row_binding_sha256",
                       "annotation_position_sha256", "placebo_position_sha256")
    seen = set()
    for row in rows:
        identity = (row.get("video_id"), row.get("qid"))
        if (any(not isinstance(value, str) or not value for value in identity)
                or identity in seen
                or any(not isinstance(row.get(key), str) or len(row[key]) != 64
                       or any(c not in "0123456789abcdef" for c in row[key])
                       for key in required_hashes)):
            raise ValueError("specificity_preflight_row_binding_invalid")
        seen.add(identity)
        groups, counts, offsets = (row.get("annotation_group_indices"),
            row.get("group_token_counts"), row.get("eligible_offsets"))
        if (type(row.get("eligible")) is not bool or type(row.get("group_count")) is not int
                or type(row.get("annotation_position_count")) is not int
                or type(row.get("placebo_position_count")) is not int
                or not isinstance(groups, list) or not isinstance(counts, list)
                or not isinstance(offsets, list) or not groups
                or groups != sorted(set(groups))
                or any(type(x) is not int for x in groups + counts + offsets)
                or len(counts) != row["group_count"]
                or any(x <= 0 for x in counts) or 0 in offsets
                or offsets != sorted(set(offsets))):
            raise ValueError("specificity_preflight_row_geometry_invalid")
        expected_offsets = eligible_translations(groups, counts)
        chosen = row.get("chosen_offset")
        translated = [] if chosen is None else [x + chosen for x in groups]
        if (tuple(offsets) != expected_offsets
                or row["eligible"] is not bool(offsets)
                or (chosen is None) != (not offsets)
                or (chosen is not None and (type(chosen) is not int or chosen not in offsets))
                or row.get("placebo_group_indices") != translated
                or row["annotation_position_count"] != sum(counts[x] for x in groups)
                or row["placebo_position_count"] != (0 if chosen is None else
                    sum(counts[x] for x in translated))):
            raise ValueError("specificity_preflight_row_translation_invalid")


def validate_preflight_bindings(rows: list[dict], items: list[dict], old_rows: list[dict]) -> None:
    if len(rows) != EXPECTED_N or len(items) != EXPECTED_N or len(old_rows) != EXPECTED_N:
        raise ValueError("specificity_preflight_binding_denominator_invalid")
    seen_bindings = set()
    for preflight_row, item, old in zip(rows, items, old_rows, strict=True):
        binding = item["planned"].get("selected_row_binding_sha256")
        if (not isinstance(binding, str) or binding in seen_bindings
                or preflight_row.get("video_id") != item.get("video_id")
                or preflight_row.get("qid") != item.get("qid")
                or preflight_row.get("row_binding_sha256") != binding
                or preflight_row.get("row_binding_sha256") != old.get("row_binding_sha256")
                or old.get("video_id") != item.get("video_id")
                or old.get("qid") != item.get("qid")):
            raise ValueError("specificity_preflight_order_or_binding_mismatch")
        seen_bindings.add(binding)


def validate_smoke_receipt(smoke: dict, *, preflight_sha256: str, old_sha: str,
                           source_hashes: dict, data_hashes: dict,
                           eligible_row: dict, old_row: dict) -> None:
    records = smoke.get("records")
    counts = {name: {"requested": 1, "attempted": 1, "completed": 1}
              for name in ALL_BRANCHES}
    actual_counts = smoke.get("branch_counts")
    exact_counts = (isinstance(actual_counts, dict) and set(actual_counts) == set(ALL_BRANCHES)
        and all(isinstance(actual_counts[name], dict)
            and set(actual_counts[name]) == {"requested", "attempted", "completed"}
            and all(type(actual_counts[name][key]) is int
                    for key in ("requested", "attempted", "completed"))
            for name in ALL_BRANCHES))
    margin_error = smoke.get("baseline_full_margin_abs_error")
    if (smoke.get("status") != "specificity_six_forward_smoke_complete"
            or smoke.get("preflight_sha256") != preflight_sha256
            or smoke.get("v4_score_sha256") != old_sha
            or smoke.get("source_sha256") != source_hashes
            or smoke.get("data_sha256") != data_hashes
            or smoke.get("row_binding_sha256") != eligible_row.get("row_binding_sha256")
            or smoke.get("requested_forwards") != 6
            or smoke.get("attempted_forwards") != 6
            or smoke.get("completed_forwards") != 6
            or not exact_counts or actual_counts != counts
            or not isinstance(records, list) or len(records) != 1
            or records[0].get("row_binding_sha256") != eligible_row.get("row_binding_sha256")
            or type(records[0].get("answer_index")) is not int
            or records[0].get("answer_index") != old_row.get("answer_index")
            or type(records[0].get("foil_index")) is not int
            or records[0].get("foil_index") != old_row.get("foil_index")
            or type(smoke.get("baseline_answer_index")) is not int
            or smoke.get("baseline_answer_index") != old_row.get("answer_index")
            or type(smoke.get("baseline_foil_index")) is not int
            or smoke.get("baseline_foil_index") != old_row.get("foil_index")
            or not isinstance(margin_error, (int, float)) or isinstance(margin_error, bool)
            or not math.isfinite(margin_error) or margin_error > 1e-5):
        raise ValueError("specificity_smoke_receipt_invalid")
    if (not 0 <= records[0]["answer_index"] < 5 or not 0 <= records[0]["foil_index"] < 5
            or records[0]["answer_index"] == records[0]["foil_index"]):
        raise ValueError("specificity_smoke_prediction_index_invalid")


def initial_branch_counts(mode: str, eligible_n: int | None = None) -> dict:
    if mode not in {"smoke", "full"}:
        return {name: {"requested": 0, "attempted": 0, "completed": 0}
                for name in ALL_BRANCHES}
    base_n = 1 if mode == "smoke" else EXPECTED_N
    placebo_n = 1 if mode == "smoke" else (eligible_n or 0)
    return {name: {"requested": placebo_n if name in PLACEBO_BRANCHES else base_n,
                   "attempted": 0, "completed": 0} for name in ALL_BRANCHES}


def validate_runtime_preflight_row(prepared: dict, expected: dict) -> None:
    sequence_fields = {"annotation_group_indices", "group_token_counts", "eligible_offsets",
                       "placebo_group_indices"}
    for key in ("group_count", "annotation_group_indices", "group_token_counts",
                "annotation_position_count", "annotation_position_sha256", "eligible_offsets",
                "chosen_offset", "placebo_group_indices", "placebo_position_count",
                "placebo_position_sha256", "eligible"):
        actual = prepared.get(key)
        wanted = expected.get(key)
        if key in sequence_fields and actual is not None:
            actual = list(actual)
        if actual != wanted:
            raise ValueError("specificity_runtime_preflight_mismatch")


def _choice_logps(output: dict, token_ids: tuple[int, ...], torch_module) -> list[float]:
    logits = output.get("logits") if isinstance(output, dict) else None
    if (not isinstance(token_ids, (list, tuple)) or len(token_ids) != 5
            or any(type(token) is not int or token < 0 for token in token_ids)
            or len(set(token_ids)) != 5):
        raise ValueError("specificity_answer_token_ids_invalid")
    if (not torch_module.is_tensor(logits) or logits.ndim != 3
            or tuple(logits.shape[:2]) != (1, 1) or not torch_module.isfinite(logits).all()):
        raise ValueError("specificity_logits_contract_invalid")
    values = torch_module.log_softmax(logits[0, -1].float(), dim=-1)
    if any(token >= values.shape[-1] for token in token_ids):
        raise ValueError("specificity_answer_token_out_of_range")
    return [float(value) for value in values[list(token_ids)].detach().cpu().tolist()]


def score_row(*, model, processor, item: dict, expected: dict, old: dict, counters: dict) -> dict:
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    from qwen_question_temporal_attention_selector import select_top_temporal_groups
    frames, decode = decode_sampled_video(item["video_path"], num_frames=kernel.FRAME_COUNT,
        uniform_tolerance_seconds=UNIFORM_TOLERANCE)
    prepared = prepare_mask_row(item=item, frames=frames, decode=decode,
        processor=processor, torch_module=torch, old=old)
    if (expected.get("video_id") != item.get("video_id")
            or expected.get("qid") != item.get("qid")
            or expected.get("row_binding_sha256")
               != item["planned"].get("selected_row_binding_sha256")):
        raise ValueError("specificity_runtime_identity_binding_mismatch")
    validate_runtime_preflight_row(prepared, expected)
    batch, groups, qpos, token_ids = (prepared["batch"], prepared["groups"],
        prepared["qpos"], prepared["token_ids"])
    outputs = {}

    def forward(name: str, **kwargs):
        counters[name]["attempted"] += 1
        output = kernel._forward_score(model, batch, token_ids, torch, **kwargs)
        counters[name]["completed"] += 1
        torch.cuda.empty_cache()
        return output

    capture = {}
    outputs["baseline_full"] = forward("baseline_full", question_positions=qpos,
        groups=groups, capture=capture)
    expected_layers = tuple(index - 1 for index in kernel.LAYERS_1_BASED)
    if tuple(sorted(capture)) != expected_layers:
        raise ValueError("specificity_baseline_capture_incomplete")
    layer_values = [capture[layer] for layer in sorted(capture)]
    group_scores = [sum(layer[g] for layer in layer_values) / len(layer_values)
                    for g in range(len(groups))]
    qta_groups = select_top_temporal_groups(group_scores,
        top_k=max(1, math.ceil(len(groups) * 0.10)))
    bias_positions = tuple(sorted(pos for idx in qta_groups for pos in groups[idx]))
    baseline_values = _choice_logps(outputs["baseline_full"], token_ids, torch)
    from fixed_prediction_selective_dependence import freeze_baseline_answer_and_foil
    answer, foil = freeze_baseline_answer_and_foil(baseline_values)
    if (answer != old.get("answer_index") or foil != old.get("foil_index")
            or not math.isclose(fixed_answer_foil_margin(baseline_values,
                    answer_index=answer, foil_index=foil),
                float(old["margin_by_condition"]["baseline_full"]),
                rel_tol=0.0, abs_tol=1e-5)):
        raise ValueError("specificity_frozen_answer_foil_reproduction_failed")
    outputs["baseline_union_mask"] = forward("baseline_union_mask",
        ablation_positions=prepared["annotation_positions"])
    outputs["intervention_full"] = forward("intervention_full", bias_positions=bias_positions)
    outputs["intervention_union_mask"] = forward("intervention_union_mask",
        bias_positions=bias_positions, ablation_positions=prepared["annotation_positions"])
    if expected["eligible"]:
        outputs["baseline_placebo_mask"] = forward("baseline_placebo_mask",
            ablation_positions=prepared["placebo_positions"])
        outputs["intervention_placebo_mask"] = forward("intervention_placebo_mask",
            bias_positions=bias_positions, ablation_positions=prepared["placebo_positions"])
    else:
        outputs["baseline_placebo_mask"] = outputs["intervention_placebo_mask"] = None
    four = {key: outputs[key] for key in V4_BRANCHES}
    lengths = {key: int(batch["input_ids"].shape[1]) for key in V4_BRANCHES}
    scored = score_four_condition_logits(condition_outputs=four,
        answer_token_ids=token_ids, input_lengths=lengths, torch_module=torch)
    contrast = scored["contrast"]
    if (contrast["answer_index"] != answer or contrast["foil_index"] != foil
            or not math.isfinite(contrast["delta_union_dependence"])):
        raise ValueError("specificity_four_branch_arithmetic_invalid")
    row = {
        "video_id": item["video_id"], "qid": item["qid"],
        "geometry_sha256": prepared["geometry_sha256"],
        "row_binding_sha256": prepared["row_binding_sha256"],
        "answer_index": answer, "foil_index": foil,
        "historical_answer_foil_match": True,
        "historical_baseline_full_margin_abs_error": abs(
            float(contrast["margin_by_condition"]["baseline_full"])
            - float(old["margin_by_condition"]["baseline_full"])),
        "qta_selected_group_count": len(qta_groups),
        "annotation_group_count": len(prepared["annotation_group_indices"]),
        "group_count": prepared["group_count"],
        "placebo_offset": prepared["chosen_offset"],
        "specificity_eligible": prepared["eligible"],
        "margin_by_condition": contrast["margin_by_condition"],
        "D_annotation_union": float(contrast["delta_union_dependence"]),
    }
    if prepared["eligible"]:
        base_placebo_values = _choice_logps(outputs["baseline_placebo_mask"], token_ids, torch)
        qta_placebo_values = _choice_logps(outputs["intervention_placebo_mask"], token_ids, torch)
        base_placebo_margin = fixed_answer_foil_margin(base_placebo_values,
            answer_index=answer, foil_index=foil)
        qta_placebo_margin = fixed_answer_foil_margin(qta_placebo_values,
            answer_index=answer, foil_index=foil)
        margin = contrast["margin_by_condition"]
        d_base_placebo = margin["baseline_full"] - base_placebo_margin
        d_qta_placebo = margin["intervention_full"] - qta_placebo_margin
        d_placebo = d_qta_placebo - d_base_placebo
        specificity = float(contrast["delta_union_dependence"]) - d_placebo
        if not all(math.isfinite(x) for x in (d_base_placebo, d_qta_placebo, d_placebo, specificity)):
            raise ValueError("specificity_placebo_arithmetic_nonfinite")
        row.update({"baseline_placebo_margin": base_placebo_margin,
            "intervention_placebo_margin": qta_placebo_margin,
            "D_baseline_placebo": d_base_placebo,
            "D_intervention_placebo": d_qta_placebo,
            "D_placebo_translation": d_placebo,
            "specificity_difference": specificity})
    del outputs, four, batch, frames, decode, prepared
    gc.collect()
    return row


def build_summaries(records: list[dict]) -> dict:
    if len(records) != EXPECTED_N or any(not isinstance(row, dict) for row in records):
        raise ValueError("specificity_summary_denominator_invalid")
    ann = [row["D_annotation_union"] for row in records]
    eligible = [row for row in records if row["specificity_eligible"] is True]
    if not eligible:
        raise ValueError("specificity_no_eligible_rows")
    return {
        "all_147_D_annotation_union": finite_cohort_summary(ann),
        "eligible_D_annotation_union": finite_cohort_summary(
            [row["D_annotation_union"] for row in eligible]),
        "eligible_D_placebo_translation": finite_cohort_summary(
            [row["D_placebo_translation"] for row in eligible]),
        "eligible_specificity_difference": finite_cohort_summary(
            [row["specificity_difference"] for row in eligible]),
        "inferential_status": "finite cohort descriptive only; no p-value or population interval",
    }


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    if ((args.mode == "preflight" and args.output_name != PREFLIGHT_NAME)
            or (args.mode == "smoke" and args.output_name != SMOKE_NAME)
            or (args.mode == "full" and not args.output_name.startswith(OUTPUT_PREFIX))):
        parser.error("output_name_does_not_match_mode_contract")
    repo = Path(__file__).resolve().parents[1]
    private = repo / PRIVATE_REL
    output_path = private / args.output_name
    if os.path.lexists(output_path):
        parser.error("output_collision")
    started = time.monotonic()
    counters = initial_branch_counts(args.mode)
    stage = "load_frozen_plan"
    failure_stage = None
    old_rows = []
    records = []
    binding = None
    try:
        base, gate, plan, census = load_v4_plan(repo)
        items = selected_items(repo, base, plan)
        old_receipt = v4.private_json(private / V4_SCORE_NAME, private)
        old_rows, old_sha = validate_old_score_receipt(old_receipt, plan=plan,
            census=census, private=private)
        source_hashes = {name: sha256_file(repo / name) for name in SOURCE_FILES}
        v4_source_hashes = {name: sha256_file(repo / name) for name in v4.SOURCE_FILES}
        data_hashes = {name: sha256_file(repo / name) for name in V4_DATA_FILES}
        if (v4_source_hashes != plan["provenance"]["source_sha256"]
                or data_hashes != plan["provenance"]["data_sha256"]
                or old_receipt.get("source_sha256") != v4_source_hashes
                or old_receipt.get("data_sha256") != data_hashes):
            raise ValueError("specificity_v4_source_or_data_binding_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / kernel.MODEL_REVISION)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        processor_manifest = processor_geometry_provenance(processor, model_path)
        if processor_manifest != plan["provenance"].get("processor_manifest"):
            raise ValueError("specificity_processor_manifest_changed")
        if args.mode == "preflight":
            stage = "outcome_blind_specificity_geometry_preflight"
            payload = geometry_preflight(items=items, old_rows=old_rows,
                processor=processor, processor_manifest=processor_manifest, repo=repo,
                source_hashes=source_hashes, data_hashes=data_hashes,
                v4_score_sha256=old_sha)
            kernel.write_private_output(repo, args.output_name, payload)
            print(json.dumps({"status": payload["status"],
                "requested_rows": EXPECTED_N, "completed_rows": payload["completed_rows"],
                "eligible_specificity_rows": payload["eligible_specificity_rows"],
                "structurally_unavailable_rows": payload["structurally_unavailable_rows"],
                "run_allowed": payload["run_allowed"], "privacy": "private receipt; no row content"}, sort_keys=True))
            return 0
        saved_preflight = v4.private_json(private / PREFLIGHT_NAME, private)
        # Bind protocol/data/code, then revalidate each media row immediately before scoring.
        stage = "validate_saved_specificity_preflight"
        expected_header = build_preflight_header(processor_manifest=processor_manifest,
            repo=repo, source_hashes=source_hashes, data_hashes=data_hashes,
            v4_score_sha256=old_sha)
        validate_preflight(saved_preflight, expected_header)
        validate_preflight_bindings(saved_preflight["rows"], items, old_rows)
        eligible_rows = [row for row in saved_preflight["rows"] if row["eligible"]]
        if not eligible_rows:
            raise ValueError("specificity_no_eligible_rows")
        counters = initial_branch_counts(args.mode,
            1 if args.mode == "smoke" else saved_preflight["eligible_specificity_rows"])
        item_by_binding = {row["planned"]["selected_row_binding_sha256"]: row for row in items}
        old_by_binding = {row["row_binding_sha256"]: row for row in old_rows}
        if args.mode == "smoke":
            expected_rows = [eligible_rows[0]]
        else:
            if not args.output_name[len(OUTPUT_PREFIX):]:
                raise ValueError("specificity_full_output_suffix_required")
            saved_smoke = v4.private_json(private / SMOKE_NAME, private)
            validate_smoke_receipt(saved_smoke,
                preflight_sha256=sha256_file(private / PREFLIGHT_NAME), old_sha=old_sha,
                source_hashes=source_hashes, data_hashes=data_hashes,
                eligible_row=eligible_rows[0],
                old_row=old_by_binding[eligible_rows[0]["row_binding_sha256"]])
            expected_rows = saved_preflight["rows"]
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_required_no_cpu_fallback")
        free, total = torch.cuda.mem_get_info()
        if free < 11 * 1024**3:
            raise ValueError("insufficient_free_vram_before_model_load")
        checkpoint_audit = kernel.audit_pinned_checkpoint(repo, model_path)
        if checkpoint_audit != old_receipt.get("checkpoint_byte_audit"):
            raise ValueError("specificity_checkpoint_audit_changed")
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(kernel.LAYERS_1_BASED)):
            raise ValueError("specificity_model_processor_contract_invalid")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        if args.mode == "full":
            eligible_n = saved_preflight["eligible_specificity_rows"]
            if counters != initial_branch_counts("full", eligible_n):
                raise ValueError("specificity_requested_counter_drift")
        effective_rows = expected_rows if args.mode == "smoke" else saved_preflight["rows"]
        if args.mode == "full" and sum(row["eligible"] for row in effective_rows) == 0:
            raise ValueError("specificity_no_eligible_rows")
        stage = "six_condition_specificity_scoring"
        for row in effective_rows:
            binding = row["row_binding_sha256"]
            if args.mode == "full" and not row["eligible"]:
                item = item_by_binding[binding]
                old = old_by_binding[binding]
                # Ineligible rows retain the four annotation conditions and 147-row denominator.
                record = score_row(model=model, processor=processor, item=item, expected=row,
                    old=old, counters=counters)
                records.append(record)
                continue
            item = item_by_binding[binding]
            old = old_by_binding[binding]
            record = score_row(model=model, processor=processor, item=item, expected=row,
                old=old, counters=counters)
            records.append(record)
        torch.cuda.synchronize()
        del model; gc.collect(); torch.cuda.empty_cache()
        if args.mode == "smoke":
            record = records[0]
            payload = {"schema_version": 1,
                "status": "specificity_six_forward_smoke_complete",
                "protocol": PROTOCOL, "v4_score_sha256": old_sha,
                "preflight_sha256": sha256_file(private / PREFLIGHT_NAME),
                "source_sha256": source_hashes, "data_sha256": data_hashes,
                "row_binding_sha256": record["row_binding_sha256"],
                "baseline_answer_index": record["answer_index"],
                "baseline_foil_index": record["foil_index"],
                "baseline_full_margin_abs_error": record["historical_baseline_full_margin_abs_error"],
                "requested_forwards": 6, "attempted_forwards": sum(x["attempted"] for x in counters.values()),
                "completed_forwards": sum(x["completed"] for x in counters.values()),
                "branch_counts": counters,
                "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                "rights_status": "unverified_local_only", "faithfulness_result": False,
                "records": records, "elapsed_seconds": time.monotonic() - started}
            if payload["completed_forwards"] != 6:
                raise ValueError("specificity_smoke_forward_count_invalid")
            if any(counters[name] != {"requested": 1, "attempted": 1, "completed": 1}
                   for name in ALL_BRANCHES):
                raise ValueError("specificity_smoke_branch_count_invalid")
        else:
            if len(records) != EXPECTED_N or any(
                    counters[name]["attempted"] != counters[name]["requested"]
                    or counters[name]["completed"] != counters[name]["requested"]
                    for name in ALL_BRANCHES):
                raise ValueError("specificity_full_denominator_or_branch_count_invalid")
            eligible_records = [row for row in records if row["specificity_eligible"]]
            payload = {"schema_version": 1,
                "status": "qta_v4_mask_specificity_exploratory_complete",
                "protocol": PROTOCOL, "scope": "test-associated annotation-mask specificity diagnostic; not faithfulness or confirmatory",
                "rights_status": "unverified_local_only", "faithfulness_result": False,
                "requested_rows": EXPECTED_N, "effective_rows": len(records),
                "eligible_specificity_rows": len(eligible_records),
                "structurally_unavailable_rows": EXPECTED_N - len(eligible_records),
                "requested_forwards": sum(x["requested"] for x in counters.values()),
                "attempted_forwards": sum(x["attempted"] for x in counters.values()),
                "completed_forwards": sum(x["completed"] for x in counters.values()),
                "branch_counts": counters, "records": records,
                "summary": build_summaries(records),
                "inferential_status": "finite cohort descriptive only; no p-value or population interval",
                "provenance": {"v4_score_sha256": old_sha,
                    "v4_plan_sha256": sha256_file(private / V4_PLAN_NAME),
                    "v4_census_sha256": sha256_file(private / V4_CENSUS_NAME),
                    "specificity_preflight_sha256": sha256_file(private / PREFLIGHT_NAME),
                    "specificity_smoke_sha256": sha256_file(private / SMOKE_NAME),
                    "source_sha256": source_hashes, "data_sha256": data_hashes,
                    "processor_manifest": processor_manifest,
                    "checkpoint_byte_audit": checkpoint_audit},
                "model_revision": kernel.MODEL_REVISION,
                "precision": "4bit_bf16_compute_bf16_eager", "seed": SEED,
                "hardware": {"device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                    "free_vram_before_load_gib": free / 1024**3,
                    "total_vram_gib": total / 1024**3,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
                "elapsed_seconds": time.monotonic() - started,
                "limitations": ["post-result exploratory comparison, not confirmatory",
                    "translated mask matches group/token counts and run shape but not content or all OOD effects",
                    "test-associated cohort, benchmark intervals unreviewed, prior exposure unknown",
                    "rights unverified; local-only protected outputs"]}
        kernel.write_private_output(repo, args.output_name, payload)
        print(json.dumps({"status": payload["status"],
            "requested_rows": payload.get("requested_rows", 1),
            "effective_rows": payload.get("effective_rows", 1),
            "eligible_specificity_rows": payload.get("eligible_specificity_rows", 1),
            "requested_forwards": payload["requested_forwards"],
            "attempted_forwards": payload["attempted_forwards"],
            "completed_forwards": payload["completed_forwards"],
            "privacy": "private receipt; no row content"}, sort_keys=True))
        return 0
    except Exception as exc:
        code = safe_error_code(exc)
        failure_stage = stage
        failure = {"schema_version": 1,
            "status": "failed_incomplete_no_cohort_estimate",
            "protocol": PROTOCOL, "failure_stage": failure_stage,
            "failure_code": code, "requested_rows": EXPECTED_N,
            "completed_rows": len(records), "failing_row_binding_sha256": binding,
            "branch_counts": counters,
            "row_failures": ([{"row_binding_sha256": binding, "failure_code": code}]
                if binding is not None else []),
            "requested_forwards": sum(x["requested"] for x in counters.values()),
            "attempted_forwards": sum(x["attempted"] for x in counters.values()),
            "completed_forwards": sum(x["completed"] for x in counters.values()),
            "eligible_specificity_rows": None,
            "inferential_status": "none; incomplete denominator",
            "faithfulness_result": False,
            "private_output_name": args.output_name}
        if not os.path.lexists(output_path):
            try:
                kernel.write_private_output(repo, args.output_name, failure)
            except Exception:
                pass
        parser.error(f"run_failed:{failure_stage}:{code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
