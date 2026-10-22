#!/usr/bin/env python3
"""Verify the three private generation-bias receipts and emit a safe summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
BRANCHES = ("baseline", "bbox_key_bias", "spatial_sham_key_bias")
RECEIPTS = [PRIVATE / f"minerva_visible_object_v2_generation_bias_row{i}_gpu_20261022_01.json"
            for i in range(3)]
CPU_RECEIPTS = [PRIVATE / f"minerva_visible_object_v2_fresh_row{i}_cpu_20261022_01.json"
                for i in range(3)]
PREFLIGHTS = [PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_01.json",
              PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json",
              PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json"]
COHORT = PRIVATE / "minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private(path: Path) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 50_000_000):
        raise ValueError("private_receipt_contract_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("private_receipt_changed")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("private_receipt_shape_invalid")
    return raw, value


def _verify_row(index: int, *, cohort_raw: bytes, cohort: dict,
                preflight_raw: bytes, preflight: dict,
                final_preflight: dict, cpu_raw: bytes, cpu: dict,
                gpu_raw: bytes, gpu: dict, source_hashes: dict) -> dict:
    from run_minerva_visible_object_generation_bias_v1 import classify_generation
    from run_minerva_generated_object_pixel_pilot import MODEL_REVISION
    selected = cohort["selection"][index]
    target = (selected.get("video_id"), selected.get("row_key_sha256"))
    row_records = [r for r in preflight.get("records", [])
                   if isinstance(r, dict)
                   and (r.get("video_id"), r.get("row_key_sha256")) == target]
    final_records = [r for r in final_preflight.get("records", [])
                     if isinstance(r, dict)
                     and (r.get("video_id"), r.get("row_key_sha256")) == target]
    if (preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or preflight.get("cohort_sha256") != _sha(cohort_raw)
            or preflight.get("manifest_sha256") != final_preflight.get("manifest_sha256")
            or any(preflight.get(key) != final_preflight.get(key) for key in (
                "preflight_source_sha256", "decoder_source_sha256",
                "generated_runner_source_sha256", "bbox_mapper_source_sha256",
                "qwen_video_processor", "rights_status"))
            or len(row_records) != 1 or len(final_records) != 1
            or row_records[0] != final_records[0]
            or row_records[0].get("status") != "pass"):
        raise ValueError("row_preflight_final_binding_invalid")
    dose = cpu.get("condition_key_counts")
    cpu_sources = cpu.get("source_sha256")
    if (not isinstance(cpu_sources, dict) or not cpu_sources
            or cpu.get("runner_sha256") != cpu_sources.get(
                "run_minerva_generated_object_pixel_pilot.py")):
        raise ValueError("cpu_source_bundle_invalid")
    for filename, digest in cpu_sources.items():
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not isinstance(digest, str) or len(digest) != 64
                or _sha((ROOT / "scripts" / filename).read_bytes()) != digest):
            raise ValueError("cpu_source_hash_mismatch")
    if (gpu.get("status") != "completed_minerva_visible_object_generation_bias_v1"
            or gpu.get("direct_faithfulness_validated") is not False
            or gpu.get("row_index") != index
            or gpu.get("cohort_sha256") != _sha(cohort_raw)
            or gpu.get("cohort_preflight_sha256") != _sha(preflight_raw)
            or gpu.get("cpu_receipt_sha256") != _sha(cpu_raw)
            or gpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or gpu.get("video_sha256") != cpu.get("video_sha256")
            or gpu.get("time_map_sha256") != cpu.get("time_map_sha256")
            or gpu.get("model_revision") != MODEL_REVISION
            or gpu.get("source_sha256") != source_hashes
            or gpu.get("branch_order") != list(BRANCHES)
            or gpu.get("frame_count") != 64
            or gpu.get("biased_layers_1_based") != list(range(5, 17))
            or gpu.get("failure_code") is not None
            or cpu.get("cohort_sha256") != _sha(cohort_raw)
            or cpu.get("cohort_cpu_preflight_sha256") != _sha(preflight_raw)
            or cpu.get("row_index") != index
            or cpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or cpu.get("annotation_source_sha256") != cohort.get("source_sha256")
            or gpu.get("base_pixel_tensor_sha256")
                != cpu.get("pixel_tensor_sha256", {}).get("base")
            or gpu.get("prompt_sha256") != cpu.get("rendered_prompt_sha256")
            or not isinstance(dose, dict)
            or dose.get("bbox") != dose.get("spatial_sham")
            or type(dose.get("bbox")) is not int or dose["bbox"] < 1
            or gpu.get("key_dose") != dose):
        raise ValueError("generation_receipt_binding_invalid")
    outcomes = gpu.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(BRANCHES):
        raise ValueError("generation_branch_set_invalid")
    branch_valid, text_hashes, audit_calls, audit_cells = {}, {}, {}, {}
    for name in BRANCHES:
        outcome = outcomes[name]
        if not isinstance(outcome, dict):
            raise ValueError("generation_branch_shape_invalid")
        text = outcome.get("generated_text")
        token_count = outcome.get("generated_token_count")
        text_hash = outcome.get("generated_text_sha256")
        token_hash = outcome.get("generated_token_ids_sha256")
        if (not isinstance(text, str) or not text.strip()
                or text_hash != _sha(text.encode())
                or type(token_count) is not int or not 1 <= token_count <= 64
                or not isinstance(token_hash, str) or len(token_hash) != 64):
            raise ValueError("generated_text_hash_or_length_invalid")
        # The runner's mechanical rule uses only nonempty token IDs and their
        # length. Actual token hashes remain bound in the private receipt.
        mechanical = classify_generation(text, [0] * token_count,
            cohort["selection"][index]["object_name"])
        if mechanical != outcome.get("mechanical_diagnostic"):
            raise ValueError("generation_mechanical_diagnostic_mismatch")
        audit = outcome.get("mask_audit")
        biased = name != "baseline"
        expected_calls = token_count * 12 if biased else 0
        expected_cells = expected_calls * (dose["bbox"] if name == "bbox_key_bias"
            else dose["spatial_sham"] if biased else 0)
        if (not isinstance(audit, dict)
                or audit.get("applied") is not biased
                or audit.get("all_selected_cells_match_bias") is not True
                or audit.get("hook_calls") != expected_calls
                or audit.get("verified_cells") != expected_cells
                or (biased and audit.get("calls_per_layer") != {
                    str(layer): token_count for layer in range(4, 16)})
                or (not biased and audit.get("calls_per_layer") not in (None, {}))
                or (biased and abs(float(audit.get("effective_mask_bias", 0))
                    - 0.69140625) > 1e-9)):
            raise ValueError("generation_hook_audit_invalid")
        branch_valid[name] = mechanical["status"] == "mechanically_valid"
        text_hashes[name] = text_hash
        audit_calls[name] = expected_calls
        audit_cells[name] = expected_cells
    return {"row_index": index, "receipt_sha256": _sha(gpu_raw),
        "cpu_receipt_sha256": _sha(cpu_raw),
        "cohort_preflight_sha256": _sha(preflight_raw),
        "key_dose_per_branch": dose["bbox"],
        "mechanically_valid": branch_valid,
        "text_changed_from_baseline": {
            "bbox_key_bias": text_hashes["bbox_key_bias"] != text_hashes["baseline"],
            "spatial_sham_key_bias":
                text_hashes["spatial_sham_key_bias"] != text_hashes["baseline"]},
        "bbox_text_differs_from_spatial_sham":
            text_hashes["bbox_key_bias"] != text_hashes["spatial_sham_key_bias"],
        "hook_calls": audit_calls, "verified_mask_cells": audit_cells,
        "peak_vram_bytes": gpu.get("peak_vram_bytes")}


def build_summary() -> dict:
    cohort_raw, cohort = _private(COHORT)
    final_preflight_raw, final_preflight = _private(PREFLIGHTS[1])
    if (cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"
            or cohort.get("requested_rows") != 3
            or not isinstance(cohort.get("selection"), list)
            or len(cohort["selection"]) != 3):
        raise ValueError("frozen_cohort_invalid")
    if (final_preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or final_preflight.get("cohort_sha256") != _sha(cohort_raw)
            or final_preflight.get("requested_rows") != 3
            or final_preflight.get("effective_rows") != 3
            or len(final_preflight.get("records", [])) != 3):
        raise ValueError("final_preflight_invalid")
    script_dir = ROOT / "scripts"
    source_names = {"runner": "run_minerva_visible_object_generation_bias_v1.py",
        "pilot": "run_minerva_generated_object_pixel_pilot.py",
        "generation_hooks": "qwen_visual_attention_boost.py",
        "generation_audit": "run_qta_intervened_interval_generation.py"}
    source_hashes = {key: _sha((script_dir / filename).read_bytes())
                     for key, filename in source_names.items()}
    rows = []
    for i in range(3):
        p_raw, preflight = _private(PREFLIGHTS[i])
        c_raw, cpu = _private(CPU_RECEIPTS[i])
        g_raw, gpu = _private(RECEIPTS[i])
        rows.append(_verify_row(i, cohort_raw=cohort_raw, cohort=cohort,
            preflight_raw=p_raw, preflight=preflight,
            final_preflight=final_preflight, cpu_raw=c_raw, cpu=cpu,
            gpu_raw=g_raw, gpu=gpu, source_hashes=source_hashes))
    return {"schema_version": 1,
        "status": "verified_three_video_exploratory_generation_comparison",
        "requested_videos": 3, "completed_videos": 3,
        "generation_calls_completed": 9,
        "cohort_sha256": _sha(cohort_raw),
        "final_preflight_sha256": _sha(final_preflight_raw),
        "source_sha256": {**source_hashes,
            "summarizer": _sha(Path(__file__).read_bytes())},
        "rows": rows,
        "mechanically_valid_counts": {name: sum(int(row["mechanically_valid"][name])
            for row in rows) for name in BRANCHES},
        "mechanical_bbox_only_validity_gain_vs_sham_rows": sum(
            int(row["mechanically_valid"]["bbox_key_bias"]
                and not row["mechanically_valid"]["spatial_sham_key_bias"])
            for row in rows),
        "nominal_bias_nats": math.log(2.0),
        "effective_bfloat16_mask_bias_nats": 0.69140625,
        "direct_faithfulness_measured": False,
        "rights_status": "unverified_local_only",
        "limitations": ["three already exposed exploratory videos; no independent confirmation",
            "mechanical text validity does not establish semantic video support",
            "bbox and sham key doses match within each video but vary across videos",
            "no human support review of these generated sentences"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if (output.parent.resolve() != (ROOT / "results").resolve()
            or output.suffix != ".json" or output.exists() or output.is_symlink()):
        parser.error("results_output_contract_invalid")
    try:
        result = build_summary()
        raw = (json.dumps(result, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False) + "\n").encode()
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "summary_failed")
    print(json.dumps({"status": result["status"],
        "completed_videos": result["completed_videos"],
        "mechanically_valid_counts": result["mechanically_valid_counts"],
        "mechanical_bbox_only_validity_gain_vs_sham_rows":
            result["mechanical_bbox_only_validity_gain_vs_sham_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
