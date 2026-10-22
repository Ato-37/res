#!/usr/bin/env python3
"""Local-only exploratory three-branch generation on the frozen Minerva v2 rows.

For each row, generate from the identical unfilled 64-frame input under no
attention bias, bbox-key +ln(2), and equal-dose spatial-sham-key +ln(2).
This measures changes in generated text, not semantic support or faithfulness.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import run_minerva_generated_object_pixel_pilot as pilot

BRANCHES = ("baseline", "bbox_key_bias", "spatial_sham_key_bias")
STATUS = "completed_minerva_visible_object_generation_bias_v1"
SEED = 20261022
MAX_NEW_TOKENS = 64


def classify_generation(text: str, token_ids: list[int], object_name: str) -> dict:
    """Frozen mechanical diagnostics; semantic support needs independent review."""
    result = {"object_name_present": False, "short_single_line": False,
              "status": "invalid", "reason": None}
    if not isinstance(text, str) or not isinstance(token_ids, list):
        result["reason"] = "generated_target_empty"
        return result
    alias = object_name.split()[-1]
    result["object_name_present"] = bool(re.search(
        rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE))
    try:
        pilot._generated_target(text, token_ids, object_name)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    result.update(short_single_line=True, status="mechanically_valid")
    return result


def _generate_one(model, processor, batch, *, torch, positions,
                  layer_indices: tuple[int, ...]) -> dict:
    from qwen_visual_attention_boost import (
        install_generation_video_key_bias_hooks, remove_hooks)
    from run_qta_intervened_interval_generation import (
        _observe_generation_shapes, validate_dynamic_shape_trace,
        validate_generation_mask_audit)

    prompt_length = int(batch["input_ids"].shape[1])
    prompt_ids = batch["input_ids"][0].detach().cpu().tolist()
    observations, observer_handles = _observe_generation_shapes(model, layer_indices)
    mask_observations, treatment_handles = [], []
    try:
        if positions is not None:
            treatment_handles = install_generation_video_key_bias_hooks(
                model=model, layer_indices=layer_indices,
                selected_positions=positions, first_query_position=prompt_length - 1,
                logit_bias=pilot.LOGIT_BIAS_NATS, torch_module=torch,
                observations=mask_observations)
        moved = {key: value.to("cuda") if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise RuntimeError("generation_free_vram_guard_failed")
        with torch.inference_mode():
            generated = model.generate(**moved, do_sample=False, num_beams=1,
                use_cache=True, cache_implementation="dynamic",
                max_new_tokens=MAX_NEW_TOKENS, return_dict_in_generate=True,
                output_logits=False)
        sequence = generated.sequences[0].detach().cpu().tolist()
        if sequence[:prompt_length] != prompt_ids or len(sequence) <= prompt_length:
            raise ValueError("generation_prefix_contract_failed")
        target_ids = sequence[prompt_length:]
        text = processor.tokenizer.decode(target_ids,
            skip_special_tokens=True).strip()
        trace = validate_dynamic_shape_trace(observations, layer_indices,
                                              prompt_length)
        audit = validate_generation_mask_audit(mask_observations, trace,
            layer_indices, expected_applied=positions is not None)
        return {"generated_text": text,
            "generated_text_sha256": pilot._sha(text.encode("utf-8")),
            "generated_token_ids_sha256": pilot._sha(json.dumps(
                target_ids, separators=(",", ":")).encode("ascii")),
            "generated_token_count": len(target_ids),
            "shape_trace": trace, "mask_audit": audit}
    finally:
        remove_hooks(treatment_handles)
        for handle in reversed(observer_handles):
            handle.remove()


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort", "cohort-preflight", "cpu-receipt", "video",
                 "time-map", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--minerva-json", type=Path,
        default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--assoc-info", type=Path,
        default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path,
        default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--manifest", type=Path,
        default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--user-authorized-local-only", action="store_true",
        required=True)
    args = parser.parse_args(argv)
    pilot._validate_private_path(args.output, must_exist=False)
    pilot._validate_private_path(args.cpu_receipt)
    if args.output.resolve() == args.cpu_receipt.resolve():
        raise ValueError("output_must_differ_from_cpu_receipt")
    prepared = pilot._prepare(cohort_path=args.cohort,
        cohort_preflight_path=args.cohort_preflight, row_index=args.row_index,
        video_path=args.video, time_map_path=args.time_map,
        minerva_json_path=args.minerva_json, assoc_info_path=args.assoc_info,
        mask_info_path=args.mask_info, manifest_path=args.manifest)
    cpu_raw = pilot._private_file(args.cpu_receipt)
    pilot._validate_cpu_receipt(json.loads(cpu_raw), prepared=prepared,
        row_index=args.row_index, receipt_raw=cpu_raw)
    support = tuple(prepared["support_positions"])
    sham = tuple(prepared["sham_positions"])
    prompt_length = int(prepared["batches"]["base"]["input_ids"].shape[1])
    if (not support or len(support) != len(sham) or set(support) & set(sham)
            or max((*support, *sham)) >= prompt_length - 1):
        raise ValueError("generation_key_sets_invalid")
    import torch
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("cuda_bf16_unavailable")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise ValueError("free_vram_below_12_gib")
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / pilot.MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    layers = tuple(index - 1 for index in pilot.LAYERS_1_BASED)
    key_sets = {"baseline": None, "bbox_key_bias": support,
                "spatial_sham_key_bias": sham}
    outcomes = {}
    try:
        for branch in BRANCHES:
            result = _generate_one(model, prepared["processor"],
                prepared["batches"]["base"], torch=torch,
                positions=key_sets[branch], layer_indices=layers)
            result["mechanical_diagnostic"] = classify_generation(
                result["generated_text"],
                [0] * result["generated_token_count"], prepared["object_name"])
            outcomes[branch] = result
    except Exception as exc:
        raw = str(exc)
        failure_code = (raw if isinstance(exc, ValueError)
                        and re.fullmatch(r"[a-z0-9_:]{1,200}", raw)
                        else type(exc).__name__)
        status = "failed_minerva_visible_object_generation_bias_v1"
    else:
        failure_code = None
        status = STATUS
    finally:
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated()
        del model
        torch.cuda.empty_cache()
    receipt = {"schema_version": 1, "status": status,
        "purpose": "exploratory effect of fixed bbox visual-key attention bias on generated object observations",
        "rights_status": "unverified; local-only per user authorization",
        "direct_faithfulness_validated": False,
        "cohort_sha256": pilot._sha(prepared["cohort_raw"]),
        "cohort_preflight_sha256": pilot._sha(prepared["cohort_preflight_raw"]),
        "cpu_receipt_sha256": pilot._sha(cpu_raw),
        "row_index": args.row_index,
        "row_key_sha256": pilot._sha(prepared["row_key"].encode()),
        "video_sha256": prepared["video_sha256"],
        "time_map_sha256": prepared["time_map_sha256"],
        "source_sha256": {"runner": pilot._sha_file(Path(__file__)),
            "pilot": pilot._sha_file(Path(pilot.__file__)),
            "generation_hooks": pilot._sha_file(Path(__file__).with_name(
                "qwen_visual_attention_boost.py")),
            "generation_audit": pilot._sha_file(Path(__file__).with_name(
                "run_qta_intervened_interval_generation.py"))},
        "model_revision": pilot.MODEL_REVISION,
        "runtime_versions": {"python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "gpu_name": torch.cuda.get_device_name(0)},
        "seed": SEED, "frame_count": pilot.FRAME_COUNT,
        "branch_order": list(BRANCHES), "bias_nats": math.log(2.0),
        "biased_layers_1_based": list(pilot.LAYERS_1_BASED),
        "key_dose": prepared["key_dose"],
        "prompt_sha256": prepared["rendered_prompt_sha256"],
        "base_pixel_tensor_sha256": prepared["pixel_tensor_sha256"]["base"],
        "outcomes": outcomes, "failure_code": failure_code,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": peak_vram, "visible_total_vram_bytes": total,
        "limitations": ["three videos; exploratory after v1/v2 outcomes",
            "mechanical text checks are not semantic support judgments",
            "annotation bbox approximates object; Qwen merges temporal frame pairs",
            "generation changes alone do not establish improved explanation faithfulness"]}
    pilot._write_private_json(args.output, receipt)
    print(json.dumps({"status": status, "row_index": args.row_index,
        "branches_completed": len(outcomes), "output": str(args.output)}))
    return 0 if status == STATUS else 1


if __name__ == "__main__":
    raise SystemExit(run())
