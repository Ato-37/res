#!/usr/bin/env python3
"""Synthetic-only Qwen 3-record x 6-branch intervention integration smoke.

Builds three 32-frame synthetic videos and full/evidence-mask/matched-control
inputs. Per row, an untreated full-video pass freezes the A--E answer/foil and
selects the top 10% of complete temporal groups from question-token attention.
The same selector is reused in all three treatment branches. It runs 18 total
model forwards and reports only synthetic
aggregate contrasts and resource counts; it is not a faithfulness experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fixed_prediction_selective_dependence import (  # noqa: E402
    INTERVENTION_CONDITIONS,
    score_six_condition_logits,
)
from derive_six_condition_binding import derive_six_condition_binding  # noqa: E402
from nextgqa_atomic_batch_freeze import (  # noqa: E402
    CONDITIONS,
    batch_sha256,
    freeze_all_batches,
)
from nextgqa_condition_batch_builder import build_exact_triplet_batches  # noqa: E402
from qwen_visual_attention_boost import (  # noqa: E402
    install_video_key_bias_hooks,
    remove_hooks,
)
from smoke_qwen_real_video_attention_boost import (  # noqa: E402
    _run_baseline_and_selector,
    _processor_fingerprint,
)
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt  # noqa: E402


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 32
LAYER_FIRST = 5
LAYER_LAST = 16
LAYER_INDICES = tuple(range(LAYER_FIRST - 1, LAYER_LAST))
VISUAL_RATIO = 0.10
LOGIT_BIAS = math.log(2.0)
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2
SEED = 20261018


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_records(np, image_type):
    y, x = np.mgrid[0:224, 0:224]
    records = []
    for row_index in range(3):
        frames = []
        for frame_index in range(FRAME_COUNT):
            rgb = np.stack((
                (x * 3 + frame_index * 7 + row_index * 17) % 256,
                (y * 5 + frame_index * 11 + row_index * 29) % 256,
                (x + y * 2 + frame_index * 13 + row_index * 43) % 256,
            ), axis=-1).astype(np.uint8)
            frames.append(image_type.fromarray(rgb, mode="RGB"))
        records.append({
            "frames": frames,
            "presentation_times": [Fraction(i, 1) for i in range(FRAME_COUNT)],
            "sampled_indices": list(range(FRAME_COUNT)),
            "intervals": [[Fraction(5), Fraction(8)]],
            "qa_item": {
                "question": "Which synthetic color channel changes most over time?",
                "choices": ["red", "green", "blue", "all equally", "cannot tell"],
                "gold": row_index % 3,
            },
            "sample_fps": Fraction(1, 1),
        })
    return records


def _write_private_json(path: Path, value: dict) -> None:
    """Write an exclusive mode-0600 artifact under a private directory."""
    if not path.parent.exists():
        if not path.parent.parent.is_dir():
            raise RuntimeError("private_output_parent_missing")
        path.parent.mkdir(mode=0o700)
    if not path.parent.is_dir() or path.parent.stat().st_mode & 0o077:
        raise RuntimeError("private_output_directory_permissions_invalid")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new aggregate-only JSON path; existing outputs are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("output_already_exists")

    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    stage = "initialize"
    model = None
    all_hooks = []
    attempted = completed = 0
    row_deltas = []
    binding_digests = []
    report = {"status": "incomplete", "scope": "synthetic_only_six_condition_integration"}
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_not_cached")
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        source_names = (
            "smoke_qwen_visual_attention_boost_six_condition.py",
            "smoke_qwen_real_video_attention_boost.py",
            "qwen_question_temporal_attention_selector.py",
            "qwen_visual_attention_boost.py",
            "nextgqa_condition_batch_builder.py",
            "nextgqa_atomic_batch_freeze.py",
            "nextgqa_batch_digest_freeze.py",
            "fixed_prediction_selective_dependence.py",
            "derive_six_condition_binding.py",
            "run_nextgqa_gold_interval_margin_pilot.py",
            "nextgqa_claim_mask_conditions.py",
        )
        source_hashes = {
            name: _sha256(Path(__file__).resolve().parent / name)
            for name in source_names
        }
        model_config_sha256 = _sha256(model_path / "config.json")
        processor_fingerprint = _processor_fingerprint(processor)
        records = synthetic_records(np, Image)
        built = build_exact_triplet_batches(records=records, processor=processor)
        if (built.get("status") != "condition_batches_constructed_unvalidated"
                or built.get("requested_records") != 3
                or built.get("requested_batches") != 9
                or built.get("model_loaded") is not False
                or built.get("model_forwards") != 0):
            raise RuntimeError("synthetic_condition_build_failed")
        label_ids = []
        for label in "ABCDE":
            pieces = processor.tokenizer(label, add_special_tokens=False).input_ids
            if len(pieces) != 1:
                raise RuntimeError("answer_label_token_contract_invalid")
            label_ids.append(int(pieces[0]))
        if len(set(label_ids)) != 5:
            raise RuntimeError("answer_label_token_ids_not_distinct")
        record_ids = [f"synthetic-{index}" for index in range(1, 4)]
        expected_hashes = {}
        for record_id, row in zip(record_ids, built["triplets"], strict=True):
            if set(row["batches"]) != set(CONDITIONS):
                raise RuntimeError("synthetic_condition_set_invalid")
            expected_hashes[record_id] = {
                condition: batch_sha256(row["batches"][condition])
                for condition in CONDITIONS
            }
        frozen = freeze_all_batches(
            triplets=[{"record_id": record_id, **row}
                      for record_id, row in zip(record_ids, built["triplets"], strict=True)],
            token_ids=label_ids, expected_record_ids=record_ids,
            expected_batch_hashes=expected_hashes,
            upstream_precheck=lambda: True,
            authenticated_gate_check=lambda: True)
        if (frozen.get("status") != "all_nine_batches_frozen"
                or frozen.get("requested_forwards") != 9
                or frozen.get("execution_authorization") != "not_granted"):
            raise RuntimeError("synthetic_all_batch_freeze_failed")

        stage = "derive_synthetic_six_condition_bindings"
        try:
            tokenizer_payload = processor.tokenizer.backend_tokenizer.to_str()
        except Exception:
            raise RuntimeError("synthetic_tokenizer_serialization_unavailable") from None
        tokenizer_hash = hashlib.sha256(tokenizer_payload.encode("utf-8")).hexdigest()
        six_condition_source = {
            "baseline_full": "full",
            "baseline_evidence_ablation": "claim_evidence_black",
            "baseline_control_ablation": "temporal_control_black",
            "intervention_full": "full",
            "intervention_evidence_ablation": "claim_evidence_black",
            "intervention_control_ablation": "temporal_control_black",
        }
        if len(built.get("condition_metadata", [])) != len(frozen["triplets"]):
            raise RuntimeError("synthetic_prompt_hash_count_invalid")
        for position, row in enumerate(frozen["triplets"]):
            source_batches = row["batches"]
            six_batches = {
                condition: source_batches[source_name]
                for condition, source_name in six_condition_source.items()
            }
            binding = derive_six_condition_binding(
                batches=six_batches,
                answer_token_ids=label_ids,
                model_revision=MODEL_REVISION,
                tokenizer_sha256=tokenizer_hash,
                prompt_sha256=built["condition_metadata"][position][
                    "rendered_prompt_sha256"],
            )
            if (binding.get("status")
                    != "derived_synthetic_batch_binding_contract_only"
                    or binding.get("bound_conditions") != 6):
                raise RuntimeError("synthetic_six_condition_binding_failed")
            # Record the complete typed-batch binding (including pixel/video
            # tensors and processor metadata), not only the text-token binding.
            binding_digests.append(binding["full_binding_sha256"])

        free_before, total = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")
        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise RuntimeError("model_processor_video_token_mismatch")
        if len(model.model.language_model.layers) < LAYER_LAST:
            raise RuntimeError("required_layers_unavailable")
        torch.cuda.reset_peak_memory_stats(device)

        def guarded_forward(batch):
            nonlocal attempted, completed
            free, _total = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("synthetic_gpu_resource_floor")
            attempted += 1
            with torch.inference_mode():
                output = model(**batch, use_cache=False, logits_to_keep=1)
            if (not torch.is_tensor(output.logits)
                    or not bool(torch.isfinite(output.logits).all())):
                raise RuntimeError("nonfinite_condition_logits")
            completed += 1
            return output

        selectors = []
        stage = "eighteen_condition_forwards"
        for row_position, (record_id, row, source_record) in enumerate(
                zip(frozen["record_ids"], frozen["triplets"], records, strict=True), 1):
            batches = row["batches"]
            hashes = frozen["batch_sha256"][record_id]
            input_ids = batches["full"]["input_ids"]
            if (not torch.is_tensor(input_ids) or input_ids.ndim != 2
                    or int(input_ids.shape[0]) != 1):
                raise RuntimeError(f"record_{row_position}_input_ids_invalid")
            for condition in CONDITIONS:
                if not torch.equal(input_ids, batches[condition]["input_ids"]):
                    raise RuntimeError(f"record_{row_position}_input_ids_differ")
            query_length = int(input_ids.shape[1])
            if any(int(batches[condition]["input_ids"].shape[1]) != query_length
                   for condition in CONDITIONS):
                raise RuntimeError(f"record_{row_position}_prompt_length_differs")
            outputs = {}
            lengths = {}
            baseline_full_cpu = batches["full"]
            if batch_sha256(baseline_full_cpu) != hashes["full"]:
                raise RuntimeError(f"record_{row_position}_baseline_batch_hash_mismatch")
            batch = baseline_full_cpu.to("cuda")
            if batch_sha256(batch) != hashes["full"]:
                raise RuntimeError(f"record_{row_position}_baseline_transfer_hash_mismatch")
            free, _total = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("synthetic_gpu_resource_floor")
            rendered_prompt, _messages = build_qa_prompt(
                processor, source_record["qa_item"], source_record["frames"],
                (source_record["sample_fps"].numerator,
                 source_record["sample_fps"].denominator))
            def baseline_forward():
                return guarded_forward(batch)
            (baseline_full_output, selected_positions, video_token_count,
             query_position, selected_groups, question_token_count,
             temporal_group_count, capture_report) = _run_baseline_and_selector(
                model=model, batch=batch, processor=processor,
                torch_module=torch, rendered_text=rendered_prompt,
                question=source_record["qa_item"]["question"],
                forward_callable=baseline_forward)
            outputs["baseline_full"] = baseline_full_output
            lengths["baseline_full"] = query_length
            selectors.append({"selected": selected_positions,
                              "visual_count": video_token_count,
                              "query_position": query_position,
                              "selected_group_indices": selected_groups,
                              "question_token_count": question_token_count,
                              "temporal_group_count": temporal_group_count,
                              "attention_capture": capture_report})

            for condition, name in (("claim_evidence_black", "baseline_evidence_ablation"),
                                    ("temporal_control_black", "baseline_control_ablation")):
                cpu_batch = batches[condition]
                if batch_sha256(cpu_batch) != hashes[condition]:
                    raise RuntimeError(f"record_{row_position}_{condition}_hash_mismatch")
                device_batch = cpu_batch.to("cuda")
                if batch_sha256(device_batch) != hashes[condition]:
                    raise RuntimeError(f"record_{row_position}_{condition}_transfer_hash_mismatch")
                outputs[name] = guarded_forward(device_batch)
                lengths[name] = query_length

            for condition, name in (("full", "intervention_full"),
                                    ("claim_evidence_black", "intervention_evidence_ablation"),
                                    ("temporal_control_black", "intervention_control_ablation")):
                cpu_batch = batches[condition]
                expected = hashes[condition]
                if batch_sha256(cpu_batch) != expected:
                    raise RuntimeError(f"record_{row_position}_{condition}_hash_changed")
                device_batch = cpu_batch.to("cuda")
                if batch_sha256(device_batch) != expected:
                    raise RuntimeError(f"record_{row_position}_{condition}_transfer_hash_changed")
                hooks = install_video_key_bias_hooks(
                    model=model, layer_indices=LAYER_INDICES,
                    selected_positions=selected_positions,
                    query_position=query_position, logit_bias=LOGIT_BIAS,
                    torch_module=torch)
                all_hooks.extend(hooks)
                try:
                    outputs[name] = guarded_forward(device_batch)
                finally:
                    remove_hooks(hooks)
                    for handle in hooks:
                        if handle in all_hooks:
                            all_hooks.remove(handle)
                lengths[name] = query_length

            if set(outputs) != set(INTERVENTION_CONDITIONS):
                raise RuntimeError(f"record_{row_position}_six_branch_set_invalid")
            score = score_six_condition_logits(
                condition_outputs=outputs, answer_token_ids=label_ids,
                input_lengths=lengths, torch_module=torch)
            row_deltas.append(float(score["contrast"]["delta_f"]))
            del outputs, batch, baseline_full_output, device_batch, cpu_batch
            torch.cuda.empty_cache()

        if (attempted != 18 or completed != 18 or len(row_deltas) != 3
                or not all(math.isfinite(value) for value in row_deltas)):
            raise RuntimeError("synthetic_complete_cohort_invariant_failed")
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        mean_delta = sum(row_deltas) / len(row_deltas)
        sd = math.sqrt(sum((value - mean_delta) ** 2 for value in row_deltas)
                       / (len(row_deltas) - 1))
        report = {
            "status": "passed",
            "scope": "synthetic_only_six_condition_integration_not_efficacy",
            "started_at_utc": started,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": model_config_sha256,
            "processor_fingerprint_sha256": processor_fingerprint,
            "source_sha256": source_hashes,
            "seed": SEED,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "precision": "bf16_4bit_bf16_compute",
            "requested_records": 3,
            "completed_records": len(row_deltas),
            "requested_forwards": 18,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "derived_input_binding_records": len(binding_digests),
            "derived_input_binding_sha256": binding_digests,
            "conditions": ["baseline_full", "baseline_evidence_ablation",
                           "baseline_control_ablation", "intervention_full",
                           "intervention_evidence_ablation", "intervention_control_ablation"],
            "selector": "top_10_percent_mean_layer5_16_question_token_attention_by_processor_temporal_group_from_baseline_full_frozen_per_record",
            "selected_video_token_counts": [item["visual_count"] for item in selectors],
            "selected_key_counts": [len(item["selected"]) for item in selectors],
            "question_token_counts": [item["question_token_count"] for item in selectors],
            "temporal_group_counts": [item["temporal_group_count"] for item in selectors],
            "selected_temporal_group_indices": [item["selected_group_indices"] for item in selectors],
            "intervention_layers_1_based": [LAYER_FIRST, LAYER_LAST],
            "key_attention_logit_bias": LOGIT_BIAS,
            "synthetic_delta_f_mean_nats": mean_delta,
            "synthetic_delta_f_sample_sd_nats": sd,
            "faithfulness_result": False,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "limitations": [
                "all three records and videos are synthetic; synthetic DeltaF has no research interpretation",
                "selector and ln(2) strength are engineering constants, not efficacy-validated",
                "human claim review, clock calibration, source authentication, and real-data gates are absent",
                "no confidence interval or hypothesis test is computed",
            ],
        }
    except Exception as exc:
        remove_hooks(all_hooks)
        failure_code = str(exc)
        if not failure_code.isascii() or not failure_code.replace("_", "").isalnum():
            failure_code = f"smoke_failed_{stage}_{type(exc).__name__}"
        report.update({
            "failure_code": failure_code,
            "failure_stage": stage,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "completed_records": len(row_deltas),
            "requested_records": 3,
            "requested_forwards": 18,
            "faithfulness_result": False,
            "execution_authorization": "not_granted",
            "elapsed_seconds": time.monotonic() - timer,
        })
        status = 2
    else:
        report["execution_authorization"] = "not_granted"
        status = 0
    finally:
        remove_hooks(all_hooks)
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        report.setdefault("execution_authorization", "not_granted")
        report.setdefault("faithfulness_result", False)
        report.setdefault("elapsed_seconds", time.monotonic() - timer)
        _write_private_json(args.output, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
