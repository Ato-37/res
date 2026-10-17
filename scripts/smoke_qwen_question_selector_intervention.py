#!/usr/bin/env python3
"""Synthetic-only GPU integration smoke for question-selected attention bias.

Runs baseline, question-selector treatment, and deterministic sham across
full/evidence-key-ablation/control-key-ablation branches. It checks hook
composition, immutable inputs, fixed-prediction score extraction, and finite
outputs. Its synthetic score diagnostics are not faithfulness evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import stat

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_internal_attention_ablation import (  # noqa: E402
    install_video_key_intervention_hooks, remove_hooks,
)
from qwen_question_temporal_attention_selector import (  # noqa: E402
    aggregate_temporal_group_attention, map_question_token_positions,
    select_hash_sham_temporal_groups, select_top_temporal_groups,
)
from fixed_prediction_selective_dependence import (  # noqa: E402
    score_bound_six_condition_logits,
)
from derive_six_condition_binding import derive_six_condition_binding  # noqa: E402
from smoke_qwen_question_temporal_selector import (  # noqa: E402
    LAYERS_1_BASED, MODEL_REVISION, _processor_fingerprint, synthetic_records,
)
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256  # noqa: E402
from qta_nine_branch_plan import build_qta_nine_branch_plan  # noqa: E402
from run_nextgqa_gold_interval_margin_pilot import (  # noqa: E402
    build_qa_prompt, process_qa_video,
)


LOGIT_BIAS = math.log(2.0)
SHAM_SEED = "qta-sham-seed-20261018-v1"
MIN_FREE_BEFORE_LOAD = 11 * 1024**3
MIN_FREE_BEFORE_FORWARD = 1024**3


def temporal_group_overlap(*, selector_groups, comparison_groups,
                           temporal_groups):
    """Derive group and visual-key overlap from one temporal partition."""
    def checked_indices(values, name):
        if (not isinstance(values, (tuple, list)) or not values
                or any(type(value) is not int or value < 0 for value in values)
                or tuple(sorted(set(values))) != tuple(values)):
            raise ValueError(f"{name}_invalid")
        return tuple(values)

    left_groups = checked_indices(selector_groups, "selector_groups")
    right_groups = checked_indices(comparison_groups, "comparison_groups")
    if (not isinstance(temporal_groups, (tuple, list)) or not temporal_groups
            or max((*left_groups, *right_groups)) >= len(temporal_groups)):
        raise ValueError("temporal_groups_invalid")
    normalized_groups = []
    for group in temporal_groups:
        if (not isinstance(group, (tuple, list)) or not group
                or any(type(position) is not int or position < 0 for position in group)
                or tuple(sorted(set(group))) != tuple(group)):
            raise ValueError("temporal_group_positions_invalid")
        normalized_groups.append(tuple(group))
    flattened_positions = [position for group in normalized_groups for position in group]
    if len(set(flattened_positions)) != len(flattened_positions):
        raise ValueError("temporal_group_positions_overlap")
    left_positions = tuple(sorted(
        position for index in left_groups for position in normalized_groups[index]))
    right_positions = tuple(sorted(
        position for index in right_groups for position in normalized_groups[index]))
    shared_groups = sorted(set(left_groups) & set(right_groups))
    shared_positions = set(left_positions) & set(right_positions)
    return {
        "overlap_group_indices": shared_groups,
        "overlap_group_count": len(shared_groups),
        "selector_group_count": len(left_groups),
        "comparison_group_count": len(right_groups),
        "selector_group_fraction": len(shared_groups) / len(left_groups),
        "comparison_group_fraction": len(shared_groups) / len(right_groups),
        "overlap_visual_key_count": len(shared_positions),
        "selector_visual_key_count": len(left_positions),
        "comparison_visual_key_count": len(right_positions),
        "selector_visual_key_fraction": len(shared_positions) / len(left_positions),
        "comparison_visual_key_fraction": len(shared_positions) / len(right_positions),
    }


def _choice_log_probabilities(logits, token_ids, torch_module):
    """Extract full-vocabulary next-token log probabilities for A–E IDs."""
    if (not torch_module.is_tensor(logits) or logits.ndim != 3
            or tuple(logits.shape[:2]) != (1, 1)
            or not torch_module.isfinite(logits).all()
            or len(token_ids) != 5
            or any(type(token_id) is not int or token_id < 0
                   or token_id >= int(logits.shape[-1]) for token_id in token_ids)
            or len(set(token_ids)) != 5):
        raise ValueError("choice_score_logits_or_token_ids_invalid")
    log_probs = torch_module.log_softmax(logits[0, -1].float(), dim=-1)
    selected = log_probs[list(token_ids)].detach().cpu().tolist()
    if len(selected) != 5 or not all(math.isfinite(float(value)) for value in selected):
        raise ValueError("choice_score_nonfinite")
    return [float(value) for value in selected]


def _resolve_choice_continuation_ids(tokenizer, rendered_prefix, labels="ABCDE"):
    """Resolve single-token labels as continuations of the exact rendered prefix."""
    if (not isinstance(rendered_prefix, str) or not rendered_prefix
            or not isinstance(labels, str) or not labels
            or len(set(labels)) != len(labels)):
        raise ValueError("choice_continuation_prefix_or_labels_invalid")
    try:
        prefix_ids = tokenizer.encode(rendered_prefix, add_special_tokens=False)
    except Exception:
        raise ValueError("choice_continuation_prefix_tokenization_failed") from None
    if (not isinstance(prefix_ids, list) or not prefix_ids
            or any(type(value) is not int or value < 0 for value in prefix_ids)):
        raise ValueError("choice_continuation_prefix_ids_invalid")
    result = []
    for label in labels:
        try:
            extended_ids = tokenizer.encode(rendered_prefix + label,
                                            add_special_tokens=False)
        except Exception:
            raise ValueError("choice_continuation_tokenization_failed") from None
        if (not isinstance(extended_ids, list)
                or extended_ids[:len(prefix_ids)] != prefix_ids):
            raise ValueError("choice_continuation_prefix_retokenized")
        suffix = extended_ids[len(prefix_ids):]
        if (len(suffix) != 1 or type(suffix[0]) is not int or suffix[0] < 0):
            raise ValueError("choice_label_not_single_continuation_token")
        result.append(suffix[0])
    if len(set(result)) != len(result):
        raise ValueError("choice_continuation_token_ids_not_distinct")
    return result


def _validate_processed_choice_suffix(base_ids, extended_ids, expected_token_id,
                                      torch_module):
    """Require the full multimodal processor batch to append exactly one label ID."""
    if (not torch_module.is_tensor(base_ids) or not torch_module.is_tensor(extended_ids)
            or base_ids.ndim != 2 or tuple(base_ids.shape[:1]) != (1,)
            or tuple(extended_ids.shape) != (1, int(base_ids.shape[1]) + 1)
            or type(expected_token_id) is not int or expected_token_id < 0):
        raise ValueError("processed_choice_suffix_shape_invalid")
    if not torch_module.equal(extended_ids[:, :-1], base_ids):
        raise ValueError("processed_choice_suffix_prefix_mismatch")
    if int(extended_ids[0, -1].item()) != expected_token_id:
        raise ValueError("processed_choice_suffix_token_mismatch")


def _score_treatment_branches(six_condition_binding, branch_logits, torch_module,
                              *, treatment_prefix="target"):
    """Bind baseline plus one treatment's three logits to the fixed scorer."""
    if treatment_prefix not in {"target", "sham"}:
        raise ValueError("treatment_prefix_invalid")
    error_code = ("target_score_branch_set_invalid" if treatment_prefix == "target"
                  else "sham_score_branch_set_invalid")
    expected = {
        "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
        f"{treatment_prefix}_full",
        f"{treatment_prefix}_evidence_ablation",
        f"{treatment_prefix}_control_ablation",
    }
    if not isinstance(branch_logits, dict) or set(branch_logits) != expected:
        raise ValueError(error_code)
    mapped = {
        (name.replace(f"{treatment_prefix}_", "intervention_", 1)
         if name.startswith(f"{treatment_prefix}_") else name): output
        for name, output in branch_logits.items()
    }
    binding_rows = (six_condition_binding.get("branch_bindings")
                    if isinstance(six_condition_binding, dict) else None)
    if not isinstance(binding_rows, dict):
        raise ValueError("six_condition_branch_bindings_missing")
    return score_bound_six_condition_logits(
        branch_bindings=binding_rows, condition_outputs=mapped,
        torch_module=torch_module,
    )


def _hash_checkpoint_shards(model_path: Path):
    """Hash the exact cached weight shards before loading the pinned model."""
    shards = sorted(model_path.glob("model-*-of-*.safetensors"))
    if len(shards) != 5 or any(not path.is_file() for path in shards):
        raise RuntimeError("pinned_checkpoint_shard_inventory_invalid")
    result = {}
    for path in shards:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


@dataclass
class _PrivateOutputTarget:
    root_fd: int
    run_fd: int
    directory_name: str
    filename: str
    display_path: Path


def _prepare_private_output(path: Path, private_root: Path) -> _PrivateOutputTarget:
    """Create/pin one fresh 0700 run dir below a non-symlink private root."""
    root = Path(os.path.abspath(private_root))
    data_dir = root.parent
    try:
        data_st = data_dir.lstat()
        root_st = root.lstat()
    except OSError:
        raise ValueError("private_output_root_invalid") from None
    if (stat.S_ISLNK(data_st.st_mode) or stat.S_ISLNK(root_st.st_mode)
            or not stat.S_ISDIR(root_st.st_mode)
            or stat.S_IMODE(root_st.st_mode) != 0o700
            or root_st.st_uid != os.getuid()):
        raise ValueError("private_output_root_mode_invalid")
    requested = path if path.is_absolute() else Path.cwd() / path
    requested = Path(os.path.abspath(requested))
    try:
        relative = requested.relative_to(root)
    except ValueError:
        raise ValueError("output_must_be_under_private_nextgqa") from None
    if len(relative.parts) != 2 or relative.parts[1] in {"", ".", ".."}:
        raise ValueError("output_must_use_fresh_private_run_directory")
    directory_name, filename = relative.parts
    if filename in {".", ".."}:
        raise ValueError("private_output_filename_invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    data_fd = os.open(data_dir, directory_flags)
    try:
        data_open_st = os.fstat(data_fd)
        if ((data_open_st.st_dev, data_open_st.st_ino) != (data_st.st_dev, data_st.st_ino)
                or data_open_st.st_uid != os.getuid()):
            raise ValueError("private_output_data_directory_changed")
        root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        root_flags |= getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root.name, root_flags, dir_fd=data_fd)
    finally:
        os.close(data_fd)
    root_open_st = os.fstat(root_fd)
    if ((root_open_st.st_dev, root_open_st.st_ino) != (root_st.st_dev, root_st.st_ino)
            or root_open_st.st_uid != os.getuid()
            or stat.S_IMODE(root_open_st.st_mode) != 0o700):
        os.close(root_fd)
        raise ValueError("private_output_root_changed")
    try:
        os.mkdir(directory_name, 0o700, dir_fd=root_fd)
    except FileExistsError:
        os.close(root_fd)
        raise ValueError("private_output_run_directory_already_exists") from None
    except BaseException:
        os.close(root_fd)
        raise
    run_fd = None
    try:
        run_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        run_flags |= getattr(os, "O_NOFOLLOW", 0)
        run_fd = os.open(directory_name, run_flags, dir_fd=root_fd)
        os.fchmod(run_fd, 0o700)
        run_st = os.fstat(run_fd)
        if (not stat.S_ISDIR(run_st.st_mode) or run_st.st_uid != os.getuid()
                or stat.S_IMODE(run_st.st_mode) != 0o700):
            raise ValueError("private_output_run_directory_invalid")
    except BaseException:
        if run_fd is not None:
            os.close(run_fd)
        try:
            os.rmdir(directory_name, dir_fd=root_fd)
        finally:
            os.close(root_fd)
        raise
    return _PrivateOutputTarget(
        root_fd, run_fd, directory_name, filename, root / directory_name / filename)


def _write_private_json(target: _PrivateOutputTarget, payload) -> None:
    """Exclusively create a 0600 JSON file at the pinned run directory fd."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target.filename, flags, 0o600, dir_fd=target.run_fd)
    except BaseException:
        try:
            os.rmdir(target.directory_name, dir_fd=target.root_fd)
        except OSError:
            pass
        raise
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(target.filename, dir_fd=target.run_fd)
        except OSError:
            pass
        try:
            os.rmdir(target.directory_name, dir_fd=target.root_fd)
        except OSError:
            pass
        raise


def _close_private_output(target: _PrivateOutputTarget | None) -> None:
    if target is None:
        return
    for fd in (target.run_fd, target.root_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new aggregate-only JSON path; existing outputs are refused")
    args = parser.parse_args(argv)
    import numpy as np
    import PIL
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    bnb_version = importlib.metadata.version("bitsandbytes")
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_impl
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processing_impl
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen_image_impl
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_impl
    import qwen_vl_utils.vision_process as qwen_utils_impl
    from qwen_vl_utils import process_vision_info

    repo = Path(__file__).resolve().parents[1]
    local_sources = {
        "runner": Path(__file__).resolve(),
        "selector_runner_fixture": repo / "scripts/smoke_qwen_question_temporal_selector.py",
        "synthetic_fixture": repo / "scripts/smoke_qwen_internal_attention_ablation.py",
        "selector": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "qta_nine_branch_plan": repo / "scripts/qta_nine_branch_plan.py",
        "attention_bias": repo / "scripts/qwen_visual_attention_boost.py",
        "internal_ablation": repo / "scripts/qwen_internal_attention_ablation.py",
        "fixed_prediction_scorer": repo / "scripts/fixed_prediction_selective_dependence.py",
        "typed_batch_hash": repo / "scripts/nextgqa_atomic_batch_freeze.py",
        "prompt_processor_adapter": repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        "processor_batch_mapping": repo / "scripts/qwen_processor_batch_review_mapping.py",
        "temporal_group_mapping": repo / "scripts/qwen_review_interval_token_groups.py",
        "condition_batch_builder": repo / "scripts/nextgqa_condition_batch_builder.py",
        "claim_mask_builder": repo / "scripts/nextgqa_claim_mask_conditions.py",
        "claim_mask_video_decoder": repo / "scripts/claim_mask_video_decoder.py",
        "mask_condition_dependency": repo / "scripts/necessity_choice_margin_utils.py",
        "presentation_time_utils": repo / "scripts/presentation_time_utils.py",
    }
    external_modules = {
        "transformers_qwen_model_impl": qwen_model_impl,
        "transformers_qwen_processing_impl": qwen_processing_impl,
        "transformers_qwen_image_processor_impl": qwen_image_impl,
        "transformers_video_processor_impl": qwen_video_impl,
        "qwen_vl_utils_video_impl": qwen_utils_impl,
    }
    report = {
        "status": "incomplete",
        "scope": "synthetic_question_selector_bias_and_key_ablation_wiring_only",
        "faithfulness_result": False,
        "treatment_frozen": False,
        "execution_authorization": "not_granted",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": importlib.metadata.version("qwen-vl-utils"),
        "bitsandbytes_version": bnb_version,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "checkpoint_weights_hash_status": "pending_preload_hash",
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in local_sources.items()
        },
    }
    report["source_sha256"].update({
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in external_modules.items()
    })
    try:
        output_target = _prepare_private_output(
            args.output, repo / "data" / "private_nextgqa")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    args.output = output_target.display_path

    timer = time.monotonic()
    model = None
    capture_handles = []
    attempted = completed = 0
    stage = "initialize"
    exit_code = 0
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_not_cached")
        stage = "hash_checkpoint_shards"
        report["checkpoint_weight_shards"] = _hash_checkpoint_shards(model_path)
        report["checkpoint_weights_hash_status"] = "preload_shards_sha256_recorded"
        stage = "load_model"
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        record = synthetic_records(np, Image)[0]
        item = {**record, **record["qa_item"]}
        sample_fps = (record["sample_fps"].numerator,
                      record["sample_fps"].denominator)
        rendered_text, messages = build_qa_prompt(
            processor, item, record["frames"], sample_fps)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or len(videos) != 1:
            raise RuntimeError("synthetic_video_processor_contract_failed")
        pixels, metadata = videos[0]
        batch = FrozenBatch(process_qa_video(processor, rendered_text, images,
                                             pixels, metadata))
        ids = batch["input_ids"]
        if ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise RuntimeError("synthetic_prompt_batch_invalid")
        option_token_ids = _resolve_choice_continuation_ids(
            processor.tokenizer, rendered_text, labels="ABCDE")
        for label, expected_token_id in zip("ABCDE", option_token_ids, strict=True):
            extended_batch = process_qa_video(
                processor, rendered_text + label, images, pixels, metadata)
            _validate_processed_choice_suffix(
                batch["input_ids"], extended_batch["input_ids"],
                expected_token_id, torch_module=torch,
            )
        del extended_batch
        six_condition_names = (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation",
        )
        tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
        six_condition_binding = derive_six_condition_binding(
            batches={name: batch for name in six_condition_names},
            answer_token_ids=option_token_ids, model_revision=MODEL_REVISION,
            tokenizer_sha256=hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest(),
            prompt_sha256=hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
        )
        preload_batch_digest = batch_sha256(batch)
        question_positions = map_question_token_positions(
            tokenizer=processor.tokenizer, rendered_text=rendered_text,
            question_text=item["question"], processed_input_ids=ids,
            video_token_id=int(processor.video_token_id))
        query_position = int(ids.shape[1]) - 1
        attention_mask = batch.get("attention_mask")
        if (attention_mask is None or tuple(attention_mask.shape) != tuple(ids.shape)
                or not bool((attention_mask == 1).all())
                or max(question_positions) >= query_position):
            raise RuntimeError("synthetic_single_prompt_query_geometry_invalid")
        visual_positions = [i for i, token in enumerate(ids[0].tolist())
                            if token == int(processor.video_token_id)]
        temporal_steps, grid_h, grid_w = map(
            int, batch["video_grid_thw"][0].tolist())
        tokens_per_group = (grid_h * grid_w
                            // int(processor.image_processor.merge_size) ** 2)
        if len(visual_positions) != temporal_steps * tokens_per_group:
            raise RuntimeError("synthetic_temporal_group_geometry_invalid")
        groups = [tuple(visual_positions[i * tokens_per_group:(i + 1) * tokens_per_group])
                  for i in range(temporal_steps)]
        if temporal_steps < 4:
            raise RuntimeError("synthetic_temporal_group_count_invalid")

        free_before, total_vram = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")
        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
            raise RuntimeError("model_processor_or_layer_contract_invalid")
        device_batch = batch.to("cuda")
        batch_digest = batch_sha256(device_batch)
        if batch_digest != preload_batch_digest:
            raise RuntimeError("synthetic_device_batch_digest_mismatch")
        free_before_forward, _ = torch.cuda.mem_get_info(device)
        if free_before_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("synthetic_gpu_resource_floor")

        captured = {}
        stage = "capture_question_attention"
        for layer_1_based in LAYERS_1_BASED:
            layer_index = layer_1_based - 1

            def capture(_module, _inputs, output, *, selected_layer=layer_index):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise RuntimeError("baseline_attention_capture_unavailable")
                captured[selected_layer] = aggregate_temporal_group_attention(
                    attention_by_layer={selected_layer: output[1]},
                    layer_indices=(selected_layer,),
                    question_positions=question_positions,
                    group_positions=groups, torch_module=torch)

            capture_handles.append(
                model.model.language_model.layers[layer_index].self_attn
                .register_forward_hook(capture))
        torch.cuda.reset_peak_memory_stats(device)
        attempted += 1
        with torch.inference_mode():
            baseline_output = model(**device_batch, use_cache=False, logits_to_keep=1)
        if (batch_sha256(device_batch) != batch_digest
                or not torch.isfinite(baseline_output.logits).all()):
            raise RuntimeError("baseline_snapshot_or_logits_invalid")
        baseline_full_choice_log_probs = _choice_log_probabilities(
            baseline_output.logits, option_token_ids, torch)
        baseline_full_logits = baseline_output.logits.detach().to("cpu")
        del baseline_output
        completed += 1
        if tuple(sorted(captured)) != tuple(i - 1 for i in LAYERS_1_BASED):
            raise RuntimeError("question_attention_capture_incomplete")
        group_scores = [sum(captured[layer][group] for layer in captured) / len(captured)
                        for group in range(temporal_steps)]
        top_k = max(1, math.ceil(temporal_steps * 0.10))
        target_groups = select_top_temporal_groups(group_scores, top_k=top_k)
        sham_groups = select_hash_sham_temporal_groups(
            temporal_group_count=temporal_steps, top_k=top_k,
            protocol_seed=SHAM_SEED, opaque_record_token="synthetic-record-0")
        target_positions = tuple(sorted(p for index in target_groups for p in groups[index]))
        sham_positions = tuple(sorted(p for index in sham_groups for p in groups[index]))

        # Synthetic diagnostic branches only. Group choices have no annotation semantics.
        evidence_groups = tuple(range(2))
        control_groups = tuple(range(temporal_steps - 2, temporal_steps))
        evidence_positions = tuple(sorted(p for index in evidence_groups for p in groups[index]))
        control_positions = tuple(sorted(p for index in control_groups for p in groups[index]))
        target_evidence_overlap = temporal_group_overlap(
            selector_groups=target_groups, comparison_groups=(0, 1),
            temporal_groups=groups)
        target_control_overlap = temporal_group_overlap(
            selector_groups=target_groups,
            comparison_groups=tuple(range(temporal_steps - 2, temporal_steps)),
            temporal_groups=groups)
        sham_evidence_overlap = temporal_group_overlap(
            selector_groups=sham_groups, comparison_groups=(0, 1),
            temporal_groups=groups)
        sham_control_overlap = temporal_group_overlap(
            selector_groups=sham_groups,
            comparison_groups=tuple(range(temporal_steps - 2, temporal_steps)),
            temporal_groups=groups)
        remove_hooks(capture_handles)
        capture_handles = []
        condition_outputs = {}
        branch_logits = {}

        def forward_condition(name, *, bias_positions=(), ablation_positions=()):
            nonlocal attempted, completed
            hooks = []
            try:
                hooks = install_video_key_intervention_hooks(
                    model=model, layer_indices=tuple(i - 1 for i in LAYERS_1_BASED),
                    biased_positions=bias_positions,
                    ablated_positions=ablation_positions,
                    query_position=int(ids.shape[1]) - 1,
                    logit_bias=LOGIT_BIAS, torch_module=torch)
                if batch_sha256(device_batch) != batch_digest:
                    raise RuntimeError("synthetic_batch_digest_changed_before_condition")
                attempted += 1
                with torch.inference_mode():
                    output = model(**device_batch, use_cache=False, logits_to_keep=1)
                if (batch_sha256(device_batch) != batch_digest
                        or not torch.isfinite(output.logits).all()):
                    raise RuntimeError("condition_snapshot_or_logits_invalid")
                condition_outputs[name] = {
                    "logits_finite": True,
                    "logits_shape": list(output.logits.shape),
                    "choice_log_probabilities": _choice_log_probabilities(
                        output.logits, option_token_ids, torch),
                }
                branch_logits[name] = {"logits": output.logits.detach().to("cpu")}
                completed += 1
                del output
            finally:
                remove_hooks(hooks)

        stage = "nine_branch_hook_composition"
        branch_plan = build_qta_nine_branch_plan(
            row_binding_sha256=hashlib.sha256(b"synthetic-record-0").hexdigest(),
            temporal_groups=groups, target_group_indices=target_groups,
            sham_group_indices=sham_groups, evidence_group_indices=evidence_groups,
            control_group_indices=control_groups,
            typed_batch_sha256=preload_batch_digest,
            score_binding_sha256=six_condition_binding["full_binding_sha256"],
            intervention_layers_1_based=LAYERS_1_BASED, logit_bias=LOGIT_BIAS,
            sham_seed=SHAM_SEED)
        branch_inputs = {branch["name"]: branch for branch in branch_plan["branches"]}
        # Reuse the already completed baseline full pass.
        condition_outputs["baseline_full"] = {
            "logits_finite": True, "forward_reused": True,
            "choice_log_probabilities": baseline_full_choice_log_probs,
        }
        branch_logits["baseline_full"] = {"logits": baseline_full_logits}
        for branch in branch_plan["branches"][1:]:
            forward_condition(
                branch["name"],
                bias_positions=branch["biased_visual_key_positions"],
                ablation_positions=branch["ablated_visual_key_positions"])
        if set(condition_outputs) != set(branch_inputs) or completed != 9 or attempted != 9:
            raise RuntimeError("synthetic_nine_condition_incomplete")

        target_score = _score_treatment_branches(
            six_condition_binding,
            {name: branch_logits[name] for name in (
                "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
                "target_full", "target_evidence_ablation", "target_control_ablation")},
            torch)
        target_contrast = target_score["contrast"]
        sham_score = _score_treatment_branches(
            six_condition_binding,
            {name: branch_logits[name] for name in (
                "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
                "sham_full", "sham_evidence_ablation", "sham_control_ablation")},
            torch, treatment_prefix="sham")
        sham_contrast = sham_score["contrast"]

        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        report.update({
            "status": "passed",
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "processor_fingerprint_sha256": _processor_fingerprint(processor),
            "precision": "bf16_4bit_bf16_compute_eager",
            "requested_records": 1,
            "completed_records": 1,
            "requested_forwards": 9,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "captured_layers_1_based": list(LAYERS_1_BASED),
            "question_token_count": len(question_positions),
            "temporal_group_count": temporal_steps,
            "tokens_per_temporal_group": tokens_per_group,
            "top_k_budget_rule": "max(1,ceil(temporal_groups/10))_engineering_smoke_only",
            "selected_target_group_indices": target_groups,
            "selected_sham_group_indices": sham_groups,
            "target_sham_overlap_count": len(set(target_groups) & set(sham_groups)),
            "question_selector_vs_synthetic_evidence_overlap": target_evidence_overlap,
            "question_selector_vs_synthetic_control_overlap": target_control_overlap,
            "sham_vs_synthetic_evidence_overlap": sham_evidence_overlap,
            "sham_vs_synthetic_control_overlap": sham_control_overlap,
            "sham_seed": SHAM_SEED,
            "sham_selector": "sha256_utf8_qta-sham-v1_seed_opaque-token_group-index",
            "logit_bias": LOGIT_BIAS,
            "intervention_layers_1_based": list(LAYERS_1_BASED),
            "condition_names": [branch["name"] for branch in branch_plan["branches"]],
            "condition_plan_sha256": branch_plan["plan_sha256"],
            "selection_spec_sha256": branch_plan["selection_spec_sha256"],
            "condition_branch_sha256": {
                branch["name"]: branch["branch_sha256"]
                for branch in branch_plan["branches"]},
            "condition_forward_checks": condition_outputs,
            "synthetic_fixed_prediction_score_diagnostic": {
                "status": "synthetic_numerical_integration_only",
                "target_score_binding_status": target_score["status"],
                "sham_score_binding_status": sham_score["status"],
                "option_label_order": list("ABCDE"),
                "option_token_ids": option_token_ids,
                "target_delta_f": target_contrast["delta_f"],
                "sham_delta_f": sham_contrast["delta_f"],
                "target_minus_sham_delta_f": (
                    target_contrast["delta_f"] - sham_contrast["delta_f"]),
                "answer_index_frozen_from_baseline_full": target_contrast["answer_index"],
                "foil_index_frozen_from_baseline_full": target_contrast["foil_index"],
                "interpretation": "synthetic score path only; no faithfulness result",
            },
            "evidence_control_key_counts_equal": len(evidence_positions) == len(control_positions),
            "typed_batch_sha256": batch_digest,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
            "free_vram_before_forward_gib": free_before_forward / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "faithfulness_result": False,
            "treatment_frozen": False,
            "limitations": [
                "synthetic video only; no semantic evidence or treatment efficacy",
                "the selector layer range, 10-percent group budget, ln(2) dose, and sham seed are engineering smoke settings",
                "synthetic evidence/control groups have no annotation meaning",
                "synthetic A-E scores and contrasts are numerical diagnostics only; no real-data effect, interval, or hypothesis test",
            ],
        })
    except Exception as exc:
        remove_hooks(capture_handles)
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"question_intervention_smoke_failed_{type(exc).__name__}"
        report.update({"failure_code": code, "failure_stage": stage,
                       "attempted_forwards": attempted,
                       "completed_forwards": completed,
                       "requested_forwards": 9,
                       "elapsed_seconds": time.monotonic() - timer})
        exit_code = 2
    finally:
        remove_hooks(capture_handles)
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            _write_private_json(output_target, report)
        finally:
            _close_private_output(output_target)
    if exit_code == 0:
        print("synthetic score-path integration passed; no faithfulness result")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
