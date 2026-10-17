#!/usr/bin/env python3
"""Local-only real-video Qwen smoke for selected visual-key attention bias.

Uses three deterministic, manifest-unseen NExT-GQA test videos with one
deterministically selected question per video and 32 PTS-sampled frames. It runs
baseline attention capture
and the fixed top-10%/ln(2) attention-bias treatment for each video (six CUDA
forwards total), stores only aggregate logit-change diagnostics, and never
uses QA answers, gold intervals, generated text, or claims. This is a real
video intervention-wiring check, not an explanation-faithfulness result.
Protected media and all derived measurements must remain local.
"""
from __future__ import annotations

import argparse
import csv
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

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels  # noqa: E402
from qwen_visual_attention_boost import (  # noqa: E402
    install_video_key_bias_hooks,
    remove_hooks,
)
from run_nextgqa_real_video_gpu_smoke import choose_videos  # noqa: E402
from smoke_qwen_stear_attention_feasibility import (  # noqa: E402
    select_top_visual_positions,
)


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 32
UNIFORM_TOLERANCE_SECONDS = Fraction(50, 1000)
RECORD_COUNT = 3
LAYER_FIRST = 5
LAYER_LAST = 16
LAYER_INDICES = tuple(range(LAYER_FIRST - 1, LAYER_LAST))
VISUAL_RATIO = 0.10
LOGIT_BIAS = math.log(2.0)
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2
PROMPT_TEMPLATE = "Answer using visible video evidence: {question}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_attention(output, layer_index, rows, torch_module):
    if (not isinstance(output, tuple) or len(output) < 2
            or not torch_module.is_tensor(output[1]) or output[1].ndim != 4
            or int(output[1].shape[0]) != 1):
        raise RuntimeError("attention_matrix_unavailable")
    rows[layer_index] = output[1][0, :, -1, :].float().mean(dim=0).detach().cpu()


def question_for_video(dataset: Path, video_path: Path, video_root: Path) -> str:
    """Return first qid-sorted eligible question; never read answer columns."""
    from run_nextgqa_real_video_gpu_smoke import video_key

    target_key = video_key(video_path.relative_to(video_root).as_posix())
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    rows = []
    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            video_id, qid = str(row.get("video_id", "")), str(row.get("qid", ""))
            info = gsub.get(video_id)
            raw = (info.get("location", {}).get(qid)
                   if isinstance(info, dict) and isinstance(info.get("location"), dict)
                   else None)
            question = row.get("question")
            if (video_id not in video_map or not isinstance(info, dict)
                    or video_key(video_map[video_id]) != target_key
                    or not isinstance(raw, list) or not raw
                    or not isinstance(question, str) or not question.strip()):
                continue
            try:
                duration = float(info["duration"])
                spans = [(float(span[0]), float(span[1])) for span in raw]
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
            if all(0 <= start <= end <= duration for start, end in spans):
                rows.append((qid, question.strip()))
    if not rows:
        raise ValueError("eligible_question_unavailable")
    return sorted(rows, key=lambda item: item[0])[0][1]


def _make_batch(*, frames, sample_fps, question, processor, torch_module):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "sample_fps": sample_fps.numerator / sample_fps.denominator},
        {"type": "text", "text": PROMPT_TEMPLATE.format(question=question)},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise RuntimeError("video_preprocess_contract_invalid")
    pixels, metadata = videos[0]
    if (not torch_module.is_tensor(pixels) or int(pixels.shape[0]) != FRAME_COUNT
            or metadata.get("frames_indices") != list(range(FRAME_COUNT))
            or abs(float(metadata.get("fps", 0)) - float(sample_fps)) > 1e-7):
        raise RuntimeError("video_processor_frame_contract_invalid")
    batch = processor(text=[prompt], images=images, videos=[pixels],
                      video_metadata=[metadata], do_sample_frames=False,
                      cap_pixels_per_frame=True, padding=True,
                      return_tensors="pt").to("cuda")
    return batch, metadata


def _run_baseline_and_selector(*, model, batch, processor, torch_module):
    rows = {}
    handles = []
    try:
        for layer_index in LAYER_INDICES:
            def capture(_module, _inputs, output, *, selected_layer=layer_index):
                _capture_attention(output, selected_layer, rows, torch_module)
            handles.append(model.model.language_model.layers[layer_index].self_attn
                           .register_forward_hook(capture))
        with torch_module.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        remove_hooks(handles)
    if tuple(sorted(rows)) != LAYER_INDICES:
        raise RuntimeError("baseline_attention_capture_incomplete")
    input_ids = batch.get("input_ids")
    if (not torch_module.is_tensor(input_ids) or input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1):
        raise RuntimeError("input_ids_shape_invalid")
    sequence_length = int(input_ids.shape[1])
    query_position = sequence_length - 1
    visual_positions = tuple(
        index for index, token in enumerate(input_ids[0].tolist())
        if int(token) == int(processor.video_token_id))
    if not visual_positions or visual_positions[-1] >= query_position:
        raise RuntimeError("video_token_positions_invalid")
    compact = [None] * LAYER_LAST
    for layer_index, row in rows.items():
        compact[layer_index] = row.view(1, 1, 1, sequence_length)
    selected, scores = select_top_visual_positions(
        compact, visual_positions, layer_indices=LAYER_INDICES,
        ratio=VISUAL_RATIO)
    if (len(selected) != max(1, math.ceil(len(visual_positions) * VISUAL_RATIO))
            or not all(math.isfinite(float(value)) for value in scores)):
        raise RuntimeError("frozen_visual_selector_invalid")
    return output, selected, len(visual_positions), query_position


def _write_private_json(path: Path, value: dict) -> None:
    if not path.parent.exists():
        # Create only the final result directory. Requiring its parent to
        # exist avoids broad recursive permission changes in user directories.
        path.parent.mkdir(mode=0o700)
    if not path.parent.is_dir() or path.parent.stat().st_mode & 0o077:
        raise RuntimeError("private_output_directory_permissions_invalid")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new private local JSON path; existing outputs are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a unique local path")

    import torch
    import transformers
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    stage = "initialize"
    model = None
    all_hooks = []
    attempted = completed = 0
    records = []
    report = {"status": "incomplete", "scope": "local_real_video_hook_smoke_not_faithfulness"}
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        free_before, total = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")
        repo = Path(__file__).resolve().parents[1]
        dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
        candidates = choose_videos(repo, dataset)
        video_root = Path.home() / "nextgqa_sample" / "NExTVideo"
        valid = []
        inspected = rejected = 0
        for candidate in candidates[:100]:
            inspected += 1
            try:
                frames, metadata = decode_sampled_video(
                    candidate, num_frames=FRAME_COUNT,
                    uniform_tolerance_seconds=UNIFORM_TOLERANCE_SECONDS)
                verify_sampled_frame_pixels(frames, metadata)
            except ValueError:
                rejected += 1
                continue
            try:
                question = question_for_video(dataset, candidate, video_root)
            except ValueError:
                raise RuntimeError("eligible_question_unavailable") from None
            valid.append((candidate, frames, metadata, question))
            if len(valid) == RECORD_COUNT:
                break
        if len(valid) != RECORD_COUNT:
            raise RuntimeError("three_uniform_pts_test_videos_unavailable")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_not_cached")
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        torch.manual_seed(20261018)
        torch.cuda.manual_seed_all(20261018)
        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < LAYER_LAST):
            raise RuntimeError("pinned_model_processor_contract_invalid")
        torch.cuda.reset_peak_memory_stats(device)
        stage = "six_real_video_forwards"
        for _position, (_video_path, frames, metadata, question) in enumerate(valid, start=1):
            fps = Fraction(*metadata["effective_sample_fps"])
            batch, video_metadata = _make_batch(
                frames=frames, sample_fps=fps, question=question, processor=processor,
                torch_module=torch)
            if int(video_metadata.get("frames_indices", [-1])[-1]) != FRAME_COUNT - 1:
                raise RuntimeError("processor_video_frame_order_invalid")
            free, _ = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("gpu_resource_floor_before_baseline")
            attempted += 1
            baseline, selected, visual_count, query_position = _run_baseline_and_selector(
                model=model, batch=batch, processor=processor, torch_module=torch)
            base_logits = baseline.logits[0, -1].float()
            if not bool(torch.isfinite(base_logits).all()):
                raise RuntimeError("baseline_logits_nonfinite")
            completed += 1
            free, _ = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("gpu_resource_floor_before_treatment")
            hooks = install_video_key_bias_hooks(
                model=model, layer_indices=LAYER_INDICES,
                selected_positions=selected, query_position=query_position,
                logit_bias=LOGIT_BIAS, torch_module=torch)
            all_hooks.extend(hooks)
            try:
                attempted += 1
                with torch.inference_mode():
                    treated = model(**batch, use_cache=False, logits_to_keep=1)
            finally:
                remove_hooks(hooks)
                for handle in hooks:
                    if handle in all_hooks:
                        all_hooks.remove(handle)
            torch.cuda.synchronize(device)
            treated_logits = treated.logits[0, -1].float()
            if not bool(torch.isfinite(treated_logits).all()):
                raise RuntimeError("treatment_logits_nonfinite")
            completed += 1
            delta = treated_logits - base_logits
            if not bool(torch.isfinite(delta).all()):
                raise RuntimeError("treatment_logit_delta_nonfinite")
            records.append({
                "frames": FRAME_COUNT,
                "video_token_count": visual_count,
                "selected_key_count": len(selected),
                "sequence_length": int(batch["input_ids"].shape[1]),
                "query_position": query_position,
                "max_abs_logit_change": float(delta.abs().max().item()),
                "rms_logit_change": float(delta.square().mean().sqrt().item()),
                "top_vocabulary_token_changed": (
                    int(base_logits.argmax()) != int(treated_logits.argmax())),
                "sampled_pts_sha256": metadata["decoded_pts_sequence_sha256"],
                "sampled_rgb_sha256": metadata["pass1_decoded_rgb_sequence_sha256"],
            })
            del baseline, treated, base_logits, treated_logits, delta, batch
            torch.cuda.empty_cache()
        if attempted != 6 or completed != 6 or len(records) != RECORD_COUNT:
            raise RuntimeError("real_video_six_forward_completion_invariant_failed")
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        rms_values = [row["rms_logit_change"] for row in records]
        report = {
            "status": "passed",
            "scope": "local_real_video_attention_bias_wiring_only",
            "started_at_utc": started,
            "runner_sha256": _sha256(Path(__file__)),
            "intervention_helper_sha256": _sha256(
                Path(__file__).with_name("qwen_visual_attention_boost.py")),
            "dataset_split": "NExT-GQA test",
            "candidate_eligibility": "valid local media and nonempty annotation geometry; question chosen by qid order",
            "prompt_type": "question-conditioned visual prompt; question text not persisted",
            "candidate_selection": "valid annotated local test videos absent from prior sample manifests; deterministic hash order",
            "candidate_videos_inspected_for_pts": inspected,
            "candidate_timing_rejections": rejected,
            "requested_videos": RECORD_COUNT,
            "effective_videos": len(records),
            "requested_frames_per_video": FRAME_COUNT,
            "uniform_grid_tolerance_seconds": [
                UNIFORM_TOLERANCE_SECONDS.numerator,
                UNIFORM_TOLERANCE_SECONDS.denominator],
            "effective_frames_per_video": [row["frames"] for row in records],
            "requested_forwards": 6,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "model_revision": MODEL_REVISION,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "precision": "bf16_4bit_bf16_compute",
            "intervention_layers_1_based": [LAYER_FIRST, LAYER_LAST],
            "selector": "baseline_full_mean_final_query_attention_top_10_percent_video_keys_frozen_per_video",
            "key_attention_logit_bias": LOGIT_BIAS,
            "video_token_counts": [row["video_token_count"] for row in records],
            "selected_key_counts": [row["selected_key_count"] for row in records],
            "sequence_lengths": [row["sequence_length"] for row in records],
            "max_abs_logit_changes": [row["max_abs_logit_change"] for row in records],
            "rms_logit_changes": rms_values,
            "top_vocabulary_token_changed_count": sum(
                row["top_vocabulary_token_changed"] for row in records),
            "mean_rms_logit_change": sum(rms_values) / len(rms_values),
            "per_video_decoder_hashes": [
                {"pts": row["sampled_pts_sha256"], "rgb": row["sampled_rgb_sha256"]}
                for row in records],
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "faithfulness_result": False,
            "qa_question_text_used": True,
            "qa_answer_text_used": False,
            "answer_or_generated_text_saved": False,
            "gold_interval_geometry_used_for_candidate_eligibility": True,
            "gold_interval_used_for_model_or_score": False,
            "local_execution_authorization": "user_authorized_local_engineering",
            "scoring_authorization": "not_granted",
            "rights_status": "unverified_local_only",
        }
        status = 0
    except Exception as exc:
        remove_hooks(all_hooks)
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"local_video_smoke_failed_{stage}_{type(exc).__name__}"
        report.update({
            "failure_code": code,
            "failure_stage": stage,
            "started_at_utc": started,
            "requested_videos": RECORD_COUNT,
            "effective_videos": len(records),
            "requested_forwards": 6,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "faithfulness_result": False,
            "local_execution_authorization": "user_authorized_local_engineering",
            "scoring_authorization": "not_granted",
            "rights_status": "unverified_local_only",
            "elapsed_seconds": time.monotonic() - timer,
        })
        # Never expose partial per-video measurements as if this fixed cohort
        # completed atomically.
        records = []
        status = 2
    finally:
        remove_hooks(all_hooks)
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        report.setdefault("local_execution_authorization",
                           "user_authorized_local_engineering")
        report.setdefault("scoring_authorization", "not_granted")
        report.setdefault("faithfulness_result", False)
        report.setdefault("rights_status", "unverified_local_only")
        report.setdefault("elapsed_seconds", time.monotonic() - timer)
        report["records"] = records
        _write_private_json(args.output, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
