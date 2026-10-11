"""Locate causal video-evidence pathways with layer-level activation patching.

For each sample, construct a natural temporal intervention by replacing the
sampled evidence frames with sampled non-evidence frames. Then compare:

* ``full``: original video;
* ``swapped``: evidence frames replaced by control frames;
* ``patched``: swapped input, but one layer's attention output is copied from
  the full-input forward pass.

If patching a layer rescues the answer sequence log-probability, that layer's
attention pathway is a candidate causal mediator. This first implementation
patches the complete attention output of a layer; it does not assume that one
head is independently causal.

Usage:
    python scripts/activation_patching_sweep.py \
        --samples-file data/nextgqa_samples_confirm_smoke.json \
        --n-samples 3 --layers 0,6,12,24,27 \
        --output results/activation_patching_smoke.json
"""

import argparse
import json
import math
import os
import shutil
import tempfile

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from necessity_mask_test import (
    choose_control_indices,
    decodable_video_info,
    grounding_frame_indices,
    sample_indices,
)
from soft_ablation_sweep import append_answer_tokens, sequence_answer_logprobs


def read_sampled_frames(video_path, total, indices, max_dim):
    wanted = set(indices)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video {video_path}")
    frames = {}
    try:
        for pos in range(total):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not sequentially read frame {pos} from {video_path}")
            if pos not in wanted:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((max_dim, max_dim))
            frames[pos] = image.copy()
            if len(frames) == len(indices):
                break
    finally:
        cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f"Decoded {len(frames)} of {len(indices)} requested frames")
    return frames


def write_frame_set(images, directory):
    os.makedirs(directory, exist_ok=True)
    paths = []
    for position, image in enumerate(images):
        path = os.path.join(directory, f"frame_{position:03d}.jpg")
        image.save(path)
        paths.append(path)
    return paths


def make_full_and_swapped_frames(frame_map, indices, evidence_indices, control_indices):
    full = [frame_map[index] for index in indices]
    swapped = list(full)
    evidence_positions = [indices.index(index) for index in evidence_indices]
    control_positions = [indices.index(index) for index in control_indices]
    for evidence_position, control_position in zip(evidence_positions, control_positions):
        swapped[evidence_position] = frame_map[indices[control_position]].copy()
    return full, swapped


def build_inputs(processor, device, frame_paths, question, cap_pixels_per_frame=None):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frame_paths},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    processor_kwargs = {
        "text": [text], "images": image_inputs, "videos": video_inputs,
        "padding": True, "return_tensors": "pt",
    }
    if cap_pixels_per_frame is not None:
        processor_kwargs["cap_pixels_per_frame"] = cap_pixels_per_frame
    return processor(**processor_kwargs).to(device)


def sequence_score(model, prompt_inputs, answer_ids):
    inputs = append_answer_tokens(prompt_inputs, answer_ids)
    with torch.no_grad():
        output = model(**inputs)
    logprobs = sequence_answer_logprobs(
        output, inputs["input_ids"].shape[1], answer_ids
    )
    return sum(logprobs[pos, token_id].item() for pos, token_id in enumerate(answer_ids))


def capture_attention_outputs(model, inputs, layers):
    captured = {}
    handles = []

    def make_hook(layer_index):
        def hook(module, module_inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_index] = value.detach().clone()
        return hook

    for layer_index in layers:
        handles.append(
            model.model.language_model.layers[layer_index].self_attn.register_forward_hook(
                make_hook(layer_index)
            )
        )
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def patch_bounds(scope, prompt_length, total_length):
    """Prompt includes the position predicting token 1; answer is teacher-forced."""
    if not 0 < prompt_length < total_length:
        raise ValueError("Need a nonempty prompt and answer")
    if scope == "all":
        return 0, total_length
    if scope == "prompt":
        return 0, prompt_length
    if scope == "answer":
        return prompt_length, total_length
    raise ValueError(f"Unknown scope {scope}")


def validate_alignment(full, swapped):
    # Pixels differ intentionally; all other processor outputs must match.
    keys = {key for key in full if not key.startswith("pixel_values")}
    if keys != {key for key in swapped if not key.startswith("pixel_values")}:
        raise ValueError("Processor metadata keys differ")
    for key in keys:
        a, b = full[key], swapped[key]
        equal = torch.equal(a, b) if torch.is_tensor(a) else a == b
        if not equal:
            raise ValueError(f"Full/swapped token alignment differs: {key}")


def patched_sequence_score(model, inputs, answer_ids, layer_index, source_output,
                           scope="all"):
    layer = model.model.language_model.layers[layer_index]
    calls = 0
    start, end = patch_bounds(scope, inputs["input_ids"].shape[1],
                              inputs["input_ids"].shape[1] + len(answer_ids))

    def patch_hook(module, module_inputs, output):
        nonlocal calls
        calls += 1
        value = output[0] if isinstance(output, (tuple, list)) else output
        if (value.shape != source_output.shape or value.dtype != source_output.dtype
                or value.device != source_output.device
                or value.shape[1] != inputs["input_ids"].shape[1] + len(answer_ids)):
            raise ValueError("Patch source/recipient shape, dtype or device mismatch")
        patched = value.clone()
        patched[:, start:end, :] = source_output[:, start:end, :]
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        if isinstance(output, list):
            return [patched] + output[1:]
        return patched

    handle = layer.self_attn.register_forward_hook(patch_hook)
    try:
        score = sequence_score(model, inputs, answer_ids)
        if calls != 1 or not math.isfinite(score):
            raise RuntimeError(f"Invalid patch forward: hook calls={calls}, score={score}")
        return score
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--layers", default="0,6,12,24,27")
    parser.add_argument("--patch-scope", choices=["all", "prompt", "answer"], default="all")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-atol", type=float, default=1e-4)
    parser.add_argument("--strict-grounding", action="store_true")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.self_check_atol) or args.self_check_atol < 0:
        parser.error("--self-check-atol must be finite and nonnegative")

    layers_to_run = [int(value) for value in args.layers.split(",")]
    samples = json.load(open(args.samples_file))
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    for sample in samples:
        sample["video"] = os.path.expanduser(sample["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant_config, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    layers = model.model.language_model.layers
    for layer_index in layers_to_run:
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"layer {layer_index} is outside 0..{len(layers) - 1}")

    records = []
    failures = []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("qid", sample.get("video_id", sample_index)))
        temp_dir = tempfile.mkdtemp(prefix="activation_patching_")
        try:
            total, fps = decodable_video_info(sample["video"])
            indices = sample_indices(total, args.num_frames)
            timestamps = [index / fps for index in indices]
            evidence_indices = grounding_frame_indices(
                indices, timestamps, sample["grounding"]
            )
            if args.strict_grounding:
                evidence_indices = [i for i, t in zip(indices, timestamps)
                                    if any(a <= t <= b for a, b in sample["grounding"])]
            control_indices = choose_control_indices(indices, evidence_indices)
            if not evidence_indices or not control_indices:
                raise RuntimeError(
                    f"Could not construct evidence/control swaps: "
                    f"evidence={evidence_indices}, control={control_indices}"
                )
            frame_map = read_sampled_frames(
                sample["video"], total, indices, args.frame_size
            )
            full_images, swapped_images = make_full_and_swapped_frames(
                frame_map, indices, evidence_indices, control_indices
            )
            full_paths = write_frame_set(full_images, os.path.join(temp_dir, "full"))
            swapped_paths = write_frame_set(swapped_images, os.path.join(temp_dir, "swapped"))
            full_inputs = build_inputs(processor, device, full_paths, sample["question"])
            swapped_inputs = build_inputs(processor, device, swapped_paths, sample["question"])
            validate_alignment(full_inputs, swapped_inputs)
            answer_ids = processor.tokenizer(
                sample["answer"], add_special_tokens=False
            ).input_ids
            if not answer_ids:
                raise ValueError("Empty answer tokens")
            full_sequence_inputs = append_answer_tokens(full_inputs, answer_ids)
            swapped_sequence_inputs = append_answer_tokens(swapped_inputs, answer_ids)
            full_outputs = capture_attention_outputs(
                model, full_sequence_inputs, layers_to_run
            )
            full_score = sequence_score(model, full_inputs, answer_ids)
            swapped_score = sequence_score(model, swapped_inputs, answer_ids)
            if not all(math.isfinite(s) for s in (full_score, swapped_score)):
                raise ValueError("Nonfinite baseline")
            self_checks = []
            if args.self_check:
                for name, prompt, sequence, baseline in (
                    ("full", full_inputs, full_sequence_inputs, full_score),
                    ("swapped", swapped_inputs, swapped_sequence_inputs, swapped_score),
                ):
                    # Capture one extra layer at a time to bound GPU memory.
                    for layer_index in layers_to_run:
                        source = (full_outputs[layer_index] if name == "full" else
                                  capture_attention_outputs(model, sequence, [layer_index])[layer_index])
                        checked = patched_sequence_score(
                            model, prompt, answer_ids, layer_index, source, args.patch_scope)
                        error = abs(checked - baseline)
                        self_checks.append({"input": name, "layer": layer_index,
                                            "absolute_error": error})
                        if error > args.self_check_atol:
                            raise RuntimeError(f"Self-patch failed: {name} layer={layer_index} "
                                               f"error={error} tolerance={args.self_check_atol}")
                        del source
            record = {
                "sample_id": sample_id,
                "grounding": sample["grounding"],
                "sampled_indices": indices,
                "sampled_timestamps": timestamps,
                "evidence_indices": evidence_indices,
                "control_indices": control_indices,
                "full_sequence_logprob": full_score,
                "swapped_sequence_logprob": swapped_score,
                "swap_drop": swapped_score - full_score,
                "layers": {},
                "self_checks": self_checks,
            }
            for layer_index in layers_to_run:
                patched_score = patched_sequence_score(
                    model, swapped_inputs, answer_ids, layer_index, full_outputs[layer_index],
                    args.patch_scope,
                )
                record["layers"][str(layer_index)] = {
                    "patched_sequence_logprob": patched_score,
                    "patch_rescue": patched_score - swapped_score,
                    "rescue_fraction": (
                        (patched_score - swapped_score) / (full_score - swapped_score)
                        if full_score != swapped_score else None
                    ),
                }
            records.append(record)
            if args.self_check:
                print(f"sample={sample_id} self_check=PASS checks={len(self_checks)}")
            print(f"sample={sample_id} full={full_score:.4f} swapped={swapped_score:.4f} "
                  f"swap_drop={record['swap_drop']:+.4f}")
            for layer_index in layers_to_run:
                print(f"  layer={layer_index} rescue="
                      f"{record['layers'][str(layer_index)]['patch_rescue']:+.4f}")
        except Exception as exc:
            failures.append({"id": sample_id, "video": sample.get("video"), "error": repr(exc)})
            print(f"WARNING: skipping sample {sample_id}: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    result = {
        "script": "activation_patching_sweep.py",
        "samples_file": args.samples_file,
        "model": args.model,
        "quant": args.quant,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "layers": layers_to_run,
        "patch_scope": args.patch_scope,
        "strict_grounding": args.strict_grounding,
        "self_check": args.self_check,
        "self_check_atol": args.self_check_atol,
        "num_frames": args.num_frames,
        "n_requested": len(samples),
        "n_effective": len(records),
        "failures": failures,
        "records": records,
        "warning": "Evidence/control frame swapping is an input-level causal probe; interpret with the swap validity and OOD limitations.",
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
