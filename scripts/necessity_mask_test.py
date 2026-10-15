"""Probe temporal necessity by masking grounded video frames.

For each sample, compare the answer sequence log-probability under:

* ``full``: the original sampled frames;
* ``evidence_mask``: sampled frames whose timestamps overlap NExT-GQA's
  grounding interval are replaced with black frames;
* ``control_mask``: the same number of frames is masked in a deterministic
  non-grounded temporal window.

This is a preliminary necessity test. Black-frame masking is an out-of-
distribution intervention, so evidence-vs-control differences must not be
reported as definitive causal proof without additional masking controls.

Usage:
    python scripts/necessity_mask_test.py \
        --samples-file data/nextgqa_samples_confirm_smoke.json \
        --n-samples 3 \
        --output results/necessity_mask_smoke.json
"""

import argparse
import json
import os
import shutil
import tempfile

import cv2
import torch
from PIL import Image

from necessity_choice_margin_utils import inspect_video_capture, uniform_sample_indices
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from soft_ablation_sweep import append_answer_tokens, sequence_answer_logprobs
from closed_research_branch import reject_inference


def decodable_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    return inspect_video_capture(cap, cv2.CAP_PROP_FPS)


def sample_indices(total, num_frames):
    return uniform_sample_indices(total, num_frames)


def overlaps(timestamp, intervals):
    return any(start <= timestamp <= end for start, end in intervals)


def grounding_frame_indices(indices, timestamps, intervals):
    selected = [idx for idx, timestamp in zip(indices, timestamps)
                if overlaps(timestamp, intervals)]
    if selected:
        return selected
    # With sparse temporal sampling, a short annotated interval may fall
    # between sampled timestamps. Use the nearest sampled frame so the sample
    # is not silently discarded solely because of temporal discretization.
    nearest = []
    for start, end in intervals:
        target = (start + end) / 2
        nearest.append(min(
            zip(indices, timestamps), key=lambda item: abs(item[1] - target)
        )[0])
    return sorted(set(nearest))


def choose_control_indices(sampled_indices, evidence_indices):
    count = len(evidence_indices)
    evidence_set = set(evidence_indices)
    if count == 0 or count >= len(sampled_indices):
        return []
    candidates = []
    for start in range(len(sampled_indices) - count + 1):
        window = set(sampled_indices[start:start + count])
        if not window & evidence_set:
            # Prefer a window near the temporal center to avoid always masking
            # the beginning or end when several controls are available.
            center = (sampled_indices[0] + sampled_indices[-1]) / 2
            window_center = sum(sampled_indices[start:start + count]) / count
            candidates.append((abs(window_center - center), start))
    if not candidates:
        return []
    _, start = min(candidates)
    return sampled_indices[start:start + count]


def extract_condition_frames(video_path, total, fps, indices, mask_indices, max_dim):
    tmp_dir = tempfile.mkdtemp(prefix="necessity_frames_")
    wanted = set(indices)
    masked = set(mask_indices)
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video {video_path}")
        for pos in range(total):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not sequentially read frame {pos} from {video_path}")
            if pos not in wanted:
                continue
            if pos in masked:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.new("RGB", (frame.shape[1], frame.shape[0]), (0, 0, 0))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            image.thumbnail((max_dim, max_dim))
            path = os.path.join(tmp_dir, f"frame_{len(frame_paths):03d}.jpg")
            image.save(path)
            frame_paths.append(path)
            if len(frame_paths) == len(indices):
                break
        if len(frame_paths) != len(indices):
            raise RuntimeError(
                f"Decoded {len(frame_paths)} of {len(indices)} requested frames from {video_path}"
            )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        cap.release()
    return frame_paths, tmp_dir


def sequence_metrics(model, processor, device, frame_paths, question, answer_ids):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frame_paths},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    inputs = append_answer_tokens(prompt_inputs, answer_ids)
    with torch.no_grad():
        output = model(**inputs)
    logprobs = sequence_answer_logprobs(
        output, inputs["input_ids"].shape[1], answer_ids
    )
    sequence_logprob = sum(
        logprobs[pos, token_id].item()
        for pos, token_id in enumerate(answer_ids)
    )
    first_token_logprob = logprobs[0, answer_ids[0]].item()
    return {
        "sequence_logprob": sequence_logprob,
        "first_token_logprob": first_token_logprob,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of uniformly sampled frames; 16 improves coverage of short intervals")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_frames < 2:
        parser.error("--num-frames must be >=2 to construct disjoint evidence/control masks")
    reject_inference(parser, "legacy_necessity_mask",
                     "the old nearest-frame QA-span scorer is superseded by the frozen timing contract")

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

    records = []
    failures = []
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("qid", sample.get("video_id", index)))
        try:
            total, fps = decodable_video_info(sample["video"])
            indices = sample_indices(total, args.num_frames)
            timestamps = [idx / fps for idx in indices]
            evidence_indices = grounding_frame_indices(
                indices, timestamps, sample["grounding"]
            )
            control_indices = choose_control_indices(indices, evidence_indices)
            if not evidence_indices or not control_indices:
                raise RuntimeError(
                    f"Could not construct evidence/control masks: "
                    f"evidence={evidence_indices}, control={control_indices}"
                )
            answer_ids = processor.tokenizer(
                sample["answer"], add_special_tokens=False
            ).input_ids
            conditions = {
                "full": [],
                "evidence_mask": evidence_indices,
                "control_mask": control_indices,
            }
            metrics = {}
            for condition, mask_indices in conditions.items():
                frame_paths, tmp_dir = extract_condition_frames(
                    sample["video"], total, fps, indices, mask_indices, args.frame_size
                )
                try:
                    metrics[condition] = sequence_metrics(
                        model, processor, device, frame_paths,
                        sample["question"], answer_ids
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            record = {
                "sample_id": sample_id,
                "video": sample["video"],
                "grounding": sample["grounding"],
                "sampled_indices": indices,
                "sampled_timestamps": timestamps,
                "evidence_mask_indices": evidence_indices,
                "control_mask_indices": control_indices,
                "metrics": metrics,
                "evidence_delta": metrics["evidence_mask"]["sequence_logprob"] - metrics["full"]["sequence_logprob"],
                "control_delta": metrics["control_mask"]["sequence_logprob"] - metrics["full"]["sequence_logprob"],
                "necessity_contrast": metrics["evidence_mask"]["sequence_logprob"] - metrics["control_mask"]["sequence_logprob"],
            }
            records.append(record)
            print(f"sample={sample_id} full={metrics['full']['sequence_logprob']:.4f} "
                  f"evidence={metrics['evidence_mask']['sequence_logprob']:.4f} "
                  f"control={metrics['control_mask']['sequence_logprob']:.4f}")
        except Exception as exc:
            failures.append({"id": sample_id, "video": sample.get("video"), "error": repr(exc)})
            print(f"WARNING: skipping sample {sample_id}: {exc}")

    output = {
        "script": "necessity_mask_test.py",
        "samples_file": args.samples_file,
        "model": args.model,
        "quant": args.quant,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "num_frames": args.num_frames,
        "n_requested": len(samples),
        "n_effective": len(records),
        "failures": failures,
        "records": records,
        "warning": "Black-frame masking is an out-of-distribution intervention; treat this as a preliminary necessity test.",
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(output, file, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
