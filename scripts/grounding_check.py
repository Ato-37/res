"""Measure a coarse attention-overlap proxy for temporal grounding.

Video tokens get temporally merged inside Qwen2.5-VL's vision tower (pairs of
sampled frames typically collapse into one "temporal chunk" of tokens), so
single-frame attention isn't recoverable -- we work at the chunk level
instead and map each chunk back to the timestamps of the original frames it
was built from.

The chunk timing is a coarse preprocessing alignment. The reported attention-overlap
hit rate is only a grounding proxy; it is not a faithfulness or causal-dependence test.
Its chunk interval is the span from the first to last sampled timestamp in each temporal
patch, so that interval can overlap an annotation even when no sampled timestamp lies
inside it. The separate sampled-point count below makes that geometric distinction
visible; sampled points inside an annotation still do not establish semantic evidence.
The last prompt-position attention predicts the first generated token; it is not
direct evidence that the generated answer depends on the attended video interval.

Usage:
    python scripts/grounding_check.py --samples-file data/nextgqa_samples.json
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
from grounding_metrics import attention_overlap_proxy, sampled_point_window_coverage, validate_temporal_intervals
from video_time_mapping import chunk_timestamp_ranges, effective_sample_fps, uniform_frame_indices


def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count from {video_path}")
    if num_frames < 2:
        cap.release()
        raise ValueError("num_frames must be at least 2")

    try:
        indices = uniform_frame_indices(total, num_frames)
    except ValueError:
        cap.release()
        raise
    tmp_dir = tempfile.mkdtemp(prefix="grounding_frames_")
    frame_paths = []

    timestamps = []
    try:
        for pos, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {idx} from {video_path}")
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(timestamp):
                raise RuntimeError(f"Could not read timestamp for frame {idx}")
            timestamps.append(timestamp)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((max_dim, max_dim))
            path = f"{tmp_dir}/frame_{pos:03d}.jpg"
            img.save(path)
            frame_paths.append(path)
        sampled_fps = effective_sample_fps(timestamps)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        cap.release()

    return frame_paths, tmp_dir, timestamps, sampled_fps


def find_video_token_id(model, processor):
    tid = getattr(model.config, "video_token_id", None)
    if tid is not None:
        return tid
    for name in ["<|video_pad|>", "<|video_placeholder|>"]:
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("Could not find a video token id; check model.config or tokenizer special tokens")


def video_span(input_ids, video_token_id):
    positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("No video tokens found in input_ids")
    return int(positions[0]), int(positions[-1]) + 1  # [start, end)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--layers", default=None,
                         help="Layers averaged after forward; does not reduce attention memory (default: all)")
    args = parser.parse_args()
    layer_filter = [int(x) for x in args.layers.split(",")] if args.layers else None

    samples = json.load(open(args.samples_file))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])
        try:
            intervals = validate_temporal_intervals(s["grounding"])
            if not intervals or any(
                not (math.isfinite(a) and math.isfinite(b) and a < b)
                for a, b in intervals
            ):
                raise ValueError("grounding intervals must be finite and positive-duration")
            s["grounding"] = intervals
        except (KeyError, TypeError, ValueError) as error:
            s["_grounding_error"] = str(error)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, n_samples={len(samples)}")

    if device != "cuda":
        parser.error("Attention extraction requires CUDA; CPU fallback is intentionally disabled")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / (1024 ** 3)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / (1024 ** 3)
    if not torch.cuda.is_bf16_supported():
        parser.error("Qwen2.5-VL must run with BF16 support in this project")
    if args.quant != "4bit" and total_gib < 24:
        parser.error("16 GiB-class GPUs must use 4-bit quantization for attention outputs")
    if args.quant == "4bit" and free_gib < 12:
        parser.error("At least 12 GiB free VRAM is required before 4-bit attention inference")
    if args.num_frames > 8 and total_gib < 24:
        parser.error("For 16 GiB GPUs, --num-frames is capped at 8 by attention memory")
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant_config, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    video_token_id = find_video_token_id(model, processor)
    print(f"video_token_id={video_token_id}")

    hits = evaluated = skipped = 0
    for i, s in enumerate(samples):
        print(f"\n=== sample {i}: {s['question']!r} -> {s['answer']!r} "
              f"ground_truth={s.get('grounding', None)} ===")
        if "_grounding_error" in s:
            print(f"  WARNING: invalid ground-truth intervals ({s['_grounding_error']}), skipping")
            skipped += 1
            continue
        frame_paths, tmp_dir, timestamps, sampled_fps = extract_frames(s["video"], args.num_frames, args.frame_size)
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": frame_paths, "sample_fps": sampled_fps},
                {"type": "text", "text": s["question"]},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(
                messages, return_video_metadata=True
            )
            if not video_inputs or len(video_inputs) != 1:
                raise RuntimeError("Expected one video input with metadata")
            video_tensor, video_metadata = video_inputs[0]
            video_kwargs = {"video_metadata": [video_metadata], "do_sample_frames": False}
            inputs = processor(
                text=[text], images=image_inputs, videos=[video_tensor], padding=True,
                cap_pixels_per_frame=True, return_tensors="pt",
                **video_kwargs,
            ).to(device)

            v_start, v_end = video_span(inputs["input_ids"], video_token_id)
            span_len = v_end - v_start
            grid_thw = inputs.get("video_grid_thw")
            if grid_thw is None:
                print("  WARNING: missing video_grid_thw, skipping sample")
                skipped += 1
                continue
            t = int(grid_thw[0, 0])
            merge_size = int(processor.video_processor.merge_size)
            expected_video_tokens = int(grid_thw[0].prod().item()) // (merge_size ** 2)
            actual_video_tokens = int(
                (inputs["input_ids"][0] == video_token_id).sum().item()
            )
            if actual_video_tokens != expected_video_tokens or span_len != actual_video_tokens:
                print(f"  WARNING: token/grid mismatch (span={span_len}, "
                      f"tokens={actual_video_tokens}, expected={expected_video_tokens}), skipping")
                skipped += 1
                continue
            temporal_patch_size = int(processor.video_processor.temporal_patch_size)
            try:
                chunk_ranges = chunk_timestamp_ranges(timestamps, temporal_patch_size)
            except ValueError as error:
                print(f"  WARNING: invalid timestamp mapping ({error}), skipping sample")
                skipped += 1
                continue
            if len(chunk_ranges) != t or span_len % t != 0:
                print(f"  WARNING: token/time grid mismatch (tokens={span_len}, grid={t}, "
                      f"timestamp_chunks={len(chunk_ranges)}), skipping sample")
                skipped += 1
                continue
            chunk_size = span_len // t
            point_coverage = sampled_point_window_coverage(
                timestamps, temporal_patch_size, s["grounding"]
            )
            print(f"  frames_requested={args.num_frames}, frames_effective={len(timestamps)}, "
                  f"sampled_fps={sampled_fps:.6f}, temporal_patch_size={temporal_patch_size}")
            print("  sampled_point_window_geometry="
                  f"{point_coverage['points_inside']}/{len(timestamps)} sampled points inside "
                  f"annotated windows; maximum_points_inside_any_chunk="
                  f"{point_coverage['maximum_points_inside_any_chunk']}; "
                  f"any_point_inside={point_coverage['any_point_inside']} "
                  "(geometric only; not semantic evidence)")

            torch.cuda.reset_peak_memory_stats(0)
            with torch.no_grad():
                out = model(**inputs, output_attentions=True, use_cache=False, logits_to_keep=1)

            peak_gib = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            print(f"  gpu_peak_memory_allocated_gib={peak_gib:.3f}")
            # average attention from the last (answer-predicting) position onto
            # the video span, across every layer and head (or a chosen subset
            # of layers, to avoid attention-sink heads elsewhere drowning out
            # a real grounding signal -- see grounding_check baseline result)
            attn_layers = out.attentions if layer_filter is None else [out.attentions[i] for i in layer_filter]
            last_pos_attn = torch.stack([layer_attn[0, :, -1, v_start:v_end] for layer_attn in attn_layers])
            per_token = last_pos_attn.float().mean(dim=(0, 1))  # [span_len]
            per_chunk = per_token.view(t, chunk_size).sum(dim=1)  # [t]
            if not bool(torch.isfinite(per_chunk).all()) or float(per_chunk.sum()) <= 0:
                print("  WARNING: non-finite or zero attention mass, skipping")
                skipped += 1
                continue

            try:
                normalized, peak_chunk, hit = attention_overlap_proxy(
                    per_chunk.detach().cpu().tolist(), chunk_ranges, s["grounding"]
                )
            except ValueError as error:
                print(f"  WARNING: invalid attention/time/grounding values ({error}), skipping")
                skipped += 1
                continue
            per_chunk = torch.tensor(normalized, dtype=torch.float32)
            print(f"  video_span=({v_start},{v_end}) t_chunks={t} chunk_size={chunk_size}")
            peak_points_inside = point_coverage['per_chunk_points_inside'][peak_chunk]
            print(f"  attention_peak_chunk_sample_points_inside={peak_points_inside}; "
                  f"attention_peak_span_only_overlap={bool(hit and peak_points_inside == 0)}")
            for c in range(t):
                lo, hi = chunk_ranges[c]
                print(f"    chunk={c} time=[{lo:.1f},{hi:.1f}]s "
                      f"sample_points_in_window={point_coverage['per_chunk_points_inside'][c]} "
                      f"attention_share={per_chunk[c].item():.3f}")

            peak_lo, peak_hi = chunk_ranges[peak_chunk]
            hits += int(hit)
            evaluated += 1
            print(f"  peak_chunk={peak_chunk} time=[{peak_lo:.1f},{peak_hi:.1f}]s "
                  f"attention_overlap_proxy_hit={hit}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    rate = hits / evaluated if evaluated else float("nan")
    print(
        "\n=== attention-overlap proxy (peak temporal-chunk intersection; not faithfulness): "
        f"{hits}/{evaluated} = {rate:.2%}; skipped={skipped} ==="
    )


if __name__ == "__main__":
    main()
