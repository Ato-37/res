"""Offline synthetic smoke for Qwen2.5-VL timestamp/grid alignment.

This loads only the cached processor (not model weights) and allocates no GPU model.
"""
import argparse

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from video_time_mapping import chunk_timestamp_ranges
from temporal_chunk_token_mapping import map_temporal_chunks_to_sequence_positions
from temporal_evidence_controls import select_temporal_evidence_controls
from smoke_video_frame_extractor import make_synthetic_video_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.frame_count < 2 or args.frame_count % 2:
        parser.error("--frame-count must be a positive even number")

    frames, timestamps, sampled_fps = make_synthetic_video_frames(args.frame_count)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": sampled_fps},
        {"type": "text", "text": "Describe the synthetic video."},
    ]}]

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(
        messages, return_video_metadata=True
    )
    if not video_inputs or len(video_inputs) != 1:
        raise RuntimeError("Expected exactly one video input with metadata")
    video_tensor, video_metadata = video_inputs[0]
    video_kwargs = {"video_metadata": [video_metadata], "do_sample_frames": False}
    batch = processor(
        text=[text],
        images=image_inputs,
        cap_pixels_per_frame=True,
        videos=[video_tensor],
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    grid = batch.get("video_grid_thw")
    if grid is None or len(grid) != 1:
        raise RuntimeError("Expected exactly one video_grid_thw entry")
    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    chunk_ranges = chunk_timestamp_ranges(timestamps, temporal_patch_size)
    temporal_grid_size = int(grid[0, 0])
    if temporal_grid_size != len(chunk_ranges):
        raise RuntimeError(
            f"Timestamp/grid mismatch: {len(chunk_ranges)} chunks vs {temporal_grid_size} grid steps"
        )
    if temporal_grid_size < 2:
        raise RuntimeError("Synthetic control smoke requires at least two temporal grid slices")

    second_per_grid_ts = batch.get("second_per_grid_ts")
    if second_per_grid_ts is None:
        raise RuntimeError("Processor did not return second_per_grid_ts")
    actual_seconds_per_grid = float(second_per_grid_ts[0])
    expected_seconds_per_grid = temporal_patch_size / sampled_fps
    if abs(actual_seconds_per_grid - expected_seconds_per_grid) > 1e-5:
        raise RuntimeError(
            f"FPS metadata mismatch: {actual_seconds_per_grid} vs {expected_seconds_per_grid}"
        )

    merge_size = int(processor.video_processor.merge_size)
    expected_video_tokens = int(grid[0].prod().item()) // (merge_size * merge_size)
    video_token_id = processor.video_token_id
    actual_video_tokens = int((batch["input_ids"] == video_token_id).sum().item())
    if actual_video_tokens != expected_video_tokens:
        raise RuntimeError(
            f"Video-token mismatch: {actual_video_tokens} vs {expected_video_tokens}"
        )
    video_token_positions = (
        batch["input_ids"][0] == video_token_id
    ).nonzero(as_tuple=True)[0].tolist()
    temporal_mapping = map_temporal_chunks_to_sequence_positions(
        video_token_positions,
        grid[0].tolist(),
        spatial_merge_size=merge_size,
        chunk_indices=range(len(chunk_ranges)),
    )
    if tuple(temporal_mapping) != tuple(range(temporal_grid_size)):
        raise RuntimeError("Mapped temporal slices do not match timestamp chunk order")
    flattened_positions = tuple(
        position for chunk in temporal_mapping.values() for position in chunk
    )
    if flattened_positions != tuple(video_token_positions):
        raise RuntimeError("Temporal slices do not partition the actual video-token block")
    tokens_per_slice = expected_video_tokens // temporal_grid_size
    if any(len(chunk) != tokens_per_slice for chunk in temporal_mapping.values()):
        raise RuntimeError("Mapped temporal slices have unequal processor token counts")

    # Fabricate one explicit interval from the middle processor chunk. This is
    # a mapping fixture only; it is not an annotation or a real data outcome.
    evidence_chunk_index = temporal_grid_size // 2
    synthetic_interval = tuple(chunk_ranges[evidence_chunk_index])
    selection = select_temporal_evidence_controls(
        [synthetic_interval],
        chunk_ranges,
        chunk_token_count=tokens_per_slice,
    )
    if not selection.eligible:
        raise RuntimeError(f"Synthetic evidence/control selection failed: {selection.reason}")
    if selection.evidence_indices != (evidence_chunk_index,):
        raise RuntimeError(
            f"Synthetic interval selected unexpected evidence chunks: {selection.evidence_indices}"
        )
    if (len(selection.control_indices) != 1
            or len(set(selection.control_indices)) != 1
            or selection.control_indices[0] == evidence_chunk_index):
        raise RuntimeError("Expected exactly one distinct non-evidence control chunk")
    if selection.matched_pairs != ((evidence_chunk_index, selection.control_indices[0]),):
        raise RuntimeError(f"One-to-one matched pair is inconsistent: {selection.matched_pairs}")

    evidence_positions = temporal_mapping[evidence_chunk_index]
    control_chunk_index = selection.control_indices[0]
    control_positions = temporal_mapping[control_chunk_index]
    if len(evidence_positions) != len(control_positions):
        raise RuntimeError("Evidence and control token counts do not match")
    if set(evidence_positions) & set(control_positions):
        raise RuntimeError("Evidence and control sequence positions overlap")
    video_position_set = set(video_token_positions)
    if (not set(evidence_positions).issubset(video_position_set)
            or not set(control_positions).issubset(video_position_set)):
        raise RuntimeError("Evidence/control sequence positions are outside the video-token block")

    print(
        "Qwen video-time smoke passed: "
        f"frames={args.frame_count}, sampled_fps={sampled_fps:.6f}, "
        f"temporal_patch_size={temporal_patch_size}, grid_thw={grid[0].tolist()}, "
        f"grid_steps={temporal_grid_size}, "
        f"seconds_per_grid={actual_seconds_per_grid:.3f}, video_tokens={actual_video_tokens}, "
        f"sequence_token_counts_per_grid_slice={[len(chunk) for chunk in temporal_mapping.values()]}, "
        f"synthetic_interval={synthetic_interval}, evidence_chunk={evidence_chunk_index}, "
        f"control_chunk={control_chunk_index}, matched_pairs={selection.matched_pairs}, "
        f"evidence/control_tokens={len(evidence_positions)}/{len(control_positions)}"
    )


if __name__ == "__main__":
    main()
