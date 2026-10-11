"""Offline synthetic smoke for Qwen2.5-VL timestamp/grid alignment.

This loads only the cached processor (not model weights) and allocates no GPU model.
"""
import argparse

from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from video_time_mapping import chunk_timestamp_ranges


def partition_video_tokens_by_temporal_grid(video_token_positions, grid_thw, spatial_merge_size):
    """Partition one contiguous Qwen video-token block in temporal-major grid order.

    Qwen's processor emits one merged image-token position per spatial merge cell
    at each temporal grid step. This is only a processor-layout check; it does
    not identify evidence, controls, or semantically meaningful intervals.
    """
    grid = tuple(int(value) for value in grid_thw)
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError(f"Expected positive [T,H,W] video grid, got {grid}")
    temporal_size, height, width = grid
    merge_size = int(spatial_merge_size)
    if merge_size <= 0 or height % merge_size or width % merge_size:
        raise ValueError(
            f"Grid spatial dimensions {height}x{width} are not divisible by merge size {merge_size}"
        )

    tokens_per_slice = (height // merge_size) * (width // merge_size)
    positions = tuple(int(position) for position in video_token_positions)
    expected_count = temporal_size * tokens_per_slice
    if len(positions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} video-token positions from grid {grid}, got {len(positions)}"
        )
    if not positions or positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ValueError("Video-token positions must be one contiguous sequence block")

    return {
        temporal_index: positions[
            temporal_index * tokens_per_slice:(temporal_index + 1) * tokens_per_slice
        ]
        for temporal_index in range(temporal_size)
    }


def make_synthetic_video_frames(frame_count, frame_size=224, sampled_fps=2.0):
    """Create in-memory synthetic frames with a uniform, explicit time axis."""
    frames = []
    for index in range(frame_count):
        value = index * 17 % 256
        frames.append(Image.new("RGB", (frame_size, frame_size), (value, value, value)))
    timestamps = [index / sampled_fps for index in range(frame_count)]
    return frames, timestamps, sampled_fps


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
        raise RuntimeError("Temporal partition smoke requires at least two grid slices")

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
    temporal_mapping = partition_video_tokens_by_temporal_grid(
        video_token_positions,
        grid[0].tolist(),
        spatial_merge_size=merge_size,
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

    print(
        "Qwen video-time smoke passed: "
        f"frames={args.frame_count}, sampled_fps={sampled_fps:.6f}, "
        f"temporal_patch_size={temporal_patch_size}, grid_thw={grid[0].tolist()}, "
        f"grid_steps={temporal_grid_size}, "
        f"seconds_per_grid={actual_seconds_per_grid:.3f}, video_tokens={actual_video_tokens}, "
        f"timestamp_ranges={chunk_ranges}, "
        f"sequence_token_counts_per_grid_slice={[len(chunk) for chunk in temporal_mapping.values()]}"
    )


if __name__ == "__main__":
    main()
