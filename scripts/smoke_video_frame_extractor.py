"""Decode a tiny synthetic MP4 through the grounding frame extractor; no dataset video."""
import shutil
import tempfile
import math

import cv2
import numpy as np
from PIL import Image

from grounding_check import extract_frames
from video_time_mapping import effective_sample_fps


def make_synthetic_video_frames(frame_count, frame_size=224, source_fps=8.0):
    """Encode/decode a synthetic clip and return sampled PIL frames plus timestamps."""
    if frame_count < 2 or frame_count % 2:
        raise ValueError("frame_count must be even and at least two")
    with tempfile.TemporaryDirectory(prefix="qwen_video_smoke_") as work_dir:
        video_path = f"{work_dir}/synthetic.mp4"
        source_count = max(frame_count * 3, 24)
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
            (frame_size, frame_size),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the synthetic MP4")
        try:
            for index in range(source_count):
                value = index * 17 % 256
                frame = np.full((frame_size, frame_size, 3), value, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        # Read the encoded file's actual metadata so the duration check is
        # grounded in the produced MP4, not just the writer inputs.
        metadata_cap = cv2.VideoCapture(video_path)
        try:
            if not metadata_cap.isOpened():
                raise RuntimeError("OpenCV could not reopen the synthetic MP4")
            encoded_frame_count = int(metadata_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            encoded_fps = float(metadata_cap.get(cv2.CAP_PROP_FPS))
        finally:
            metadata_cap.release()
        if encoded_frame_count <= 0 or not math.isfinite(encoded_fps) or encoded_fps <= 0:
            raise RuntimeError(
                f"Invalid encoded-video metadata: frames={encoded_frame_count}, fps={encoded_fps}"
            )
        nominal_duration = encoded_frame_count / encoded_fps

        # extract_frames currently returns paths, its temporary directory,
        # sample timestamps, and effective sampled FPS.
        frame_paths, frame_dir, timestamps, sampled_fps = extract_frames(
            video_path, frame_count, frame_size
        )
        try:
            frames = []
            for path in frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("RGB"))
            if len(frames) != frame_count:
                raise RuntimeError(
                    f"Extractor returned {len(frames)} of {frame_count} requested frames"
                )
            effective_sample_fps(timestamps)
            if timestamps[-1] >= nominal_duration:
                raise RuntimeError("Synthetic sample timestamp exceeds nominal duration")
            return frames, timestamps, sampled_fps
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    frames, timestamps, sampled_fps = make_synthetic_video_frames(8)
    print(
        "Synthetic video decode smoke passed: "
        f"frames={len(frames)}, sample_timestamp_span={timestamps[-1] - timestamps[0]:.3f}s, "
        f"sampled_fps={sampled_fps:.6f}"
    )


if __name__ == "__main__":
    main()
