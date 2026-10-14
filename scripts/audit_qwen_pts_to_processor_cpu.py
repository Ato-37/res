"""Bind synthetic decoder PTS, exact pixels, Qwen video metadata, and RoPE IDs.

CPU-only integration audit: generates temporary lossless synthetic media, decodes
it with PyAV, and sends those exact PIL frames through the pinned Qwen processor.
No project videos, model weights, CUDA, QA, or annotations are used.

Usage::
    .venv/bin/python scripts/audit_qwen_pts_to_processor_cpu.py --output results/NEW.json
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile

import av
import numpy as np
from PIL import Image
import torch

from presentation_time_utils import (
    presentation_times_from_pts, uniform_sampling_rate_from_times,
)
from necessity_choice_margin_utils import uniform_sample_indices

REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PALETTE = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (160, 80, 20), (40, 90, 180))


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_digest(image: Image.Image) -> str:
    return digest_bytes(image.convert("RGB").tobytes())


def encode_fixture(path: Path, *, pts: tuple[int, ...], time_base: Fraction) -> None:
    """Encode unique-color frames losslessly with explicitly assigned PTS."""
    rate = Fraction(1, 1) / time_base
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv444p"
        stream.time_base = time_base
        if len(pts) > len(PALETTE):
            raise ValueError("synthetic fixture has more PTS than unique frame witnesses")
        for timestamp, color in zip(pts, PALETTE, strict=False):
            rgb = np.empty((64, 64, 3), dtype=np.uint8)
            rgb[:] = color
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = timestamp
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decode_fixture(path: Path):
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
    if not decoded or any(frame.pts is None for frame in decoded):
        raise ValueError("decoded frame has no PTS")
    time_bases = {Fraction(frame.time_base) for frame in decoded}
    if len(time_bases) != 1:
        raise ValueError("decoded frame time bases differ")
    pts = tuple(int(frame.pts) for frame in decoded)
    times = presentation_times_from_pts(pts, next(iter(time_bases)))
    images = [frame.to_image().convert("RGB") for frame in decoded]
    return pts, next(iter(time_bases)), times, images


def run_audit() -> dict:
    repo = Path(__file__).resolve().parents[1]
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION)
    if not model_path.is_dir():
        raise RuntimeError("pinned local processor cache missing; downloads are disabled")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    with tempfile.TemporaryDirectory(prefix="qwen-pts-fixture-") as temp_dir:
        root = Path(temp_dir)
        cfr_path = root / "synthetic_cfr.mkv"
        encode_fixture(cfr_path, pts=tuple(range(7)), time_base=Fraction(1, 2))
        pts, time_base, times, all_frames = decode_fixture(cfr_path)
        sample_indices = uniform_sample_indices(len(all_frames), 4)
        frames = [all_frames[index] for index in sample_indices]
        sampled_times = [times[index] for index in sample_indices]
        if len(all_frames) != 7 or sample_indices != [0, 2, 4, 6]:
            raise RuntimeError("CFR endpoint sampler changed requested source-frame selection")
        if sampled_times != [Fraction(0), Fraction(1), Fraction(2), Fraction(3)]:
            raise RuntimeError("selected CFR frame times changed unexpectedly")
        rate, deviation = uniform_sampling_rate_from_times(sampled_times, Fraction(0))
        if rate != 1 or deviation != 0:
            raise RuntimeError("sampled CFR fixture did not yield exact 1 FPS")
        frame_hashes = [image_digest(frame) for frame in frames]
        if len(set(frame_hashes)) != 4:
            raise RuntimeError("synthetic frame identities collapsed after decode")

        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": float(rate)},
            {"type": "text", "text": "Describe the video briefly."},
        ]}]
        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=28 * 28, max_pixels=28 * 28,
        )
        vision_config = model_config.vision_config
        tokens_per_second = int(vision_config.tokens_per_second)
        if int(vision_config.spatial_merge_size) != int(processor.video_processor.merge_size):
            raise RuntimeError("checkpoint/processor spatial merge sizes differ")
        if int(vision_config.temporal_patch_size) != int(processor.video_processor.temporal_patch_size):
            raise RuntimeError("checkpoint/processor temporal patch sizes differ")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or not videos or len(videos) != 1:
            raise RuntimeError("Qwen utility did not return exactly one video")
        video_tensor, metadata = videos[0]
        if not isinstance(video_tensor, torch.Tensor) or video_tensor.shape[0] != len(frames):
            raise RuntimeError("Qwen utility did not preserve the decoded frame count")
        # The list-frame utility resizes each image and stacks TCHW uint8 data.
        # Distinct constant-color frame witnesses survive that deterministic
        # resize and bind tensor order back to the decoded frame sequence.
        input_signatures = torch.stack([
            torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
            for frame in frames
        ])
        output_signatures = video_tensor.float().mean(dim=(-1, -2))
        if output_signatures.shape != input_signatures.shape or not torch.allclose(
                output_signatures, input_signatures, rtol=0, atol=1.0):
            raise RuntimeError("Qwen utility video tensor changed/reordered frame identity")
        if metadata.get("frames_indices") != [0, 1, 2, 3]:
            raise RuntimeError("Qwen metadata does not point to the decoded frame order")
        if float(metadata.get("fps", 0)) != float(rate):
            raise RuntimeError("Qwen FPS metadata differs from the exact CFR grid")
        if int(metadata.get("total_num_frames", 0)) != len(frames):
            raise RuntimeError("Qwen total frame count differs from decoded frame list")

        batch = processor(
            text=[text], images=images, videos=[video_tensor],
            video_metadata=[metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt",
        )
        if batch.get("video_grid_thw") is None or batch["video_grid_thw"].shape[0] != 1:
            raise RuntimeError("processor did not return one video grid")
        native, _ = Qwen2_5_VLModel.get_rope_index(
            SimpleModelConfig(model_config, Qwen2_5_VLModel), input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=batch["attention_mask"],
        )
        grid = batch["video_grid_thw"][0].tolist()
        merge = int(processor.video_processor.merge_size)
        video_positions = (batch["input_ids"][0] == processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")).nonzero(as_tuple=True)[0]
        if len(video_positions) == 0 or grid[0] != 2:
            raise RuntimeError("unexpected processor temporal grid/video token mapping")
        tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
        anchors = [int(native[0, 0, video_positions[0] + i * tokens_per_step]) for i in range(grid[0])]
        seconds_per_grid = float(batch["second_per_grid_ts"][0])
        expected = [anchors[0] + int(i * seconds_per_grid * tokens_per_second)
                    for i in range(grid[0])]
        if anchors != expected:
            raise RuntimeError("actual installed Qwen position IDs differ from fixed temporal rule")

        # VFR is deliberately rejected before giving the frame list to Qwen,
        # because its list-input interface exposes one FPS for the whole video.
        vfr_path = root / "synthetic_vfr.mkv"
        encode_fixture(vfr_path, pts=(0, 100, 300, 400), time_base=Fraction(1, 1000))
        vfr_pts, vfr_tb, vfr_times, _ = decode_fixture(vfr_path)
        try:
            uniform_sampling_rate_from_times(vfr_times, Fraction(1, 1000))
        except ValueError:
            vfr_rejected = True
        else:
            vfr_rejected = False
        if not vfr_rejected:
            raise RuntimeError("nonuniform VFR fixture unexpectedly passed uniform-FPS gate")

        irregular_path = root / "synthetic_cfr_irregular_sample.mkv"
        encode_fixture(irregular_path, pts=tuple(range(8)), time_base=Fraction(1, 2))
        irregular_pts, irregular_tb, irregular_times, irregular_frames = decode_fixture(irregular_path)
        irregular_indices = uniform_sample_indices(len(irregular_frames), 4)
        irregular_sampled_times = [irregular_times[index] for index in irregular_indices]
        try:
            uniform_sampling_rate_from_times(irregular_sampled_times, Fraction(0))
        except ValueError:
            irregular_sampling_rejected = True
        else:
            irregular_sampling_rejected = False
        if irregular_indices != [0, 2, 5, 7] or not irregular_sampling_rejected:
            raise RuntimeError("endpoint sampling's nonuniform CFR times were not rejected")

        model_file = inspect.getfile(Qwen2_5_VLModel)
        processor_file = inspect.getfile(type(processor))
        vision_utils_file = inspect.getfile(process_vision_info)
        processor_config_path = model_path / "preprocessor_config.json"
        if not processor_config_path.is_file():
            raise RuntimeError("pinned processor configuration file is missing")
        return {
            "schema_version": 1,
            "purpose": "synthetic_pts_pixels_qwen_processor_rope_integration",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "runner_sha256": digest_bytes(Path(__file__).read_bytes()),
            "presentation_time_utils_sha256": digest_bytes((repo / "scripts/presentation_time_utils.py").read_bytes()),
            "model_revision": REVISION,
            "model_config_sha256": digest_bytes((model_path / "config.json").read_bytes()),
            "processor_config_sha256": digest_bytes(processor_config_path.read_bytes()),
            "installed_model_source_sha256": digest_bytes(Path(model_file).read_bytes()),
            "installed_processor_source_sha256": digest_bytes(Path(processor_file).read_bytes()),
            "installed_vision_utils_source_sha256": digest_bytes(Path(vision_utils_file).read_bytes()),
            "versions": {
                "pyav": av.__version__,
                "av_linked_libraries": {name: list(version) for name, version in av.library_versions.items()},
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            },
            "cfr": {"requested_frames": 4, "decoded_source_frames": len(all_frames),
                    "decoded_frames": len(frames), "pts": list(pts),
                    "time_base": str(time_base), "presentation_seconds": [str(x) for x in times],
                    "selected_source_indices": sample_indices,
                    "selected_presentation_seconds": [str(x) for x in sampled_times],
                    "frame_sha256": frame_hashes, "qwen_metadata_indices": metadata["frames_indices"],
                    "qwen_metadata_fps": metadata["fps"], "qwen_video_grid_thw": grid,
                    "tokens_per_second": tokens_per_second,
                    "seconds_per_grid": seconds_per_grid, "native_temporal_anchors": anchors,
                    "temporal_anchor_reference": expected},
            "vfr_gate": {"pts": list(vfr_pts), "time_base": str(vfr_tb),
                         "presentation_seconds": [str(x) for x in vfr_times],
                         "uniform_fps_gate_rejected": vfr_rejected},
            "irregular_endpoint_sample_gate": {
                "decoded_source_frames": len(irregular_frames),
                "pts": list(irregular_pts), "time_base": str(irregular_tb),
                "selected_source_indices": irregular_indices,
                "selected_presentation_seconds": [str(x) for x in irregular_sampled_times],
                "uniform_fps_gate_rejected": irregular_sampling_rejected,
            },
            "limitations": ["Synthetic CFR/VFR only; no project media, model weights, CUDA, QA, or annotations.",
                            "PTS origin is the first decoded frame and is not calibrated to annotation-player time.",
                            "This verifies the local PyAV→Qwen processor fixture boundary, not production decoder equivalence.",
                            "No faithfulness, answer accuracy, or semantic evidence validity is measured."],
        }


class SimpleModelConfig:
    """Expose the actual pinned checkpoint config to get_rope_index."""
    def __init__(self, model_config, model_class):
        from types import MethodType
        self.config = model_config
        self.get_vision_position_ids = MethodType(model_class.get_vision_position_ids, self)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / "results") or output.exists():
        parser.error("new output under results/ required; overwrite refused")
    result = run_audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"CPU PTS→pixels→Qwen integration passed: CFR {result['cfr']['decoded_frames']}/4; VFR rejected; output {args.output}")

if __name__ == "__main__":
    main()
