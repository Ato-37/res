"""CPU-only reconstruction of legacy vs explicit Qwen video-time metadata.

Uses one fixed set of 16 deterministic JPEG frames at 2 FPS and the cached
Qwen2.5-VL processor/config, without model weights, CUDA, annotations, or QA.
The legacy branch calls the existing ``activation_patching_sweep.build_inputs``
unchanged. The comparison branch passes the same frames and prompt with the
explicit metadata and ``do_sample_frames=False`` returned by qwen-vl-utils.

This audits today's code and installed dependencies. It does not establish
which exact source/dependency versions or temporal coordinates were used by any
historical run, nor attribute previous outcomes to metadata behavior.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/audit_legacy_video_time_metadata.py \
      --output results/legacy_video_time_metadata_20261012_01.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import subprocess
from typing import Any

from activation_patching_sweep import build_inputs
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
FIXTURE_FPS = 2.0
FIXTURE_FRAME_COUNT = 16
PROMPT = "Describe the synthetic video briefly. Use only visual information."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_synthetic_jpegs(directory: Path) -> tuple[list[str], str]:
    """Write a deterministic 16-frame, 2 FPS JPEG fixture and return its hash."""
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    digest = hashlib.sha256()
    digest.update(b"legacy-video-time-metadata/synthetic-jpeg-fixture/v1\0")
    digest.update(f"{FIXTURE_FRAME_COUNT}:{FIXTURE_FPS}:224x224:RGB\0".encode("ascii"))
    for index in range(FIXTURE_FRAME_COUNT):
        base = (index * 29) % 256
        image = Image.new("RGB", (224, 224), (base, (base + 61) % 256, (base + 127) % 256))
        draw = ImageDraw.Draw(image)
        draw.rectangle((index * 7, 20, index * 7 + 35, 80), fill=(255 - base, base, 31))
        path = directory / f"frame_{index:03d}.jpg"
        image.save(path, format="JPEG", quality=95, subsampling=0, optimize=False)
        jpeg_bytes = path.read_bytes()
        digest.update(index.to_bytes(4, "big"))
        digest.update(len(jpeg_bytes).to_bytes(8, "big"))
        digest.update(jpeg_bytes)
        paths.append(str(path))
    return paths, digest.hexdigest()


def build_explicit_metadata_inputs(processor, frame_paths: list[str], prompt: str):
    """Build the corrected comparison with explicit video metadata, no sampling."""
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frame_paths, "sample_fps": FIXTURE_FPS},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_metadata=True, return_video_kwargs=True
    )
    if not video_inputs or len(video_inputs) != 1:
        raise RuntimeError("Expected exactly one explicit-metadata video")
    video_tensor, video_metadata = video_inputs[0]
    inputs = processor(
        text=[text], images=image_inputs, videos=[video_tensor],
        video_metadata=[video_metadata], padding=True, return_tensors="pt",
        cap_pixels_per_frame=True,
        **video_kwargs,
    )
    return inputs, video_tensor, video_metadata, video_kwargs


def _position_ids(inputs, config):
    class ModelGeometryMethods:
        get_rope_index = Qwen2_5_VLModel.get_rope_index
        get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

    model = ModelGeometryMethods()
    model.config = config
    return model.get_rope_index(
        input_ids=inputs["input_ids"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        video_grid_thw=inputs["video_grid_thw"],
        second_per_grid_ts=inputs["second_per_grid_ts"],
        attention_mask=inputs.get("attention_mask"),
    )[0]


def _video_summary(inputs, position_ids, processor, config) -> dict[str, Any]:
    grid = tuple(int(value) for value in inputs["video_grid_thw"][0].tolist())
    seconds = float(inputs["second_per_grid_ts"][0])
    video_token_id = getattr(config, "video_token_id", None)
    if video_token_id is None:
        video_token_id = processor.video_token_id
    positions = (inputs["input_ids"][0] == video_token_id).nonzero(as_tuple=True)[0].tolist()
    if not positions:
        raise RuntimeError("Processor emitted no video tokens")
    merge = int(processor.video_processor.merge_size)
    tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
    anchors = [
        int(position_ids[0, 0, positions[0] + step * tokens_per_step])
        for step in range(grid[0])
    ]
    return {
        "grid_thw": list(grid),
        "second_per_grid_seconds": seconds,
        "implied_sampled_fps": int(processor.video_processor.temporal_patch_size) / seconds,
        "video_token_count": len(positions),
        "position_id_shape": list(position_ids.shape),
        "temporal_anchors": anchors,
    }


def _pixel_tensors(inputs):
    return {
        key: value for key, value in inputs.items()
        if "pixel_values" in key and hasattr(value, "shape")
    }


def compare_metadata_paths(processor, config, frame_paths: list[str]) -> dict[str, Any]:
    """Compare exact legacy builder call to explicit-metadata processor inputs."""
    legacy = build_inputs(processor, "cpu", frame_paths, PROMPT, cap_pixels_per_frame=True)
    explicit, explicit_video_tensor, explicit_video_metadata, explicit_video_kwargs = (
        build_explicit_metadata_inputs(processor, frame_paths, PROMPT)
    )
    legacy_positions = _position_ids(legacy, config)
    explicit_positions = _position_ids(explicit, config)
    pixel_values_legacy = _pixel_tensors(legacy)
    pixel_values_explicit = _pixel_tensors(explicit)
    pixel_keys_equal = set(pixel_values_legacy) == set(pixel_values_explicit)
    pixel_values_equal = pixel_keys_equal and all(
        (pixel_values_legacy[key] == pixel_values_explicit[key]).all().item()
        for key in pixel_values_legacy
    )
    token_ids_equal = bool((legacy["input_ids"] == explicit["input_ids"]).all().item())
    grid_equal = bool((legacy["video_grid_thw"] == explicit["video_grid_thw"]).all().item())
    same_position_shape = tuple(legacy_positions.shape) == tuple(explicit_positions.shape)
    changed_entries = (
        int((legacy_positions != explicit_positions).sum().item())
        if same_position_shape else None
    )
    return {
        "legacy_build_inputs": _video_summary(legacy, legacy_positions, processor, config),
        "explicit_qwen_metadata": _video_summary(explicit, explicit_positions, processor, config),
        "identity_checks": {
            "input_token_ids_equal": token_ids_equal,
            "video_grid_thw_equal": grid_equal,
            "video_token_count_equal": (
                _video_summary(legacy, legacy_positions, processor, config)["video_token_count"]
                == _video_summary(explicit, explicit_positions, processor, config)["video_token_count"]
            ),
            "processor_pixel_value_keys_equal": pixel_keys_equal,
            "processor_pixel_values_equal": pixel_values_equal,
            "explicit_input_has_one_video_tensor": len(explicit_video_tensor.shape) == 4,
            "explicit_video_metadata_fps": float(explicit_video_metadata["fps"]),
            "explicit_video_metadata_frame_count": len(explicit_video_metadata["frames_indices"]),
            "explicit_do_sample_frames": explicit_video_kwargs.get("do_sample_frames"),
            "position_id_tensor_shapes_equal": same_position_shape,
        },
        "changed_position_id_entries": changed_entries,
    }


def run_audit(output: Path) -> dict[str, Any]:
    revision_root = (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
        / MODEL_REVISION
    )
    if not revision_root.is_dir():
        raise RuntimeError("Pinned cached model processor/config missing; downloads disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(
        str(revision_root), local_files_only=True,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    config = AutoConfig.from_pretrained(str(revision_root), local_files_only=True)
    with tempfile.TemporaryDirectory(prefix="legacy_video_metadata_audit_") as temp:
        frame_paths, fixture_sha256 = make_synthetic_jpegs(Path(temp) / "frames")
        comparison = compare_metadata_paths(processor, config, frame_paths)

    legacy = comparison["legacy_build_inputs"]
    explicit = comparison["explicit_qwen_metadata"]
    if legacy["grid_thw"] != [8, 32, 32] or abs(legacy["second_per_grid_seconds"] - (1 / 12)) > 1e-6:
        raise RuntimeError(f"Legacy path differs from pinned expected reconstruction: {legacy}")
    if explicit["grid_thw"] != legacy["grid_thw"] or abs(explicit["second_per_grid_seconds"] - 1.0) > 1e-6:
        raise RuntimeError(f"Explicit metadata path failed fixed 2 FPS expectation: {explicit}")
    identity = comparison["identity_checks"]
    if not all(identity[key] for key in (
        "input_token_ids_equal", "video_grid_thw_equal", "video_token_count_equal",
        "processor_pixel_value_keys_equal", "processor_pixel_values_equal",
        "position_id_tensor_shapes_equal",
    )):
        raise RuntimeError(f"Legacy and explicit paths differ beyond time metadata: {identity}")

    source_root = Path(__file__).resolve().parents[1]
    processor_config_path = revision_root / "preprocessor_config.json"
    model_config_path = revision_root / "config.json"
    vision_processor_config = processor.video_processor.to_dict()
    canonical_vision = json.dumps(
        vision_processor_config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    installed_model_file = Path(inspect.getfile(Qwen2_5_VLModel))
    installed_processor_file = Path(inspect.getfile(Qwen2_5_VLProcessor))
    qwen_utils_file = Path(inspect.getfile(process_vision_info))
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
        "purpose": "current_legacy_video_time_metadata_reconstruction",
        "status": "complete",
        "fixture": {
            "kind": "deterministic_synthetic_jpeg_sequence",
            "frame_count": FIXTURE_FRAME_COUNT,
            "sampled_fps": FIXTURE_FPS,
            "frame_timestamps_seconds": [index / FIXTURE_FPS for index in range(FIXTURE_FRAME_COUNT)],
            "sha256": fixture_sha256,
            "pixel_size": [224, 224],
            "prompt": PROMPT,
        },
        "processor_settings": {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "cap_pixels_per_frame": True,
            "model_weights_loaded": False,
            "cuda_used": False,
        },
        "legacy_default_metadata_provenance": (
            "Current-source inference, not historical-run provenance: build_inputs calls "
            "process_vision_info with default return_video_metadata=False and does not pass "
            "fps/video_metadata to the processor. Installed Transformers synthesizes frame "
            "indices but fps=None; VideoMetadata.sampled_fps therefore uses its documented "
            "24 FPS fallback. The measured legacy second_per_grid value is reported above."
        ),
        "comparison": comparison,
        "changed_position_id_entries": comparison["changed_position_id_entries"],
        "source_sha256": {
            "audit_script": sha256_file(Path(__file__)),
            "legacy_builder": sha256_file(source_root / "scripts" / "activation_patching_sweep.py"),
            "model_config": sha256_file(model_config_path),
            "processor_config": sha256_file(processor_config_path),
            "vision_processor_config": hashlib.sha256(canonical_vision).hexdigest(),
            "installed_rope_source": sha256_file(installed_model_file),
            "installed_processor_source": sha256_file(installed_processor_file),
            "installed_qwen_vl_utils_source": sha256_file(qwen_utils_file),
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "qwen-vl-utils", "Pillow")
        },
        "scope_limit": (
            "Reconstructs the current legacy builder and explicit-metadata paths on one "
            "synthetic 16-frame CPU fixture. Does not establish historical run coordinates, "
            "attribute prior results, use QA data, or claim temporal evidence validity."
        ),
        "privacy": "No raw frame files, local paths, QA text, labels, qids, model weights, or logits are emitted.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
        help="New result path under results/; existing files are never overwritten")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a short summary")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / "results") or output.exists():
        parser.error("Choose a new output path under results/; overwrite refused")
    result = run_audit(output)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        old = result["comparison"]["legacy_build_inputs"]
        explicit = result["comparison"]["explicit_qwen_metadata"]
        print(
            "Legacy metadata reconstruction passed: "
            f"legacy grid={old['grid_thw']} dt={old['second_per_grid_seconds']:.6f}s "
            f"anchors={old['temporal_anchors']}; explicit dt={explicit['second_per_grid_seconds']:.6f}s "
            f"anchors={explicit['temporal_anchors']}; "
            f"changed_position_entries={result['changed_position_id_entries']}; "
            f"output={args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
