#!/usr/bin/env python3
"""CPU-only, aggregate-only processor preflight for oracle-claim v1."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from qta_synthetic_oracle_claim_v1 import (
    FRAME_COUNT, GENERATOR_VERSION, assignment_manifest, build_assignment,
    build_clip, claims_for, control_temporal_groups, event_temporal_groups,
    map_exact_claim_positions, render_claim_prompt, sham_temporal_groups,
)


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SNAPSHOT_RELATIVE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct" / "snapshots" / MODEL_REVISION
PRIVATE_RELATIVE = Path("data/private_nextgqa")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _smoke_design():
    import random
    from qta_synthetic_oracle_claim_v1 import SceneDesign, _seed_population, BACKGROUNDS
    seed = _seed_population()[0]
    rng = random.Random(seed)
    return SceneDesign(
        family="gate", onset=18, replicate=-1, scene_seed=seed,
        target_object="star", label_order="supported_A",
        gate_x=rng.choice((112, 128, 144)),
        track_offset_y=rng.choice((-8, 0, 8)),
        base_radius=rng.choice((24, 28, 32)),
        background_rgb=rng.choice(BACKGROUNDS), star_upper=bool(rng.randrange(2)),
        star_uses_color0=bool(rng.randrange(2)))


def _answer_token_ids(tokenizer, rendered: str) -> tuple[int, int]:
    base = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(base, list) or not base:
        raise ValueError("answer_prompt_tokenization_invalid")
    result = []
    for label in ("A", "B"):
        extended = tokenizer.encode(rendered + label, add_special_tokens=False)
        if not isinstance(extended, list) or extended[:len(base)] != base:
            raise ValueError("answer_token_prompt_prefix_retokenized")
        suffix = extended[len(base):]
        if len(suffix) != 1 or type(suffix[0]) is not int or suffix[0] < 0:
            raise ValueError("answer_label_not_single_token")
        result.append(suffix[0])
    if result[0] == result[1]:
        raise ValueError("answer_label_token_collision")
    return result[0], result[1]


def _process(processor, process_vision_info, rendered, messages):
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("processor_video_input_contract_invalid")
    pixels, metadata = videos[0]
    if len(pixels) != FRAME_COUNT:
        raise ValueError("processor_input_frame_count_invalid")
    batch = processor(text=[rendered], images=images, videos=[pixels],
                      video_metadata=[metadata], do_sample_frames=False,
                      cap_pixels_per_frame=True, padding=True, return_tensors="pt")
    if batch["input_ids"].device.type != "cpu":
        raise ValueError("cpu_preflight_tensor_device_invalid")
    return batch, images, videos, pixels, metadata


def run_preflight(*, mode: str, output: Path,
                  repo_root: Path | None = None) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("preflight_mode_invalid")
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    output = Path(os.path.abspath(output))
    private = root / PRIVATE_RELATIVE
    if output.parent != private or output.exists():
        raise ValueError("preflight_output_must_be_new_private_child")
    snapshot = SNAPSHOT_RELATIVE
    if not snapshot.is_dir():
        raise RuntimeError("pinned_processor_snapshot_unavailable")

    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_processor
    import qwen_vl_utils
    from qta_synthetic_oracle_claim_v1 import (
        _seed_population, sha256_canonical,
    )

    source_paths = (
        Path("experiments/2026-10-21_synthetic_claim_evidence_dependence_protocol_v1.md"),
        Path("scripts/qta_synthetic_oracle_claim_v1.py"),
        Path("scripts/preflight_qta_synthetic_oracle_claim_v1.py"),
        Path("scripts/qta_synthetic_event_construct_v1.py"),
        Path("scripts/run_nextgqa_gold_interval_margin_pilot.py"),
    )
    source_hashes = {path.as_posix(): _file_sha256(root / path)
                     for path in source_paths}
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    worktree_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True).strip())
    assignment = assignment_manifest()
    full_assignment = build_assignment()
    if mode == "smoke":
        # Balance is checked against the frozen full cohort even during smoke.
        balance_rows = full_assignment
    else:
        balance_rows = designs = full_assignment
    cell_counts = Counter((row.family, row.onset) for row in balance_rows)
    assignment_cells = Counter((row.family, row.onset, row.target_object,
                                row.label_order) for row in balance_rows)
    if (len(balance_rows) != 48 or len(cell_counts) != 12
            or any(count != 4 for count in cell_counts.values())
            or len(assignment_cells) != 48
            or any(count != 1 for count in assignment_cells.values())):
        raise ValueError("frozen_assignment_balance_invalid")
    if mode == "smoke":
        designs = (_smoke_design(),)
    processor = AutoProcessor.from_pretrained(
        str(snapshot), revision=MODEL_REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        trust_remote_code=False,
    )
    if not getattr(processor.tokenizer, "is_fast", False):
        raise ValueError("fast_tokenizer_required_for_claim_offsets")
    processor_descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_sha256": _sha256(
            processor.tokenizer.backend_tokenizer.to_str().encode("utf-8")),
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    processor_sha = _sha256(_canonical(processor_descriptor))
    implementation_sources = {
        "transformers_qwen2_vl_video_processor": Path(qwen_video_processor.__file__).resolve(),
        "qwen_vl_utils_package": Path(qwen_vl_utils.__file__).resolve(),
    }
    implementation_fingerprints = {
        name: {"basename": path.name, "sha256": _file_sha256(path)}
        for name, path in implementation_sources.items()
    }
    processor_hash = hashlib.sha256()
    frame_hash = hashlib.sha256(b"qta-oracle-claim-frame-cohort-v1\0")
    prompt_hash = hashlib.sha256(b"qta-oracle-claim-prompt-cohort-v1\0")
    geometry_counts: Counter[tuple[int, int, int]] = Counter()
    sequence_counts: Counter[int] = Counter()
    claim_token_counts: Counter[int] = Counter()
    event_overlap_counts: Counter[int] = Counter()
    control_overlap_counts: Counter[int] = Counter()
    successful = failed = prompt_count = 0
    merge = int(processor.image_processor.merge_size)
    for design in designs:
        clip = build_clip(design)
        frame_digest = hashlib.sha256(b"scene-frames-v1\0")
        for index, image in enumerate(clip.frames):
            frame_digest.update(index.to_bytes(4, "big"))
            frame_digest.update(image.tobytes())
        true_claim, false_claim = claims_for(design)
        claim_kinds = (("true", true_claim), ("false", false_claim))
        for claim_kind, claim in claim_kinds:
            prompt, supported_label = render_claim_prompt(claim, design.label_order)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": list(clip.frames), "sample_fps": 1.0},
                {"type": "text", "text": prompt},
            ]}]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch, images, videos, pixels, metadata = _process(
                processor, process_vision_info, rendered, messages)
            ids = batch.get("input_ids")
            grid = batch.get("video_grid_thw")
            mask = batch.get("attention_mask")
            if (not torch.is_tensor(ids) or ids.ndim != 2 or tuple(ids.shape[:1]) != (1,)
                    or not torch.is_tensor(grid) or tuple(grid.shape) != (1, 3)
                    or not torch.is_tensor(mask) or mask.shape != ids.shape
                    or not bool((mask == 1).all())):
                raise ValueError("processor_tensor_contract_invalid")
            positions = map_exact_claim_positions(
                tokenizer=processor.tokenizer, rendered_text=rendered,
                claim=claim, processed_input_ids=ids, video_token_id=int(processor.video_token_id))
            a_id, b_id = _answer_token_ids(processor.tokenizer, rendered)
            label_id = a_id if supported_label == "A" else b_id
            for label, expected_id in (("A", a_id), ("B", b_id)):
                extended_rendered = rendered + label
                extended, e_images, e_videos, e_pixels, e_metadata = _process(
                    processor, process_vision_info, extended_rendered, messages)
                base_ids = ids[0].tolist()
                extended_ids = extended["input_ids"][0].tolist()
                if extended_ids != base_ids + [expected_id]:
                    raise ValueError("processed_answer_continuation_mismatch")
                del extended, e_images, e_videos, e_pixels, e_metadata, extended_ids
            t, gh, gw = map(int, grid[0].tolist())
            if t != 16 or gh % merge or gw % merge:
                raise ValueError("processor_video_geometry_invalid")
            tokens_per_group = gh * gw // merge**2
            visual_positions = [i for i, value in enumerate(ids[0].tolist())
                                if int(value) == int(processor.video_token_id)]
            if len(visual_positions) != t * tokens_per_group:
                raise ValueError("processor_video_token_geometry_invalid")
            groups = tuple(tuple(visual_positions[g * tokens_per_group:(g + 1) * tokens_per_group])
                           for g in range(t))
            if any(not group for group in groups):
                raise ValueError("processor_temporal_group_empty")
            event_groups = event_temporal_groups(design.onset, t)
            clip_token = f"oracle-v1:{design.family}:{design.onset}:{design.scene_seed}"
            controls = control_temporal_groups(scene_seed=design.scene_seed,
                                               clip_token=clip_token,
                                               event_groups=event_groups,
                                               temporal_group_count=t)
            sham = sham_temporal_groups(scene_seed=design.scene_seed,
                                        clip_token=clip_token, claim_kind=claim_kind,
                                        temporal_group_count=t)
            if len(event_groups) != 2 or len(controls) != 2 or len(sham) != 2:
                raise ValueError("processor_group_assignment_invalid")
            geometry_counts[tuple(map(int, grid[0].tolist()))] += 1
            sequence_counts[int(ids.shape[1])] += 1
            claim_token_counts[len(positions)] += 1
            event_overlap_counts[len(set(event_groups) & set(sham))] += 1
            control_overlap_counts[len(set(controls) & set(sham))] += 1
            frame_hash.update(bytes.fromhex(frame_digest.hexdigest()))
            prompt_hash.update(_canonical({
                "claim_kind": claim_kind,
                "scene_seed_sha256": _sha256(str(design.scene_seed).encode("ascii")),
                "rendered_prompt_sha256": _sha256(rendered.encode("utf-8")),
                "claim_positions": list(positions),
                "supported_label": supported_label,
                "answer_token_ids": [a_id, b_id],
                "video_grid_thw": [t, gh, gw],
                "event_groups": list(event_groups), "control_groups": list(controls),
                "sham_groups": list(sham),
            }))
            processor_hash.update(bytes.fromhex(_sha256(_canonical({
                "input_length": int(ids.shape[1]), "grid": [t, gh, gw],
                "frame_hash": frame_digest.hexdigest(),
            }))))
            prompt_count += 1
            successful += 1
            del batch, ids, grid, mask, images, videos, pixels, metadata, messages
        failed += 0

    expected_clips = len(designs)
    expected_prompts = 2 * expected_clips
    if successful != expected_prompts or prompt_count != expected_prompts or failed:
        raise RuntimeError("preflight_effective_count_mismatch")
    result = {
        "schema_version": 1,
        "status": "passed_cpu_processor_preflight_only",
        "mode": mode,
        "generator_version": GENERATOR_VERSION,
        "model_revision": MODEL_REVISION,
        "processor_sha256": processor_sha,
        "tokenizer_sha256": processor_descriptor["tokenizer_sha256"],
        "processor_descriptor_sha256": processor_sha,
        "runtime_versions": {
            "torch": str(torch.__version__),
            "transformers": str(importlib.metadata.version("transformers")),
            "qwen_vl_utils": str(importlib.metadata.version("qwen-vl-utils")),
        },
        "processor_implementation_sha256": implementation_fingerprints,
        "assignment_balance": {
            "cell_count": len(cell_counts),
            "cell_sizes": sorted(cell_counts.values()),
            "unique_cell_target_label_combinations": len(assignment_cells),
            "combination_counts": sorted(assignment_cells.values()),
        },
        "assignment_manifest_sha256": _sha256(_canonical(assignment)),
        "assignment_scene_seed_sha256": _sha256(_canonical([
            _sha256(str(row.scene_seed).encode("ascii")) for row in designs])),
        "source_commit": source_commit,
        "worktree_dirty": worktree_dirty,
        "source_sha256": source_hashes,
        "requested_clips": expected_clips,
        "effective_clips": expected_clips,
        "requested_claim_prompts": expected_prompts,
        "effective_claim_prompts": successful,
        "failure_count": failed,
        "model_loaded": False,
        "gpu_used": False,
        "frame_cohort_digest_sha256": frame_hash.hexdigest(),
        "prompt_binding_digest_sha256": prompt_hash.hexdigest(),
        "processor_geometry_digest_sha256": processor_hash.hexdigest(),
        "geometry_counts": [{"video_grid_thw": list(k), "prompts": v}
                            for k, v in sorted(geometry_counts.items())],
        "sequence_token_counts": [{"tokens": k, "prompts": v}
                                  for k, v in sorted(sequence_counts.items())],
        "claim_token_count_counts": [{"tokens": k, "prompts": v}
                                     for k, v in sorted(claim_token_counts.items())],
        "sham_event_overlap_counts": [{"groups": k, "prompts": v}
                                      for k, v in sorted(event_overlap_counts.items())],
        "sham_control_overlap_counts": [{"groups": k, "prompts": v}
                                        for k, v in sorted(control_overlap_counts.items())],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    try:
        os.write(fd, _canonical(result) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    if stat.S_IMODE(output.stat().st_mode) != 0o600:
        raise RuntimeError("preflight_receipt_mode_invalid")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(mode=args.mode, output=args.output)
    print(json.dumps({k: result[k] for k in (
        "status", "mode", "requested_clips", "effective_clips",
        "requested_claim_prompts", "effective_claim_prompts", "failure_count",
        "geometry_counts", "sequence_token_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
