#!/usr/bin/env python3
"""CPU-only processor geometry preflight for the frozen 40-pair cohort.

Loads the pinned local processor only. It disables CUDA visibility before
imports, streams a single clip at a time, and writes aggregate hashes/counts
only to a new mode-0600 file under data/private_nextgqa/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import time
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from qta_synthetic_event_construct_v1 import (  # noqa: E402
    CLIP_COUNT, EVENT_LENGTH as EVENT_FRAMES, FRAME_COUNT, LABELS,
    ONSET_STRATA as ONSET_FRAMES, PAIR_COUNT, TEMPORAL_PATCH_SIZE,
    WIDTH, HEIGHT, PairDesign, build_assignment, build_construct_clip,
    external_smoke_seed, assignment_manifest, sha256_canonical,
)
TEMPORAL_GROUP_COUNT = FRAME_COUNT // TEMPORAL_PATCH_SIZE

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_SNAPSHOT_RELATIVE = Path(
    ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
) / "snapshots" / MODEL_REVISION
PROTOCOL_RELATIVE = "experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md"
FORBIDDEN_KEYS = {
    "question", "choices", "answer", "answers", "gold", "frames", "frame",
    "pixels", "pixel_values", "pixel_values_videos", "input_ids", "prompt",
    "messages", "records", "transcript", "nickname", "absolute_path",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new aggregate-only JSON under data/private_nextgqa/")
    return parser


def validate_output_path(path: Path, repo_root: Path) -> Path:
    private = repo_root / "data" / "private_nextgqa"
    info = private.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("private_output_directory_permissions_invalid")
    resolved = private.resolve(strict=True)
    output = path if path.is_absolute() else repo_root / path
    if output.parent.resolve(strict=False) != resolved:
        raise ValueError("output_must_be_direct_child_of_private_directory")
    if output.suffix.lower() != ".json" or output.exists() or output.is_symlink():
        raise ValueError("output_must_be_new_json_file")
    return resolved / output.name


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def event_and_control_groups(onset_frame: int, temporal_groups: int,
                             patch_size: int = TEMPORAL_PATCH_SIZE
                             ) -> tuple[list[int], list[int]]:
    if (type(onset_frame) is not int or onset_frame < 0
            or type(temporal_groups) is not int
            or type(patch_size) is not int or patch_size <= 0
            or onset_frame % patch_size
            or onset_frame + EVENT_FRAMES > FRAME_COUNT
            or FRAME_COUNT % patch_size
            or temporal_groups != FRAME_COUNT // patch_size):
        raise ValueError("event_temporal_geometry_invalid")
    event_end = onset_frame + EVENT_FRAMES
    evidence = [group for group in range(temporal_groups)
                if max(onset_frame, group * patch_size)
                < min(event_end, (group + 1) * patch_size)]
    if len(evidence) != 2:
        raise ValueError("event_did_not_map_to_exactly_two_groups")
    event_center_twice = onset_frame * 2 + EVENT_FRAMES
    ranked = sorted(
        (group for group in range(temporal_groups) if group not in evidence),
        key=lambda group: (abs((2 * group + 1) * patch_size - event_center_twice),
                           group),
    )
    control = sorted(ranked[:2])
    if len(control) != 2 or set(evidence).intersection(control):
        raise ValueError("matched_control_groups_invalid")
    return evidence, control


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processor_fingerprint(processor: Any) -> tuple[str, str]:
    tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
    tokenizer_hash = sha256_bytes(tokenizer_json.encode("utf-8"))
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_sha256": tokenizer_hash,
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    return sha256_bytes(canonical_bytes(descriptor)), tokenizer_hash


def _frame_digest(clip: Any) -> str:
    digest = hashlib.sha256(b"qta-synthetic-event-construct-clip-v1\0")
    for index, frame in enumerate(clip.frames):
        if frame.mode != "RGB" or frame.size != (WIDTH, HEIGHT):
            raise ValueError("fixture_frame_contract_invalid")
        digest.update(index.to_bytes(4, "big"))
        digest.update(frame.tobytes())
    return digest.hexdigest()


def _batch_geometry(batch: Mapping[str, Any], torch_module: Any,
                    merge_size: int) -> tuple[dict[str, Any], str]:
    # Transformers returns BatchFeature, a Mapping rather than a plain dict.
    if not isinstance(batch, Mapping):
        raise ValueError("processor_batch_schema_invalid")
    for value in batch.values():
        if isinstance(value, torch_module.Tensor) and value.device.type != "cpu":
            raise ValueError("processor_batch_not_cpu")
    ids = batch.get("input_ids")
    grid = batch.get("video_grid_thw")
    if (not isinstance(ids, torch_module.Tensor) or ids.device.type != "cpu"
            or ids.ndim != 2 or ids.shape[0] != 1
            or not isinstance(grid, torch_module.Tensor) or grid.device.type != "cpu"
            or grid.ndim != 2 or tuple(grid.shape) != (1, 3)):
        raise ValueError("processor_batch_geometry_invalid")
    t, h, w = (int(value) for value in grid[0].tolist())
    if (t != FRAME_COUNT // TEMPORAL_PATCH_SIZE or h <= 0 or w <= 0
            or h % merge_size or w % merge_size or bool((grid <= 0).any().item())):
        raise ValueError("processor_grid_contract_invalid")
    geometry = {
        "sequence_tokens": int(ids.shape[1]),
        "video_grid_thw": [t, h, w],
        "merged_visual_tokens": t * (h // merge_size) * (w // merge_size),
    }
    return geometry, sha256_bytes(canonical_bytes(geometry))


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lower = key.lower() if isinstance(key, str) else ""
            if (not isinstance(key, str) or lower in FORBIDDEN_KEYS
                    or lower.startswith("raw_") or lower.endswith("_text")
                    or any(marker in lower for marker in
                           ("pixel_values", "token_ids", "frame_bytes"))):
                raise ValueError("aggregate_report_contains_private_field")
            _assert_aggregate_only(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_aggregate_only(child)
    elif isinstance(value, str) and (len(value) > 2048 or value.startswith(("/", "\\"))):
        raise ValueError("aggregate_report_string_invalid")


def write_report_exclusive(path: Path, report: dict[str, Any]) -> None:
    _assert_aggregate_only(report)
    payload = canonical_bytes(report) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _design_summary() -> dict[str, Any]:
    designs = build_assignment()
    smoke_seed = external_smoke_seed()
    if (len(designs) != PAIR_COUNT
            or len({row.scene_seed for row in designs}) != PAIR_COUNT
            or smoke_seed in {row.scene_seed for row in designs}):
        raise ValueError("frozen_seed_assignment_invalid")
    onset_counts = Counter(row.onset_frame for row in designs)
    orientation_counts: dict[int, Counter[bool]] = {}
    combination_counts: dict[int, Counter[tuple[bool, int]]] = {}
    correct_labels = {"circle": Counter(), "triangle": Counter()}
    for design in designs:
        onset, orientation, rotation = design.onset_frame, design.circle_left, design.choice_rotation
        orientation_counts.setdefault(onset, Counter())[orientation] += 1
        combination_counts.setdefault(onset, Counter())[(orientation, rotation)] += 1
        rotated = tuple(text for _label, text in design.choices)
        correct_labels["circle"][LABELS[rotated.index("The circle")]] += 1
        correct_labels["triangle"][LABELS[rotated.index("The triangle")]] += 1
    expected_combinations = {(orientation, rotation)
                             for orientation in (True, False) for rotation in range(5)}
    if (onset_counts != Counter({onset: 10 for onset in ONSET_FRAMES})
            or any(counts != Counter({True: 5, False: 5})
                   for counts in orientation_counts.values())
            or any(counts != Counter({combo: 1 for combo in expected_combinations})
                   for counts in combination_counts.values())
            or any(counts != Counter({label: 8 for label in LABELS})
                   for counts in correct_labels.values())):
        raise ValueError("frozen_design_balance_check_failed")
    return {
        "reserved_engineering_smoke_seed_sha256": sha256_bytes(str(smoke_seed).encode()),
        "sampled_cohort_seeds_unique": len({row.scene_seed for row in designs}),
        "onset_pair_counts": [{"start_frame": onset, "pairs": onset_counts[onset]}
                              for onset in ONSET_FRAMES],
        "orientation_pair_counts_by_onset": [
            {"start_frame": onset, "circle_left": orientation_counts[onset][True],
             "circle_right": orientation_counts[onset][False]}
            for onset in ONSET_FRAMES],
        "orientation_rotation_combinations_per_onset": 10,
        "circle_target_correct_label_counts": dict(sorted(correct_labels["circle"].items())),
        "triangle_target_correct_label_counts": dict(sorted(correct_labels["triangle"].items())),
    }


CHOICES_ROTATED = tuple(
    ("Both shapes", "The circle", "Neither shape", "The triangle", "The background")[i:]
    + ("Both shapes", "The circle", "Neither shape", "The triangle", "The background")[:i]
    for i in range(5)
)


def run_preflight(*, output: Path, repo_root: Path | None = None) -> int:
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    output_path = validate_output_path(output, repo_root)
    started = time.monotonic()
    stage = "imports"
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "qta_synthetic_event_construct_processor_preflight",
        "status": "incomplete",
        "scope": "synthetic_processor_geometry_only_no_qta_overlap",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _hash_file(repo_root / PROTOCOL_RELATIVE),
        "source_sha256": {},
        "requested_pairs": PAIR_COUNT,
        "effective_pairs": 0,
        "requested_clips": CLIP_COUNT,
        "effective_clips": 0,
        "frames_per_clip": FRAME_COUNT,
        "model_revision": MODEL_REVISION,
        "model_weights_loaded": False,
        "gpu_used": False,
        "model_forwards": 0,
        "qta_selector_overlap_measured": False,
        "faithfulness_result": False,
        "authorization": "processor_preflight_only_no_scoring_authorization",
    }
    try:
        stage = "source_hashes"
        source_files = {
            "runner": Path(__file__).resolve(),
            "fixture": Path(__file__).resolve().with_name("qta_synthetic_event_construct_v1.py"),
            "protocol": repo_root / PROTOCOL_RELATIVE,
            "qa_video_builder": repo_root / "scripts" / "run_nextgqa_gold_interval_margin_pilot.py",
        }
        report["source_sha256"] = {key: _hash_file(value)
                                   for key, value in source_files.items()}
        report["design_balance"] = _design_summary()
        manifest = assignment_manifest(python_version=platform.python_version())
        report["assignment_manifest_sha256"] = sha256_canonical(manifest)
        report["assignment_python_version"] = platform.python_version()

        stage = "cpu_processor_load"
        import torch
        import transformers
        import PIL
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video
        snapshot = Path.home() / MODEL_SNAPSHOT_RELATIVE
        if not snapshot.is_dir():
            raise FileNotFoundError("pinned_processor_snapshot_unavailable_offline")
        processor = AutoProcessor.from_pretrained(
            str(snapshot), revision=MODEL_REVISION, local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
            trust_remote_code=False,
        )
        processor_hash, tokenizer_hash = _processor_fingerprint(processor)
        merge_size = int(processor.image_processor.merge_size)
        patch_size = int(processor.video_processor.temporal_patch_size)
        if patch_size != TEMPORAL_PATCH_SIZE:
            raise ValueError("processor_temporal_patch_size_mismatch")
        report.update({
            "processor_sha256": processor_hash,
            "tokenizer_sha256": tokenizer_hash,
            "runtime": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "transformers": str(transformers.__version__), "pillow": str(PIL.__version__),
                "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
                "cuda_visible_devices": "empty",
            },
            "processor_temporal_patch_size": patch_size,
            "processor_spatial_merge_size": merge_size,
        })

        stage = "stream_pairs_through_cpu_processor"
        geometry_counts: Counter[tuple[int, int, int]] = Counter()
        sequence_counts: Counter[int] = Counter()
        evidence_counts: Counter[int] = Counter()
        control_counts: Counter[int] = Counter()
        cohort_digest = hashlib.sha256(b"qta-construct-cohort-pairs-v1\0")
        group_digest = hashlib.sha256(b"qta-construct-pair-geometry-v1\0")
        for design in build_assignment():
            ordinal = design.ordinal
            hashes: list[dict[str, str]] = []
            group_geometry: list[tuple[list[int], list[int], list[int]]] = []
            for changed_object in ("circle", "triangle"):
                clip = build_construct_clip(design, changed_object)
                frame_hash = _frame_digest(clip)
                item = {"question": clip.question,
                        "choices": [text for _label, text in clip.choices]}
                rendered_text, messages = build_qa_prompt(processor, item, clip.frames, (1, 1))
                rendered_hash = sha256_bytes(rendered_text.encode("utf-8"))
                choice_hash = sha256_bytes(canonical_bytes(item["choices"]))
                images, videos = process_vision_info(messages, return_video_metadata=True)
                if images is not None or not isinstance(videos, list) or len(videos) != 1:
                    raise ValueError("video_processor_input_contract_invalid")
                pixels, video_metadata = videos[0]
                if len(pixels) != FRAME_COUNT:
                    raise ValueError("processor_video_frame_count_mismatch")
                batch = process_qa_video(processor, rendered_text, images, pixels, video_metadata)
                geometry, geometry_hash = _batch_geometry(batch, torch, merge_size)
                grid = geometry["video_grid_thw"]
                evidence, control = event_and_control_groups(design.onset_frame, grid[0], patch_size)
                if len(evidence) != 2 or len(control) != 2 or set(evidence) & set(control):
                    raise ValueError("event_control_group_contract_invalid")
                hashes.append({
                    "frame_sha256": frame_hash, "rendered_qa_sha256": rendered_hash,
                    "choice_list_sha256": choice_hash, "processor_geometry_sha256": geometry_hash,
                })
                group_geometry.append((evidence, control, grid))
                geometry_counts[tuple(grid)] += 1
                sequence_counts[geometry["sequence_tokens"]] += 1
                evidence_counts.update(evidence)
                control_counts.update(control)
                del batch, pixels, video_metadata, videos, messages, rendered_text, clip
                del images, item, geometry, grid, evidence, control

            if len({row["rendered_qa_sha256"] for row in hashes}) != 1:
                raise ValueError("paired_prompt_or_options_mismatch")
            if len({row["choice_list_sha256"] for row in hashes}) != 1:
                raise ValueError("paired_choice_list_mismatch")
            if len({row["processor_geometry_sha256"] for row in hashes}) != 1:
                raise ValueError("paired_processor_geometry_mismatch")
            if group_geometry[0] != group_geometry[1]:
                raise ValueError("paired_evidence_control_mapping_mismatch")
            cohort_digest.update(canonical_bytes({
                "ordinal": ordinal, "scene_seed_sha256": sha256_bytes(str(design.scene_seed).encode()),
                "event_start_frame": design.onset_frame,
                "circle_left": design.circle_left,
                "answer_rotation": design.choice_rotation,
                "clip_frame_hashes": [row["frame_sha256"] for row in hashes],
                "rendered_qa_sha256": hashes[0]["rendered_qa_sha256"],
                "choice_list_sha256": hashes[0]["choice_list_sha256"],
                "processor_geometry_sha256": hashes[0]["processor_geometry_sha256"],
            }))
            group_digest.update(canonical_bytes({
                "evidence": group_geometry[0][0], "control": group_geometry[0][1],
                "video_grid_thw": group_geometry[0][2],
            }))
            report["effective_pairs"] += 1
            report["effective_clips"] += 2
            del design, hashes, group_geometry

        if report["effective_pairs"] != PAIR_COUNT or report["effective_clips"] != CLIP_COUNT:
            raise ValueError("cohort_effective_count_mismatch")
        if len(geometry_counts) != 1:
            raise ValueError("cohort_processor_geometry_not_constant")
        report.update({
            "status": "passed_cpu_processor_preflight_no_model_forward",
            "cohort_frames_sha256": cohort_digest.hexdigest(),
            "paired_group_geometry_sha256": group_digest.hexdigest(),
            "checks": {
                "paired_rendered_qa_equal": PAIR_COUNT,
                "paired_choice_lists_equal": PAIR_COUNT,
                "paired_processor_geometry_equal": PAIR_COUNT,
                "clips_with_exactly_two_event_groups": CLIP_COUNT,
                "clips_with_two_disjoint_nearest_controls": CLIP_COUNT,
            },
            "aggregate_processor_geometry": {
                "grid_counts": [{"video_grid_thw": list(grid), "clips": count}
                                for grid, count in sorted(geometry_counts.items())],
                "sequence_token_count_min": min(sequence_counts),
                "sequence_token_count_max": max(sequence_counts),
                "evidence_group_clip_counts": [
                    {"group": group, "clips": evidence_counts[group]}
                    for group in range(TEMPORAL_GROUP_COUNT)],
                "control_group_clip_counts": [
                    {"group": group, "clips": control_counts[group]}
                    for group in range(TEMPORAL_GROUP_COUNT)],
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
    except Exception as exc:
        report.update({
            "status": "failed_cpu_processor_preflight",
            "failure_stage": stage,
            "failure_type": type(exc).__name__,
            "failure_code": str(exc) if str(exc).isidentifier() else "preflight_failed",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        write_report_exclusive(output_path, report)
        return 1
    write_report_exclusive(output_path, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_preflight(output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
