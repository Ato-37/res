#!/usr/bin/env python3
"""Local-only exploratory QTA selected-group stability at 16 vs 32 frames.

For the same three deterministic NExT-GQA test videos/questions, perform one
untreated Qwen attention capture per video at each resolution. Report only
finite-cohort interval Jaccard and selected-center Wasserstein-1, with paired
values identified only by fixed ordinal; no row-level selection, IDs, text,
logits, or media-derived values are written. This is selector stability only,
not a faithfulness result. Rights remain unverified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
COUNTS = (16, 32)
RECORDS = 3
LAYERS = tuple(range(4, 16))
TOLERANCE = Fraction(50, 1000)
PROMPT = "Answer using visible video evidence: {question}"
RANDOM_SEED = 20261018
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processor_group_support(frame_times: list[float], temporal_steps: int,
                            temporal_patch_size: int) -> tuple[list[tuple[float, float]], list[float]]:
    """Map consecutive processor frame patches to existing midpoint-cell support.

    Boundaries use normalized sampled-frame PTS `[0, adjacent midpoints, 1]`.
    A processor group's support spans boundaries at its first and final frame
    offsets; its center for supplemental W1 is the patch PTS mean.
    """
    if (type(temporal_steps) is not int or temporal_steps < 2
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or len(frame_times) != temporal_steps * temporal_patch_size):
        raise ValueError("processor_temporal_mapping_shape_invalid")
    if (len(frame_times) < 2 or frame_times[0] != 0
            or any(not math.isfinite(x) for x in frame_times)
            or any(b <= a for a, b in zip(frame_times, frame_times[1:]))):
        raise ValueError("processor_frame_times_invalid")
    end = frame_times[-1]
    normalized = [value / end for value in frame_times]
    boundaries = [0.0] + [(a + b) / 2 for a, b in zip(normalized, normalized[1:])] + [1.0]
    intervals = []
    centers = []
    for group in range(temporal_steps):
        start = group * temporal_patch_size
        stop = start + temporal_patch_size
        intervals.append((boundaries[start], boundaries[stop]))
        centers.append(sum(normalized[start:stop]) / temporal_patch_size)
    return intervals, centers


def interval_jaccard(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    def union_length(rows):
        total = 0.0
        right = -1.0
        for left, end in sorted(rows):
            if not (0 <= left <= end <= 1):
                raise ValueError("interval_bounds_invalid")
            total += max(0.0, end - max(left, right))
            right = max(right, end)
        return total
    intersection = union_length([(max(x0, y0), min(x1, y1)) for x0, x1 in a for y0, y1 in b if max(x0, y0) < min(x1, y1)])
    union = union_length(a) + union_length(b) - intersection
    if union <= 0:
        raise ValueError("interval_union_empty")
    return intersection / union


def wasserstein1_empirical(a: list[float], b: list[float]) -> float:
    """Exact 1-D W1 between empirical center distributions, any sample sizes."""
    if not a or not b or any(not math.isfinite(x) or not 0 <= x <= 1 for x in a + b):
        raise ValueError("center_samples_invalid")
    points = sorted(set(a + b))
    result = 0.0
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2
        cdf_a = sum(x <= midpoint for x in a) / len(a)
        cdf_b = sum(x <= midpoint for x in b) / len(b)
        result += (right - left) * abs(cdf_a - cdf_b)
    return result


def selected_union(group_intervals, selected):
    if not selected or len(set(selected)) != len(selected) or any(
            type(i) is not int or i < 0 or i >= len(group_intervals) for i in selected):
        raise ValueError("selected_groups_invalid")
    return [group_intervals[index] for index in sorted(selected)]


def compare_selections(intervals16, centers16, groups16, intervals32, centers32, groups32):
    intervals16 = selected_union(intervals16, groups16)
    intervals32 = selected_union(intervals32, groups32)
    centers16 = [centers16[i] for i in groups16]
    centers32 = [centers32[i] for i in groups32]
    return interval_jaccard(intervals16, intervals32), wasserstein1_empirical(centers16, centers32)


def summarize_pairs(pairs):
    if not pairs or any(len(pair) != 2 or any(not math.isfinite(value) for value in pair)
                        for pair in pairs):
        raise ValueError("pair_metrics_invalid")
    return {"interval_jaccard_mean": sum(x[0] for x in pairs) / len(pairs),
            "interval_jaccard_min": min(x[0] for x in pairs),
            "selected_center_wasserstein1_mean": sum(x[1] for x in pairs) / len(pairs),
            "selected_center_wasserstein1_max": max(x[1] for x in pairs)}


def pair_metric_rows(pairs):
    if not pairs:
        raise ValueError("pair_metrics_empty")
    return [{"fixed_pair_ordinal": index + 1,
             "interval_jaccard": values[0],
             "selected_center_wasserstein1": values[1]}
            for index, values in enumerate(pairs)]


def _write_private(path: Path, payload: dict, *, safe_roots=None):
    path = Path(os.path.abspath(path))
    if path.is_symlink() or path.exists():
        raise FileExistsError("output_exists")
    probe = path.parent
    while probe != probe.parent:
        if probe.is_symlink():
            raise PermissionError("output_parent_symlink")
        probe = probe.parent
    roots = [Path.home().resolve(), Path(__file__).resolve().parents[1], Path("/tmp").resolve()]
    if safe_roots is not None:
        roots = [Path(root).resolve() for root in safe_roots]
    if not path.parent.exists():
        path.parent.mkdir(mode=0o700)
    parent = path.parent.resolve(strict=True)
    if not any(parent == root or root in parent.parents for root in roots):
        raise PermissionError("output_parent_outside_safe_roots")
    parent_stat = parent.stat()
    if (parent.is_symlink() or parent_stat.st_uid != os.getuid()
            or parent_stat.st_mode & 0o077):
        raise PermissionError("output_directory_must_be_private")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


_RUN_STATE = {"completed_forwards": 0, "effective_video_pairs": 0,
              "candidate_videos_inspected": 0,
              "requested_video_pairs": RECORDS,
              "requested_forwards": RECORDS * 2,
              "smoke_only": False,
              "started_at_utc": None,
              "started_monotonic": None}


def _execute(args):
    import torch
    import transformers
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_processor_module
    import qwen_vl_utils.vision_process as qwen_video_utils_module
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from claim_mask_video_decoder import decode_sampled_video_sets, verify_sampled_frame_pixels
    from qwen_question_temporal_attention_selector import map_question_token_positions, select_top_temporal_groups
    from qta_attention_forward_capture import capture_qta_attention_forward_once, aggregate_captured_qta_rows
    from run_nextgqa_real_video_gpu_smoke import choose_videos
    from smoke_qwen_real_video_attention_boost import _make_batch, _temporal_visual_groups, question_for_video

    repo = Path(__file__).resolve().parents[1]
    record_count = 1 if args.smoke else RECORDS
    _RUN_STATE["requested_video_pairs"] = record_count
    _RUN_STATE["requested_forwards"] = record_count * 2
    _RUN_STATE["smoke_only"] = bool(args.smoke)
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    candidates = choose_videos(repo, dataset)
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    examples = []
    # Freeze exactly the first three eligible candidates before attempting
    # either-resolution decode. A decode/timing failure fails this cohort; it
    # must not trigger replacement based on observed model eligibility.
    inspection_count = 0
    _RUN_STATE["candidate_videos_inspected"] = inspection_count
    for video in candidates[:record_count]:
        inspection_count += 1
        _RUN_STATE["candidate_videos_inspected"] = inspection_count
        try:
            frames, metadata = decode_sampled_video_sets(video, frame_counts=COUNTS, uniform_tolerance_seconds=TOLERANCE)
            for count in COUNTS:
                verify_sampled_frame_pixels(frames[count], metadata[count])
            question = question_for_video(dataset, video, video_root)
        except ValueError as exc:
            raise RuntimeError("frozen_candidate_16_32_decode_or_question_failure") from exc
        examples.append((video, frames, metadata, question))
    if len(examples) != record_count:
        raise RuntimeError("fixed_cohort_shared_16_32_uniform_pts_videos_unavailable")

    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / MODEL_REVISION
    if not model_path.is_dir():
        raise RuntimeError("pinned_checkpoint_not_cached")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_bf16_unavailable")
    device = torch.cuda.current_device()
    free_before, total = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise RuntimeError("insufficient_free_vram_before_model_load")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    free_after_load, _ = torch.cuda.mem_get_info(device)
    if free_after_load < MIN_FREE_BEFORE_FORWARD:
        raise RuntimeError("insufficient_free_vram_after_model_load")
    if int(model.config.video_token_id) != int(processor.video_token_id) or len(model.model.language_model.layers) < 16:
        raise RuntimeError("pinned_model_processor_contract_invalid")

    per_resolution = {count: [] for count in COUNTS}
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    video_processor = getattr(processor, "video_processor", processor.image_processor)
    if int(video_processor.temporal_patch_size) != temporal_patch_size:
        raise RuntimeError("processor_temporal_patch_size_disagreement")
    model_temporal_patch = getattr(model.config.vision_config, "temporal_patch_size", None)
    if model_temporal_patch is None or int(model_temporal_patch) != temporal_patch_size:
        raise RuntimeError("model_processor_temporal_patch_size_disagreement")
    processor_geometry = None
    processing_started = time.monotonic()
    completed_forwards = 0
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    for video, frame_sets, metadata_sets, question in examples:
        for count in COUNTS:
            meta = metadata_sets[count]
            fps = Fraction(*meta["effective_sample_fps"])
            batch, _, rendered = _make_batch(frames=frame_sets[count], sample_fps=fps, question=question,
                processor=processor, torch_module=torch, frame_count=count)
            ids = batch["input_ids"]
            steps, gh, gw = map(int, batch["video_grid_thw"][0].tolist())
            frame_times = [Fraction(row["relative_presentation_time"][0], row["relative_presentation_time"][1])
                           for row in meta["sampled_frames"]]
            if steps * temporal_patch_size != count:
                raise RuntimeError("processor_temporal_patch_count_contract_invalid")
            geometry = (temporal_patch_size, int(processor.image_processor.patch_size),
                        int(processor.image_processor.merge_size), str(MODEL_REVISION))
            if processor_geometry is None:
                processor_geometry = geometry
            elif geometry != processor_geometry:
                raise RuntimeError("processor_revision_or_temporal_geometry_changed")
            group_intervals, group_centers = processor_group_support(
                frame_times, steps, temporal_patch_size)
            groups = _temporal_visual_groups(token_ids=[int(x) for x in ids[0].tolist()],
                video_token_id=int(processor.video_token_id), temporal_steps=steps, grid_h=gh, grid_w=gw,
                merge_size=int(processor.image_processor.merge_size))
            if len(group_intervals) != len(groups) or len(group_centers) != steps:
                raise RuntimeError("processor_temporal_group_time_count_mismatch")
            qpos = map_question_token_positions(tokenizer=processor.tokenizer, rendered_text=rendered,
                question_text=question, processed_input_ids=ids, video_token_id=int(processor.video_token_id))
            free_before_forward, _ = torch.cuda.mem_get_info(device)
            if free_before_forward < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("insufficient_free_vram_before_forward")
            _, rows, _ = capture_qta_attention_forward_once(model=model,
                forward=lambda: model(**batch, use_cache=False, logits_to_keep=1), question_positions=qpos,
                input_length=int(ids.shape[1]), torch_module=torch)
            completed_forwards += 1
            _RUN_STATE["completed_forwards"] = completed_forwards
            scores = aggregate_captured_qta_rows(rows=rows, group_positions=groups, torch_module=torch)
            selected = select_top_temporal_groups(scores, top_k=max(1, math.ceil(steps * 0.10)))
            per_resolution[count].append({"intervals": group_intervals,
                "centers": group_centers, "groups": selected,
                "source_hash": meta["source_sha256"], "pts_hash": meta["decoded_pts_sequence_sha256"],
                "rgb_hash": meta["pass1_decoded_rgb_sequence_sha256"],
                "sampled_pts_hash": hashlib.sha256(json.dumps(
                    [row["relative_presentation_time"] for row in meta["sampled_frames"]],
                    separators=(",", ":")).encode("ascii")).hexdigest(),
                "selected_rgb_hash": hashlib.sha256(json.dumps(
                    [row["rgb_pixel_sha256"] for row in meta["sampled_frames"]],
                    separators=(",", ":")).encode("ascii")).hexdigest()})
            del batch, rows
            torch.cuda.empty_cache()
        if count == 32:
            _RUN_STATE["effective_video_pairs"] += 1

    pairs = [compare_selections(per_resolution[16][i]["intervals"], per_resolution[16][i]["centers"],
             per_resolution[16][i]["groups"], per_resolution[32][i]["intervals"],
             per_resolution[32][i]["centers"], per_resolution[32][i]["groups"])
             for i in range(record_count)]
    sources = {"runner": Path(__file__), "decoder": repo / "scripts/claim_mask_video_decoder.py",
        "candidate_selector": repo / "scripts/run_nextgqa_real_video_gpu_smoke.py",
        "question_mapping": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "capture": repo / "scripts/qta_attention_forward_capture.py",
        "selector_wrapper": repo / "scripts/smoke_qwen_real_video_attention_boost.py",
        "time_mapper": repo / "scripts/qwen_review_interval_token_groups.py",
        "qwen_video_processor": Path(qwen_video_processor_module.__file__),
        "qwen_video_utils": Path(qwen_video_utils_module.__file__)}
    manifests = sorted((repo / "data").glob("nextgqa_samples_*.json"))
    manifest_hashes = [(path.name, _sha256(path)) for path in manifests]
    payload = {"status": "passed", "scope": ("exploratory_qta_selector_resolution_stability_smoke"
                  if args.smoke else "exploratory_qta_selector_resolution_stability"),
        "smoke_only": bool(args.smoke),
        "started_at_utc": _RUN_STATE["started_at_utc"], "source_sha256": {k: _sha256(v) for k, v in sources.items()},
        "dataset_input_sha256": {name: _sha256(dataset / name) for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json")},
        "model_revision": MODEL_REVISION, "model_config_sha256": _sha256(model_path / "config.json"),
        "processor_revision": MODEL_REVISION, "processor_temporal_patch_size": temporal_patch_size,
        "processor_geometry_sha256": hashlib.sha256(json.dumps({
            "temporal_patch_size": temporal_patch_size,
            "patch_size": int(processor.image_processor.patch_size),
            "merge_size": int(processor.image_processor.merge_size),
            "image_processor": processor.image_processor.to_dict(),
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(device),
        "precision": "bf16_weights_4bit_bf16_compute", "dataset_split": "NExT-GQA test",
        "candidate_selection": f"first {record_count} from existing deterministic choose_videos; same qid-sorted question per video; frozen before decoding and never replaced",
        "candidate_exclusion_manifest_count": len(manifest_hashes),
        "candidate_exclusion_manifests_sha256": hashlib.sha256(json.dumps(
            manifest_hashes, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "requested_videos": record_count, "effective_video_pairs": len(pairs),
        "failed_video_pairs": record_count - len(pairs), "candidate_videos_inspected": inspection_count,
        "requested_forwards": record_count * 2,
        "completed_forwards": completed_forwards,
        "failed_forwards": record_count * 2 - completed_forwards,
        "frame_counts": [16, 32], "uniform_grid_tolerance_seconds": [TOLERANCE.numerator, TOLERANCE.denominator],
        "free_vram_before_model_load_gib": free_before / 1024**3,
        "free_vram_after_model_load_gib": free_after_load / 1024**3,
        "total_vram_gib": total / 1024**3,
        "selector": "untreated question-token attention averaged across heads/layers 5-16; top ceil(0.10*T) complete processor temporal groups",
        "temporal_support_mapping": "normalized sampled-span frame cells [0, adjacent midpoints, last sampled PTS]; processor group g spans boundaries[g*temporal_patch_size:(g+1)*temporal_patch_size]; center W1 uses patch PTS mean; not full-duration coverage",
        "primary_statistic": f"mean per-video interval Jaccard across {record_count} fixed pair(s)",
        "metrics": summarize_pairs(pairs),
        "random_seed": RANDOM_SEED,
        "fixed_pair_metrics": pair_metric_rows(pairs),
        "input_hash_aggregates": {"source_video_sha256_aggregate": hashlib.sha256("".join(sorted(x["source_hash"] for x in per_resolution[16])).encode()).hexdigest(),
            "full_decoded_pts_sequence_sha256_aggregate": hashlib.sha256("".join(sorted(x["pts_hash"] for x in per_resolution[16])).encode()).hexdigest(),
            "full_decoded_rgb_sequence_sha256_aggregate": hashlib.sha256("".join(sorted(x["rgb_hash"] for x in per_resolution[16])).encode()).hexdigest(),
            "sampled_pts_sequence_16_sha256_aggregate": hashlib.sha256("".join(sorted(x["sampled_pts_hash"] for x in per_resolution[16])).encode()).hexdigest(),
            "sampled_pts_sequence_32_sha256_aggregate": hashlib.sha256("".join(sorted(x["sampled_pts_hash"] for x in per_resolution[32])).encode()).hexdigest(),
            "selected_rgb_sequence_16_sha256_aggregate": hashlib.sha256("".join(sorted(x["selected_rgb_hash"] for x in per_resolution[16])).encode()).hexdigest(),
            "selected_rgb_sequence_32_sha256_aggregate": hashlib.sha256("".join(sorted(x["selected_rgb_hash"] for x in per_resolution[32])).encode()).hexdigest()},
        "attention_capture_and_analysis_elapsed_seconds": time.monotonic() - processing_started,
        "elapsed_seconds": time.monotonic() - _RUN_STATE["started_monotonic"], "faithfulness_result": False,
        "limitations": ["selected-center Wasserstein-1 is supplemental descriptive only", "interval support is normalized over sampled endpoint span and is not full-duration coverage", "same-group stability is confounded by changed visual tokenization and pre/post frame sampling", f"fixed finite cohort contains only {record_count} video pair(s)", "QTA attention selection is descriptive and does not establish causal or explanation faithfulness"],
        "ids_questions_answers_raw_logits_and_per_video_selections_saved": False, "local_only": True, "rights_status": "unverified_local_only"}
    _write_private(args.output, payload)
    print(json.dumps({k: payload[k] for k in ("status", "scope", "requested_videos", "effective_video_pairs", "requested_forwards", "completed_forwards", "metrics", "faithfulness_result")}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new private local JSON summary; overwrite refused")
    parser.add_argument("--smoke", action="store_true",
                        help="run only the first frozen candidate pair as a non-inferential path smoke")
    args = parser.parse_args(argv)
    _RUN_STATE.update(completed_forwards=0, effective_video_pairs=0,
                      candidate_videos_inspected=0,
                      requested_video_pairs=1 if args.smoke else RECORDS,
                      requested_forwards=(1 if args.smoke else RECORDS) * 2,
                      smoke_only=bool(args.smoke),
                      started_at_utc=datetime.now(timezone.utc).isoformat(),
                      started_monotonic=time.monotonic())
    try:
        _execute(args)
        return 0
    except Exception as exc:
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"local_run_failed_{type(exc).__name__}"
        requested_pairs = _RUN_STATE["requested_video_pairs"]
        requested_forwards = _RUN_STATE["requested_forwards"]
        failed = max(0, requested_pairs - _RUN_STATE["effective_video_pairs"])
        failure = {"status": "failed", "scope": ("exploratory_qta_selector_resolution_stability_smoke"
            if _RUN_STATE["smoke_only"] else "exploratory_qta_selector_resolution_stability"),
            "smoke_only": _RUN_STATE["smoke_only"],
            "started_at_utc": _RUN_STATE["started_at_utc"],
            "elapsed_seconds": time.monotonic() - _RUN_STATE["started_monotonic"],
            "runner_sha256": _sha256(Path(__file__)),
            "failure_code": code, "requested_videos": requested_pairs,
            "effective_video_pairs": _RUN_STATE["effective_video_pairs"],
            "failed_video_pairs": failed, "candidate_videos_inspected": _RUN_STATE["candidate_videos_inspected"],
            "requested_forwards": requested_forwards,
            "completed_forwards": _RUN_STATE["completed_forwards"],
            "failed_forwards": requested_forwards - _RUN_STATE["completed_forwards"],
            "faithfulness_result": False, "local_only": True,
            "rights_status": "unverified_local_only"}
        try:
            _write_private(args.output, failure)
            artifact = "written"
        except Exception:
            artifact = "unavailable_output_path_unsafe_or_exists"
        print(json.dumps({"status": "failed", "failure_code": code,
            "failure_artifact": artifact, "requested_videos": requested_pairs,
            "effective_video_pairs": _RUN_STATE["effective_video_pairs"],
            "failed_video_pairs": failed, "requested_forwards": requested_forwards,
            "completed_forwards": _RUN_STATE["completed_forwards"]}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
