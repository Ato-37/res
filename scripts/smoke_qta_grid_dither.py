#!/usr/bin/env python3
"""Local-only exploratory QTA selection stability under a 32-frame grid dither.

The fixed baseline is endpoint-inclusive uniform sampling at 32 frames. The
comparison shifts odd interior sample slots +1 decoded index and leaves even
interior slots and both endpoints unchanged. Every row must remain sorted,
unique, and within the fixed 50 ms actual-PTS uniformity tolerance. This is a
small descriptive selector diagnostic, not a faithfulness result. Rights are
unverified; output is private and local.
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
FRAME_COUNT = 32
RECORDS = 3
LAYERS = tuple(range(5, 17))
TOLERANCE = Fraction(1, 20)
RANDOM_SEED = 20261018
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2
SOURCE_NAMES = ("smoke_qta_grid_dither.py", "claim_mask_video_decoder.py", "presentation_time_utils.py",
                "necessity_choice_margin_utils.py", "nextgqa_atomic_batch_freeze.py", "run_nextgqa_real_video_gpu_smoke.py",
                "smoke_qwen_real_video_attention_boost.py", "qwen_question_temporal_attention_selector.py",
                "qta_attention_forward_capture.py", "qwen_review_interval_token_groups.py")
STATE = {"completed_forwards": 0, "effective_videos": 0,
         "attempted_forwards": 0,
         "attempted_videos": 0,
         "inspected": 0, "requested_videos": RECORDS,
         "requested_forwards": RECORDS * 2, "smoke": False,
         "started_at": None, "started_monotonic": None,
         "exclusion_manifests": None, "model_snapshot_before": None,
         "model_snapshot_after": None, "model_snapshot_hash_before_seconds": None,
         "model_snapshot_hash_after_seconds": None, "source_hashes_before": None,
         "source_hashes_after": None}


def dithered_indices(frame_total: int, sample_count: int = FRAME_COUNT) -> tuple[list[int], list[int]]:
    """Return baseline and frozen +1/0 dither grids, rejecting invalid grids."""
    from necessity_choice_margin_utils import uniform_sample_indices
    if type(frame_total) is not int or type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_grid_arguments_invalid")
    baseline = uniform_sample_indices(frame_total, sample_count)
    dithered = list(baseline)
    for slot in range(1, sample_count - 1):
        if slot % 2 == 1:
            dithered[slot] += 1
    validate_grid(baseline, frame_total, sample_count)
    validate_grid(dithered, frame_total, sample_count)
    if dithered[0] != baseline[0] or dithered[-1] != baseline[-1]:
        raise ValueError("dither_endpoint_changed")
    return baseline, dithered


def validate_grid(indices, frame_total: int, expected_count: int = FRAME_COUNT) -> None:
    if (type(frame_total) is not int or frame_total < expected_count
            or not isinstance(indices, (list, tuple)) or len(indices) != expected_count
            or any(type(i) is not int or i < 0 or i >= frame_total for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError("sample_grid_not_sorted_unique_in_range")
    if indices[0] != 0 or indices[-1] != frame_total - 1:
        raise ValueError("sample_grid_endpoints_invalid")


def require_uniform_pts(times, tolerance=TOLERANCE):
    from presentation_time_utils import uniform_sampling_rate_from_times
    if len(times) != FRAME_COUNT or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("sample_pts_not_strictly_increasing")
    return uniform_sampling_rate_from_times(times, tolerance)


def group_set_jaccard(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        raise ValueError("selected_group_set_empty")
    return len(a & b) / len(a | b)


def validate_capture_receipt(receipt):
    expected_layers = list(LAYERS)
    if (not isinstance(receipt, dict)
            or receipt.get("status") != "one_forward_attention_rows_captured_non_authorizing"
            or receipt.get("forward_calls") != 1
            or receipt.get("layer_indices_1_based") != expected_layers
            or receipt.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
        raise RuntimeError("attention_capture_receipt_invalid")
    calls = receipt.get("layer_calls")
    if not isinstance(calls, dict) or calls != {str(layer): 1 for layer in expected_layers}:
        raise RuntimeError("attention_capture_layer_receipt_invalid")
    deviation = receipt.get("attention_row_sum_max_abs_deviation")
    if (isinstance(deviation, bool) or not isinstance(deviation, (int, float))
            or not math.isfinite(float(deviation)) or deviation < 0):
        raise RuntimeError("attention_capture_deviation_invalid")
    return float(deviation)


def empirical_wasserstein1(a, b):
    if not a or not b or any(not math.isfinite(x) or not 0 <= x <= 1 for x in list(a) + list(b)):
        raise ValueError("center_samples_invalid")
    points = sorted(set(a + b))
    return sum((right - left) * abs(sum(x <= (left + right) / 2 for x in a) / len(a)
                - sum(x <= (left + right) / 2 for x in b) / len(b))
               for left, right in zip(points, points[1:]))


def support_intervals(pts, temporal_steps, temporal_patch_size):
    """Use sampled-frame midpoint cells; a patch group spans its constituent cells."""
    if (len(pts) != temporal_steps * temporal_patch_size or len(pts) < 2
            or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("processor_temporal_mapping_shape_invalid")
    start = pts[0]
    duration = pts[-1] - start
    normalized = [(x - start) / duration for x in pts]
    boundaries = [0.0] + [(a + b) / 2 for a, b in zip(normalized, normalized[1:])] + [1.0]
    intervals = [(boundaries[g * temporal_patch_size], boundaries[(g + 1) * temporal_patch_size])
                 for g in range(temporal_steps)]
    centers = [sum(normalized[g * temporal_patch_size:(g + 1) * temporal_patch_size]) / temporal_patch_size
               for g in range(temporal_steps)]
    return intervals, centers


def interval_union_jaccard(left, right):
    def length(rows):
        total, end = 0.0, 0.0
        for a, b in sorted(rows):
            if not (0 <= a <= b <= 1):
                raise ValueError("interval_bounds_invalid")
            total += max(0.0, b - max(a, end))
            end = max(end, b)
        return total
    intersections = [(max(a, c), min(b, d)) for a, b in left for c, d in right if max(a, c) < min(b, d)]
    union = length(left) + length(right) - length(intersections)
    if union <= 0:
        raise ValueError("interval_union_empty")
    return length(intersections) / union


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot_sha256(root):
    """Hash exact bytes of every regular pinned snapshot file (HF blob links included)."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError("pinned_snapshot_path_invalid")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), path.stat().st_size, _sha(path)))
    if not entries:
        raise RuntimeError("pinned_snapshot_empty")
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def exclusion_manifest_receipt(repo):
    entries = [(path.name, _sha(path)) for path in sorted((Path(repo) / "data").glob("nextgqa_samples_*.json"))]
    return {"names_and_sha256": entries, "count": len(entries),
            "digest_sha256": hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()}


def forward_count_receipt(requested, attempted, completed):
    if (any(type(value) is not int or value < 0 for value in (requested, attempted, completed))
            or completed > attempted or attempted > requested):
        raise ValueError("forward_count_contract_invalid")
    return {"requested_forwards": requested, "attempted_forwards": attempted,
            "completed_forwards": completed, "failed_forwards": attempted-completed,
            "not_attempted_forwards": requested-attempted}


def scope_name(smoke):
    return "exploratory_qta_32_frame_grid_dither_smoke" if smoke else "exploratory_qta_32_frame_grid_dither"


def video_pair_count_receipt(requested, attempted, completed):
    if (any(type(value) is not int or value < 0 for value in (requested, attempted, completed))
            or completed > attempted or attempted > requested):
        raise ValueError("video_pair_count_contract_invalid")
    return {"requested_video_pairs": requested, "attempted_video_pairs": attempted,
            "completed_video_pairs": completed, "failed_video_pairs": attempted-completed,
            "not_attempted_video_pairs": requested-attempted}


def source_hash_receipt(repo):
    hashes = {name: _sha(Path(repo) / "scripts" / name) for name in SOURCE_NAMES}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": hashes, "digest_sha256": digest}


def _write_private(path, payload):
    path = Path(os.path.abspath(path))
    if path.is_symlink() or path.exists():
        raise FileExistsError("output_exists")
    parent = path.parent
    if not parent.exists():
        parent.mkdir(mode=0o700)
    resolved = parent.resolve(strict=True)
    roots = [Path.home().resolve(), Path(__file__).resolve().parents[1], Path("/tmp").resolve()]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PermissionError("output_parent_outside_safe_roots")
    ancestor = parent
    while ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise PermissionError("output_parent_symlink")
        ancestor = ancestor.parent
    if parent.is_symlink() or resolved.stat().st_uid != os.getuid() or resolved.stat().st_mode & 0o077:
        raise PermissionError("output_directory_must_be_private")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _decode_two_grids(path):
    """Use decoder's two-pass contract for the union of frozen baseline/dither indices."""
    import av
    from claim_mask_video_decoder import (_decode_selected_frames, _file_sha256, _image_sha256,
                                          _stream_and_decode)
    from presentation_time_utils import presentation_times_from_pts
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("video_input_missing")
    with path.open("rb") as source:
        before = _file_sha256(source)
        pts, time_base, rel_times, pixel_spec, pixel_hashes, stream_info = _stream_and_decode(source)
        base, dither = dithered_indices(len(pts))
        base_rate, base_dev = require_uniform_pts([rel_times[i] for i in base])
        dith_rate, dith_dev = require_uniform_pts([rel_times[i] for i in dither])
        selected, selected_pts = _decode_selected_frames(source, pts, time_base, pixel_hashes,
                                                         set(base) | set(dither), pixel_spec)
        after = _file_sha256(source)
        if before != after:
            raise ValueError("video_input_changed_during_decode")
    rows = {}
    for name, indices, rate, dev in (("baseline", base, base_rate, base_dev),
                                     ("dither", dither, dith_rate, dith_dev)):
        rows[name] = {"frames": [selected[i] for i in indices],
                      "pts": [rel_times[i] for i in indices],
                      "indices": indices,
                      "rate": rate,
                      "max_deviation": dev,
                      "sampled_rgb_hash": hashlib.sha256("".join(_image_sha256(selected[i]) for i in indices).encode()).hexdigest()}
    pts_blob = json.dumps([[p, time_base.numerator, time_base.denominator] for p in pts], separators=(",", ":")).encode()
    return rows, {"source_hash": before, "pts_hash": hashlib.sha256(pts_blob).hexdigest(),
                  "rgb_hash": hashlib.sha256("".join(pixel_hashes).encode()).hexdigest(),
                  "decoder_version": str(av.__version__), "stream_info": stream_info}


def _execute(args):
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from run_nextgqa_real_video_gpu_smoke import choose_videos
    from smoke_qwen_real_video_attention_boost import _make_batch, _temporal_visual_groups, question_for_video
    from qwen_question_temporal_attention_selector import map_question_token_positions, select_top_temporal_groups
    from qta_attention_forward_capture import capture_qta_attention_forward_once, aggregate_captured_qta_rows
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    count = 1 if args.smoke else RECORDS
    STATE["requested_videos"], STATE["requested_forwards"] = count, count * 2
    source_hashes_before = source_hash_receipt(repo)
    STATE["source_hashes_before"] = source_hashes_before
    exclusions_before = exclusion_manifest_receipt(repo)
    STATE["exclusion_manifests"] = exclusions_before
    candidates = choose_videos(repo, dataset)
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    examples = []
    requested_count = 1 if args.smoke else RECORDS
    for video in candidates[:requested_count]:
        STATE["inspected"] += 1
        STATE["attempted_videos"] += 1
        rows, media = _decode_two_grids(video)
        question = question_for_video(dataset, video, video_root)
        examples.append((video, rows, media, question))
    if len(examples) != requested_count:
        raise RuntimeError("fixed_candidate_cohort_unavailable")
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / MODEL_REVISION
    if not model_path.is_dir():
        raise RuntimeError("pinned_checkpoint_not_cached")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_bf16_unavailable")
    device = torch.cuda.current_device()
    free_before, total = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise RuntimeError("insufficient_free_vram_before_preflight")
    hash_started = time.monotonic()
    model_snapshot_sha256_before = _snapshot_sha256(model_path)
    model_snapshot_hash_seconds = time.monotonic() - hash_started
    STATE["model_snapshot_before"] = model_snapshot_sha256_before
    STATE["model_snapshot_hash_before_seconds"] = model_snapshot_hash_seconds
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    prepared = []
    patch = int(processor.image_processor.temporal_patch_size)
    with (model_path / "config.json").open(encoding="utf-8") as config_stream:
        config_temporal_patch = int(json.load(config_stream)["vision_config"]["temporal_patch_size"])
    if config_temporal_patch != patch:
        raise RuntimeError("model_config_processor_temporal_patch_disagreement")
    for video, rows, media, question in examples:
        pair = {}
        for condition in ("baseline", "dither"):
            row = rows[condition]
            batch, metadata, rendered = _make_batch(frames=row["frames"], sample_fps=row["rate"],
                question=question, processor=processor, torch_module=torch, frame_count=FRAME_COUNT)
            if metadata.get("frames_indices") != list(range(FRAME_COUNT)) or abs(float(metadata.get("fps", 0)) - float(row["rate"])) > 1e-7:
                raise RuntimeError("processor_frame_or_fps_contract_invalid")
            steps, gh, gw = map(int, batch["video_grid_thw"][0].tolist())
            if steps * patch != FRAME_COUNT:
                raise RuntimeError("processor_temporal_patch_contract_invalid")
            ids = batch["input_ids"]
            qpos = map_question_token_positions(tokenizer=processor.tokenizer, rendered_text=rendered,
                question_text=question, processed_input_ids=ids, video_token_id=int(processor.video_token_id))
            groups = _temporal_visual_groups(token_ids=[int(x) for x in ids[0].tolist()],
                video_token_id=int(processor.video_token_id), temporal_steps=steps, grid_h=gh, grid_w=gw,
                merge_size=int(processor.image_processor.merge_size))
            from nextgqa_atomic_batch_freeze import batch_sha256
            input_hash = batch_sha256(batch)
            batch = batch.to("cpu")
            pair[condition] = {"batch": batch, "batch_hash": input_hash,
                "qpos": qpos, "groups": groups, "steps": steps,
                "ids_hash": hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                "grid": [steps, gh, gw]}
        a, b = pair["baseline"], pair["dither"]
        if (not torch.equal(a["batch"]["input_ids"], b["batch"]["input_ids"])
                or not torch.equal(a["batch"]["attention_mask"], b["batch"]["attention_mask"])
                or a["grid"] != b["grid"] or a["steps"] != b["steps"]
                or rows["baseline"]["rate"] != rows["dither"]["rate"]
                or a["batch"]["pixel_values_videos"].shape != b["batch"]["pixel_values_videos"].shape
                or a["batch"]["pixel_values_videos"].dtype != b["batch"]["pixel_values_videos"].dtype):
            raise RuntimeError("processor_condition_compatibility_failed")
        hashes = {name: entry["batch_hash"] for name, entry in pair.items()}
        for condition in ("baseline", "dither"):
            rows[condition].pop("frames", None)
        prepared.append((pair, rows, media, hashes, question))
        del pair
        torch.cuda.empty_cache()
    examples.clear()
    free_before, total = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise RuntimeError("insufficient_free_vram_before_model_load")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.vision_config.temporal_patch_size) != patch:
        raise RuntimeError("model_processor_temporal_patch_disagreement")
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    metrics, batch_hashes, capture_receipts = [], [], []
    max_capture_deviation = 0.0
    from nextgqa_atomic_batch_freeze import batch_sha256
    processing_start = time.monotonic()
    for ordinal, (pair, rows, media, preflight_hashes, question) in enumerate(prepared, 1):
        selected = {}
        intervals = {}
        centers = {}
        for condition in ("baseline", "dither"):
            entry = pair[condition]
            batch = entry["batch"].to("cuda")
            free, _ = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("insufficient_free_vram_before_forward")
            if batch_sha256(batch) != preflight_hashes[condition]:
                raise RuntimeError("processor_batch_changed_after_preflight")
            batch_hashes.append(preflight_hashes[condition])
            def counted_forward(batch=batch):
                STATE["attempted_forwards"] += 1
                result = model(**batch, use_cache=False, logits_to_keep=1)
                STATE["completed_forwards"] += 1
                return result
            _, captured, receipt = capture_qta_attention_forward_once(model=model,
                forward=counted_forward,
                question_positions=entry["qpos"], input_length=int(batch["input_ids"].shape[1]), torch_module=torch)
            deviation = validate_capture_receipt(receipt)
            max_capture_deviation = max(max_capture_deviation, deviation)
            capture_receipts.append({"forward_calls": receipt["forward_calls"],
                "layer_calls_all_once": all(value == 1 for value in receipt["layer_calls"].values()),
                "layers_captured": len(receipt["layer_calls"]),
                "rows_normalized_within_dtype_tolerance": receipt["attention_rows_satisfy_source_dtype_normalization_tolerance"]})
            scores = aggregate_captured_qta_rows(rows=captured, group_positions=entry["groups"], torch_module=torch)
            selected[condition] = select_top_temporal_groups(scores, top_k=max(1, math.ceil(.10 * entry["steps"])))
            pts = rows[condition]["pts"]
            intervals[condition], centers[condition] = support_intervals(pts, entry["steps"], patch)
            del captured
            torch.cuda.empty_cache()
        base_sel, dith_sel = selected["baseline"], selected["dither"]
        ordinal_jaccard = group_set_jaccard(base_sel, dith_sel)
        actual_jaccard = interval_union_jaccard([intervals["baseline"][i] for i in base_sel],
                                                 [intervals["dither"][i] for i in dith_sel])
        w1 = empirical_wasserstein1([centers["baseline"][i] for i in base_sel],
                                    [centers["dither"][i] for i in dith_sel])
        metrics.append({"fixed_video_ordinal": ordinal, "group_index_set_jaccard": ordinal_jaccard,
                        "actual_pts_support_jaccard": actual_jaccard, "selected_center_wasserstein1": w1,
                        "baseline_actual_pts_max_deviation_seconds": [rows["baseline"]["max_deviation"].numerator,
                                                                         rows["baseline"]["max_deviation"].denominator],
                        "dither_actual_pts_max_deviation_seconds": [rows["dither"]["max_deviation"].numerator,
                                                                      rows["dither"]["max_deviation"].denominator]})
        STATE["effective_videos"] += 1
    source_hashes_after = source_hash_receipt(repo)
    STATE["source_hashes_after"] = source_hashes_after
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source_files_changed_during_run")
    after_hash_started = time.monotonic()
    model_snapshot_sha256_after = _snapshot_sha256(model_path)
    model_snapshot_after_hash_seconds = time.monotonic() - after_hash_started
    STATE["model_snapshot_after"] = model_snapshot_sha256_after
    STATE["model_snapshot_hash_after_seconds"] = model_snapshot_after_hash_seconds
    if model_snapshot_sha256_after != model_snapshot_sha256_before:
        raise RuntimeError("pinned_snapshot_changed_during_run")
    exclusions = exclusion_manifest_receipt(repo)
    if exclusions != exclusions_before:
        raise RuntimeError("candidate_exclusion_manifests_changed_during_run")
    forward_receipt = forward_count_receipt(count*2, STATE["attempted_forwards"], STATE["completed_forwards"])
    video_receipt = video_pair_count_receipt(count, STATE["attempted_videos"], STATE["effective_videos"])
    payload = {"status": "passed", "scope": scope_name(args.smoke),
        "started_at_utc": STATE["started_at"], "source_sha256": source_hashes_before["files"],
        "source_sha256_digest": source_hashes_before["digest_sha256"],
        "source_hashes_rechecked_after_forwards": source_hashes_after["digest_sha256"],
        "model_snapshot_sha256_before_load": model_snapshot_sha256_before,
        "model_snapshot_sha256_after_forwards": model_snapshot_sha256_after,
        "model_snapshot_hash_before_seconds": model_snapshot_hash_seconds,
        "model_snapshot_hash_after_seconds": model_snapshot_after_hash_seconds,
        "model_revision": MODEL_REVISION, "model_config_sha256": _sha(model_path / "config.json"),
        "processor_revision": MODEL_REVISION, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device), "precision": "bf16_weights_4bit_bf16_compute",
        "dataset_split": "NExT-GQA test", "dataset_input_sha256": {
            name: _sha(dataset / name) for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json")},
        "candidate_selection": "same first three deterministic choose_videos; fixed cohort, no replacement",
        "candidate_exclusion_manifests": exclusions,
        **video_receipt,
        "requested_videos": count, "effective_videos": len(metrics),
        **forward_receipt, "frame_count": FRAME_COUNT,
        "dither_rule": "endpoint-inclusive baseline; odd 1-based interior sample slots +1 decoded index; even interior slots unchanged; endpoints fixed",
        "uniform_actual_pts_tolerance_seconds": [1,20], "processor_temporal_patch_size": patch,
        "selector": "untreated QTA question-token attention, layers 5-16, top ceil(0.10*T) complete groups",
        "primary_statistic": "mean fixed-video ordinal group-index-set Jaccard; ordinal slots are same patch partition",
        "primary_mean_group_index_set_jaccard": sum(r["group_index_set_jaccard"] for r in metrics)/len(metrics),
        "supplemental_mean_actual_pts_support_jaccard": sum(r["actual_pts_support_jaccard"] for r in metrics)/len(metrics),
        "supplemental_mean_selected_center_wasserstein1": sum(r["selected_center_wasserstein1"] for r in metrics)/len(metrics),
        "fixed_pair_metrics": metrics, "processor_input_batch_sha256": batch_hashes,
        "capture_receipt_aggregate": {"successful_captures": len(capture_receipts),
            "all_one_forward_each_layer_once": all(r["forward_calls"] == 1 and r["layer_calls_all_once"] for r in capture_receipts),
            "all_rows_normalized": all(r["rows_normalized_within_dtype_tolerance"] for r in capture_receipts),
            "maximum_row_sum_deviation": max_capture_deviation},
        "input_hash_aggregates": {
            key: hashlib.sha256("".join(sorted(media[key] for _, _, media, _, _ in prepared)).encode()).hexdigest()
            for key in ("source_hash", "pts_hash", "rgb_hash")},
        "sampled_media_hash_aggregates": {
            condition: hashlib.sha256("".join(sorted(rows[condition]["sampled_rgb_hash"]
                for _, rows, _, _, _ in prepared)).encode()).hexdigest()
            for condition in ("baseline", "dither")},
        "random_seed": RANDOM_SEED, "elapsed_attention_analysis_seconds": time.monotonic()-processing_start,
        "elapsed_seconds": time.monotonic()-STATE["started_monotonic"],
        "free_vram_before_model_load_gib": free_before/1024**3, "total_vram_gib": total/1024**3,
        "faithfulness_result": False, "rights_status": "unverified_local_only", "local_only": True,
        "limitations": ["only fixed first three candidates; no replacement", "ordinal group stability does not establish temporal evidence validity", "actual support changes with sampled PTS", "sampling-density and visual-tokenization changes are confounded", "no threshold or tuning", "attention selection does not establish causal or explanation faithfulness"],
        "ids_questions_answers_selected_group_indices_raw_logits_or_video_rows_saved": False}
    _write_private(args.output, payload)
    print(json.dumps({k: payload[k] for k in ("status", "requested_video_pairs", "attempted_video_pairs",
        "completed_video_pairs", "not_attempted_video_pairs", "requested_forwards", "completed_forwards",
        "primary_mean_group_index_set_jaccard", "faithfulness_result")}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private local JSON path; overwrite refused")
    parser.add_argument("--smoke", action="store_true", help="one fixed candidate, two forwards")
    args = parser.parse_args(argv)
    STATE.update(attempted_forwards=0, completed_forwards=0, attempted_videos=0,
        effective_videos=0, inspected=0,
        requested_videos=1 if args.smoke else RECORDS, requested_forwards=(1 if args.smoke else RECORDS)*2,
        smoke=bool(args.smoke), started_at=datetime.now(timezone.utc).isoformat(), started_monotonic=time.monotonic(),
        exclusion_manifests=None, model_snapshot_before=None, model_snapshot_after=None,
        model_snapshot_hash_before_seconds=None, model_snapshot_hash_after_seconds=None,
        source_hashes_before=None, source_hashes_after=None)
    try:
        _execute(args)
        return 0
    except Exception as exc:
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"local_run_failed_{type(exc).__name__}"
        requested, forwards = STATE["requested_videos"], STATE["requested_forwards"]
        forward_receipt = forward_count_receipt(forwards, STATE["attempted_forwards"], STATE["completed_forwards"])
        video_receipt = video_pair_count_receipt(requested, STATE["attempted_videos"], STATE["effective_videos"])
        failure = {"status": "failed", "scope": scope_name(STATE["smoke"]),
            "smoke_only": STATE["smoke"],
            "started_at_utc": STATE["started_at"], "failure_code": code,
            **video_receipt, "effective_videos": STATE["effective_videos"],
            "candidate_videos_inspected": STATE["inspected"],
            **forward_receipt,
            "source_sha256_at_start": STATE["source_hashes_before"],
            "source_sha256_after_forwards": STATE["source_hashes_after"],
            "candidate_exclusion_manifests_at_start": STATE["exclusion_manifests"],
            "model_snapshot_sha256_before_load": STATE["model_snapshot_before"],
            "model_snapshot_sha256_after_forwards": STATE["model_snapshot_after"],
            "model_snapshot_hash_before_seconds": STATE["model_snapshot_hash_before_seconds"],
            "model_snapshot_hash_after_seconds": STATE["model_snapshot_hash_after_seconds"],
            "elapsed_seconds": time.monotonic()-STATE["started_monotonic"],
            "faithfulness_result": False, "local_only": True, "rights_status": "unverified_local_only"}
        try:
            _write_private(args.output, failure)
            artifact = "written"
        except Exception:
            artifact = "unavailable_output_path_unsafe_or_exists"
        print(json.dumps({"status": "failed", "failure_code": code, "failure_artifact": artifact,
            **video_receipt, "effective_videos": STATE["effective_videos"],
            **forward_receipt}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
