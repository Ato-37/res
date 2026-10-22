#!/usr/bin/env python3
"""Score fixed claim dependence on reviewed frames under QTA (local exploratory).

The full mode requires a response bound to the exact local review artifact.
The synthetic smoke exercises the GPU code path with a fabricated frame-group
mapping and is never a human-review or faithfulness result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_nextgqa"
PACKET_REL = Path("data/private_nextgqa/claim_generation_cohort11_gpu_v3_20261021_02.json")
ARTIFACT_REL = Path("data/private_nextgqa/claim_support_frame_review_cohort11_v3_20261021_08.html")
LOCK_REL = Path("data/private_nextgqa/claim_cohort_lock_20261021_11.json")
PAIR_FILTER_REL = Path("data/private_nextgqa/claim_cohort_pair_filter_20261021_11.json")
EXCLUSION_REL = Path("data/private_nextgqa/claim_cohort_media_exclusions_20261021_11.json")
TRANSLATION_REL = Path("data/private_nextgqa/test_split_internal_bias_review_translations_20261021_01.json")
FRAME_COUNT = 16
TEMPORAL_PATCH_SIZE = 2
GROUP_COUNT = FRAME_COUNT // TEMPORAL_PATCH_SIZE
LAYERS = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
SOURCE_FILES = (
    "scripts/run_claim_support_qta_16frame_exploratory.py",
    "scripts/adapt_claim_support_frame_review_response.py",
    "scripts/inspect_claim_support_frame_review_response.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/run_qta_intervened_interval_generation.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/run_claim_text_qa_interval_key_bias.py",
    "scripts/run_saved_claim_frame_permutation_attention_smoke.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/build_test_split_claim_support_frame_review.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/generate_nextgqa_claim_review_pilot.py",
    "scripts/run_saved_claim_video_content_dependence.py",
    "scripts/private_output_paths.py",
    "scripts/presentation_time_utils.py",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_manifest(model_path: Path) -> dict:
    package_versions = {}
    for distribution in ("torch", "torchvision", "transformers", "bitsandbytes",
                         "qwen-vl-utils", "Pillow", "av", "numpy"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    files = sorted((path.relative_to(model_path).as_posix(), path.stat().st_size)
                   for path in model_path.rglob("*") if path.is_file())
    return {
        "python_version": os.sys.version.split()[0],
        "package_versions": package_versions,
        "source_sha256": {relative: _sha_file(ROOT / relative) for relative in SOURCE_FILES},
        "checkpoint_config_sha256": _sha_file(model_path / "config.json"),
        "checkpoint_snapshot_inventory_sha256": _sha(json.dumps(
            files, separators=(",", ":")).encode("utf-8")),
        "checkpoint_snapshot_inventory_file_count": len(files),
        "checkpoint_snapshot_inventory_semantics": "relative file names and sizes; model revision pinned separately; weight bytes not rehashed",
    }


def _read_private(path: Path, *, max_bytes: int = 50_000_000) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > max_bytes):
            raise ValueError("private_input_contract_invalid")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("private_input_too_large")
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_bound_response(path: Path, *, max_bytes: int = 2_000_000) -> bytes:
    """Read a user-supplied local response safely, including Windows Downloads files.

    Input permissions vary on drvfs; output permissions remain strictly private.
    The response is still bounded, no-follow, regular-file-only, and checked for
    mutation while read. Its exact artifact fingerprint is validated downstream.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("bound_response_file_contract_invalid")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(256 * 1024, max_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("bound_response_too_large")
            chunks.append(block)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or total != before.st_size):
            raise ValueError("bound_response_changed_while_reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def select_control_groups(evidence: tuple[int, ...], qta: tuple[int, ...],
                          group_count: int = GROUP_COUNT) -> tuple[int, ...]:
    """Equal-count control, disjoint from support and QTA, minimum rank distance."""
    if (type(group_count) is not int or group_count < 2
            or not evidence or tuple(sorted(set(evidence))) != evidence
            or not qta or tuple(sorted(set(qta))) != qta
            or any(type(i) is not int or not 0 <= i < group_count
                   for i in (*evidence, *qta))):
        raise ValueError("group_mapping_invalid")
    forbidden = set(evidence) | set(qta)
    candidates = [i for i in range(group_count) if i not in forbidden]
    k = len(evidence)
    if len(candidates) < k:
        raise ValueError("matched_control_unavailable")
    # Dynamic programming over sorted candidate groups; deterministic tie break.
    dp: list[tuple[int, tuple[int, ...]] | None] = [None] * (k + 1)
    dp[0] = (0, ())
    for candidate in candidates:
        for count in range(k, 0, -1):
            prev = dp[count - 1]
            if prev is None:
                continue
            proposal = (prev[0] + abs(evidence[count - 1] - candidate),
                        prev[1] + (candidate,))
            if dp[count] is None or proposal < dp[count]:
                dp[count] = proposal
    if dp[k] is None:
        raise ValueError("matched_control_unavailable")
    return dp[k][1]


def compute_selective_dependence(*, baseline_support: float, baseline_control: float,
                                 qta_support: float, qta_control: float) -> dict:
    values = (baseline_support, baseline_control, qta_support, qta_control)
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in values):
        raise ValueError("branch_score_nonfinite")
    d_base = float(baseline_control) - float(baseline_support)
    d_qta = float(qta_control) - float(qta_support)
    return {"D_baseline_nats_per_token": d_base,
            "D_qta_nats_per_token": d_qta,
            "delta_D_qta_minus_baseline_nats_per_token": d_qta - d_base}


def _artifact_parts(artifact_raw: bytes):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from inspect_claim_support_frame_review_response import (
        ITEMS_RE, PROVENANCE_RE, FINGERPRINT_RE,
    )
    items_match, provenance_match, fingerprint_match = (
        ITEMS_RE.search(artifact_raw), PROVENANCE_RE.search(artifact_raw),
        FINGERPRINT_RE.search(artifact_raw))
    if not (items_match and provenance_match and fingerprint_match):
        raise ValueError("artifact_manifest_missing")
    return (json.loads(items_match.group(1)), json.loads(provenance_match.group(1)),
            fingerprint_match.group(1).decode("ascii"))


def bind_rows(packet_raw: bytes, artifact_raw: bytes, mapping: dict,
              translation_raw: bytes | None = None) -> list[dict]:
    """Bind mapping to exact packet and all row hashes before any model load."""
    if mapping.get("status") != "mapped_unvalidated" or mapping.get("model_load_permitted") is not False:
        raise ValueError("frame_mapping_not_validated")
    packet = json.loads(packet_raw)
    items, provenance, fingerprint = _artifact_parts(artifact_raw)
    translations = None
    if "translation_sha256" in provenance:
        if translation_raw is None:
            raise ValueError("translation_source_binding_invalid")
        translations = json.loads(translation_raw)
        if (not isinstance(translations, dict) or translations.get("schema_version") != 1
                or not isinstance(translations.get("items"), dict)
                or _sha(translation_raw) != provenance.get("translation_sha256")):
            raise ValueError("translation_source_binding_invalid")
    if (_sha(packet_raw) != provenance.get("packet_sha256")
            or mapping.get("artifact_fingerprint") != fingerprint
            or mapping.get("source_manifest_sha256") != _sha(json.dumps(
                provenance, sort_keys=True, separators=(",", ":")).encode())):
        raise ValueError("artifact_packet_binding_mismatch")
    frozen_sources = ((LOCK_REL, provenance.get("lock_sha256")),
                      (PAIR_FILTER_REL, provenance.get("pair_filter_sha256")),
                      (EXCLUSION_REL, provenance.get("exclusion_manifest_sha256")))
    for relative, expected_sha in frozen_sources:
        if _sha(_read_private(ROOT / relative)) != expected_sha:
            raise ValueError("frozen_cohort_source_hash_mismatch")
    if (packet.get("status") != "passed" or packet.get("dataset_split") != "test"
            or packet.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
            or packet.get("generation_prompt_version") != "v3_sequence_complete"
            or packet.get("frames_per_video") != FRAME_COUNT
            or packet.get("effective_videos") != len(items)
            or len(packet.get("records", [])) != len(items)
            or len(mapping.get("rows", [])) != len(items)):
        raise ValueError("packet_contract_mismatch")
    for name, expected in packet.get("dataset_inputs_sha256", {}).items():
        source_path = ROOT / "NExT-GQA" / "datasets" / "nextgqa" / name
        if (not source_path.is_file() or source_path.is_symlink()
                or _sha_file(source_path) != expected):
            raise ValueError("dataset_source_hash_mismatch")
    source_by_id = {row["item_id"]: row for row in provenance["items"]}
    record_by_index = {index: record for index, record in enumerate(packet["records"])}
    bound = []
    for item, mapped in zip(items, mapping["rows"], strict=True):
        item_id = item["item_id"]
        source = source_by_id.get(item_id)
        index = source.get("record_index") if isinstance(source, dict) else None
        record = record_by_index.get(index)
        claim = record.get("generated_parsed", {}).get("claim") if isinstance(record, dict) else None
        qid = record.get("dataset_question_id") if isinstance(record, dict) else None
        frames = record.get("sampled_frame_relative_pts_seconds") if isinstance(record, dict) else None
        artifact_times = [x.get("time") for x in item.get("frames", [])]
        qid_sha = _sha(str(qid).encode()) if isinstance(qid, str) else None
        candidate_sha = record.get("candidate_pair_sha256") if isinstance(record, dict) else None
        if translations is not None:
            translation_key = (_sha((record["question"] + "\0" + claim).encode("utf-8"))
                               if isinstance(record, dict) and isinstance(claim, str) else None)
            translated = translations["items"].get(translation_key)
            text_binding_ok = (isinstance(translated, dict)
                and translated.get("claim_ja") == item.get("claim")
                and translated.get("question_ja") == item.get("question"))
        else:
            text_binding_ok = (isinstance(claim, str)
                and _sha(claim.encode()) == mapped.get("claim_sha256")
                and isinstance(record.get("question") if record else None, str)
                and _sha(record["question"].encode()) == mapped.get("question_sha256"))
        checks = (
            mapped.get("item_id_sha256") == _sha(item_id.encode()),
            source is not None,
            mapped.get("source_record_index") == index,
            mapped.get("video_sha256") == (record.get("video_sha256") if record else None),
            mapped.get("video_sha256") == (source.get("video_sha256") if source else None),
            mapped.get("qid_sha256") == qid_sha == (source.get("qid_sha256") if source else None),
            mapped.get("candidate_pair_sha256") == candidate_sha == (source.get("candidate_pair_sha256") if source else None),
            text_binding_ok,
            isinstance(frames, list) and len(frames) == FRAME_COUNT
                and len(artifact_times) == FRAME_COUNT
                and all(abs(float(a) - float(b)) <= 1e-6
                        for a, b in zip(frames, artifact_times, strict=True)),
            type(mapped.get("eligible_for_exploratory_score")) is bool,
        )
        if not all(checks):
            raise ValueError("review_packet_row_binding_mismatch")
        video_path = Path(record["local_video_path"])
        if (video_path.is_symlink() or not video_path.is_file()
                or _sha_file(video_path) != record["video_sha256"]):
            raise ValueError("source_video_hash_mismatch")
        bound.append({"item": item, "mapping": mapped, "record": record,
                      "source": source, "record_index": index})
    return bound


def _preflight_media(bound: dict) -> dict:
    """Decode and validate exact 16-frame media before loading model weights."""
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    record = bound["record"]
    video_path = Path(record["local_video_path"])
    frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
        uniform_tolerance_seconds=Fraction(1, 20))
    verify_sampled_frame_pixels(frames, decoded)
    if decoded.get("source_sha256") != record.get("video_sha256"):
        raise ValueError("decoded_video_hash_mismatch")
    pts = [float(Fraction(*row["relative_presentation_time"]))
           for row in decoded["sampled_frames"]]
    saved_pts = record.get("sampled_frame_relative_pts_seconds")
    artifact_pts = [frame.get("time") for frame in bound["item"]["frames"]]
    if (len(frames) != FRAME_COUNT or len(saved_pts) != FRAME_COUNT
            or len(artifact_pts) != FRAME_COUNT
            or any(abs(a - float(b)) > 1e-6
                   for a, b in zip(pts, saved_pts, strict=True))
            or any(abs(a - float(b)) > 1e-6
                   for a, b in zip(pts, artifact_pts, strict=True))):
        raise ValueError("source_or_review_pts_mismatch")
    from build_test_split_claim_support_frame_review import frame_jpeg_data_url
    artifact_frames = bound["item"]["frames"]
    if any(frame_jpeg_data_url(frame) != artifact_frame.get("data")
           for frame, artifact_frame in zip(frames, artifact_frames, strict=True)):
        raise ValueError("artifact_thumbnail_source_frame_mismatch")
    return {"frames": frames, "decoded": decoded, "pts": pts,
            "source_sha256": decoded["source_sha256"]}


def _read_host_available_bytes() -> int:
    from run_saved_claim_frame_permutation_attention_smoke import _host_available_bytes
    return _host_available_bytes()


def _build_batch(processor, process_vision_info, record: dict, frames, fps: float,
                 prompt_suffix: str, masked_group_indices: tuple[int, ...], torch):
    from necessity_choice_margin_utils import apply_mask_fill
    from run_saved_claim_frame_permutation_attention_smoke import _build_prompt
    mask_frames = {i for group in masked_group_indices
                   for i in range(group * TEMPORAL_PATCH_SIZE,
                                  (group + 1) * TEMPORAL_PATCH_SIZE)}
    condition_frames = [apply_mask_fill(frame, "black") if i in mask_frames else frame.copy()
                        for i, frame in enumerate(frames)]
    rendered, messages = _build_prompt(processor, record, condition_frames, fps, prompt_suffix)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    pixels, metadata = videos[0]
    if metadata.get("frames_indices") != list(range(FRAME_COUNT)):
        raise ValueError("processor_frame_indices_changed")
    batch = processor(text=[rendered], images=images, videos=[pixels],
        video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
        padding=True, return_tensors="pt")
    return batch, rendered, metadata


def _video_groups(batch: dict, processor) -> tuple[tuple[int, ...], ...]:
    ids = batch["input_ids"][0].tolist()
    grid = [int(x) for x in batch["video_grid_thw"][0].tolist()]
    temporal, height, width = grid
    patch = int(processor.video_processor.temporal_patch_size)
    merge = int(processor.image_processor.merge_size)
    positions = [i for i, token in enumerate(ids) if token == int(processor.video_token_id)]
    per_group = (height // merge) * (width // merge)
    if (patch != TEMPORAL_PATCH_SIZE or temporal != GROUP_COUNT
            or height % merge or width % merge or len(positions) != temporal * per_group
            or positions != list(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("processor_group_geometry_mismatch")
    return tuple(tuple(positions[i * per_group:(i + 1) * per_group])
                 for i in range(temporal))


def _score_branch(*, model, batch_cpu: dict, target_ids: tuple[int, ...],
                  biased_positions: tuple[int, ...], torch,
                  logit_bias: float = LOGIT_BIAS,
                  ablated_positions: tuple[int, ...] = ()) -> tuple[float, dict]:
    from qta_multiquery_video_key_intervention import (
        append_target_to_unpadded_batch, install_multiquery_video_key_intervention_hooks,
        remove_hooks, score_target_logits_window,
    )
    batch = {key: value.to("cuda") if torch.is_tensor(value) else value
             for key, value in batch_cpu.items()}
    prompt_len = int(batch["input_ids"].shape[1])
    extended, queries = append_target_to_unpadded_batch(
        batch=batch, target_token_ids=target_ids, torch_module=torch)
    visual_positions = tuple((batch["input_ids"][0] == int(model.config.video_token_id))
                             .nonzero(as_tuple=True)[0].detach().cpu().tolist())
    if not visual_positions or not queries or max(visual_positions) >= min(queries):
        raise ValueError("visual_keys_not_causal_for_claim")
    observed = {layer: 0 for layer in LAYERS}
    hooks = (install_multiquery_video_key_intervention_hooks(
        model=model, layer_indices=tuple(layer - 1 for layer in LAYERS),
        biased_positions=biased_positions, ablated_positions=ablated_positions,
        query_positions=queries, logit_bias=logit_bias, torch_module=torch)
        if biased_positions or ablated_positions else [])
    observers = []
    try:
        layers = model.model.language_model.layers
        for layer in LAYERS:
            def audit(_module, args, kwargs, *, layer_i=layer):
                mask = kwargs.get("attention_mask")
                if not torch.is_tensor(mask) or mask.ndim != 4:
                    raise ValueError("attention_mask_audit_missing")
                for q in queries:
                    for key in biased_positions:
                        if not math.isclose(float(mask[0, 0, q, key].item()), logit_bias,
                                rel_tol=0, abs_tol=2e-3):
                            raise ValueError("bias_mask_not_observed")
                    for key in ablated_positions:
                        if not bool(torch.isneginf(mask[0, 0, q, key])):
                            raise ValueError("ablation_mask_not_observed")
                observed[layer_i] += 1
                return args, kwargs
            observers.append(layers[layer - 1].self_attn.register_forward_pre_hook(
                audit, with_kwargs=True))
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024**3 or _read_host_available_bytes() < 3 * 1024**3:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            output = model(**extended, use_cache=False,
                           logits_to_keep=len(target_ids) + 1)
        if observed != {layer: 1 for layer in LAYERS}:
            raise ValueError("attention_hook_count_mismatch")
        scores = score_target_logits_window(logits=output.logits,
            target_token_ids=target_ids, prompt_length=prompt_len,
            first_query_position=prompt_len - 1, torch_module=torch)
        del output
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return float(scores["target_logprob_mean"]), {
            "target_token_count": len(target_ids), "target_token_ids_sha256": _sha(
                json.dumps(target_ids, separators=(",", ":")).encode("ascii")),
            "biased_key_count": len(biased_positions),
            "deleted_key_count": len(ablated_positions),
            "logit_bias_nats": logit_bias,
            "intervention": ("QTA visual-key bias and/or ablation" if
                biased_positions and ablated_positions else
                "QTA visual-key logit bias" if biased_positions else
                "QTA visual-key ablation" if ablated_positions else "none"),
            "target_query_count": len(queries), "layer_hook_calls": {str(k): v for k, v in observed.items()},
        }
    finally:
        for handle in reversed(observers):
            handle.remove()
        if hooks:
            remove_hooks(hooks)
        del extended, batch
        torch.cuda.empty_cache()


def _run_one(bound: dict, media: dict, *, processor, process_vision_info, model, torch,
             synthetic: bool, counters: dict) -> dict:
    from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
    from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
    from run_qta_intervened_interval_generation import _attention_selector
    from qwen_question_temporal_attention_selector import select_top_temporal_groups

    record, mapping = bound["record"], bound["mapping"]
    frames, decoded, pts = media["frames"], media["decoded"], media["pts"]
    prompt_version, suffix = resolve_claim_prompt_suffix(record, PROMPT_SUFFIX_BY_VERSION)
    fps = float(Fraction(*decoded["effective_sample_fps"]))
    full_batch, rendered, metadata = _build_batch(processor, process_vision_info,
        record, frames, fps, suffix, (), torch)
    group_positions = _video_groups(full_batch, processor)
    counters["attempted"] += 1
    question_scores, selector_info = _attention_selector(
        model, processor, rendered, record["question"].strip(), full_batch,
        list(group_positions), torch)
    counters["completed"] += 1
    top_count = max(1, math.ceil(GROUP_COUNT * 0.10))
    qta_groups = tuple(sorted(select_top_temporal_groups(question_scores, top_k=top_count)))
    if synthetic:
        support_groups = (1,)
    else:
        support_groups = tuple(sorted({i // TEMPORAL_PATCH_SIZE
            for i in mapping["selected_frame_indices_zero_based"]}))
    from run_claim_text_qa_interval_key_bias import map_groups_to_visual_keys
    qta_positions = tuple(sorted(p for i in qta_groups for p in group_positions[i]))
    # First map validates exact processor key geometry and equal visual-key budget.
    empty_control = tuple(i for i in range(GROUP_COUNT)
                          if i not in set(support_groups) | set(qta_groups))
    if len(empty_control) < len(support_groups):
        raise ValueError("matched_control_unavailable")
    control_groups = select_control_groups(support_groups, qta_groups, GROUP_COUNT)
    input_ids = full_batch["input_ids"][0].detach().cpu().tolist()
    grid = [int(x) for x in full_batch["video_grid_thw"][0].detach().cpu().tolist()]
    group_map = map_groups_to_visual_keys(input_ids=input_ids,
        video_token_id=int(processor.video_token_id), video_grid_thw=grid,
        spatial_merge_size=int(processor.image_processor.merge_size),
        temporal_patch_size=TEMPORAL_PATCH_SIZE, frame_count=FRAME_COUNT,
        evidence_group_indices=support_groups, control_group_indices=control_groups)
    control_positions = group_map["control_key_positions"]
    support_positions = group_map["evidence_key_positions"]
    if (set(qta_groups) & set(control_groups) or len(qta_positions) == 0
            or len(support_positions) != len(control_positions)):
        raise ValueError("key_budget_or_overlap_invalid")
    target_ids = tuple(processor.tokenizer(
        record["generated_parsed"]["claim"].strip(), add_special_tokens=False)["input_ids"])
    if not target_ids or len(target_ids) > 512 or any(type(i) is not int or i < 0 for i in target_ids):
        raise ValueError("frozen_claim_tokenization_invalid")
    batch_by_mask = {}
    for name, groups in (("support", support_groups), ("control", control_groups)):
        batch, masked_rendered, masked_metadata = _build_batch(processor,
            process_vision_info, record, frames, fps, suffix, groups, torch)
        if (masked_rendered != rendered or masked_metadata.get("video_grid_thw")
                != metadata.get("video_grid_thw") or
                not torch.equal(batch["input_ids"], full_batch["input_ids"])):
            raise ValueError("mask_branch_prompt_or_geometry_changed")
        if not torch.equal(full_batch["video_grid_thw"], batch["video_grid_thw"]):
            raise ValueError("mask_branch_video_grid_changed")
        for key in full_batch:
            if key == "pixel_values_videos":
                continue
            left, right = full_batch[key], batch[key]
            if torch.is_tensor(left) and (not torch.is_tensor(right) or not torch.equal(left, right)):
                raise ValueError("mask_branch_nonvisual_input_changed")
        batch_by_mask[name] = batch
    scores, calls = {}, {}
    for condition, bias_positions in (("baseline", ()), ("qta", qta_positions)):
        for mask in ("support", "control"):
            name = f"{condition}_{mask}"
            counters["attempted"] += 1
            value, detail = _score_branch(model=model, batch_cpu=batch_by_mask[mask],
                target_ids=target_ids, biased_positions=bias_positions, torch=torch)
            counters["completed"] += 1
            scores[name] = value
            calls[name] = detail
    primary = compute_selective_dependence(
        baseline_support=scores["baseline_support"],
        baseline_control=scores["baseline_control"],
        qta_support=scores["qta_support"], qta_control=scores["qta_control"])
    return {
        "source_record_index": bound["record_index"],
        "video_sha256": record["video_sha256"],
        "question_sha256": _sha(record["question"].encode()),
        "claim_sha256": _sha(record["generated_parsed"]["claim"].strip().encode()),
        "reviewer_count": mapping["reviewer_count"],
        "review_decision_counts": mapping["decision_counts"],
        "scoring_eligible_by_review": mapping["eligible_for_exploratory_score"],
        "synthetic_support_mapping": synthetic,
        "frame_count": FRAME_COUNT, "sampled_pts_sha256": _sha(
            json.dumps(pts, separators=(",", ":")).encode()),
        "generation_prompt_version": prompt_version,
        "claim_target_semantics": "deterministic re-tokenization of saved parsed claim; original generation IDs unavailable",
        "target_token_count": len(target_ids),
        "target_token_ids_sha256": _sha(json.dumps(target_ids,
            separators=(",", ":")).encode("ascii")),
        "support_frame_indices_zero_based": (None if synthetic else
            mapping["selected_frame_indices_zero_based"]),
        "support_group_indices": list(support_groups),
        "control_group_indices": list(control_groups),
        "qta_group_count": GROUP_COUNT, "qta_selected_group_indices": list(qta_groups),
        "qta_key_count": len(qta_positions),
        "support_key_count": len(support_positions), "control_key_count": len(control_positions),
        "qta_support_group_overlap": sorted(set(qta_groups) & set(support_groups)),
        "score_mean_logprob_nats_per_token": scores,
        "primary": primary, "calls": calls,
        "attention_selector": selector_info,
        "forward_count": 5,
    }


def _write_private(output: Path, payload: dict):
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", type=Path, default=ROOT / ARTIFACT_REL)
    parser.add_argument("--packet", type=Path, default=ROOT / PACKET_REL)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--synthetic-frame-group", type=int, default=1,
                        help="smoke only: fabricated group index, never human evidence")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("local_only_authorization_required")
    if args.mode == "full" and args.response is None:
        parser.error("full_mode_requires_artifact_bound_response")
    if args.mode == "smoke" and not 0 <= args.synthetic_frame_group < GROUP_COUNT:
        parser.error("synthetic_frame_group_out_of_range")
    output = validate_private_output(args.output, ROOT)
    if output.parent != PRIVATE or output.suffix != ".json":
        raise ValueError("output_must_be_new_private_json")
    started = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    attempted = completed = 0
    stage = "private_artifact_and_packet_binding"
    try:
        from adapt_claim_support_frame_review_response import adapt
        artifact_raw = _read_private(args.artifact)
        packet_raw = _read_private(args.packet)
        if args.mode == "full":
            response = json.loads(_read_bound_response(args.response))
            mapping = adapt(response, artifact_raw)
        else:
            from inspect_claim_support_frame_review_response import inspect
            items, _provenance, fingerprint = _artifact_parts(artifact_raw)
            smoke_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            smoke_response = {"schema_version": 1,
                "purpose": "local_claim_support_frame_review",
                "status": "human_responses_self_reported_unvalidated",
                "artifact_fingerprint": fingerprint,
                "created_at_utc": smoke_time,
                "reviewers": [{"nickname": "synthetic-smoke", "answers": [
                    {"item_id": item["item_id"], "decision": "supports",
                     "frame_indices": [args.synthetic_frame_group * TEMPORAL_PATCH_SIZE]}
                    for item in items]}],
                "limitations": ["synthetic only", "not a human review", "not a score"]}
            artifact_check = inspect(smoke_response, artifact_raw)
            if artifact_check.get("structurally_valid") is not True:
                raise ValueError("artifact_self_validation_failed")
            mapping = {"status": "mapped_unvalidated", "model_load_permitted": False,
                "artifact_fingerprint": fingerprint,
                "source_manifest_sha256": _sha(json.dumps(
                    _artifact_parts(artifact_raw)[1], sort_keys=True,
                    separators=(",", ":")).encode()),
                "rows": [{"item_id_sha256": _sha(item["item_id"].encode()),
                    "source_record_index": index,
                    "video_sha256": _artifact_parts(artifact_raw)[1]["items"][index]["video_sha256"],
                    "qid_sha256": _artifact_parts(artifact_raw)[1]["items"][index]["qid_sha256"],
                    "candidate_pair_sha256": _artifact_parts(artifact_raw)[1]["items"][index]["candidate_pair_sha256"],
                    "question_sha256": _sha(item["question"].encode()),
                    "claim_sha256": _sha(item["claim"].encode()),
                    "reviewer_count": 0, "decision_counts": {"supports":0,"contradicts":0,
                        "not_visible_in_model_frames":0,"cannot_judge":0},
                    "eligible_for_exploratory_score": False,
                    "selected_frame_indices_zero_based": []}
                    for index, item in enumerate(items)]}
        translation_raw = (_read_private(ROOT / TRANSLATION_REL)
                           if "translation_sha256" in _artifact_parts(artifact_raw)[1] else None)
        bound = bind_rows(packet_raw, artifact_raw, mapping, translation_raw)
        selected = bound[:1] if args.mode == "smoke" else bound
        if args.mode == "full" and len(selected) != 3:
            raise ValueError("full_cohort_size_invalid")
        eligible = [row for row in selected if args.mode == "smoke"
                    or row["mapping"]["eligible_for_exploratory_score"]]
        ineligible_rows = [row for row in selected if row not in eligible]
        preflight_media = {}
        stage = "cpu_media_preflight"
        for row in eligible:
            preflight_media[row["record_index"]] = _preflight_media(row)
        if args.mode == "full" and not eligible:
            payload = {"schema_version": 1, "status": "no_eligible_human_support_rows",
                "purpose": "direct claim-frame measurement blocked before model load",
                "direct_claim_frame_dependence_measured": False,
                "human_review_performed": True, "requested_rows": len(bound),
                "effective_eligible_rows": 0, "attempted_forwards": 0,
                "completed_forwards": 0, "artifact_sha256": _sha(artifact_raw),
                "packet_sha256": _sha(packet_raw), "rights_status": "unverified; local-only",
                "records": [{"record_index": row["record_index"],
                             "review_decision_counts": row["mapping"]["decision_counts"],
                             "status": "review_ineligible"} for row in ineligible_rows]}
            _write_private(output, payload)
            print(json.dumps({"status": payload["status"], "requested_rows": len(bound),
                "effective_eligible_rows": 0, "completed_forwards": 0}, sort_keys=True))
            return 2
        stage = "cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_available = _read_host_available_bytes()
        if free < MIN_FREE_VRAM or host_available < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / "cc594898137f460bfe9f0759e9844b3ce807cfb5")
        if not model_path.is_dir():
            raise ValueError("pinned_checkpoint_missing")
        runtime_manifest = _runtime_manifest(model_path)
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
        from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_saved_claim_frame_permutation_attention_smoke import _host_available_bytes
        if _host_available_bytes() < MIN_HOST_AVAILABLE:
            raise RuntimeError("host_memory_guard_failed")
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261022); torch.cuda.manual_seed_all(20261022)
        torch.cuda.reset_peak_memory_stats()
        counters = {"attempted": 0, "completed": 0}
        records = [{"source_record_index": row["record_index"],
                    "status": "review_ineligible",
                    "review_decision_counts": row["mapping"]["decision_counts"],
                    "scoring_eligible_by_review": False} for row in ineligible_rows]
        for row in eligible:
            try:
                records.append(_run_one(row, preflight_media[row["record_index"]],
                    processor=processor, process_vision_info=process_vision_info,
                    model=model, torch=torch, synthetic=args.mode == "smoke",
                    counters=counters))
            except Exception as row_error:
                records.append({"source_record_index": row["record_index"],
                    "status": "row_failed", "failure_code": str(row_error)[:160],
                    "scoring_eligible_by_review": row["mapping"]["eligible_for_exploratory_score"]})
        records.sort(key=lambda row: row.get("source_record_index", -1))
        attempted, completed = counters["attempted"], counters["completed"]
        scored_rows = sum("primary" in row for row in records)
        all_complete = scored_rows == len(eligible) and completed == 5 * len(eligible)
        delta_values = [row["primary"]["delta_D_qta_minus_baseline_nats_per_token"]
                        for row in records if "primary" in row]
        descriptive_mean_delta = sum(delta_values) / len(delta_values) if delta_values else None
        payload = {
            "schema_version": 1,
            "status": ("completed_synthetic_mapping_smoke" if args.mode == "smoke" and all_complete
                       else "completed_exploratory_claim_support_qta" if all_complete and len(eligible) == 3
                       else "completed_partial_exploratory_claim_support_qta" if all_complete
                       else "incomplete"),
            "purpose": "synthetic GPU wiring smoke; no human support labels" if args.mode == "smoke"
                       else "post-hoc exploratory reviewed-claim support dependence",
            "direct_faithfulness": False,
            "direct_claim_frame_dependence_measured": args.mode == "full" and scored_rows > 0,
            "faithfulness_result": False,
            "synthetic_support_mapping": args.mode == "smoke",
            "human_review_performed": args.mode == "full",
            "human_review_authenticated": False,
            "rights_status": "unverified; local-only per user authorization",
            "started_utc": started, "elapsed_seconds": time.monotonic() - start_time,
            "packet_sha256": _sha(packet_raw),
            "artifact_sha256": _sha(artifact_raw),
            "artifact_fingerprint": mapping["artifact_fingerprint"],
            "source_manifest_sha256": mapping["source_manifest_sha256"],
            "requested_rows": 1 if args.mode == "smoke" else len(bound),
            "effective_scored_rows": scored_rows,
            "eligible_rows": len(eligible), "ineligible_rows_retained": len(ineligible_rows),
            "requested_forwards": 5 * len(eligible), "attempted_forwards": attempted,
            "completed_forwards": completed,
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "precision": "4-bit weights; BF16 compute; eager attention",
            "reproducibility_manifest": runtime_manifest,
            "seed": 20261022, "frames_per_video": FRAME_COUNT,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
            "qta": {"selector": "unmasked question-token attention averaged over heads and layers 5-16",
                    "top_k": 1, "logit_bias": LOGIT_BIAS, "layers_1_based": list(LAYERS)},
            "estimand": "D_g=meanLP(claim|control pixel mask,g)-meanLP(claim|support pixel mask,g); primary DeltaD=D_QTA-D_baseline; no visual-key deletion",
            "descriptive_mean_delta_D_over_scored_rows": descriptive_mean_delta,
            "records": records,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3, "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_available / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "limitations": [
                "synthetic smoke contains no human support labels and is not an efficacy result" if args.mode == "smoke" else
                    "self-reported review is not authenticated; support is limited to 16 sampled thumbnails",
                "re-tokenized claim likelihood is not original generation-tail probability",
                "black masking is out-of-distribution and does not isolate one internal causal route",
                "only three previously explored test videos; post-hoc finite-cohort descriptive result",
                "does not assess claim truth or link the claim to a frozen model prediction",
            ],
        }
        _write_private(output, payload)
        print(json.dumps({"status": payload["status"], "requested_rows": payload["requested_rows"],
            "effective_scored_rows": payload["effective_scored_rows"],
            "completed_forwards": completed, "synthetic_support_mapping": payload["synthetic_support_mapping"],
            "output_name": output.name}, sort_keys=True))
        return 0 if all_complete else 2
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_claim_support_qta_exploratory",
            "stage": stage, "exception_type": type(exc).__name__,
            "failure_code": str(exc)[:160], "started_utc": started,
            "requested_rows": (1 if args.mode == "smoke" else 3),
            "attempted_forwards": attempted, "completed_forwards": completed,
            "direct_faithfulness": False, "model_load_permitted": False,
            "rights_status": "unverified; local-only"}
        try:
            _write_private(output, failure)
        except FileExistsError:
            pass
        print(json.dumps({"status": failure["status"], "stage": stage,
            "failure_code": failure["failure_code"], "completed_forwards": completed,
            "output_name": output.name}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
