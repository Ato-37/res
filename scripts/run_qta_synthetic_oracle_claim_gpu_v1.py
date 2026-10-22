#!/usr/bin/env python3
"""Run private GPU smoke/full scoring for synthetic oracle claims v1.

This is a procedural claim-evidence sensitivity construct experiment, not a
natural-video or generated-explanation faithfulness test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import time
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from qta_synthetic_oracle_claim_v1 import (
    FRAME_COUNT, GENERATOR_VERSION, SceneDesign, _seed_population,
    assignment_manifest, build_assignment, build_clip, claims_for,
    control_temporal_groups, event_temporal_groups, map_exact_claim_positions,
    render_claim_prompt, select_attention_groups, sham_temporal_groups,
)


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
MIN_FREE_BEFORE_LOAD = int(13.5 * 1024**3)
MIN_FREE_BEFORE_FORWARD = 3 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_LOAD = 8 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_FORWARD = 3 * 1024**3
PRIVATE_RELATIVE = Path("data/private_nextgqa")
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host_available_memory_unavailable")


def _checkpoint_manifest() -> dict[str, dict[str, Any]]:
    files = sorted(SNAPSHOT.glob("model-*-of-*.safetensors"))
    if len(files) != 5:
        raise ValueError("checkpoint_shard_inventory_invalid")
    return {path.name: {"size_bytes": path.stat().st_size,
                        "sha256": _file_sha256(path)} for path in files}


def _processor_fingerprint(processor) -> tuple[str, str]:
    tokenizer_sha = _sha256(processor.tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_sha256": tokenizer_sha,
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    return _sha256(_canonical(descriptor)), tokenizer_sha


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        Path("experiments/2026-10-21_synthetic_claim_evidence_dependence_protocol_v1.md"),
        Path("scripts/qta_synthetic_oracle_claim_v1.py"),
        Path("scripts/preflight_qta_synthetic_oracle_claim_v1.py"),
        Path("scripts/run_qta_synthetic_oracle_claim_gpu_v1.py"),
        Path("scripts/qta_synthetic_event_construct_v1.py"),
        Path("scripts/run_nextgqa_gold_interval_margin_pilot.py"),
        Path("scripts/qta_attention_forward_capture.py"),
        Path("scripts/qta_multiquery_video_key_intervention.py"),
    )
    return {path.as_posix(): _file_sha256(root / path) for path in paths}


def _read_private_json(path: Path, private_root: Path) -> tuple[dict[str, Any], str]:
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(private_root))
    if path.parent != root:
        raise ValueError("receipt_must_be_direct_private_child")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("receipt_permissions_invalid")
        raw = bytearray()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(fd)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("receipt_schema_invalid")
    return value, _sha256(bytes(raw))


def _write_private_json(path: Path, payload: dict[str, Any], private_root: Path) -> None:
    root = Path(os.path.abspath(private_root))
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_directory_permissions_invalid")
    output = Path(os.path.abspath(path))
    if output.parent != root or output.suffix != ".json" or output.exists():
        raise ValueError("output_path_invalid_or_exists")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            output.unlink()
        except OSError:
            pass
        raise


def _validate_preflight(receipt: dict[str, Any], *, mode: str,
                        receipt_sha: str, current_sources: dict[str, str],
                        assignment_sha: str, processor_sha: str,
                        tokenizer_sha: str, source_commit: str) -> None:
    expected_count = 2 if mode == "smoke" else 96
    expected_clips = 1 if mode == "smoke" else 48
    required_sources = {
        "experiments/2026-10-21_synthetic_claim_evidence_dependence_protocol_v1.md",
        "scripts/qta_synthetic_oracle_claim_v1.py",
        "scripts/preflight_qta_synthetic_oracle_claim_v1.py",
        "scripts/qta_synthetic_event_construct_v1.py",
        "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    }
    if (receipt.get("schema_version") != 1
            or receipt.get("status") != "passed_cpu_processor_preflight_only"
            or receipt.get("mode") != mode
            or receipt.get("requested_clips") != expected_clips
            or receipt.get("effective_clips") != expected_clips
            or receipt.get("requested_claim_prompts") != expected_count
            or receipt.get("effective_claim_prompts") != expected_count
            or receipt.get("failure_count") != 0
            or receipt.get("model_loaded") is not False
            or receipt.get("gpu_used") is not False
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("assignment_manifest_sha256") != assignment_sha
            or receipt.get("processor_sha256") != processor_sha
            or receipt.get("tokenizer_sha256") != tokenizer_sha
            or receipt.get("source_commit") != source_commit):
        raise ValueError("cpu_preflight_receipt_invalid")
    preflight_sources = receipt.get("source_sha256")
    if not isinstance(preflight_sources, dict) or any(
            preflight_sources.get(path) != current_sources.get(path)
            for path in required_sources):
        raise ValueError("cpu_preflight_source_binding_invalid")
    if len(receipt_sha) != 64:
        raise ValueError("cpu_preflight_digest_invalid")


def _answer_margin(log_probs, *, a_id: int, b_id: int,
                   supported_label: str) -> float:
    if supported_label == "A":
        value = float(log_probs[a_id].item() - log_probs[b_id].item())
    elif supported_label == "B":
        value = float(log_probs[b_id].item() - log_probs[a_id].item())
    else:
        raise ValueError("supported_label_invalid")
    if not math.isfinite(value):
        raise ValueError("support_margin_nonfinite")
    return value


def _answer_token_ids(tokenizer, rendered: str) -> tuple[int, int]:
    """Bind the binary labels to the exact continuation after this prompt."""
    base = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(base, list) or not base:
        raise ValueError("answer_prompt_tokenization_invalid")
    result = []
    for label in ("A", "B"):
        extended = tokenizer.encode(rendered + label, add_special_tokens=False)
        if extended[:len(base)] != base:
            raise ValueError("answer_token_prompt_prefix_retokenized")
        suffix = extended[len(base):]
        if len(suffix) != 1 or type(suffix[0]) is not int or suffix[0] < 0:
            raise ValueError("answer_label_not_single_token")
        result.append(suffix[0])
    if result[0] == result[1]:
        raise ValueError("answer_label_token_collision")
    return result[0], result[1]


def _stratified_summary(cell_rows: dict[tuple[str, int], list[float]]) -> dict[str, Any]:
    if len(cell_rows) != 12 or any(len(values) != 4 for values in cell_rows.values()):
        raise ValueError("primary_cell_denominator_invalid")
    means, variances, df_terms = [], [], []
    for cell in sorted(cell_rows):
        values = cell_rows[cell]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        means.append(mean)
        variances.append(variance)
    theta = sum(means) / 12
    components = [variance / (12**2 * 4) for variance in variances]
    total_variance = sum(components)
    if total_variance == 0:
        return {"estimate": theta, "standard_error": None,
                "degrees_of_freedom": None, "ci95": None, "p_two_sided": None,
                "zero_variance": True}
    denominator = sum((component**2) / 3 for component in components if component > 0)
    if denominator == 0:
        return {"estimate": theta, "standard_error": math.sqrt(total_variance),
                "degrees_of_freedom": None, "ci95": None, "p_two_sided": None,
                "zero_variance": True}
    df = total_variance**2 / denominator
    # Use scipy's Student-t distribution for the planned interval and test.
    from scipy.stats import t as student_t
    se = math.sqrt(total_variance)
    critical = float(student_t.ppf(0.975, df))
    p = float(2 * student_t.sf(abs(theta / se), df))
    return {"estimate": theta, "standard_error": se,
            "degrees_of_freedom": df,
            "ci95": [theta - critical * se, theta + critical * se],
            "p_two_sided": p, "zero_variance": False}


def _smoke_design() -> SceneDesign:
    import random
    from qta_synthetic_oracle_claim_v1 import BACKGROUNDS
    seed = _seed_population()[0]
    rng = random.Random(seed)
    return SceneDesign(
        family="gate", onset=18, replicate=-1, scene_seed=seed,
        target_object="star", label_order="supported_A",
        gate_x=rng.choice((112, 128, 144)), track_offset_y=rng.choice((-8, 0, 8)),
        base_radius=rng.choice((24, 28, 32)), background_rgb=rng.choice(BACKGROUNDS),
        star_upper=bool(rng.randrange(2)), star_uses_color0=bool(rng.randrange(2)))


def execute(*, mode: str, output: Path, preflight_path: Path,
            smoke_path: Path | None = None,
            repo_root: Path | None = None) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("run_mode_invalid")
    started_at = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    private = repo / PRIVATE_RELATIVE
    output = Path(os.path.abspath(output))
    if output.parent != private or output.exists():
        raise ValueError("output_must_be_new_private_child")
    if not SNAPSHOT.is_dir():
        raise RuntimeError("pinned_checkpoint_unavailable")
    if mode == "full" and smoke_path is None:
        raise ValueError("full_mode_requires_external_smoke")

    sources = _source_hashes(repo)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assignment = assignment_manifest()
    assignment_sha = _sha256(_canonical(assignment))
    preflight, preflight_sha = _read_private_json(preflight_path, private)

    import torch
    import transformers
    import PIL
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from qta_attention_forward_capture import (
        aggregate_captured_qta_rows, capture_qta_attention_forward_once,
    )
    from qta_multiquery_video_key_intervention import (
        install_multiquery_video_key_intervention_hooks, remove_hooks,
    )
    from run_nextgqa_gold_interval_margin_pilot import process_qa_video

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("pinned_cuda_bf16_runtime_unavailable")
    free_before, total_vram = torch.cuda.mem_get_info()
    host_before = _host_available_bytes()
    if (free_before < MIN_FREE_BEFORE_LOAD
            or host_before < MIN_HOST_AVAILABLE_BEFORE_LOAD):
        raise RuntimeError("preload_resource_guard_failed")
    checkpoint_shards = _checkpoint_manifest()
    checkpoint_sha = _sha256(_canonical(checkpoint_shards))
    model_config_sha = _file_sha256(SNAPSHOT / "config.json")

    processor = AutoProcessor.from_pretrained(
        str(SNAPSHOT), revision=MODEL_REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        trust_remote_code=False)
    processor_sha, tokenizer_sha = _processor_fingerprint(processor)
    _validate_preflight(preflight, mode=mode, receipt_sha=preflight_sha,
                        current_sources=sources, assignment_sha=assignment_sha,
                        processor_sha=processor_sha, tokenizer_sha=tokenizer_sha,
                        source_commit=source_commit)
    expected_grid = tuple(preflight["geometry_counts"][0]["video_grid_thw"])
    if mode == "full":
        smoke, smoke_sha = _read_private_json(smoke_path, private)
        embedded_preflight = smoke.get("cpu_preflight_receipt", {})
        embedded_sha = _sha256(_canonical(embedded_preflight) + b"\n")
        if embedded_sha != smoke.get("cpu_preflight_receipt_sha256"):
            raise ValueError("external_smoke_embedded_preflight_digest_invalid")
        _validate_preflight(embedded_preflight, mode="smoke",
                            receipt_sha=smoke.get("cpu_preflight_receipt_sha256", ""),
                            current_sources=sources, assignment_sha=assignment_sha,
                            processor_sha=processor_sha, tokenizer_sha=tokenizer_sha,
                            source_commit=source_commit)
        if (smoke.get("status") != "passed_external_gpu_smoke_only"
                or smoke.get("mode") != "smoke"
                or smoke.get("completed_forwards") != 18
                or smoke.get("failure_count") != 0
                or smoke.get("gpu_used") is not True
                or smoke.get("checkpoint_sha256") != checkpoint_sha
                or smoke.get("model_config_sha256") != model_config_sha
                or smoke.get("source_sha256") != sources
                or smoke.get("processor_sha256") != processor_sha
                or smoke.get("assignment_manifest_sha256") != assignment_sha):
            raise ValueError("external_gpu_smoke_binding_invalid")
    else:
        smoke_sha = None

    free_immediate, _ = torch.cuda.mem_get_info()
    host_immediate = _host_available_bytes()
    if (free_immediate < MIN_FREE_BEFORE_LOAD
            or host_immediate < MIN_HOST_AVAILABLE_BEFORE_LOAD):
        raise RuntimeError("immediate_preload_resource_guard_failed")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(SNAPSHOT), revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    ).eval()
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
        raise RuntimeError("model_processor_contract_invalid")
    torch.cuda.reset_peak_memory_stats()

    designs = (_smoke_design(),) if mode == "smoke" else build_assignment()
    cell_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    baseline_cell_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    family_onset_summary: dict[str, dict[str, float]] = {}
    attempted = completed = accuracy_correct = accuracy_total = 0
    selection_overlap_counts: dict[str, int] = defaultdict(int)
    capture_positions_hash = hashlib.sha256(b"oracle-claim-selector-positions-v1\0")
    assignment_cursor = 0
    start_loop = time.monotonic()
    try:
        for design in designs:
            clip = build_clip(design)
            true_claim, false_claim = claims_for(design)
            claim_results: dict[str, dict[str, Any]] = {}
            for claim_kind, claim in (("true", true_claim), ("false", false_claim)):
                prompt, supported_label = render_claim_prompt(claim, design.label_order)
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": list(clip.frames), "sample_fps": 1.0},
                    {"type": "text", "text": prompt},
                ]}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                a_id, b_id = _answer_token_ids(processor.tokenizer, rendered)
                images, videos = process_vision_info(messages, return_video_metadata=True)
                if images is not None or not isinstance(videos, list) or len(videos) != 1:
                    raise ValueError("video_input_contract_invalid")
                pixels, metadata = videos[0]
                if len(pixels) != FRAME_COUNT:
                    raise ValueError("processor_frame_count_invalid")
                cpu_batch = process_qa_video(processor, rendered, images, pixels, metadata)
                ids, grid, mask = (cpu_batch.get("input_ids"),
                                   cpu_batch.get("video_grid_thw"),
                                   cpu_batch.get("attention_mask"))
                if (not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1
                        or not torch.is_tensor(grid) or tuple(grid.shape) != (1, 3)
                        or not torch.is_tensor(mask) or mask.shape != ids.shape
                        or not bool((mask == 1).all())):
                    raise ValueError("processor_tensor_contract_invalid")
                positions = map_exact_claim_positions(
                    tokenizer=processor.tokenizer, rendered_text=rendered, claim=claim,
                    processed_input_ids=ids, video_token_id=int(processor.video_token_id))
                selector_position = positions[-1]
                query_position = int(ids.shape[1]) - 1
                if selector_position >= query_position:
                    raise ValueError("claim_selector_not_before_answer_query")
                t, gh, gw = map(int, grid[0].tolist())
                if (tuple((t, gh, gw)) != expected_grid or t != 16
                        or gh % int(processor.image_processor.merge_size)
                        or gw % int(processor.image_processor.merge_size)):
                    raise ValueError("processor_grid_binding_invalid")
                tokens_per_group = (gh * gw
                                    // int(processor.image_processor.merge_size) ** 2)
                visual_positions = [i for i, token in enumerate(ids[0].tolist())
                                    if int(token) == int(processor.video_token_id)]
                if len(visual_positions) != t * tokens_per_group:
                    raise ValueError("visual_key_count_invalid")
                groups = tuple(tuple(visual_positions[g * tokens_per_group:(g + 1) * tokens_per_group])
                               for g in range(t))
                event_groups = event_temporal_groups(design.onset, t)
                clip_token = f"oracle-v1:{design.family}:{design.onset}:{design.scene_seed}"
                controls = control_temporal_groups(
                    scene_seed=design.scene_seed, clip_token=clip_token,
                    event_groups=event_groups, temporal_group_count=t)
                sham_groups = sham_temporal_groups(
                    scene_seed=design.scene_seed, clip_token=clip_token,
                    claim_kind=claim_kind, temporal_group_count=t)
                event_keys = tuple(sorted(k for group in event_groups for k in groups[group]))
                control_keys = tuple(sorted(k for group in controls for k in groups[group]))
                device_batch = {key: value.to("cuda") if torch.is_tensor(value) else value
                                for key, value in cpu_batch.items()}
                forward_count = 0

                def forward(*, bias_keys=(), ablate_keys=(), capture=False):
                    nonlocal attempted, completed, forward_count
                    free, _ = torch.cuda.mem_get_info()
                    if (free < MIN_FREE_BEFORE_FORWARD
                            or _host_available_bytes() < MIN_HOST_AVAILABLE_BEFORE_FORWARD):
                        raise RuntimeError("per_forward_resource_guard_failed")
                    if capture:
                        def do_forward():
                            with torch.inference_mode():
                                return model(**device_batch, use_cache=False, logits_to_keep=1)
                        attempted += 1
                        output, rows, capture_receipt = capture_qta_attention_forward_once(
                            model=model, forward=do_forward,
                            question_positions=(selector_position,),
                            input_length=int(ids.shape[1]), torch_module=torch)
                        if (capture_receipt.get("forward_calls") != 1
                                or capture_receipt.get("layer_indices_1_based") != list(LAYERS_1_BASED)
                                or capture_receipt.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
                            raise RuntimeError("claim_attention_capture_invalid")
                        masses = aggregate_captured_qta_rows(
                            rows=rows, group_positions=groups, torch_module=torch)
                        del rows, capture_receipt
                    else:
                        if bias_keys or ablate_keys:
                            handles = install_multiquery_video_key_intervention_hooks(
                                model=model,
                                layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
                                biased_positions=tuple(bias_keys),
                                ablated_positions=tuple(ablate_keys),
                                query_positions=(query_position,), logit_bias=LOGIT_BIAS,
                                torch_module=torch)
                        else:
                            handles = []
                        try:
                            attempted += 1
                            with torch.inference_mode():
                                output = model(**device_batch, use_cache=False, logits_to_keep=1)
                        finally:
                            remove_hooks(handles)
                        masses = None
                    logits = getattr(output, "logits", None)
                    if (not torch.is_tensor(logits) or logits.ndim != 3
                            or tuple(logits.shape[:2]) != (1, 1)
                            or not bool(torch.isfinite(logits).all())):
                        raise RuntimeError("answer_prediction_logits_invalid")
                    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
                    margin = _answer_margin(log_probs, a_id=a_id, b_id=b_id,
                                            supported_label=supported_label)
                    # Accuracy is defined within the frozen binary choice set,
                    # consistently with the A-vs-B log-odds endpoint.
                    predicted = "A" if log_probs[a_id] >= log_probs[b_id] else "B"
                    completed += 1
                    forward_count += 1
                    del output, logits, log_probs
                    torch.cuda.empty_cache()
                    return margin, predicted, masses

                # Full-video baseline is the only selector source.
                base_full, base_pred, masses = forward(capture=True)
                selected_groups = select_attention_groups(masses)
                selected_keys = tuple(sorted(k for group in selected_groups for k in groups[group]))
                sham_keys = tuple(sorted(k for group in sham_groups for k in groups[group]))
                selection_overlap_counts["qta_event"] += len(set(selected_groups) & set(event_groups))
                selection_overlap_counts["qta_control"] += len(set(selected_groups) & set(controls))
                selection_overlap_counts["qta_sham"] += len(set(selected_groups) & set(sham_groups))
                selection_overlap_counts["sham_event"] += len(set(sham_groups) & set(event_groups))
                selection_overlap_counts["sham_control"] += len(set(sham_groups) & set(controls))
                capture_positions_hash.update(_canonical({
                    "family": design.family, "onset": design.onset,
                    "seed_sha256": _sha256(str(design.scene_seed).encode("ascii")),
                    "claim_kind": claim_kind, "selector_position": selector_position,
                    "query_position": query_position,
                    "selected_groups": list(selected_groups),
                    "sham_groups": list(sham_groups),
                }))
                base_event, _p, _ = forward(ablate_keys=event_keys)
                base_control, _p, _ = forward(ablate_keys=control_keys)
                qta_full, qta_pred, _ = forward(bias_keys=selected_keys)
                qta_event, _p, _ = forward(bias_keys=selected_keys, ablate_keys=event_keys)
                qta_control, _p, _ = forward(bias_keys=selected_keys, ablate_keys=control_keys)
                sham_full, sham_pred, _ = forward(bias_keys=sham_keys)
                sham_event, _p, _ = forward(bias_keys=sham_keys, ablate_keys=event_keys)
                sham_control, _p, _ = forward(bias_keys=sham_keys, ablate_keys=control_keys)
                if forward_count != 9:
                    raise RuntimeError("claim_branch_forward_count_invalid")
                claim_results[claim_kind] = {
                    "baseline_full": base_full,
                    "baseline_event": base_event,
                    "baseline_control": base_control,
                    "qta_full": qta_full,
                    "qta_event": qta_event,
                    "qta_control": qta_control,
                    "sham_full": sham_full,
                    "sham_event": sham_event,
                    "sham_control": sham_control,
                    "baseline_pred": base_pred,
                    "qta_pred": qta_pred,
                    "sham_pred": sham_pred,
                    "selected_groups": selected_groups,
                    "sham_groups": sham_groups,
                    "forward_count": forward_count,
                }
                del device_batch, cpu_batch, ids, grid, mask, groups, positions, pixels
                del metadata, videos, images, messages, rendered, masses
                gc.collect()
                torch.cuda.empty_cache()
            if set(claim_results) != {"true", "false"}:
                raise RuntimeError("true_false_claim_pair_incomplete")
            def m(arm: str, condition: str) -> float:
                return (claim_results["true"][f"{arm}_{condition}"]
                        - claim_results["false"][f"{arm}_{condition}"])
            d_baseline = m("baseline", "control") - m("baseline", "event")
            d_qta = m("qta", "control") - m("qta", "event")
            d_sham = m("sham", "control") - m("sham", "event")
            scene_theta = d_qta - d_sham
            cell = (design.family, design.onset)
            if mode == "full":
                cell_values[cell].append(scene_theta)
                baseline_cell_values[cell].append(d_baseline)
            family_onset_summary[f"{design.family}|{design.onset}"] = {
                "n": len(cell_values[cell]) if mode == "full" else 1,
                "baseline_D_mean": d_baseline,
                "qta_D_mean": d_qta,
                "sham_D_mean": d_sham,
                "primary_scene_contrast": scene_theta,
            }
            # A/B choice correctness on the full-video branch for true and false claims.
            for kind, expected_supported in (("true", True), ("false", False)):
                row = claim_results[kind]
                prediction = row["qta_pred"]
                supported_label = "A" if design.label_order == "supported_A" else "B"
                predicted_supported = prediction == supported_label
                accuracy_correct += int(predicted_supported == expected_supported)
                accuracy_total += 1
            assignment_cursor += 1
            if assignment_cursor % 4 == 0:
                torch.cuda.synchronize()
            del claim_results, clip, true_claim, false_claim, design
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        requested_clips = 1 if mode == "smoke" else 48
        requested_claims = 2 * requested_clips
        requested_forwards = requested_claims * 9
        if completed != requested_forwards or attempted != requested_forwards:
            raise RuntimeError("requested_forward_denominator_incomplete")
        summary = _stratified_summary(dict(cell_values)) if mode == "full" else None
        payload = {
            "schema_version": 1,
            "status": ("passed_external_gpu_smoke_only" if mode == "smoke"
                       else "passed_full_synthetic_oracle_claim_construct"),
            "purpose": "synthetic_claim_conditioned_video_evidence_sensitivity_not_faithfulness",
            "mode": mode,
            "started_utc": started_utc,
            "requested_clips": requested_clips,
            "effective_clips": requested_clips,
            "requested_claim_prompts": requested_claims,
            "effective_claim_prompts": requested_claims,
            "requested_forwards": requested_forwards,
            "completed_forwards": completed,
            "attempted_forwards": attempted,
            "failure_count": 0,
            "primary": "equal_mean_over_12_family_onset_cells_of[(true_minus_false support-logodds control_minus_event under claim-selected bias) minus same under independent hash sham]",
            "primary_summary": summary,
            "cell_sufficient_statistics": {
                f"{family}|{onset}": {
                    "n": len(values), "sum": sum(values),
                    "sum_squares": sum(x * x for x in values),
                    "positive_count": sum(x > 0 for x in values),
                    "negative_count": sum(x < 0 for x in values),
                    "zero_count": sum(x == 0 for x in values),
                    "baseline_D_mean": sum(baseline_cell_values[(family, onset)]) /
                                       len(baseline_cell_values[(family, onset)])
                                       if baseline_cell_values[(family, onset)] else None,
                } for (family, onset), values in sorted(cell_values.items())
            } if mode == "full" else {},
            "smoke_only_summary": family_onset_summary if mode == "smoke" else None,
            "full_video_qta_accuracy": {
                "correct": accuracy_correct, "n": accuracy_total,
                "fraction": accuracy_correct / accuracy_total if accuracy_total else None,
            },
            "selector_and_sham_overlap_group_sums": dict(sorted(selection_overlap_counts.items())),
            "selector_positions_digest_sha256": capture_positions_hash.hexdigest(),
            "assignment_manifest_sha256": assignment_sha,
            "cpu_preflight_receipt_sha256": preflight_sha,
            "cpu_preflight_receipt": preflight,
            "external_smoke_receipt_sha256": smoke_sha,
            "source_commit": source_commit,
            "source_sha256": sources,
            "checkpoint_shards": checkpoint_shards,
            "checkpoint_sha256": checkpoint_sha,
            "model_config_sha256": model_config_sha,
            "processor_sha256": processor_sha,
            "tokenizer_sha256": tokenizer_sha,
            "model_revision": MODEL_REVISION,
            "precision": "4bit_bf16_compute_bf16_eager",
            "layers_1_based": list(LAYERS_1_BASED),
            "logit_bias": LOGIT_BIAS,
            "video_grid_thw": list(expected_grid),
            "started_loop_elapsed_seconds": round(time.monotonic() - start_loop, 3),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "runtime": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "transformers": str(transformers.__version__), "pillow": str(PIL.__version__),
                "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
                "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "free_vram_before_load_bytes": free_before,
                "total_vram_bytes": total_vram,
                "host_available_before_load_bytes": host_before,
                "free_vram_immediately_before_load_bytes": free_immediate,
                "host_available_immediately_before_load_bytes": host_immediate,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "model_loaded": True,
            "gpu_used": True,
            "cohort_included": mode == "full",
            "faithfulness_result": False,
            "rights_status": "unverified_local_only",
            "data_handling": "aggregate-only private local receipt; no frames, claims, or raw prompt text",
            "limitations": [
                "Procedural synthetic claims only; not natural-video explanation faithfulness.",
                "Oracle claim judgments are derived from the generator and are not human reviewed.",
                "Attention bias and key ablation can induce distribution shift and interact.",
                "Four scenes per family-by-onset cell support only a small benchmark-specific estimate.",
            ],
        }
        _write_private_json(output, payload, private)
        return payload
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(mode=args.mode, output=args.output,
                     preflight_path=args.preflight_receipt,
                     smoke_path=args.smoke_receipt)
    print(json.dumps({key: result[key] for key in (
        "status", "mode", "requested_forwards", "completed_forwards",
        "failure_count", "elapsed_seconds", "primary_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
