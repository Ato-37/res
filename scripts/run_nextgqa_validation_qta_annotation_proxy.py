#!/usr/bin/env python3
"""Exploratory QTA benchmark-annotation proxy on a fixed historical cohort.

This runner reuses the exact 30-row 2026-10-08 validation pilot manifest. It
measures how QTA-Bias changes dependence on NExT-GQA's annotated intervals
under equal-count attention-key ablation on one unchanged processor batch. It
is not explanation faithfulness or a fresh-cohort inference. Outputs are private.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from qta_source_capsule import capture_source_capsule, recover_source_capsule

PILOT_MANIFEST_SHA256 = "968ed130684510de8c6fe7e91a4c0ed90db2cac90a8b2200ff3b55254cd77875"
PILOT_RUN_MANIFEST = "results/faithfulness_validation_pilot_20261008_run_manifest.json"
PILOT_RUN_MANIFEST_SHA256 = "20da1bc521c67aad20f6c9593ca7639ddd1b4eaec750263c2edd812493025f83"
PRIVATE_PILOT_COPY = "data/private_nextgqa/faithfulness_validation_pilot_20261008.raw.json"
V6_INVENTORY = ("data/private_nextgqa/qta_pre_review_inventory_20261018_ongoing_01/"
                "pre_review_candidate_byte_inventory.json")
V6_PROVENANCE = ("data/private_nextgqa/qta_pre_review_inventory_20261018_ongoing_01/"
                 "source_provenance.json")
V6_INVENTORY_SHA256 = "c54b7262dfea04572bbdb86beeef3d7e149199e63586d5c2c38e0d69a3d62b77"
V6_PROVENANCE_SHA256 = "9b493ecbc1731712d42ac26b177e20940cb350a0f4581eeaab754f8dee3d85e6"
FRAME_COUNT = 32
GOLD_COHORT_N = 30
SEED = 20261018
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATASET_SOURCE_SHA256 = {
    "NExT-GQA/datasets/nextgqa/val.csv": "ac5a1eb1d6dee9cd84ab7f9c94ab345b2404646f24b79ac182cd2aa9e2d424c1",
    "NExT-GQA/datasets/nextgqa/gsub_val.json": "4feb23e53102cb2d4779e857ff8f9fd9b552fc38435621406b2ed9b0bf76204b",
    "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json": "1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685",
}
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
SOURCE_FILES = (
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/qwen_visual_attention_boost.py",
    "scripts/qwen_internal_attention_ablation.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/fixed_prediction_selective_dependence.py",
    "scripts/audit_qwen_checkpoint_bytes_cpu.py",
    "scripts/qta_source_capsule.py",
    "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
    "experiments/2026-10-18_qta_treatment_selection.md",
    "experiments/2026-10-18_qta_annotation_anchor_proxy_v1.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def processor_provenance(processor, model_path: Path) -> dict:
    """Fingerprint tokenizer/image processor plus the pinned config bytes."""
    tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
    tokenizer_sha = hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_json_sha256": tokenizer_sha,
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    fingerprint = hashlib.sha256(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    return {"model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "tokenizer_serialization_sha256": tokenizer_sha,
            "processor_fingerprint_sha256": fingerprint,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "fingerprint_scope": "tokenizer serialization + image processor config + video token ID; separate video processor config is not included"}


def validate_pilot_bytes(raw: bytes, run_manifest: dict, *, smoke: bool = False) -> list[dict]:
    """Bind exact historical rows to their published run manifest without IDs in output."""
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != PILOT_MANIFEST_SHA256:
        raise ValueError("historical_pilot_manifest_hash_mismatch")
    validate_pilot_run_receipt(run_manifest)
    rows = json.loads(raw.decode("utf-8"))
    if (not isinstance(rows, list) or len(rows) != GOLD_COHORT_N
            or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("historical_pilot_rows_invalid")
    required = {"video", "video_id", "qid", "question", "answer",
                "answer_choice_index", "choices"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("historical_pilot_row_schema_invalid")
    validate_pilot_cohort_identity(rows)
    if (any(type(row["answer_choice_index"]) is not int
                   or not 0 <= row["answer_choice_index"] < 5
                   or not isinstance(row["choices"], list) or len(row["choices"]) != 5
                   for row in rows)):
        raise ValueError("historical_pilot_cohort_not_one_row_per_video")
    return rows[:1] if smoke else rows


def validate_pilot_run_receipt(run_manifest: dict) -> None:
    if (not isinstance(run_manifest, dict)
            or run_manifest.get("samples_sha256") != PILOT_MANIFEST_SHA256
            or run_manifest.get("n_requested") != GOLD_COHORT_N
            or run_manifest.get("n_effective") != GOLD_COHORT_N
            or run_manifest.get("n_failures") != 0):
        raise ValueError("historical_pilot_run_manifest_mismatch")


def validate_pilot_cohort_identity(rows: list[dict]) -> None:
    """Require one row per video; question IDs are video-local in this data."""
    if (not isinstance(rows, list) or len(rows) != GOLD_COHORT_N
            or len({str(row.get("video_id")) for row in rows
                    if isinstance(row, dict)}) != GOLD_COHORT_N
            or len({(str(row.get("video_id")), str(row.get("qid")))
                    for row in rows if isinstance(row, dict)}) != GOLD_COHORT_N):
        raise ValueError("historical_pilot_cohort_not_one_row_per_video")


def temporal_support_end(annotation_duration: Fraction,
                         sampled_pts: list[Fraction]) -> Fraction:
    """Keep the dataset's annotation bound separate from sampled-video support."""
    if not sampled_pts:
        raise ValueError("sampled_pts_empty")
    return max(annotation_duration, sampled_pts[-1])


def validate_pilot_run_manifest_bytes(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != PILOT_RUN_MANIFEST_SHA256:
        raise ValueError("historical_pilot_receipt_hash_mismatch")
    receipt = json.loads(raw.decode("utf-8"))
    validate_pilot_run_receipt(receipt)
    return receipt


def validate_v6_inventory(inventory_raw: bytes, provenance_raw: bytes) -> frozenset[str]:
    """Read hash-only inventory metadata; never open review package/media files."""
    if hashlib.sha256(inventory_raw).hexdigest() != V6_INVENTORY_SHA256:
        raise ValueError("v6_candidate_inventory_hash_mismatch")
    if hashlib.sha256(provenance_raw).hexdigest() != V6_PROVENANCE_SHA256:
        raise ValueError("v6_source_provenance_hash_mismatch")
    inventory = json.loads(inventory_raw.decode("utf-8"))
    provenance = json.loads(provenance_raw.decode("utf-8"))
    return validate_v6_inventory_metadata(inventory, provenance)


def validate_v6_inventory_metadata(inventory: dict, provenance: dict) -> frozenset[str]:
    """Validate the hash-only inventory/source-provenance join."""
    if (provenance.get("inventory_path") != V6_INVENTORY
            or provenance.get("inventory_sha256") != V6_INVENTORY_SHA256
            or provenance.get("package_dir") != "data/private_nextgqa/reviewer_package_v6_20261018_0243"
            or provenance.get("execution_authorization") != "not_granted"):
        raise ValueError("v6_source_provenance_binding_invalid")
    artifacts = inventory.get("artifacts") if isinstance(inventory, dict) else None
    if not isinstance(artifacts, list):
        raise ValueError("v6_candidate_inventory_invalid")
    entries = {row.get("role"): row.get("sha256") for row in artifacts
               if isinstance(row, dict)}
    roles = {f"review_package_media_{index:02d}" for index in (1, 2, 3)}
    hashes = {entries.get(role) for role in roles}
    if (None in hashes or len(hashes) != 3
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes)):
        raise ValueError("v6_review_media_exclusion_set_mismatch")
    return frozenset(hashes)


def validate_dataset_pins(repo: Path) -> dict[str, str]:
    actual = {relative: sha256_file(repo / relative)
              for relative in DATASET_SOURCE_SHA256}
    if actual != DATASET_SOURCE_SHA256:
        raise ValueError("validation_dataset_source_hash_mismatch")
    return actual


def resolve_pilot_manifest(repo: Path, supplied: str | None) -> Path:
    """Resolve the default beside the checkout, not beside the user's home."""
    return Path(supplied) if supplied is not None else repo / PRIVATE_PILOT_COPY


def validate_preflight_plans(plans, requested: int, *, smoke=False):
    expected = 1 if smoke else GOLD_COHORT_N
    if (type(requested) is not int or requested != expected
            or not isinstance(plans, list) or len(plans) != expected
            or any(not isinstance(plan, dict)
                   or not isinstance(plan.get("geometry_sha256"), str)
                   or len(plan["geometry_sha256"]) != 64
                   or plan.get("evidence_count", 0) <= 0
                   or plan.get("evidence_count") != plan.get("control_count")
                   or plan.get("group_count", 0) <= 0
                   for plan in plans)):
        raise ValueError("preflight_effective_denominator_or_control_invalid")
    return True


def validate_checkpoint_audit(report: dict) -> dict:
    """Fail closed unless the CPU byte audit authenticated the pinned revision."""
    if (not isinstance(report, dict)
            or report.get("model_id") != "Qwen/Qwen2.5-VL-7B-Instruct"
            or report.get("revision") != MODEL_REVISION
            or report.get("byte_authentication_status") != "verified_current_local_bytes"
            or report.get("requested_shards") != 5
            or report.get("effective_shards") != 5
            or report.get("failure_count") != 0
            or report.get("failure_codes") != []):
        raise ValueError("pinned_checkpoint_byte_audit_failed")
    shards = report.get("shards")
    expected_names = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
    if (not isinstance(shards, list) or len(shards) != 5
            or [item.get("filename") for item in shards if isinstance(item, dict)] != expected_names
            or any(item.get("sha256_matches") is not True
                   or item.get("stable_during_read") is not True
                   or item.get("failure_code") is not None
                   or not isinstance(item.get("observed_sha256"), str)
                   or len(item["observed_sha256"]) != 64
                   for item in shards)):
        raise ValueError("pinned_checkpoint_shard_audit_incomplete")
    if report.get("api_metadata_hash_matches_pinned_artifact") is not True:
        raise ValueError("pinned_checkpoint_metadata_audit_failed")
    return {"status": report["byte_authentication_status"],
            "revision": report["revision"],
            "requested_shards": report["requested_shards"],
            "effective_shards": report["effective_shards"],
            "failure_count": report["failure_count"],
            "shards": [{"filename": item["filename"],
                        "size_bytes": item["observed_size_bytes"],
                        "sha256": item["observed_sha256"]} for item in shards]}


def audit_pinned_checkpoint(repo: Path, snapshot_dir: Path) -> dict:
    """Run the established CPU-only streaming audit; do not persist its raw report."""
    import audit_qwen_checkpoint_bytes_cpu as audit
    metadata_bytes = (repo / audit.API_METADATA_RELATIVE).read_bytes()
    report = audit.audit_checkpoint(
        metadata_bytes=metadata_bytes, snapshot_dir=snapshot_dir,
        expected_metadata_sha256=audit.API_METADATA_SHA256, repo_root=repo)
    return validate_checkpoint_audit(report)


BRANCH_NAMES = (
    "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
    "intervention_full", "intervention_evidence_ablation",
    "intervention_control_ablation",
)


def validate_shared_batch_fingerprints(fingerprints: dict) -> str:
    if (not isinstance(fingerprints, dict) or set(fingerprints) != set(BRANCH_NAMES)
            or any(not isinstance(value, str) or len(value) != 64
                   for value in fingerprints.values())
            or len(set(fingerprints.values())) != 1):
        raise ValueError("six_branch_original_batch_binding_mismatch")
    return next(iter(fingerprints.values()))


def failure_receipt(*, requested_n: int, stage: str, error_type: str,
                    preflight_attempted: int, preflight_completed: int,
                    branch_counts: dict, failure_detail: str | None = None,
                    evidence_selection_policy="all_annotations",
                    source_capsule: dict | None = None) -> dict:
    return {"status": "failed_incomplete_no_cohort_estimate",
            "scope": "NExT-GQA annotation-interval dependence proxy; not explanation faithfulness",
            "evidence_selection_policy": evidence_selection_policy,
            "requested_n": requested_n, "effective_n": 0, "failures": 1,
            "failure_stage": stage, "failure_type": error_type,
            "failure_detail": failure_detail[:240] if isinstance(failure_detail, str) else None,
            "preflight_requested": requested_n,
            "preflight_attempted": preflight_attempted,
            "preflight_completed": preflight_completed,
            "branch_counts": {name: {**counts, "effective": 0}
                              for name, counts in branch_counts.items()}, "records": [],
            "finite_cohort_delta_f": None,
            "inference": "none; incomplete denominator",
            "source_capsule": source_capsule,
            "execution_authorization": "local_only_user_authorized"}


def safe_failure_detail(exc: Exception) -> str | None:
    """Keep only bounded runtime diagnostics or fixed underscore error codes."""
    detail = str(exc)
    if isinstance(exc, ValueError) and not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", detail):
        return None
    if not isinstance(exc, (ValueError, RuntimeError)):
        return None
    return detail[:240] or None


def read_private_source(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_source_file_mode_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("private_source_hash_mismatch")
    return raw


def score_six_calls(*, baseline_full, baseline_evidence, baseline_control,
                    qta_full, qta_evidence, qta_control) -> dict:
    """Use the existing six-branch scorer with p/q frozen from baseline full."""
    from fixed_prediction_selective_dependence import (
        INTERVENTION_CONDITIONS, fixed_prediction_intervention_contrast)
    values = {"baseline_full": baseline_full,
              "baseline_evidence_ablation": baseline_evidence,
              "baseline_control_ablation": baseline_control,
              "intervention_full": qta_full,
              "intervention_evidence_ablation": qta_evidence,
              "intervention_control_ablation": qta_control}
    if set(values) != INTERVENTION_CONDITIONS:
        raise ValueError("exact_six_condition_scores_required")
    result = fixed_prediction_intervention_contrast(values)
    # The helper's synthetic-only status describes its API boundary; this
    # caller has real model outputs, but the score remains exploratory only.
    return {key: result[key] for key in (
        "answer_index", "foil_index", "margin_by_condition",
        "baseline_evidence_drop", "baseline_control_drop",
        "baseline_selective_dependence", "intervention_evidence_drop",
        "intervention_control_drop", "intervention_selective_dependence", "delta_f")}


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty_finite_cohort")
    mean = sum(values) / len(values)
    return {"n": len(values), "mean": mean,
            "sample_sd": (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** .5
            if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def summarize_baseline_accuracy(rows: list[dict], requested: int, *, smoke: bool) -> dict | None:
    expected = 1 if smoke else GOLD_COHORT_N
    if (requested != expected or len(rows) != expected
            or any(type(row.get("baseline_full_correct")) is not int
                   or row["baseline_full_correct"] not in (0, 1) for row in rows)):
        raise ValueError("baseline_full_accuracy_denominator_invalid")
    if smoke:
        return None
    correct = sum(row["baseline_full_correct"] for row in rows)
    return {"n": expected, "correct": correct, "accuracy": correct / expected,
            "interpretation": "descriptive full-input A-E benchmark accuracy; not faithfulness"}


def source_hashes(repo: Path) -> tuple[str, dict[str, str]]:
    """Record exact executed sources without requiring a clean/committed tree."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    return commit, {relative: sha256_file(repo / relative) for relative in SOURCE_FILES}


class SourceCapsuleMismatch(ValueError):
    def __init__(self, metadata: dict):
        super().__init__("source_capsule_source_hash_mismatch")
        self.source_capsule = metadata


def capture_verified_source_capsule(repo: Path, capsule_directory: Path | str,
                                   output_name: str,
                                   expected_source_hashes: dict[str, str]) -> dict:
    """Capture and reverify the frozen source allowlist before data/model work."""
    if (not isinstance(output_name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", output_name)):
        raise ValueError("source_capsule_output_name_invalid")
    if (not isinstance(expected_source_hashes, dict)
            or tuple(expected_source_hashes) != SOURCE_FILES):
        raise ValueError("source_capsule_expected_map_invalid")
    capsule_name = f"{output_name}.source-capsule.zip"
    capsule_path = Path(capsule_directory) / capsule_name
    captured = capture_source_capsule(source_root=repo, ordered_paths=SOURCE_FILES,
                                      output_path=capsule_path)
    verified = recover_source_capsule(
        capsule_path=capsule_path, expected_paths=SOURCE_FILES,
        expected_archive_sha256=captured["archive_sha256"])
    verified_map = dict(zip(SOURCE_FILES, verified["file_sha256"], strict=True))
    metadata = {"basename": capsule_name,
                "archive_sha256": verified["archive_sha256"],
                "verified_source_sha256": verified_map,
                "verified_before_data_access": True,
                "provenance_authenticated": False}
    if verified_map != expected_source_hashes:
        raise SourceCapsuleMismatch(metadata)
    return metadata


def _video_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def load_fixed_rows(repo: Path, manifest_path: Path, video_root: Path,
                    *, smoke: bool = False):
    v6_exclusion_hashes = validate_v6_inventory(
        read_private_source(repo / V6_INVENTORY, V6_INVENTORY_SHA256),
        read_private_source(repo / V6_PROVENANCE, V6_PROVENANCE_SHA256))
    dataset_hashes = validate_dataset_pins(repo)
    run_manifest_raw = (repo / PILOT_RUN_MANIFEST).read_bytes()
    run_manifest = validate_pilot_run_manifest_bytes(run_manifest_raw)
    raw = read_private_source(Path(manifest_path), PILOT_MANIFEST_SHA256)
    rows = validate_pilot_bytes(raw, run_manifest, smoke=smoke)
    video_map = json.loads((repo / "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json")
                           .read_text(encoding="utf-8"))
    val_rows = {}
    with (repo / "NExT-GQA/datasets/nextgqa/val.csv").open(
            newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            val_rows[(str(row["video_id"]), str(row["qid"]))] = row
    gsub = json.loads((repo / "NExT-GQA/datasets/nextgqa/gsub_val.json")
                      .read_text(encoding="utf-8"))
    paths = {}
    for path in Path(video_root).rglob("*.mp4"):
        paths.setdefault(_video_key(path.relative_to(video_root).as_posix()), []).append(path)
    result = []
    research_media = set()
    for row in rows:
        key = (str(row["video_id"]), str(row["qid"]))
        source = val_rows.get(key)
        if source is None or key[0] not in video_map:
            raise ValueError("historical_pilot_dataset_join_missing")
        choices = [str(source[f"a{i}"]).strip() for i in range(5)]
        try:
            answer_index = int(row["answer_choice_index"])
        except (TypeError, ValueError):
            raise ValueError("historical_pilot_answer_index_invalid") from None
        if (str(source["question"]).strip() != row["question"]
                or choices != row["choices"]
                or str(source["answer"]).strip() != row["answer"]
                or answer_index < 0 or answer_index >= 5
                or choices[answer_index] != row["answer"]):
            raise ValueError("historical_pilot_question_choice_join_mismatch")
        target_key = _video_key(video_map[key[0]])
        if target_key != _video_key(row["video"]):
            raise ValueError("historical_pilot_media_mapping_mismatch")
        candidates = paths.get(target_key, [])
        if len(candidates) != 1:
            raise ValueError("historical_pilot_local_media_missing_or_ambiguous")
        ann = gsub.get(key[0], {})
        spans = ann.get("location", {}).get(key[1]) if isinstance(ann, dict) else None
        duration = ann.get("duration") if isinstance(ann, dict) else None
        if (not isinstance(spans, list) or not spans
                or not isinstance(duration, (int, float)) or not math.isfinite(duration)
                or duration <= 0):
            raise ValueError("historical_pilot_gsub_interval_missing")
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in span)
                    or span[0] < 0 or span[0] >= span[1] or span[1] > duration):
                raise ValueError("historical_pilot_gsub_interval_invalid")
        result.append({"video_id": key[0], "qid": key[1], "question": row["question"],
                       "choices": choices, "gold": answer_index, "spans": spans,
                       "duration": duration, "video_path": candidates[0]})
    # Unique exact-pilot media, disjoint from all three v6 review-media byte hashes.
    for item in result:
        digest = sha256_file(item["video_path"])
        if digest in v6_exclusion_hashes:
            raise ValueError("historical_pilot_overlaps_v6_review_media")
        research_media.add(digest)
    if len(result) != (1 if smoke else GOLD_COHORT_N) or len(research_media) != len(result):
        raise ValueError("historical_pilot_effective_count_or_video_uniqueness_mismatch")
    cohort_digest = _canonical_hash([
        [hashlib.sha256(item["video_id"].encode()).hexdigest(),
         hashlib.sha256(item["qid"].encode()).hexdigest(),
         sha256_file(item["video_path"])] for item in result])
    return result, cohort_digest, dataset_hashes, PILOT_RUN_MANIFEST_SHA256, v6_exclusion_hashes


def _choice_token_ids(tokenizer, rendered_prefix):
    from qta_choice_token_binding import resolve_choice_continuation_ids
    answer_ids, _ = resolve_choice_continuation_ids(tokenizer, rendered_prefix)
    return answer_ids


def _prepare_batch(processor, item, frames, fps, torch):
    from qwen_vl_utils import process_vision_info
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video
    import numpy as np
    rendered, messages = build_qa_prompt(processor, item, frames, fps)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_failed")
    pixels, metadata = videos[0]
    if (not torch.is_tensor(pixels) or int(pixels.shape[0]) != FRAME_COUNT
            or metadata.get("frames_indices") != list(range(FRAME_COUNT))
            or not math.isclose(float(metadata.get("fps", 0)),
                                fps[0] / fps[1], rel_tol=1e-9)):
        raise ValueError("qwen_sampled_frame_indices_mismatch")
    source_signature = torch.stack([torch.from_numpy(np.asarray(frame).copy()).float()
                                    .mean(dim=(0, 1)) for frame in frames])
    pixel_signature = pixels.float().mean(dim=(-1, -2))
    if (pixel_signature.shape != source_signature.shape
            or not torch.allclose(pixel_signature, source_signature, rtol=0, atol=2.0)):
        raise ValueError("qwen_frame_identity_or_order_mismatch")
    batch = process_qa_video(processor, rendered, images, pixels, metadata)
    return rendered, batch


def _forward_score(model, batch, token_ids, torch, *, bias_positions=(),
                   ablation_positions=(), question_positions=(), groups=(), capture=None):
    from qwen_internal_attention_ablation import install_video_key_intervention_hooks, remove_hooks
    from qwen_question_temporal_attention_selector import aggregate_temporal_group_attention
    handles = []
    inter = []
    try:
        if capture is not None:
            for layer in LAYERS_1_BASED:
                index = layer - 1
                def hook(_module, _inputs, output, *, layer_index=index):
                    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                        raise ValueError("baseline_attention_capture_missing")
                    capture[layer_index] = aggregate_temporal_group_attention(
                        attention_by_layer={layer_index: output[1]},
                        layer_indices=(layer_index,), question_positions=question_positions,
                        group_positions=groups, torch_module=torch)
                handles.append(model.model.language_model.layers[index].self_attn
                               .register_forward_hook(hook))
        if bias_positions or ablation_positions:
            inter = install_video_key_intervention_hooks(
                model=model, layer_indices=tuple(index - 1 for index in LAYERS_1_BASED),
                biased_positions=bias_positions, ablated_positions=ablation_positions,
                query_position=int(batch["input_ids"].shape[1]) - 1,
                logit_bias=LOGIT_BIAS, torch_module=torch)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 2 * 1024**3:
            raise RuntimeError(
                "per_forward_free_vram_guard_failed:"
                f"free_bytes={free_bytes};allocated_bytes={torch.cuda.memory_allocated()};"
                f"reserved_bytes={torch.cuda.memory_reserved()}")
        available = next((int(line.split()[1]) * 1024
                          for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemAvailable:")), 0)
        if available < 1024**3:
            raise RuntimeError("per_forward_host_memory_guard_failed")
        with torch.inference_mode():
            output = model(**{key: value.to("cuda") if torch.is_tensor(value) else value
                              for key, value in batch.items()},
                           use_cache=False, logits_to_keep=1)
        logits = output.logits
        if (logits.ndim != 3 or tuple(logits.shape[:2]) != (1, 1)
                or not torch.isfinite(logits).all()):
            raise ValueError("next_token_logits_invalid")
        cpu_logits = logits.detach().to("cpu")
        del output
        return {"logits": cpu_logits}
    finally:
        remove_hooks(handles + inter)


def prepare_geometry(processor, item, frames, decode, torch,
                     evidence_selection_policy="all_annotations"):
    from claim_mask_video_decoder import verify_sampled_frame_pixels
    from qwen_question_temporal_attention_selector import (
        map_question_token_positions)
    from qwen_review_interval_token_groups import map_review_intervals_to_qwen_groups
    verify_sampled_frame_pixels(frames, decode)
    fps = decode["effective_sample_fps"]
    rendered, full_batch = _prepare_batch(processor, item, frames, fps, torch)
    grid = [int(x) for x in full_batch["video_grid_thw"][0].tolist()]
    temporal_steps, height, width = grid
    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    spatial_merge = int(processor.image_processor.merge_size)
    sampled = decode["sampled_frames"]
    if (len(sampled) != FRAME_COUNT or len(frames) != FRAME_COUNT
            or type(decode.get("decoded_frame_count")) is not int
            or decode["decoded_frame_count"] < FRAME_COUNT):
        raise ValueError("sampled_frame_count_invalid")
    pts = [Fraction(*row["relative_presentation_time"]) for row in sampled]
    support_end = temporal_support_end(Fraction(str(item["duration"])), pts)
    frame_indices = [int(row["decoded_frame_index"]) for row in sampled]
    source_frame_count = decode["decoded_frame_count"]
    if (len(set(frame_indices)) != FRAME_COUNT or frame_indices != sorted(frame_indices)
            or any(index < 0 or index >= source_frame_count for index in frame_indices)):
        raise ValueError("sampled_frame_source_indices_invalid")
    mapped = map_review_intervals_to_qwen_groups(
        input_ids=full_batch["input_ids"][0].tolist(),
        video_token_id=int(processor.video_token_id), video_grid_thw=grid,
        spatial_merge_size=spatial_merge, temporal_patch_size=temporal_patch_size,
        sampled_pts_seconds=pts, video_duration_seconds=support_end,
        annotation_duration_seconds=item["duration"],
        reviewed_intervals_seconds=item["spans"], interval_boundary_guard_seconds=0,
        require_processor_uniform_grid_assignment=True,
        evidence_selection_policy=evidence_selection_policy)
    visual = [i for i, token in enumerate(full_batch["input_ids"][0].tolist())
              if token == int(processor.video_token_id)]
    tokens_per_group = (height // spatial_merge) * (width // spatial_merge)
    if len(visual) != temporal_steps * tokens_per_group:
        raise ValueError("qwen_temporal_visual_token_geometry_invalid")
    groups = [tuple(visual[index * tokens_per_group:(index + 1) * tokens_per_group])
              for index in range(temporal_steps)]
    question_positions = map_question_token_positions(
        tokenizer=processor.tokenizer, rendered_text=rendered,
        question_text=item["question"], processed_input_ids=full_batch["input_ids"],
        video_token_id=int(processor.video_token_id))
    token_ids = _choice_token_ids(processor.tokenizer, rendered)
    geometry_fingerprint = _canonical_hash({
        "source_sha256": decode["source_sha256"],
        "grid": grid,
        "input_ids_sha256": hashlib.sha256(
            full_batch["input_ids"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "sampled_pts": [str(value) for value in pts],
        "annotation_duration_seconds": str(item["duration"]),
        "temporal_support_end_seconds": str(support_end),
        "annotation_intervals_seconds": [
            [str(Fraction(str(start))), str(Fraction(str(end)))]
            for start, end in item["spans"]],
        "annotation_covered_groups": mapped["annotation_covered_group_indices"],
        "evidence_groups": mapped["evidence_group_indices"],
        "control_groups": mapped["control_group_indices"],
        "evidence_selection_policy": evidence_selection_policy,
        "representative_overlap_duration_fraction": (
            mapped["representative_annotation_overlap_duration_fraction"]),
        "representative_overlap_fraction": (
            mapped["representative_annotation_overlap_fraction"]),
        "question_positions": question_positions,
        "selector_group_count": len(groups),
    })
    del rendered
    return full_batch, mapped, groups, question_positions, token_ids, geometry_fingerprint


def preflight_cohort(processor, torch, selected, *, smoke=False, counters=None,
                     evidence_selection_policy="all_annotations"):
    """Decode, preprocess, map annotations, and enforce controls before model load."""
    from claim_mask_video_decoder import decode_sampled_video
    from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
    plans = []
    for item in selected:
        if counters is not None:
            counters["attempted"] += 1
        frames, decode = decode_sampled_video(
            item["video_path"], num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        _batch, mapped, groups, _question_positions, _token_ids, fingerprint = (
            prepare_geometry(processor, item, frames, decode, torch,
                             evidence_selection_policy))
        evidence = mapped["evidence_group_indices"]
        control = mapped["control_group_indices"]
        if (not evidence or not control or len(evidence) != len(control)
                or set(evidence) & set(control)
                or any(index >= len(groups) for index in (*evidence, *control))):
            raise ValueError("preflight_equal_count_disjoint_group_control_invalid")
        plans.append({"geometry_sha256": fingerprint,
                      "group_count": len(groups),
                      "annotation_covered_group_count": (
                          mapped["annotation_covered_group_count"]),
                      "representative_group_index": (
                          mapped["representative_annotation_group_index"]),
                      "representative_overlap_fraction": (
                          mapped["representative_annotation_overlap_duration_fraction"]),
                      "evidence_count": len(evidence),
                      "control_count": len(control)})
        if counters is not None:
            counters["completed"] += 1
        del _batch, frames, decode
        gc.collect()
    validate_preflight_plans(plans, len(selected), smoke=smoke)
    return plans


def score_one(model, processor, torch, item, *, frames, decode, expected_plan,
              branch_counts, evidence_selection_policy="all_annotations"):
    from qwen_question_temporal_attention_selector import (
        aggregate_temporal_group_attention, select_top_temporal_groups)
    full_batch, mapped, groups, question_positions, token_ids, fingerprint = (
        prepare_geometry(processor, item, frames, decode, torch,
                         evidence_selection_policy))
    if (fingerprint != expected_plan["geometry_sha256"]
            or len(groups) != expected_plan["group_count"]
            or len(mapped["evidence_group_indices"]) != expected_plan["evidence_count"]
            or len(mapped["control_group_indices"]) != expected_plan["control_count"]):
        raise ValueError("score_input_geometry_changed_after_preflight")
    temporal_steps = len(groups)
    captured = {}
    from nextgqa_atomic_batch_freeze import batch_sha256
    batch_hash = batch_sha256(full_batch)
    branch_batch_hashes = {branch: batch_hash for branch in BRANCH_NAMES}
    validate_shared_batch_fingerprints(branch_batch_hashes)
    outputs = {}
    def run(name, **kwargs):
        branch_counts[name]["attempted"] += 1
        values = _forward_score(model, full_batch, token_ids, torch, **kwargs)
        # The result is copied to CPU by _forward_score; release inactive
        # allocator blocks before the next identical full-batch forward.
        torch.cuda.empty_cache()
        branch_counts[name]["completed"] += 1
        return values
    outputs["baseline_full"] = run(
        "baseline_full", question_positions=question_positions,
        groups=groups, capture=captured)
    if tuple(sorted(captured)) != tuple(index - 1 for index in LAYERS_1_BASED):
        raise ValueError("question_attention_capture_incomplete")
    layer_scores = [captured[layer] for layer in sorted(captured)]
    group_scores = [sum(row[index] for row in layer_scores) / len(layer_scores)
                    for index in range(len(groups))]
    selected_groups = select_top_temporal_groups(
        group_scores, top_k=max(1, math.ceil(len(groups) * 0.10)))
    selected_positions = tuple(sorted(position for group in selected_groups
                                       for position in groups[group]))
    outputs["baseline_evidence_ablation"] = run(
        "baseline_evidence_ablation",
        ablation_positions=mapped["evidence_token_positions"])
    outputs["baseline_control_ablation"] = run(
        "baseline_control_ablation",
        ablation_positions=mapped["control_token_positions"])
    outputs["intervention_full"] = run(
        "intervention_full", bias_positions=selected_positions)
    outputs["intervention_evidence_ablation"] = run(
        "intervention_evidence_ablation", bias_positions=selected_positions,
        ablation_positions=mapped["evidence_token_positions"])
    outputs["intervention_control_ablation"] = run(
        "intervention_control_ablation", bias_positions=selected_positions,
        ablation_positions=mapped["control_token_positions"])
    from fixed_prediction_selective_dependence import score_six_condition_logits
    input_length = int(full_batch["input_ids"].shape[1])
    scored = score_six_condition_logits(
        condition_outputs=outputs, answer_token_ids=token_ids,
        input_lengths={name: input_length for name in BRANCH_NAMES},
        torch_module=torch)
    contrast = scored["contrast"]
    result = {key: contrast[key] for key in (
        "answer_index", "foil_index", "margin_by_condition", "baseline_evidence_drop",
        "baseline_control_drop", "baseline_selective_dependence",
        "intervention_evidence_drop", "intervention_control_drop",
        "intervention_selective_dependence", "delta_f")}
    result.update({"selector_evidence_group_overlap": len(
                       set(selected_groups) & set(mapped["evidence_group_indices"])),
                   "selector_control_group_overlap": len(
                       set(selected_groups) & set(mapped["control_group_indices"])),
                   "annotation_covered_group_indices": (
                       mapped["annotation_covered_group_indices"]),
                   "evidence_group_indices": mapped["evidence_group_indices"],
                   "control_group_indices": mapped["control_group_indices"],
                   "evidence_group_count": len(mapped["evidence_group_indices"]),
                   "control_group_count": len(mapped["control_group_indices"]),
                   "annotation_covered_group_count": (
                       mapped["annotation_covered_group_count"]),
                   "representative_annotation_group_index": (
                       mapped["representative_annotation_group_index"]),
                   "representative_annotation_overlap_duration_fraction": (
                       mapped["representative_annotation_overlap_duration_fraction"]),
                   "representative_annotation_overlap_fraction": (
                       mapped["representative_annotation_overlap_fraction"]),
                   "representative_group_position_fraction": (
                       mapped["representative_annotation_group_index"]
                       / max(1, temporal_steps - 1)
                       if mapped["representative_annotation_group_index"] is not None
                       else None),
                   "temporal_group_count": temporal_steps,
                   "support_over_annotation_duration_seconds": (
                       mapped["support_duration_seconds"]
                       - mapped["annotation_duration_seconds"])})
    result["baseline_full_correct"] = int(contrast["answer_index"] == item["gold"])
    del full_batch, outputs
    gc.collect()
    return result


def write_private_output(repo: Path, filename: str, payload: dict):
    """Create an exclusive mode-0600 file directly under private_nextgqa."""
    validate_private_output_name(filename)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    repo_fd = os.open(repo, directory_flags)
    try:
        data_fd = os.open("data", directory_flags, dir_fd=repo_fd)
        try:
            private_fd = os.open("private_nextgqa", directory_flags, dir_fd=data_fd)
        finally:
            os.close(data_fd)
    finally:
        os.close(repo_fd)
    try:
        root_stat = os.fstat(private_fd)
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise ValueError("private_output_root_invalid")
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=private_fd)
        try:
            data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
    finally:
        os.close(private_fd)


def validate_private_output_name(filename: str) -> None:
    if (not isinstance(filename, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", filename)):
        raise ValueError("private_output_filename_invalid")


def assert_private_output_slot_free(repo: Path, filename: str) -> None:
    """Fail before data/model work when a receipt name already exists."""
    validate_private_output_name(filename)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    repo_fd = os.open(repo, directory_flags)
    try:
        data_fd = os.open("data", directory_flags, dir_fd=repo_fd)
        try:
            private_fd = os.open("private_nextgqa", directory_flags, dir_fd=data_fd)
        finally:
            os.close(data_fd)
    finally:
        os.close(repo_fd)
    try:
        root_stat = os.fstat(private_fd)
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise ValueError("private_output_root_invalid")
        try:
            os.stat(filename, dir_fd=private_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError("private_output_collision")
    finally:
        os.close(private_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(1, 30), required=True,
                        help="1 is engineering smoke; 30 is the only cohort analysis")
    parser.add_argument("--output-name", required=True,
                        help="new filename directly under data/private_nextgqa")
    parser.add_argument("--source-capsule-dir", required=True,
                        help="existing owner-only mode-0700 directory outside the Git worktree")
    parser.add_argument("--pilot-manifest", default=None,
                        help="defaults to the fixed private manifest inside this repository")
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    parser.add_argument("--evidence-selection-policy",
                        choices=("all_annotations", "single_overlap_anchor"),
                        default="all_annotations",
                        help="frozen annotation/control mapping policy")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    try:
        assert_private_output_slot_free(repo, args.output_name)
    except FileExistsError:
        parser.error("private_output_collision")
    except ValueError:
        parser.error("private_output_contract_invalid")
    pilot_manifest = resolve_pilot_manifest(repo, args.pilot_manifest)
    smoke = args.n == 1
    branch_counts = {name: {"requested": args.n, "attempted": 0, "completed": 0}
                     for name in BRANCH_NAMES}
    preflight_counts = {"attempted": 0, "completed": 0}
    stage = "source_preflight"
    source_capsule = None
    outputs = []
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        commit, executed_source_hashes = source_hashes(repo)
        stage = "source_capsule_capture_verify"
        source_capsule = capture_verified_source_capsule(
            repo, args.source_capsule_dir, args.output_name, executed_source_hashes)
        selected, cohort_sha, dataset_hashes, run_manifest_hash, v6_exclusion_hashes = load_fixed_rows(
            repo, pilot_manifest, Path(args.video_root), smoke=smoke)
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        stage = "checkpoint_byte_audit"
        checkpoint_audit = audit_pinned_checkpoint(repo, model_path)
        import torch
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, min_pixels=256 * 28 * 28,
            max_pixels=256 * 28 * 28)
        processor_pins = processor_provenance(processor, model_path)
        # The complete fixed denominator is decoded, processed, and mapped before
        # model construction. The smoke runs exactly the first fixed row only.
        stage = "full_cpu_preflight"
        plans = preflight_cohort(processor, torch, selected, smoke=smoke,
                                 counters=preflight_counts,
                                 evidence_selection_policy=args.evidence_selection_policy)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA_BF16_required_no_CPU_fallback")
        free, total = torch.cuda.mem_get_info()
        if free < 11 * 1024**3:
            raise ValueError("insufficient_free_vram_before_model_load")
        stage = "model_load"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
            raise ValueError("model_processor_layer_contract_invalid")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        from claim_mask_video_decoder import decode_sampled_video
        from run_nextgqa_gold_interval_margin_pilot import UNIFORM_TOLERANCE
        stage = "six_branch_score"
        for ordinal, (item, plan) in enumerate(zip(selected, plans, strict=True), 1):
            frames, decode = decode_sampled_video(
                item["video_path"], num_frames=FRAME_COUNT,
                uniform_tolerance_seconds=UNIFORM_TOLERANCE)
            row_result = score_one(
                model, processor, torch, item, frames=frames, decode=decode,
                expected_plan=plan, branch_counts=branch_counts,
                evidence_selection_policy=args.evidence_selection_policy)
            row_result["ordinal"] = ordinal
            outputs.append(row_result)
            del frames, decode
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if (len(outputs) != args.n or any(branch_counts[name]["completed"] != args.n
                                         for name in BRANCH_NAMES)):
            raise ValueError("complete_six_branch_denominator_not_met")
        values = [row["delta_f"] for row in outputs]
        payload = {
            "status": "engineering_smoke_only" if smoke else (
                "completed_exploratory_finite_cohort" if args.evidence_selection_policy == "all_annotations"
                else "completed_exploratory_single_anchor_finite_cohort"),
            "scope": ("NExT-GQA annotation-interval dependence proxy; not explanation faithfulness"),
            "evidence_selection_policy": args.evidence_selection_policy,
            "authorization": "local_only_user_authorized; rights_status_unverified",
            "cohort": "exact_2026-10-08_validation_pilot_manifest",
            "requested_n": args.n, "effective_n": len(outputs), "failures": 0,
            "preflight_requested": args.n, "preflight_attempted": preflight_counts["attempted"],
            "preflight_completed": preflight_counts["completed"],
            "branch_counts": {name: {**counts, "effective": len(outputs)}
                              for name, counts in branch_counts.items()},
            "forward_plan": (
                "baseline_and_QTA_full_plus_one_representative_group_and_one_uncovered_control"
                if args.evidence_selection_policy == "single_overlap_anchor" else
                "baseline_and_QTA_full_plus_all_annotation_groups_and_equal_count_control"),
            "requested_forwards": 6 * args.n,
            "attempted_forwards": sum(branch_counts[name]["attempted"] for name in BRANCH_NAMES),
            "completed_forwards": sum(branch_counts[name]["completed"] for name in BRANCH_NAMES),
            "primary": "delta_f = (QTA_control_margin - QTA_evidence_margin) - (baseline_control_margin - baseline_evidence_margin); p/q frozen from baseline full",
            "records": outputs,
            "finite_cohort_delta_f": None if smoke else summarize(values),
            "finite_cohort_baseline_full_restricted_choice_accuracy":
                summarize_baseline_accuracy(outputs, args.n, smoke=smoke),
            "anchor_selection_descriptive": (
                {"selected_annotation_overlap_fraction": summarize([
                    float(Fraction(row["representative_annotation_overlap_fraction"]))
                    for row in outputs]),
                 "representative_group_position_fraction": summarize([
                    row["representative_group_position_fraction"] for row in outputs]),
                 "covered_group_count": summarize([
                    row["annotation_covered_group_count"] for row in outputs])}
                if args.evidence_selection_policy == "single_overlap_anchor" and not smoke
                else None),
            "inference": "engineering_only" if smoke else "descriptive_only_no_p_value_or_population_CI",
            "started_utc": started, "elapsed_seconds": time.monotonic() - start,
            "hardware": {"device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                         "free_vram_before_load_gib": free / 1024**3,
                         "total_vram_gib": total / 1024**3,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "model_revision": MODEL_REVISION, "precision": "4bit_bf16_compute_bf16_eager",
            "processor_provenance": processor_pins,
            "seed": SEED, "frame_count": FRAME_COUNT,
            "qta_layers_1_based": list(LAYERS_1_BASED), "qta_logit_bias": LOGIT_BIAS,
            "qta_budget": "max(1,ceil(0.10*T)); fixed from baseline question-token attention",
            "intervention": "attention-key ablation on same original Qwen processor batch; no pixel masking",
            "historical_pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "historical_pilot_run_manifest_sha256": run_manifest_hash,
            "protocol_sha256": executed_source_hashes[
                "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md"],
            "treatment_selection_sha256": executed_source_hashes[
                "experiments/2026-10-18_qta_treatment_selection.md"],
            "anchor_protocol_sha256": executed_source_hashes[
                "experiments/2026-10-18_qta_annotation_anchor_proxy_v1.md"],
            "dataset_source_sha256": dataset_hashes,
            "cohort_fingerprint_sha256": cohort_sha,
            "v6_inventory_sha256": V6_INVENTORY_SHA256,
            "v6_source_provenance_sha256": V6_PROVENANCE_SHA256,
            "v6_exclusion_set_sha256": _canonical_hash(sorted(v6_exclusion_hashes)),
            "checkpoint_byte_audit": checkpoint_audit,
            "source_commit": commit, "source_sha256": executed_source_hashes,
            "source_capsule": source_capsule,
            "environment": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "bitsandbytes", "av", "qwen-vl-utils")},
            "execution_authorization": "local_only_user_authorized",
            "interpretation_limit": "reused development pilot; gsub annotation dependence only; no generated-claim semantics, explanation faithfulness, fresh-cohort inference, or causal efficacy claim",
            "provenance_limit": "processor/tokenizer/config hashes are measured for this run and bound to the pinned local revision; no independently predeclared expected digest for these artifacts was available",
        }
    except Exception as exc:
        if isinstance(exc, SourceCapsuleMismatch):
            source_capsule = exc.source_capsule
        receipt = failure_receipt(requested_n=args.n, stage=stage,
                                  error_type=type(exc).__name__,
                                  preflight_attempted=preflight_counts["attempted"],
                                  preflight_completed=preflight_counts["completed"],
                                  branch_counts=branch_counts,
                                  failure_detail=safe_failure_detail(exc),
                                  evidence_selection_policy=args.evidence_selection_policy,
                                  source_capsule=source_capsule)
        try:
            write_private_output(repo, args.output_name, receipt)
        except Exception as write_exc:
            parser.error(f"private_failure_receipt_write_failed:{type(write_exc).__name__}")
        print(f"Incomplete run; private failure receipt={args.output_name}; no estimate written")
        return 2
    try:
        write_private_output(repo, args.output_name, payload)
    except (OSError, ValueError) as exc:
        parser.error(f"private_output_failed:{type(exc).__name__}")
    print(f"Completed {len(outputs)} rows; private summary={args.output_name}; no raw QA/media/logits written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
