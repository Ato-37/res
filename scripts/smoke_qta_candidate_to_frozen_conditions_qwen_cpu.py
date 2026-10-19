#!/usr/bin/env python3
"""Synthetic CPU-only composition smoke for QTA candidate -> 3x3 freeze.

Uses the locally cached Qwen processor and generated RGB clips only. It never
loads model weights or protected media. The two freeze callbacks are test
stubs; a passing result is not execution authorization or faithfulness evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from nextgqa_atomic_batch_freeze import (
    CONDITIONS, batch_sha256, freeze_all_batches,
)
from nextgqa_condition_batch_builder import build_exact_triplet_batches
from qta_actual_processor_row_candidate import (
    build_actual_processor_row_candidate,
    verify_actual_processor_row_candidate,
)
from qta_prescore_mapping_binding import interval_list_sha256


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_fixtures(processor, snapshot: Path):
    tokenizer = processor.tokenizer
    tokenizer_sha = _file_sha256(snapshot / "tokenizer.json")
    processor_files = ("preprocessor_config.json", "tokenizer.json",
                      "tokenizer_config.json", "chat_template.json",
                      "merges.txt", "vocab.json")
    processor_sha = hashlib.sha256(_canonical([
        [name, _file_sha256(snapshot / name)] for name in processor_files
    ])).hexdigest()
    revision = snapshot.name
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    spatial_merge_size = int(processor.image_processor.merge_size)

    candidate_rows, expected_rows, sampler_rows, condition_records = [], [], [], []
    for row_index in range(3):
        token = f"synthetic-row-{row_index + 1}"
        source_sha = hashlib.sha256(f"synthetic-source-{row_index + 1}".encode()).hexdigest()
        question = f"Which color appears in generated clip {row_index + 1}?"
        choices = [f"color {letter}" for letter in "ABCDE"]
        claim = f"Generated clip {row_index + 1} changes color over time."
        frames = [Image.new("RGB", (56, 56),
                            ((frame_index * 7) % 256,
                             (row_index * 37 + frame_index * 3) % 256,
                             ((frame_index + row_index) * 13) % 256))
                  for frame_index in range(32)]
        pts = [Fraction(frame_index, 2) for frame_index in range(32)]
        interval = [[2, 6]]
        sampler = {
            "source_sha256": source_sha,
            "decoded_frame_count": 32,
            "sampled_frame_count": 32,
            "sampled_indices": list(range(32)),
            "sampled_pts": [[value.numerator, value.denominator] for value in pts],
            "effective_sample_fps": [2, 1],
            "sampled_rgb_sha256": [
                hashlib.sha256(frame.convert("RGB").tobytes()).hexdigest()
                for frame in frames
            ],
        }
        sampler_rows.append(sampler)
        candidate_rows.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question": question,
            "choices": choices,
            "claim": claim,
            "frames": frames,
            "video_duration_seconds": 16,
            "reviewed_intervals_seconds": interval,
            "interval_boundary_guard_seconds": 0,
            "processor_revision": revision,
            "processor_fingerprint_sha256": processor_sha,
            "video_token_id": video_token_id,
            "spatial_merge_size": spatial_merge_size,
            "temporal_patch_size": temporal_patch_size,
        })
        expected_rows.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "options_sha256": hashlib.sha256(_canonical(choices)).hexdigest(),
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "reviewed_intervals_sha256": interval_list_sha256([["2", "6"]]),
            "tokenizer_sha256": tokenizer_sha,
            "processor_revision": revision,
            "processor_fingerprint_sha256": processor_sha,
            "sampler_row_sha256": hashlib.sha256(_canonical(sampler)).hexdigest(),
            "video_duration_seconds": "16",
            "interval_boundary_guard_seconds": "0",
        })
        condition_records.append({
            "frames": frames,
            "presentation_times": pts,
            "sampled_indices": list(range(32)),
            "intervals": [[Fraction(2), Fraction(6)]],
            "qa_item": {"question": question, "choices": choices,
                        "gold": row_index},
            "sample_fps": Fraction(2, 1),
        })
    return (candidate_rows, expected_rows, sampler_rows, condition_records,
            tokenizer_sha)


def main() -> int:
    default_snapshot = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen2.5-VL-7B-Instruct" / "snapshots"
        / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, default=default_snapshot)
    args = parser.parse_args()
    snapshot = args.model_snapshot.resolve(strict=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("expected_cpu_only_but_cuda_was_initialized")

    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    (source_rows, expected_rows, sampler_rows, records,
     tokenizer_sha) = _make_fixtures(processor, snapshot)
    candidate = build_actual_processor_row_candidate(
        rows=source_rows, expected_rows=expected_rows,
        sampler_rows=sampler_rows, processor=processor,
        tokenizer=processor.tokenizer, tokenizer_sha256=tokenizer_sha,
        vision_info_fn=process_vision_info, torch_module=torch,
    )
    verify_actual_processor_row_candidate(candidate, expected_rows=expected_rows)

    conditions = build_exact_triplet_batches(
        records=records, processor=processor, vision_info_fn=process_vision_info)
    if (candidate["requested_rows"] != 3 or len(candidate["candidate_rows"]) != 3
            or conditions["requested_records"] != 3
            or conditions["requested_batches"] != 9
            or conditions["model_forwards"] != 0
            or conditions["execution_authorization"] != "not_granted"):
        raise AssertionError("three_by_three_shape_or_authority_contract_failed")

    record_ids = [row["review_token"] for row in candidate["candidate_rows"]]
    triplets = []
    for position, (candidate_row, condition_row, record_id) in enumerate(
            zip(candidate["candidate_rows"], conditions["triplets"], record_ids, strict=True)):
        candidate_full_sha = batch_sha256(candidate_row["batch"])
        built_full_sha = batch_sha256(condition_row["batches"]["full"])
        if candidate_full_sha != built_full_sha:
            raise AssertionError(f"candidate_full_batch_mismatch_row_{position + 1}")
        if candidate_row["answer_token_ids"] != candidate["candidate_rows"][0]["answer_token_ids"]:
            raise AssertionError("answer_label_token_contract_varies_by_row")
        if candidate_row["input_ids_sha256"] != hashlib.sha256(
                _canonical(condition_row["batches"]["full"]["input_ids"][0]
                           .detach().cpu().tolist())).hexdigest():
            raise AssertionError(f"prompt_token_ids_mismatch_row_{position + 1}")
        triplets.append({"record_id": record_id,
                         "gold_index": condition_row["gold_index"],
                         "batches": condition_row["batches"]})

    token_ids = candidate["candidate_rows"][0]["answer_token_ids"]
    hashes = {
        row["record_id"]: {condition: batch_sha256(row["batches"][condition])
                            for condition in CONDITIONS}
        for row in triplets
    }
    record_ids = [row["record_id"] for row in triplets]
    callback_counts = {"precheck": 0, "authenticated": 0}

    def precheck():
        callback_counts["precheck"] += 1
        return True  # synthetic harness only; not a real authorization gate

    def authenticated():
        callback_counts["authenticated"] += 1
        return True  # synthetic harness only; not a real authorization gate

    frozen = freeze_all_batches(
        triplets=triplets, token_ids=token_ids,
        expected_record_ids=record_ids, expected_batch_hashes=hashes,
        upstream_precheck=precheck, authenticated_gate_check=authenticated)
    if (frozen["status"] != "all_nine_batches_frozen"
            or frozen["requested_records"] != 3
            or frozen["requested_forwards"] != 9
            or frozen["execution_authorization"] != "not_granted"
            or callback_counts != {"precheck": 1, "authenticated": 1}
            or candidate["model_load_permitted"]
            or candidate["scoring_authorized"]
            or candidate["faithfulness_scored"]):
        raise AssertionError("atomic_freeze_or_non_authorization_contract_failed")

    print(json.dumps({
        "status": "synthetic_candidate_to_3x3_freeze_smoke_passed",
        "processor_revision": snapshot.name,
        "rows": 3,
        "condition_batches": 9,
        "candidate_full_batch_hashes_equal_builder_full": True,
        "ordered_record_ids": record_ids,
        "atomic_freeze_composition": "passed_with_synthetic_gate_stubs",
        "synthetic_gate_callbacks_only": True,
        "model_weights_loaded": False,
        "model_forwards": 0,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
        "protected_media_used": False,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
