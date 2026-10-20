"""Synthetic, valid v3 baseline manifest fixture for review-pipeline tests."""
from __future__ import annotations

import hashlib
import json

from qta_selection_provenance import FROZEN_SELECTOR_SOURCE_SHA256


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def build_synthetic_baseline_v3(video_review_sha256: str):
    rows = []
    for index in range(3):
        token = f"{index + 1:024d}"
        question = ("What happened?", "Where is the object?", "What color is it?")[index]
        options = (["A", "B", "C", "D", "E"],
                   ["left", "right", "up", "down", "center"],
                   ["red", "blue", "green", "white", "black"])[index]
        claim = ("A person waves.", "The item is on the left.",
                 "The object is blue.")[index]
        scores = [-1.0, -1.2, -2.0, -2.5, -3.0]
        interval_sha = "d" * 64
        positions = [2, 3, 4]
        binding = {
            "schema_version": 2,
            "review_token": token,
            "answer_label_order": ["A", "B", "C", "D", "E"],
            "answer_token_ids": [100 + 5 * index + offset for offset in range(5)],
            "choice_token_binding_sha256": "1" * 64,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "options_sha256": hashlib.sha256(_canonical(options)).hexdigest(),
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "source_video_sha256": f"{index + 2:064x}",
            "reviewed_intervals_sha256": interval_sha,
            "video_support_decision": "supports_claim",
            "prompt_sha256": "3" * 64,
            "input_ids_sha256": "4" * 64,
            "input_length": 11,
            "score_position": 10,
            "score_input_token_id": 100 + index,
            "video_token_id": 32000,
            "question_positions": positions,
            "question_positions_sha256": hashlib.sha256(
                _canonical(positions)).hexdigest(),
            "question_position_mapper_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
            "processor_batch_sha256": "5" * 64,
            "option_log_probabilities": scores,
            "p_index": 0,
            "q_index": 1,
            "run_digests": {
                "protocol_sha256": "6" * 64,
                "model_sha256": "7" * 64,
                "tokenizer_sha256": "8" * 64,
                "scorer_sha256": "9" * 64,
                "processor_sha256": "a" * 64,
                "runtime_sha256": "b" * 64,
            },
        }
        rows.append({
            "review_token": token,
            "question": question,
            "options": options,
            "claim": claim,
            "option_log_probabilities": scores,
            "p_index": 0,
            "q_index": 1,
            "video_support_decision": "supports_claim",
            "reviewed_intervals_sha256": interval_sha,
            "score_binding": binding,
            "score_binding_sha256": hashlib.sha256(_canonical(binding)).hexdigest(),
        })

    manifest = {
        "schema_version": 3,
        "purpose": "qta_baseline_prediction_link_manifest_v3",
        "video_claim_review_receipt_sha256": video_review_sha256,
        "records": rows,
    }
    manifest_raw = _canonical(manifest)
    source_hashes = {
        row["review_token"]: {
            "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
            "options_sha256": hashlib.sha256(_canonical(row["options"])).hexdigest(),
            "claim_sha256": hashlib.sha256(row["claim"].encode()).hexdigest(),
            "video_support_decision": row["video_support_decision"],
            "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
        } for row in rows
    }
    return manifest_raw, rows, source_hashes
