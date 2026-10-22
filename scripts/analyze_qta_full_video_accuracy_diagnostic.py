#!/usr/bin/env python3
"""Verify and summarize the local QTA full-video accuracy receipt.

The analyzer prints aggregate QA-choice counts only. It verifies private
receipt/anchor/source bindings and recomputes the finite-cohort arithmetic;
it does not establish semantic faithfulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_private(repo: Path, relative: str) -> tuple[dict, str]:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("private_path_invalid")
    path = repo / candidate
    private = (repo / "data/private_nextgqa").resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(private)
    except (OSError, ValueError) as exc:
        raise ValueError("private_path_invalid") from exc
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_file_permissions_invalid")
        blocks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(blocks)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("private_json_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def _check_choice(condition: dict, gold: int) -> bool:
    values = condition.get("choice_logprobs_nats")
    if (not isinstance(values, list) or len(values) != 5
            or any(type(x) not in (int, float) or not math.isfinite(float(x)) for x in values)):
        raise ValueError("choice_vector_invalid")
    predicted = max(range(5), key=lambda i: (float(values[i]), -i))
    if predicted != condition.get("prediction_index"):
        raise ValueError("prediction_argmax_mismatch")
    margin = float(values[gold]) - max(float(values[i]) for i in range(5) if i != gold)
    if not math.isclose(margin, float(condition.get("gold_minus_best_foil_margin_nats")),
                        rel_tol=0, abs_tol=1e-10):
        raise ValueError("choice_margin_mismatch")
    correct = predicted == gold
    if condition.get("gold_choice_correct") is not correct:
        raise ValueError("choice_correctness_mismatch")
    if not math.isclose(float(values[gold]), float(condition.get("gold_logprob_nats")),
                        rel_tol=0, abs_tol=1e-10):
        raise ValueError("gold_logprob_mismatch")
    return correct


def analyze(repo: Path, relative: str) -> dict:
    receipt, digest = _read_private(repo, relative)
    if (receipt.get("status") != "completed_exploratory_full_video_answer_accuracy"
            or receipt.get("requested_records") != 27
            or receipt.get("effective_records") != 27
            or receipt.get("attempted_forwards") != 81
            or receipt.get("completed_forwards") != 81):
        raise ValueError("completed_cohort_required")
    for name, expected in receipt.get("source_sha256", {}).items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_path_invalid")
        source = (repo / path).resolve(strict=True)
        try:
            source.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError("source_path_invalid") from exc
        if _hash(source) != expected:
            raise ValueError("source_hash_mismatch")
    anchors = {}
    for item in receipt.get("anchor_receipts", []):
        anchor, anchor_hash = _read_private(repo, item["path"])
        if (anchor_hash != item["sha256"]
                or anchor.get("status") != "completed_exploratory_internal_key_ablation"
                or len(anchor.get("records", [])) != 3):
            raise ValueError("anchor_receipt_invalid")
        anchors[Path(item["path"]).stem] = anchor
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != 27:
        raise ValueError("record_count_invalid")
    if (len({r.get("row_token_sha256") for r in records}) != 27
            or len({r.get("source_video_sha256") for r in records}) != 27):
        raise ValueError("rows_not_unique")
    correct = {name: 0 for name in ("baseline", "qta", "sham")}
    flips = {"baseline_to_qta_wrong_to_correct": 0,
             "baseline_to_qta_correct_to_wrong": 0,
             "qta_to_sham_prediction_changes": 0,
             "baseline_to_qta_prediction_changes": 0}
    max_margin_change = {"qta_vs_baseline": 0.0, "sham_vs_baseline": 0.0}
    for record in records:
        anchor = anchors.get(record.get("cohort"))
        pos = record.get("row_position")
        if anchor is None or type(pos) is not int or pos not in (1, 2, 3):
            raise ValueError("anchor_row_reference_invalid")
        anchor_row = anchor["records"][pos - 1]
        if (record.get("row_token_sha256") != anchor_row.get("row_token_sha256")
                or record.get("source_video_sha256") != anchor_row.get("source_video_sha256")
                or record.get("input_batch_sha256") != anchor_row.get("input_batch_sha256")
                or record.get("qta_group_indices") != anchor_row.get("question_attention_group_indices")):
            raise ValueError("anchor_row_binding_mismatch")
        gold = record.get("gold_choice_index")
        if type(gold) is not int or not 0 <= gold < 5:
            raise ValueError("gold_choice_index_invalid")
        conditions = record.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != {"baseline", "qta", "sham"}:
            raise ValueError("condition_set_invalid")
        for name in correct:
            correct[name] += int(_check_choice(conditions[name], gold))
        for name in ("qta", "sham"):
            audit = conditions[name].get("mask_audit", {})
            hook_counts = audit.get("hook_calls_by_layer", {})
            if (len(hook_counts) != 12 or set(hook_counts.values()) != {1}
                    or audit.get("effective_mask_bias") != 0.69140625):
                raise ValueError("attention_mask_audit_invalid")
        qta_audit = conditions["qta"]["mask_audit"]
        sham_audit = conditions["sham"]["mask_audit"]
        if (qta_audit.get("biased_key_count") != sham_audit.get("biased_key_count")
                or set(record["qta_group_indices"]) == set(record["sham_group_indices"])):
            raise ValueError("sham_budget_or_independence_invalid")
        evidence, control = set(anchor_row["evidence_group_indices"]), set(anchor_row["control_group_indices"])
        qta, sham = set(record["qta_group_indices"]), set(record["sham_group_indices"])
        if (len(qta) != len(sham) or len(qta & evidence) != len(sham & evidence)
                or len(qta & control) != len(sham & control)):
            raise ValueError("sham_overlap_match_invalid")
        predictions = {name: conditions[name]["prediction_index"] for name in correct}
        flips["baseline_to_qta_wrong_to_correct"] += int(
            not conditions["baseline"]["gold_choice_correct"]
            and conditions["qta"]["gold_choice_correct"])
        flips["baseline_to_qta_correct_to_wrong"] += int(
            conditions["baseline"]["gold_choice_correct"]
            and not conditions["qta"]["gold_choice_correct"])
        flips["baseline_to_qta_prediction_changes"] += int(predictions["baseline"] != predictions["qta"])
        flips["qta_to_sham_prediction_changes"] += int(predictions["qta"] != predictions["sham"])
        max_margin_change["qta_vs_baseline"] = max(max_margin_change["qta_vs_baseline"],
            abs(float(conditions["qta"]["gold_minus_best_foil_margin_nats"])
                - float(conditions["baseline"]["gold_minus_best_foil_margin_nats"])))
        max_margin_change["sham_vs_baseline"] = max(max_margin_change["sham_vs_baseline"],
            abs(float(conditions["sham"]["gold_minus_best_foil_margin_nats"])
                - float(conditions["baseline"]["gold_minus_best_foil_margin_nats"])))
    result = {"receipt_sha256": digest, "records": len(records),
        "forwards": receipt["completed_forwards"], "correct": correct,
        "accuracy": {name: correct[name] / len(records) for name in correct},
        "paired_flip_counts": flips, "max_absolute_gold_margin_shift_nats": max_margin_change,
        "runtime_seconds": receipt["runtime_seconds"],
        "peak_vram_gib": receipt["peak_vram_allocated_bytes"] / 1024**3,
        "direct_faithfulness": False,
        "interpretation": "post-hoc finite-cohort QA accuracy only; not explanation faithfulness"}
    stored = receipt.get("finite_cohort_summary", {})
    for name in correct:
        cell = stored.get(name, {})
        if cell.get("correct") != correct[name] or cell.get("n") != len(records):
            raise ValueError("stored_accuracy_mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?",
        default="data/private_nextgqa/qta_full_video_accuracy_full_20261021_02.json")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(analyze(repo, args.receipt), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
