"""Compute fixed five-choice margins for a preflighted 3 x 3 input bundle.

This module is the inference/math core only. It does not authenticate inputs,
load a model, authorize execution, or make a draft protocol scoreable. A
production caller must complete and hash-freeze every all-row review, clock,
source, target, condition-input, and code gate before loading model weights.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256


CONDITIONS = ("full", "claim_evidence_black", "temporal_control_black")
REQUESTED_RECORDS = 3
LABEL_COUNT = 5


def validate_scoring_inputs(triplets: Any, token_ids: Any) -> None:
    """Validate the fixed cohort's already-frozen CPU inputs before forwarding.

    Do not call ``FrozenBatch.to('cpu')`` here: the freezer has already made
    the authoritative CPU snapshot, and ``FrozenBatch.to`` clones tensors.
    Hashing the existing snapshot below is sufficient to validate its bytes.
    """
    if (not isinstance(token_ids, (list, tuple)) or len(token_ids) != LABEL_COUNT
            or any(type(token) is not int or token < 0 for token in token_ids)
            or len(set(token_ids)) != LABEL_COUNT):
        raise ValueError("answer_label_token_contract_invalid")
    if not isinstance(triplets, (list, tuple)) or len(triplets) != REQUESTED_RECORDS:
        raise ValueError("requested_record_count_mismatch")
    for row_index, row in enumerate(triplets):
        if not isinstance(row, Mapping) or set(row) != {"gold_index", "batches"}:
            raise ValueError(f"record_{row_index + 1}_schema_invalid")
        gold = row["gold_index"]
        if type(gold) is not int or not 0 <= gold < LABEL_COUNT:
            raise ValueError(f"record_{row_index + 1}_gold_index_invalid")
        batches = row["batches"]
        if not isinstance(batches, Mapping) or set(batches) != set(CONDITIONS):
            raise ValueError(f"record_{row_index + 1}_condition_set_invalid")
        for condition in CONDITIONS:
            batch = batches[condition]
            if (type(batch) is not FrozenBatch
                    or not isinstance(batch, Mapping) or not batch):
                raise ValueError(f"record_{row_index + 1}_{condition}_batch_invalid")
            _validate_cpu_tree(batch, row_index + 1, condition)


def _validate_cpu_tree(value: Any, row_position: int, condition: str) -> None:
    """Reject non-CPU tensor leaves without copying the frozen snapshot."""
    if (callable(getattr(value, "detach", None))
            and callable(getattr(value, "cpu", None))
            and hasattr(value, "device")):
        if str(value.device) != "cpu":
            raise ValueError(f"record_{row_position}_{condition}_batch_not_cpu")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_cpu_tree(child, row_position, condition)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_cpu_tree(child, row_position, condition)


def _margin(log_probabilities: list[float], gold_index: int) -> float:
    gold = log_probabilities[gold_index]
    foil = max(value for index, value in enumerate(log_probabilities)
               if index != gold_index)
    return gold - foil


def _validate_frozen_hashes(*, triplets: Any, record_ids: Any,
                            expected_batch_hashes: Any) -> None:
    if (not isinstance(record_ids, (list, tuple)) or len(record_ids) != REQUESTED_RECORDS
            or any(not isinstance(value, str) or not value for value in record_ids)
            or len(set(record_ids)) != REQUESTED_RECORDS
            or not isinstance(expected_batch_hashes, Mapping)
            or set(expected_batch_hashes) != set(record_ids)):
        raise ValueError("frozen_batch_hash_binding_invalid")
    for row_position, (record_id, row) in enumerate(zip(record_ids, triplets, strict=True), 1):
        expected_conditions = expected_batch_hashes[record_id]
        if (not isinstance(expected_conditions, Mapping)
                or set(expected_conditions) != set(CONDITIONS)):
            raise ValueError(f"record_{row_position}_frozen_hash_conditions_invalid")
        for condition in CONDITIONS:
            expected = expected_conditions[condition]
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(char not in "0123456789abcdef" for char in expected)):
                raise ValueError(f"record_{row_position}_{condition}_frozen_hash_invalid")
            try:
                observed = batch_sha256(row["batches"][condition])
            except Exception:
                raise ValueError(f"record_{row_position}_{condition}_frozen_hash_unavailable") from None
            if observed != expected:
                raise ValueError(f"record_{row_position}_{condition}_frozen_hash_mismatch")


def score_preflighted_triplets(*, model: Any, triplets: Any, token_ids: Any,
                               record_ids: Any, expected_batch_hashes: Any,
                               torch_module: Any, device: str = "cuda",
                               before_forward: Callable[[int, int, str], None] | None = None,
                               model_call: Callable[..., Any] | None = None,
                               ) -> dict:
    """Run the fixed nine forwards and return a finite-cohort descriptive result.

    ``before_forward`` is a caller-provided live resource guard; it receives
    1-based row/condition positions and runs immediately before each forward.
    A failure stops execution and leaves ``D3`` null. This function assumes the
    caller has already authenticated and frozen all nine exact model inputs.
    """
    # Gate the exact freezer-produced type before generic validation calls
    # `.to()`. This excludes subclasses that could override transfer behavior.
    if isinstance(triplets, (list, tuple)) and len(triplets) == REQUESTED_RECORDS:
        for row in triplets:
            if isinstance(row, Mapping) and isinstance(row.get("batches"), Mapping):
                if any(type(batch) is not FrozenBatch for batch in row["batches"].values()):
                    raise ValueError("frozen_batch_snapshot_required")
    validate_scoring_inputs(triplets, token_ids)
    # Recheck all nine snapshots before the first forward, then hash each
    # device batch after transfer to catch later edits and transfer changes.
    _validate_frozen_hashes(triplets=triplets, record_ids=record_ids,
                            expected_batch_hashes=expected_batch_hashes)
    if model is None or torch_module is None or not isinstance(device, str) or not device:
        raise ValueError("scoring_runtime_invalid")

    try:
        token_tensor = torch_module.tensor(list(token_ids), device=device)
    except Exception as exc:
        return {
            "status": "failed",
            "scope": "finite_three_record_descriptive_only",
            "requested_records": REQUESTED_RECORDS,
            "cohort_denominator": REQUESTED_RECORDS,
            "requested_forwards": REQUESTED_RECORDS * len(CONDITIONS),
            "attempted_forwards": 0,
            "completed_forwards": 0,
            "completed_triplets": 0,
            "failure_code": "answer_token_tensor_creation_failed_" + type(exc).__name__,
            "records": [],
            "D3_control_minus_claim_mask_nats": None,
            "confidence_interval": None,
            "p_value": None,
            "execution_authorization": "not_granted",
        }
    records: list[dict] = []
    attempted = completed = 0
    failure: str | None = None
    for row_position, row in enumerate(triplets, start=1):
        row_scores: dict[str, list[float]] = {}
        for condition_position, condition in enumerate(CONDITIONS, start=1):
            try:
                expected_hash = expected_batch_hashes[record_ids[row_position - 1]][condition]
                if before_forward is not None:
                    before_forward(row_position, condition_position, condition)
                batch = row["batches"][condition].to(device)
                if batch_sha256(batch) != expected_hash:
                    failure = "frozen_batch_hash_changed_after_device_transfer"
                    break
                attempted += 1
                call_kwargs = dict(batch)
                call_kwargs.update(use_cache=False, logits_to_keep=1)
                with torch_module.inference_mode():
                    if model_call is None:
                        output = model(**call_kwargs)
                    else:
                        output = model_call(
                            model=model, call_kwargs=call_kwargs,
                            expected_batch_keys=tuple(batch),
                            expected_batch_sha256=expected_hash,
                            row_position=row_position,
                            condition_position=condition_position,
                            condition=condition)
                logits = output.logits[0, -1].float()
                if not bool(torch_module.isfinite(logits).all()):
                    failure = "nonfinite_next_token_logits"
                    break
                log_probs = torch_module.log_softmax(logits, dim=-1)
                scores = log_probs[token_tensor].float().detach().cpu().tolist()
                if (len(scores) != LABEL_COUNT
                        or any(not math.isfinite(float(value)) for value in scores)):
                    failure = "answer_label_log_probability_invalid"
                    break
                row_scores[condition] = [float(value) for value in scores]
                completed += 1
            except Exception as exc:
                # Do not serialize exception text; it may contain paths or input data.
                failure = "forward_or_score_failed_" + type(exc).__name__
                break
        if row_scores:
            margins = {condition: _margin(values, row["gold_index"])
                       for condition, values in row_scores.items()}
            row_result = {
                "record_position": row_position,
                "choice_log_probabilities_nats": row_scores,
                "gold_vs_best_foil_margins_nats": margins,
                "best_foil_index_by_condition": {
                    condition: max((index for index in range(LABEL_COUNT)
                                   if index != row["gold_index"]),
                                  key=lambda index: values[index])
                    for condition, values in row_scores.items()
                },
                "d_control_minus_claim_mask_nats": None,
            }
            if set(margins) == set(CONDITIONS):
                row_result["d_control_minus_claim_mask_nats"] = (
                    margins["temporal_control_black"] - margins["claim_evidence_black"])
            records.append(row_result)
        if failure is not None:
            break

    complete = (failure is None and completed == REQUESTED_RECORDS * len(CONDITIONS)
                and len(records) == REQUESTED_RECORDS
                and all(row["d_control_minus_claim_mask_nats"] is not None
                        for row in records))
    complete_triplets = sum(row["d_control_minus_claim_mask_nats"] is not None
                            for row in records)
    return {
        "status": "completed" if complete else "failed",
        "scope": "finite_three_record_descriptive_only",
        "requested_records": REQUESTED_RECORDS,
        "cohort_denominator": REQUESTED_RECORDS,
        "completed_triplets": complete_triplets,
        "requested_forwards": REQUESTED_RECORDS * len(CONDITIONS),
        "attempted_forwards": attempted,
        "completed_forwards": completed,
        "failure_code": failure,
        "records": records,
        "D3_control_minus_claim_mask_nats": (
            sum(row["d_control_minus_claim_mask_nats"] for row in records) / REQUESTED_RECORDS
            if complete else None),
        "confidence_interval": None,
        "p_value": None,
        "execution_authorization": "not_granted",
    }
