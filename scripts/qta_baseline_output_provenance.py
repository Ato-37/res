"""Pure synthetic fixture calculation; this module performs no model forwards.

Explicitly tagged fixture logits are converted to A--E probabilities and bound
to structural row/input metadata for regression tests. This is not model-output
provenance, evidence of model execution, a production receipt, or authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256


RUN_DIGEST_FIELDS = (
    "protocol_sha256", "model_sha256", "tokenizer_sha256",
    "scorer_sha256", "processor_sha256",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ROW_COUNT = 3


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{field}_invalid")
    return value


def _log_probs(logits: Any, token_ids: list[int]) -> list[float]:
    # Fixture schema is exactly [batch=1, sequence=1, vocabulary].
    if (not isinstance(logits, list) or len(logits) != 1
            or not isinstance(logits[0], list) or len(logits[0]) != 1
            or not isinstance(logits[0][0], list) or not logits[0][0]):
        raise ValueError("synthetic_fixture_logits_shape_invalid")
    vocab_logits = logits[0][0]
    if any(isinstance(x, (list, tuple, dict)) for x in vocab_logits):
        raise ValueError("synthetic_fixture_logits_shape_invalid")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in vocab_logits):
        raise ValueError("synthetic_fixture_logits_invalid")
    if any(token >= len(vocab_logits) for token in token_ids):
        raise ValueError("answer_token_out_of_vocabulary")
    maximum = max(float(x) for x in vocab_logits)
    log_normalizer = maximum + math.log(math.fsum(
        math.exp(float(x) - maximum) for x in vocab_logits))
    return [float(vocab_logits[token]) - log_normalizer for token in token_ids]


def _margin(values: list[float], gold_index: int) -> tuple[float, int, int]:
    p = min(range(5), key=lambda i: (-values[i], i))
    q = min((i for i in range(5) if i != p), key=lambda i: (-values[i], i))
    margin = values[gold_index] - max(values[i] for i in range(5) if i != gold_index)
    return margin, p, q


def _batch_input_ids(batch: FrozenBatch, expected: list[int], position: int) -> None:
    if "input_ids" not in batch:
        raise ValueError(f"row_{position}_batch_input_ids_missing")
    value = batch["input_ids"]
    if callable(getattr(value, "detach", None)):
        try:
            value = value.detach().cpu()
            if str(getattr(value, "device", "cpu")) != "cpu":
                raise ValueError
            value = value.tolist()
        except Exception:
            raise ValueError(f"row_{position}_batch_input_ids_unreadable") from None
    if isinstance(value, tuple):
        value = list(value)
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], (list, tuple))):
        value = list(value[0])
    if (not isinstance(value, list)
            or any(type(token) is not int for token in value)
            or value != expected):
        raise ValueError(f"row_{position}_batch_input_ids_mismatch")


def derive_synthetic_baseline_fixture(*, rows: Any,
                                      run_digests: dict[str, str]) -> dict:
    """Calculate fixture A--E values without a model or callable.

    Each row contains ``fixture_schema='synthetic_logits_1x1_vocab_v1'`` and
    ``fixture_logits`` shaped exactly ``[1, 1, vocabulary]``. The explicit
    values are synthetic test fixtures, never model outputs.
    """
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or len(rows) != _ROW_COUNT):
        raise ValueError("requested_three_rows_required")
    if (not isinstance(run_digests, dict)
            or set(run_digests) != set(RUN_DIGEST_FIELDS)):
        raise ValueError("run_digest_inventory_invalid")
    run_digests = {key: _digest(run_digests[key], key) for key in RUN_DIGEST_FIELDS}
    prepared = []
    seen = set()
    for position, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or set(row) != {
                "record_id", "prompt", "input_ids", "score_position",
                "answer_token_ids", "gold_index", "batch", "fixture_schema",
                "fixture_logits"}:
            raise ValueError(f"row_{position}_schema_invalid")
        record_id, prompt, ids = row["record_id"], row["prompt"], row["input_ids"]
        token_ids, gold, batch = (row["answer_token_ids"], row["gold_index"],
                                  row["batch"])
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"row_{position}_identity_invalid")
        seen.add(record_id)
        if (not isinstance(prompt, str) or not prompt.strip()
                or not isinstance(ids, list) or not ids
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError(f"row_{position}_prompt_or_ids_invalid")
        if (type(row["score_position"]) is not int
                or row["score_position"] != len(ids) - 1):
            raise ValueError(f"row_{position}_score_position_invalid")
        if (not isinstance(token_ids, list) or len(token_ids) != 5
                or any(type(token) is not int or token < 0 for token in token_ids)
                or len(set(token_ids)) != 5 or type(gold) is not int
                or not 0 <= gold < 5):
            raise ValueError(f"row_{position}_choice_contract_invalid")
        if type(batch) is not FrozenBatch or not batch:
            raise ValueError(f"row_{position}_frozen_batch_required")
        if row["fixture_schema"] != "synthetic_logits_1x1_vocab_v1":
            raise ValueError(f"row_{position}_synthetic_fixture_schema_invalid")
        values = _log_probs(row["fixture_logits"], token_ids)
        # Atomic structural preflight: all rows and ID bindings pass before any
        # fixture calculations are serialized.
        _batch_input_ids(batch, ids, position)
        prepared.append((position, row, batch_sha256(batch), values))

    records = []
    for position, row, batch_hash, values in prepared:
        record_id, prompt, ids = row["record_id"], row["prompt"], row["input_ids"]
        token_ids, gold = row["answer_token_ids"], row["gold_index"]
        margin, p, q = _margin(values, gold)
        records.append({
            "record_position": position,
            "record_id": record_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_ids_sha256": hashlib.sha256(_canonical(ids)).hexdigest(),
            "score_position": row["score_position"],
            "score_input_token_id": ids[-1],
            "answer_token_ids": list(token_ids),
            "gold_index": gold,
            "processor_batch_sha256": batch_hash,
            "option_log_probabilities": values,
            "p_index": p,
            "q_index": q,
            "gold_vs_best_foil_margin_nats": margin,
        })
    payload = {
        "schema_version": 1,
        "purpose": "synthetic_qta_baseline_fixture_calculation",
        "run_digests": run_digests,
        "records": records,
        "execution_authorization": "not_granted",
        "fixture_math_only": True,
        "fixture_logits_are_synthetic": True,
    }
    raw = _canonical(payload)
    return {
        "status": "synthetic_fixture_calculation_complete",
        "payload": payload,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_row_count": _ROW_COUNT,
        "fixture_logits_are_synthetic": True,
        "fixture_inputs_unverified": True,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
        "cryptographic_attestation": False,
    }


def verify_synthetic_baseline_binding(result: Any, *, expected_record_ids: Sequence[str],
                                      expected_run_digests: dict[str, str]) -> bool:
    """Check the fixture envelope/order; this does not prove model execution."""
    if (not isinstance(result, dict)
            or result.get("status") != "synthetic_fixture_calculation_complete"):
        raise ValueError("binding_result_invalid")
    payload = result.get("payload")
    if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "purpose", "run_digests", "records",
            "execution_authorization", "fixture_math_only",
            "fixture_logits_are_synthetic"}:
        raise ValueError("binding_payload_invalid")
    if (payload["schema_version"] != 1
            or payload["purpose"] != "synthetic_qta_baseline_fixture_calculation"
            or payload["fixture_math_only"] is not True
            or payload["fixture_logits_are_synthetic"] is not True
            or result.get("fixture_logits_are_synthetic") is not True
            or result.get("fixture_inputs_unverified") is not True
            or payload["execution_authorization"] != "not_granted"
            or result.get("execution_authorization") != "not_granted"
            or result.get("fixture_row_count") != _ROW_COUNT
            or result.get("scoring_authorized") is not False
            or result.get("cryptographic_attestation") is not False):
        raise ValueError("binding_scope_invalid")
    if (not isinstance(expected_run_digests, dict)
            or set(expected_run_digests) != set(RUN_DIGEST_FIELDS)):
        raise ValueError("binding_expected_run_digest_inventory_invalid")
    expected = {key: _digest(expected_run_digests[key], key)
                for key in RUN_DIGEST_FIELDS}
    if payload["run_digests"] != expected:
        raise ValueError("binding_run_digests_mismatch")
    records = payload["records"]
    if (not isinstance(expected_record_ids, Sequence)
            or len(expected_record_ids) != _ROW_COUNT
            or not isinstance(records, list) or len(records) != _ROW_COUNT
            or [record.get("record_id") for record in records] != list(expected_record_ids)
            or [record.get("record_position") for record in records] != [1, 2, 3]):
        raise ValueError("binding_row_order_mismatch")
    if hashlib.sha256(_canonical(payload)).hexdigest() != result.get("payload_sha256"):
        raise ValueError("binding_payload_digest_mismatch")
    return True
