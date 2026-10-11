"""Pure scoring for a frozen raw token tail under full teacher forcing.

The caller supplies one generated sequence as raw token IDs. This module
freezes its tail after the prompt boundary and never decodes/re-tokenizes text
or selects replacement tokens from later logits. It supports only one
unpadded sequence with full-sequence logits; it does not establish whether the
target corresponds to an annotated field or explanation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any


SOURCE_POLICY = "generation_sequence_tail"


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int; bool is not accepted")
    return value


def _torch_if_installed():
    try:
        import torch
    except ImportError:
        return None
    return torch


@dataclass(frozen=True)
class FrozenTarget:
    """Immutable snapshot of a raw generation tail and its provenance limits."""

    token_ids: tuple[int, ...]
    prompt_length: int
    target_sha256: str
    eos_token_id: int | None
    terminal_seen: bool
    source_policy: str = SOURCE_POLICY
    caller_provenance_unverified: bool = True

    @property
    def termination_status(self) -> str:
        return "terminal_seen" if self.terminal_seen else "termination_unverified"


def freeze_generation_tail(
    generated_sequence_ids: Sequence[int],
    *,
    prompt_length: int,
    eos_token_id: int | None = None,
) -> FrozenTarget:
    """Freeze the exact raw generation suffix after the supplied prompt length.

    ``generated_sequence_ids`` must be a single sequence, usually the raw
    ``generate().sequences[0]`` IDs. The helper cannot verify that provenance;
    it records ``caller_provenance_unverified=True``. Only one EOS ID is
    supported; callers with multi-EOS configurations must handle that
    explicitly before calling.
    """
    prompt_length = _strict_int("prompt_length", prompt_length)
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if eos_token_id is not None:
        eos_token_id = _strict_int("eos_token_id", eos_token_id)
        if eos_token_id < 0:
            raise ValueError("eos_token_id must be nonnegative")
    torch = _torch_if_installed()
    if torch is not None and isinstance(generated_sequence_ids, torch.Tensor):
        if generated_sequence_ids.ndim != 1:
            raise ValueError("generated_sequence_ids tensor must be one-dimensional")
        if generated_sequence_ids.dtype == torch.bool or generated_sequence_ids.dtype.is_floating_point or generated_sequence_ids.dtype.is_complex:
            raise TypeError("generated_sequence_ids tensor must have an integer dtype")
        sequence_values = generated_sequence_ids.detach().to(device="cpu").tolist()
    elif isinstance(generated_sequence_ids, (str, bytes)) or not isinstance(
        generated_sequence_ids, Sequence
    ):
        raise TypeError("generated_sequence_ids must be a one-dimensional sequence of ints")
    else:
        sequence_values = generated_sequence_ids
    sequence_ids = tuple(
        _strict_int(f"generated_sequence_ids[{index}]", token_id)
        for index, token_id in enumerate(sequence_values)
    )
    if any(token_id < 0 for token_id in sequence_ids):
        raise ValueError("generated token IDs must be nonnegative")
    if len(sequence_ids) <= prompt_length:
        raise ValueError("generation tail must contain at least one token")

    target_ids = sequence_ids[prompt_length:]
    target_digest = hashlib.sha256(
        json.dumps(target_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    terminal_seen = eos_token_id is not None and target_ids[-1] == eos_token_id
    return FrozenTarget(
        token_ids=target_ids,
        prompt_length=prompt_length,
        target_sha256=target_digest,
        eos_token_id=eos_token_id,
        terminal_seen=terminal_seen,
    )


def _selected_logits(logits: Any, expected_sequence_length: int,
                     row_indices: tuple[int, ...]) -> list[list[float]]:
    """Validate exact full shape and return only scored rows as CPU floats."""
    torch = _torch_if_installed()
    if torch is not None and isinstance(logits, torch.Tensor):
        if logits.ndim != 3:
            raise ValueError("tensor logits must have shape [1, sequence, vocab]")
        if logits.shape[0] != 1:
            raise ValueError("only batch size 1 is supported")
        if logits.shape[1] != expected_sequence_length:
            raise ValueError("logit sequence length must equal prompt_length + target length")
        if logits.shape[2] <= 0 or not logits.dtype.is_floating_point:
            raise TypeError("tensor logits must have a positive vocabulary axis and floating dtype")
        selected = logits[0, list(row_indices), :].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("selected logits must be finite")
        return selected.tolist()

    if isinstance(logits, (str, bytes)) or not isinstance(logits, Sequence):
        raise TypeError("logits must be a nested sequence with shape [1, sequence, vocab]")
    if len(logits) != 1:
        raise ValueError("only batch size 1 is supported")
    sequence_logits = logits[0]
    if isinstance(sequence_logits, (str, bytes)) or not isinstance(sequence_logits, Sequence):
        raise TypeError("logits must have shape [1, sequence, vocab]")
    if len(sequence_logits) != expected_sequence_length:
        raise ValueError("logit sequence length must equal prompt_length + target length")

    validated = []
    vocab_size = None
    for row_index, row in enumerate(sequence_logits):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"logit row {row_index} must be a sequence")
        if vocab_size is None:
            vocab_size = len(row)
            if vocab_size <= 0:
                raise ValueError("vocabulary width must be positive")
        elif len(row) != vocab_size:
            raise ValueError("all logit rows must have the same vocabulary width")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("logit values must be numeric and non-bool")
            numeric = float(value)
            if row_index in row_indices and not math.isfinite(numeric):
                raise ValueError("logits must be finite")
            values.append(numeric)
        validated.append(values)
    return [validated[row_index] for row_index in row_indices]


def _logprob(row: list[float], token_id: int) -> float:
    peak = max(row)
    shifted_log_partition = math.log(sum(math.exp(value - peak) for value in row))
    return (row[token_id] - peak) - shifted_log_partition


def score_frozen_target(logits: Any, target: FrozenTarget) -> dict:
    """Score frozen target IDs at their teacher-forced next-token rows.

    For prompt length ``P`` and target length ``L``, target token ``i`` is
    scored from logit row ``P - 1 + i``. Requires exact full-sequence length
    ``P + L``; padding, cache-sliced logits, multi-sequence batches, and beams
    are unsupported. Tensor logits are accepted directly; only the selected
    ``L`` rows are detached and copied to CPU for scalar aggregation. Python
    nested sequences are also accepted for small synthetic checks.
    """
    if not isinstance(target, FrozenTarget):
        raise TypeError("target must be a FrozenTarget created by freeze_generation_tail")
    if type(target.token_ids) is not tuple:
        raise TypeError("frozen target token IDs must be an immutable tuple")
    _strict_int("target.prompt_length", target.prompt_length)
    if not target.token_ids:
        raise ValueError("target must contain at least one token")
    if target.prompt_length <= 0:
        raise ValueError("frozen target prompt_length must be positive")
    if any(type(token_id) is not int or token_id < 0 for token_id in target.token_ids):
        raise ValueError("frozen target IDs must be nonnegative ints")
    if target.source_policy != SOURCE_POLICY or target.caller_provenance_unverified is not True:
        raise ValueError("frozen target provenance fields are invalid")
    if target.eos_token_id is not None:
        _strict_int("target.eos_token_id", target.eos_token_id)
        if target.eos_token_id < 0:
            raise ValueError("frozen EOS ID must be nonnegative")
    expected_digest = hashlib.sha256(
        json.dumps(target.token_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if target.target_sha256 != expected_digest:
        raise ValueError("frozen target digest does not match token IDs")
    expected_terminal_seen = (
        target.eos_token_id is not None and target.token_ids[-1] == target.eos_token_id
    )
    if target.terminal_seen is not expected_terminal_seen:
        raise ValueError("frozen terminal provenance does not match target tail")

    row_indices = tuple(
        target.prompt_length - 1 + index for index in range(len(target.token_ids))
    )
    selected_logits = _selected_logits(
        logits,
        expected_sequence_length=target.prompt_length + len(target.token_ids),
        row_indices=row_indices,
    )
    if max(target.token_ids) >= len(selected_logits[0]):
        raise ValueError("target token ID is outside the logits vocabulary")
    token_logprobs = tuple(
        _logprob(row, token_id)
        for row, token_id in zip(selected_logits, target.token_ids)
    )
    if not all(math.isfinite(value) for value in token_logprobs):
        raise ValueError("selected token scores exceed finite scalar precision")
    full_sum = sum(token_logprobs)
    if not math.isfinite(full_sum):
        raise ValueError("summed target score exceeds finite scalar precision")
    full_count = len(token_logprobs)

    content_count = full_count - int(target.terminal_seen)
    content_values = token_logprobs[:content_count]
    content_sum = sum(content_values)
    content_mean = content_sum / content_count if content_count else None
    return {
        "source_policy": target.source_policy,
        "caller_provenance_unverified": target.caller_provenance_unverified,
        "target_sha256": target.target_sha256,
        "prompt_length": target.prompt_length,
        "target_token_count": full_count,
        "eos_token_id": target.eos_token_id,
        "terminal_seen": target.terminal_seen,
        "termination_status": target.termination_status,
        "all_generated_score_includes_terminal": target.terminal_seen,
        "content_score_excludes_terminal": target.terminal_seen,
        "score_unit": "nats",
        "prediction_logit_rows": row_indices,
        "token_logprobs_nats": token_logprobs,
        "all_generated_sum_nats": full_sum,
        "all_generated_mean_nats_per_token": full_sum / full_count,
        "content_token_count": content_count,
        "content_sum_nats": content_sum,
        "content_mean_nats_per_token": content_mean,
    }
