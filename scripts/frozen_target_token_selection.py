"""Freeze and score caller-selected positions in an already frozen target.

Selection indices are zero-based and relative to ``FrozenTarget.token_ids``.
This module only computes arithmetic over the corresponding per-token scores.
It cannot identify claim text, JSON wrappers, EOS meaning, or semantic claim
boundaries; the caller supplies those decisions and their provenance remains
unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from frozen_teacher_forced_score import FrozenTarget


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    return value


def _selection_digest(target_sha256: str, indices: tuple[int, ...]) -> str:
    payload = json.dumps(
        {"target_sha256": target_sha256, "indices": list(indices)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validated_target(target: FrozenTarget) -> tuple[str, int]:
    if not isinstance(target, FrozenTarget):
        raise TypeError("target must be a FrozenTarget")
    if type(target.token_ids) is not tuple or not target.token_ids:
        raise ValueError("target token IDs must be a nonempty immutable tuple")
    if any(type(token_id) is not int or token_id < 0 for token_id in target.token_ids):
        raise ValueError("target token IDs must be nonnegative integers")
    digest = hashlib.sha256(
        json.dumps(target.token_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if target.target_sha256 != digest:
        raise ValueError("frozen target digest does not match token IDs")
    return digest, len(target.token_ids)


@dataclass(frozen=True)
class FrozenTokenSelection:
    """Immutable selection bound to one target digest and exact target length."""

    target_sha256: str
    target_length: int
    indices: tuple[int, ...]
    selection_sha256: str


@dataclass(frozen=True)
class SelectedTokenScore:
    """Mechanical aggregate over selected target-token log probabilities."""

    target_sha256: str
    target_length: int
    selection_sha256: str
    indices: tuple[int, ...]
    count: int
    sum_nats: float
    mean_nats_per_token: float
    semantic_selection_verified: bool = False


def freeze_token_selection(
    target: FrozenTarget, indices: Sequence[int]
) -> FrozenTokenSelection:
    """Freeze sorted, unique, nonempty target-relative indices.

    This does not infer what a selected position represents. In particular, it
    does not include or exclude EOS, wrappers, or claim text automatically.
    """
    target_digest, target_length = _validated_target(target)
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise TypeError("indices must be a sequence of strict integers")
    frozen_indices = tuple(indices)
    if not frozen_indices:
        raise ValueError("indices must be nonempty")
    for index in frozen_indices:
        _strict_int("selection index", index)
    if any(index < 0 or index >= target_length for index in frozen_indices):
        raise ValueError("selection index is outside the frozen target")
    if tuple(sorted(set(frozen_indices))) != frozen_indices:
        raise ValueError("indices must be sorted and unique")
    return FrozenTokenSelection(
        target_sha256=target_digest,
        target_length=target_length,
        indices=frozen_indices,
        selection_sha256=_selection_digest(target_digest, frozen_indices),
    )


def score_frozen_selection(
    full_score: Mapping[str, object], selection: FrozenTokenSelection
) -> SelectedTokenScore:
    """Aggregate selected per-token scores from ``score_frozen_target`` output.

    The score must match the frozen target digest and exact length. Only
    mechanically selected positions are summed/averaged; semantic selection
    correctness and caller provenance are not verified here.
    """
    if not isinstance(full_score, Mapping):
        raise TypeError("full_score must be the mapping returned by score_frozen_target")
    if not isinstance(selection, FrozenTokenSelection):
        raise TypeError("selection must be a FrozenTokenSelection")
    if type(selection.target_length) is not int or selection.target_length <= 0:
        raise ValueError("selection target length is invalid")
    if type(selection.indices) is not tuple or not selection.indices:
        raise ValueError("selection indices must be a nonempty immutable tuple")
    for index in selection.indices:
        _strict_int("selection index", index)
    if any(index < 0 or index >= selection.target_length for index in selection.indices):
        raise ValueError("selection index is outside the frozen target")
    if tuple(sorted(set(selection.indices))) != selection.indices:
        raise ValueError("selection indices must be sorted and unique")
    if type(selection.target_sha256) is not str or len(selection.target_sha256) != 64:
        raise ValueError("selection target digest is invalid")
    expected_selection_digest = _selection_digest(
        selection.target_sha256, selection.indices
    )
    if selection.selection_sha256 != expected_selection_digest:
        raise ValueError("selection digest does not match target and indices")

    if full_score.get("target_sha256") != selection.target_sha256:
        raise ValueError("scored target digest does not match frozen selection")
    score_length = full_score.get("target_token_count")
    if type(score_length) is not int or score_length != selection.target_length:
        raise ValueError("scored target length does not match frozen selection")
    per_token = full_score.get("token_logprobs_nats")
    if isinstance(per_token, (str, bytes)) or not isinstance(per_token, Sequence):
        raise TypeError("full score must include a per-token score sequence")
    if len(per_token) != selection.target_length:
        raise ValueError("per-token score length does not match frozen target")

    selected_values: list[float] = []
    for index in selection.indices:
        value = per_token[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("selected per-token scores must be numeric and non-bool")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("selected per-token scores must be finite")
        selected_values.append(value)
    total = sum(selected_values)
    if not math.isfinite(total):
        raise ValueError("selected score sum must be finite")
    return SelectedTokenScore(
        target_sha256=selection.target_sha256,
        target_length=selection.target_length,
        selection_sha256=selection.selection_sha256,
        indices=selection.indices,
        count=len(selected_values),
        sum_nats=total,
        mean_nats_per_token=total / len(selected_values),
    )
