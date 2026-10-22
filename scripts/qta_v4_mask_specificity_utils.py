"""CPU-only pure utilities for the frozen v4 mask-specificity design.

No model, dataset, protected artifact, filesystem, or random state is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Sequence
from typing import Any

PROTOCOL_ID = "qta-v4-mask-specificity-v1"
FIXED_SEED = 20261018


class MaskSpecificityInputError(ValueError):
    """Malformed synthetic or preflight geometry input."""


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise MaskSpecificityInputError(label + "_must_be_exact_int")
    return value


def _mask_and_counts(mask_groups: Sequence[int], group_token_counts: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(mask_groups, (str, bytes)) or not isinstance(mask_groups, Sequence):
        raise MaskSpecificityInputError("mask_groups_invalid")
    if isinstance(group_token_counts, (str, bytes)) or not isinstance(group_token_counts, Sequence):
        raise MaskSpecificityInputError("group_token_counts_invalid")
    if not group_token_counts:
        raise MaskSpecificityInputError("group_token_counts_empty")
    counts = tuple(_exact_int(v, "group_token_count") for v in group_token_counts)
    if any(v <= 0 for v in counts):
        raise MaskSpecificityInputError("group_token_count_nonpositive")
    mask = tuple(_exact_int(v, "mask_group") for v in mask_groups)
    if not mask:
        raise MaskSpecificityInputError("mask_groups_empty")
    if len(set(mask)) != len(mask):
        raise MaskSpecificityInputError("mask_groups_duplicate")
    if any(v < 0 or v >= len(counts) for v in mask):
        raise MaskSpecificityInputError("mask_group_out_of_range")
    return tuple(sorted(mask)), counts


def eligible_translations(mask_groups: Sequence[int],
                          group_token_counts: Sequence[int]) -> tuple[int, ...]:
    """Return sorted eligible nonzero integer offsets, without wraparound.

    Translation preserves the mask's set cardinality and exact run shape by
    construction. A candidate is retained only if every shifted group stays
    inside `[0, T)` and the sum of per-group visual-token counts is unchanged.
    """
    mask, counts = _mask_and_counts(mask_groups, group_token_counts)
    total_groups = len(counts)
    lower = -mask[0]
    upper = total_groups - 1 - mask[-1]
    original_positions = sum(counts[group] for group in mask)
    result = []
    for offset in range(lower, upper + 1):
        if offset == 0:
            continue
        shifted = tuple(group + offset for group in mask)
        if (min(shifted) < 0 or max(shifted) >= total_groups
                or len(shifted) != len(mask)
                or sum(counts[group] for group in shifted) != original_positions):
            continue
        result.append(offset)
    return tuple(result)


def choose_translation(mask_groups: Sequence[int],
                       group_token_counts: Sequence[int], *,
                       video_id: str, qid: str,
                       protocol_id: str = PROTOCOL_ID,
                       fixed_seed: int = FIXED_SEED) -> int | None:
    """Select the hash-min eligible offset using the frozen protocol bytes.

    `None` means structurally unavailable. Hash ties are resolved by smaller
    integer offset, making behavior deterministic even under a digest collision.
    """
    if not isinstance(protocol_id, str) or not protocol_id or "\0" in protocol_id:
        raise MaskSpecificityInputError("protocol_id_invalid")
    if not isinstance(video_id, str) or not video_id or "\0" in video_id:
        raise MaskSpecificityInputError("video_id_invalid")
    if not isinstance(qid, str) or not qid or "\0" in qid:
        raise MaskSpecificityInputError("qid_invalid")
    seed = _exact_int(fixed_seed, "fixed_seed")
    offsets = eligible_translations(mask_groups, group_token_counts)
    if not offsets:
        return None
    def offset_digest(offset: int) -> bytes:
        payload = json.dumps([protocol_id, str(seed), video_id, qid, str(offset)],
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).digest()
    return min(offsets, key=lambda offset: (offset_digest(offset), offset))


def finite_cohort_summary(values: Iterable[float]) -> dict[str, int | float | None]:
    """Describe a complete finite cohort; sample SD is None for n < 2.

    Booleans, non-numeric values, NaN, and infinities are rejected instead of
    silently omitted. Mean and sample SD use population-free finite-cohort
    arithmetic only; this function computes no interval or inferential test.
    """
    try:
        materialized = list(values)
    except TypeError as exc:
        raise MaskSpecificityInputError("values_not_iterable") from exc
    if not materialized:
        raise MaskSpecificityInputError("finite_cohort_empty")
    numbers_list: list[float] = []
    for value in materialized:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise MaskSpecificityInputError("finite_cohort_value_not_real")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MaskSpecificityInputError("finite_cohort_value_not_finite")
        numbers_list.append(numeric)
    n = len(numbers_list)
    try:
        mean = math.fsum(numbers_list) / n
    except OverflowError as exc:
        raise MaskSpecificityInputError("finite_cohort_mean_overflow") from exc
    if not math.isfinite(mean):
        raise MaskSpecificityInputError("finite_cohort_mean_overflow")
    if n < 2:
        sample_sd = None
    else:
        try:
            squared = math.fsum((value - mean) ** 2 for value in numbers_list)
        except OverflowError as exc:
            raise MaskSpecificityInputError("finite_cohort_sd_overflow") from exc
        sample_sd = math.sqrt(squared / (n - 1))
        if not math.isfinite(sample_sd):
            raise MaskSpecificityInputError("finite_cohort_sd_overflow")
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "minimum": min(numbers_list),
        "maximum": max(numbers_list),
        "positive": sum(value > 0 for value in numbers_list),
        "negative": sum(value < 0 for value in numbers_list),
        "zero": sum(value == 0 for value in numbers_list),
    }


def specificity_descriptives(annotation_union_d: Sequence[float],
                              placebo_translation_d: Sequence[float]) -> dict[str, dict[str, int | float | None]]:
    """Summarize paired finite-cohort D values and their rowwise difference."""
    if (isinstance(annotation_union_d, (str, bytes))
            or isinstance(placebo_translation_d, (str, bytes))
            or not isinstance(annotation_union_d, Sequence)
            or not isinstance(placebo_translation_d, Sequence)):
        raise MaskSpecificityInputError("paired_cohort_sequences_invalid")
    if len(annotation_union_d) != len(placebo_translation_d):
        raise MaskSpecificityInputError("paired_cohort_length_mismatch")
    # Validate each input before subtraction, preserving fail-closed behavior.
    annotation_summary = finite_cohort_summary(annotation_union_d)
    placebo_summary = finite_cohort_summary(placebo_translation_d)
    differences = [float(a) - float(p) for a, p in zip(annotation_union_d, placebo_translation_d)]
    difference_summary = finite_cohort_summary(differences)
    return {
        "D_annotation_union": annotation_summary,
        "D_placebo_translation": placebo_summary,
        "specificity_difference": difference_summary,
    }
