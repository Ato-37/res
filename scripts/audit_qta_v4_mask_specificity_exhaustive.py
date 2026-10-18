"""Independent finite exhaustive audit of placebo translation selection.

This deliberately uses synthetic geometry only.  The reference candidate
enumerator and hash input construction below do not call the production helpers.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qta_v4_mask_specificity_utils import (  # noqa: E402
    FIXED_SEED, PROTOCOL_ID, choose_translation, eligible_translations,
)


def reference_offsets(mask: tuple[int, ...], counts: tuple[int, ...]) -> tuple[int, ...]:
    original_shape = tuple(b - a for a, b in zip(mask, mask[1:]))
    original_tokens = sum(counts[i] for i in mask)
    candidates: list[int] = []
    for offset in range(-(len(counts) - 1), len(counts)):
        if offset == 0:
            continue
        shifted = tuple(i + offset for i in mask)
        if any(i < 0 or i >= len(counts) for i in shifted):
            continue
        shape = tuple(b - a for a, b in zip(shifted, shifted[1:]))
        if shape != original_shape or sum(counts[i] for i in shifted) != original_tokens:
            continue
        candidates.append(offset)
    return tuple(candidates)


def reference_choice(offsets: tuple[int, ...], video_id: str, qid: str) -> int | None:
    if not offsets:
        return None
    ranked: list[tuple[bytes, int]] = []
    for offset in offsets:
        # Spell out the frozen JSON bytes independently from production code.
        fields = [PROTOCOL_ID, str(FIXED_SEED), video_id, qid, str(offset)]
        wire = ("[" + ",".join(json.dumps(x, ensure_ascii=False) for x in fields) + "]").encode("utf-8")
        ranked.append((hashlib.sha256(wire).digest(), offset))
    return min(ranked)[1]


def run_audit() -> dict[str, int]:
    geometries = masks_checked = candidates_checked = unavailable = 0
    for size in range(1, 7):
        for counts in itertools.product((1, 2), repeat=size):
            geometries += 1
            for mask_size in range(1, size + 1):
                for mask in itertools.combinations(range(size), mask_size):
                    masks_checked += 1
                    expected_offsets = reference_offsets(mask, counts)
                    actual_offsets = eligible_translations(mask, counts)
                    if actual_offsets != expected_offsets:
                        raise AssertionError(("offset_mismatch", counts, mask, expected_offsets, actual_offsets))
                    for offset in actual_offsets:
                        shifted = tuple(i + offset for i in mask)
                        if (min(shifted) < 0 or max(shifted) >= size
                                or tuple(b-a for a, b in zip(shifted, shifted[1:]))
                                != tuple(b-a for a, b in zip(mask, mask[1:]))
                                or sum(counts[i] for i in shifted) != sum(counts[i] for i in mask)):
                            raise AssertionError(("invariant_failure", counts, mask, offset))
                    candidates_checked += len(actual_offsets)
                    video_id, qid = f"synthetic-video-{size}-{counts}", f"synthetic-q-{mask}"
                    expected_choice = reference_choice(expected_offsets, video_id, qid)
                    actual_choice = choose_translation(mask, counts, video_id=video_id, qid=qid)
                    if actual_choice != expected_choice:
                        raise AssertionError(("choice_mismatch", counts, mask, expected_choice, actual_choice))
                    if not expected_offsets:
                        unavailable += 1
                    reversed_input_choice = choose_translation(tuple(reversed(mask)), counts,
                                                               video_id=video_id, qid=qid)
                    if reversed_input_choice != expected_choice:
                        raise AssertionError(("order_dependence", counts, mask))
    return {"count_vectors": geometries, "masks": masks_checked,
            "eligible_offsets": candidates_checked, "no_candidate_masks": unavailable}


if __name__ == "__main__":
    print(json.dumps(run_audit(), sort_keys=True))
