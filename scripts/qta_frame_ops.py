"""Input-level frame operations for WP5 (frame count, geometry and timestamps are unchanged; one temporal group = `patch` consecutive frames)."""
from __future__ import annotations

from typing import Sequence


def group_frames(groups: Sequence[int], patch: int, n_frames: int) -> set:
    out = set()
    for g in groups:
        for j in range(patch):
            i = patch * g + j
            if not 0 <= i < n_frames:
                raise ValueError("group_outside_video")
            out.add(i)
    return out


def replace_indices(frames: Sequence, indices: set) -> list:
    """Replace each frame in `indices` by a copy of the nearest frame outside `indices` (ties -> the earlier frame)."""
    n = len(frames)
    keep = [i for i in range(n) if i not in indices]
    if not keep:
        raise ValueError("nothing_left_to_copy_from")
    out = []
    for i in range(n):
        if i in indices:
            j = min(keep, key=lambda k: (abs(k - i), k))
            out.append(frames[j].copy())
        else:
            out.append(frames[i].copy())
    return out


def replace_groups(frames: Sequence, groups: Sequence[int], patch: int) -> list:
    return replace_indices(frames, group_frames(groups, patch, len(frames)))


def keep_only_groups(frames: Sequence, groups: Sequence[int], patch: int) -> list:
    n_groups = len(frames) // patch
    other = [g for g in range(n_groups) if g not in set(groups)]
    return replace_indices(frames, group_frames(other, patch, len(frames)))


def all_static(frames: Sequence) -> list:
    mid = len(frames) // 2
    return [frames[mid].copy() for _ in frames]
