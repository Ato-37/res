"""Evidence-restricted attention-knockout edges (WP1, experiments/2026-10-23_evidence_pathway_knockout_plan.md).

Builds knockout blocks (rows, cols) like qta_edge_knockout, but the blocked key columns are the video keys of selected temporal groups
(annotated evidence groups or an equal-count sham). Hook installation is reused from qta_edge_knockout.
"""
from __future__ import annotations

from typing import Sequence

from qta_edge_knockout import (block_cell_count, install_knockout_hooks,  # noqa: F401  (re-exported)
                               remove_hooks)

EVIDENCE_EDGES = ("video_to_text", "video_to_last")


def evidence_blocks(edge: str, *, video_groups: Sequence[Sequence[int]], seq_len: int,
                    key_groups: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    groups = tuple(tuple(g) for g in video_groups)
    sel = tuple(sorted(set(key_groups)))
    if (edge not in EVIDENCE_EDGES or not groups or not sel or type(seq_len) is not int
            or any(type(i) is not int or not 0 <= i < len(groups) for i in sel)
            or any(not g or tuple(sorted(set(g))) != g for g in groups)):
        raise ValueError("evidence_blocks_inputs_invalid")
    cols = tuple(sorted(p for i in sel for p in groups[i]))
    last_video = max(p for g in groups for p in g)
    if seq_len < last_video + 3:
        raise ValueError("sequence_too_short_for_text_rows")
    if edge == "video_to_text":
        return [(tuple(range(last_video + 1, seq_len - 1)), cols)]
    return [((seq_len - 1,), cols)]


def all_video_blocks(edge: str, *, video_groups: Sequence[Sequence[int]], seq_len: int):
    """Positive-control block: the same edge but over every video key."""
    return evidence_blocks(edge, video_groups=video_groups, seq_len=seq_len, key_groups=range(len(video_groups)))
