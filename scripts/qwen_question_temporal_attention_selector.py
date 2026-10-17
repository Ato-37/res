"""Outcome-independent question-token to temporal-group attention selector.

This is a proposed selector component, not a frozen treatment or scorer. It
maps the exact natural-language question span in a rendered video-first prompt
through the Qwen processor's single video-pad expansion, then aggregates an
untreated baseline attention map over complete temporal token groups. It does
not inspect claims, reviewed intervals, gold answers, options, or logits.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any


def map_question_token_positions(*, tokenizer: Any, rendered_text: str,
                                 question_text: str, processed_input_ids: Any,
                                 video_token_id: int) -> list[int]:
    """Map question character offsets into processor IDs after video expansion.

    The rendered prompt must contain exactly one video placeholder and one
    exact question string after it. The complete processor ID vector must equal
    the rendered token IDs with that placeholder expanded to its repeated
    visual-token run. Any additional processor rewrite fails closed.
    """
    if (not isinstance(rendered_text, str) or not rendered_text
            or not isinstance(question_text, str) or not question_text.strip()
            or type(video_token_id) is not int or video_token_id < 0):
        raise ValueError("question_prompt_input_invalid")
    occurrences = 0
    cursor = 0
    while True:
        match = rendered_text.find(question_text, cursor)
        if match < 0:
            break
        occurrences += 1
        if occurrences > 1:
            break
        cursor = match + 1
    if occurrences != 1:
        raise ValueError("question_text_not_unique_in_prompt")
    encoded = tokenizer(rendered_text, add_special_tokens=False,
                        return_offsets_mapping=True)
    raw_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes))
            or not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes))
            or len(raw_ids) != len(offsets) or not raw_ids):
        raise ValueError("rendered_tokenization_invalid")
    raw_ids = list(raw_ids)
    raw_video = [i for i, token in enumerate(raw_ids)
                 if type(token) is int and token == video_token_id]
    if len(raw_video) != 1:
        raise ValueError("rendered_video_placeholder_count_invalid")
    placeholder = raw_video[0]
    if any(type(token) is not int or token < 0 for token in raw_ids):
        raise ValueError("rendered_token_ids_invalid")

    if hasattr(processed_input_ids, "detach"):
        processed_input_ids = processed_input_ids.detach().cpu().tolist()
    if (not isinstance(processed_input_ids, Sequence)
            or isinstance(processed_input_ids, (str, bytes))):
        raise ValueError("processed_input_ids_invalid")
    processed = list(processed_input_ids)
    if (processed and isinstance(processed[0], Sequence)
            and not isinstance(processed[0], (str, bytes))):
        if len(processed) != 1:
            raise ValueError("processed_input_batch_size_invalid")
        processed = list(processed[0])
    if any(type(token) is not int or token < 0 for token in processed):
        raise ValueError("processed_input_ids_invalid")
    visual = [i for i, token in enumerate(processed) if token == video_token_id]
    if (not visual or visual != list(range(visual[0], visual[-1] + 1))
            or len(processed) < len(raw_ids)):
        raise ValueError("processed_video_token_run_invalid")
    expected = (raw_ids[:placeholder] + [video_token_id] * len(visual)
                + raw_ids[placeholder + 1:])
    if processed != expected:
        raise ValueError("processor_ids_not_single_video_expansion")

    char_start = rendered_text.find(question_text)
    char_end = char_start + len(question_text)
    raw_question_positions = []
    for index, offset in enumerate(offsets):
        if (not isinstance(offset, Sequence) or len(offset) != 2
                or type(offset[0]) is not int or type(offset[1]) is not int
                or offset[0] < 0 or offset[1] < offset[0]):
            raise ValueError("token_offsets_invalid")
        if offset[1] > char_start and offset[0] < char_end:
            raw_question_positions.append(index)
    if (not raw_question_positions
            or raw_question_positions[0] <= placeholder
            or raw_question_positions != list(range(raw_question_positions[0],
                                                      raw_question_positions[-1] + 1))):
        raise ValueError("question_token_span_invalid")
    shift = len(visual) - 1
    mapped = [index if index < placeholder else index + shift
              for index in raw_question_positions]
    if mapped != list(range(mapped[0], mapped[-1] + 1)):
        raise ValueError("mapped_question_span_invalid")
    return mapped


def aggregate_temporal_group_attention(*, attention_by_layer: Mapping[int, Any],
                                       layer_indices: Sequence[int],
                                       question_positions: Sequence[int],
                                       group_positions: Sequence[Sequence[int]],
                                       torch_module: Any) -> list[float]:
    """Average baseline attention mass from question queries into each group."""
    if (torch_module is None or not isinstance(attention_by_layer, Mapping)
            or not isinstance(layer_indices, Sequence) or not layer_indices
            or any(type(layer) is not int or layer < 0 for layer in layer_indices)
            or tuple(sorted(set(layer_indices))) != tuple(layer_indices)
            or not isinstance(question_positions, Sequence) or not question_positions
            or any(type(p) is not int or p < 0 for p in question_positions)
            or tuple(question_positions) != tuple(range(question_positions[0],
                                                       question_positions[-1] + 1))
            or not isinstance(group_positions, Sequence) or not group_positions):
        raise ValueError("selector_inputs_invalid")
    if set(attention_by_layer) != set(layer_indices):
        raise ValueError("attention_layer_set_mismatch")
    groups = [tuple(group) for group in group_positions]
    if any(not group or any(type(p) is not int or p < 0 for p in group)
           or tuple(sorted(set(group))) != group
           or group != tuple(range(group[0], group[-1] + 1)) for group in groups):
        raise ValueError("temporal_group_positions_invalid")
    if len({len(group) for group in groups}) != 1:
        raise ValueError("temporal_group_size_mismatch")
    if groups != sorted(groups, key=lambda group: group[0]):
        raise ValueError("temporal_group_order_invalid")
    flattened = [p for group in groups for p in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("temporal_groups_overlap")

    layer_scores = []
    reference_shape = None
    for layer in layer_indices:
        attention = attention_by_layer[layer]
        if (not torch_module.is_tensor(attention) or attention.ndim != 4
                or int(attention.shape[0]) != 1 or int(attention.shape[1]) < 1):
            raise ValueError("attention_tensor_shape_invalid")
        if reference_shape is None:
            reference_shape = tuple(attention.shape)
        elif tuple(attention.shape) != reference_shape:
            raise ValueError("attention_layer_shape_mismatch")
        query_length, key_length = map(int, attention.shape[-2:])
        if (max(question_positions) >= query_length
                or max(flattened) >= key_length
                or max(flattened) >= min(question_positions)):
            raise ValueError("question_group_positions_not_causal")
        rows = attention[0, :, list(question_positions), :].float()
        if (not bool(torch_module.isfinite(rows).all())
                or bool((rows < 0).any())):
            raise ValueError("attention_weights_invalid")
        mean_question_attention = rows.mean(dim=(0, 1))
        layer_scores.append([
            float(mean_question_attention[list(group)].sum().item())
            for group in groups
        ])
    means = [sum(scores[index] for scores in layer_scores) / len(layer_scores)
             for index in range(len(groups))]
    if any(not math.isfinite(score) for score in means):
        raise ValueError("temporal_group_score_nonfinite")
    return means


def select_top_temporal_groups(scores: Sequence[float], *, top_k: int) -> list[int]:
    """Select highest-scoring temporal groups, resolving ties by earliest index."""
    if (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))
            or not scores or type(top_k) is not int or not 1 <= top_k <= len(scores)):
        raise ValueError("top_k_or_scores_invalid")
    values = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("temporal_group_score_invalid")
        value = float(score)
        if not math.isfinite(value) or value < 0:
            raise ValueError("temporal_group_score_invalid")
        values.append(value)
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:top_k]


def select_hash_sham_temporal_groups(*, temporal_group_count: int, top_k: int,
                                     protocol_seed: str,
                                     opaque_record_token: str) -> list[int]:
    """Select a stable, outcome-independent sham subset by SHA-256 ordering.

    Ranking byte strings are UTF-8 encodings of
    ``qta-sham-v1\\0{seed}\\0{opaque_token}\\0{group_index}``; ties use the
    ascending group index. This avoids dependence on a runtime PRNG version.
    The caller remains responsible for keeping the opaque token private.
    """
    if (type(temporal_group_count) is not int or temporal_group_count <= 0
            or type(top_k) is not int or not 1 <= top_k <= temporal_group_count
            or not isinstance(protocol_seed, str) or not protocol_seed
            or "\x00" in protocol_seed
            or not isinstance(opaque_record_token, str) or not opaque_record_token
            or "\x00" in opaque_record_token):
        raise ValueError("hash_sham_selector_inputs_invalid")
    prefix = f"qta-sham-v1\0{protocol_seed}\0{opaque_record_token}\0"
    ranked = sorted(
        range(temporal_group_count),
        key=lambda index: (hashlib.sha256(f"{prefix}{index}".encode("utf-8")).digest(),
                           index),
    )
    return sorted(ranked[:top_k])
