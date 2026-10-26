"""Apply a frozen QTA key bias and/or visual-key ablation on target-token rows.

This helper only edits Qwen eager additive attention masks. It does not select
groups, authenticate their source, score faithfulness, or authorize execution.
The caller must supply fixed causal video-key positions and query rows.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Any


def _positions(name: str, value: Sequence[int], *, key_length: int,
               query_length: int) -> tuple[int, ...]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or not value or any(type(x) is not int for x in value)
            or tuple(sorted(set(value))) != tuple(value)
            or any(x < 0 or x >= key_length for x in value)):
        raise ValueError(f"{name}_positions_invalid")
    return tuple(value)


def apply_multiquery_video_key_intervention(*, attention_mask: Any,
        biased_positions: Sequence[int], ablated_positions: Sequence[int],
        query_positions: Sequence[int], logit_bias: float,
        torch_module: Any) -> Any:
    """Return a copy with fixed key operations on a fixed set of query rows.

    Ablation takes precedence where biased and ablated keys overlap. Each
    selected key must be visible and exactly zero in every targeted row before
    editing. Non-target query rows and all other keys are preserved.
    """
    if (torch_module is None or not callable(getattr(torch_module, "is_tensor", None))
            or not torch_module.is_tensor(attention_mask)):
        raise ValueError("attention_mask_tensor_required")
    if (attention_mask.ndim != 4 or tuple(attention_mask.shape[:2]) != (1, 1)
            or not attention_mask.dtype.is_floating_point):
        raise ValueError("additive_attention_mask_shape_or_dtype_invalid")
    if bool(torch_module.isnan(attention_mask).any()) or bool(torch_module.isposinf(attention_mask).any()):
        raise ValueError("attention_mask_contains_invalid_nonfinite")
    qlen, klen = map(int, attention_mask.shape[-2:])
    if qlen != klen:
        raise ValueError("uncached_square_attention_mask_required")
    queries = _positions("query", query_positions, key_length=qlen, query_length=qlen)
    if any(b <= a for a, b in zip(queries, queries[1:])):
        raise ValueError("query_positions_must_be_unique_sorted")
    biased = tuple(biased_positions) if biased_positions else ()
    ablated = tuple(ablated_positions) if ablated_positions else ()
    if not biased and not ablated:
        raise ValueError("at_least_one_key_operation_required")
    if biased:
        biased = _positions("biased", biased, key_length=klen, query_length=qlen)
    if ablated:
        ablated = _positions("ablated", ablated, key_length=klen, query_length=qlen)
    if biased and (isinstance(logit_bias, bool) or not isinstance(logit_bias, (int, float))
                   or not math.isfinite(float(logit_bias)) or float(logit_bias) == 0):
        raise ValueError("finite_nonzero_logit_bias_required")
    keys = tuple(sorted(set(biased) | set(ablated)))
    if any(key >= query for query in queries for key in keys):
        raise ValueError("selected_video_keys_must_precede_every_query")
    rows = torch_module.tensor(queries, device=attention_mask.device, dtype=torch_module.long)
    cols = torch_module.tensor(keys, device=attention_mask.device, dtype=torch_module.long)
    current = attention_mask[0, 0][rows[:, None], cols[None, :]]
    if not bool(torch_module.isfinite(current).all()) or not bool((current == 0).all()):
        raise ValueError("selected_video_key_not_visible_zero_in_every_query")

    result = attention_mask.clone()
    if biased:
        bcols = torch_module.tensor(biased, device=result.device, dtype=torch_module.long)
        result[0, 0][rows[:, None], bcols[None, :]] += float(logit_bias)
    if ablated:
        acols = torch_module.tensor(ablated, device=result.device, dtype=torch_module.long)
        result[0, 0][rows[:, None], acols[None, :]] = float("-inf")
    if bool(torch_module.isnan(result).any()) or bool(torch_module.isposinf(result).any()):
        raise ValueError("intervened_attention_mask_invalid")
    return result


def score_target_logits_window(*, logits: Any, target_token_ids: Sequence[int],
                               prompt_length: int, first_query_position: int,
                               torch_module: Any) -> dict:
    """Score frozen target IDs from a compact tail-logits window.

    For prompt length P and target y[0:L], logits must contain rows for
    absolute positions P-1 through P+L-1 (L+1 rows). Only the first L rows
    predict y[0:L]; the final row is required to prove the offset/window shape
    but is not included in the target score.
    """
    if (torch_module is None or not torch_module.is_tensor(logits)
            or logits.ndim != 3 or int(logits.shape[0]) != 1
            or not logits.dtype.is_floating_point):
        raise ValueError("target_logits_shape_or_dtype_invalid")
    if (type(prompt_length) is not int or prompt_length < 1
            or type(first_query_position) is not int
            or first_query_position != prompt_length - 1):
        raise ValueError("target_logits_offset_invalid")
    if (not isinstance(target_token_ids, Sequence)
            or isinstance(target_token_ids, (str, bytes)) or not target_token_ids
            or any(type(token_id) is not int or token_id < 0
                   for token_id in target_token_ids)):
        raise ValueError("target_token_ids_invalid")
    target = tuple(target_token_ids)
    length = len(target)
    if int(logits.shape[1]) != length + 1 or int(logits.shape[2]) <= max(target):
        raise ValueError("target_logits_window_length_or_vocab_invalid")
    selected = logits[0, :length, :].to(dtype=torch_module.float32)
    if not bool(torch_module.isfinite(selected).all()):
        raise ValueError("target_logits_nonfinite")
    ids = torch_module.tensor(target, device=selected.device, dtype=torch_module.long)
    token_logprobs = torch_module.log_softmax(selected, dim=-1).gather(
        1, ids[:, None]).squeeze(1)
    if not bool(torch_module.isfinite(token_logprobs).all()):
        raise ValueError("target_logprob_nonfinite")
    values = [float(x) for x in token_logprobs.detach().cpu().tolist()]
    return {"target_token_count": length, "first_query_position": first_query_position,
            "last_target_query_position": first_query_position + length - 1,
            "target_token_logprobs": values,
            "target_logprob_sum": sum(values),
            "target_logprob_mean": sum(values) / length,
            "score_semantics": "mean conditional logprob of frozen target token IDs"}


def freeze_generated_continuation(*, prompt_token_ids: Sequence[int],
                                  generated_sequence_ids: Sequence[int],
                                  eos_token_id: int | None) -> dict:
    """Freeze the raw `generate().sequences[0]` tail after exact prefix checking."""
    def ids(name: str, value: Sequence[int]) -> tuple[int, ...]:
        if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
                or not value or any(type(x) is not int or x < 0 for x in value)):
            raise ValueError(f"{name}_invalid")
        return tuple(value)
    prompt = ids("prompt_token_ids", prompt_token_ids)
    generated = ids("generated_sequence_ids", generated_sequence_ids)
    if len(generated) <= len(prompt) or generated[:len(prompt)] != prompt:
        raise ValueError("generated_prompt_prefix_mismatch")
    if eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0):
        raise ValueError("eos_token_id_invalid")
    target = generated[len(prompt):]
    digest = hashlib.sha256(json.dumps(target, separators=(",", ":")).encode("ascii")).hexdigest()
    return {"prompt_length": len(prompt), "target_token_ids": target,
            "target_token_count": len(target), "target_token_ids_sha256": digest,
            "eos_token_id": eos_token_id,
            "terminal_seen": eos_token_id is not None and target[-1] == eos_token_id,
            "source_policy": "raw_generate_sequences_suffix_after_exact_prompt_prefix"}


def append_target_to_unpadded_batch(*, batch: dict, target_token_ids: Sequence[int],
                                   torch_module: Any) -> tuple[dict, tuple[int, ...]]:
    """Append raw target IDs for uncached teacher forcing and return query rows."""
    if not isinstance(batch, dict) or torch_module is None:
        raise ValueError("teacher_forcing_batch_invalid")
    ids = batch.get("input_ids")
    mask = batch.get("attention_mask")
    if (not torch_module.is_tensor(ids) or ids.ndim != 2 or tuple(ids.shape[:1]) != (1,)
            or not torch_module.is_tensor(mask) or tuple(mask.shape) != tuple(ids.shape)
            or not bool((mask == 1).all())):
        raise ValueError("teacher_forcing_requires_single_unpadded_batch")
    if (not isinstance(target_token_ids, Sequence) or isinstance(target_token_ids, (str, bytes))
            or not target_token_ids or any(type(x) is not int or x < 0 for x in target_token_ids)):
        raise ValueError("teacher_forcing_target_ids_invalid")
    prompt_length = int(ids.shape[1])
    target = torch_module.tensor([list(target_token_ids)], dtype=ids.dtype, device=ids.device)
    extended = dict(batch)
    extended["input_ids"] = torch_module.cat((ids, target), dim=1)
    extended["attention_mask"] = torch_module.cat(
        (mask, torch_module.ones((1, len(target_token_ids)), dtype=mask.dtype, device=mask.device)), dim=1)
    for name in ("mm_token_type_ids", "token_type_ids"):
        value = extended.get(name)
        if torch_module.is_tensor(value) and value.ndim >= 2 and int(value.shape[-1]) == prompt_length:
            zeros = torch_module.zeros((*value.shape[:-1], len(target_token_ids)),
                                       dtype=value.dtype, device=value.device)
            extended[name] = torch_module.cat((value, zeros), dim=-1)
    # Qwen recomputes multimodal RoPE positions for the extended sequence.
    extended.pop("position_ids", None)
    extended.pop("cache_position", None)
    query_positions = tuple(range(prompt_length - 1, prompt_length + len(target_token_ids) - 1))
    return extended, query_positions


def validate_evidence_control_key_sets(*, evidence_positions: Sequence[int],
                                       control_positions: Sequence[int]) -> dict:
    """Require disjoint, exact equal-size flattened visual-key sets."""
    def keys(name: str, value: Sequence[int]) -> tuple[int, ...]:
        if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
                or not value or any(type(x) is not int or x < 0 for x in value)
                or tuple(sorted(set(value))) != tuple(value)):
            raise ValueError(f"{name}_visual_key_positions_invalid")
        return tuple(value)
    evidence = keys("evidence", evidence_positions)
    control = keys("control", control_positions)
    if len(evidence) != len(control):
        raise ValueError("evidence_control_visual_key_budget_mismatch")
    if set(evidence).intersection(control):
        raise ValueError("evidence_control_visual_key_overlap")
    return {"evidence_key_count": len(evidence), "control_key_count": len(control),
            "disjoint": True, "equal_flattened_key_budget": True}


def aggregate_captured_question_group_attention(*, attention_by_layer: dict,
        layer_indices: Sequence[int], group_positions: Sequence[Sequence[int]],
        question_token_count: int, torch_module: Any) -> list[float]:
    """Aggregate compact `[heads, question_tokens, sequence]` capture rows."""
    if (not isinstance(attention_by_layer, dict) or not layer_indices
            or tuple(sorted(attention_by_layer)) != tuple(layer_indices)
            or type(question_token_count) is not int or question_token_count <= 0
            or not group_positions):
        raise ValueError("captured_question_attention_inputs_invalid")
    groups = tuple(tuple(group) for group in group_positions)
    if any(not group or tuple(sorted(set(group))) != group for group in groups):
        raise ValueError("captured_temporal_group_positions_invalid")
    flattened = tuple(position for group in groups for position in group)
    if len(set(flattened)) != len(flattened):
        raise ValueError("captured_temporal_groups_overlap")
    reference = None
    scores_by_layer = []
    for layer in layer_indices:
        rows = attention_by_layer[layer]
        if (not torch_module.is_tensor(rows) or rows.ndim != 3
                or int(rows.shape[0]) < 1 or int(rows.shape[1]) != question_token_count):
            raise ValueError("captured_attention_shape_invalid")
        shape = tuple(rows.shape)
        if reference is None:
            reference = shape
        elif shape != reference:
            raise ValueError("captured_attention_layer_shape_mismatch")
        if max(flattened) >= int(rows.shape[-1]):
            raise ValueError("captured_group_key_position_out_of_range")
        cpu = rows.to(device="cpu", dtype=torch_module.float32)
        if not bool(torch_module.isfinite(cpu).all()) or bool((cpu < 0).any()):
            raise ValueError("captured_attention_values_invalid")
        mean = cpu.mean(dim=(0, 1))
        scores_by_layer.append([float(mean[list(group)].sum().item()) for group in groups])
    result = [sum(layer[index] for layer in scores_by_layer) / len(scores_by_layer)
              for index in range(len(groups))]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("captured_group_score_invalid")
    return result


def report_qta_key_overlap(*, qta_positions: Sequence[int],
                           evidence_positions: Sequence[int],
                           control_positions: Sequence[int]) -> dict:
    """Report overlap without using it to alter cohort membership or selection."""
    selected = set(_positions("qta", qta_positions, key_length=2**63, query_length=2**63))
    evidence = set(_positions("evidence", evidence_positions,
                              key_length=2**63, query_length=2**63))
    control = set(_positions("control", control_positions,
                             key_length=2**63, query_length=2**63))
    return {"qta_key_count": len(selected),
            "qta_evidence_overlap_count": len(selected & evidence),
            "qta_control_overlap_count": len(selected & control),
            "qta_evidence_overlap_fraction": len(selected & evidence) / len(selected),
            "qta_control_overlap_fraction": len(selected & control) / len(selected)}


def compute_text_annotation_selective_dependence(*, baseline_evidence_logprob: float,
        baseline_control_logprob: float, qta_evidence_logprob: float,
        qta_control_logprob: float) -> dict:
    """Compute the frozen exploratory four-condition text-score contrast.

    Positive values mean the generated target tokens' mean log-likelihood is
    more sensitive to benchmark-annotation-group removal than to matched
    control removal under QTA than under baseline. This is conditional text
    sensitivity, not claim truth or explanation faithfulness.
    """
    values = (baseline_evidence_logprob, baseline_control_logprob,
              qta_evidence_logprob, qta_control_logprob)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("text_dependence_logprobs_must_be_finite")
    baseline = float(baseline_control_logprob) - float(baseline_evidence_logprob)
    qta = float(qta_control_logprob) - float(qta_evidence_logprob)
    delta = qta - baseline
    return {"baseline_annotation_selective_text_dependence": baseline,
            "qta_annotation_selective_text_dependence": qta,
            "delta_annotation_selective_text_dependence": delta,
            "interpretation": "conditional generated-text sensitivity proxy; not faithfulness"}


def install_multiquery_video_key_intervention_hooks(*, model: Any,
        layer_indices: Sequence[int], biased_positions: Sequence[int],
        ablated_positions: Sequence[int], query_positions: Sequence[int],
        logit_bias: float, torch_module: Any) -> list[Any]:
    """Install the exact same fixed multi-query mask edit on selected layers."""
    try:
        layers = model.model.language_model.layers
    except AttributeError:
        raise ValueError("qwen_language_layers_unavailable") from None
    if (not isinstance(layer_indices, Sequence) or isinstance(layer_indices, (str, bytes))
            or not layer_indices
            or any(type(i) is not int or i < 0 or i >= len(layers) for i in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)):
        raise ValueError("intervention_layer_indices_invalid")
    frozen_layers = tuple(layer_indices)
    frozen_bias = tuple(biased_positions)
    frozen_ablation = tuple(ablated_positions)
    frozen_queries = tuple(query_positions)
    handles = []
    # One causal mask object is shared by all layers of a forward and the edit is identical per layer: edit and validate once per mask object.
    cache = {"mask": None, "edited": None, "built": None}
    try:
        for layer_index in frozen_layers:
            attention = getattr(layers[layer_index], "self_attn", None)
            register = getattr(attention, "register_forward_pre_hook", None)
            if not callable(register):
                raise ValueError("self_attention_pre_hook_unavailable")

            def pre_hook(_module, args, kwargs):
                if "attention_mask" not in kwargs:
                    raise ValueError("attention_mask_keyword_missing")
                kwargs = dict(kwargs)
                original = kwargs["attention_mask"]
                if original is None and torch_module.is_tensor(kwargs.get("hidden_states")):
                    # SDPA hands None for a plain causal forward: build the explicit mask once per installation (hooks are installed per forward)
                    if cache.get("built") is None:
                        from qta_mask_utils import causal_additive_mask
                        cache["built"] = causal_additive_mask(kwargs["hidden_states"], torch_module)
                    original = cache["built"]
                if cache["mask"] is not original:
                    cache["edited"] = apply_multiquery_video_key_intervention(
                        attention_mask=original,
                        biased_positions=frozen_bias, ablated_positions=frozen_ablation,
                        query_positions=frozen_queries, logit_bias=logit_bias,
                        torch_module=torch_module)
                    cache["mask"] = original
                kwargs["attention_mask"] = cache["edited"]
                return args, kwargs

            try:
                handles.append(register(pre_hook, with_kwargs=True))
            except TypeError:
                raise ValueError("pytorch_keyword_pre_hook_unavailable") from None
    except Exception:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles


def remove_hooks(handles: Sequence[Any]) -> None:
    first_error = None
    for handle in reversed(tuple(handles)):
        try:
            handle.remove()
        except Exception as exc:  # defensive cleanup
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError("multiquery_hook_cleanup_failed") from first_error
