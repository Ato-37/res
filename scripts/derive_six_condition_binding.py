"""Derive six-branch scoring bindings from CPU-resident batch tensors.

This helper establishes consistency of the supplied synthetic/model batches;
it does not authenticate their origin or satisfy any human-review/data gate.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from fixed_prediction_selective_dependence import (
    ANSWER_LABEL_ORDER,
    INTERVENTION_CONDITIONS,
    validate_six_condition_input_binding,
)
from nextgqa_atomic_batch_freeze import batch_sha256


BRANCH_PAIRS = (
    ("baseline_full", "intervention_full"),
    ("baseline_evidence_ablation", "intervention_evidence_ablation"),
    ("baseline_control_ablation", "intervention_control_ablation"),
)
REQUIRED_QWEN_VIDEO_BATCH_KEYS = frozenset({
    "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
})


def _tensor_array(value: object, *, field: str, condition: str):
    if not callable(getattr(value, "detach", None)):
        raise ValueError(f"{condition}_{field}_tensor_required")
    device = getattr(value, "device", None)
    if device is not None and getattr(device, "type", None) != "cpu":
        raise ValueError(f"{condition}_{field}_must_be_cpu_resident")
    try:
        array = value.detach().contiguous().numpy()
    except Exception:
        raise ValueError(f"{condition}_{field}_conversion_failed") from None
    if getattr(array, "ndim", None) != 2 or int(array.shape[0]) != 1:
        raise ValueError(f"{condition}_{field}_shape_invalid")
    return array


def _tensor_digest(array) -> str:
    header = json.dumps(
        {"shape": list(array.shape), "dtype": str(array.dtype)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _dtype_kind(dtype: object) -> str | None:
    kind = getattr(dtype, "kind", None)
    if isinstance(kind, str):
        return kind
    value = str(dtype).lower().replace("torch.", "")
    if value in {"bool", "bool_"}:
        return "b"
    if value.startswith(("int", "uint")):
        return "i" if value.startswith("int") else "u"
    if value.startswith(("float", "double", "half", "bfloat")):
        return "f"
    return None


def derive_six_condition_binding(
    *,
    batches: Mapping[str, Mapping],
    answer_token_ids: Sequence[int],
    model_revision: str,
    tokenizer_sha256: str,
    prompt_sha256: str,
    condition_declarations: Mapping[str, Mapping] | None = None,
) -> dict:
    """Build validator-compatible bindings from the actual six batch tensors.

    Explicit metadata is shared across branches. Optional condition declarations
    are consistency assertions and must match derived hashes and metadata; they
    cannot authenticate the caller or model provenance.
    """
    if not isinstance(batches, Mapping) or set(batches) != INTERVENTION_CONDITIONS:
        raise ValueError("exact_six_intervention_batches_required")
    if (not isinstance(answer_token_ids, Sequence)
            or isinstance(answer_token_ids, (str, bytes))
            or len(answer_token_ids) != 5
            or any(type(token_id) is not int or token_id < 0
                   for token_id in answer_token_ids)
            or len(set(answer_token_ids)) != 5):
        raise ValueError("answer_token_ids_invalid")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("model_revision_invalid")
    for name, digest in (("tokenizer_sha256", tokenizer_sha256),
                         ("prompt_sha256", prompt_sha256)):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError(f"{name}_invalid")
    if condition_declarations is not None and (
            not isinstance(condition_declarations, Mapping)
            or set(condition_declarations) != INTERVENTION_CONDITIONS):
        raise ValueError("condition_declarations_invalid")

    rows = {}
    batch_hashes = {}
    canonical_input = canonical_mask = canonical_length = None
    for condition in sorted(INTERVENTION_CONDITIONS):
        batch = batches[condition]
        required_keys = {"input_ids", "attention_mask"} | REQUIRED_QWEN_VIDEO_BATCH_KEYS
        if not isinstance(batch, Mapping) or not required_keys.issubset(batch):
            raise ValueError(f"{condition}_required_tensors_missing")
        input_array = _tensor_array(batch["input_ids"], field="input_ids", condition=condition)
        mask_array = _tensor_array(batch["attention_mask"], field="attention_mask", condition=condition)
        if input_array.shape != mask_array.shape or int(input_array.shape[1]) <= 0:
            raise ValueError(f"{condition}_input_mask_shape_mismatch")
        if _dtype_kind(input_array.dtype) not in {"i", "u"}:
            raise ValueError(f"{condition}_input_ids_integer_dtype_required")
        if _dtype_kind(mask_array.dtype) not in {"i", "u", "b"}:
            raise ValueError(f"{condition}_attention_mask_integer_or_bool_dtype_required")
        mask_values = mask_array.tolist()[0]
        if any(value not in (0, 1, False, True) for value in mask_values):
            raise ValueError(f"{condition}_attention_mask_not_binary")
        input_digest = _tensor_digest(input_array)
        mask_digest = _tensor_digest(mask_array)
        length = int(input_array.shape[1])
        if canonical_input is None:
            canonical_input, canonical_mask, canonical_length = input_array, mask_array, length
        else:
            if (input_array.shape != canonical_input.shape
                    or input_array.dtype != canonical_input.dtype
                    or input_array.tobytes(order="C") != canonical_input.tobytes(order="C")):
                raise ValueError("six_condition_input_ids_mismatch")
            if (mask_array.shape != canonical_mask.shape
                    or mask_array.dtype != canonical_mask.dtype
                    or mask_array.tobytes(order="C") != canonical_mask.tobytes(order="C")):
                raise ValueError("six_condition_attention_mask_mismatch")
            if length != canonical_length:
                raise ValueError("six_condition_score_position_mismatch")
        row = {
            "answer_label_order": list(ANSWER_LABEL_ORDER),
            "answer_token_ids": list(answer_token_ids),
            "input_ids_sha256": input_digest,
            "attention_mask_sha256": mask_digest,
            "tokenizer_sha256": tokenizer_sha256,
            "model_revision": model_revision,
            "prompt_sha256": prompt_sha256,
            "input_length": length,
            "score_position": length - 1,
        }
        if condition_declarations is not None:
            declared = condition_declarations[condition]
            if not isinstance(declared, Mapping) or dict(declared) != row:
                raise ValueError(f"{condition}_binding_declaration_mismatch")
        rows[condition] = row
        try:
            batch_hashes[condition] = batch_sha256(batch)
        except Exception as exc:
            code = str(exc)
            if not code.isascii() or not code.replace("_", "").isalnum():
                code = "typed_batch_hash_failed"
            raise ValueError(f"{condition}_{code}") from None

    for baseline, treated in BRANCH_PAIRS:
        if batch_hashes[baseline] != batch_hashes[treated]:
            raise ValueError(f"paired_batch_hash_mismatch_{baseline.removeprefix('baseline_')}")

    validated = validate_six_condition_input_binding(rows)
    full_binding_payload = json.dumps(
        {
            "text_binding_sha256": validated["binding_sha256"],
            "batch_sha256_by_condition": batch_hashes,
        }, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        **validated,
        "status": "derived_synthetic_batch_binding_contract_only",
        "input_ids_sha256": rows["baseline_full"]["input_ids_sha256"],
        "attention_mask_sha256": rows["baseline_full"]["attention_mask_sha256"],
        "text_binding_sha256": validated["binding_sha256"],
        "batch_sha256_by_condition": batch_hashes,
        "full_binding_sha256": hashlib.sha256(full_binding_payload).hexdigest(),
        "interpretation_limit": (
            "text hashes derived from supplied CPU tensors and complete typed-batch hashes "
            "including video/pixel tensors and processor metadata; explicit metadata "
            "and optional declarations remain caller-supplied, with no tokenizer/model "
            "provenance authentication, human review, real-data authorization, or efficacy"
        ),
    }
