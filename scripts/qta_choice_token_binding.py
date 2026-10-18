"""Derive A–E next-token IDs from the exact rendered QTA prompt.

This CPU helper verifies tokenizer continuation semantics. The caller still
owns the renderer, pinned-tokenizer provenance, multimodal processor call, and
model output; hashes and structural checks are not authentication or scoring
authorization.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
import re


LABELS = tuple("ABCDE")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def resolve_choice_continuation_ids(tokenizer, rendered_prompt: str,
                                    labels: str = "ABCDE") -> tuple[list[int], list[int]]:
    """Return ``(A–E continuation IDs, exact prompt-prefix IDs)``."""
    if (not isinstance(rendered_prompt, str) or not rendered_prompt
            or labels != "ABCDE"):
        raise ValueError("choice_prompt_or_label_order_invalid")
    try:
        prefix_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    except Exception:
        raise ValueError("choice_prompt_tokenization_failed") from None
    if (not isinstance(prefix_ids, list) or not prefix_ids
            or any(type(value) is not int or value < 0 for value in prefix_ids)):
        raise ValueError("choice_prompt_token_ids_invalid")
    result = []
    for label in labels:
        try:
            extended_ids = tokenizer.encode(rendered_prompt + label,
                                            add_special_tokens=False)
        except Exception:
            raise ValueError("choice_continuation_tokenization_failed") from None
        if (not isinstance(extended_ids, list)
                or extended_ids[:len(prefix_ids)] != prefix_ids):
            raise ValueError("choice_prompt_prefix_retokenized")
        suffix = extended_ids[len(prefix_ids):]
        if (len(suffix) != 1 or type(suffix[0]) is not int or suffix[0] < 0):
            raise ValueError("choice_label_not_single_continuation_token")
        result.append(suffix[0])
    if len(set(result)) != len(LABELS):
        raise ValueError("choice_continuation_ids_not_distinct")
    return result, list(prefix_ids)


def build_choice_token_binding(*, tokenizer, rendered_prompt: str,
                               tokenizer_sha256: str) -> dict:
    """Return a per-prompt digest and IDs derived from the supplied tokenizer."""
    if not isinstance(tokenizer_sha256, str) or not SHA256.fullmatch(tokenizer_sha256):
        raise ValueError("choice_tokenizer_digest_invalid")
    token_ids, prefix_ids = resolve_choice_continuation_ids(tokenizer, rendered_prompt)
    canonical_ids = _canonical(prefix_ids)
    prompt_sha = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    prefix_sha = hashlib.sha256(canonical_ids).hexdigest()
    return {
        "labels": list(LABELS),
        "prompt_sha256": prompt_sha,
        "tokenizer_sha256": tokenizer_sha256,
        "prompt_token_ids_sha256": prefix_sha,
        "answer_token_ids": token_ids,
        "binding_sha256": hashlib.sha256(_canonical({
            "labels": list(LABELS), "prompt_sha256": prompt_sha,
            "tokenizer_sha256": tokenizer_sha256,
            "prompt_token_ids_sha256": prefix_sha,
            "answer_token_ids": token_ids,
        })).hexdigest(),
        "verification_scope": "rendered_text_prompt_tokenizer_continuations_only",
    }


def validate_processed_choice_suffix(base_ids, extended_ids,
                                     expected_token_id: int) -> None:
    """Require a multimodal processor to append exactly one derived ID."""
    if (not isinstance(base_ids, Sequence) or isinstance(base_ids, (str, bytes))
            or not base_ids or any(type(value) is not int or value < 0 for value in base_ids)
            or not isinstance(extended_ids, Sequence)
            or isinstance(extended_ids, (str, bytes))
            or not extended_ids
            or any(type(value) is not int or value < 0 for value in extended_ids)
            or type(expected_token_id) is not int or expected_token_id < 0):
        raise ValueError("processed_choice_suffix_input_invalid")
    if list(extended_ids[:len(base_ids)]) != list(base_ids):
        raise ValueError("processed_choice_suffix_prefix_mismatch")
    if list(extended_ids[len(base_ids):]) != [expected_token_id]:
        raise ValueError("processed_choice_suffix_token_mismatch")
