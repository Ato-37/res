"""CPU-check tokenizer choice continuations against multimodal processor IDs."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from qta_choice_token_binding import resolve_choice_continuation_ids
from qta_choice_token_binding import validate_processed_choice_suffix


def verify_processor_choice_suffix(*, tokenizer, rendered_prompt: str,
                                    process_text: Callable[[str], Sequence[int]],
                                    labels: str = "ABCDE") -> dict:
    """Require processor token IDs to append the tokenizer-derived choice ID.

    `process_text` must process each text with the same fixed multimodal inputs.
    This helper authenticates neither that those inputs are real nor the model's
    output formatting; it checks only continuation-ID alignment.
    """
    if not callable(process_text):
        raise ValueError("processor_choice_callback_invalid")
    token_ids, _ = resolve_choice_continuation_ids(tokenizer, rendered_prompt, labels)
    base_ids = process_text(rendered_prompt)
    if not isinstance(base_ids, Sequence) or isinstance(base_ids, (str, bytes)):
        raise ValueError("processor_choice_base_ids_invalid")
    for label, token_id in zip(labels, token_ids, strict=True):
        extended_ids = process_text(rendered_prompt + label)
        try:
            validate_processed_choice_suffix(base_ids, extended_ids, token_id)
        except ValueError:
            raise ValueError("processor_choice_suffix_mismatch") from None
    return {"labels": list(labels), "answer_token_ids": token_ids,
            "processed_suffix_matches": len(labels),
            "verification_scope": "same_multimodal_inputs_and_text_continuation_ids_only"}
