"""Conservative syntax gate for short claim text sent to human review."""
from __future__ import annotations

import re


def claim_text_quality_failure(value: object) -> str | None:
    """Return a format-only failure code; do not judge semantic truth/support."""
    if not isinstance(value, str) or not value.strip():
        return "claim_empty"
    if value != value.strip():
        return "claim_outer_whitespace"
    if "\n" in value or "\r" in value:
        return "claim_multiline"
    if re.search(r"addcriterion|<\|[^>]+\|>|</?[a-z][^>]*>", value, re.I):
        return "claim_template_or_token_artifact"
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", value)
    if len(words) < 2 or len(words) > 40 or not value[0].isupper():
        return "claim_sentence_shape_invalid"
    if value[-1] not in ".?!" or len(re.findall(r"[.!?](?:\s|$)", value)) > 1:
        return "claim_sentence_shape_invalid"
    return None
