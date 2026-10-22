#!/usr/bin/env python3
"""Validate and aggregate a local exploratory review-v2 JSON without PII output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

LABELS = {"roughly_matches", "partly_differs", "cannot_tell"}
FINGERPRINT_RE = re.compile(
    r'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">')


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _embedded_binding(html: str) -> tuple[str, dict, list[dict]]:
    match = FINGERPRINT_RE.search(html)
    if not match:
        raise ValueError("artifact_fingerprint_missing")
    canonical = html[:match.start(1)] + "__FP__" + html[match.end(1):]
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if fingerprint != match.group(1):
        raise ValueError("artifact_fingerprint_invalid")
    marker = "<script>const items="
    start = html.find(marker)
    if start < 0:
        raise ValueError("artifact_items_missing")
    start += len(marker)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    items, end = decoder.raw_decode(html[start:])
    suffix = html[start + end:]
    binding_marker = ",binding="
    if not suffix.startswith(binding_marker):
        raise ValueError("artifact_binding_missing")
    binding, _ = decoder.raw_decode(suffix[len(binding_marker):])
    if not isinstance(binding, dict) or not isinstance(items, list) or len(items) != 3:
        raise ValueError("artifact_schema_invalid")
    return match.group(1), binding, items


def inspect(response_path: Path, artifact_path: Path) -> dict:
    html = artifact_path.read_text(encoding="utf-8")
    fingerprint, binding, items = _embedded_binding(html)
    response = json.loads(response_path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object)
    if not isinstance(response, dict):
        raise ValueError("response_object_required")
    if (response.get("schema_version") != 1
            or response.get("purpose") != "exploratory_local_human_review"
            or response.get("status") != "informal_observations_only"):
        raise ValueError("response_scope_mismatch")
    if response.get("artifact_fingerprint") != fingerprint:
        raise ValueError("response_artifact_fingerprint_mismatch")
    if response.get("source_binding") != binding:
        raise ValueError("response_source_binding_mismatch")
    respondents = response.get("respondents")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("respondent_sessions_required")
    expected = {item.get("item_id"): i for i, item in enumerate(items, start=1)}
    if None in expected or len(expected) != 3:
        raise ValueError("artifact_item_ids_invalid")
    counts = {label: 0 for label in sorted(LABELS)}
    sessions = 0
    for respondent in respondents:
        if not isinstance(respondent, dict):
            raise ValueError("respondent_schema_invalid")
        nickname = respondent.get("nickname")
        if not isinstance(nickname, str) or not nickname.strip() or len(nickname) > 40:
            raise ValueError("nickname_invalid")
        answers = respondent.get("answers")
        if not isinstance(answers, list) or len(answers) != 3:
            raise ValueError("review_session_incomplete")
        seen = set()
        for answer in answers:
            if not isinstance(answer, dict):
                raise ValueError("answer_schema_invalid")
            item_id = answer.get("item_id")
            if item_id not in expected or item_id in seen:
                raise ValueError("answer_item_binding_invalid")
            if answer.get("item_number") != expected[item_id]:
                raise ValueError("answer_item_order_invalid")
            if answer.get("judgment") not in LABELS:
                raise ValueError("answer_label_invalid")
            note = answer.get("note")
            if note is not None and (not isinstance(note, str) or len(note) > 500):
                raise ValueError("answer_note_invalid")
            seen.add(item_id)
            counts[answer["judgment"]] += 1
        sessions += 1
    return {"analysis": "aggregate_only_informal_video_claim_review",
            "artifact_fingerprint_valid": True,
            "source_binding_valid": True,
            "complete_reviewer_sessions": sessions,
            "items_per_session": 3,
            "judgment_counts": counts,
            "formal_faithfulness_evidence": False,
            "interpretation": "informal local observations only; reviewer identity is not authenticated"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.response, args.artifact), ensure_ascii=False,
                     sort_keys=True))


if __name__ == "__main__":
    main()
