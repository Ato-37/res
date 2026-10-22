#!/usr/bin/env python3
"""Bind a local frame-review response to private, exploratory frame-time rows.

This adapter does not authenticate reviewers, prove semantic support, authorize
model loading, or score faithfulness. It preserves every item and marks rows
eligible only when all submitted reviewers selected ``supports``; disagreements
remain in the denominator as ineligible rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

from inspect_claim_support_frame_review_response import (
    FINGERPRINT_RE,
    ITEMS_RE,
    PROVENANCE_RE,
    inspect,
)


def _read_regular(path: Path, *, limit: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > limit:
        raise ValueError("input_file_invalid")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input_file_too_large")
    return raw


def adapt(response: object, artifact_raw: bytes) -> dict:
    """Return aggregate-only, item-complete time mappings after exact binding."""
    validation = inspect(response, artifact_raw)
    if validation.get("structurally_valid") is not True:
        raise ValueError("review_binding_or_schema_invalid")

    items_match = ITEMS_RE.search(artifact_raw)
    provenance_match = PROVENANCE_RE.search(artifact_raw)
    fingerprint_match = FINGERPRINT_RE.search(artifact_raw)
    if not (items_match and provenance_match and fingerprint_match):
        raise ValueError("artifact_fields_missing")
    items = json.loads(items_match.group(1))
    provenance = json.loads(provenance_match.group(1))
    source_by_id = {row["item_id"]: row for row in provenance["items"]}
    answers_by_reviewer = [
        {answer["item_id"]: answer for answer in reviewer["answers"]}
        for reviewer in response["reviewers"]
    ]

    rows = []
    for item_index, item in enumerate(items):
        item_id = item["item_id"]
        times = [frame["time"] for frame in item["frames"]]
        if (len(times) != 16 or any(type(t) not in (int, float) or not math.isfinite(t)
                                    or t < 0 for t in times)
                or any(right <= left for left, right in zip(times, times[1:]))):
            raise ValueError("artifact_frame_times_invalid")
        decisions = [answer_map[item_id]["decision"] for answer_map in answers_by_reviewer]
        unanimous_support = bool(decisions) and all(d == "supports" for d in decisions)
        if unanimous_support:
            per_reviewer_frames = [set(answer_map[item_id]["frame_indices"])
                                   for answer_map in answers_by_reviewer]
            selected = sorted(set.intersection(*per_reviewer_frames)) if per_reviewer_frames else []
        else:
            selected = []
        eligible = unanimous_support and bool(selected)
        source = source_by_id[item_id]
        rows.append({
            "item_ordinal": item_index,
            "item_id_sha256": hashlib.sha256(item_id.encode("utf-8")).hexdigest(),
            "source_record_index": source.get("record_index"),
            "video_sha256": source.get("video_sha256"),
            "qid_sha256": source.get("qid_sha256"),
            "candidate_pair_sha256": source.get("candidate_pair_sha256"),
            "question_sha256": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
            "claim_sha256": hashlib.sha256(item["claim"].encode("utf-8")).hexdigest(),
            "reviewer_count": len(decisions),
            "decision_counts": {key: decisions.count(key) for key in
                                ("supports", "contradicts", "not_visible_in_model_frames", "cannot_judge")},
            "unanimous_support": unanimous_support,
            "eligible_for_exploratory_score": eligible,
            "selected_frame_indices_zero_based": selected,
            "selected_frame_times_seconds": [times[i] for i in selected],
        })
    return {
        "schema_version": 1,
        "purpose": "exploratory_claim_support_frame_time_mapping",
        "status": "mapped_unvalidated",
        "artifact_fingerprint": validation["artifact_fingerprint"],
        "source_manifest_sha256": validation["source_manifest_sha256"],
        "canonical_response_sha256": hashlib.sha256(json.dumps(
            response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "requested_rows": len(items),
        "effective_eligible_rows": sum(row["eligible_for_exploratory_score"] for row in rows),
        "rows": rows,
        "semantic_support_validated": False,
        "reviewer_identity_or_independence_verified": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
        "limitations": [
            "self-reported judgments are not authenticated or independently adjudicated",
            "support is limited to the 16 displayed model-input frame thumbnails",
            "all submitted reviewers must agree on supports; disagreements remain in requested_rows",
            "scored frame indices are the intersection selected by every reviewer; no union of unilateral selections",
            "this mapping is not a model score or a faithfulness result",
        ],
    }


def _write_private_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode) or parent.st_uid != os.getuid():
        raise ValueError("output_parent_invalid")
    os.chmod(path.parent, 0o700)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new local-only JSON path; existing outputs are never overwritten")
    args = parser.parse_args(argv)
    try:
        response_raw = _read_regular(args.response, limit=2_000_000)
        artifact_raw = _read_regular(args.artifact, limit=20_000_000)
        response = json.loads(response_raw)
        result = adapt(response, artifact_raw)
        _write_private_exclusive(args.output, result)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        print(json.dumps({"status": "blocked", "failure": "local_adapter_failed",
                          "model_load_permitted": False, "execution_authorization": "not_granted"}))
        return 2
    print(json.dumps({"status": result["status"], "requested_rows": result["requested_rows"],
                      "effective_eligible_rows": result["effective_eligible_rows"],
                      "model_load_permitted": False, "output_mode": "0600"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
