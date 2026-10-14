#!/usr/bin/env python3
"""Freeze already-generated claim text into the private review-input schema.

This CPU-only builder performs no model/video access and grants no execution
authorization. It accepts an eligible manifest and a private generation export,
then writes a closed v1 claim-input manifest without overwriting existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[a-f0-9]{64}$")
MODEL_REVISION = re.compile(r"^.+@[a-f0-9]{40}$")
OCCURRENCE_POLICIES = {"one_sufficient_occurrence", "all_relevant_occurrences"}
GENERATION_KEYS = {"video_id", "claim_id", "claim_text", "model_revision", "generation_run_sha256"}


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_nonfinite(key) or _nonfinite(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_nonfinite(item) for item in value)
    return False


def _read(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
    )
    if _nonfinite(value):
        raise ValueError("non_finite_json")
    return value, raw


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _raw_matches(value: Any, raw: Any) -> bool:
    if not isinstance(raw, bytes):
        return False
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
        )
        if _nonfinite(parsed) or _nonfinite(value):
            return False
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (AttributeError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return False


def build(eligible: Any, eligible_raw: bytes, generation: Any, dataset_id: str,
          claim_policy_sha256: str, occurrence_policy: str) -> tuple[dict, bytes]:
    failures: set[str] = set()
    if not _raw_matches(eligible, eligible_raw):
        failures.add("eligible_manifest_bytes_invalid")
    if not isinstance(eligible, list) or not eligible:
        failures.add("eligible_manifest_invalid")
        eligible_rows = []
    else:
        eligible_rows = eligible
    eligible_ids = []
    for row in eligible_rows:
        video_id = row.get("video_id") if isinstance(row, dict) else None
        if not _text(video_id):
            failures.add("eligible_video_id_invalid")
        else:
            eligible_ids.append(video_id)
    if len(eligible_ids) != len(set(eligible_ids)):
        failures.add("eligible_video_id_duplicate")
    if not _text(dataset_id):
        failures.add("eligible_dataset_id_invalid")
    if not isinstance(claim_policy_sha256, str) or SHA256.fullmatch(claim_policy_sha256) is None:
        failures.add("claim_policy_hash_invalid")
    if not isinstance(occurrence_policy, str) or occurrence_policy not in OCCURRENCE_POLICIES:
        failures.add("occurrence_policy_invalid")
    if (not isinstance(generation, dict) or set(generation) != {"schema_version", "claims"}
            or type(generation.get("schema_version")) is not int or generation.get("schema_version") != 1):
        failures.add("generation_manifest_schema_invalid")
        generation = {}
    generation_rows = generation.get("claims")
    if not isinstance(generation_rows, list):
        failures.add("generation_rows_invalid")
        generation_rows = []
    by_video = {}
    for row in generation_rows:
        if not isinstance(row, dict) or set(row) != GENERATION_KEYS:
            failures.add("generation_row_invalid")
            continue
        video_id = row.get("video_id")
        claim_text = row.get("claim_text")
        if (not _text(video_id) or not _text(row.get("claim_id"))
                or not isinstance(claim_text, str) or not claim_text.strip()
                or not isinstance(row.get("model_revision"), str)
                or MODEL_REVISION.fullmatch(row["model_revision"]) is None
                or not isinstance(row.get("generation_run_sha256"), str)
                or SHA256.fullmatch(row["generation_run_sha256"]) is None):
            failures.add("generation_row_invalid")
            continue
        try:
            claim_text_sha256 = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            failures.add("claim_text_invalid_unicode")
            continue
        if video_id in by_video:
            failures.add("generation_video_id_duplicate")
        by_video[video_id] = {
            "video_id": video_id, "claim_id": row["claim_id"], "claim_text": claim_text,
            "claim_text_sha256": claim_text_sha256, "model_revision": row["model_revision"],
            "generation_run_sha256": row["generation_run_sha256"],
        }
    expected, supplied = set(eligible_ids), set(by_video)
    if expected - supplied:
        failures.add("generation_rows_missing")
    if supplied - expected:
        failures.add("generation_rows_outside_eligible_cohort")
    revisions = {row["model_revision"] for row in by_video.values()}
    if len(revisions) > 1:
        failures.add("generation_model_revision_mixed")
    if failures:
        raise ValueError(",".join(sorted(failures)))
    manifest = {
        "schema_version": 1,
        "eligible_dataset_id": dataset_id,
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "claim_policy_sha256": claim_policy_sha256,
        "occurrence_policy": occurrence_policy,
        "claims": [by_video[video_id] for video_id in eligible_ids],
    }
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return manifest, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--generation-export", required=True, type=Path)
    parser.add_argument("--eligible-dataset-id", required=True)
    parser.add_argument("--claim-policy-sha256", required=True)
    parser.add_argument("--occurrence-policy", required=True, choices=sorted(OCCURRENCE_POLICIES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        eligible, eligible_raw = _read(args.eligible_manifest)
        generation, _generation_raw = _read(args.generation_export)
        _, output_raw = build(eligible, eligible_raw, generation, args.eligible_dataset_id,
                              args.claim_policy_sha256, args.occurrence_policy)
        output_fd = None
        created = False
        try:
            output_fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            os.fchmod(output_fd, 0o600)
            with os.fdopen(output_fd, "wb") as stream:
                output_fd = None
                stream.write(output_raw)
        except OSError:
            if output_fd is not None:
                os.close(output_fd)
            if created:
                args.output.unlink(missing_ok=True)
            raise
        report = {"status": "created", "requested_eligible_video_count": len(eligible),
                  "effective_claim_count": len(eligible),
                  "output_sha256": hashlib.sha256(output_raw).hexdigest(),
                  "execution_authorization": "not_granted"}
        code = 0
    except FileExistsError:
        report = {"status": "failed", "failure_codes": ["output_already_exists"],
                  "execution_authorization": "not_granted"}
        code = 1
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        report = {"status": "failed", "failure_codes": ["private_input_invalid_or_inconsistent"],
                  "execution_authorization": "not_granted"}
        code = 1
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
