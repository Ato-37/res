#!/usr/bin/env python3
"""Build a private, blinded draft worksheet from a local claim-review packet.

This CPU-only builder copies only the frozen question and parsed claim plus
local source references. It excludes answers, choices, gold intervals, model
frame selections, and raw generations. The resulting worksheet is not a human
review, protocol approval, or authorization to score data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SHA256 = re.compile(r"^[a-f0-9]{64}$")
GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")
PURPOSE = "private_local_nextgqa_model_generated_claim_review_packet"
FROZEN_POLICY_SHA256 = "0fe92fff658aa25280b0e32c6db979d8903c9ad4d1ddfc592bb38601d459bfb7"
DATASET_INPUT_KEYS = {
    "test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json",
}


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
        return any(_nonfinite(k) or _nonfinite(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_nonfinite(v) for v in value)
    return False


def _text(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
    )
    if _nonfinite(value):
        raise ValueError("non_finite_json")
    return value, raw


def _raw_matches(value: Any, raw: bytes) -> bool:
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


def build(packet: Any, packet_raw: bytes) -> tuple[dict, bytes]:
    """Validate a v3 review packet and return a blinded worksheet and bytes."""
    if not isinstance(packet, dict) or packet.get("schema_version") != 1 \
            or packet.get("purpose") != PURPOSE:
        raise ValueError("claim_packet_schema_invalid")
    if not _raw_matches(packet, packet_raw):
        raise ValueError("claim_packet_bytes_invalid")
    if _nonfinite(packet):
        raise ValueError("claim_packet_nonfinite")
    records = packet.get("records")
    if not isinstance(records, list) or len(records) > 3:
        raise ValueError("claim_packet_records_invalid")
    requested = packet.get("requested_videos")
    effective = packet.get("effective_videos")
    if (isinstance(requested, bool) or not isinstance(requested, int)
            or requested not in (0, 1, 2, 3)
            or isinstance(effective, bool) or not isinstance(effective, int)
            or effective != len(records) or effective > requested):
        raise ValueError("claim_packet_counts_invalid")
    cohort_sha = packet.get("cohort_sha256")
    if not _sha(cohort_sha):
        raise ValueError("claim_packet_cohort_hash_invalid")
    model_revision = packet.get("model_revision")
    if not _text(model_revision):
        raise ValueError("claim_packet_model_revision_invalid")
    source_commit = packet.get("source_commit")
    if source_commit is not None and (
            not isinstance(source_commit, str) or not GIT_COMMIT.fullmatch(source_commit)):
        raise ValueError("claim_packet_source_commit_invalid")
    runner_hashes = {}
    for packet_key, output_key in (("runner_sha256", "claim_packet_runner_sha256"),
                                   ("pilot_runner_sha256", "cohort_selector_runner_sha256")):
        digest = packet.get(packet_key)
        if digest is not None:
            if not _sha(digest):
                raise ValueError("claim_packet_runner_hash_invalid")
            runner_hashes[output_key] = digest
    dataset_hashes = packet.get("dataset_inputs_sha256")
    if (not isinstance(dataset_hashes, dict)
            or set(dataset_hashes) != DATASET_INPUT_KEYS
            or any(not _sha(digest) for digest in dataset_hashes.values())):
        raise ValueError("claim_packet_dataset_hashes_invalid")

    result_records = []
    seen_video_hashes: set[str] = set()
    seen_claim_ids: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("claim_packet_record_invalid")
        video_path = row.get("local_video_path")
        video_sha = row.get("video_sha256")
        question = row.get("question")
        qid = row.get("dataset_question_id")
        parsed = row.get("generated_parsed")
        if (not _text(video_path) or not _sha(video_sha) or not _text(question)
                or not _text(qid) or not isinstance(parsed, dict)
                or not _text(parsed.get("claim"))):
            raise ValueError("claim_packet_required_claim_missing_or_invalid")
        if video_sha in seen_video_hashes:
            raise ValueError("claim_packet_video_duplicate")
        seen_video_hashes.add(video_sha)
        claim = parsed["claim"]
        claim_sha = hashlib.sha256(claim.encode("utf-8")).hexdigest()
        video_id = "video-" + video_sha
        claim_id = "claim-" + hashlib.sha256(
            (video_sha + "\0" + qid + "\0" + claim_sha).encode("utf-8")
        ).hexdigest()
        if claim_id in seen_claim_ids:
            raise ValueError("claim_packet_claim_duplicate")
        seen_claim_ids.add(claim_id)
        result_records.append({
            "video_path": video_path,
            "video_sha256": video_sha,
            "video_id": video_id,
            "claim_id": claim_id,
            "question": question,
            "claim_text": claim,
            "claim_text_sha256": claim_sha,
            "review": {
                "source_identity_verified": None,
                "full_source_video_reviewed": None,
                "question_and_claim_only_context_shown": None,
                "gold_and_seed_hidden_until_mapping_frozen": None,
                "outcome_scores_not_accessed": None,
                "occurrence_policy_applied": None,
                "support_decision": None,
                "evidence_intervals_seconds": None,
                "rationale": None,
                "reviewer_role": None,
                "review_record_reference": None,
                "reviewed_at_utc": None,
                "review_status": "not_started",
            },
        })

    output = {
        "schema_version": 1,
        "purpose": "private_blinded_claim_to_evidence_mapping_worksheet",
        "status": "draft_not_reviewed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_packet_sha256": hashlib.sha256(packet_raw).hexdigest(),
        "cohort_sha256": cohort_sha,
        "model_revision": model_revision,
        "source_commit": source_commit,
        "claim_packet_runner_sha256": runner_hashes.get("claim_packet_runner_sha256"),
        "cohort_selector_runner_sha256": runner_hashes.get("cohort_selector_runner_sha256"),
        "dataset_inputs_sha256": dataset_hashes,
        "occurrence_policy": "unfrozen",
        "review_procedure_status": "unfrozen",
        "requested_records": requested,
        "effective_records": len(result_records),
        "claim_records_available": len(result_records),
        "review_completed": False,
        "execution_authorization": "not_granted",
        "records": result_records,
    }
    encoded = (json.dumps(output, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    return output, encoded


def apply_review_policy(output: dict, policy: Any, policy_raw: bytes) -> dict:
    """Bind a worksheet to the exact pre-review procedure artifact."""
    expected = {
        "schema_version", "policy_id", "purpose", "occurrence_policy",
        "review_scope",
        "display_procedure", "permitted_evidence", "support_rule",
        "interval_rule", "failure_rule", "review_order",
    }
    if not isinstance(policy, dict) or set(policy) != expected:
        raise ValueError("review_policy_schema_invalid")
    if not _raw_matches(policy, policy_raw):
        raise ValueError("review_policy_bytes_invalid")
    if (policy.get("schema_version") != 1
            or policy.get("policy_id") != "nextgqa_claim_review_v1"
            or policy.get("purpose") != "pre_outcome_human_claim_to_video_mapping"
            or policy.get("occurrence_policy") != "all_relevant_occurrences"
            or hashlib.sha256(policy_raw).hexdigest() != FROZEN_POLICY_SHA256
            or output.get("requested_records") != 3
            or output.get("effective_records") != 3
            or not all(_text(policy.get(key)) for key in expected - {"schema_version"})):
        raise ValueError("review_policy_values_invalid")
    output["claim_review_policy_id"] = policy["policy_id"]
    output["claim_review_policy_sha256"] = hashlib.sha256(policy_raw).hexdigest()
    output["occurrence_policy"] = policy["occurrence_policy"]
    output["review_procedure_status"] = "frozen_pre_review"
    return output


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _verify_source_media(payload: dict) -> None:
    """Verify media references before writing, without exposing failure paths."""
    for row in payload["records"]:
        path = Path(row["video_path"])
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ValueError("source_video_missing_or_unreadable") from exc
        if digest.hexdigest() != row["video_sha256"]:
            raise ValueError("source_video_hash_mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="private local v3 claim-review JSON")
    parser.add_argument("--policy", required=True, help="frozen pre-review policy JSON")
    parser.add_argument("--output", required=True, help="new private draft worksheet JSON")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    source = Path(args.input).resolve()
    target = Path(args.output).resolve()
    if source == target:
        parser.error("input_and_output_paths_must_differ")
    repo_root = Path(__file__).resolve().parents[1]
    if target.is_relative_to(repo_root):
        parser.error("output_must_be_outside_repository")
    if target.exists():
        parser.error("refusing_to_overwrite_output")
    try:
        packet, raw = _read_json(source)
        payload, encoded = build(packet, raw)
        policy, policy_raw = _read_json(Path(args.policy).resolve())
        payload = apply_review_policy(payload, policy, policy_raw)
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n").encode("utf-8")
        _verify_source_media(payload)
        _write_exclusive(target, encoded)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except OSError:
        parser.error("input_or_output_io_error")
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        # Error codes only: never echo paths, claims, questions, or raw input.
        code = str(exc).split(":", 1)[0] or "claim_packet_invalid"
        parser.error(code)
    print(f"Draft mapping worksheet created: {len(payload['records'])} records; "
          "status=draft_not_reviewed; execution_authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
