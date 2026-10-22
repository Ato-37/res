#!/usr/bin/env python3
"""Freeze all eligible 16-frame v2_short claim packets for proxy replication.

Selection depends only on frozen claim validity, one QA interval, sampled-frame
coverage, equal-size disjoint control availability, a disjoint hash sham, and
unique source media. No outcome from the internal-bias endpoint is consulted.
The input/output JSON files contain protected local paths/text and must remain
mode 0600 under data/private_nextgqa.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from run_claim_text_qa_interval_key_bias import (
    MODEL_REVISION, SHAM_PROTOCOL, align_interval_and_control_groups,
    read_private_json, select_disjoint_sham_groups,
)

PACKET_NAMES = tuple(
    f"claim_likelihood_claims_remaining_20261021_{suffix}.json"
    for suffix in ("06", "07", "08", "09", "14", "15", "16", "17",
                   "18", "19", "20", "21", "22")
)
FRAME_COUNT = 16
TEMPORAL_PATCH_SIZE = 2


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_packet(repo: Path, output: Path) -> dict:
    private = (repo / "data/private_nextgqa").resolve(strict=True)
    eligible: list[dict] = []
    packet_hashes: dict[str, str] = {}
    dataset_hashes = None
    valid_candidates = 0
    exclusion_counts: dict[str, int] = {}
    seen_media: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for name in PACKET_NAMES:
        path = private / name
        packet, raw = read_private_json(path, private)
        packet_hashes[name] = sha256_bytes(raw)
        if (packet.get("status") not in {"passed", "partial"}
                or packet.get("dataset_split") != "test"
                or packet.get("model_revision") != MODEL_REVISION
                or packet.get("generation_prompt_version") != "v2_short"
                or packet.get("frames_per_video") != FRAME_COUNT
                or not isinstance(packet.get("records"), list)):
            raise ValueError(f"source_packet_contract_invalid:{name}")
        current_hashes = packet.get("dataset_inputs_sha256")
        if not isinstance(current_hashes, dict) or dataset_hashes is not None and current_hashes != dataset_hashes:
            raise ValueError(f"source_dataset_hashes_differ:{name}")
        dataset_hashes = current_hashes

        for position, row in enumerate(packet["records"]):
            parsed = row.get("generated_parsed") if isinstance(row, dict) else None
            claim = parsed.get("claim") if isinstance(parsed, dict) else None
            if (not isinstance(parsed, dict) or parsed.get("valid") is not True
                    or not isinstance(claim, str) or not claim.strip()
                    or row.get("generation_prompt_version") != "v2_short"
                    or not isinstance(row.get("sampled_frame_relative_pts_seconds"), list)
                    or len(row["sampled_frame_relative_pts_seconds"]) != FRAME_COUNT):
                exclusion_counts["invalid_or_non_v2_claim"] = exclusion_counts.get("invalid_or_non_v2_claim", 0) + 1
                continue
            valid_candidates += 1
            video_sha = row.get("video_sha256")
            qid = str(row.get("dataset_question_id", ""))
            pair = (video_sha, qid)
            if (not isinstance(video_sha, str) or len(video_sha) != 64
                    or video_sha in seen_media or pair in seen_pairs):
                raise ValueError("duplicate_or_invalid_source_media_pair")
            try:
                mapping = align_interval_and_control_groups(
                    sampled_pts=row["sampled_frame_relative_pts_seconds"],
                    intervals=row["gold_interval_seconds"], frame_count=FRAME_COUNT)
                row_token = sha256_bytes((video_sha + qid).encode())
                sham = select_disjoint_sham_groups(
                    group_count=FRAME_COUNT // TEMPORAL_PATCH_SIZE,
                    group_budget=len(mapping["evidence_group_indices"]),
                    evidence_groups=mapping["evidence_group_indices"],
                    control_groups=mapping["control_group_indices"], row_token=row_token)
            except (KeyError, TypeError, ValueError) as exc:
                reason = str(exc).split(":", 1)[0]
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                continue
            seen_media.add(video_sha)
            seen_pairs.add(pair)
            eligible.append(row)

    if len(eligible) != 19 or valid_candidates != 32:
        raise ValueError(f"frozen_pool_count_mismatch:{valid_candidates}:{len(eligible)}")
    output = output.resolve(strict=False)
    if output.parent != private or output.suffix != ".json":
        raise ValueError("output_must_be_private_json")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "dataset_split": "test",
        "model_revision": MODEL_REVISION,
        "generation_prompt_version": "v2_short",
        "frames_per_video": FRAME_COUNT,
        "claim_review_completed": False,
        "requested_videos": len(eligible),
        "effective_videos": len(eligible),
        "fully_structured_claim_records": len(eligible),
        "selection_protocol": "all valid saved v2_short claims from frozen source packet list with exactly one QA interval, at least one sampled QA frame, same-length disjoint nearest temporal-group control, available hash-ranked disjoint sham, and unique source video; no internal-bias outcome consulted",
        "sham_protocol": SHAM_PROTOCOL,
        "candidate_valid_claim_rows": valid_candidates,
        "eligible_rows": len(eligible),
        "exclusion_counts": exclusion_counts,
        "source_packets_sha256": packet_hashes,
        "dataset_inputs_sha256": dataset_hashes,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "records": eligible,
        "rights_status": "unverified; local-only per user instruction",
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = repo / output
    result = build_packet(repo, output)
    print(json.dumps({"status": result["status"], "candidate_valid_claim_rows": result["candidate_valid_claim_rows"],
                      "eligible_rows": result["eligible_rows"], "excluded": result["exclusion_counts"],
                      "output_name": output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
