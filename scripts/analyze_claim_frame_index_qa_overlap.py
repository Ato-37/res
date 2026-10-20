#!/usr/bin/env python3
"""Exploratory overlap of machine-localized spans with NExT-GQA QA intervals.

This is a dataset-timing alignment diagnostic, not claim-specific grounding.
Run locally: inputs are protected mode-0600 private artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from fractions import Fraction
from pathlib import Path

from presentation_time_utils import strict_evidence_indices_from_times


def read_private(path: Path) -> tuple[dict, str]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def span_jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        raise ValueError("empty_interval_frame_set")
    return len(a & b) / len(a | b)


def analyze_rows(packet: dict, result: dict) -> list[dict]:
    records, rows = packet.get("records"), result.get("rows")
    if (packet.get("status") != "passed" or result.get("status") != "passed"
            or result.get("mode") != "full" or not isinstance(records, list)
            or len(records) != 3 or not isinstance(rows, list) or len(rows) != 3
            or result.get("effective_rows") != 3
            or result.get("completed_scorer_forwards") not in (9, 15)
            or result.get("completed_scorer_forwards") != result.get("requested_scorer_forwards")):
        raise ValueError("incomplete_pilot_pair")
    source_hashes = [record.get("video_sha256") for record in records]
    if (source_hashes != result.get("source_video_sha256")
            or len(set(source_hashes)) != 3):
        raise ValueError("packet_result_source_binding_invalid")

    output = []
    for index, (record, row) in enumerate(zip(records, rows)):
        if row.get("row_index") != index:
            raise ValueError("row_binding_invalid")
        pts_values = record.get("sampled_frame_relative_pts_seconds")
        spans = record.get("gold_interval_seconds")
        if (not isinstance(pts_values, list) or len(pts_values) != 16
                or not isinstance(spans, list) or not spans):
            raise ValueError("packet_timing_metadata_invalid")
        pts = [Fraction(str(value)) for value in pts_values]
        evidence = strict_evidence_indices_from_times(list(range(16)), pts, spans)
        start, end = row.get("start_frame_index"), row.get("end_frame_index")
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end < 16
                or not 1 <= end - start + 1 <= 4):
            raise ValueError("localized_span_invalid")
        selected = list(range(start, end + 1))
        observed = span_jaccard(selected, evidence)
        same_length_scores = []
        length = len(selected)
        for candidate_start in range(17 - length):
            candidate = list(range(candidate_start, candidate_start + length))
            same_length_scores.append(span_jaccard(candidate, evidence))
        percentile = sum(score <= observed for score in same_length_scores) / len(same_length_scores)
        upper_rank_fraction = sum(score >= observed for score in same_length_scores) / len(same_length_scores)
        output.append({
            "row_index": index,
            "localized_length_frames": length,
            "qa_interval_frame_count": len(evidence),
            "jaccard": observed,
            "same_length_jaccard_percentile": percentile,
            "same_length_at_least_as_high_fraction": upper_rank_fraction,
        })
    return output


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty_row_set")
    jaccard = [row["jaccard"] for row in rows]
    rank = [row["same_length_jaccard_percentile"] for row in rows]
    return {
        "n": len(rows),
        "mean_jaccard": sum(jaccard) / len(jaccard),
        "median_jaccard": sorted(jaccard)[len(jaccard) // 2]
            if len(jaccard) % 2 else sum(sorted(jaccard)[len(jaccard)//2-1:len(jaccard)//2+1]) / 2,
        "mean_same_length_percentile": sum(rank) / len(rank),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    packet, packet_sha = read_private(args.packet)
    result, result_sha = read_private(args.result)
    summary = summarize(analyze_rows(packet, result))
    summary.update({"packet_sha256": packet_sha, "result_sha256": result_sha,
                   "scope": "qa_interval_timing_alignment_not_claim_faithfulness"})
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
