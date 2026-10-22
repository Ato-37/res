#!/usr/bin/env python3
"""Aggregate-only audit of NExT-GQA QA-to-evidence-interval coverage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(split_csv: Path, gsub_json: Path) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "qid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("split_csv_missing_video_id_or_qid")
        for row in reader:
            video_id, qid = row.get("video_id"), row.get("qid")
            if not video_id or qid is None or not qid:
                raise ValueError("split_csv_missing_join_key")
            rows.append((video_id, qid))
    if len(rows) != len(set(rows)):
        raise ValueError("duplicate_split_video_qid")

    annotations = json.loads(gsub_json.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        raise ValueError("gsub_root_must_be_object")
    annotated = {
        (video_id, str(qid))
        for video_id, record in annotations.items()
        for qid in record.get("location", {})
    }
    split = set(rows)
    counts = Counter()
    alternatives_per_qa: Counter[int] = Counter()
    spans = 0
    for video_id, record in annotations.items():
        duration = record.get("duration")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("invalid_video_duration")
        locations = record.get("location")
        if not isinstance(locations, dict):
            raise ValueError("invalid_location_map")
        for qid, alternatives in locations.items():
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError("empty_or_invalid_alternative_spans")
            alternatives_per_qa[len(alternatives)] += 1
            for span in alternatives:
                spans += 1
                if (not isinstance(span, list) or len(span) != 2
                        or not all(not isinstance(x, bool) and isinstance(x, (int, float))
                                   and math.isfinite(x) for x in span)):
                    counts["malformed"] += 1
                    continue
                start, end = span
                if start < 0:
                    counts["negative_start"] += 1
                elif start >= end:
                    counts["nonpositive_or_reversed"] += 1
                elif end > duration:
                    counts["end_after_video_duration"] += 1
                else:
                    counts["valid_positive_interval"] += 1

    return {
        "csv_sha256": sha256(split_csv),
        "gsub_sha256": sha256(gsub_json),
        "csv_rows": len(rows),
        "csv_unique_videos": len({video_id for video_id, _ in rows}),
        "annotated_unique_video_qids": len(annotated),
        "exact_csv_qa_rows_with_intervals": len(split & annotated),
        "csv_qa_rows_without_intervals": len(split - annotated),
        "annotation_video_qids_not_in_csv": len(annotated - split),
        "alternative_interval_count": spans,
        "qa_rows_by_alternative_count": {str(k): alternatives_per_qa[k] for k in sorted(alternatives_per_qa)},
        "interval_validation_counts": {key: counts[key] for key in (
            "valid_positive_interval", "negative_start", "nonpositive_or_reversed",
            "end_after_video_duration", "malformed") if counts[key]},
    }


def summarize_packet_frame_overlap(records: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize QA-span overlap with packet sample timestamps; never emit rows."""
    item_count = interval_count = sampled_frame_count = inside_count = items_with_inside = 0
    for record in records:
        spans = record.get("gold_interval_seconds")
        timestamps = record.get("sampled_frame_relative_pts_seconds")
        if not isinstance(spans, list) or not isinstance(timestamps, list):
            raise ValueError("packet_interval_or_frame_times_missing")
        valid = []
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2
                    or not all(not isinstance(x, bool) and isinstance(x, (int, float))
                               and math.isfinite(x) for x in span)
                    or span[0] < 0 or span[0] >= span[1]):
                raise ValueError("packet_interval_invalid")
            valid.append(span)
        if not timestamps or any(isinstance(t, bool) or not isinstance(t, (int, float))
                                 or not math.isfinite(t) for t in timestamps):
            raise ValueError("packet_frame_timestamp_invalid")
        overlap = sum(any(start <= t <= end for start, end in valid)
                      for t in timestamps)
        item_count += 1
        interval_count += len(valid)
        sampled_frame_count += len(timestamps)
        inside_count += overlap
        items_with_inside += overlap > 0
    return {
        "review_items": item_count,
        "qa_reference_intervals": interval_count,
        "sampled_frame_timestamps": sampled_frame_count,
        "timestamps_inside_qa_interval_union": inside_count,
        "items_with_at_least_one_inside_timestamp": items_with_inside,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("NExT-GQA/datasets/nextgqa"))
    parser.add_argument("--include-current-review-packet-overlap", action="store_true",
                        help="add counts-only overlap for the exact three frozen review rows")
    args = parser.parse_args()
    summaries = {}
    for split in ("val", "test"):
        summaries[split] = summarize(args.dataset_dir / f"{split}.csv",
                                     args.dataset_dir / f"gsub_{split}.json")
        if summaries[split]["csv_qa_rows_without_intervals"]:
            raise SystemExit(f"{split}: missing interval annotations for CSV QA rows")
    if args.include_current_review_packet_overlap:
        from run_saved_claim_frame_permutation_attention_smoke import _read_packet
        from run_test_split_internal_bias_frame_mean_interaction import PACKETS
        repo = Path(__file__).resolve().parents[1]
        records = [_read_packet(repo, packet, index, expected_split="test")[0]
                   for packet, index, _ in PACKETS.values()]
        summaries["current_review_packet_overlap"] = summarize_packet_frame_overlap(records)
    print(json.dumps(summaries, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
