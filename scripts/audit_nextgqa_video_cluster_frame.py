#!/usr/bin/env python3
"""Aggregate NExT-GQA split coverage without emitting IDs or QA text.

This metadata-only audit measures the candidate video-ID frame and its QA
clustering. It does not read media bytes or use/emit question or answer text.
The CSV parser transiently parses every column, but only video_id/qid are
retained and all other row fields are discarded immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("metadata_json_invalid") from None


def audit_split(test_csv: Path, gsub_json: Path, video_map_json: Path) -> dict:
    try:
        qa_by_video: Counter[str] = Counter()
        pairs: set[tuple[str, str]] = set()
        row_count = duplicate_pairs = 0
        with test_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not {"video_id", "qid"}.issubset(reader.fieldnames)):
                raise ValueError("test_csv_required_columns_missing")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("test_csv_row_width_invalid")
                video_id, qid = row.get("video_id"), row.get("qid")
                if (not isinstance(video_id, str) or not video_id.strip()
                        or not isinstance(qid, str) or not qid.strip()):
                    raise ValueError("test_csv_identity_fields_invalid")
                pair = (video_id, qid)
                if pair in pairs:
                    duplicate_pairs += 1
                pairs.add(pair)
                qa_by_video[video_id] += 1
                row_count += 1
                del row
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("test_csv_invalid") from None
    gsub = _load_json(gsub_json)
    video_map = _load_json(video_map_json)
    if (not isinstance(gsub, dict) or not isinstance(video_map, dict)
            or any(not isinstance(k, str) for k in gsub)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in video_map.items())):
        raise ValueError("metadata_mapping_schema_invalid")
    for video_id, payload in gsub.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("location"), dict):
            raise ValueError("gsub_location_schema_invalid")
    test_ids = set(qa_by_video)
    mapped_paths = [video_map[video_id] for video_id in test_ids if video_id in video_map]
    gsub_qids = {video_id: set(payload["location"]) for video_id, payload in gsub.items()}
    qa_pairs = pairs
    test_qids_by_video: dict[str, set[str]] = {}
    for video_id, qid in qa_pairs:
        test_qids_by_video.setdefault(video_id, set()).add(qid)
    interval_count = 0
    matched_interval_count = 0
    for payload in gsub.values():
        for qid, spans in payload["location"].items():
            if not isinstance(spans, list):
                raise ValueError("gsub_intervals_schema_invalid")
            interval_count += len(spans)
    for video_id, qids in test_qids_by_video.items():
        for qid in qids & gsub_qids.get(video_id, set()):
            matched_interval_count += len(gsub[video_id]["location"][qid])
    histogram = Counter(qa_by_video.values())
    sorted_counts = sorted(qa_by_video.values())
    quantiles = {}
    for label, q in (("min", 0.0), ("p25_nearest_rank", 0.25),
                     ("median_nearest_rank", 0.5), ("p75_nearest_rank", 0.75),
                     ("p90_nearest_rank", 0.9), ("max", 1.0)):
        index = 0 if q == 0 else (len(sorted_counts) - 1 if q == 1
                                 else math.ceil(len(sorted_counts) * q) - 1)
        quantiles[label] = sorted_counts[index] if sorted_counts else 0
    return {
        "schema_version": 1,
        "purpose": "nextgqa_test_video_id_cluster_frame_aggregate",
        "scope": "metadata_only_no_media_bytes_or_qa_text_use_or_emission",
        "qa_rows": row_count,
        "unique_test_video_ids": len(test_ids),
        "unique_test_video_question_pairs": len(qa_pairs),
        "duplicate_video_question_pairs": duplicate_pairs,
        "gsub_video_ids": len(gsub),
        "test_video_ids_with_gsub": len(test_ids & set(gsub)),
        "gsub_video_ids_outside_test": len(set(gsub) - test_ids),
        "test_video_ids_with_video_map": len(test_ids & set(video_map)),
        "test_video_ids_missing_video_map": len(test_ids - set(video_map)),
        "distinct_mapped_video_paths_for_test_ids": len(set(mapped_paths)),
        "gsub_question_ids_for_test_pairs": sum(
            1 for video_id, qid in qa_pairs if qid in gsub_qids.get(video_id, set())),
        "gsub_intervals_all_ids": interval_count,
        "gsub_intervals_for_test_pairs": matched_interval_count,
        "qa_rows_per_video_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "qa_rows_per_video_quantiles": quantiles,
        "candidate_video_id_units_for_unique_video_sampling": len(test_ids),
        "source_cluster_independence_verified": False,
        "review_eligibility_verified": False,
        "limitations": [
            "video ID and mapped path uniqueness do not prove independent source content",
            "media-byte duplicates and cross-video source/event clusters were not audited",
            "question/answer columns are parsed transiently by the CSV reader, then discarded; they are not used or emitted",
            "claim support, prior exposure, reviewer completion, and score eligibility were not assessed",
            "counts are availability diagnostics, not an eligible or randomized cohort",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--gsub-json", required=True, type=Path)
    parser.add_argument("--video-map-json", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_split(args.test_csv, args.gsub_json, args.video_map_json),
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
