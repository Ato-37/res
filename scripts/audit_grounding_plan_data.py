#!/usr/bin/env python3
"""CPU-only, counts-only audit for the proposed NExT-GQA grounding study.

The output intentionally excludes question/answer text, video identifiers,
sample identifiers, annotation payloads, and local absolute paths.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_video_key(value):
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts and "." in parts[-1]:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def normalize_intervals(value):
    """Return valid [start,end] intervals and a count of malformed intervals."""
    if not isinstance(value, list):
        return [], 1
    valid, invalid = [], 0
    for span in value:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            invalid += 1
            continue
        start, end = (_finite_number(item) for item in span)
        if start is None or end is None or start < 0 or end < start:
            invalid += 1
            continue
        valid.append((start, end))
    return valid, invalid

def interval_value_shape(value):
    """Classify annotation container shape without exposing annotation values."""
    if not isinstance(value, list):
        return "not_list"
    if not value:
        return "empty_list"
    if len(value) == 2 and all(_finite_number(item) is not None for item in value):
        return "flat_pair"
    if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
        return "list_of_pairs"
    return "mixed_or_malformed_list"




def _get_gsub_intervals(gsub, video_id, qid):
    video_data = gsub.get(str(video_id), {})
    if not isinstance(video_data, dict):
        return None
    location = video_data.get("location", {})
    if not isinstance(location, dict):
        return None
    value = location.get(str(qid))
    if value is None:
        value = location.get(qid)
    return value

def audit_time_grid(csv_path, gsub_path, frame2time, video_map):
    """Summarize key coverage and gold-interval alignment to sampled timestamps."""
    gsub = json.loads(Path(gsub_path).read_text(encoding="utf-8"))
    csv_video_ids = set()
    rows_with_gsub = rows_with_grid = rows_with_all_maps = 0
    malformed_rows = past_duration_rows = 0
    interval_count = covered_interval_count = mapped_interval_count = 0
    midpoint_errors, endpoint_errors, positive_gaps, unique_counts = [], [], [], []
    non_monotonic_vectors = duplicate_adjacent_vectors = 0

    for video_id, value in frame2time.items():
        if not isinstance(value, list):
            continue
        times = [_finite_number(item) for item in value]
        if not times or any(item is None for item in times):
            continue
        unique_counts.append(len(set(times)))
        non_monotonic_vectors += any(b < a for a, b in zip(times, times[1:]))
        duplicate_adjacent_vectors += any(b == a for a, b in zip(times, times[1:]))
        positive_gaps.extend(b - a for a, b in zip(times, times[1:]) if b > a)

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            video_id, qid = str(row["video_id"]), str(row["qid"])
            csv_video_ids.add(video_id)
            has_gsub = video_id in gsub and _get_gsub_intervals(gsub, video_id, qid) is not None
            has_grid = video_id in frame2time and isinstance(frame2time.get(video_id), list)
            has_map = video_id in video_map
            rows_with_gsub += has_gsub
            rows_with_grid += has_grid
            rows_with_all_maps += has_gsub and has_grid and has_map
            if not has_gsub:
                continue

            raw = _get_gsub_intervals(gsub, video_id, qid)
            intervals, invalid = normalize_intervals(raw)
            if invalid:
                malformed_rows += 1
                continue
            video_data = gsub.get(video_id, {})
            duration = _finite_number(video_data.get("duration")) if isinstance(video_data, dict) else None
            if duration is not None and any(end > duration + 1e-6 for _, end in intervals):
                past_duration_rows += 1
                continue
            times = frame2time.get(video_id)
            if not isinstance(times, list):
                continue
            times = [_finite_number(item) for item in times]
            if not times or any(item is None for item in times):
                continue

            for start, end in intervals:
                interval_count += 1
                mapped_interval_count += 1
                covered_interval_count += any(start <= timestamp <= end for timestamp in times)
                midpoint = (start + end) / 2
                midpoint_errors.append(min(abs(timestamp - midpoint) for timestamp in times))
                endpoint_errors.extend((
                    min(abs(timestamp - start) for timestamp in times),
                    min(abs(timestamp - end) for timestamp in times),
                ))

    unique_gsub_video_ids = set(gsub)
    unique_grid_video_ids = set(frame2time)
    unique_video_map_ids = set(video_map)
    shared_all = csv_video_ids & unique_gsub_video_ids & unique_grid_video_ids & unique_video_map_ids
    return {
        "csv_unique_video_count": len(csv_video_ids),
        "gsub_video_entry_count": len(gsub),
        "frame2time_video_entry_count": len(frame2time),
        "video_map_entry_count": len(video_map),
        "csv_video_ids_with_gsub_key": len(csv_video_ids & unique_gsub_video_ids),
        "csv_video_ids_with_frame2time_key": len(csv_video_ids & unique_grid_video_ids),
        "csv_video_ids_with_video_map_key": len(csv_video_ids & unique_video_map_ids),
        "csv_video_ids_with_all_three_metadata_keys": len(shared_all),
        "rows_with_gsub_annotation": rows_with_gsub,
        "rows_with_frame2time_vector": rows_with_grid,
        "rows_with_all_metadata_keys": rows_with_all_maps,
        "valid_gold_interval_count": interval_count,
        "intervals_with_any_sampled_timestamp": covered_interval_count,
        "intervals_with_any_sampled_timestamp_fraction": (
            covered_interval_count / mapped_interval_count if mapped_interval_count else None
        ),
        "rows_with_malformed_interval_values": malformed_rows,
        "rows_with_interval_past_video_duration": past_duration_rows,
        "timestamp_vector_unique_point_count": _distribution(unique_counts),
        "positive_adjacent_grid_spacing_seconds": _distribution(positive_gaps),
        "nearest_interval_midpoint_seconds": _distribution(midpoint_errors),
        "nearest_interval_endpoint_seconds": _distribution(endpoint_errors),
        "non_monotonic_timestamp_vector_count": non_monotonic_vectors,
        "duplicate_adjacent_timestamp_vector_count": duplicate_adjacent_vectors,
        "grid_coverage_denominator": "duration-valid, structurally valid intervals with a finite frame2time vector",
    }


def _distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def audit_split(csv_path, gsub_path, video_map, excluded_vidor_ids):
    gsub = json.loads(Path(gsub_path).read_text(encoding="utf-8"))
    rows = []
    duplicate_rows = 0
    seen_questions = set()
    type_counts = Counter()
    annotation_shape_counts = Counter()
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "qid", "question", "answer"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{Path(csv_path).name} lacks required columns: {sorted(missing_columns)}")
        for row in reader:
            video_id, qid = str(row["video_id"]), str(row["qid"])
            key = (video_id, qid)
            duplicate_rows += int(key in seen_questions)
            seen_questions.add(key)
            raw = _get_gsub_intervals(gsub, video_id, qid)
            intervals, invalid = normalize_intervals(raw) if raw is not None else ([], 0)
            if raw is not None:
                annotation_shape_counts[interval_value_shape(raw)] += 1
            video_data = gsub.get(video_id, {})
            duration = _finite_number(video_data.get("duration")) if isinstance(video_data, dict) else None
            vidor = video_map.get(video_id)
            rows.append({
                "has_annotation": raw is not None,
                "intervals": intervals,
                "video_duration": duration,
                "ends_after_duration": bool(
                    duration is not None and any(end > duration + 1e-6 for _, end in intervals)
                ),
                "invalid": invalid,
                "has_video_map": vidor is not None,
                "video_key": canonical_video_key(vidor) if vidor is not None else None,
                "question_type": str(row.get("type") or "unknown"),
            })
            type_counts[str(row.get("type") or "unknown")] += 1

    matched = [row for row in rows if row["has_annotation"]]
    structurally_valid_rows = [
        row for row in matched if row["intervals"] and row["invalid"] == 0
    ]
    valid_rows = [row for row in structurally_valid_rows if not row["ends_after_duration"]]
    annotated_video_ids = {row["video_key"] for row in matched if row["video_key"]}
    excluded_rows = [row for row in rows
                     if row["video_key"] and row["video_key"] in excluded_vidor_ids]
    eligible_annotated_videos = {
        row["video_key"] for row in valid_rows
        if row["video_key"] and row["video_key"] not in excluded_vidor_ids
    }
    spans = [end - start for row in valid_rows for start, end in row["intervals"]]
    starts = [start for row in valid_rows for start, _ in row["intervals"]]
    ends = [end for row in valid_rows for _, end in row["intervals"]]
    interval_counts = [len(row["intervals"]) for row in valid_rows]
    video_durations = [row["video_duration"] for row in rows if row["video_duration"] is not None]
    return {
        "csv_rows": len(rows),
        "unique_video_questions": len(seen_questions),
        "duplicate_video_question_rows": duplicate_rows,
        "question_type_counts": dict(sorted(type_counts.items())),
        "gsub_video_entries": len(gsub),
        "rows_with_gsub_key": len(matched),
        "rows_without_gsub_key": len(rows) - len(matched),
        "rows_with_valid_intervals": len(valid_rows),
        "annotation_value_shape_counts": dict(sorted(annotation_shape_counts.items())),
        "rows_with_structurally_valid_intervals": len(structurally_valid_rows),
        "rows_with_any_malformed_interval": sum(bool(row["invalid"]) for row in matched),
        "rows_with_interval_ending_after_video_duration": sum(
            row["ends_after_duration"] for row in matched
        ),
        "video_duration_seconds": _distribution(video_durations),
        "interval_count_total": sum(len(row["intervals"]) for row in valid_rows),
        "intervals_per_valid_question": _distribution(interval_counts),
        "interval_duration_seconds": _distribution(spans),
        "interval_start_seconds": _distribution(starts),
        "interval_end_seconds": _distribution(ends),
        "rows_with_video_map": sum(row["has_video_map"] for row in rows),
        "unique_annotated_video_count": len(annotated_video_ids),
        "rows_on_previously_used_videos": len(excluded_rows),
        "unique_eligible_annotated_videos_after_exclusion": len(eligible_annotated_videos),
        "interval_unit": "seconds, as documented in the local NExT-GQA dataset README",
    }


def _shape_summary(value):
    kinds = Counter()
    numeric_finite = 0
    numeric_vectors = []

    def visit(node, depth=0):
        nonlocal numeric_finite
        kinds[type(node).__name__] += 1
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            numeric_finite += int(math.isfinite(float(node)))
        elif isinstance(node, dict):
            for child in node.values():
                visit(child, depth + 1)
        elif isinstance(node, list):
            values = [_finite_number(item) for item in node]
            if values and all(item is not None for item in values):
                numeric_vectors.append(values)
            for child in node:
                visit(child, depth + 1)

    visit(value)
    return {
        "top_level_type": type(value).__name__,
        "top_level_entries": len(value) if isinstance(value, (dict, list)) else None,
        "nested_value_type_counts": dict(sorted(kinds.items())),
        "finite_numeric_value_count": numeric_finite,
        "numeric_time_vector_count": len(numeric_vectors),
        "time_vector_length": _distribution([len(vector) for vector in numeric_vectors]),
        "non_monotonic_time_vector_count": sum(
            any(right < left for left, right in zip(vector, vector[1:]))
            for vector in numeric_vectors
        ),
        "duplicate_adjacent_time_vector_count": sum(
            any(right == left for left, right in zip(vector, vector[1:]))
            for vector in numeric_vectors
        ),
        "timestamp_value_range": _distribution([item for vector in numeric_vectors for item in vector]),
    }


def audit(args):
    source_paths = {
        "val_csv": Path(args.val_csv),
        "gsub_val_json": Path(args.gsub_val_json),
        "test_csv": Path(args.test_csv),
        "gsub_test_json": Path(args.gsub_test_json),
        "video_map_json": Path(args.video_map),
        "frame2time_val_json": Path(args.frame2time_val_json),
        "frame2time_test_json": Path(args.frame2time_test_json),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path.name}")
    video_map = json.loads(source_paths["video_map_json"].read_text(encoding="utf-8"))
    if not isinstance(video_map, dict):
        raise ValueError("video map must be a JSON object")

    excluded_ids = set()
    manifests = []
    for manifest_path in args.exclude_samples:
        path = Path(manifest_path)
        if not path.is_file():
            continue
        samples = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            continue
        found = 0
        for sample in samples:
            if isinstance(sample, dict) and sample.get("video"):
                key = canonical_video_key(sample["video"])
                if key:
                    excluded_ids.add(key)
                    found += 1
        manifests.append({"name": path.name, "sample_records_with_video_path": found})

    frame2time_objects = {
        "validation": json.loads(source_paths["frame2time_val_json"].read_text(encoding="utf-8")),
        "test": json.loads(source_paths["frame2time_test_json"].read_text(encoding="utf-8")),
    }
    splits = {}
    for name, csv_key, gsub_key in (
        ("validation", "val_csv", "gsub_val_json"),
        ("test", "test_csv", "gsub_test_json"),
    ):
        splits[name] = audit_split(
            source_paths[csv_key], source_paths[gsub_key], video_map, excluded_ids
        )
    temporal_grid = {}
    for name, csv_key, gsub_key in (
        ("validation", "val_csv", "gsub_val_json"),
        ("test", "test_csv", "gsub_test_json"),
    ):
        temporal_grid[name] = audit_time_grid(
            source_paths[csv_key], source_paths[gsub_key], frame2time_objects[name], video_map
        )

    mappings = {}
    for name in ("frame2time_val_json", "frame2time_test_json"):
        split = "validation" if "val" in name else "test"
        obj = frame2time_objects[split]
        mappings[name.removeprefix("frame2time_").removesuffix("_json")] = _shape_summary(obj)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "counts_only_grounding_data_feasibility",
        "inputs": {
            key: {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in source_paths.items()
        },
        "excluded_manifest_count": len(manifests),
        "excluded_sample_record_count": sum(item["sample_records_with_video_path"] for item in manifests),
        "unique_previously_used_video_count": len(excluded_ids),
        "excluded_manifests": manifests,
        "video_map_entries": len(video_map),
        "splits": splits,
        "frame2time_shape": mappings,
        "temporal_grid_alignment": temporal_grid,
        "privacy": {
            "contains_raw_question_or_answer_text": False,
            "contains_video_or_question_identifiers": False,
            "contains_absolute_paths": False,
        },
        "limitations": [
            "Counts and interval-shape checks do not establish that an interval semantically supports a generated claim.",
            "Local dataset docs define intervals and frame2time values in seconds; correspondence to Qwen visual chunks remains unverified.",
            "No video decoding, model inference, intervention, or faithfulness test was run.",
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--gsub-val-json", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--gsub-test-json", required=True)
    parser.add_argument("--video-map", required=True)
    parser.add_argument("--frame2time-val-json", required=True)
    parser.add_argument("--frame2time-test-json", required=True)
    parser.add_argument("--exclude-samples", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    print("CPU grounding data feasibility audit")
    for split, values in result["splits"].items():
        print(f"{split}: rows={values['csv_rows']}, interval_coverage="
              f"{values['rows_with_valid_intervals']}/{values['csv_rows']}, "
              f"eligible_unique_videos={values['unique_eligible_annotated_videos_after_exclusion']}, "
              f"malformed_interval_rows={values['rows_with_any_malformed_interval']}")
    print(f"excluded unique videos={result['unique_previously_used_video_count']}")
    print(f"wrote counts-only summary: {Path(args.output).name}")


if __name__ == "__main__":
    main()
