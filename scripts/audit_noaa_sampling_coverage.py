"""Counts-only geometry audit of fixed NOAA QA windows under uniform sampling.

Compares whole-video uniform 4/8-frame samples with independent uniform
4/8-frame samples of each existing question window. It reads only window bounds
from the local sample manifests; it performs no model inference, annotation,
question/answer output, or faithfulness evaluation. Counts describe geometric
sample/window intersection only, not claim fit, evidence validity, or efficacy.

Default output is a short human-readable summary. Use ``--json`` for the
publish-safe derived summary, or ``--output results/<unique-name>.json`` to
write that JSON without overwriting an existing file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
import subprocess
from typing import Any

import cv2

from audit_noaa_qa_gate import GROUPS
from video_time_mapping import uniform_frame_indices


REQUESTED_SAMPLE_COUNTS = (4, 8)
ENDPOINT_PAIR_SIZE_FRAMES = 2
EXPECTED_GROUPS = 3
EXPECTED_WINDOWS = 6


class SourceFailure(Exception):
    """A source could not be safely used for geometric counting."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def read_video_geometry(path: Path) -> tuple[int, float, float]:
    """Return OpenCV frame count, nominal FPS, and frame_count/FPS duration."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SourceFailure("video_open_failed")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0:
        raise SourceFailure("invalid_frame_count")
    if not math.isfinite(fps) or fps <= 0:
        raise SourceFailure("invalid_nominal_fps")
    return frame_count, fps, frame_count / fps


def _window_frame_bounds(
    start_seconds: Any,
    end_seconds: Any,
    *,
    fps: float,
    frame_count: int,
    duration_seconds: float,
) -> tuple[int, int] | str:
    """Return bounded inclusive frame indices, or a privacy-safe error class."""
    if not _finite_real(start_seconds) or not _finite_real(end_seconds):
        return "nonfinite_window_bounds"
    start = float(start_seconds)
    end = float(end_seconds)
    if start < 0 or end <= start or end > duration_seconds:
        return "window_bounds_outside_source"
    first = max(0, math.ceil(start * fps))
    last = min(frame_count - 1, math.floor(end * fps))
    if first > last:
        return "window_contains_no_frame_presentation_timestamp"
    return first, last


def _selected_frames_per_window(
    windows: list[tuple[int, int] | None],
    *,
    frame_count: int,
    requested_count: int,
    mode: str,
) -> list[tuple[int, ...]]:
    if mode == "whole_video":
        selected = tuple(uniform_frame_indices(frame_count, requested_count))
        return [selected for _ in windows]
    if mode == "per_question_window":
        samples = []
        for bounds in windows:
            if bounds is None:
                samples.append(())
                continue
            first, last = bounds
            available = last - first + 1
            # For a one-frame interval there is no constant stride to infer;
            # use the only available frame and report that actual count.
            if available == 1:
                samples.append((first,))
            else:
                samples.append(tuple(
                    first + index
                    for index in uniform_frame_indices(available, requested_count)
                ))
        return samples
    raise ValueError(f"Unknown sampling mode: {mode}")


def _count_summary(sample_counts: list[int]) -> dict[str, int]:
    return {
        "frames_inside_windows": sum(sample_counts),
        "windows_with_zero_sampled_frames": sum(count == 0 for count in sample_counts),
        "windows_with_fewer_than_two_sampled_frames": sum(count < 2 for count in sample_counts),
        "windows_with_at_least_two_sampled_frames": sum(count >= 2 for count in sample_counts),
    }


def _point_counts_per_window(
    sampled_indices_per_window: list[tuple[int, ...]],
    raw_intervals: list[tuple[float, float] | None],
    *,
    fps: float,
) -> list[int]:
    counts = []
    for sampled_indices, interval in zip(sampled_indices_per_window, raw_intervals):
        if interval is None:
            counts.append(0)
            continue
        start_seconds, end_seconds = interval
        counts.append(sum(
            start_seconds <= index / fps <= end_seconds
            for index in sampled_indices
        ))
    return counts


def _endpoint_span_summary(
    sampled_indices_per_window: list[tuple[int, ...]],
    raw_intervals: list[tuple[float, float] | None],
    *,
    fps: float,
) -> dict[str, Any]:
    """Compare sample points with paired endpoint spans; neither implies evidence."""
    point_windows = 0
    span_windows = 0
    span_pair_intersections = 0
    span_only_windows = 0
    nearest_gaps: list[float] = []
    for sampled_indices, interval in zip(sampled_indices_per_window, raw_intervals):
        if interval is None:
            continue
        start_seconds, end_seconds = interval
        point_times = tuple(index / fps for index in sampled_indices)
        points_inside = sum(start_seconds <= time <= end_seconds for time in point_times)
        point_windows += points_inside > 0
        if point_times:
            nearest_gaps.append(min(
                max(start_seconds - time, time - end_seconds, 0.0)
                for time in point_times
            ))
        has_span_overlap = False
        for index in range(0, len(point_times) - 1, ENDPOINT_PAIR_SIZE_FRAMES):
            span_start, span_end = point_times[index], point_times[index + 1]
            if max(span_start, start_seconds) < min(span_end, end_seconds):
                span_pair_intersections += 1
                has_span_overlap = True
        span_windows += has_span_overlap
        span_only_windows += has_span_overlap and points_inside == 0
    return {
        "windows_with_any_sampled_point": int(point_windows),
        "windows_with_any_positive_paired_endpoint_span_overlap": int(span_windows),
        "positive_paired_endpoint_span_intersections": span_pair_intersections,
        "span_intersects_but_zero_actual_sample_points_windows": int(span_only_windows),
        "windows_with_nearest_sample_gap": len(nearest_gaps),
        "max_nearest_sample_gap_seconds": max(nearest_gaps) if nearest_gaps else None,
    }


def audit_window_bounds(
    raw_windows: list[tuple[Any, Any] | None],
    *,
    frame_count: int,
    fps: float,
    duration_seconds: float,
) -> dict[str, Any]:
    """Compare fixed whole-clip and per-window policies for one clip."""
    valid_bounds: list[tuple[int, int] | None] = []
    valid_intervals: list[tuple[float, float] | None] = []
    finite_failures = 0
    bounds_failures = 0
    no_frame_presentation_timestamp = 0
    for window in raw_windows:
        if window is None or len(window) != 2:
            result: tuple[int, int] | str = "nonfinite_window_bounds"
        else:
            result = _window_frame_bounds(
                window[0], window[1], fps=fps, frame_count=frame_count,
                duration_seconds=duration_seconds,
            )
        if result == "nonfinite_window_bounds":
            finite_failures += 1
            valid_bounds.append(None)
            valid_intervals.append(None)
        elif result == "window_bounds_outside_source":
            bounds_failures += 1
            valid_bounds.append(None)
            valid_intervals.append(None)
        elif result == "window_contains_no_frame_presentation_timestamp":
            no_frame_presentation_timestamp += 1
            valid_bounds.append(None)
            valid_intervals.append((float(window[0]), float(window[1])))
        else:
            valid_bounds.append(result)
            valid_intervals.append((float(window[0]), float(window[1])))

    policy_counts: dict[str, dict[str, dict[str, Any]]] = {}
    for requested_count in REQUESTED_SAMPLE_COUNTS:
        policy_counts[str(requested_count)] = {}
        for mode in ("whole_video", "per_question_window"):
            sampled_frames = _selected_frames_per_window(
                valid_bounds,
                frame_count=frame_count,
                requested_count=requested_count,
                mode=mode,
            )
            summary = _count_summary(_point_counts_per_window(
                sampled_frames, valid_intervals, fps=fps,
            ))
            summary.update(_endpoint_span_summary(sampled_frames, valid_intervals, fps=fps))
            policy_counts[str(requested_count)][mode] = summary

    return {
        "windows_requested": len(raw_windows),
        "windows_valid": sum(bounds is not None for bounds in valid_bounds),
        "nonfinite_window_bounds_failures": finite_failures,
        "window_bounds_failures": bounds_failures,
        "windows_without_source_frame_presentation_timestamps": no_frame_presentation_timestamp,
        "sampling": policy_counts,
    }


def _read_sample_windows(path: Path) -> list[tuple[Any, Any] | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceFailure("sample_manifest_missing") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceFailure("sample_manifest_invalid") from exc
    records = value.get("samples") if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise SourceFailure("sample_manifest_invalid")
    windows: list[tuple[Any, Any] | None] = []
    for record in records:
        if not isinstance(record, dict):
            windows.append(None)
        else:
            windows.append((record.get("window_start_seconds"), record.get("window_end_seconds")))
    return windows


def run_audit(data_root: Path) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    failure_reasons: dict[str, int] = {}
    total_windows = 0
    total_valid_windows = 0
    total_finite_failures = 0
    total_bounds_failures = 0
    total_no_frame_presentation_timestamps = 0
    combined = {
        str(count): {
        mode: {
                "frames_inside_windows": 0,
                "windows_with_zero_sampled_frames": 0,
                "windows_with_fewer_than_two_sampled_frames": 0,
                "windows_with_at_least_two_sampled_frames": 0,
                "windows_with_any_sampled_point": 0,
                "windows_with_any_positive_paired_endpoint_span_overlap": 0,
                "positive_paired_endpoint_span_intersections": 0,
                "span_intersects_but_zero_actual_sample_points_windows": 0,
                "windows_with_nearest_sample_gap": 0,
                "max_nearest_sample_gap_seconds": None,
            }
            for mode in ("whole_video", "per_question_window")
        }
        for count in REQUESTED_SAMPLE_COUNTS
    }

    for group in GROUPS:
        group_id = group["group_id"]
        group_root = data_root / group["directory"]
        media_path = group_root / "media" / "source.mp4"
        samples_path = group_root / "private" / "samples.json"
        record: dict[str, Any] = {"group_id": group_id, "failures": []}
        try:
            media_hash = sha256_file(media_path)
            sample_hash = sha256_file(samples_path)
            record["media_sha256"] = media_hash
            record["samples_sha256"] = sample_hash
            if media_hash != group["media_sha256"]:
                raise SourceFailure("media_hash_mismatch")
            frame_count, fps, duration = read_video_geometry(media_path)
            raw_windows = _read_sample_windows(samples_path)
            audit = audit_window_bounds(
                raw_windows, frame_count=frame_count, fps=fps,
                duration_seconds=duration,
            )
            record.update({
                "source_frame_count": frame_count,
                "nominal_fps": fps,
                "source_duration_seconds": duration,
                "windows": audit,
            })
            total_windows += audit["windows_requested"]
            total_valid_windows += audit["windows_valid"]
            total_finite_failures += audit["nonfinite_window_bounds_failures"]
            total_bounds_failures += audit["window_bounds_failures"]
            total_no_frame_presentation_timestamps += audit["windows_without_source_frame_presentation_timestamps"]
            for count, by_mode in audit["sampling"].items():
                for mode, counts in by_mode.items():
                    for key, value in counts.items():
                        if key == "max_nearest_sample_gap_seconds":
                            current = combined[count][mode][key]
                            if value is not None and (current is None or value > current):
                                combined[count][mode][key] = value
                        else:
                            combined[count][mode][key] += value
            for field, reason in (
                ("nonfinite_window_bounds_failures", "nonfinite_window_bounds"),
                ("window_bounds_failures", "window_bounds_outside_source"),
                ("windows_without_source_frame_presentation_timestamps", "window_without_frame_presentation_timestamp"),
            ):
                n = audit[field]
                if n:
                    record["failures"].append(reason)
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + n
        except (OSError, SourceFailure) as exc:
            reason = exc.reason if isinstance(exc, SourceFailure) else "source_read_error"
            record["failures"].append(reason)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        groups.append(record)

    valid_group_count = sum(not record["failures"] for record in groups)
    if len(GROUPS) != EXPECTED_GROUPS:
        failure_reasons["canonical_group_count_mismatch"] = 1
    if total_windows != EXPECTED_WINDOWS:
        failure_reasons["canonical_window_count_mismatch"] = 1
    complete = (
        len(GROUPS) == EXPECTED_GROUPS
        and valid_group_count == EXPECTED_GROUPS
        and total_windows == EXPECTED_WINDOWS
        and total_valid_windows == EXPECTED_WINDOWS
        and not failure_reasons
    )
    source_root = Path(__file__).resolve().parent
    head = subprocess.run(
        ["git", "-C", str(source_root.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "purpose": "noaa_sampling_coverage_geometry_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "opencv_version": cv2.__version__,
        "status": "complete" if complete else "partial_or_failed",
        "scope_limit": (
            "Counts are geometric sample/window intersections only; they do not establish "
            "claim fit, evidence validity, QA accuracy, efficacy, or faithfulness."
        ),
        "human_qa_attestation_status": "pending; this audit does not satisfy or alter that gate",
        "requested_sample_counts": list(REQUESTED_SAMPLE_COUNTS),
        "endpoint_span_pair_size_frames": ENDPOINT_PAIR_SIZE_FRAMES,
        "metadata_method": "OpenCV VideoCapture CAP_PROP_FRAME_COUNT and CAP_PROP_FPS; nominal constant-rate geometry",
        "source_groups_requested": EXPECTED_GROUPS,
        "source_groups_valid": valid_group_count,
        "windows_expected": EXPECTED_WINDOWS,
        "windows_requested": total_windows,
        "windows_valid": total_valid_windows,
        "nonfinite_window_bounds_failures": total_finite_failures,
        "window_bounds_failures": total_bounds_failures,
        "windows_without_source_frame_presentation_timestamps": total_no_frame_presentation_timestamps,
        "sampling_counts": combined,
        "failure_reasons": failure_reasons,
        "groups": groups,
        "code_sha256": {
            "audit_script": sha256_file(Path(__file__)),
            "frame_index_helper": sha256_file(source_root / "video_time_mapping.py"),
            "canonical_group_manifest": sha256_file(source_root / "audit_noaa_qa_gate.py"),
        },
        "privacy": "No question text, choices, answers, qids, local paths, or people are emitted.",
    }


def _human_summary(result: dict[str, Any]) -> str:
    full4 = result["sampling_counts"]["4"]["whole_video"]
    full8 = result["sampling_counts"]["8"]["whole_video"]
    gap4 = full4["max_nearest_sample_gap_seconds"]
    gap8 = full8["max_nearest_sample_gap_seconds"]
    format_gap = lambda value: "n/a" if value is None else f"{value:.3f}"
    return (
        f"NOAA sampling geometry: {result['status']}; "
        f"groups={result['source_groups_valid']}/{result['source_groups_requested']}, "
        f"windows={result['windows_valid']}/{result['windows_expected']}; "
        f"whole-video point-hit windows 4/8="
        f"{full4['windows_with_any_sampled_point']}/{full8['windows_with_any_sampled_point']}; "
        f"paired-endpoint-span overlaps 4/8="
        f"{full4['windows_with_any_positive_paired_endpoint_span_overlap']}/"
        f"{full8['windows_with_any_positive_paired_endpoint_span_overlap']}; "
        f"span-only windows 4/8="
        f"{full4['span_intersects_but_zero_actual_sample_points_windows']}/"
        f"{full8['span_intersects_but_zero_actual_sample_points_windows']}; "
        f"max nearest-point gap seconds 4/8="
        f"{format_gap(gap4)}/{format_gap(gap8)}; "
        f"failures={sum(result['failure_reasons'].values())}. "
        "Geometric counts only; endpoint spans do not establish temporal coverage or evidence."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "noaa_qa_pilot_20261011",
        help="Local NOAA sample/media root (default: datasets/noaa_qa_pilot_20261011)",
    )
    parser.add_argument("--json", action="store_true", help="Print publish-safe JSON instead of a short summary")
    parser.add_argument("--output", type=Path, help="Write JSON to a new path; existing files are never overwritten")
    args = parser.parse_args(argv)

    result = run_audit(args.root)
    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        repo = Path(__file__).resolve().parents[1]
        output = (repo / args.output).resolve()
        if output.parent != (repo / "results").resolve() or output.suffix.lower() != ".json":
            parser.error("--output must be a .json file directly under repository results/")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        except FileExistsError:
            parser.error("--output already exists; choose a new unique path")
    print(serialized if args.json else _human_summary(result))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
