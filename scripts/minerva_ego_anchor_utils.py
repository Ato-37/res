"""Deterministic helpers for Minerva-Ego object/time-linked rationale pilots.

The helpers bind one-second rationale timestamps, HD-EPIC MP4 frame numbers,
and Qwen temporal groups. They establish metadata alignment only; they do not
validate video content, object visibility, or explanation truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from bisect import bisect_left
import math
import re


_TIMESTAMP_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_TIMESTAMP_RANGE_RE = re.compile(
    r"(?<![\d:])(\d{1,2}):(\d{2})\s*(?:-|–|—|\bto\b)\s*"
    r"(\d{1,2}):(\d{2})(?![\d:])", re.IGNORECASE)
_OBJECT_ID_RE = re.compile(r"(?<![0-9a-fA-F]){oid}(?![0-9a-fA-F])")


def parse_timestamp_bins(text: str) -> list[tuple[int, int]]:
    """Return distinct [start,end) one-second bins for mm:ss anchors."""
    if not isinstance(text, str):
        raise ValueError("timestamp_text_must_be_string")
    seconds = sorted({int(mm) * 60 + int(ss)
                      for mm, ss in _TIMESTAMP_RE.findall(text)
                      if int(ss) < 60})
    return [(second, second + 1) for second in seconds]


def parse_rationale_time_intervals(text: str) -> list[tuple[int, int]]:
    """Parse explicit timestamp ranges as spans; retain bins for standalone times.

    Displayed endpoints have one-second precision, so a range ``01:17-01:46``
    maps to the half-open interval [77, 107). This makes interior annotated
    frames eligible without widening the annotation track tolerance.
    """
    if not isinstance(text, str):
        raise ValueError("timestamp_text_must_be_string")
    ranges: list[tuple[int, int]] = []
    covered: list[tuple[int, int]] = []
    for match in _TIMESTAMP_RANGE_RE.finditer(text):
        # Consume paired endpoints even when invalid so malformed/reversed
        # ranges cannot silently degrade into two standalone timestamp bins.
        covered.append(match.span())
        start_min, start_sec, end_min, end_sec = map(int, match.groups())
        if start_sec >= 60 or end_sec >= 60:
            continue
        start = start_min * 60 + start_sec
        end = end_min * 60 + end_sec
        if end < start:
            continue
        ranges.append((start, end + 1))
    standalone = [second for match in _TIMESTAMP_RE.finditer(text)
                  if not any(left <= match.start() and match.end() <= right
                             for left, right in covered)
                  for mm, ss in [match.groups()]
                  if int(ss) < 60
                  for second in [int(mm) * 60 + int(ss)]]
    return sorted(set(ranges + [(second, second + 1) for second in standalone]))


def timestamp_bins_overlap_segment(bins: Sequence[Sequence[float]],
                                   segment: Sequence[float]) -> bool:
    """Whether a rounded timestamp bin overlaps a closed track interval."""
    if (not isinstance(segment, Sequence) or isinstance(segment, (str, bytes))
            or len(segment) != 2):
        raise ValueError("track_segment_invalid")
    start, end = segment
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or start < 0 or end < start):
        raise ValueError("track_segment_invalid")
    for row in bins:
        if (not isinstance(row, Sequence) or isinstance(row, (str, bytes))
                or len(row) != 2):
            raise ValueError("timestamp_bin_invalid")
        left, right = row
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not math.isfinite(float(left)) or not math.isfinite(float(right))
                or left < 0 or right <= left):
            raise ValueError("timestamp_bin_invalid")
        if float(left) < float(end) and float(right) > float(start):
            return True
    return False


def supported_rationale_lines(*, reasoning: str,
                               object_tracks: Mapping[str, Sequence[Sequence[float]]]
                               ) -> list[str]:
    """Keep object-linked lines whose rounded timestamp bin overlaps its track."""
    if (not isinstance(reasoning, str) or not isinstance(object_tracks, Mapping)
            or not object_tracks):
        raise ValueError("rationale_anchor_input_invalid")
    lines = []
    for line in reasoning.splitlines():
        matched = [oid for oid in object_tracks
                   if re.search(_OBJECT_ID_RE.pattern.format(oid=re.escape(oid)), line)]
        if not matched:
            continue
        intervals = parse_rationale_time_intervals(line)
        if intervals and any(timestamp_bins_overlap_segment(intervals, segment)
                        for oid in matched for segment in object_tracks[oid]):
            lines.append(line.strip())
    return lines


def select_rationale_mask_anchor(*, reasoning: str, object_id: str,
                                 tracks: Sequence[Mapping[str, object]],
                                 mask_frame_numbers: Mapping[str, int],
                                 mp4_frame_times_seconds: Sequence[float],
                                 track_boundary_tolerance_seconds: float = 0.05) -> dict:
    """Choose an annotated mask frame inside the first supported rationale bin."""
    segments = [track.get("time_segment") for track in tracks
                if isinstance(track, Mapping) and track.get("time_segment")]
    lines = supported_rationale_lines(
        reasoning=reasoning, object_tracks={object_id: segments})
    if not lines:
        raise ValueError("supported_rationale_line_missing")
    times = [_as_seconds(value) for value in mp4_frame_times_seconds]
    if (any(right <= left for left, right in zip(times, times[1:]))
            or not isinstance(mask_frame_numbers, Mapping)
            or isinstance(track_boundary_tolerance_seconds, bool)
            or not isinstance(track_boundary_tolerance_seconds, (int, float))
            or not math.isfinite(float(track_boundary_tolerance_seconds))
            or track_boundary_tolerance_seconds < 0
            or track_boundary_tolerance_seconds > 0.1):
        raise ValueError("rationale_mask_anchor_input_invalid")
    line = lines[0]
    for lo, hi in parse_rationale_time_intervals(line):
        candidates = []
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            segment = track.get("time_segment")
            if not isinstance(segment, Sequence) or len(segment) != 2:
                continue
            if not timestamp_bins_overlap_segment([(lo, hi)], segment):
                continue
            for mask_id in track.get("masks", []):
                frame_number = mask_frame_numbers.get(mask_id)
                if (type(frame_number) is not int
                        or not 0 <= frame_number < len(times)):
                    continue
                time_seconds = times[frame_number]
                if (lo <= time_seconds < hi
                        and segment[0] - track_boundary_tolerance_seconds
                            <= time_seconds <= segment[1] + track_boundary_tolerance_seconds):
                    candidates.append((abs(time_seconds - (lo + hi) / 2),
                                       frame_number, time_seconds))
        if candidates:
            error, frame_number, time_seconds = min(candidates)
            return {"rationale_line": line,
                    "timestamp_support_interval_seconds": [lo, hi],
                    "mask_frame_number": frame_number,
                    "mask_time_seconds": time_seconds,
                    "timestamp_support_center_error_seconds": error,
                    "track_boundary_tolerance_seconds": track_boundary_tolerance_seconds,
                    "supported_rationale_line_count": len(lines)}
    raise ValueError("rationale_mask_frame_missing_in_timestamp_bin")


def _as_seconds(value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError("presentation_time_invalid")
    return float(value)


def build_uniform_sample_indices(*, total_frames: int, sample_count: int,
                                 anchor_frames: Sequence[int] = ()) -> list[int]:
    """Build uniform indices while replacing nearest slots with exact anchors."""
    if (type(total_frames) is not int or total_frames < 1
            or type(sample_count) is not int or not 1 <= sample_count <= total_frames
            or not isinstance(anchor_frames, Sequence)
            or isinstance(anchor_frames, (str, bytes))
            or len(set(anchor_frames)) != len(anchor_frames)
            or any(type(index) is not int or not 0 <= index < total_frames
                   for index in anchor_frames)
            or len(anchor_frames) > sample_count):
        raise ValueError("frame_sampling_geometry_invalid")
    ranks = [round(i * (total_frames - 1) / (sample_count - 1))
             for i in range(sample_count)] if sample_count > 1 else [0]
    for anchor in anchor_frames:
        if anchor in ranks:
            continue
        slot = min(range(len(ranks)), key=lambda i: (abs(ranks[i] - anchor), i))
        ranks[slot] = anchor
    ranks = sorted(ranks)
    if len(set(ranks)) != sample_count:
        raise ValueError("anchored_frame_indices_not_unique")
    return ranks


def build_anchor_centered_sample_indices(*, mp4_frame_times_seconds: Sequence[float],
                                         anchor_frame_number: int,
                                         sample_count: int = 64,
                                         sample_fps: float = 2.0) -> dict:
    """Build a fixed-rate local clip centered on one annotation-backed frame."""
    if (not isinstance(mp4_frame_times_seconds, Sequence)
            or isinstance(mp4_frame_times_seconds, (str, bytes))
            or len(mp4_frame_times_seconds) < sample_count
            or type(anchor_frame_number) is not int
            or not 0 <= anchor_frame_number < len(mp4_frame_times_seconds)
            or type(sample_count) is not int or sample_count < 2
            or isinstance(sample_fps, bool)
            or not isinstance(sample_fps, (int, float))
            or not math.isfinite(float(sample_fps)) or sample_fps <= 0):
        raise ValueError("anchor_centered_sampling_input_invalid")
    times = [_as_seconds(value) for value in mp4_frame_times_seconds]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("mp4_frame_times_not_strictly_increasing")
    step = 1.0 / float(sample_fps)
    clip_span = (sample_count - 1) * step
    duration = times[-1] - times[0]
    if duration < clip_span:
        raise ValueError("video_shorter_than_anchor_clip")
    anchor_time = times[anchor_frame_number]
    start = min(max(times[0], anchor_time - (sample_count // 2) * step),
                times[-1] - clip_span)
    targets = [start + rank * step for rank in range(sample_count)]
    indices = []
    sample_errors = []
    for target in targets:
        right = bisect_left(times, target)
        choices = [index for index in (right - 1, right) if 0 <= index < len(times)]
        index = min(choices, key=lambda item: (abs(times[item] - target), item))
        indices.append(index)
        sample_errors.append(abs(times[index] - target))
    if len(set(indices)) != sample_count or indices != sorted(indices):
        raise ValueError("anchor_centered_sample_indices_not_unique")
    anchor_rank = min(range(sample_count), key=lambda rank: abs(indices[rank] - anchor_frame_number))
    return {"sample_frame_indices": indices,
            "sample_fps": float(sample_fps),
            "clip_start_source_seconds": times[indices[0]],
            "clip_end_source_seconds": times[indices[-1]],
            "target_clip_span_seconds": clip_span,
            "anchor_sample_rank": anchor_rank,
            "anchor_time_error_seconds": abs(times[indices[anchor_rank]] - anchor_time),
            "maximum_grid_time_error_seconds": max(sample_errors)}


def include_exact_anchor_sample(*, sample_frame_indices: Sequence[int],
                                mp4_frame_times_seconds: Sequence[float],
                                anchor_frame_number: int) -> dict:
    """Replace the nearest sampled slot so an annotated source frame is exact.

    This preserves sample count and ordering. The neighboring temporal gaps may
    differ from the nominal sampling interval; callers must record actual PTS
    and use the same resulting sequence in every condition.
    """
    if (not isinstance(sample_frame_indices, Sequence)
            or isinstance(sample_frame_indices, (str, bytes))
            or not isinstance(mp4_frame_times_seconds, Sequence)
            or isinstance(mp4_frame_times_seconds, (str, bytes))
            or type(anchor_frame_number) is not int
            or not 0 <= anchor_frame_number < len(mp4_frame_times_seconds)
            or not sample_frame_indices):
        raise ValueError("exact_anchor_sampling_input_invalid")
    times = [_as_seconds(value) for value in mp4_frame_times_seconds]
    indices = list(sample_frame_indices)
    if (any(right <= left for left, right in zip(times, times[1:]))
            or any(type(index) is not int or not 0 <= index < len(times)
                   for index in indices)
            or indices != sorted(set(indices))
            ):
        raise ValueError("exact_anchor_sampling_geometry_invalid")
    if anchor_frame_number < indices[0] or anchor_frame_number > indices[-1]:
        raise ValueError("exact_anchor_outside_sampled_clip")
    if anchor_frame_number not in indices:
        candidates = [rank for rank, index in enumerate(indices)
                      if (rank == 0 or indices[rank - 1] < anchor_frame_number)
                      and (rank == len(indices) - 1
                           or anchor_frame_number < indices[rank + 1])]
        if not candidates:
            raise ValueError("exact_anchor_has_no_replaceable_slot")
        rank = min(candidates,
                   key=lambda item: (abs(times[indices[item]]
                                       - times[anchor_frame_number]), item))
        indices[rank] = anchor_frame_number
    indices.sort()
    if len(set(indices)) != len(sample_frame_indices):
        raise ValueError("exact_anchor_sampling_created_duplicate")
    rank = indices.index(anchor_frame_number)
    gaps = [times[right] - times[left]
            for left, right in zip(indices, indices[1:])]
    return {"sample_frame_indices": indices,
            "anchor_sample_rank": rank,
            "anchor_time_error_seconds": 0.0,
            "maximum_sample_gap_seconds": max(gaps) if gaps else 0.0,
            "minimum_sample_gap_seconds": min(gaps) if gaps else 0.0}


def evidence_temporal_groups(*, mask_frame_numbers: Sequence[int],
                             mp4_frame_times_seconds: Sequence[float],
                             sampled_frames: Sequence[Mapping[str, object]],
                             temporal_patch_size: int,
                             temporal_step_count: int) -> dict:
    """Map HD-EPIC MP4 mask frames to nearest sampled Qwen temporal groups.

    A mask frame maps by row index into the official MP4-frame time table.
    Uniformly sampled decoded frame records carry the sampler's source frame
    indices and actual relative PTS. Ties resolve to the earlier sampled
    frame. Every selected sample rank then maps to its fixed-size Qwen temporal
    patch group.
    """
    if (not isinstance(mask_frame_numbers, Sequence)
            or isinstance(mask_frame_numbers, (str, bytes)) or not mask_frame_numbers
            or not isinstance(mp4_frame_times_seconds, Sequence)
            or isinstance(mp4_frame_times_seconds, (str, bytes))
            or not isinstance(sampled_frames, Sequence)
            or isinstance(sampled_frames, (str, bytes)) or not sampled_frames
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or type(temporal_step_count) is not int or temporal_step_count < 1):
        raise ValueError("evidence_temporal_mapping_input_invalid")
    if any(type(index) is not int or not 0 <= index < len(mp4_frame_times_seconds)
           for index in mask_frame_numbers):
        raise ValueError("mask_frame_number_out_of_range")
    times = [_as_seconds(value) for value in mp4_frame_times_seconds]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("mp4_frame_times_not_strictly_increasing")
    sampled_indices = []
    sampled_times = []
    for frame in sampled_frames:
        if not isinstance(frame, Mapping):
            raise ValueError("sampled_frame_record_invalid")
        index = frame.get("decoded_frame_index")
        if type(index) is not int or not 0 <= index < len(times):
            raise ValueError("sampled_frame_index_out_of_range")
        sampled_indices.append(index)
        pair = frame.get("relative_presentation_time")
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes))
                or len(pair) != 2 or type(pair[0]) is not int
                or type(pair[1]) is not int or pair[1] <= 0):
            raise ValueError("sampled_frame_pts_invalid")
        pts_seconds = pair[0] / pair[1]
        if abs(pts_seconds - times[index]) > 0.001:
            raise ValueError("sampled_frame_pts_map_mismatch")
        sampled_times.append(pts_seconds)
    if sampled_indices != sorted(set(sampled_indices)):
        raise ValueError("sampled_frame_indices_not_strictly_increasing")
    if len(sampled_indices) % temporal_patch_size:
        raise ValueError("sampled_frame_count_not_patch_aligned")
    if len(sampled_indices) // temporal_patch_size != temporal_step_count:
        raise ValueError("qwen_temporal_step_count_mismatch")

    mapped = []
    groups = set()
    for frame_number in mask_frame_numbers:
        target_time = times[frame_number]
        rank = min(range(len(sampled_times)),
                   key=lambda i: (abs(sampled_times[i] - target_time), i))
        group = rank // temporal_patch_size
        groups.add(group)
        mapped.append({
            "mask_frame_number": frame_number,
            "mask_time_seconds": target_time,
            "sample_rank": rank,
            "sampled_frame_index": sampled_indices[rank],
            "sampled_time_seconds": sampled_times[rank],
            "absolute_time_error_seconds": abs(sampled_times[rank] - target_time),
            "temporal_group": group,
        })
    return {"mapped_mask_frames": mapped,
            "evidence_temporal_groups": sorted(groups),
            "temporal_step_count": temporal_step_count}


def validate_sampled_evidence_alignment(*, mapping: Mapping[str, object],
                                        track_groups: Sequence[int],
                                        maximum_error_seconds: float = 1.0) -> dict:
    """Fail closed unless every annotation-backed key group is a close track sample."""
    if (not isinstance(mapping, Mapping)
            or not isinstance(mapping.get("mapped_mask_frames"), Sequence)
            or not mapping["mapped_mask_frames"]
            or not isinstance(mapping.get("evidence_temporal_groups"), Sequence)
            or not mapping["evidence_temporal_groups"]
            or not isinstance(track_groups, Sequence)
            or isinstance(track_groups, (str, bytes))
            or type(maximum_error_seconds) not in (int, float)
            or not math.isfinite(float(maximum_error_seconds))
            or maximum_error_seconds <= 0):
        raise ValueError("evidence_alignment_input_invalid")
    errors = []
    for item in mapping["mapped_mask_frames"]:
        if not isinstance(item, Mapping):
            raise ValueError("evidence_alignment_mapping_invalid")
        error = item.get("absolute_time_error_seconds")
        if (isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(float(error)) or error < 0):
            raise ValueError("evidence_alignment_mapping_invalid")
        errors.append(float(error))
    evidence = mapping["evidence_temporal_groups"]
    if any(type(group) is not int or group < 0 for group in evidence):
        raise ValueError("evidence_alignment_group_invalid")
    if any(type(group) is not int or group < 0 for group in track_groups):
        raise ValueError("evidence_alignment_group_invalid")
    max_error = max(errors)
    if max_error > maximum_error_seconds:
        raise ValueError("mask_to_sample_time_error_above_threshold")
    if not set(evidence).issubset(set(track_groups)):
        raise ValueError("evidence_group_not_sampled_inside_track")
    return {"mapped_mask_frame_count": len(errors),
            "maximum_time_error_seconds": max_error,
            "evidence_group_count": len(evidence),
            "track_overlapping_evidence_group_count": len(evidence)}


def track_overlapping_temporal_groups(*, sampled_times_seconds: Sequence[float],
                                      temporal_patch_size: int,
                                      track_segments: Sequence[Sequence[float]],
                                      boundary_tolerance_seconds: float = 0.05) -> list[int]:
    """Return temporal groups with any sampled frame inside an annotated track."""
    if (not isinstance(sampled_times_seconds, Sequence)
            or isinstance(sampled_times_seconds, (str, bytes))
            or not sampled_times_seconds
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or len(sampled_times_seconds) % temporal_patch_size
            or not isinstance(track_segments, Sequence) or not track_segments
            or isinstance(boundary_tolerance_seconds, bool)
            or not isinstance(boundary_tolerance_seconds, (int, float))
            or not math.isfinite(float(boundary_tolerance_seconds))
            or boundary_tolerance_seconds < 0
            or boundary_tolerance_seconds > 0.1):
        raise ValueError("track_group_input_invalid")
    times = [_as_seconds(value) for value in sampled_times_seconds]
    normalized = []
    for segment in track_segments:
        if (not isinstance(segment, Sequence) or isinstance(segment, (str, bytes))
                or len(segment) != 2):
            raise ValueError("track_segment_invalid")
        start, end = segment
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start)) or not math.isfinite(float(end))
                or start < 0 or end < start):
            raise ValueError("track_segment_invalid")
        normalized.append((float(start), float(end)))
    return [group for group in range(len(times) // temporal_patch_size)
            if any(start - boundary_tolerance_seconds <= times[index]
                   <= end + boundary_tolerance_seconds
                   for index in range(group * temporal_patch_size,
                                      (group + 1) * temporal_patch_size)
                   for start, end in normalized)]


def score_generated_movement_anchors(*, text: str,
                                     track_segments: Sequence[Sequence[float]],
                                     object_names: Sequence[str] = ("mug", "cup")) -> dict:
    """Score named-object timestamp anchors against annotated movement tracks.

    This lexical/time metric ignores whether the described action is semantically
    correct. It is suitable only when the generation prompt asks exclusively
    for movement and to repeat the object name in each timed sentence.
    """
    if (not isinstance(text, str) or not isinstance(track_segments, Sequence)
            or not track_segments or not isinstance(object_names, Sequence)
            or isinstance(object_names, (str, bytes)) or not object_names
            or any(not isinstance(name, str) or not name.strip() for name in object_names)):
        raise ValueError("generated_anchor_input_invalid")
    object_pattern = re.compile(r"(?i)(?<!\w)(?:" + "|".join(
        re.escape(name.strip()) for name in sorted(set(object_names), key=len, reverse=True))
        + r")(?!\w)")
    sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
    anchors = []
    segments = []
    for segment in track_segments:
        if (not isinstance(segment, Sequence) or isinstance(segment, (str, bytes))
                or len(segment) != 2):
            raise ValueError("track_segment_invalid")
        start, end = segment
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start)) or not math.isfinite(float(end))
                or start < 0 or end <= start):
            raise ValueError("track_segment_invalid")
        segments.append((float(start), float(end)))
    for sentence in sentences:
        bins = parse_timestamp_bins(sentence)
        for start, end in bins:
            object_named = object_pattern.search(sentence) is not None
            aligned = object_named and any(
                timestamp_bins_overlap_segment([(start, end)], segment)
                for segment in segments)
            anchors.append({"time_seconds": start, "time_bin": [start, end],
                            "object_named": object_named,
                            "aligned_to_track": aligned})
    supported_segments = sum(any(a["object_named"]
        and timestamp_bins_overlap_segment([a["time_bin"]], (start, end))
        for a in anchors) for start, end in segments)
    return {"timestamp_anchor_count": len(anchors),
            "aligned_anchor_count": sum(item["aligned_to_track"] for item in anchors),
            "anchor_precision": (sum(item["aligned_to_track"] for item in anchors) / len(anchors)
                                 if anchors else 0.0),
            "movement_segment_count": len(segments),
            "covered_segment_count": supported_segments,
            "segment_recall": supported_segments / len(segments),
            "anchors": [{key: value for key, value in item.items()
                         if key != "time_bin"} for item in anchors]}


def matched_control_groups(*, temporal_step_count: int,
                           evidence_groups: Sequence[int],
                           excluded_groups: Sequence[int],
                           seed: str, record_token: str) -> list[int]:
    """Choose a stable equal-count sham group set outside evidence tracks."""
    if (type(temporal_step_count) is not int or temporal_step_count < 1
            or not isinstance(seed, str) or not seed or "\0" in seed
            or not isinstance(record_token, str) or not record_token or "\0" in record_token):
        raise ValueError("control_group_input_invalid")
    def validate(indices, name):
        if (not isinstance(indices, Sequence) or isinstance(indices, (str, bytes))
                or any(type(i) is not int or not 0 <= i < temporal_step_count for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError(name + "_invalid")
    validate(evidence_groups, "evidence_groups")
    validate(excluded_groups, "excluded_groups")
    if not evidence_groups:
        raise ValueError("evidence_groups_empty")
    forbidden = set(evidence_groups) | set(excluded_groups)
    candidates = [i for i in range(temporal_step_count) if i not in forbidden]
    if len(candidates) < len(evidence_groups):
        raise ValueError("insufficient_control_groups")
    prefix = f"minerva-ego-sham-v1\0{seed}\0{record_token}\0"
    chosen = sorted(candidates, key=lambda i: (hashlib.sha256(
        f"{prefix}{i}".encode("utf-8")).digest(), i))[:len(evidence_groups)]
    return sorted(chosen)


def bbox_to_merged_patch_offsets(*, bbox_xyxy: Sequence[float],
                                 source_width: int, source_height: int,
                                 grid_height: int, grid_width: int,
                                 patch_size: int, spatial_merge_size: int,
                                 maximum_clipped_fraction: float = 0.05) -> dict:
    """Map an original-frame XYXY box to overlapping merged visual patch cells.

    `grid_height`/`grid_width` are Qwen's pre-merge patch grid from
    `video_grid_thw`. The processor resizes the complete frame to this grid;
    the function therefore scales x and y independently to the realized grid
    dimensions. The returned cell offsets use row-major order after spatial
    merge, matching the vision merger's restored sequence order.

    Boxes are half-open `[xmin, ymin, xmax, ymax]` pixel coordinates. Small
    out-of-frame excursions are clipped only when their area fraction is within
    the caller's explicit threshold; the clipped fraction is returned.
    """
    if (not isinstance(bbox_xyxy, Sequence)
            or isinstance(bbox_xyxy, (str, bytes)) or len(bbox_xyxy) != 4
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in bbox_xyxy)):
        raise ValueError("bbox_xyxy_invalid")
    if (any(type(value) is not int or value < 1 for value in
            (source_width, source_height, grid_height, grid_width,
             patch_size, spatial_merge_size))
            or grid_height % spatial_merge_size
            or grid_width % spatial_merge_size):
        raise ValueError("bbox_patch_geometry_invalid")
    if (isinstance(maximum_clipped_fraction, bool)
            or not isinstance(maximum_clipped_fraction, (int, float))
            or not math.isfinite(float(maximum_clipped_fraction))
            or not 0 <= maximum_clipped_fraction < 1):
        raise ValueError("bbox_clipping_threshold_invalid")

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy_nonpositive_area")
    original_area = (x2 - x1) * (y2 - y1)
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(source_width), x2), min(float(source_height), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("bbox_outside_source_frame")
    clipped_area = (cx2 - cx1) * (cy2 - cy1)
    clipped_fraction = 1.0 - clipped_area / original_area
    if clipped_fraction > float(maximum_clipped_fraction):
        raise ValueError("bbox_clipped_fraction_above_threshold")

    merged_rows = grid_height // spatial_merge_size
    merged_cols = grid_width // spatial_merge_size
    processed_width = grid_width * patch_size
    processed_height = grid_height * patch_size
    merged_cell_width = spatial_merge_size * patch_size
    merged_cell_height = spatial_merge_size * patch_size
    px1 = cx1 * processed_width / source_width
    px2 = cx2 * processed_width / source_width
    py1 = cy1 * processed_height / source_height
    py2 = cy2 * processed_height / source_height
    col_start = max(0, min(merged_cols - 1,
                           math.floor(px1 / merged_cell_width)))
    col_stop = max(col_start + 1, min(merged_cols,
                                     math.ceil(px2 / merged_cell_width)))
    row_start = max(0, min(merged_rows - 1,
                           math.floor(py1 / merged_cell_height)))
    row_stop = max(row_start + 1, min(merged_rows,
                                     math.ceil(py2 / merged_cell_height)))
    cells = tuple(row * merged_cols + col
                  for row in range(row_start, row_stop)
                  for col in range(col_start, col_stop))
    return {"merged_grid_hw": [merged_rows, merged_cols],
            "patch_cell_offsets": cells,
            "patch_cell_rc": tuple((offset // merged_cols,
                                    offset % merged_cols) for offset in cells),
            "source_bbox_xyxy": (x1, y1, x2, y2),
            "clipped_bbox_xyxy": (cx1, cy1, cx2, cy2),
            "clipped_area_fraction": clipped_fraction,
            "maximum_clipped_fraction": float(maximum_clipped_fraction)}


def bbox_to_video_visual_key_positions(*, bbox_xyxy: Sequence[float],
                                       frame_rank: int, frame_count: int,
                                       temporal_patch_size: int,
                                       visual_token_groups: Sequence[Sequence[int]],
                                       source_width: int, source_height: int,
                                       grid_height: int, grid_width: int,
                                       patch_size: int,
                                       spatial_merge_size: int,
                                       maximum_clipped_fraction: float = 0.05) -> dict:
    """Map one processor-input frame-rank box to Qwen merged-token keys.

    `frame_rank` is the frame's position in the exact video tensor passed to
    the processor, not its source-video frame number. Temporal patches fuse
    adjacent frames, so ranks in the same temporal pair map to the same grid.
    """
    if (type(frame_count) is not int or frame_count < 1
            or type(frame_rank) is not int or not 0 <= frame_rank < frame_count
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or frame_count % temporal_patch_size
            or not isinstance(visual_token_groups, Sequence)
            or isinstance(visual_token_groups, (str, bytes))
            or len(visual_token_groups) != frame_count // temporal_patch_size):
        raise ValueError("bbox_video_group_geometry_invalid")
    mapped = bbox_to_merged_patch_offsets(
        bbox_xyxy=bbox_xyxy, source_width=source_width,
        source_height=source_height, grid_height=grid_height,
        grid_width=grid_width, patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        maximum_clipped_fraction=maximum_clipped_fraction)
    group_index = frame_rank // temporal_patch_size
    group = visual_token_groups[group_index]
    expected_tokens = math.prod(mapped["merged_grid_hw"])
    if (not isinstance(group, Sequence) or isinstance(group, (str, bytes))
            or len(group) != expected_tokens
            or any(type(position) is not int or position < 0 for position in group)
            or len(set(group)) != len(group)
            or any(left >= right for left, right in zip(group, group[1:]))):
        raise ValueError("bbox_visual_group_positions_invalid")
    return {**mapped,
            "frame_rank": frame_rank,
            "temporal_patch_size": temporal_patch_size,
            "temporal_group_index": group_index,
            "visual_key_positions": tuple(
                group[offset] for offset in mapped["patch_cell_offsets"])}


def matched_spatial_sham_offsets(*, merged_cell_count: int,
                                 support_offsets: Sequence[int],
                                 seed: str, record_token: str) -> tuple[int, ...]:
    """Choose a deterministic equal-dose set outside the annotated bbox cells."""
    if (type(merged_cell_count) is not int or merged_cell_count < 1
            or not isinstance(support_offsets, Sequence)
            or isinstance(support_offsets, (str, bytes)) or not support_offsets
            or any(type(value) is not int or not 0 <= value < merged_cell_count
                   for value in support_offsets)
            or len(set(support_offsets)) != len(support_offsets)
            or not isinstance(seed, str) or not seed
            or not isinstance(record_token, str) or not record_token):
        raise ValueError("spatial_sham_input_invalid")
    candidates = [value for value in range(merged_cell_count)
                  if value not in set(support_offsets)]
    if len(candidates) < len(support_offsets):
        raise ValueError("spatial_sham_dose_unavailable")
    prefix = f"{seed}\0{record_token}\0"
    ranked = sorted(candidates,
        key=lambda value: hashlib.sha256(f"{prefix}{value}".encode()).digest())
    return tuple(sorted(ranked[:len(support_offsets)]))
