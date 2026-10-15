"""Fail-closed PyAV adapter for exact PTS-based sampled video frames.

This module does not authorize inference or calibrate an annotation player's
clock. It decodes twice so it can choose the endpoint-inclusive index sample
after counting every displayed frame without retaining the full video in RAM.
The returned RGB PIL buffers are the same buffers a caller must pass to Qwen;
``verify_sampled_frame_pixels`` can check them immediately before processing.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
from PIL import Image

from necessity_choice_margin_utils import uniform_sample_indices
from presentation_time_utils import (
    presentation_times_from_pts,
    uniform_sampling_rate_from_times,
)


def _file_sha256(source) -> str:
    """Hash one already-open, seekable file object without changing its cursor."""
    digest = hashlib.sha256()
    old_position = source.tell()
    source.seek(0)
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    source.seek(old_position)
    return digest.hexdigest()


def _rational(value, label: str) -> Fraction:
    if value is None or isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{label}_missing_or_invalid")
    try:
        if isinstance(value, tuple):
            if len(value) != 2 or any(type(part) is not int for part in value):
                raise ValueError
            result = Fraction(*value)
        elif isinstance(value, (Fraction, int)):
            result = Fraction(value)
        else:
            raise ValueError
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{label}_missing_or_invalid") from exc
    if result <= 0:
        raise ValueError(f"{label}_must_be_positive")
    return result


def _seconds_fraction(value, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label}_must_be_finite_nonnegative")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{label}_must_be_finite_nonnegative") from exc
    if result < 0:
        raise ValueError(f"{label}_must_be_finite_nonnegative")
    return result


def _stream_and_decode(source):
    """Decode one video stream in display order and return PTS metadata."""
    source.seek(0)
    with av.open(source, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("video_stream_count_must_equal_one")
        stream = streams[0]
        stream.codec_context.thread_count = 1
        pts_values: list[int] = []
        pixel_hashes: list[str] = []
        time_bases: set[Fraction] = set()
        pixel_specs: set[tuple[int, int, str]] = set()
        for frame in container.decode(stream):
            if getattr(frame, "is_corrupt", False):
                raise ValueError("decoded_frame_marked_corrupt")
            if frame.pts is None:
                raise ValueError("decoded_frame_pts_missing")
            pts = int(frame.pts)
            if pts < -(1 << 63) or pts > (1 << 63) - 1:
                raise ValueError("decoded_frame_pts_out_of_range")
            raw_time_base = frame.time_base if frame.time_base is not None else stream.time_base
            time_base = _rational(raw_time_base, "frame_time_base")
            pts_values.append(pts)
            time_bases.add(time_base)
            pixel_specs.add((int(frame.width), int(frame.height), frame.format.name))
            pixel_hashes.append(_image_sha256(frame.to_image().convert("RGB")))
        if not pts_values:
            raise ValueError("decoded_video_has_no_frames")
        if len(time_bases) != 1:
            raise ValueError("decoded_frame_time_bases_differ")
        if len(pixel_specs) != 1:
            raise ValueError("decoded_frame_pixel_specs_differ")
        time_base = next(iter(time_bases))
        # The helper rejects duplicate or non-monotonic PTS in display order.
        relative_times = presentation_times_from_pts(pts_values, time_base)
        stream_info = {
            "stream_index": int(stream.index),
            "codec_name": str(stream.codec_context.name or "unknown"),
            "stream_time_base": _rational(stream.time_base, "stream_time_base"),
            "decoder_thread_count": int(stream.codec_context.thread_count),
            "container_format": str(container.format.name or "unknown"),
            "reported_stream_frame_count": int(stream.frames),
        }
        return (pts_values, time_base, relative_times, next(iter(pixel_specs)),
                pixel_hashes, stream_info)


def _decode_selected_frames(source, expected_pts: list[int], expected_time_base: Fraction,
                            expected_pixel_hashes: list[str],
                            selected_indices: set[int], expected_pixel_spec):
    frames: dict[int, Image.Image] = {}
    selected_pts: dict[int, int] = {}
    pts_values: list[int] = []
    time_bases: set[Fraction] = set()
    pixel_specs: set[tuple[int, int, str]] = set()
    source.seek(0)
    with av.open(source, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("video_stream_count_changed_between_decodes")
        stream = streams[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if getattr(frame, "is_corrupt", False):
                raise ValueError("sampled_pass_frame_marked_corrupt")
            if frame.pts is None:
                raise ValueError("decoded_frame_pts_missing_on_sample_pass")
            pts = int(frame.pts)
            raw_time_base = frame.time_base if frame.time_base is not None else stream.time_base
            time_base = _rational(raw_time_base, "frame_time_base")
            pts_values.append(pts)
            time_bases.add(time_base)
            pixel_spec = (int(frame.width), int(frame.height), frame.format.name)
            pixel_specs.add(pixel_spec)
            if index in selected_indices:
                if pixel_spec != expected_pixel_spec:
                    raise ValueError("sampled_frame_pixel_spec_changed_between_decodes")
                image = frame.to_image().convert("RGB")
                if _image_sha256(image) != expected_pixel_hashes[index]:
                    raise ValueError("sampled_frame_pixels_changed_between_decodes")
                frames[index] = image
                selected_pts[index] = pts
    if pts_values != expected_pts or time_bases != {expected_time_base}:
        raise ValueError("decoded_pts_changed_between_passes")
    if pixel_specs != {expected_pixel_spec}:
        raise ValueError("decoded_pixel_spec_changed_between_passes")
    if set(frames) != selected_indices:
        raise ValueError("sampled_frame_count_mismatch")
    return frames, selected_pts


def _image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def decode_sampled_video(path, *, num_frames: int, uniform_tolerance_seconds) -> tuple[list[Image.Image], dict]:
    """Decode one video twice and return exact sampled RGB frames plus metadata.

    The time-origin is the first decoded frame's PTS. The declared tolerance
    and sampled PTS must yield a uniform temporal grid, because the current
    Qwen list-frame input and rank-matched control cannot express arbitrary VFR
    timestamps. This function does not calibrate that origin to annotations.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError("video_input_missing")
    if type(num_frames) is not int or num_frames < 2:
        raise ValueError("num_frames_must_be_integer_at_least_two")
    tolerance = _seconds_fraction(uniform_tolerance_seconds, "uniform_tolerance")
    # Pin one inode for both PyAV passes. Reopening the path would allow a path
    # replacement between passes to make the recorded hash describe other bytes.
    with video_path.open("rb") as source:
        source_before = _file_sha256(source)
        (pts, time_base, relative_times, pixel_spec,
         pixel_hashes, stream_info) = _stream_and_decode(source)
        indices = uniform_sample_indices(len(pts), num_frames)
        if len(set(indices)) != num_frames:
            raise ValueError("insufficient_distinct_decoded_frames")
        sampled_times = [relative_times[index] for index in indices]
        sample_rate, max_deviation = uniform_sampling_rate_from_times(sampled_times, tolerance)
        selected_set = set(indices)
        selected_frames, selected_pts = _decode_selected_frames(
            source, pts, time_base, pixel_hashes, selected_set, pixel_spec)
        source_after = _file_sha256(source)
        if source_before != source_after:
            raise ValueError("video_input_changed_during_decode")
    ordered_frames = [selected_frames[index] for index in indices]
    frame_rows = []
    for index, frame in zip(indices, ordered_frames, strict=True):
        frame_rows.append({
            "decoded_frame_index": index,
            "pts": selected_pts[index],
            "relative_presentation_time": [
                relative_times[index].numerator, relative_times[index].denominator,
            ],
            "rgb_pixel_sha256": _image_sha256(frame),
            "rgb_size": [int(frame.width), int(frame.height)],
        })
    pts_payload = json.dumps(
        [[value, time_base.numerator, time_base.denominator] for value in pts],
        separators=(",", ":"),
    ).encode("ascii")
    pixel_hash_payload = json.dumps(pixel_hashes, separators=(",", ":")).encode("ascii")
    metadata = {
        "schema_version": 1,
        "purpose": "claim_mask_pyav_decode_sample_provenance",
        "source_sha256": source_before,
        "decoder": {
            "name": "PyAV",
            "version": str(av.__version__),
            "ffmpeg_libraries": {
                name: list(version) for name, version in sorted(av.library_versions.items())
            },
        },
        "stream": stream_info,
        "decoded_frame_count": len(pts),
        "decoded_pts_sequence_sha256": hashlib.sha256(pts_payload).hexdigest(),
        "pass1_decoded_rgb_sequence_sha256": hashlib.sha256(pixel_hash_payload).hexdigest(),
        "reported_stream_frame_count_matches_decoded": stream_info["reported_stream_frame_count"] == len(pts),
        "decoded_time_base": [time_base.numerator, time_base.denominator],
        "timestamp_origin": "first_decoded_presentation_timestamp",
        "decoded_pixel_spec": {
            "width": pixel_spec[0], "height": pixel_spec[1], "pixel_format": pixel_spec[2],
        },
        "sampler": "endpoint_inclusive_uniform_decoded_frame_index_v1",
        "requested_frame_count": num_frames,
        "sampled_frames": frame_rows,
        "uniform_grid_tolerance_seconds": [tolerance.numerator, tolerance.denominator],
        "effective_sample_fps": [sample_rate.numerator, sample_rate.denominator],
        "max_sampled_grid_deviation_seconds": [
            max_deviation.numerator, max_deviation.denominator,
        ],
        "annotation_clock_calibrated": False,
        "execution_authorization": "not_granted",
    }
    return ordered_frames, metadata


def verify_sampled_frame_pixels(frames, metadata: dict) -> None:
    """Raise unless current RGB frame buffers match their decoded hashes."""
    records = metadata.get("sampled_frames") if isinstance(metadata, dict) else None
    if not isinstance(frames, (list, tuple)) or not isinstance(records, list):
        raise ValueError("sampled_frame_pixel_manifest_invalid")
    if len(frames) != len(records):
        raise ValueError("sampled_frame_pixel_count_mismatch")
    for image, record in zip(frames, records, strict=True):
        if not isinstance(image, Image.Image) or not isinstance(record, dict):
            raise ValueError("sampled_frame_pixel_record_invalid")
        actual_size = [int(image.width), int(image.height)]
        if actual_size != record.get("rgb_size"):
            raise ValueError("sampled_frame_dimensions_mismatch")
        if _image_sha256(image) != record.get("rgb_pixel_sha256"):
            raise ValueError("sampled_frame_pixel_hash_mismatch")
