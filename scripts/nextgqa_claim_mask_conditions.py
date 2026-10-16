"""Construct exact full/evidence-mask/control-mask RGB frame lists.

This dependency-light helper performs no media I/O, prompting, model loading,
scoring, or authorization. Inputs are already-decoded sampled RGB frames, their
full-video exact PTS map, the sampled decoded-frame indices, and reviewed time
intervals supplied by the caller. Callers must authenticate interval provenance,
clock calibration, and interval bounds against independently verified media
duration separately; the last decoded PTS need not equal playable duration.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib

from necessity_choice_margin_utils import apply_mask_fill
from presentation_time_utils import assign_evidence_control_frames_from_times

FRAME_COUNT = 32
CONDITIONS = ("full", "claim_evidence_black", "temporal_control_black")


def _frame_sha256(image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _validate_frames(frames) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ValueError("pillow_unavailable") from exc
    if not isinstance(frames, (list, tuple)) or len(frames) != FRAME_COUNT:
        raise ValueError("sampled_frame_count_mismatch")
    if any(not isinstance(frame, Image.Image) or frame.mode != "RGB" for frame in frames):
        raise ValueError("sampled_frames_must_be_rgb_images")
    dimensions = {(int(frame.width), int(frame.height)) for frame in frames}
    if len(dimensions) != 1:
        raise ValueError("sampled_frame_dimensions_mismatch")
    width, height = next(iter(dimensions))
    if width < 1 or height < 1:
        raise ValueError("sampled_frame_dimensions_invalid")
    return width, height


def build_claim_mask_conditions(*, frames, sampled_indices,
                                presentation_times, intervals,
                                include_metadata: bool = False):
    """Return full, black evidence-mask, and matched black-control frame lists.

    `presentation_times` is the complete decoder map as nonnegative exact
    ``Fraction`` seconds relative to its first frame. `sampled_indices` must
    equal the frozen endpoint-inclusive 32-index sampler. Evidence membership
    uses inclusive interval endpoints; temporal controls use the shared
    minimum-rank-distance/lexicographic tie-break helper.

    The input frames are never modified. When requested, metadata contains only
    numeric assignments, dimensions, and per-frame pixel SHA-256 digests.
    """
    if type(include_metadata) is not bool:
        raise ValueError("include_metadata_must_be_bool")
    dimensions = _validate_frames(frames)
    if (not isinstance(presentation_times, (list, tuple))
            or not presentation_times
            or any(not isinstance(value, Fraction) for value in presentation_times)):
        raise ValueError("presentation_times_must_be_exact_fractions")
    if (not isinstance(sampled_indices, (list, tuple))
            or len(sampled_indices) != FRAME_COUNT
            or any(type(index) is not int for index in sampled_indices)):
        raise ValueError("sampled_indices_invalid")

    assignment = assign_evidence_control_frames_from_times(
        len(presentation_times), FRAME_COUNT, presentation_times, intervals)
    if assignment.get("status") != "ok":
        raise ValueError("frame_assignment_" + str(assignment.get("status", "invalid")))
    if list(sampled_indices) != assignment["sampled_indices"]:
        raise ValueError("sampled_indices_do_not_match_frozen_sampler")

    evidence = assignment["evidence_indices"]
    controls = assignment["control_indices"]
    sampled = assignment["sampled_indices"]
    if (not evidence or len(evidence) != len(controls)
            or len(set(evidence)) != len(evidence)
            or len(set(controls)) != len(controls)
            or set(evidence) & set(controls)
            or not set(evidence).issubset(sampled)
            or not set(controls).issubset(sampled)):
        raise ValueError("evidence_control_assignment_invalid")

    rank_by_index = {frame_index: rank for rank, frame_index in enumerate(sampled)}
    evidence_ranks = {rank_by_index[index] for index in evidence}
    control_ranks = {rank_by_index[index] for index in controls}
    if (len(evidence_ranks) != len(evidence)
            or len(control_ranks) != len(controls)
            or evidence_ranks & control_ranks):
        raise ValueError("evidence_control_rank_assignment_invalid")

    full = [frame.copy() for frame in frames]
    evidence_masked = [
        apply_mask_fill(frame, "black") if rank in evidence_ranks else frame.copy()
        for rank, frame in enumerate(frames)
    ]
    control_masked = [
        apply_mask_fill(frame, "black") if rank in control_ranks else frame.copy()
        for rank, frame in enumerate(frames)
    ]
    conditions = {
        "full": full,
        "claim_evidence_black": evidence_masked,
        "temporal_control_black": control_masked,
    }
    if any(len(group) != FRAME_COUNT for group in conditions.values()):
        raise ValueError("condition_frame_count_mismatch")
    if any((frame.width, frame.height) != dimensions
           for group in conditions.values() for frame in group):
        raise ValueError("condition_frame_dimensions_mismatch")

    if not include_metadata:
        return conditions
    metadata = {
        "status": "assignment_constructed_unvalidated",
        "frame_count": FRAME_COUNT,
        "sampled_indices": list(sampled),
        "evidence_indices": list(evidence),
        "control_indices": list(controls),
        "evidence_ranks": sorted(evidence_ranks),
        "control_ranks": sorted(control_ranks),
        "dimensions": list(dimensions),
        "matched_rank_distance": assignment["matched_rank_distance"],
        "input_frame_sha256": [_frame_sha256(frame) for frame in frames],
        "condition_frame_sha256": {
            name: [_frame_sha256(frame) for frame in group]
            for name, group in conditions.items()
        },
        "clock_calibration_authenticated": False,
        "review_provenance_authenticated": False,
        "execution_authorization": "not_granted",
    }
    return {"conditions": conditions, "metadata": metadata}
