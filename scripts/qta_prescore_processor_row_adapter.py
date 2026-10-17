"""Project a processor mapper result into a non-authorizing QTA row binding."""
from __future__ import annotations

import re
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def processor_binding_row(*, review_token: str, source_video_sha256: str,
                          mapping: dict[str, Any]) -> dict[str, Any]:
    """Return only the mapper fields consumed by the three-row QTA join.

    This projection preserves no tensor or private path and authenticates no
    source, calibration, or human review. External pins are checked later by
    `audit_qta_prescore_mapping_bundle`.
    """
    if not isinstance(review_token, str) or not review_token:
        raise ValueError("processor_row_review_token_invalid")
    if (not isinstance(source_video_sha256, str)
            or not SHA256.fullmatch(source_video_sha256)):
        raise ValueError("processor_row_source_hash_invalid")
    if (not isinstance(mapping, dict)
            or mapping.get("status") != "processor_batch_snapshot_and_caller_metadata_bound"):
        raise ValueError("processor_mapping_status_invalid")
    batch_sha = _sha(mapping.get("typed_batch_sha256"), "typed_batch_sha256")
    sampling_sha = _sha(mapping.get("sampling_metadata_sha256"),
                        "sampling_metadata_sha256")
    grid = mapping.get("video_grid_thw")
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)):
        raise ValueError("processor_video_grid_invalid")
    intervals = mapping.get("reviewed_intervals_rational_seconds")
    if (not isinstance(intervals, list) or not intervals
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(not isinstance(value, str) or not value for value in pair)
                   for pair in intervals)):
        raise ValueError("processor_review_intervals_invalid")
    guard = mapping.get("interval_boundary_guard_rational_seconds")
    if not isinstance(guard, str) or not guard:
        raise ValueError("processor_clock_guard_invalid")
    evidence = _indices(mapping.get("evidence_group_indices"), "evidence_groups")
    control = _indices(mapping.get("control_group_indices"), "control_groups")
    return {
        "review_token": review_token,
        "source_video_sha256": source_video_sha256,
        "typed_batch_sha256": batch_sha,
        "sampling_metadata_sha256": sampling_sha,
        "video_grid_thw": list(grid),
        "reviewed_intervals_rational_seconds": [list(pair) for pair in intervals],
        "interval_boundary_guard_rational_seconds": guard,
        "evidence_group_indices": evidence,
        "control_group_indices": control,
    }


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"processor_row_{field}_invalid")
    return value


def _indices(value: Any, field: str) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(type(index) is not int or index < 0 for index in value)
            or value != sorted(set(value))):
        raise ValueError(f"processor_row_{field}_invalid")
    return list(value)
