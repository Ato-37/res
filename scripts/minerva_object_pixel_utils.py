"""Strict, deterministic pixel-box helpers for Minerva-Ego pilot inputs.

These helpers operate on decoded RGB frames and XYXY half-open pixel boxes.
They do not establish that a box is a valid segmentation mask or that its
contents support an explanation.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from math import gcd


def _finite_xyxy(value, label: str) -> tuple[float, float, float, float]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 4):
        raise ValueError(f"{label}_invalid")
    coords = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}_invalid")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}_invalid")
        coords.append(number)
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{label}_invalid")
    return x0, y0, x1, y1


def _positive_dimension(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}_invalid")
    return value


def clip_bbox_xyxy(bbox_xyxy: Sequence[float], width: int, height: int,
                   max_clipped_fraction: float = 0.5) -> dict:
    """Clip finite XYXY coordinates and rasterize to a half-open integer box.

    The left/top edges are floored and right/bottom edges are ceiled so the
    returned pixel rectangle covers the continuous clipped annotation. The
    clipping fraction is the continuous area removed from the original box.
    Boxes exceeding ``max_clipped_fraction`` fail closed.
    """
    width = _positive_dimension(width, "frame_width")
    height = _positive_dimension(height, "frame_height")
    if (isinstance(max_clipped_fraction, bool)
            or not isinstance(max_clipped_fraction, (int, float))
            or not math.isfinite(float(max_clipped_fraction))
            or not 0.0 <= float(max_clipped_fraction) <= 1.0):
        raise ValueError("max_clipped_fraction_invalid")
    x0, y0, x1, y1 = _finite_xyxy(bbox_xyxy, "bbox")
    original_area = (x1 - x0) * (y1 - y0)
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(width), x1), min(float(height), y1)
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError("bbox_outside_frame")
    clipped_fraction = 1.0 - ((cx1 - cx0) * (cy1 - cy0) / original_area)
    # Protect against tiny floating point excursions around zero.
    clipped_fraction = min(1.0, max(0.0, clipped_fraction))
    if clipped_fraction > float(max_clipped_fraction) + 1e-12:
        raise ValueError("bbox_clipping_fraction_exceeded")
    integer_box = (max(0, math.floor(cx0)), max(0, math.floor(cy0)),
                   min(width, math.ceil(cx1)), min(height, math.ceil(cy1)))
    if integer_box[2] <= integer_box[0] or integer_box[3] <= integer_box[1]:
        raise ValueError("bbox_integer_box_empty")
    return {"box_xyxy": integer_box,
            "clipped_bbox_xyxy": (cx0, cy0, cx1, cy1),
            "clipped_area_fraction": clipped_fraction,
            "original_area": original_area,
            "clipped_area": (cx1 - cx0) * (cy1 - cy0)}


def _validated_integer_box(box, *, frame_width: int,
                           frame_height: int, label: str) -> tuple[int, int, int, int]:
    if (not isinstance(box, Sequence) or isinstance(box, (str, bytes))
            or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in box)):
        raise ValueError(f"{label}_invalid")
    x0, y0, x1, y1 = box
    if (x0 < 0 or y0 < 0 or x1 > frame_width or y1 > frame_height
            or x1 <= x0 or y1 <= y0):
        raise ValueError(f"{label}_out_of_bounds")
    return x0, y0, x1, y1


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def select_spatial_sham_box(support_box: Sequence[int], width: int, height: int,
                            forbidden_boxes: Sequence[Sequence[int]],
                            seed: str) -> tuple[int, int, int, int]:
    """Choose a deterministic same-size integer box disjoint from exclusions.

    A SHA-256-seeded modular traversal visits every possible top-left pixel
    position once at most, without allocating a list proportional to frame area.
    The target box is always included among exclusions.
    """
    width = _positive_dimension(width, "frame_width")
    height = _positive_dimension(height, "frame_height")
    if not isinstance(seed, str) or not seed:
        raise ValueError("spatial_sham_seed_invalid")
    target = _validated_integer_box(support_box,
        frame_width=width, frame_height=height, label="target_box")
    exclusions = [target]
    if (not isinstance(forbidden_boxes, Sequence)
            or isinstance(forbidden_boxes, (str, bytes))):
        raise ValueError("excluded_boxes_invalid")
    exclusions.extend(_validated_integer_box(box, frame_width=width,
        frame_height=height, label="excluded_box") for box in forbidden_boxes)
    box_width, box_height = target[2] - target[0], target[3] - target[1]
    positions_w, positions_h = width - box_width + 1, height - box_height + 1
    if positions_w < 1 or positions_h < 1:
        raise ValueError("target_box_out_of_bounds")
    position_count = positions_w * positions_h
    digest = hashlib.sha256(f"minerva-object-pixel-sham-v1\0{seed}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % position_count
    step = int.from_bytes(digest[8:16], "big") % position_count or 1
    while gcd(step, position_count) != 1:
        step = (step + 1) % position_count or 1
    for ordinal in range(position_count):
        linear = (start + ordinal * step) % position_count
        x0, y0 = linear % positions_w, linear // positions_w
        candidate = (x0, y0, x0 + box_width, y0 + box_height)
        if all(not _boxes_overlap(candidate, box) for box in exclusions):
            return candidate
    raise ValueError("spatial_sham_unavailable")


def mean_fill_box(image, integer_box_xyxy: Sequence[int]):
    """Return an RGB PIL copy with a box filled by the source frame RGB mean.

    Pixels outside the half-open box remain byte-identical. The global frame
    mean is rounded to nearest uint8 with NumPy's ties-to-even rule.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("numpy_and_pillow_required") from exc
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("image_must_be_rgb_pil")
    width, height = image.size
    box = _validated_integer_box(integer_box_xyxy,
        frame_width=width, frame_height=height, label="integer_box")
    source = np.asarray(image, dtype=np.uint8)
    mean_rgb = np.rint(source.mean(axis=(0, 1), dtype=np.float64)).astype(np.uint8)
    output = source.copy()
    x0, y0, x1, y1 = box
    output[y0:y1, x0:x1] = mean_rgb
    return Image.fromarray(output, mode="RGB")
