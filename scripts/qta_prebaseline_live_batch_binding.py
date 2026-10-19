"""Bind live in-process QTA processor batches to the ordered private cohort.

This CPU-side provenance leaf checks that the exact ``FrozenBatch`` objects
presented to a later scorer match caller-pinned digests and source/review row
order. The pins remain unauthenticated here. This module never loads a model,
returns a batch, authorizes inference, or reports faithfulness.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
import math
from typing import Any

import torch

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256

ROW_COUNT = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")


def _validate_cpu_value(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise ValueError("processor_batch_mapping_key_invalid")
        for child in value.values():
            _validate_cpu_value(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_cpu_value(child)
    elif isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.is_quantized or value.is_complex()):
            raise ValueError("processor_batch_tensor_unsupported")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError("processor_batch_nonfinite_tensor")
    elif value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("processor_batch_nonfinite_value")
    else:
        raise ValueError("processor_batch_value_unsupported")


def validate_live_processor_batches(*, expected_rows: Any, live_rows: Any,
                                   source_crosswalk: Any) -> dict[str, Any]:
    """Atomically validate exactly three ordered live batches.

    Rows have private schema ``{review_token, source_video_sha256,
    batch_sha256, batch}``. Expected rows have the same shape except ``batch``.
    ``source_crosswalk`` is the ordered output of the source/review byte join.
    No partial row count is returned on failure and no hashes or identifiers
    are included in the public result.
    """
    try:
        if (not isinstance(expected_rows, (list, tuple))
                or len(expected_rows) != ROW_COUNT
                or not isinstance(live_rows, (list, tuple))
                or len(live_rows) != ROW_COUNT
                or not isinstance(source_crosswalk, (list, tuple))
                or len(source_crosswalk) != ROW_COUNT):
            raise ValueError("processor_batch_three_row_inventory_required")

        seen_tokens: set[str] = set()
        seen_sources: set[str] = set()
        for expected, live, source in zip(
                expected_rows, live_rows, source_crosswalk, strict=True):
            if (not isinstance(expected, Mapping)
                    or set(expected) != {"review_token", "source_video_sha256", "batch_sha256"}
                    or not isinstance(live, Mapping)
                    or set(live) != {"review_token", "source_video_sha256", "batch_sha256", "batch"}
                    or not isinstance(source, Mapping)):
                raise ValueError("processor_batch_row_schema_invalid")
            token = expected["review_token"]
            source_sha = expected["source_video_sha256"]
            expected_digest = expected["batch_sha256"]
            if (not isinstance(token, str) or not _TOKEN.fullmatch(token)
                    or not isinstance(source_sha, str) or not _SHA256.fullmatch(source_sha)
                    or not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest)):
                raise ValueError("processor_batch_expected_pin_invalid")
            if token in seen_tokens or source_sha in seen_sources:
                raise ValueError("processor_batch_duplicate_source_or_token")
            seen_tokens.add(token)
            seen_sources.add(source_sha)
            if (source.get("review_token") != token
                    or source.get("video_sha256") != source_sha
                    or live.get("review_token") != token
                    or live.get("source_video_sha256") != source_sha
                    or live.get("batch_sha256") != expected_digest):
                raise ValueError("processor_batch_source_order_or_pin_mismatch")
            batch = live["batch"]
            if type(batch) is not FrozenBatch or not batch:
                raise ValueError("processor_batch_frozen_cpu_batch_required")
            _validate_cpu_value(batch)
            actual_digest = batch_sha256(batch)
            if actual_digest != expected_digest:
                raise ValueError("processor_batch_live_digest_mismatch")

        # Return aggregate-only evidence. A caller must still use the same
        # pinned digest in the scoring primitive, which rehashes its input.
        return {"status": "passed", "requested_rows": ROW_COUNT,
                "validated_rows": ROW_COUNT, "failure_code": None}
    except Exception as exc:
        code = str(exc)
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,119}", code):
            code = "processor_batch_validation_failed"
        return {"status": "blocked", "requested_rows": ROW_COUNT,
                "validated_rows": 0, "failure_code": code}
