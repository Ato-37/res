"""Compare two published Qwen sequence-length diagnostic artifacts.

This is a scalar/artifact reproducibility check. It deliberately ignores run
time, source/provenance fields, and the new load-time checkpoint witness while
requiring all fixed inputs, traces, comparison values, counts, and numeric
cross-checks to match exactly. It cannot reconstruct omitted tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IGNORED_KEYS = {
    "source_commit", "source_sha256", "started_utc", "elapsed_seconds",
    "checkpoint_shards_before_model_load", "checkpoint_shards_after_model_load", "checkpoint_byte_audit",
    "checkpoint_provenance_limitations", "gpu_memory", "created_utc",
}


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalized(v) for k, v in value.items() if k not in IGNORED_KEYS}
    if isinstance(value, list):
        return [_normalized(v) for v in value]
    return value


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_bytes = left_path.read_bytes()
    right_bytes = right_path.read_bytes()
    left = json.loads(left_bytes)
    right = json.loads(right_bytes)
    for name, report in (("left", left), ("right", right)):
        if report.get("status") != "diagnostic_complete":
            raise ValueError(f"{name} artifact is not complete")
        if report.get("completed_model_calls") != 5 or report.get("effective_comparison_pairs") != 19:
            raise ValueError(f"{name} artifact has unexpected fixed counts")
        if report.get("effective_hidden_row_comparisons") != 9 or report.get("effective_lm_head_projection_calls") != 10:
            raise ValueError(f"{name} artifact has unexpected projection counts")
        if report.get("numeric_crosscheck_passed") is not True:
            raise ValueError(f"{name} numeric cross-check did not pass")
        if report.get("failures") != 0:
            raise ValueError(f"{name} artifact records failures")
        before = report.get("checkpoint_shards_before_model_load")
        after = report.get("checkpoint_shards_after_model_load")
        if before is not None:
            if before.get("status") != "verified_immediately_before_model_load":
                raise ValueError(f"{name} pre-load witness status is invalid")
            if len(before.get("shards", [])) != 5:
                raise ValueError(f"{name} pre-load witness shard count is invalid")
        if after is not None:
            if after.get("status") != "verified_immediately_after_model_load":
                raise ValueError(f"{name} post-load witness status is invalid")
            if len(after.get("shards", [])) != 5:
                raise ValueError(f"{name} post-load witness shard count is invalid")
            if before is None or before.get("shards") != after.get("shards"):
                raise ValueError(f"{name} pre/post-load shard witness differs")
    normalized_equal = _normalized(left) == _normalized(right)
    if not normalized_equal:
        raise ValueError("fixed diagnostic fields differ")
    return {
        "analysis": "qwen_sequence_length_artifact_reproducibility",
        "left_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "right_sha256": hashlib.sha256(right_bytes).hexdigest(),
        "fixed_fields_equal": True,
        "tensor_recalculation": "not_possible_outputs_not_in_artifacts",
        "left_source_commit": left.get("source_commit"),
        "right_source_commit": right.get("source_commit"),
        "left_load_witness": left.get("checkpoint_shards_before_model_load", {}).get("status"),
        "right_load_witness": right.get("checkpoint_shards_before_model_load", {}).get("status"),
        "left_post_load_witness": left.get("checkpoint_shards_after_model_load", {}).get("status"),
        "right_post_load_witness": right.get("checkpoint_shards_after_model_load", {}).get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
