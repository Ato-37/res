#!/usr/bin/env python3
"""Compare frozen head rankings across query position and temporal resolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

import run_head_query_position_exploration as query


PROTOCOL = Path("experiments/2026-10-19_head_query_by_temporal_resolution_protocol.md")
EXPECTED_16_SUMMARY_SHA256 = "9efd7d6b57bdbea210093b4c4c590b1f31ec0fe208056f62507f8be1b3dc41c2"
EXPECTED_16_METRICS_SHA256 = "11735b2fb92b8c0aa8db7c72bffd74e1cccf17f94c7492f14b3a23dc33d71d82"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays(archive, prefix):
    return tuple(np.asarray(archive[f"{prefix}_{metric}"])
                 for metric in ("mass", "entropy", "peak"))


def _top20(metrics, temporal_bins):
    eligible, elbow_n = query.summarize_all_heads(
        *metrics, temporal_bins=temporal_bins)
    return {
        "elbow_survivors": elbow_n,
        "non_sink_survivors": len(eligible),
        "top20": [(int(row[0]), int(row[1])) for row in eligible[:20]],
    }


def _jaccard(left, right):
    a, b = set(left["top20"]), set(right["top20"])
    union = a | b
    return {"intersection": len(a & b), "union": len(union),
            "jaccard": len(a & b) / len(union) if union else None}


def compare_four_strata(metrics8, indices8, metrics16, indices16):
    """Re-rank four settings on the exact intersection of successful rows."""
    index8 = np.asarray(indices8, dtype=np.int64)
    index16 = np.asarray(indices16, dtype=np.int64)
    if (index8.ndim != 1 or index16.ndim != 1 or len(set(index8.tolist())) != len(index8)
            or len(set(index16.tolist())) != len(index16)):
        raise ValueError("sample_indices_must_be_unique_vectors")
    common = sorted(set(index8.tolist()) & set(index16.tolist()))
    if not common:
        raise ValueError("no_common_successful_samples")
    pos8 = {sample: i for i, sample in enumerate(index8.tolist())}
    pos16 = {sample: i for i, sample in enumerate(index16.tolist())}

    def aligned(arrays, positions):
        result = tuple(np.asarray(array)[positions] for array in arrays)
        if any(array.shape[0] != len(common) for array in result):
            raise ValueError("aligned_metric_sample_count_mismatch")
        return result

    m8 = {name: aligned(value, [pos8[i] for i in common])
          for name, value in metrics8.items()}
    m16 = {name: aligned(value, [pos16[i] for i in common])
           for name, value in metrics16.items()}
    ranked = {
        "8_frame_generation_prefix": _top20(m8["generation_prefix"], 4),
        "8_frame_question_body_mean": _top20(m8["question_body_mean"], 4),
        "16_frame_generation_prefix": _top20(m16["generation_prefix"], 8),
        "16_frame_question_body_mean": _top20(m16["question_body_mean"], 8),
    }
    q8 = _jaccard(ranked["8_frame_generation_prefix"],
                  ranked["8_frame_question_body_mean"])
    q16 = _jaccard(ranked["16_frame_generation_prefix"],
                   ranked["16_frame_question_body_mean"])
    rprefix = _jaccard(ranked["8_frame_generation_prefix"],
                       ranked["16_frame_generation_prefix"])
    rquestion = _jaccard(ranked["8_frame_question_body_mean"],
                         ranked["16_frame_question_body_mean"])
    return {
        "matched_sample_count": len(common),
        "matched_sample_index_sha256": hashlib.sha256(
            np.asarray(common, dtype=np.int32).tobytes()).hexdigest(),
        "ranked_conditions": ranked,
        "contrasts": {
            "query_position_jaccard_within_8_frames": q8,
            "query_position_jaccard_within_16_frames": q16,
            "resolution_jaccard_generation_prefix": rprefix,
            "resolution_jaccard_question_body_mean": rquestion,
            "query_position_jaccard_difference_16_minus_8":
                q16["jaccard"] - q8["jaccard"]
                if q8["jaccard"] is not None and q16["jaccard"] is not None else None,
        },
    }


def load_run(summary_path: Path, metrics_path: Path, expected_bins: int):
    summary = json.loads(summary_path.read_text())
    if summary.get("frames") != expected_bins * 2 or summary.get("temporal_bins") != expected_bins:
        raise ValueError("unexpected_resolution")
    if summary.get("metrics_sha256") != sha256(metrics_path):
        raise ValueError("metrics_hash_mismatch")
    with np.load(metrics_path, allow_pickle=False) as archive:
        indices = archive["sample_index"].astype(np.int64)
        metrics = {name: _arrays(archive, name)
                   for name in query.MODE_NAMES}
    for name, triplet in metrics.items():
        if triplet[0].shape[0] != len(indices):
            raise ValueError(f"{name}_sample_axis_mismatch")
        query.validate_paired_metrics(triplet, triplet, temporal_bins=expected_bins)
    return summary, indices, metrics


def _write_exclusive(path: Path, data: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-8", required=True, type=Path)
    parser.add_argument("--metrics-8", required=True, type=Path)
    parser.add_argument("--summary-16", required=True, type=Path)
    parser.add_argument("--metrics-16", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing_to_overwrite_output")
    if sha256(args.summary_16) != EXPECTED_16_SUMMARY_SHA256:
        parser.error("unexpected_16_frame_summary_hash")
    if sha256(args.metrics_16) != EXPECTED_16_METRICS_SHA256:
        parser.error("unexpected_16_frame_metrics_hash")
    s8, i8, m8 = load_run(args.summary_8, args.metrics_8, 4)
    s16, i16, m16 = load_run(args.summary_16, args.metrics_16, 8)
    if (s8.get("protocol_file") != PROTOCOL.as_posix()
            or s8.get("protocol_sha256") != sha256(PROTOCOL)):
        parser.error("8_frame_protocol_hash_mismatch")
    for key in ("data_manifest_sha256", "model_revision", "max_pixels"):
        if s8.get(key) != s16.get(key):
            parser.error(f"run_mismatch_{key}")
    result = compare_four_strata(m8, i8, m16, i16)
    payload = {
        "schema": "head_query_temporal_resolution_comparison_v1",
        "research_status": "exploratory_post_selection_only",
        "claim": "attention-shape ranking robustness only; not faithfulness",
        "protocol_sha256": sha256(PROTOCOL),
        "analyzer_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "summary_8_sha256": sha256(args.summary_8),
        "metrics_8_sha256": sha256(args.metrics_8),
        "summary_16_sha256": sha256(args.summary_16),
        "metrics_16_sha256": sha256(args.metrics_16),
        "matched_sample_count": result["matched_sample_count"],
        "matched_sample_index_sha256": result["matched_sample_index_sha256"],
        "contrasts": result["contrasts"],
        "ranked_conditions": result["ranked_conditions"],
        "interpretation_limit": "descriptive post-selection comparison; no inferential test or faithfulness claim",
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.output.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("private_output_directory_must_be_0700")
    _write_exclusive(args.output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"matched_samples": result["matched_sample_count"],
                      "contrasts": result["contrasts"], "output": str(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
