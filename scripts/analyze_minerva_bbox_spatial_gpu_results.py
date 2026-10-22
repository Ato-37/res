#!/usr/bin/env python3
"""Verify and summarize the fixed three-row Minerva bbox spatial GPU pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/private_minerva"
DEFAULT_RESULTS = [
    PRIVATE / "minerva_ego_bbox_spatial_gpu_20261022_P05_02.json",
    PRIVATE / "minerva_ego_bbox_spatial_gpu_20261022_P06_01.json",
    PRIVATE / "minerva_ego_bbox_spatial_gpu_20261022_P07_01.json",
]
DEFAULT_TOKEN_PREFLIGHTS = [
    PRIVATE / "minerva_ego_bbox_ablation_token_preflight_20261022_P05_01.json",
    PRIVATE / "minerva_ego_bbox_ablation_token_preflight_20261022_P06_01.json",
    PRIVATE / "minerva_ego_bbox_ablation_token_preflight_20261022_P07_01.json",
]
EXPECTED_ROWS = {
    "P05-20240425-125459": "1b337cd3246461596a11f1b08108606776eb3764ad617949e6db7e6b37ac015a",
    "P06-20240510-140459": "3ba12ffa507dec3e2009f7ec315b21c6e591e073f31f71455bde060657b06f26",
    "P07-20240529-131737": "76face140234c2eda1bf1678326a8817929befc9784abf75a4744d86f6ab5501",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=3, type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--preflight", type=Path,
        help="Matching frozen CPU preflight; inferred for the built-in positive or ablation result set")
    parser.add_argument("--token-preflights", nargs=3, type=Path,
        help="Optional row-aligned full-prompt CPU token receipts")
    args = parser.parse_args(argv)
    if args.preflight is None:
        if args.results == DEFAULT_RESULTS:
            suffix = "08"
        elif all("ablation" in path.name for path in args.results):
            suffix = "12"
        else:
            parser.error("--preflight is required for a custom non-ablation result set")
        args.preflight = PRIVATE / f"minerva_ego_bbox_patch_preflight_20261022_{suffix}.json"
    preflight = json.loads(args.preflight.read_text())
    preflight_sha = sha256_file(args.preflight)
    if (preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or preflight.get("requested_rows") != 3
            or preflight.get("effective_rows") != 3):
        raise ValueError("three_row_preflight_not_complete")

    records = []
    observed_signature = None
    token_paths = args.token_preflights
    if token_paths is None and all("ablation" in path.name for path in args.results):
        token_paths = DEFAULT_TOKEN_PREFLIGHTS
    for index, path in enumerate(args.results):
        result = json.loads(path.read_text())
        if result.get("status") != "completed_single_video_exploratory_reference_likelihood_pilot":
            raise ValueError("gpu_result_status_invalid")
        row_hash = result.get("row_key_sha256")
        matching = [name for name, expected in EXPECTED_ROWS.items()
                    if expected == row_hash]
        if len(matching) != 1:
            raise ValueError("gpu_result_row_not_in_frozen_cohort")
        operation = result.get("intervention_operation", "key_bias")
        suffix = "ablation" if operation == "key_ablation" else "bias"
        if operation == "key_ablation":
            # Legacy receipts carried an unused CLI bias value even though the
            # intervention deleted keys with -inf. Report the effective action.
            signature = (operation, None, None)
        else:
            signature = (operation, result.get("bias_sign"), result.get("bias_nats"))
        if observed_signature is None:
            observed_signature = signature
        elif signature != observed_signature:
            raise ValueError("mixed_intervention_signatures")
        support_key = f"object_bbox_support_{suffix}"
        spatial_key = f"equal_dose_same_pair_spatial_sham_{suffix}"
        temporal_key = f"equal_dose_temporal_bbox_sham_{suffix}"
        if (result.get("cohort_sha256") != preflight.get("cohort_sha256")
                or result.get("preflight_receipt_sha256") != preflight_sha
                or result.get("intervention") != "spatial_bbox"
                or result.get("forward_count") != 4
                or result.get("direct_generated_explanation_faithfulness") is not False):
            raise ValueError("gpu_result_provenance_or_scope_mismatch")
        if token_paths is not None:
            token_path = token_paths[index]
            token_receipt = json.loads(token_path.read_text())
            if (result.get("token_preflight_receipt_sha256") != sha256_file(token_path)
                    or token_receipt.get("row_key_sha256") != row_hash
                    or token_receipt.get("preflight_receipt_sha256") != preflight_sha
                    or token_receipt.get("runtime_source_sha256")
                       != result.get("runtime_source_sha256")):
                raise ValueError("gpu_token_preflight_receipt_binding_mismatch")
        scores = result.get("target_mean_logprob_nats_per_token")
        if not isinstance(scores, dict) or set(scores) != {
                "baseline", support_key, spatial_key, temporal_key}:
            raise ValueError("gpu_result_conditions_incomplete")
        mapped = result.get("mapped_bbox_to_video_visual_keys") or {}
        dose = len(mapped.get("patch_cell_offsets") or [])
        sham_counts = [len(result.get("spatial_sham_offsets") or [])] * 2
        if dose < 1 or any(count != dose for count in sham_counts):
            raise ValueError("spatial_intervention_dose_mismatch")
        records.append({"row": matching[0], "path": str(path),
            "sha256": sha256_file(path), "dose_keys": dose,
            "scores": scores,
            "support_minus_baseline": scores[support_key] - scores["baseline"],
            "support_minus_spatial_sham": scores[support_key] - scores[spatial_key],
            "support_minus_temporal_sham": scores[support_key] - scores[temporal_key],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "elapsed_scoring_seconds": result["elapsed_scoring_seconds"]})
    if {record["row"] for record in records} != set(EXPECTED_ROWS):
        raise ValueError("frozen_cohort_row_coverage_mismatch")

    fields = ("support_minus_baseline", "support_minus_spatial_sham",
              "support_minus_temporal_sham")
    summary = {}
    for field in fields:
        values = [record[field] for record in records]
        summary[field] = {"mean": statistics.mean(values),
                          "minimum": min(values), "maximum": max(values)}
    print(json.dumps({"requested_rows": 3, "effective_rows": 3,
        "preflight_sha256": preflight_sha,
        "results": records, "paired_descriptive_contrasts": summary,
        "intervention_signature": {"operation": observed_signature[0],
            "bias_sign": observed_signature[1], "bias_nats": observed_signature[2]},
        "inferential_test": None,
        "interpretation_scope": "exploratory fixed-rationale likelihood; not generated-explanation faithfulness"},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
