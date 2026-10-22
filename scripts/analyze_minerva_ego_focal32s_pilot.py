#!/usr/bin/env python3
"""Recompute descriptive paired contrasts for the private focal-clip pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
CONDITIONS = ("baseline", "object_time_support_bias", "equal_dose_temporal_sham_bias")
LAYERS = {str(layer) for layer in range(5, 17)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_results(results: list[dict], result_hashes: list[str] | None = None) -> dict:
    if len(results) != 3:
        raise ValueError("expected_exactly_three_rows")
    keys = [item.get("row_key_sha256") for item in results]
    if any(not isinstance(key, str) or len(key) != 64 for key in keys) or len(set(keys)) != 3:
        raise ValueError("row_binding_invalid")
    cohort_hashes = {item.get("cohort_sha256") for item in results}
    runners = {item.get("runtime_source_sha256", {}).get(
        "run_minerva_ego_anchor_gpu_pilot.py") for item in results}
    helpers = {item.get("runtime_source_sha256", {}).get(
        "minerva_ego_anchor_utils.py") for item in results}
    revisions = {item.get("model_revision") for item in results}
    if (len(cohort_hashes) != 1 or None in cohort_hashes
            or len(runners) != 1 or None in runners
            or len(helpers) != 1 or None in helpers
            or len(revisions) != 1 or None in revisions):
        raise ValueError("cohort_source_or_model_mismatch")
    rows = []
    for item in results:
        if (item.get("status") != "completed_single_video_exploratory_reference_likelihood_pilot"
                or item.get("forward_count") != 3
                or item.get("direct_generated_explanation_faithfulness") is not False
                or item.get("human_review_performed") is not False):
            raise ValueError("result_scope_or_completion_invalid")
        audits = item.get("scoring_audit", {})
        if set(audits) != set(CONDITIONS):
            raise ValueError("scoring_conditions_invalid")
        target_hashes = {audits[name].get("target_token_ids_sha256") for name in CONDITIONS}
        if len(target_hashes) != 1 or None in target_hashes:
            raise ValueError("target_token_binding_invalid")
        hook_ok = True
        for name in CONDITIONS:
            audit = audits[name]
            expected_keys = 0 if name == "baseline" else 121
            hook_ok &= (audit.get("biased_key_count") == expected_keys
                        and audit.get("deleted_key_count") == 0
                        and set(audit.get("layer_hook_calls", {})) == LAYERS
                        and set(audit["layer_hook_calls"].values()) == {1})
        if not hook_ok:
            raise ValueError("attention_intervention_audit_invalid")
        scores = item.get("target_mean_logprob_nats_per_token", {})
        support = scores.get("object_time_support_bias")
        baseline = scores.get("baseline")
        sham = scores.get("equal_dose_temporal_sham_bias")
        if any(type(value) not in (int, float) for value in (support, baseline, sham)):
            raise ValueError("score_values_invalid")
        d_baseline, d_sham = support - baseline, support - sham
        if (abs(d_baseline - item.get("support_minus_baseline", float("inf"))) > 1e-9
                or abs(d_sham - item.get("support_minus_sham", float("inf"))) > 1e-9):
            raise ValueError("stored_contrast_mismatch")
        rows.append({"row_key_sha256": item["row_key_sha256"],
                     "support_minus_baseline_nats_per_token": d_baseline,
                     "support_minus_sham_nats_per_token": d_sham,
                     "scoring_seconds": item["elapsed_scoring_seconds"],
                     "peak_vram_bytes": item["peak_vram_bytes"]})
    if result_hashes is not None and (len(result_hashes) != 3
                                      or any(len(value) != 64 for value in result_hashes)):
        raise ValueError("result_hash_manifest_invalid")
    baseline_diffs = [row["support_minus_baseline_nats_per_token"] for row in rows]
    sham_diffs = [row["support_minus_sham_nats_per_token"] for row in rows]
    return {"schema_version": 1,
            "status": "completed_descriptive_exploratory_analysis",
            "cohort_sha256": next(iter(cohort_hashes)),
            "runner_sha256": next(iter(runners)),
            "helper_sha256": next(iter(helpers)),
            "model_revision": next(iter(revisions)),
            "requested_rows": 3, "effective_rows": len(rows),
            "forward_count": sum(item["forward_count"] for item in results),
            "human_review_performed": False,
            "direct_generated_explanation_faithfulness": False,
            "rows": rows,
            "result_sha256": result_hashes,
            "support_minus_baseline_mean_nats_per_token": statistics.mean(baseline_diffs),
            "support_minus_baseline_range_nats_per_token": [min(baseline_diffs), max(baseline_diffs)],
            "support_minus_sham_mean_nats_per_token": statistics.mean(sham_diffs),
            "support_minus_sham_range_nats_per_token": [min(sham_diffs), max(sham_diffs)],
            "statistical_test": None,
            "interpretation": "descriptive only; n=3, previously frozen exploratory cohort; not generated-explanation faithfulness"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (len(args.result) != 3 or any(path.parent.resolve() != PRIVATE.resolve()
            or not path.is_file() or path.stat().st_mode & 0o077 for path in args.result)
            or args.output.parent.resolve() != PRIVATE.resolve()
            or not PRIVATE.is_dir() or PRIVATE.stat().st_mode & 0o077
            or args.output.exists()):
        raise ValueError("private_result_contract_failed")
    results = [json.loads(path.read_text()) for path in args.result]
    hashes = [sha256_file(path) for path in args.result]
    summary = summarize_results(results, hashes)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({key: summary[key] for key in (
        "status", "effective_rows", "forward_count",
        "support_minus_baseline_mean_nats_per_token",
        "support_minus_sham_mean_nats_per_token")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
