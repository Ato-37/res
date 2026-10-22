"""Deterministic, exploratory bootstrap of attention-shape candidate stability."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from run_head_bootstrap_exploration import summarize, sha256, media_inventory
from head_selection_unsupervised import elbow_index
from run_head_bootstrap_exploration import MODEL_REVISION


def sink_fractions(peak_rows):
    n_rows = peak_rows.shape[0]
    sink = np.zeros(peak_rows.shape[1:], dtype=np.float64)
    for li in range(sink.shape[0]):
        for hi in range(sink.shape[1]):
            sink[li, hi] = np.bincount(peak_rows[:, li, hi].astype(int)).max() / n_rows
    return sink


def bootstrap_summary(mass_rows, entropy_rows, peak_rows, group_ids, temporal_bins=8,
                      replicates=1000, seed=20261019):
    if mass_rows.shape != entropy_rows.shape or mass_rows.shape != peak_rows.shape:
        raise ValueError("metric_array_shape_mismatch")
    if mass_rows.ndim != 3 or mass_rows.shape[1:] != (28, 28):
        raise ValueError("expected_[n,28,28]_arrays")
    n = mass_rows.shape[0]
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (n,) or not np.issubdtype(group_ids.dtype, np.integer):
        raise ValueError("invalid_media_group_ids")
    if n < 2 or len(np.unique(group_ids)) < 2 or replicates < 1:
        raise ValueError("insufficient_bootstrap_input")
    if not (np.isfinite(mass_rows).all() and np.isfinite(entropy_rows).all()):
        raise ValueError("nonfinite_attention_metrics")
    if (mass_rows < 0).any() or (mass_rows > 1.0001).any():
        raise ValueError("attention_mass_out_of_range")
    if (entropy_rows < 0).any() or (entropy_rows > np.log(8) + 1e-4).any():
        raise ValueError("temporal_entropy_out_of_range")
    if temporal_bins not in (4, 8) or (peak_rows >= temporal_bins).any():
        raise ValueError("peak_bin_out_of_range")
    if (entropy_rows > np.log(temporal_bins) + 1e-4).any():
        raise ValueError("temporal_entropy_out_of_range")
    rng = np.random.default_rng(seed)
    counts = np.zeros((28, 28), dtype=np.int32)
    rank_sum = np.zeros((28, 28), dtype=np.float64)
    rank_counts = np.zeros((28, 28), dtype=np.int32)
    survivor_counts = np.zeros((28, 28), dtype=np.int32)
    eligible_counts = []
    groups = np.unique(group_ids)
    rows_by_group = {group: np.flatnonzero(group_ids == group) for group in groups}
    for _ in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        ix = np.concatenate([rows_by_group[group] for group in sampled_groups])
        ranked, _, eligible_n = summarize(mass_rows[ix], entropy_rows[ix], peak_rows[ix])
        eligible_counts.append(eligible_n)
        for rank, row in enumerate(ranked, 1):
            li, hi = row[0], row[1]
            counts[li, hi] += 1
            rank_sum[li, hi] += rank
            rank_counts[li, hi] += 1
        # Reconstruct all non-sink survivors to measure candidate-pool inclusion.
        mass = mass_rows[ix].mean(axis=0)
        entropy = entropy_rows[ix].mean(axis=0)
        sink = sink_fractions(peak_rows[ix])
        flat = [(li, hi, float(mass[li, hi])) for li in range(28) for hi in range(28)]
        flat.sort(key=lambda x: (-x[2], x[0], x[1]))
        # Same chord/elbow rule and sink threshold as the runner.
        keep = elbow_index([x[2] for x in flat], min_keep=10)
        for li, hi, _ in flat[:keep]:
            if sink[li, hi] < 0.5:
                survivor_counts[li, hi] += 1
    candidates = []
    for li in range(28):
        for hi in range(28):
            if counts[li, hi]:
                candidates.append({"layer": li, "head": hi,
                                   "top20_selection_frequency": counts[li, hi] / replicates,
                                   "mean_rank_when_selected": rank_sum[li, hi] / rank_counts[li, hi],
                                   "eligible_survivor_frequency": survivor_counts[li, hi] / replicates})
    candidates.sort(key=lambda x: (-x["top20_selection_frequency"],
                                   x["mean_rank_when_selected"], x["layer"], x["head"]))
    return {"replicates": replicates, "seed": seed, "sample_count": n,
            "unique_media_clusters": len(groups),
            "resampling_unit": "exact-media-content cluster sampled with replacement; assumes distinct media clusters are IID",
            "median_eligible_candidate_count": float(np.median(eligible_counts)),
            "top20_selection_frequency": candidates[:40],
            "interpretation": "bootstrap resampling stability on this exploratory cohort; no population CI or faithfulness claim"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--exploratory", action="store_true", required=True)
    args = p.parse_args()
    summary_path, metrics_path, output_path = map(Path, (args.summary, args.metrics, args.output))
    if output_path.exists():
        p.error("refusing_to_overwrite_output")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "head_bootstrap_exploration_v1" or summary.get("research_status") != "exploratory_only":
        p.error("summary_not_exploratory")
    runner_path = Path(__file__).with_name("run_head_bootstrap_exploration.py")
    helper_path = Path(__file__).with_name("head_selection_unsupervised.py")
    if summary.get("runner_sha256") != sha256(runner_path) or summary.get("helper_sha256") != sha256(helper_path):
        p.error("source_hash_mismatch")
    if sha256(metrics_path) != summary.get("metrics_sha256"):
        p.error("metrics_hash_mismatch")
    if (summary.get("metrics_artifact") != metrics_path.name or
            summary.get("data_manifest_sha256") != sha256(args.manifest)):
        p.error("artifact_manifest_binding_mismatch")
    if (summary.get("model_revision") != MODEL_REVISION or
            summary.get("model_loaded_revision") != MODEL_REVISION or
            len(MODEL_REVISION) != 40):
        p.error("model_revision_mismatch")
    with np.load(metrics_path, allow_pickle=False) as data:
        sample_index = data["sample_index"]
        group_ids = data["media_group_id"]
        failed_index = data["failed_sample_index"]
        arrays = [data["attention_mass"], data["temporal_entropy"], data["peak_bin"]]
    n = summary.get("requested_samples")
    source_rows = json.loads(Path(args.manifest).read_text())
    if summary.get("source_manifest_rows") != len(source_rows):
        p.error("source_manifest_row_count_mismatch")
    limit = summary.get("sample_limit", 0)
    cohort = source_rows[:limit] if limit else source_rows
    if len(cohort) != n:
        p.error("cohort_definition_mismatch")
    for row in cohort:
        row["video"] = os.path.expanduser(row["video"])
    _, all_groups, media_commitment = media_inventory(cohort)
    if (summary.get("ordered_media_content_commitment_sha256") != media_commitment or
            summary.get("unique_media_count") != len(np.unique(all_groups))):
        p.error("media_identity_binding_mismatch")
    expected = np.arange(n, dtype=np.int32)
    sample_index = np.asarray(sample_index)
    failed_index = np.asarray(failed_index)
    if (sample_index.shape != (summary.get("effective_samples"),) or
            failed_index.shape != (summary.get("failed_sample_count"),) or
            len(np.unique(sample_index)) != len(sample_index) or
            len(np.unique(failed_index)) != len(failed_index) or
            set(sample_index.tolist()) & set(failed_index.tolist()) or
            not np.array_equal(np.sort(np.concatenate([sample_index, failed_index])), expected)):
        p.error("sample_index_accounting_mismatch")
    if any(array.shape != (len(sample_index), 28, 28) for array in arrays):
        p.error("metric_array_shape_mismatch")
    if len(group_ids) != len(sample_index):
        p.error("media_group_count_mismatch")
    if not np.array_equal(all_groups[sample_index], group_ids):
        p.error("sample_media_group_mapping_mismatch")
    if int(sum(summary.get("failures", {}).values())) != len(failed_index):
        p.error("failed_sample_count_mismatch")
    if hashlib.sha256(sample_index.astype(np.int32).tobytes()).hexdigest() != summary.get("sample_index_sha256"):
        p.error("sample_index_hash_mismatch")
    if hashlib.sha256(failed_index.astype(np.int32).tobytes()).hexdigest() != summary.get("failed_sample_index_sha256"):
        p.error("failed_sample_index_hash_mismatch")
    result = bootstrap_summary(*arrays, group_ids, temporal_bins=summary.get("temporal_bins"),
                              replicates=args.replicates, seed=args.seed)
    if result["unique_media_clusters"] != summary.get("effective_unique_media_count"):
        p.error("unique_media_cluster_count_mismatch")
    result.update({"source_summary_sha256": sha256(summary_path),
                   "metrics_sha256": sha256(metrics_path),
                   "sample_index_sha256": hashlib.sha256(sample_index.tobytes()).hexdigest(),
                   "runner_sha256": summary["runner_sha256"],
                   "analysis_sha256": sha256(Path(__file__))})
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_path.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("private_output_directory_must_be_0700")
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"replicates": args.replicates, "samples": result["sample_count"],
                      "top_candidates": result["top20_selection_frequency"][:5],
                      "output": str(output_path)}))


if __name__ == "__main__":
    main()
