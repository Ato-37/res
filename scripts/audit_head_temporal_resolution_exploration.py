"""Recompute aggregate checks for a private temporal-resolution head scan."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and values[order[k]] == values[order[j]]:
            k += 1
        rank = ((j + 1) + k) / 2.0
        for pos in range(j, k):
            ranks[order[pos]] = rank
        j = k
    return ranks


def spearman_average_ties(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) *
                            sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else None


def check_run(result_path, manifest_path, runner_path, helper_path):
    result_path = Path(result_path)
    data = json.loads(result_path.read_text())
    rows = json.loads(Path(manifest_path).read_text())
    if result_path.stat().st_mode & 0o777 != 0o600:
        raise ValueError("result_file_mode_must_be_0600")
    manifest_hash = sha256(manifest_path)
    if data.get("research_status") != "exploratory_only":
        raise ValueError("not_marked_exploratory_only")
    if data.get("data_manifest_sha256") != manifest_hash:
        raise ValueError("manifest_hash_mismatch")
    if data.get("requested_samples") != len(rows):
        raise ValueError("requested_sample_count_mismatch")
    source_hashes = data.get("sources", {})
    if source_hashes.get("runner_sha256") != sha256(runner_path):
        raise ValueError("runner_hash_mismatch")
    if source_hashes.get("head_helper_sha256") != sha256(helper_path):
        raise ValueError("head_helper_hash_mismatch")

    conditions = {}
    for key, bins in (("8", 4), ("16", 8)):
        c = data["conditions"][key]
        metrics = c["all_head_metrics"]
        if c["effective_samples"] + c["failures"] != data["requested_samples"]:
            raise ValueError(f"sample_accounting_mismatch_{key}")
        if len(metrics) != 28 * 28 or len({(x["layer"], x["head"]) for x in metrics}) != 784:
            raise ValueError(f"head_inventory_mismatch_{key}")
        survivors = [x for x in metrics if x["survives_elbow_and_sink_filter"]]
        expected = sorted(survivors, key=lambda x: (
            x["mean_temporal_entropy_nats"], -x["mean_video_mass"], x["layer"], x["head"]))[:20]
        actual = c["top20"]
        if [(x["layer"], x["head"]) for x in expected] != [(x["layer"], x["head"]) for x in actual]:
            raise ValueError(f"top20_not_reproduced_{key}")
        if c["temporal_bins_observed"] != bins:
            raise ValueError(f"unexpected_temporal_bins_{key}")
        conditions[key] = {"effective_samples": c["effective_samples"],
                           "failed_samples": c["failures"],
                           "elbow_survivors": c["elbow_survivors"],
                           "non_sink_survivors": c["non_sink_survivors"],
                           "top20_n": len(actual),
                           "max_allocated_vram_gib_recorded": c["max_allocated_vram_gib"],
                           "elapsed_seconds_recorded": c["elapsed_seconds"],
                           "sanitized_error_messages": c["sanitized_error_messages"],
                           "top20_pairs": [(x["layer"], x["head"]) for x in actual],
                           "survivor_metrics": {(x["layer"], x["head"]): x
                                                for x in survivors}}
    a, b = conditions["8"], conditions["16"]
    shared = sorted(set(a["survivor_metrics"]) & set(b["survivor_metrics"]))
    rho = spearman_average_ties(
        [a["survivor_metrics"][key]["mean_temporal_entropy_nats"] for key in shared],
        [b["survivor_metrics"][key]["mean_temporal_entropy_nats"] for key in shared])
    top_a, top_b = set(a["top20_pairs"]), set(b["top20_pairs"])
    overlap, union = len(top_a & top_b), len(top_a | top_b)
    return {"artifact": str(result_path), "artifact_sha256": sha256(result_path),
            "manifest_sha256": manifest_hash,
            "runner_sha256": source_hashes["runner_sha256"],
            "helper_sha256": source_hashes["head_helper_sha256"],
            "requested": data["requested_samples"],
            "hardware": data.get("hardware"), "model_revision_recorded": data.get("model_revision"),
            "conditions": {k: {x: y for x, y in v.items() if x != "survivor_metrics"}
                           for k, v in conditions.items()},
            "top20_overlap": overlap, "top20_union": union,
            "top20_jaccard_recomputed": overlap / union if union else None,
            "shared_non_sink_survivors": len(shared),
            "spearman_average_ties_recomputed": rho,
            "interpretation_scope": "descriptive, selected-survivor attention-shape stability only"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--helper", required=True)
    args = p.parse_args()
    print(json.dumps(check_run(args.result, args.manifest, args.runner, args.helper),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
