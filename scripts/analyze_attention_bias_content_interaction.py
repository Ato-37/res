#!/usr/bin/env python3
"""Summarize local one-row receipts for the exploratory bias × content interaction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(paths: list[Path]) -> dict:
    values = []
    excluded = []
    seen = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "completed_one_row_posthoc_attention_bias_content_interaction":
            excluded.append({"name": path.name, "status": data.get("status"),
                             "completed_forwards": data.get("completed_forwards")})
            continue
        if (data.get("direct_faithfulness") is not False
                or data.get("human_review", {}).get("review_actually_performed") is not False
                or data.get("requested_records") != 1
                or data.get("effective_records") != 1
                or data.get("requested_forwards") != 1
                or data.get("attempted_forwards") != 1
                or data.get("completed_forwards") != 1):
            raise ValueError(f"receipt_scope_or_count_invalid:{path.name}")
        token = data.get("row_token_sha256")
        if not isinstance(token, str) or len(token) != 64 or token in seen:
            raise ValueError(f"invalid_or_duplicate_row_token:{path.name}")
        seen.add(token)
        expected = data["primary_descriptive_interaction"]
        scores = data["new_gray_video_biased_score"]
        reused = data["reused_frozen_receipts"]
        original_base = reused["original_baseline_mean"]
        original_bias = reused["original_bias_mean"]
        gray_base = reused["gray_baseline_mean"]
        gray_bias = scores["target_logprob_mean"]
        baseline_gap = original_base - gray_base
        biased_gap = original_bias - gray_bias
        interaction = biased_gap - baseline_gap
        for key, value in (("baseline_original_minus_gray_nats_per_token", baseline_gap),
                           ("biased_original_minus_gray_nats_per_token", biased_gap),
                           ("bias_by_video_content_interaction_nats_per_token", interaction)):
            if not math.isclose(float(expected[key]), value, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"interaction_arithmetic_mismatch:{path.name}:{key}")
        values.append({"name": path.name, "receipt_sha256": sha256_file(path),
                       "row_token_sha256": token,
                       "baseline_original_minus_gray": baseline_gap,
                       "biased_original_minus_gray": biased_gap,
                       "interaction": interaction})
    if not values:
        raise ValueError("no_completed_receipts")
    xs = [item["interaction"] for item in values]
    ordered = sorted(xs)
    n = len(xs)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    mean = sum(xs) / n
    return {"schema_version": 1, "purpose": "exploratory_posthoc_attention_bias_video_content_interaction",
            "completed_rows": n, "completed_forwards": n, "excluded_receipts": excluded,
            "rows": values, "interaction_mean": mean, "interaction_median": median,
            "interaction_min": min(xs), "interaction_max": max(xs),
            "baseline_gap_mean": sum(x["baseline_original_minus_gray"] for x in values) / n,
            "biased_gap_mean": sum(x["biased_original_minus_gray"] for x in values) / n,
            "all_interactions_positive": all(x > 0 for x in xs),
            "inferential_status": "finite selected-cohort description only; no CI or test",
            "direct_faithfulness": False, "human_review_performed": False,
            "rights_status": "unverified; private local-only data"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.receipts), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
