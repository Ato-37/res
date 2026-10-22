#!/usr/bin/env python3
"""Summarize fixed-row private video-content sensitivity receipts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


DEFAULT_RECEIPTS = (
    "data/private_nextgqa/saved_claim_video_content_dependence_20261021_smoke_01.json",
    "data/private_nextgqa/saved_claim_video_content_dependence_20261021_remaining_00.json",
    "data/private_nextgqa/saved_claim_video_content_dependence_20261021_remaining_01.json",
)


def analyze(paths: tuple[Path, ...]) -> dict:
    if len(paths) != 3:
        raise ValueError("expected_exactly_three_fixed_receipts")
    rows = []
    seen = set()
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (receipt.get("status") != "completed_one_row_exploratory_video_content_sensitivity"
                or receipt.get("direct_faithfulness") is not False
                or receipt.get("faithfulness_result") is not False
                or receipt.get("human_review", {}).get("review_actually_performed") is not False
                or (receipt.get("requested_forwards"), receipt.get("attempted_forwards"),
                    receipt.get("completed_forwards")) != (2, 2, 2)):
            raise ValueError("receipt_not_a_complete_exploratory_row")
        token = receipt.get("row_token_sha256")
        if not isinstance(token, str) or len(token) != 64 or token in seen:
            raise ValueError("row_token_invalid_or_duplicated")
        seen.add(token)
        branches = receipt.get("branch_target_scores")
        if not isinstance(branches, dict) or set(branches) != {"original_video", "constant_gray_video"}:
            raise ValueError("branch_set_invalid")
        original = branches["original_video"].get("target_logprob_mean")
        gray = branches["constant_gray_video"].get("target_logprob_mean")
        contrast = receipt.get("primary_descriptive_contrast_original_minus_gray_nats_per_token")
        values = (original, gray, contrast)
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(float(x)) for x in values):
            raise ValueError("branch_score_nonfinite")
        if not math.isclose(float(original) - float(gray), float(contrast),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("stored_contrast_arithmetic_mismatch")
        rows.append(float(contrast))
    return {
        "requested_rows": 3,
        "effective_rows": 3,
        "completed_forwards": 6,
        "row_contrasts_original_minus_gray_nats_per_token": rows,
        "finite_cohort_mean_nats_per_token": statistics.mean(rows),
        "finite_cohort_median_nats_per_token": statistics.median(rows),
        "finite_cohort_range_nats_per_token": [min(rows), max(rows)],
        "positive_rows": sum(value > 0 for value in rows),
        "inferential_status": "finite selected development cohort; no population interval or significance test",
        "direct_faithfulness": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", type=Path,
                        default=[Path(value) for value in DEFAULT_RECEIPTS])
    args = parser.parse_args()
    print(json.dumps(analyze(tuple(args.receipts)), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
