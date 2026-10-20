#!/usr/bin/env python3
"""Summarize the fixed three-row post-result frame-mean control."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path

DEFAULT=("data/private_nextgqa/saved_claim_frame_mean_control_20261021_smoke_01.json",
"data/private_nextgqa/saved_claim_frame_mean_control_20261021_remaining_00.json",
"data/private_nextgqa/saved_claim_frame_mean_control_20261021_remaining_01.json")

def analyze(paths):
    if len(paths)!=3: raise ValueError("expected_exactly_three_fixed_receipts")
    seen=set();values=[]
    for path in paths:
        r=json.loads(Path(path).read_text(encoding="utf-8"))
        if (r.get("status")!="completed_one_row_posthoc_frame_mean_sensitivity"
                or r.get("post_result_exploration") is not True
                or r.get("direct_faithfulness") is not False
                or r.get("human_review",{}).get("review_actually_performed") is not False
                or (r.get("requested_forwards"),r.get("attempted_forwards"),r.get("completed_forwards"))!=(2,2,2)):
            raise ValueError("receipt_not_complete_posthoc_row")
        token=r.get("row_token_sha256")
        if not isinstance(token,str) or len(token)!=64 or token in seen:
            raise ValueError("row_token_invalid_or_duplicated")
        seen.add(token);branches=r.get("branch_target_scores",{})
        if set(branches)!={"original_video","per_frame_rgb_mean_video"}:
            raise ValueError("branch_set_invalid")
        original=branches["original_video"].get("target_logprob_mean")
        mean=branches["per_frame_rgb_mean_video"].get("target_logprob_mean")
        difference=r.get("primary_descriptive_contrast_original_minus_frame_mean_nats_per_token")
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in (original,mean,difference)):
            raise ValueError("branch_score_nonfinite")
        if not math.isclose(float(original)-float(mean),float(difference),rel_tol=0,abs_tol=1e-12):
            raise ValueError("contrast_arithmetic_mismatch")
        values.append(float(difference))
    return {"requested_rows":3,"effective_rows":3,"completed_forwards":6,
        "row_contrasts_original_minus_frame_mean_nats_per_token":values,
        "finite_cohort_mean_nats_per_token":statistics.mean(values),
        "finite_cohort_median_nats_per_token":statistics.median(values),
        "finite_cohort_range_nats_per_token":[min(values),max(values)],
        "positive_rows":sum(v>0 for v in values),
        "inferential_status":"post-result exploratory selected development cohort; no population inference",
        "direct_faithfulness":False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("receipts",nargs="*",type=Path,default=[Path(x) for x in DEFAULT])
    print(json.dumps(analyze(tuple(parser.parse_args().receipts)),indent=2,allow_nan=False));return 0

if __name__=="__main__":raise SystemExit(main())
