#!/usr/bin/env python3
"""Audit and summarize private receipts for claim-likelihood × QA interval masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict:
    rows=[]; excluded=[]; seen=set()
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "completed_one_row_exploratory_qa_interval_claim_likelihood_interaction":
            excluded.append({"name":path.name,"status":data.get("status"),
                "attempted_forwards":data.get("attempted_forwards"),
                "completed_forwards":data.get("completed_forwards")})
            continue
        if (data.get("direct_faithfulness") is not False
                or data.get("human_review",{}).get("review_actually_performed") is not False
                or data.get("requested_records")!=1 or data.get("effective_records")!=1
                or data.get("requested_forwards")!=4 or data.get("attempted_forwards")!=4
                or data.get("completed_forwards")!=4):
            raise ValueError(f"receipt_scope_or_count_invalid:{path.name}")
        token=data.get("row_token_sha256")
        if not isinstance(token,str) or len(token)!=64 or token in seen:
            raise ValueError(f"invalid_or_duplicate_row_token:{path.name}")
        seen.add(token)
        if data.get("qa_interval_frame_count")!=data.get("control_frame_count") or data["qa_interval_frame_count"]<1:
            raise ValueError(f"frame_count_pair_invalid:{path.name}")
        scores=data.get("condition_scores",{})
        expected_names={"qa_interval_baseline","control_baseline","qa_interval_biased","control_biased"}
        if set(scores)!=expected_names:
            raise ValueError(f"condition_set_invalid:{path.name}")
        for name,biased in (("qa_interval_baseline",False),("control_baseline",False),
                             ("qa_interval_biased",True),("control_biased",True)):
            score=scores[name]
            if (score.get("score_semantics")!="mean conditional logprob of frozen target token IDs"
                    or score.get("target_token_count")!=data.get("target_token_count")
                    or score.get("visual_key_position_count",0)<1):
                raise ValueError(f"score_binding_invalid:{path.name}:{name}")
            expected_hook=1 if biased else 0
            hooks=score.get("intervention_hook_invocations_by_layer_1_based")
            expected={str(layer):expected_hook for layer in range(5,17)}
            if hooks!=expected:
                raise ValueError(f"hook_audit_invalid:{path.name}:{name}")
        qib=scores["qa_interval_baseline"]["target_logprob_mean"]
        cb=scores["control_baseline"]["target_logprob_mean"]
        qiv=scores["qa_interval_biased"]["target_logprob_mean"]
        cv=scores["control_biased"]["target_logprob_mean"]
        base=cb-qib; biased=cv-qiv; did=biased-base
        stored=data.get("primary_descriptive_interaction",{})
        for key,value in (("baseline_control_minus_qa_interval_nats_per_token",base),
            ("biased_control_minus_qa_interval_nats_per_token",biased),
            ("bias_by_qa_interval_interaction_nats_per_token",did)):
            if not math.isclose(float(stored.get(key,float("nan"))),value,rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"interaction_arithmetic_mismatch:{path.name}:{key}")
        rows.append({"name":path.name,"receipt_sha256":sha256_file(path),
            "row_token_sha256":token,"qa_interval_frame_count":data["qa_interval_frame_count"],
            "claim_tokens":data["target_token_count"],"baseline_control_minus_interval":base,
            "biased_control_minus_interval":biased,"interaction":did})
    if not rows: raise ValueError("no_completed_receipts")
    values=[r["interaction"] for r in rows]; ordered=sorted(values); n=len(values)
    median=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    return {"schema_version":1,"purpose":"post_result_exploratory_claim_likelihood_qa_interval_interaction",
        "completed_rows":n,"completed_forwards":sum(4 for _ in rows),"excluded_receipts":excluded,
        "rows":rows,"interaction_mean":sum(values)/n,"interaction_median":median,
        "interaction_min":min(values),"interaction_max":max(values),
        "positive_interaction_rows":sum(x>0 for x in values),
        "inferential_status":"finite selected n=3 cohort only; no CI or test",
        "direct_faithfulness":False,"human_review_performed":False,
        "rights_status":"unverified; private local-only data"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("receipts",nargs="+",type=Path)
    args=parser.parse_args();print(json.dumps(summarize(args.receipts),sort_keys=True,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
