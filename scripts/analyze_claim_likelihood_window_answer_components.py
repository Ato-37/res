#!/usr/bin/env python3
"""Decompose selected-window QA-margin changes into gold and best-foil logprob shifts."""
from __future__ import annotations
import argparse,hashlib,json,os,stat
from pathlib import Path
from analyze_claim_likelihood_window_gold_overlap import _read_private,PACKETS

CONDITIONS=("full","3b_likelihood_selected_window_black","temporal_control_black")

def decompose(gold_idx,logprobs):
    if type(gold_idx)is not int or not 0<=gold_idx<5:raise ValueError("gold_index_invalid")
    out={}
    for condition in CONDITIONS:
        values=logprobs.get(condition)
        if not isinstance(values,list) or len(values)!=5:raise ValueError("choice_logprob_shape_invalid")
        gold=float(values[gold_idx]);foil=max(float(v) for i,v in enumerate(values) if i!=gold_idx)
        out[condition]={"gold_choice_logprob":gold,"best_foil_logprob":foil,"margin":gold-foil}
    selected=out["3b_likelihood_selected_window_black"];control=out["temporal_control_black"]
    d_gold=control["gold_choice_logprob"]-selected["gold_choice_logprob"]
    d_foil=control["best_foil_logprob"]-selected["best_foil_logprob"]
    out["control_minus_selected_gold_component"]=d_gold
    out["control_minus_selected_best_foil_component"]=d_foil
    out["control_minus_selected_margin_reconstructed"]=d_gold-d_foil
    return out

def analyze(repo,selection_rel,score_rel,output_rel):
    selection,sraw=_read_private(repo,selection_rel);score,qraw=_read_private(repo,score_rel)
    if selection.get("status")!="completed_window_likelihood_selection" or score.get("status")!="completed" or len(selection.get("records",[]))!=3 or len(score.get("records",[]))!=3 or score.get("completed_forwards")!=9:
        raise ValueError("complete_input_receipts_required")
    sh=hashlib.sha256(sraw).hexdigest()
    if score.get("inputs_sha256",{}).get(str(selection_rel))!=sh:raise ValueError("score_selector_input_hash_mismatch")
    packets={};ph={}
    for rel in PACKETS:
        p,raw=_read_private(repo,Path(rel));ph[Path(rel).name]=hashlib.sha256(raw).hexdigest()
        for row in p["records"]:packets[row["video_sha256"]]=row
    rows=[]
    for n,(selected,result) in enumerate(zip(selection["records"],score["records"],strict=True),1):
        media=selected["source_video_sha256"];source=packets[media]
        if (selected["packet_sha256"]!=ph.get(selected["packet_name"])
                or selected["pair_sha256"]!=source["candidate_pair_sha256"]
                or selected["saved_claim_sha256"]!=hashlib.sha256(source["generated_parsed"]["claim"].encode()).hexdigest()
                or result.get("record_position")!=n):raise ValueError("row_provenance_binding_invalid")
        golds=[i for i,v in enumerate(source["choices"]) if v==source["gold_answer"]]
        if len(golds)!=1:raise ValueError("gold_answer_binding_invalid")
        components=decompose(golds[0],result["choice_log_probabilities_nats"])
        for condition in CONDITIONS:
            if abs(components[condition]["margin"]-result["gold_vs_best_foil_margins_nats"][condition])>1e-6:
                raise ValueError("saved_margin_reconstruction_mismatch")
        if abs(components["control_minus_selected_margin_reconstructed"]-result["d_control_minus_3b_likelihood_window_nats"])>1e-6:
            raise ValueError("saved_delta_reconstruction_mismatch")
        rows.append({"row_token_sha256":selected["row_token_sha256"],"components":components})
    payload={"schema_version":1,"status":"completed_exploratory_margin_component_audit",
        "purpose":"gold_and_best_foil_logprob_decomposition_of_primary_margin_contrast",
        "selection_receipt_sha256":sh,"score_receipt_sha256":hashlib.sha256(qraw).hexdigest(),
        "records":rows,"rights_status":"unverified; local-only per user authorization",
        "interpretation_limitations":["decomposition explains the same exploratory margin contrast, not evidence truth or faithfulness",
            "best foil may differ by condition; values are condition-wise maximum non-gold choices"]}
    output=repo/output_rel
    if not output_rel.is_relative_to(Path("data/private_nextgqa")) or output.exists():raise ValueError("private_output_invalid_or_exists")
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"w") as f:json.dump(payload,f,sort_keys=True,separators=(",",":"),allow_nan=False);f.write("\n")
    print(json.dumps({"status":payload["status"],"rows":len(rows)},sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--selection-receipt",type=Path,required=True);p.add_argument("--score-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();analyze(Path(__file__).resolve().parents[1],a.selection_receipt,a.score_receipt,a.output)
if __name__=="__main__":main()
