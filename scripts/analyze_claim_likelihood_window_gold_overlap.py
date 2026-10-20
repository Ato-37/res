#!/usr/bin/env python3
"""Local exploratory audit of selected chunks against NExT-GQA QA time intervals.

This is an annotation-overlap proxy, not a claim-truth or explanation-faithfulness
test. Inputs and row-level derived values remain in a private mode-0600 receipt.
"""
from __future__ import annotations
import argparse, hashlib, json, os, stat
from pathlib import Path
from claim_likelihood_window_ranking import partition_window_indices

PACKETS=("data/private_nextgqa/claim_generation_window_test_smoke_20261021_03.json",
         "data/private_nextgqa/claim_generation_window_test_remaining_20261021_03.json")

def union_intervals(intervals):
    values=[]
    for pair in intervals:
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2):raise ValueError("interval_invalid")
        a,b=map(float,pair)
        if not (a>=0 and b>=a):raise ValueError("interval_bounds_invalid")
        if b>a:values.append((a,b))
    values.sort();merged=[]
    for a,b in values:
        if not merged or a>merged[-1][1]:merged.append([a,b])
        else:merged[-1][1]=max(merged[-1][1],b)
    return tuple((a,b) for a,b in merged)

def overlap_metrics(candidate, annotation_intervals):
    start,end=map(float,candidate)
    if start<0 or end<=start:raise ValueError("candidate_interval_invalid")
    gold=union_intervals(annotation_intervals)
    candidate_len=end-start;gold_len=sum(b-a for a,b in gold)
    intersection=sum(max(0.0,min(end,b)-max(start,a)) for a,b in gold)
    union=candidate_len+gold_len-intersection
    return {"intersection_over_candidate":intersection/candidate_len,
            "intersection_over_annotation":intersection/gold_len if gold_len else None,
            "temporal_iou":intersection/union if union else 0.0,
            "intersection_seconds":intersection}

def _read_private(repo,relative):
    rel=Path(relative)
    if rel.is_absolute() or not rel.is_relative_to(Path("data/private_nextgqa")):
        raise ValueError("private_input_path_invalid")
    path=(repo/rel).resolve(strict=True);root=(repo/"data/private_nextgqa").resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError("private_input_path_invalid")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError("private_input_permissions_invalid")
        chunks=[]
        while True:
            b=os.read(fd,1<<20)
            if not b:break
            chunks.append(b)
    finally:os.close(fd)
    raw=b"".join(chunks);return json.loads(raw.decode()),raw

def analyze(repo,selection_rel,output_rel):
    selected,selected_raw=_read_private(repo,selection_rel)
    if selected.get("status")!="completed_window_likelihood_selection" or len(selected.get("records",[]))!=3 or selected.get("completed_windows")!=12:
        raise ValueError("selector_receipt_incomplete")
    packets={};packet_hashes={}
    for rel in PACKETS:
        packet,raw=_read_private(repo,Path(rel));packet_hashes[Path(rel).name]=hashlib.sha256(raw).hexdigest()
        if packet.get("status")!="passed" or packet.get("dataset_split")!="test":raise ValueError("claim_packet_invalid")
        for row in packet["records"]:
            if row["video_sha256"] in packets:raise ValueError("duplicate_packet_media")
            packets[row["video_sha256"]]=row
    rows=[]
    for rec in selected["records"]:
        media=rec.get("source_video_sha256");source=packets.get(media)
        if (source is None or rec.get("packet_sha256")!=packet_hashes.get(rec.get("packet_name"))
                or rec.get("pair_sha256")!=source.get("candidate_pair_sha256")
                or rec.get("saved_claim_sha256")!=hashlib.sha256(source["generated_parsed"]["claim"].encode()).hexdigest()
                or rec.get("row_token_sha256")!=hashlib.sha256((media+str(source["dataset_question_id"])).encode()).hexdigest()):
            raise ValueError("selector_packet_binding_invalid")
        pts=rec.get("sampled_pts_seconds");chosen=rec.get("selected_window_index")
        if len(pts)!=32 or type(chosen)is not int or not 0<=chosen<4:raise ValueError("sampled_pts_contract_invalid")
        ann=source.get("gold_interval_seconds")
        if not isinstance(ann,list) or not ann:raise ValueError("gold_intervals_missing")
        windows=partition_window_indices(32,4)
        scores=[]
        for idxs in windows:
            scores.append(overlap_metrics((pts[idxs[0]],pts[idxs[-1]]),ann))
        rows.append({"row_token_sha256":rec["row_token_sha256"],"selected_window_index":chosen,
            "selected_window_metrics":scores[chosen],"all_window_temporal_iou":[x["temporal_iou"] for x in scores],
            "selected_iou_rank_among_four":1+sum(x["temporal_iou"]>scores[chosen]["temporal_iou"] for x in scores),
            "annotation_interval_count":len(ann)})
    if len({r["row_token_sha256"] for r in rows})!=3:raise ValueError("unique_row_count_invalid")
    selected_ious=[r["selected_window_metrics"]["temporal_iou"] for r in rows]
    all_iou=[x for r in rows for x in r["all_window_temporal_iou"]]
    source_paths=[Path(__file__),Path(__file__).with_name("claim_likelihood_window_ranking.py")]
    payload={"schema_version":1,"status":"completed_exploratory_overlap_audit",
        "purpose":"posthoc_selected_window_vs_benchmark_qa_interval_overlap",
        "cohort_lock_sha256":selected["cohort_lock_sha256"],"selection_receipt_sha256":hashlib.sha256(selected_raw).hexdigest(),
        "claim_packet_sha256":packet_hashes,"dataset_split":"test","records":rows,
        "selected_window_temporal_iou_mean":sum(selected_ious)/3,
        "all_four_windows_temporal_iou_mean":sum(all_iou)/12,
        "selected_window_top_iou_count":sum(r["selected_iou_rank_among_four"]==1 for r in rows),
        "interpretation_limitations":["dataset QA intervals target benchmark QA, not reviewed claim evidence",
            "post-hoc n=3 descriptive overlap only; no inference","selected chunk was chosen by claim likelihood, not interval overlap",
            "overlap does not validate claim truth or causal model dependence"],
        "rights_status":"unverified; local-only per user authorization",
        "source_sha256":{p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    output=repo/output_rel
    if not output_rel.is_relative_to(Path("data/private_nextgqa")) or output.exists():raise ValueError("private_output_invalid_or_exists")
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as f:json.dump(payload,f,sort_keys=True,separators=(",",":"),allow_nan=False);f.write("\n")
    print(json.dumps({k:payload[k] for k in ("status","selected_window_temporal_iou_mean","all_four_windows_temporal_iou_mean","selected_window_top_iou_count")},sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--selection-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];analyze(repo,a.selection_receipt,a.output)
if __name__=="__main__":main()
