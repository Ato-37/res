#!/usr/bin/env python3
"""Summarize the fixed three-row post-result frame-mean control."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
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
                or r.get("dataset_split") not in (None,"test","val")
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

def audit_provenance(receipt_paths, lock_path, selected_filter_path,
                     packet_paths, batch_filter_paths):
    """Bind each private scoring receipt to its packet row and frozen cohort."""
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    lock_raw=Path(lock_path).read_bytes(); lock=json.loads(lock_raw)
    if lock.get("dataset_split")!="test" or lock.get("status")!="frozen_before_claim_generation_and_scoring":
        raise ValueError("cohort_lock_contract_invalid")
    selected_raw=Path(selected_filter_path).read_bytes()
    selected=json.loads(selected_raw)
    if digest(selected_raw)!=lock.get("selected_pair_filter_sha256"):
        raise ValueError("selected_filter_hash_mismatch")
    if len(selected.get("pair_sha256",[]))!=3 or len(set(selected["pair_sha256"]))!=3:
        raise ValueError("selected_filter_entries_invalid")
    locked={(x["media_sha256"],x["qid_sha256"]) for x in lock.get("assignments",[])}
    if len(locked)!=3:
        raise ValueError("cohort_lock_assignments_invalid")
    seen=set();checks=[]
    for receipt_path in receipt_paths:
        receipt=json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        packet_name=receipt.get("private_input_packet_name")
        if packet_name not in packet_paths or packet_name not in batch_filter_paths:
            raise ValueError("receipt_packet_mapping_missing")
        packet_path=Path(packet_paths[packet_name]); packet_raw=packet_path.read_bytes()
        packet=json.loads(packet_raw); index=receipt.get("packet_record_index_zero_based")
        if (digest(packet_raw)!=receipt.get("private_input_packet_sha256")
                or packet.get("status")!="passed" or packet.get("dataset_split")!="test"
                or type(index)is not int or index<0 or index>=len(packet.get("records",[]))):
            raise ValueError("receipt_packet_binding_invalid")
        row=packet["records"][index]
        batch_path=Path(batch_filter_paths[packet_name]); batch_raw=batch_path.read_bytes()
        batch=json.loads(batch_raw)
        if (digest(batch_raw)!=packet.get("candidate_pair_filter_sha256")
                or row.get("candidate_pair_sha256") not in batch.get("pair_sha256",[])
                or row.get("candidate_pair_sha256") not in selected.get("pair_sha256",[])):
            raise ValueError("receipt_pair_filter_binding_invalid")
        media=row.get("video_sha256"); qid=row.get("dataset_question_id")
        qid_hash=hashlib.sha256(str(qid).encode("utf-8")).hexdigest()
        token=hashlib.sha256((media+str(qid)).encode("utf-8")).hexdigest()
        claim=row.get("generated_parsed",{}).get("claim","")
        video_path=Path(row.get("local_video_path",""))
        if ((media,qid_hash) not in locked or token!=receipt.get("row_token_sha256")
                or digest(claim.encode("utf-8"))!=receipt.get("saved_claim_text_sha256")
                or media!=receipt.get("source_video_sha256")
                or not video_path.is_file() or digest(video_path.read_bytes())!=media
                or receipt.get("human_review",{}).get("review_actually_performed") is not False):
            raise ValueError("receipt_row_lock_binding_invalid")
        if token in seen: raise ValueError("receipt_row_token_duplicated")
        seen.add(token)
        source_hashes=receipt.get("source_sha256")
        if not isinstance(source_hashes,dict) or not source_hashes:
            raise ValueError("receipt_source_hashes_missing")
        for relative,expected in source_hashes.items():
            source=Path(__file__).resolve().parents[1]/relative
            if not source.is_file() or digest(source.read_bytes())!=expected:
                raise ValueError("receipt_source_hash_mismatch")
        checks.append({"receipt":"bound","source_hashes_verified":len(source_hashes)})
    packet_pairs=set()
    for path in packet_paths.values():
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        packet_pairs.update((row["video_sha256"],hashlib.sha256(
            str(row["dataset_question_id"]).encode("utf-8")).hexdigest())
            for row in payload.get("records",[]))
    if len(seen)!=3 or locked!=packet_pairs:
        raise ValueError("receipt_set_does_not_cover_exact_locked_cohort")
    return {"status":"provenance_verified","receipts_verified":len(checks),
            "locked_rows_verified":len(locked),"batch_filters_verified":len(batch_filter_paths),
            "source_hash_checks":sum(x["source_hashes_verified"] for x in checks),
            "private_inputs_remain_local":True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("receipts",nargs="*",type=Path,default=[Path(x) for x in DEFAULT])
    parser.add_argument("--lock",type=Path);parser.add_argument("--selected-filter",type=Path)
    parser.add_argument("--smoke-packet",type=Path);parser.add_argument("--remaining-packet",type=Path)
    parser.add_argument("--smoke-filter",type=Path);parser.add_argument("--remaining-filter",type=Path)
    args=parser.parse_args();result=analyze(tuple(args.receipts))
    audit_args=(args.lock,args.selected_filter,args.smoke_packet,args.remaining_packet,args.smoke_filter,args.remaining_filter)
    if any(x is not None for x in audit_args):
        if not all(x is not None for x in audit_args): parser.error("all_provenance_paths_are_required_together")
        packets={args.smoke_packet.name:args.smoke_packet,args.remaining_packet.name:args.remaining_packet}
        filters={args.smoke_packet.name:args.smoke_filter,args.remaining_packet.name:args.remaining_filter}
        result["provenance_audit"]=audit_provenance(args.receipts,args.lock,args.selected_filter,packets,filters)
    print(json.dumps(result,indent=2,allow_nan=False));return 0

if __name__=="__main__":raise SystemExit(main())
