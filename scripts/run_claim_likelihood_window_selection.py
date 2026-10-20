#!/usr/bin/env python3
"""Select one equal sampled-frame chunk by 3B teacher-forced saved-claim likelihood."""
from __future__ import annotations
import argparse,hashlib,json,os,stat,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path

from private_output_paths import validate_private_output
from claim_likelihood_window_ranking import partition_window_indices,select_max_mean_logprob,selected_interval
from run_saved_claim_frame_permutation_attention_smoke import (
    _build_prompt,_score_branch,_sha256_bytes,_sha256_file,validate_video_metadata,
)
from run_nextgqa_gold_interval_margin_pilot import candidate_pair_sha256, video_key
from run_nextgqa_gold_interval_margin_pilot import _hash as _canonical_hash

LOCALIZER_REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"
SCORER_REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"
MIN_FREE_VRAM=12*1024**3;MIN_HOST_AVAILABLE=8*1024**3;MIN_FORWARD_VRAM=2*1024**3
WINDOW_FRAMES=8;WINDOW_COUNT=4;DECODE_FRAMES=32

def _read_private_json(repo:Path, relative:Path):
    if relative.is_absolute() or not relative.is_relative_to(Path("data/private_nextgqa")):
        raise ValueError("private_input_path_invalid")
    path=(repo/relative).resolve(strict=True);root=(repo/"data/private_nextgqa").resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError("private_input_path_invalid")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError("private_input_permissions_invalid")
        parts=[]
        while True:
            block=os.read(fd,1024*1024)
            if not block:break
            parts.append(block)
    finally:os.close(fd)
    raw=b"".join(parts);return json.loads(raw.decode("utf-8")),raw

def _read_selected_inputs(repo,lock_rel,selected_filter_rel,packet_rels,batch_filter_rels,indices):
    lock,lock_raw=_read_private_json(repo,lock_rel);selected,selected_raw=_read_private_json(repo,selected_filter_rel)
    if (lock.get("status")!="frozen_before_claim_generation_and_scoring"
            or lock.get("dataset_split")!="test"
            or lock.get("requested_videos")!=3 or lock.get("effective_videos")!=3
            or lock.get("purpose")!="exploratory_claim_likelihood_selected_window_prediction_dependence"
            or hashlib.sha256(selected_raw).hexdigest()!=lock.get("selected_pair_filter_sha256")
            or len(selected.get("pair_sha256",[]))!=3
            or len(set(selected.get("pair_sha256",[])))!=3):
        raise ValueError("frozen_cohort_contract_invalid")
    for relative,expected in lock.get("source_sha256",{}).items():
        source=repo/relative
        if not source.is_file() or _sha256_file(source)!=expected:
            raise ValueError("frozen_lock_source_hash_mismatch")
    locked={(x["media_sha256"],x["qid_sha256"]) for x in lock.get("assignments",[])}
    if len(locked)!=3:raise ValueError("frozen_assignments_invalid")
    if not(len(packet_rels)==len(batch_filter_rels)==len(indices)):
        raise ValueError("packet_filter_index_count_mismatch")
    records=[];bindings=[];seen=set()
    for packet_rel,filter_rel,index in zip(packet_rels,batch_filter_rels,indices,strict=True):
        packet,raw=_read_private_json(repo,packet_rel);batch,batch_raw=_read_private_json(repo,filter_rel)
        if (packet.get("status")!="passed" or packet.get("dataset_split")!="test"
                or packet.get("model_revision")!=SCORER_REVISION
                or packet.get("runner_sha256") != _sha256_file(repo/"scripts/generate_nextgqa_claim_review_pilot.py")
                or packet.get("pilot_runner_sha256") != lock.get("source_sha256",{}).get("scripts/run_nextgqa_gold_interval_margin_pilot.py")
                or packet.get("candidate_pair_filter_digest") != _canonical_hash(sorted(batch.get("pair_sha256",[])))
                or packet.get("claim_review_completed") is not False
                or type(index)is not int or not 0<=index<len(packet.get("records",[]))
                or hashlib.sha256(batch_raw).hexdigest()!=packet.get("candidate_pair_filter_sha256")):
            raise ValueError("claim_packet_or_filter_contract_invalid")
        row=packet["records"][index];claim=row.get("generated_parsed",{}).get("claim")
        media=row.get("video_sha256");qid=row.get("dataset_question_id")
        qid_hash=hashlib.sha256(str(qid).encode("utf-8")).hexdigest()
        pair=row.get("candidate_pair_sha256")
        dataset=repo/"NExT-GQA"/"datasets"/"nextgqa"
        video_map=json.loads((dataset/"map_vid_vidorID.json").read_text(encoding="utf-8"))
        video_path=Path(row["local_video_path"]).resolve(strict=True)
        suffix=video_key("/".join(video_path.parts[-2:]))
        video_ids=[str(vid) for vid, mapped in video_map.items() if video_key(mapped)==suffix]
        pair_ids=[vid for vid in video_ids if candidate_pair_sha256(vid,str(qid))==pair]
        if (not isinstance(claim,str) or not claim.strip() or pair not in batch.get("pair_sha256",[])
                or pair not in selected.get("pair_sha256",[])
                or len(pair_ids)!=1
                or (media,qid_hash) not in locked
                or row.get("human_review",{}).get("claim_visually_supported_by_reported_frames") is not None
                or media in seen):
            raise ValueError("claim_row_lock_binding_invalid")
        video=Path(row["local_video_path"])
        if not video.is_file() or _sha256_file(video)!=media:raise ValueError("claim_video_hash_mismatch")
        seen.add(media);records.append(row)
        bindings.append({"packet_name":packet_rel.name,"packet_sha256":hashlib.sha256(raw).hexdigest(),
            "record_index_zero_based":index,"batch_filter_name":filter_rel.name,
            "batch_filter_sha256":hashlib.sha256(batch_raw).hexdigest(),
            "pair_sha256":pair,"row_token_sha256":hashlib.sha256((media+str(qid)).encode()).hexdigest(),
            "source_video_sha256":media,"saved_claim_sha256":hashlib.sha256(claim.encode()).hexdigest()})
    if len(records)>3 or len(seen)!=len(records):raise ValueError("selected_record_count_invalid")
    return records,bindings,hashlib.sha256(lock_raw).hexdigest(),hashlib.sha256(selected_raw).hexdigest()

def _prepare_window(processor,process_vision_info,record,frames,fps,suffix,torch):
    rendered,messages=_build_prompt(processor,record,frames,float(fps),suffix)
    images,videos=process_vision_info(messages,return_video_metadata=True)
    if images is not None or not isinstance(videos,list) or len(videos)!=1:
        raise ValueError("window_video_preprocess_invalid")
    pixels,metadata=videos[0]
    if int(pixels.shape[0])!=WINDOW_FRAMES:raise ValueError("window_frame_count_invalid")
    validate_video_metadata(metadata,sample_fps=float(fps),frame_count=WINDOW_FRAMES)
    batch=processor(text=[rendered],images=images,videos=[pixels],video_metadata=[metadata],
        do_sample_frames=False,cap_pixels_per_frame=True,padding=True,return_tensors="pt")
    return batch

def _selection_status(rows, completed_windows):
    identities={row.get("row_token_sha256") for row in rows}
    complete=(len(rows)==3 and len(identities)==3 and None not in identities
              and completed_windows==12 and all(
                  len(row.get("window_mean_claim_logprobs_nats_per_token",[]))==4 for row in rows))
    return "completed_window_likelihood_selection" if complete else "partial_window_likelihood_smoke"

def combine_receipts(repo:Path, receipt_paths:list[Path], output:Path):
    """Combine partial runs only when they cover the frozen three-row cohort once."""
    payloads=[]
    for relative in receipt_paths:
        receipt,_=_read_private_json(repo,relative)
        if (receipt.get("status")!="partial_window_likelihood_smoke"
                or receipt.get("dataset_split")!="test"
                or receipt.get("purpose")!="exploratory_claim_likelihood_selected_window_prediction_dependence"):
            raise ValueError("partial_selection_receipt_contract_invalid")
        payloads.append(receipt)
    if len({p["cohort_lock_sha256"] for p in payloads})!=1 or len({p["selected_pair_filter_sha256"] for p in payloads})!=1:
        raise ValueError("partial_selection_receipt_cohort_mismatch")
    if len({json.dumps(p["source_sha256"],sort_keys=True) for p in payloads})!=1:
        raise ValueError("partial_selection_receipt_source_mismatch")
    rows=[r for p in payloads for r in p["records"]]
    if _selection_status(rows,sum(p["completed_windows"] for p in payloads))!="completed_window_likelihood_selection":
        raise ValueError("partial_selection_receipts_do_not_cover_cohort")
    merged=dict(payloads[0]);merged["records"]=sorted(rows,key=lambda r:r["row_token_sha256"])
    merged["completed_windows"]=sum(p["completed_windows"] for p in payloads)
    merged["requested_windows"]=12;merged["selected_cohort_complete"]=True
    merged["status"]="completed_window_likelihood_selection"
    merged["combined_receipt_sha256"]=[hashlib.sha256(_read_private_json(repo,p)[1]).hexdigest() for p in receipt_paths]
    raw=(json.dumps(merged,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())

def run(args):
    repo=Path(__file__).resolve().parents[1];output=validate_private_output(args.output,repo)
    if output.parent!=repo/"data"/"private_nextgqa" or output.suffix!=".json":
        raise ValueError("output_path_invalid")
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat();stage="private_input_validation"
    completed=0
    try:
        records,bindings,lock_hash,filter_hash=_read_selected_inputs(repo,args.lock,args.selected_filter,
            args.packet,args.batch_filter,args.record_index)
        from run_machine_localized_claim_mask_pilot import _host_available_bytes
        from claim_mask_video_decoder import decode_sampled_video,verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError("cuda_bf16_unavailable")
        free,total=torch.cuda.mem_get_info();host=_host_available_bytes()
        if free<MIN_FREE_VRAM or host<MIN_HOST_AVAILABLE:raise RuntimeError("localizer_resource_guard_failed")
        localizer_path=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"/"snapshots"/LOCALIZER_REVISION
        if not localizer_path.is_dir():raise ValueError("pinned_localizer_missing")
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        stage="load_localizer";processor=AutoProcessor.from_pretrained(str(localizer_path),local_files_only=True,
            min_pixels=256*28*28,max_pixels=256*28*28)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(localizer_path),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation="eager",device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id)!=int(processor.video_token_id):raise ValueError("localizer_video_token_mismatch")
        torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021);torch.cuda.reset_peak_memory_stats()
        rows=[];partition=partition_window_indices(DECODE_FRAMES,WINDOW_COUNT)
        for row_index,(record,binding) in enumerate(zip(records,bindings,strict=True)):
            stage=f"decode_video_{row_index}";frames,decoded=decode_sampled_video(record["local_video_path"],
                num_frames=DECODE_FRAMES,uniform_tolerance_seconds=Fraction(1,20),include_timing_map=True)
            verify_sampled_frame_pixels(frames,decoded)
            if decoded["source_sha256"]!=record["video_sha256"]:raise ValueError("decoded_video_digest_mismatch")
            pts=[float(Fraction(*x["relative_presentation_time"])) for x in decoded["sampled_frames"]]
            fps=Fraction(*decoded["effective_sample_fps"])
            version=record.get("generation_prompt_version")
            if version not in PROMPT_SUFFIX_BY_VERSION:raise ValueError("generation_prompt_version_invalid")
            suffix=PROMPT_SUFFIX_BY_VERSION[version];target=tuple(processor.tokenizer(
                record["generated_parsed"]["claim"],add_special_tokens=False)["input_ids"])
            if not target or len(target)>512:raise ValueError("claim_target_tokens_invalid")
            scores=[]
            for window_index,frame_indices in enumerate(partition):
                stage=f"score_window_{row_index}_{window_index}"
                free_now,_=torch.cuda.mem_get_info()
                host_now=_host_available_bytes()
                if free_now<MIN_FORWARD_VRAM or host_now<MIN_HOST_AVAILABLE:
                    raise RuntimeError("localizer_forward_resource_guard_failed")
                batch=_prepare_window(processor,process_vision_info,record,
                    [frames[i] for i in frame_indices],fps,suffix,torch)
                scored=_score_branch(model=model,batch_cpu=batch,target_ids=target,
                    video_token_id=int(processor.video_token_id),biased=False,torch=torch)
                score=float(scored["target_logprob_mean"])
                if not torch.isfinite(torch.tensor(score)):raise ValueError("window_score_nonfinite")
                scores.append(score);completed+=1
                del batch,scored;torch.cuda.empty_cache()
            chosen=select_max_mean_logprob(scores);frame_indices,interval=selected_interval(pts,chosen)
            rows.append({**binding,"dataset_split":"test","window_count":WINDOW_COUNT,
                "frames_per_window":WINDOW_FRAMES,"window_mean_claim_logprobs_nats_per_token":scores,
                "selected_window_index":chosen,"selected_sampled_frame_indices":list(frame_indices),
                "selected_interval_start_seconds":interval[0],"selected_interval_end_seconds":interval[1],
                "sampled_pts_seconds":pts,"decoded_pts_sha256":decoded["decoded_pts_sequence_sha256"],
                "target_token_count":len(target),"target_token_ids_sha256":_sha256_bytes(
                    json.dumps(target,separators=(",",":")).encode("ascii")),
                "effective_sample_fps":float(fps)})
        del model,processor;torch.cuda.synchronize();torch.cuda.empty_cache()
        source_files=[Path(__file__).resolve(),Path(__file__).resolve().parent/"claim_likelihood_window_ranking.py",
            Path(__file__).resolve().parent/"claim_mask_video_decoder.py",
            Path(__file__).resolve().parent/"run_saved_claim_frame_permutation_attention_smoke.py",
            Path(__file__).resolve().parent/"qta_multiquery_video_key_intervention.py",
            Path(__file__).resolve().parent/"run_nextgqa_gold_interval_margin_pilot.py",
            Path(__file__).resolve().parent/"generate_nextgqa_claim_review_pilot.py"]
        status=_selection_status(rows,completed)
        complete=status=="completed_window_likelihood_selection"
        payload={"schema_version":1,"status":status,
            "purpose":"exploratory_claim_likelihood_selected_window_prediction_dependence",
            "started_utc":started_utc,"dataset_split":"test","cohort_lock_sha256":lock_hash,
            "selected_pair_filter_sha256":filter_hash,"records":rows,"requested_windows":len(rows)*WINDOW_COUNT,
            "completed_windows":completed,"window_count":WINDOW_COUNT,"frames_per_window":WINDOW_FRAMES,
            "selection_rule":"highest 3B teacher-forced mean claim-token logprob; earliest window breaks ties",
            "localizer_model":"Qwen/Qwen2.5-VL-3B-Instruct","localizer_revision":LOCALIZER_REVISION,
            "scorer_model_revision":SCORER_REVISION,"precision":"4-bit weights; BF16 compute; eager attention",
            "selected_cohort_complete":complete,
            "seed":20261021,"human_review":{"review_actually_performed":False,"collection_waived_by_user":True},
            "direct_faithfulness":False,"rights_status":"unverified; local-only per user authorization",
            "hardware":{"device":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,
                "free_vram_before_load_gib":free/1024**3,"total_vram_gib":total/1024**3,
                "host_available_before_load_gib":host/1024**3,
                "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3},
            "source_sha256":{p.relative_to(repo).as_posix():_sha256_file(p) for p in source_files},
            "model_config_sha256":{"Qwen2.5-VL-3B-Instruct/config.json":_sha256_file(localizer_path/"config.json"),
                "Qwen2.5-VL-3B-Instruct/preprocessor_config.json":_sha256_file(localizer_path/"preprocessor_config.json")},
            "elapsed_seconds":time.monotonic()-started,
            "limitations":["four-window max selection induces winner's-curse bias",
                "3B and 7B checkpoints share Qwen family priors","automated claim-window match is unreviewed",
                "claim likelihood is not semantic evidence validation","test cohort is exploratory and no population inference"]}
        raw=(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        print(json.dumps({"status":payload["status"],"requested_windows":payload["requested_windows"],
            "completed_windows":completed,"selected_rows":len(rows),"output_name":output.name},sort_keys=True))
        return 0
    except Exception as exc:
        failed={"schema_version":1,"status":"failed_window_likelihood_selection",
            "stage":stage,"exception_type":type(exc).__name__,"failure_code":str(exc)[:120],
            "started_utc":started_utc,"requested_windows":len(records)*WINDOW_COUNT if 'records' in locals() else 0,
            "completed_windows":completed,"direct_faithfulness":False,
            "human_review_performed":False,"rights_status":"unverified; local-only"}
        try:
            raw=(json.dumps(failed,sort_keys=True,separators=(",",":"))+"\n").encode()
            fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
            with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        except FileExistsError:pass
        print(json.dumps({"status":failed["status"],"stage":stage,"completed_windows":completed,
            "failure_code":failed["failure_code"],"output_name":output.name},sort_keys=True));return 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output")
    p.add_argument("--lock",type=Path);p.add_argument("--selected-filter",type=Path)
    p.add_argument("--packet",type=Path,action="append")
    p.add_argument("--batch-filter",type=Path,action="append")
    p.add_argument("--record-index",type=int,action="append")
    p.add_argument("--user-authorized-local-only",action="store_true",required=True)
    p.add_argument("--combine-receipt",type=Path,action="append")
    p.add_argument("--combine-output",type=Path)
    p.add_argument("--combine-only",action="store_true")
    args=p.parse_args()
    if args.combine_only:
        if not args.combine_receipt or not args.combine_output: p.error("combine-only requires --combine-receipt and --combine-output")
        repo=Path(__file__).resolve().parents[1]
        out=validate_private_output(args.combine_output,repo)
        if out.parent!=repo/"data"/"private_nextgqa":p.error("combine output must be private")
        combine_receipts(repo,args.combine_receipt,out)
        print(json.dumps({"status":"completed_window_likelihood_selection","output_name":out.name,"combined_receipts":len(args.combine_receipt)}));return 0
    if not args.output or not args.packet or not args.batch_filter or not args.record_index or not args.lock or not args.selected_filter:
        p.error("selection run requires --output, --lock, --selected-filter, --packet, --batch-filter, and --record-index")
    return run(args)

if __name__=="__main__":raise SystemExit(main())
