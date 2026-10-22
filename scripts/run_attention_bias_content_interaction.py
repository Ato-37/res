#!/usr/bin/env python3
"""Post-result exploratory attention-bias × original/gray claim-likelihood interaction."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,math,os,time
from pathlib import Path

from private_output_paths import validate_private_output
from run_saved_claim_frame_permutation_attention_smoke import (
    FRAME_COUNT,LAYERS_1_BASED,LOGIT_BIAS,MODEL_REVISION,MIN_FREE_VRAM,
    MIN_HOST_AVAILABLE,PACKET_REL,REMAINING_PACKET_REL,_host_available_bytes,
    _prepare_processor_batch,_read_packet,_score_branch,_sha256_bytes,_sha256_file,
    _tensor_sha256,
)
from run_saved_claim_video_content_dependence import make_constant_gray_frames

RECEIPTS={0:("saved_claim_frame_permutation_attention_smoke_20261021_02.json","saved_claim_video_content_dependence_20261021_smoke_01.json"),
1:("saved_claim_frame_permutation_attention_smoke_20261021_remaining_00_retry_01.json","saved_claim_video_content_dependence_20261021_remaining_00.json"),
2:("saved_claim_frame_permutation_attention_smoke_20261021_remaining_01.json","saved_claim_video_content_dependence_20261021_remaining_01.json")}

def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def interaction(original_base,gray_base,original_bias,gray_bias):
    vals=(original_base,gray_base,original_bias,gray_bias)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)) for v in vals):
        raise ValueError("branch_logprob_must_be_finite")
    base=float(original_base)-float(gray_base);biased=float(original_bias)-float(gray_bias)
    return {"baseline_original_minus_gray_nats_per_token":base,
        "biased_original_minus_gray_nats_per_token":biased,
        "bias_by_video_content_interaction_nats_per_token":biased-base}

def fixed_global_index(packet_rel:Path,record_index:int)->int:
    if type(record_index)is not int or record_index<0:
        raise ValueError("fixed_row_index_invalid")
    if packet_rel==PACKET_REL:
        if record_index!=0:raise ValueError("smoke_packet_row_index_invalid")
        return 0
    if packet_rel==REMAINING_PACKET_REL and record_index in (0,1):
        return record_index+1
    raise ValueError("fixed_packet_row_index_invalid")

def load_joined_receipts(repo,index):
    if index not in RECEIPTS:raise ValueError("fixed_row_index_invalid")
    attention_name,content_name=RECEIPTS[index];root=repo/"data"/"private_nextgqa"
    ap=root/attention_name;cp=root/content_name
    a=json.loads(ap.read_text(encoding="utf-8"));c=json.loads(cp.read_text(encoding="utf-8"))
    if a.get("status")!="completed_one_row_exploratory_sensitivity_smoke" or c.get("status")!="completed_one_row_exploratory_video_content_sensitivity":
        raise ValueError("baseline_receipt_status_invalid")
    if (a.get("completed_forwards"),c.get("completed_forwards"))!=(4,2):raise ValueError("baseline_forward_count_invalid")
    if a.get("model_revision")!=MODEL_REVISION or c.get("model_revision")!=MODEL_REVISION:
        raise ValueError("baseline_checkpoint_revision_not_pinned")
    for key in ("row_token_sha256","source_video_sha256","saved_claim_text_sha256","private_input_packet_sha256","model_revision"):
        if a.get(key)!=c.get(key):raise ValueError("baseline_receipt_binding_mismatch_"+key)
    if (a.get("direct_faithfulness") is not False or c.get("direct_faithfulness") is not False
            or a.get("human_review",{}).get("review_actually_performed") is not False
            or c.get("human_review",{}).get("review_actually_performed") is not False):
        raise ValueError("baseline_scope_or_review_state_invalid")
    if a.get("reencoded_claim_token_ids_sha256")!=c.get("retokenized_claim_ids_sha256"):
        raise ValueError("baseline_target_token_binding_mismatch")
    if a.get("target_token_count")!=c.get("target_token_count"):
        raise ValueError("baseline_target_length_mismatch")
    common_sources=("scripts/qta_multiquery_video_key_intervention.py",
        "scripts/claim_mask_video_decoder.py","scripts/generate_nextgqa_claim_review_pilot.py")
    for name in common_sources:
        if a.get("source_sha256",{}).get(name)!=c.get("source_sha256",{}).get(name):
            raise ValueError("baseline_source_hash_join_mismatch")
        if not a.get("source_sha256",{}).get(name):raise ValueError("baseline_source_hash_missing")
    attention_runner_hash=a.get("source_sha256",{}).get("scripts/run_saved_claim_frame_permutation_attention_smoke.py")
    current_runner=repo/"scripts"/"run_saved_claim_frame_permutation_attention_smoke.py"
    preserved_runner=repo/"scripts"/"run_saved_claim_frame_permutation_attention_smoke_20261021_v1.py"
    if attention_runner_hash not in {_sha256_file(current_runner),_sha256_file(preserved_runner)}:
        raise ValueError("attention_runner_source_not_preserved")
    if c.get("source_sha256",{}).get("scripts/run_saved_claim_frame_permutation_attention_smoke.py")!=_sha256_file(current_runner):
        raise ValueError("content_runner_helper_source_mismatch")
    for name in ("original_baseline","original_visual_bias"):
        score=a["branch_target_scores"].get(name,{})
        if (score.get("score_semantics")!="mean conditional logprob of frozen target token IDs"
                or score.get("target_token_count")!=a.get("target_token_count")
                or score.get("visual_key_position_count",0)<1
                or score.get("visual_key_positions_sha256")!=a["branch_target_scores"]["original_baseline"].get("visual_key_positions_sha256")
                or score.get("first_query_position")!=a["branch_target_scores"]["original_baseline"].get("first_query_position")
                or score.get("last_target_query_position")!=a["branch_target_scores"]["original_baseline"].get("last_target_query_position")):
            raise ValueError("attention_score_geometry_join_mismatch")
        hooks=score.get("intervention_hook_invocations_by_layer_1_based")
        expected=0 if name=="original_baseline" else 1
        if not isinstance(hooks,dict) or hooks!={str(layer):expected for layer in LAYERS_1_BASED}:
            raise ValueError("attention_receipt_hook_count_invalid")
    content_original=c["branch_target_scores"]["original_video"]
    content_gray=c["branch_target_scores"]["constant_gray_video"]
    for key in ("visual_key_position_count","visual_key_positions_sha256","first_query_position","last_target_query_position"):
        if content_original.get(key)!=content_gray.get(key) or content_original.get(key)!=a["branch_target_scores"]["original_baseline"].get(key):
            raise ValueError("cross_receipt_visual_query_positions_mismatch")
    if a.get("attention_intervention",{}).get("layers_1_based")!=list(LAYERS_1_BASED) or not math.isclose(a.get("attention_intervention",{}).get("logit_bias",float("nan")),LOGIT_BIAS,abs_tol=1e-15):
        raise ValueError("baseline_attention_intervention_mismatch")
    original_base=a["branch_target_scores"]["original_baseline"]["target_logprob_mean"]
    content_base=c["branch_target_scores"]["original_video"]["target_logprob_mean"]
    if not math.isclose(original_base,content_base,rel_tol=0,abs_tol=1e-12):
        raise ValueError("original_baseline_reuse_mismatch")
    return a,c,ap,cp

def run(output_arg:str,packet_rel:Path,record_index:int)->int:
    repo=Path(__file__).resolve().parents[1];output=validate_private_output(output_arg,repo)
    if output.parent!=repo/"data"/"private_nextgqa" or output.suffix!=".json":raise ValueError("private_output_must_be_json_child_of_private_data_root")
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat();attempted=completed=0;stage="receipt_join"
    try:
        global_index=fixed_global_index(packet_rel,record_index)
        attention,content,attention_path,content_path=load_joined_receipts(repo,global_index)
        record,packet_hash=_read_packet(repo,packet_rel,record_index)
        claim=record.get("generated_parsed",{}).get("claim");video_path=record.get("local_video_path");video_hash=record.get("video_sha256")
        row_token=_sha256_bytes((video_hash+str(record.get("dataset_question_id",""))).encode())
        if (row_token!=attention["row_token_sha256"] or video_hash!=attention["source_video_sha256"]
                or packet_hash!=attention["private_input_packet_sha256"]
                or _sha256_bytes(claim.encode())!=attention["saved_claim_text_sha256"]):
            raise ValueError("packet_row_does_not_match_frozen_baselines")
        stage="cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError("cuda_bf16_unavailable")
        free,total=torch.cuda.mem_get_info();host_free=_host_available_bytes()
        if free<MIN_FREE_VRAM or host_free<MIN_HOST_AVAILABLE:raise RuntimeError("pre_model_resource_guard_failed")
        model_path=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"/"snapshots"/MODEL_REVISION
        if not model_path.is_dir():raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video,verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
        from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
        prompt_version,prompt_suffix=resolve_claim_prompt_suffix(record,PROMPT_SUFFIX_BY_VERSION)
        stage="decode_and_rebuild_inputs"
        processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,min_pixels=256*28*28,max_pixels=256*28*28)
        frames,decoded=decode_sampled_video(video_path,num_frames=FRAME_COUNT,uniform_tolerance_seconds=Fraction(1,20));verify_sampled_frame_pixels(frames,decoded)
        if decoded["source_sha256"]!=video_hash:raise ValueError("claim_video_digest_mismatch")
        saved_pts=record.get("sampled_frame_relative_pts_seconds")
        decoded_pts=[float(Fraction(*x["relative_presentation_time"])) for x in decoded["sampled_frames"]]
        if len(frames)!=FRAME_COUNT or not isinstance(saved_pts,list) or len(saved_pts)!=FRAME_COUNT or any(abs(float(x)-y)>1e-6 for x,y in zip(saved_pts,decoded_pts,strict=True)):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        fps=float(Fraction(*decoded["effective_sample_fps"]))
        gray_frames=make_constant_gray_frames(frames)
        original,original_meta=_prepare_processor_batch(processor,process_vision_info,record,frames,fps,prompt_suffix,torch)
        gray,gray_meta=_prepare_processor_batch(processor,process_vision_info,record,gray_frames,fps,prompt_suffix,torch)
        if original_meta!=gray_meta or set(original)!=set(gray):raise ValueError("condition_metadata_or_keys_changed")
        for key in original:
            x,y=original[key],gray[key]
            if torch.is_tensor(x) and torch.is_tensor(y):
                if tuple(x.shape)!=tuple(y.shape):raise ValueError("condition_tensor_shape_changed")
                if key!="pixel_values_videos" and not torch.equal(x,y):raise ValueError("condition_nonvisual_tensor_changed")
            elif x!=y:raise ValueError("condition_non_tensor_field_changed")
        orig_pixels=_tensor_sha256(original["pixel_values_videos"],torch);gray_pixels=_tensor_sha256(gray["pixel_values_videos"],torch)
        if orig_pixels!=attention["frames"]["original_processor_pixels_sha256"] or orig_pixels!=content["video_conditions"]["original_processor_pixels_sha256"] or gray_pixels!=content["video_conditions"]["constant_gray_processor_pixels_sha256"]:
            raise ValueError("reconstructed_processor_pixel_hash_mismatch")
        if original["video_grid_thw"][0].tolist()!=attention["frames"]["video_grid_thw"] or [float(x) for x in original["second_per_grid_ts"].tolist()]!=attention["frames"]["second_per_grid_ts"]:
            raise ValueError("reconstructed_temporal_geometry_mismatch")
        if original_meta.get("frames_indices")!=list(range(FRAME_COUNT)) or len(original_meta.get("frames_indices",[]))!=FRAME_COUNT:
            raise ValueError("reconstructed_frame_indices_invalid")
        target_ids=tuple(processor.tokenizer(claim,add_special_tokens=False)["input_ids"])
        target_hash=_sha256_bytes(json.dumps(target_ids,separators=(",",":")).encode("ascii"))
        if target_hash!=attention["reencoded_claim_token_ids_sha256"]:raise ValueError("target_token_ids_mismatch")
        stage="load_pinned_scorer"
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.bfloat16,
            attn_implementation="eager",device_map="cuda",quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id)!=int(processor.video_token_id):raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021);torch.cuda.reset_peak_memory_stats()
        stage="gray_visual_bias_forward";attempted=1
        gray_bias=_score_branch(model=model,batch_cpu=gray,target_ids=target_ids,video_token_id=int(processor.video_token_id),biased=True,torch=torch)
        completed=1
        stage="receipt_finalize"
        original_base=attention["branch_target_scores"]["original_baseline"]["target_logprob_mean"]
        original_bias=attention["branch_target_scores"]["original_visual_bias"]["target_logprob_mean"]
        gray_base=content["branch_target_scores"]["constant_gray_video"]["target_logprob_mean"]
        estimand=interaction(original_base,gray_base,original_bias,gray_bias["target_logprob_mean"])
        source_files=[Path(__file__).resolve(),Path(__file__).resolve().parent/"run_saved_claim_frame_permutation_attention_smoke.py",
            Path(__file__).resolve().parent/"run_saved_claim_video_content_dependence.py",
            Path(__file__).resolve().parent/"qta_multiquery_video_key_intervention.py",
            Path(__file__).resolve().parent/"claim_mask_video_decoder.py",
            Path(__file__).resolve().parent/"generate_nextgqa_claim_review_pilot.py"]
        payload={"schema_version":1,"status":"completed_one_row_posthoc_attention_bias_content_interaction",
            "started_utc":started_utc,"post_result_exploration":True,"direct_faithfulness":False,"faithfulness_result":False,
            "human_review":{"review_actually_performed":False,"collection_waived_by_user_for_this_exploratory_run":True},
            "rights_status":"unverified; local-only per user authorization","requested_records":1,"effective_records":1,
            "requested_forwards":1,"attempted_forwards":attempted,"completed_forwards":completed,
            "row_token_sha256":row_token,"source_video_sha256":video_hash,"private_input_packet_sha256":packet_hash,
            "generation_prompt_version":prompt_version,"prompt_suffix_sha256":_sha256_bytes(prompt_suffix.encode("utf-8")),
            "saved_claim_text_sha256":attention["saved_claim_text_sha256"],"retokenized_claim_ids_sha256":target_hash,
            "target_token_count":len(target_ids),"model":"Qwen/Qwen2.5-VL-7B-Instruct","model_revision":MODEL_REVISION,
            "precision":"4-bit weights; BF16 compute; eager attention","seed":20261021,
            "attention_intervention":{"layers_1_based":list(LAYERS_1_BASED),"logit_bias":LOGIT_BIAS,
                "key_set":"all processor-expanded video-token positions","query_set":"all re-tokenized claim-token prediction rows"},
            "reused_frozen_receipts":{"attention_receipt_sha256":sha_file(attention_path),"content_receipt_sha256":sha_file(content_path),
                "original_baseline_mean":original_base,"original_bias_mean":original_bias,"gray_baseline_mean":gray_base,
                "original_processor_pixels_sha256":orig_pixels,"gray_processor_pixels_sha256":gray_pixels},
            "new_gray_video_biased_score":gray_bias,
            "primary_descriptive_interaction":estimand,
            "hardware":{"device":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,"free_vram_before_load_gib":free/1024**3,
                "total_vram_gib":total/1024**3,"host_available_before_load_gib":host_free/1024**3,
                "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3},
            "source_sha256":{str(p.relative_to(repo)):sha_file(p) for p in source_files},"elapsed_seconds":time.monotonic()-started,
            "inferential_status":"post-result exploratory selected three-row cohort; report finite descriptive values only",
            "limitations":["gray video is a severe out-of-distribution control","global attention bias is not claim-specific",
                "same-model generated claims and re-tokenized target IDs","baseline scores are reused from prior exact-hash receipts",
                "positive interaction is not improved claim truth, grounding, or explanation faithfulness",
                "human review was not performed","rights unverified; all protected data/results remain local"]}
        raw=(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        print(json.dumps({"status":payload["status"],"completed_forwards":completed,
            "interaction_nats_per_token":estimand["bias_by_video_content_interaction_nats_per_token"],"output_name":output.name},sort_keys=True));return 0
    except Exception as exc:
        failure={"schema_version":1,"status":"failed_posthoc_attention_bias_content_interaction","stage":stage,
            "exception_type":type(exc).__name__,"failure_code":str(exc)[:160],"started_utc":started_utc,
            "requested_forwards":1,"attempted_forwards":attempted,"completed_forwards":completed,
            "direct_faithfulness":False,"human_review_performed":False,"rights_status":"unverified; local-only",
            "row_token_sha256":locals().get("row_token"),"source_video_sha256":locals().get("video_hash"),
            "private_input_packet_sha256":locals().get("packet_hash")}
        try:
            raw=(json.dumps(failure,sort_keys=True,separators=(",",":"))+"\n").encode();fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
            with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        except FileExistsError:pass
        print(json.dumps({"status":failure["status"],"stage":stage,"completed_forwards":completed,
            "failure_code":failure["failure_code"],"output_name":output.name},sort_keys=True));return 2

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    parser.add_argument("--packet",choices=("smoke","remaining"),default="smoke");parser.add_argument("--record-index",type=int,default=0)
    parser.add_argument("--user-authorized-local-only",action="store_true",required=True);args=parser.parse_args()
    return run(args.output,PACKET_REL if args.packet=="smoke" else REMAINING_PACKET_REL,args.record_index)

if __name__=="__main__":raise SystemExit(main())
