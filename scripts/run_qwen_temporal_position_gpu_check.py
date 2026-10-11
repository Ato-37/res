"""Video-only runtime regression check of native vs fractional temporal IDs.

One fixed NOAA source, four frames, generic prompt, no QA, labels, generation,
head/layer search or efficacy outcome. Three sequential model forwards: native
automatic IDs, explicit identical IDs (no-op), and a temporal-row-only candidate
matching current official upstream fractional arithmetic. Does not edit .venv.
Usage: PYTHONPATH=scripts .venv/bin/python scripts/run_qwen_temporal_position_gpu_check.py --output results/NEW.json
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from run_dataset_video_gpu_smoke import sha256
from qwen_temporal_geometry import replace_video_temporal_rope_row, qwen_temporal_geometry

RUN_PROGRESS = {'stage':'preflight', 'completed_forwards':0}


def compare_logits(reference, candidate):
    """Scalar next-token sensitivity at temperature 1; no raw tokens exported."""
    import torch
    p = reference.double().log_softmax(-1)
    q = candidate.double().log_softmax(-1)
    middle = torch.logaddexp(p,q) - math.log(2)
    js = ((p.exp()*(p-middle)).sum() + (q.exp()*(q-middle)).sum())/2
    return {
        'maximum_absolute_logit_difference':float((reference-candidate).abs().max()),
        'mean_absolute_logit_difference':float((reference-candidate).abs().mean()),
        'jensen_shannon_divergence_nats':max(0.0,float(js)),
        'top1_equal':bool(reference.argmax()==candidate.argmax()),
        'bitwise_equal':bool(torch.equal(reference,candidate))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo/'results') or output.exists():
        parser.error('New output under results/ required; overwrite refused')
    revision = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    expected_media = '13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126'
    video = repo/'datasets/noaa_qa_pilot_20261011/media/source.mp4'
    if not video.is_file() or sha256(video)!=expected_media:
        parser.error('Pinned NOAA media missing/hash mismatch')
    model_path = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots'/revision
    if not model_path.is_dir():
        parser.error('Pinned cached model missing; download disabled')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from grounding_check import extract_frames
    from qwen_vl_utils import process_vision_info
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('CUDA/BF16 required; no CPU fallback')
    free,total = torch.cuda.mem_get_info()
    if free<12*1024**3:
        parser.error('At least 12 GiB free VRAM required')
    random.seed(20261012); torch.manual_seed(20261012); torch.cuda.manual_seed_all(20261012)
    started=datetime.now(timezone.utc).isoformat(); timer=time.monotonic()
    RUN_PROGRESS['stage']='extraction_and_model_load'
    prompt='Describe the visible scene briefly. Use only visual information.'
    frames,tmp,timestamps,fps=extract_frames(str(video),4,224)
    try:
        if len(timestamps)!=4 or len(frames)!=4:
            raise RuntimeError('Four effective frames required')
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map='cuda').eval()
        processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,
            min_pixels=256*28*28,max_pixels=256*28*28)
        messages=[{'role':'user','content':[
            {'type':'video','video':frames,'sample_fps':fps}, {'type':'text','text':prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos=process_vision_info(messages,return_video_metadata=True)
        if not videos or len(videos)!=1:
            raise RuntimeError('Exactly one processed video required')
        tensor,metadata=videos[0]
        batch=processor(text=[text],images=images,videos=[tensor],video_metadata=[metadata],
            do_sample_frames=False,cap_pixels_per_frame=True,padding=True,return_tensors='pt').to('cuda')
        grid=batch['video_grid_thw'][0].tolist()
        merge=int(processor.video_processor.merge_size)
        positions=(batch['input_ids'][0]==model.config.video_token_id).nonzero(as_tuple=True)[0].tolist()
        if positions!=list(range(positions[0],positions[-1]+1)):
            raise RuntimeError('Video tokens not contiguous')
        types=batch['mm_token_type_ids'][0].tolist()
        if [i for i,x in enumerate(types) if x==2]!=positions or any(x==1 for x in types):
            raise RuntimeError('Video modality mapping mismatch or image input')
        seconds=float(batch['second_per_grid_ts'][0])
        config=model.config.vision_config
        native,delta=model.model.get_rope_index(
            input_ids=batch['input_ids'],mm_token_type_ids=batch['mm_token_type_ids'],
            video_grid_thw=batch['video_grid_thw'],second_per_grid_ts=batch['second_per_grid_ts'],
            attention_mask=batch['attention_mask'])
        native_list=native[:,0,:].detach().cpu().tolist()
        candidate_rows=replace_video_temporal_rope_row(native_list,
            video_token_spans=[(positions[0],positions[-1]+1)],image_token_spans=[],
            attention_mask=batch['attention_mask'][0].tolist(),video_grid_thw=grid,
            spatial_merge_size=merge,second_per_grid_seconds=seconds,
            tokens_per_second=int(config.tokens_per_second))
        candidate=torch.tensor(candidate_rows,dtype=native.dtype,device=native.device).unsqueeze(1)
        video_mask=batch['mm_token_type_ids'][0]==2
        if not torch.equal(native[1:],candidate[1:]) or not torch.equal(native[0,0,~video_mask],candidate[0,0,~video_mask]):
            raise RuntimeError('Candidate changed non-temporal-video coordinates')
        captures=[]
        def capture(module, positional, keyword):
            captures.append(keyword['position_ids'].detach().cpu().clone())
        handle=model.model.language_model.register_forward_pre_hook(capture,with_kwargs=True)
        conditions=[]; logits=[]
        try:
            for name,ids in [('native_auto',None),('explicit_native_noop',native),('fractional_video_time_only',candidate)]:
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                begin=time.monotonic()
                kw={} if ids is None else {'position_ids':ids}
                with torch.inference_mode():
                    out=model(**batch,**kw,use_cache=False,output_attentions=False,logits_to_keep=1)
                torch.cuda.synchronize()
                values=out.logits[0,-1].detach().float().cpu()
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError('Nonfinite next-token logits')
                RUN_PROGRESS['completed_forwards'] += 1
                RUN_PROGRESS['stage']='forward_'+name
                conditions.append({'condition':name,'finite_logits':True,
                    'forward_seconds':time.monotonic()-begin,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
                logits.append(values); del out
        finally:
            handle.remove()
        RUN_PROGRESS['stage']='identity_and_position_validation'
        if len(captures)!=3 or not torch.equal(captures[0],native.cpu()) or not torch.equal(captures[1],native.cpu()) or not torch.equal(captures[2],candidate.cpu()):
            raise RuntimeError('Actual language-model position IDs differ from requested IDs')
        noop=compare_logits(logits[0],logits[1])
        if not noop['bitwise_equal']:
            raise RuntimeError('Explicit native no-op failed exact output identity')
        step_tokens=(grid[1]//merge)*(grid[2]//merge)
        anchors_native=[int(native[0,0,positions[0]+i*step_tokens]) for i in range(grid[0])]
        anchors_candidate=[int(candidate[0,0,positions[0]+i*step_tokens]) for i in range(grid[0])]
        result={
            'schema_version':1,'purpose':'real_video_temporal_position_runtime_regression_check',
            'started_utc':started,'elapsed_seconds':time.monotonic()-timer,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'source_sha256':{n:sha256(repo/'scripts'/n) for n in [Path(__file__).name,
                'run_dataset_video_gpu_smoke.py','qwen_temporal_geometry.py','grounding_check.py','video_time_mapping.py']},
            'installed_model_source_sha256':sha256(inspect.getfile(Qwen2_5_VLModel)),
            'model':'Qwen/Qwen2.5-VL-7B-Instruct','model_revision':revision,'model_config_sha256':sha256(model_path/'config.json'),
            'media_sha256':expected_media,'source_record':'https://www.ncei.noaa.gov/metadata/granule/geoportal/rest/metadata/item/EX2503_DIVE03_20250413T192459Z_ROVHD/html',
            'rights_source':'https://oceanexplorer.noaa.gov/okeanos-beyond-the-blue-media-resources/',
            'credit':'NOAA Ocean Exploration','cruise_citation':'NOAA Ocean Exploration (2025), EX2503, NCEI, https://doi.org/10.25921/vgk8-m692 (subset used)',
            'seed':20261012,'quantization':'4bit','compute_dtype':'bfloat16','attention_implementation':'eager',
            'prompt':prompt,'audio_used':False,'annotations_used':False,'human_qa_attestation':'pending; no QA evaluation',
            'requested_samples':1,'effective_samples':1,'requested_forwards':3,'effective_forwards':3,'failures':0,
            'requested_frames':4,'effective_frames':len(timestamps),'frame_max_dimension':224,'max_pixels':256*28*28,
            'sample_timestamps_seconds':timestamps,'sampled_fps':fps,'grid_thw':grid,'video_tokens':len(positions),
            'input_tokens':int(batch['input_ids'].shape[1]),'seconds_per_grid':seconds,
            'native_temporal_anchors':anchors_native,'fractional_temporal_anchors':anchors_candidate,
            'changed_position_id_entries':int((native!=candidate).sum()),
            'spatial_and_text_position_ids_preserved':True,'actual_forward_position_ids_verified':True,
            'noop_comparison':noop,'fractional_vs_native_sensitivity':compare_logits(logits[0],logits[2]),
            'temporal_geometry':qwen_temporal_geometry(timestamps,int(processor.video_processor.temporal_patch_size),
                tokens_per_second=int(config.tokens_per_second),second_per_grid=seconds,video_start_position=anchors_native[0]),
            'conditions':conditions,'gpu':torch.cuda.get_device_name(0),'gpu_free_before_gib':free/1024**3,'gpu_total_gib':total/1024**3,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','bitsandbytes','qwen-vl-utils','opencv-python-headless']},
            'cuda_version':torch.version.cuda,
            'limitations':['One sparse real video; no claim generation, QA accuracy, gold intervals or efficacy.',
                'Sensitivity metrics are next-token distribution changes caused by position IDs, not faithfulness or improvement.',
                'Candidate fixes only temporal video row; native text/spatial layout retained, not full4.49 backport.',
                'Explicit position IDs tested only for batch1/singlevideo/allone-mask/full-prompt/no cache; generation unsupported.',
                'No uncertainty interval/statistical test for one deterministic engineering fixture.'],
            'privacy':'No raw media, dataset QA, labels, full token/logit vectors, persons or absolute local paths.'}
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
        print(f'GPU position check passed: 1 video, 3/3 forwards; no-op identical; native anchors {anchors_native}, candidate {anchors_candidate}; output {args.output}')
    finally:
        shutil.rmtree(tmp,ignore_errors=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import sys
        arguments=sys.argv[1:]
        if '--output' in arguments:
            index=arguments.index('--output')
            if index+1<len(arguments):
                repo=Path(__file__).resolve().parents[1]
                output=(repo/arguments[index+1]).resolve()
                if output.is_relative_to(repo/'results') and not output.exists():
                    output.parent.mkdir(parents=True,exist_ok=True)
                    with output.open('x',encoding='utf-8') as stream:
                        json.dump({'schema_version':1,'status':'failed',
                            'purpose':'real_video_temporal_position_runtime_regression_check',
                            'requested_samples':1,'effective_samples':0,'requested_forwards':3,
                            'completed_finite_forwards':RUN_PROGRESS['completed_forwards'],
                            'failures':1,'failure_type':type(error).__name__,
                            'stage':RUN_PROGRESS['stage'],'runner_sha256':sha256(__file__),
                            'created_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
                        stream.write('\n')
        print(f'GPU position check failed: {type(error).__name__}; inspect failure artifact.',file=sys.stderr)
        sys.exit(1)
