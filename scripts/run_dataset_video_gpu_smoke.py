"""Video-only GPU execution validation; no annotation, QA accuracy, or faithfulness.

Usage: .venv/bin/python scripts/run_dataset_video_gpu_smoke.py --output results/UNIQUE.json
Uses the locally retained public-domain NOAA EX2503 dive03 video. Annotation
attestation remains pending and is not needed or asserted for this label-free
execution protocol. Media, source metadata, and annotations are never model text.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-frames', type=int, default=4, choices=[4])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / 'results') or output.exists():
        parser.error('Use a new output inside results/; overwrite refused')
    expected_sha256 = '13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126'
    video = (repo / 'datasets/noaa_qa_pilot_20261011/media/source.mp4').resolve()
    if not video.is_file() or sha256(video) != expected_sha256:
        parser.error('Missing video or media hash mismatch')
    revision = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    model_path = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots' / revision
    if not model_path.is_dir():
        parser.error('Pinned checkpoint missing; no automatic download')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import shutil
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from grounding_check import extract_frames
    from video_time_mapping import chunk_timestamp_ranges

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('CUDA and BF16 required; CPU fallback disabled')
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 12 * 1024**3:
        parser.error('At least 12 GiB free VRAM required')
    random.seed(20261012)
    torch.manual_seed(20261012)
    torch.cuda.manual_seed_all(20261012)
    prompt = 'Describe the visible scene briefly. Use only visual information.'
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    paths, tmp, timestamps, fps = extract_frames(str(video), args.num_frames, 224)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager', device_map='cuda',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256*28*28, max_pixels=256*28*28)
        messages = [{'role':'user','content':[
            {'type':'video','video':paths,'sample_fps':fps},
            {'type':'text','text':prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        tensor, metadata = videos[0]
        batch = processor(text=[text], images=images, videos=[tensor],
            video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
            padding=True, return_tensors='pt').to('cuda')
        grid = batch['video_grid_thw'][0].tolist()
        patch = int(processor.video_processor.temporal_patch_size)
        ranges = chunk_timestamp_ranges(timestamps, patch)
        positions = (batch['input_ids'][0] == model.config.video_token_id).nonzero(as_tuple=True)[0]
        expected_tokens = int(batch['video_grid_thw'][0].prod()) // int(processor.video_processor.merge_size)**2
        if len(ranges) != grid[0] or len(positions) != expected_tokens:
            raise RuntimeError('Processor token/time grid mismatch')
        seconds_per_grid = float(batch['second_per_grid_ts'][0])
        if abs(seconds_per_grid - patch/fps) > 1e-5:
            raise RuntimeError('Processor FPS metadata mismatch')
        torch.cuda.reset_peak_memory_stats()
        forward_started = time.monotonic()
        with torch.inference_mode():
            out = model(**batch, output_attentions=True, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        finite_logits = bool(torch.isfinite(out.logits).all())
        finite_attention = all(bool(torch.isfinite(a).all()) for a in out.attentions)
        if not finite_logits or not finite_attention:
            raise RuntimeError('Nonfinite logits/attention')
        result = {
            'schema_version':1, 'purpose':'real_dataset_video_only_gpu_execution_validation',
            'started_utc':started, 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'source_sha256':{name:sha256(repo/'scripts'/name) for name in [Path(__file__).name,'grounding_check.py','video_time_mapping.py']},
            'model':'Qwen/Qwen2.5-VL-7B-Instruct', 'model_revision':revision,
            'model_config_sha256':sha256(model_path/'config.json'),
            'media_sha256':expected_sha256,
            'selection':'NOAA EX2503 dive03 fixed before outcomes; one local real video; entire 300-second source',
            'source_record':'https://www.ncei.noaa.gov/metadata/granule/geoportal/rest/metadata/item/EX2503_DIVE03_20250413T192459Z_ROVHD/html',
            'rights_source':'https://oceanexplorer.noaa.gov/okeanos-beyond-the-blue-media-resources/',
            'credit':'NOAA Ocean Exploration', 'cruise_citation':'NOAA Ocean Exploration (2025), EX2503: Beyond the Blue: Papahanaumokuakea ROV and Mapping, NCEI, https://doi.org/10.25921/vgk8-m692 (subset used)',
            'seed':20261012, 'quantization':'4bit', 'compute_dtype':'bfloat16',
            'attention_implementation':'eager', 'prompt':prompt, 'audio_used':False,
            'annotations_used':False, 'human_qa_attestation':'pending; no QA evaluation in this run',
            'requested_samples':1,'effective_samples':1,'failures':0,
            'requested_frames':args.num_frames,'effective_frames':len(timestamps),
            'frame_max_dimension':224,'max_pixels':256*28*28,
            'sample_timestamps_seconds':timestamps,'sampled_fps':fps,'chunk_ranges_seconds':ranges,
            'grid_thw':grid,'seconds_per_grid':seconds_per_grid,'video_tokens':len(positions),
            'input_tokens':int(batch['input_ids'].shape[1]),'attention_layers':len(out.attentions),
            'finite_logits':finite_logits,'finite_attentions':finite_attention,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'forward_seconds':time.monotonic()-forward_started,'elapsed_seconds':time.monotonic()-timer,
            'gpu':torch.cuda.get_device_name(0),'gpu_free_before_gib':free_bytes/1024**3,'gpu_total_gib':total_bytes/1024**3,
            'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','bitsandbytes','qwen-vl-utils','opencv-python']},
            'cuda_version':torch.version.cuda,
            'limitations':['One video, sparse frames, one forward, no generation or answer accuracy.',
                'Chunk ranges represent coarse sampled-frame spans, not validated evidence intervals.',
                'No treatment/control, effect estimate, confidence interval or significance test.',
                'No grounding improvement, novelty, or faithfulness claim.'],
            'privacy':'No raw media, dataset QA, labels, identifiers of persons, or absolute local paths.'
        }
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write('\n')
        print(f'Passed real-video GPU forward: 1/1 video; {len(timestamps)}/{args.num_frames} frames; '
            f'peak {result["peak_allocated_gib"]:.3f} GiB; elapsed {result["elapsed_seconds"]:.1f}s. Output: {args.output}')
    finally:
        shutil.rmtree(tmp,ignore_errors=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Preserve failed GPU attempts without serializing potentially sensitive
        # exception messages (which can contain paths, prompts, or model inputs).
        import sys
        arguments = sys.argv[1:]
        if '--output' in arguments:
            index = arguments.index('--output')
            if index + 1 < len(arguments):
                repo = Path(__file__).resolve().parents[1]
                output = (repo / arguments[index + 1]).resolve()
                if output.is_relative_to(repo / 'results') and not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('x', encoding='utf-8') as stream:
                        json.dump({'schema_version':1, 'status':'failed',
                            'purpose':'real_dataset_video_only_gpu_execution_validation',
                            'requested_samples':1, 'effective_samples':0, 'failures':1,
                            'created_utc':datetime.now(timezone.utc).isoformat(),
                            'runner_sha256':sha256(Path(__file__)),
                            'failure_type':type(error).__name__}, stream, indent=2)
                        stream.write('\n')
        print(f'GPU smoke failed: {type(error).__name__}; no research outcome.', file=sys.stderr)
        sys.exit(1)
