"""Screen singleton and grouped attention-head activation patches.

The hook is placed at ``self_attn.o_proj`` input, where Qwen exposes the
concatenated per-head context before the output projection. This is a
head-level diagnostic; it is not a deployable correction and assumes the
model's attention implementation exposes that layout.

Usage:
    python scripts/head_group_activation_patching.py \
        --samples-file data/nextgqa_samples_confirm_smoke.json --n-samples 3 \
        --layers 6,12 --groups singletons,all --self-check \
        --strict-grounding --output results/head_group_patch_smoke.json
"""

import argparse
import json
import math
import os
import shutil
import tempfile

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import (
    build_inputs, make_full_and_swapped_frames, read_sampled_frames,
    validate_alignment, write_frame_set,
)
from necessity_mask_test import (
    choose_control_indices, decodable_video_info, grounding_frame_indices, sample_indices,
)
from qwen_vl_utils import process_vision_info  # noqa: F401 (dependency check)
from soft_ablation_sweep import append_answer_tokens, sequence_answer_logprobs


def parse_groups(spec, n_heads):
    groups = []
    for item in spec.split(','):
        item = item.strip().lower()
        if not item:
            continue
        if item == 'singletons':
            groups.extend([[head] for head in range(n_heads)])
        elif item in ('adjacent', 'adjacent_pairs'):
            groups.extend([[head, head + 1] for head in range(n_heads - 1)])
        elif item in ('skip1', 'gap2'):
            groups.extend([[head, head + 2] for head in range(n_heads - 2)])
        elif item == 'all':
            groups.append(list(range(n_heads)))
        else:
            try:
                group = sorted(set(int(value) for value in item.split('+')))
            except ValueError as exc:
                raise ValueError(f'Invalid head group: {item}') from exc
            if not group or group[0] < 0 or group[-1] >= n_heads:
                raise ValueError(f'Head group outside 0..{n_heads - 1}: {item}')
            groups.append(group)
    unique = []
    seen = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            unique.append(group)
            seen.add(key)
    if not unique:
        raise ValueError('No head groups specified')
    return unique


def patch_head_bounds(scope, prompt_length, total_length):
    if scope == 'all':
        return 0, total_length
    if scope == 'prompt':
        return 0, prompt_length
    if scope == 'answer':
        return prompt_length, total_length
    raise ValueError(f'Unknown patch scope: {scope}')


def parse_layer_spec(spec, n_layers):
    spec = spec.strip().lower()
    if spec == 'all':
        return list(range(n_layers))
    try:
        layers = [int(value) for value in spec.split(',') if value.strip()]
    except ValueError as exc:
        raise ValueError(f'Invalid layer specification: {spec}') from exc
    if not layers or any(layer < 0 or layer >= n_layers for layer in layers):
        raise ValueError(f'Layer outside 0..{n_layers - 1}: {spec}')
    if len(set(layers)) != len(layers):
        raise ValueError(f'Duplicate layer in specification: {spec}')
    return layers


def capture_o_proj_inputs(model, inputs, layers):
    captured = {}
    handles = []

    def make_hook(layer_index):
        def hook(module, module_inputs):
            if not module_inputs:
                raise RuntimeError('o_proj pre-hook received no inputs')
            captured[layer_index] = module_inputs[0].detach().clone()
        return hook

    for layer_index in layers:
        attention = model.model.language_model.layers[layer_index].self_attn
        if not hasattr(attention, 'o_proj'):
            raise RuntimeError(f'Layer {layer_index} has no self_attn.o_proj')
        handles.append(attention.o_proj.register_forward_pre_hook(make_hook(layer_index)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f'Missing o_proj captures: expected={layers}, got={sorted(captured)}')
    return captured


def patched_head_score(model, inputs, answer_ids, layer_index, source, heads,
                       num_heads, head_dim, scope):
    attention = model.model.language_model.layers[layer_index].self_attn
    prompt_length = inputs['input_ids'].shape[1]
    total_length = prompt_length + len(answer_ids)
    start, end = patch_head_bounds(scope, prompt_length, total_length)
    calls = 0

    def patch_hook(module, module_inputs):
        nonlocal calls
        calls += 1
        if not module_inputs:
            raise RuntimeError('o_proj pre-hook received no inputs')
        value = module_inputs[0]
        if value.shape != source.shape or value.dtype != source.dtype or value.device != source.device:
            raise ValueError('o_proj source/recipient shape, dtype or device mismatch')
        if value.shape[-1] < num_heads * head_dim:
            raise ValueError('o_proj input is smaller than the configured head layout')
        patched = value.clone()
        for head in heads:
            left, right = head * head_dim, (head + 1) * head_dim
            patched[:, start:end, left:right] = source[:, start:end, left:right]
        return (patched,) + tuple(module_inputs[1:])

    handle = attention.o_proj.register_forward_pre_hook(patch_hook)
    try:
        from activation_patching_sweep import sequence_score
        score = sequence_score(model, inputs, answer_ids)
    finally:
        handle.remove()
    if calls != 1 or not math.isfinite(score):
        raise RuntimeError(f'Invalid head patch forward: hook calls={calls}, score={score}')
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--layers', default='6,12')
    parser.add_argument('--groups', default='singletons,all')
    parser.add_argument('--patch-scope', choices=['all', 'prompt', 'answer'], default='all')
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--self-check-atol', type=float, default=1e-4)
    parser.add_argument('--strict-grounding', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not math.isfinite(args.self_check_atol) or args.self_check_atol < 0:
        parser.error('--self-check-atol must be finite and nonnegative')
    samples = json.loads(open(args.samples_file).read())
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    for sample in samples:
        sample['video'] = os.path.expanduser(sample['video'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant_config = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16} if args.quant == '4bit'
           else {'load_in_8bit': True})
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant_config, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    model_layers = model.model.language_model.layers
    layers_to_run = parse_layer_spec(args.layers, len(model_layers))
    config = model.model.language_model.config
    num_heads = int(getattr(config, 'num_attention_heads', 0))
    if not num_heads:
        raise RuntimeError('Could not read num_attention_heads from language model config')
    hidden_size = int(getattr(config, 'hidden_size', 0))
    if not hidden_size or hidden_size % num_heads:
        raise RuntimeError(f'Invalid head layout hidden_size={hidden_size}, heads={num_heads}')
    head_dim = hidden_size // num_heads
    groups = parse_groups(args.groups, num_heads)
    print(f'device={device}, quant={args.quant}, layers={layers_to_run}, heads={num_heads}, groups={len(groups)}')

    records, failures = [], []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get('qid', sample.get('video_id', sample_index)))
        temp_dir = tempfile.mkdtemp(prefix='head_group_patch_')
        try:
            total, fps = decodable_video_info(sample['video'])
            indices = sample_indices(total, args.num_frames)
            timestamps = [index / fps for index in indices]
            evidence = grounding_frame_indices(indices, timestamps, sample['grounding'])
            if args.strict_grounding:
                evidence = [i for i, t in zip(indices, timestamps)
                            if any(a <= t <= b for a, b in sample['grounding'])]
            control = choose_control_indices(indices, evidence)
            if not evidence or not control:
                raise RuntimeError(f'Could not construct evidence/control swaps: evidence={evidence}, control={control}')
            frame_map = read_sampled_frames(sample['video'], total, indices, args.frame_size)
            full_images, swapped_images = make_full_and_swapped_frames(frame_map, indices, evidence, control)
            full_paths = write_frame_set(full_images, os.path.join(temp_dir, 'full'))
            swapped_paths = write_frame_set(swapped_images, os.path.join(temp_dir, 'swapped'))
            full_inputs = build_inputs(processor, device, full_paths, sample['question'])
            swapped_inputs = build_inputs(processor, device, swapped_paths, sample['question'])
            validate_alignment(full_inputs, swapped_inputs)
            answer_ids = processor.tokenizer(sample['answer'], add_special_tokens=False).input_ids
            if not answer_ids:
                raise ValueError('Empty answer tokens')
            full_sequence = append_answer_tokens(full_inputs, answer_ids)
            swapped_sequence = append_answer_tokens(swapped_inputs, answer_ids)
            full_source = capture_o_proj_inputs(model, full_sequence, layers_to_run)
            full_score = __import__('activation_patching_sweep', fromlist=['sequence_score']).sequence_score(model, full_inputs, answer_ids)
            swapped_score = __import__('activation_patching_sweep', fromlist=['sequence_score']).sequence_score(model, swapped_inputs, answer_ids)
            checks = []
            if args.self_check:
                swapped_source = capture_o_proj_inputs(model, swapped_sequence, layers_to_run)
                for name, prompt, source in (('full', full_inputs, full_source), ('swapped', swapped_inputs, swapped_source)):
                    baseline = full_score if name == 'full' else swapped_score
                    for layer in layers_to_run:
                        for group in groups:
                            checked = patched_head_score(model, prompt, answer_ids, layer, source[layer], group, num_heads, head_dim, args.patch_scope)
                            error = abs(checked - baseline)
                            if error > args.self_check_atol:
                                raise RuntimeError(f'Self-patch failed: {name} layer={layer} group={group} error={error}')
                            checks.append({'input': name, 'layer': layer, 'heads': group, 'absolute_error': error})
            record = {'sample_id': sample_id, 'grounding': sample['grounding'], 'sampled_indices': indices,
                      'sampled_timestamps': timestamps, 'evidence_indices': evidence, 'control_indices': control,
                      'full_sequence_logprob': full_score, 'swapped_sequence_logprob': swapped_score,
                      'swap_drop': swapped_score - full_score, 'self_checks': checks, 'layers': {}}
            for layer in layers_to_run:
                record['layers'][str(layer)] = {}
                for group in groups:
                    patched = patched_head_score(model, swapped_inputs, answer_ids, layer, full_source[layer], group, num_heads, head_dim, args.patch_scope)
                    key = '+'.join(map(str, group))
                    record['layers'][str(layer)][key] = {'heads': group, 'patched_sequence_logprob': patched,
                        'patch_rescue': patched - swapped_score,
                        'rescue_fraction': ((patched - swapped_score) / (full_score - swapped_score)
                                            if full_score != swapped_score else None)}
            records.append(record)
            if args.self_check:
                print(f'sample={sample_id} self_check=PASS checks={len(checks)}')
            print(f'sample={sample_id} full={full_score:.4f} swapped={swapped_score:.4f} swap_drop={record["swap_drop"]:+.4f}')
        except Exception as exc:
            failures.append({'id': sample_id, 'video': sample.get('video'), 'error': repr(exc)})
            print(f'WARNING: skipping sample {sample_id}: {exc}')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    result = {'script': 'head_group_activation_patching.py', 'samples_file': args.samples_file,
              'model': args.model, 'quant': args.quant, 'dtype': 'bfloat16', 'attn_implementation': 'eager',
              'layers': layers_to_run, 'groups_spec': args.groups, 'groups': groups, 'num_heads': num_heads,
              'head_dim': head_dim, 'patch_scope': args.patch_scope, 'strict_grounding': args.strict_grounding,
              'self_check': args.self_check, 'self_check_atol': args.self_check_atol,
              'self_check_checks_per_record': (2 * len(layers_to_run) * len(groups)
                                               if args.self_check else 0),
              'n_requested': len(samples), 'n_effective': len(records), 'failures': failures, 'records': records,
              'warning': 'Head patching at o_proj input is a diagnostic intervention; validate layout and OOD swap limits.'}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
