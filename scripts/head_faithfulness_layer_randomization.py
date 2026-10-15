"""Cascading layer-parameter randomization sanity check for head attribution.

Randomizes complete Qwen2.5-VL language decoder blocks from last to first,
keeping each randomized block perturbed for subsequent cumulative conditions.
This is a parameter-randomization diagnostic, not a faithfulness effect test.
"""

import argparse
import json
import math
import os
import platform
import shutil
import tempfile
import time

import bitsandbytes as bnb
import cv2
import torch
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import build_inputs, read_sampled_frames, write_frame_set
from faithfulness_randomization_utils import compare_attribution_rankings
from faithfulness_layer_randomization_utils import summarize_cumulative_records
from head_faithfulness_test import (
    DEFAULT_MAX_PIXELS,
    LABELS,
    attribute_heads,
    parse_layers,
    sample_key,
    validate_choice_tokens,
)
from necessity_mask_test import decodable_video_info, sample_indices
from closed_research_branch import reject_inference


def randomize_decoder_layer(layer, initializer_std, generator):
    """Reinitialize one decoder block in place and re-encode its 4-bit linears."""
    linear_modules = norm_parameters = 0
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules += 1
                weight = module.weight
                quant_state = getattr(weight, 'quant_state', None)
                if quant_state is not None:
                    random_weight = torch.empty(
                        (module.out_features, module.in_features),
                        device=weight.device, dtype=torch.bfloat16)
                    random_weight.normal_(mean=0.0, std=initializer_std,
                                          generator=generator)
                    quant_args = {
                        'blocksize': weight.blocksize,
                        'compress_statistics': weight.compress_statistics,
                        'quant_type': weight.quant_type,
                    }
                    try:
                        quantized, new_quant_state = bnb.functional.quantize_4bit(
                            random_weight, quant_storage=weight.quant_storage, **quant_args)
                    except TypeError as exc:
                        if 'quant_storage' not in str(exc):
                            raise
                        quantized, new_quant_state = bnb.functional.quantize_4bit(
                            random_weight, **quant_args)
                    weight.data = quantized
                    weight.quant_state = new_quant_state
                    module.quant_state = new_quant_state
                else:
                    weight.normal_(mean=0.0, std=initializer_std, generator=generator)
                if module.bias is not None:
                    module.bias.zero_()
            elif 'RMSNorm' in module.__class__.__name__ or 'LayerNorm' in module.__class__.__name__:
                for name, parameter in module.named_parameters(recurse=False):
                    if name == 'weight':
                        parameter.fill_(1.0)
                    elif name == 'bias':
                        parameter.zero_()
                    norm_parameters += parameter.numel()
    if linear_modules == 0:
        raise RuntimeError('decoder layer contained no linear modules')
    return {'linear_modules': linear_modules, 'norm_parameters': norm_parameters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--max-randomized-layers', type=int,
                        help='limit the smoke to this many final decoder layers')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reject_inference(parser, "closed_head_faithfulness",
                     "post-confirmation layer search is closed")
    if args.n_samples < 1:
        parser.error('--n-samples must be positive')
    if args.num_frames != 16:
        parser.error('this diagnostic is fixed to the 16-frame faithfulness protocol')
    if args.max_randomized_layers is not None and args.max_randomized_layers < 1:
        parser.error('--max-randomized-layers must be positive')
    if os.path.exists(args.output):
        parser.error(f'refusing to overwrite existing output: {args.output}')

    samples = json.loads(open(args.samples_file).read())[:args.n_samples]
    if not samples or len(samples) != args.n_samples:
        parser.error(f'manifest has fewer than {args.n_samples} requested samples')
    if any(len(sample.get('choices') or []) != 5 or not sample.get('question')
           or not sample.get('video') for sample in samples):
        parser.error('each sample needs a question, video, and five choices')
    identities = [str(sample.get('video_id', sample['video'])) for sample in samples]
    if len(set(identities)) != len(identities):
        parser.error('sample videos must be unique')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        parser.error('layer randomization requires the registered CUDA/4-bit environment')
    quant = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    choice_token_ids = validate_choice_tokens(processor.tokenizer)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=quant, device_map=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    if not getattr(model, 'is_gradient_checkpointing', False):
        parser.error('non-reentrant gradient checkpointing failed to activate')
    if bool(getattr(model.config, 'tie_word_embeddings', False)):
        parser.error('tied embeddings are unsupported for this diagnostic')

    language_model = model.model.language_model
    layer_modules = language_model.layers
    all_layers = parse_layers('all', len(layer_modules))
    randomized_order = list(reversed(all_layers))
    if args.max_randomized_layers is not None:
        if args.max_randomized_layers > len(all_layers):
            parser.error('--max-randomized-layers exceeds model depth')
        randomized_order = randomized_order[:args.max_randomized_layers]
    text_config = language_model.config
    num_heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.hidden_size) // num_heads
    initializer_std = float(getattr(text_config, 'initializer_range', 0.02))
    print(f'device={device} quant=4bit layers={len(all_layers)} heads={num_heads} '
          f'randomized_layers={len(randomized_order)} order=last_to_first '
          f'initializer_std={initializer_std}', flush=True)

    failures = []
    started_run = time.perf_counter()
    active = {}
    temp_dirs = []
    try:
        for index, sample in enumerate(samples):
            key = sample_key(index)
            temp = tempfile.mkdtemp(prefix='faithfulness_layer_randomization_')
            temp_dirs.append(temp)
            try:
                video = os.path.expanduser(sample['video'])
                total_frames, _ = decodable_video_info(video)
                indices = sorted(set(sample_indices(total_frames, args.num_frames)))
                if len(indices) != args.num_frames:
                    raise RuntimeError('could not select the registered number of unique frames')
                frames = read_sampled_frames(video, total_frames, indices, args.frame_size)
                frame_paths = write_frame_set([frames[position] for position in indices],
                                              os.path.join(temp, 'frames'))
                prompt = (f"{sample['question']}\nOptions:\n" +
                          '\n'.join(f'{label}. {choice}'
                                    for label, choice in zip(LABELS, sample['choices'])) +
                          '\nChoose the best answer. Respond with one letter only.\nAnswer:')
                inputs = build_inputs(processor, device, frame_paths, prompt,
                                      cap_pixels_per_frame=True)
                input_cpu = {name: value.cpu() if torch.is_tensor(value) else value
                             for name, value in inputs.items()}
                baseline_margin, baseline_prediction, _, _, baseline_rows, _ = attribute_heads(
                    model, inputs, choice_token_ids, all_layers, num_heads, head_dim)
                baseline_map = {(row['layer'], row['head']): float(row['score'])
                                for row in baseline_rows}
                baseline_keys = sorted(baseline_map)
                baseline_scores = [baseline_map[item] for item in baseline_keys]
                if not math.isfinite(baseline_margin) or sum(baseline_scores) <= 0:
                    raise RuntimeError('trained-model attribution is non-finite or all-zero')
                active[key] = {
                    'inputs': input_cpu,
                    'baseline_keys': baseline_keys,
                    'baseline_scores': baseline_scores,
                    'baseline_prediction_index': int(baseline_prediction),
                    'baseline_margin': float(baseline_margin),
                    'cumulative_conditions': [],
                }
                del inputs
            except Exception as exc:
                failures.append({'sample_key': key, 'error_type': type(exc).__name__})
                print(f'WARNING: {key} baseline failed ({type(exc).__name__}); details withheld',
                      flush=True)
                if 'out of memory' in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for pending in range(index + 1, len(samples)):
                        failures.append({'sample_key': sample_key(pending),
                                         'error_type': 'NotRunAfterOutOfMemory'})
                    for pending_key in active:
                        failures.append({'sample_key': pending_key,
                                         'error_type': 'NotRunAfterOutOfMemory'})
                    active.clear()
                    break

        randomized_layer_counts = []
        for step, layer_index in enumerate(randomized_order, start=1):
            if not active:
                break
            try:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + layer_index)
                randomized_layer_counts.append(randomize_decoder_layer(
                    layer_modules[layer_index], initializer_std, generator))
                for key in list(active):
                    item = active[key]
                    inputs = {name: value.to(device) if torch.is_tensor(value) else value
                              for name, value in item['inputs'].items()}
                    try:
                        margin, prediction, _, _, randomized_rows, _ = attribute_heads(
                            model, inputs, choice_token_ids, all_layers, num_heads, head_dim)
                        randomized_map = {(row['layer'], row['head']): float(row['score'])
                                          for row in randomized_rows}
                        if set(randomized_map) != set(item['baseline_keys']):
                            raise RuntimeError('randomization changed the set of attributed heads')
                        comparison = compare_attribution_rankings(
                            item['baseline_scores'],
                            [randomized_map[index] for index in item['baseline_keys']], top_k=4)
                        item['cumulative_conditions'].append({
                            'randomized_layer': int(layer_index),
                            'cumulative_randomized_layers': step,
                            'spearman_rho': comparison['spearman_rho'],
                            'top4_overlap_count': comparison['top_k_overlap_count'],
                            'top4_overlap_fraction': comparison['top_k_overlap_fraction'],
                            'randomized_prediction_index': int(prediction),
                            'prediction_changed': bool(
                                prediction != item['baseline_prediction_index']),
                            'randomized_margin': float(margin),
                        })
                        del inputs
                    except Exception as exc:
                        failures.append({'sample_key': key, 'error_type': type(exc).__name__})
                        del active[key]
                        print(f'WARNING: {key} failed at depth {step} '
                              f'({type(exc).__name__}); details withheld', flush=True)
                        if 'out of memory' in str(exc).lower():
                            del inputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            for pending_key in active:
                                failures.append({'sample_key': pending_key,
                                                 'error_type': 'NotRunAfterOutOfMemory'})
                            active.clear()
                            raise StopIteration
            except StopIteration:
                break
            except Exception as exc:
                for key in list(active):
                    failures.append({'sample_key': key, 'error_type': type(exc).__name__})
                active.clear()
                print(f'WARNING: layer {layer_index} randomization failed '
                      f'({type(exc).__name__}); details withheld', flush=True)
                break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        records = []
        for key, item in active.items():
            if len(item['cumulative_conditions']) != len(randomized_order):
                failures.append({'sample_key': key, 'error_type': 'IncompleteRandomizationSequence'})
                continue
            records.append({
                'sample_key': key,
                'baseline_prediction_index': item['baseline_prediction_index'],
                'baseline_margin': item['baseline_margin'],
                'randomized_layer_order': randomized_order,
                'cumulative_conditions': item['cumulative_conditions'],
                'elapsed_seconds': time.perf_counter() - started_run,
            })
    finally:
        for temp in temp_dirs:
            shutil.rmtree(temp, ignore_errors=True)

    result = {
        'schema_version': 1,
        'experiment': 'head_attribution_cascading_layer_randomization',
        'status': 'exploratory_sanity_check_not_faithfulness_test',
        'n_requested': len(samples), 'n_effective': len(records),
        'n_failures': len(failures), 'failures': failures, 'records': records,
        'randomized_layer_order': randomized_order,
        'layer_reinitialization_counts': [
            {'layer': int(layer), **counts}
            for layer, counts in zip(randomized_order, randomized_layer_counts)
        ],
        'descriptive_by_randomized_depth': summarize_cumulative_records(
            records, len(randomized_order)) if records else [],
        'randomization_mode': 'cumulative decoder-block reinitialization, last-to-first',
        'randomized_parameters': 'all decoder-block linear weights/biases and RMS/LayerNorm affine parameters',
        'initializer_std': initializer_std,
        'seed': args.seed, 'model': args.model,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'quant': '4bit', 'dtype': 'bfloat16', 'device': device,
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'bitsandbytes': bnb.__version__, 'opencv': cv2.__version__,
        'python': platform.python_version(),
        'attention_implementation': 'sdpa',
        'gradient_checkpointing': 'non_reentrant',
        'num_frames': args.num_frames,
        'num_layers_total': len(all_layers), 'num_heads_per_layer': num_heads,
        'heads_per_ranking': len(all_layers) * num_heads, 'top_k': 4,
        'runtime_seconds': time.perf_counter() - started_run,
        'limitations': [
            'Decoder block parameters are reset from the configured initialization distribution, not restored from a trained checkpoint distribution.',
            'Cumulative last-to-first randomization is a saliency sanity check, not a causal faithfulness test.',
            'The primary comparison uses only three validation-development videos and is descriptive.',
            'Quantized randomized weights are re-encoded as NF4/FP4 according to each module quantization metadata; quantization itself can affect the perturbation.',
            'A-E choice-token margins are not free-generation accuracy.',
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'n_requested={len(samples)} n_effective={len(records)} n_failures={len(failures)}', flush=True)
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
