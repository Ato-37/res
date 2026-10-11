"""Sanity-check head attribution after randomizing the five choice-readout rows.

This is a limited last-layer parameter-randomization diagnostic, not full
transformer-layer randomization and not a faithfulness effect test.
"""

import argparse
import json
import math
import os
import platform
import shutil
import tempfile
import time

import cv2
import torch
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import build_inputs, read_sampled_frames, write_frame_set
from faithfulness_randomization_utils import compare_attribution_rankings
from head_faithfulness_test import (
    DEFAULT_MAX_PIXELS,
    LABELS,
    attribute_heads,
    parse_layers,
    sample_key,
)
from necessity_mask_test import decodable_video_info, sample_indices


def random_readout_hook(choice_token_ids, rows):
    """Replace only final-position A-E logits with logits from random readout rows."""
    def hook(module, module_inputs, output):
        if not module_inputs or module_inputs[0].ndim != 3 or output.ndim != 3:
            raise RuntimeError('unexpected language-model readout tensor shape')
        hidden = module_inputs[0]
        if hidden.shape[-1] != rows.shape[-1] or output.shape[0] != 1:
            raise RuntimeError('random readout dimensions do not match model output')
        replacement = hidden[:, -1, :].float() @ rows.to(hidden.device).T
        patched = output.clone()
        patched[0, -1, choice_token_ids] = replacement[0].to(patched.dtype)
        return patched
    return hook


def validate_choices(tokenizer):
    ids = []
    for label in LABELS:
        tokens = tokenizer(label, add_special_tokens=False).input_ids
        if len(tokens) != 1:
            raise ValueError(f'choice label {label} is not a single token')
        ids.append(int(tokens[0]))
    if len(set(ids)) != len(ids):
        raise ValueError('choice labels A-E must map to distinct tokens')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--randomizations', type=int, default=1)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.n_samples < 1 or args.randomizations < 1:
        parser.error('--n-samples and --randomizations must be positive')
    if args.num_frames != 16:
        parser.error('this diagnostic is fixed to the 16-frame faithfulness protocol')
    if os.path.exists(args.output):
        parser.error(f'refusing to overwrite existing output: {args.output}')

    samples = json.loads(open(args.samples_file).read())[:args.n_samples]
    if not samples or len(samples) != args.n_samples:
        parser.error(f'manifest has fewer than {args.n_samples} requested samples')
    if any(len(sample.get('choices') or []) != 5 or not sample.get('question')
           or not sample.get('video') for sample in samples):
        parser.error('each sample needs a question, video, and five choices')
    ids = [str(sample.get('video_id', sample['video'])) for sample in samples]
    if len(ids) != len(set(ids)):
        parser.error('sample videos must be unique')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    choice_token_ids = validate_choices(processor.tokenizer)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=quant, device_map=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, 'gradient_checkpointing_enable'):
        parser.error('model does not support gradient checkpointing')
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    if not getattr(model, 'is_gradient_checkpointing', False):
        parser.error('non-reentrant gradient checkpointing failed to activate')
    tied = bool(getattr(model.config, 'tie_word_embeddings', False))
    if tied:
        parser.error('output readout is tied to input embeddings; refusing ambiguous randomization')

    language_model = model.model.language_model
    layers = parse_layers('all', len(language_model.layers))
    config = language_model.config
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size) // num_heads
    if not hasattr(model, 'lm_head'):
        parser.error('model does not expose an lm_head output module')
    hidden_size = int(language_model.config.hidden_size)
    init_std = float(getattr(model.config, 'initializer_range', 0.02))
    print(f'device={device} quant={args.quant} layers={len(layers)} heads={num_heads} '
          f'randomizations={args.randomizations} readout_init_std={init_std}', flush=True)

    records, failures = [], []
    started_run = time.perf_counter()
    for index, sample in enumerate(samples):
        key = sample_key(index)
        temp = tempfile.mkdtemp(prefix='faithfulness_readout_randomization_')
        started = time.perf_counter()
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
            baseline_margin, baseline_prediction, _, _, baseline_rows, _ = attribute_heads(
                model, inputs, choice_token_ids, layers, num_heads, head_dim)
            baseline_map = {(row['layer'], row['head']): float(row['score'])
                            for row in baseline_rows}
            ordered_keys = sorted(baseline_map)
            baseline_scores = [baseline_map[item] for item in ordered_keys]
            if not math.isfinite(baseline_margin) or sum(baseline_scores) <= 0:
                raise RuntimeError('trained-model attribution is non-finite or all-zero')

            random_runs = []
            for replicate in range(args.randomizations):
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + index * 1009 + replicate)
                rows = torch.randn((len(choice_token_ids), hidden_size), generator=generator,
                                   device=device, dtype=torch.float32) * init_std
                hook = model.lm_head.register_forward_hook(
                    random_readout_hook(choice_token_ids, rows))
                try:
                    margin, prediction, _, _, randomized_rows, _ = attribute_heads(
                        model, inputs, choice_token_ids, layers, num_heads, head_dim)
                finally:
                    hook.remove()
                randomized_map = {(row['layer'], row['head']): float(row['score'])
                                  for row in randomized_rows}
                if set(randomized_map) != set(baseline_map):
                    raise RuntimeError('randomization changed the set of attributed heads')
                comparison = compare_attribution_rankings(
                    baseline_scores,
                    [randomized_map[item] for item in ordered_keys], top_k=4)
                comparison['randomized_prediction_index'] = int(prediction)
                comparison['randomized_margin'] = float(margin)
                random_runs.append(comparison)
            records.append({
                'sample_key': key,
                'baseline_prediction_index': int(baseline_prediction),
                'baseline_margin': float(baseline_margin),
                'randomized_readout_runs': random_runs,
                'mean_spearman_rho': sum(row['spearman_rho'] for row in random_runs) / len(random_runs),
                'mean_top4_overlap_fraction': sum(
                    row['top_k_overlap_fraction'] for row in random_runs) / len(random_runs),
                'elapsed_seconds': time.perf_counter() - started,
            })
            print(f'{key} readout_randomizations={len(random_runs)} '
                  f'mean_rho={records[-1]["mean_spearman_rho"]:+.4f} '
                  f'mean_top4_overlap={records[-1]["mean_top4_overlap_fraction"]:.3f}',
                  flush=True)
        except Exception as exc:
            failures.append({'sample_key': key, 'error_type': type(exc).__name__})
            print(f'WARNING: {key} failed ({type(exc).__name__}); details withheld', flush=True)
            if 'out of memory' in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for pending in range(index + 1, len(samples)):
                    failures.append({
                        'sample_key': sample_key(pending),
                        'error_type': 'NotRunAfterOutOfMemory',
                    })
                break
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    rho_values = [row['mean_spearman_rho'] for row in records]
    overlap_values = [row['mean_top4_overlap_fraction'] for row in records]
    result = {
        'schema_version': 1,
        'experiment': 'head_attribution_output_readout_randomization_sanity',
        'status': 'exploratory_sanity_check_not_faithfulness_test',
        'n_requested': len(samples), 'n_effective': len(records),
        'n_failures': len(failures), 'failures': failures, 'records': records,
        'randomizations_per_sample': args.randomizations,
        'randomized_parameter_scope': 'five A-E output-readout rows only; transformer layers unchanged',
        'initializer_std': init_std,
        'mean_sample_spearman_rho': sum(rho_values) / len(rho_values) if rho_values else None,
        'mean_sample_top4_overlap_fraction': sum(overlap_values) / len(overlap_values)
        if overlap_values else None,
        'seed': args.seed, 'model': args.model,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'quant': args.quant, 'dtype': 'bfloat16', 'device': device,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'opencv': cv2.__version__, 'python': platform.python_version(),
        'attention_implementation': 'sdpa',
        'gradient_checkpointing': 'non_reentrant',
        'num_frames': args.num_frames, 'layers': layers,
        'num_heads_per_layer': num_heads, 'top_k': 4,
        'runtime_seconds': time.perf_counter() - started_run,
        'limitations': [
            'Only the five choice-token output-readout rows are randomized; transformer layers are unchanged.',
            'This is a limited parameter-randomization sanity diagnostic, not the full layer-wise Adebayo test.',
            'Rank change does not establish causal faithfulness; rank stability is a warning signal, not a standalone rejection rule.',
            'The random readout rows use the model config initializer_range and may not match trained-row statistics.',
            'A-E choice-token margins are tied to the registered prompt and are not free-generation accuracy.',
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
