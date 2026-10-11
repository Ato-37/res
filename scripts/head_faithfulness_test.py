"""Test whether gradient-attributed Qwen attention heads support its answer.

The script selects the top-K heads by absolute gradient-times-activation at
the final prompt position, attenuates them, and compares their margin drop to
five layer-count- and activation-norm-matched control groups. Per-sample
outputs contain only anonymous ordinal keys and derived scores.
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
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import build_inputs, read_sampled_frames, write_frame_set
from faithfulness_utils import (
    bootstrap_mean_interval,
    build_matched_control_groups,
    format_choice_prompt,
    gradient_activation_scores,
    mean_t_interval,
    select_prediction_and_foil,
    select_top_heads,
    signflip_p_against,
)
from necessity_mask_test import decodable_video_info, sample_indices


LABELS = 'ABCDE'
DEFAULT_MAX_PIXELS = 256 * 28 * 28


def is_out_of_memory_error(exc):
    oom_types = tuple(
        error_type for error_type in (
            getattr(torch.cuda, 'OutOfMemoryError', None),
            getattr(torch, 'OutOfMemoryError', None),
        ) if isinstance(error_type, type)
    )
    return ((bool(oom_types) and isinstance(exc, oom_types))
            or 'out of memory' in str(exc).lower())


def parse_layers(spec, count):
    if spec.strip().lower() == 'all':
        return list(range(count))
    try:
        values = [int(value.strip()) for value in spec.split(',') if value.strip()]
    except ValueError as exc:
        raise ValueError(f'invalid layer list: {spec}') from exc
    if not values or len(values) != len(set(values)):
        raise ValueError('layers must be non-empty and unique')
    if any(value < 0 or value >= count for value in values):
        raise ValueError(f'layer index outside 0..{count - 1}')
    return values


def score_margin(logits, choice_token_ids, fixed_prediction=None):
    """Compute model-choice margin, optionally keeping baseline prediction fixed."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    ids = torch.tensor(choice_token_ids, device=logprobs.device)
    choice_scores = logprobs.index_select(0, ids)
    prediction, foil, _ = select_prediction_and_foil(
        choice_scores.detach().float().cpu().tolist(), fixed_prediction)
    return choice_scores[prediction] - choice_scores[foil], prediction, foil, choice_scores


def make_capture_hook(captured, layer_index, expected_width):
    def hook(module, module_inputs):
        if not module_inputs or module_inputs[0].ndim != 3:
            raise RuntimeError('o_proj input must have shape [batch, sequence, hidden]')
        value = module_inputs[0]
        if value.shape[-1] != expected_width:
            raise RuntimeError(f'o_proj width mismatch: expected {expected_width}, got {value.shape[-1]}')
        if layer_index in captured:
            # Non-reentrant checkpointing recomputes decoder layers during
            # backward. Do not replace or recapture the original activation.
            return None
        activation = value.clone()
        if not activation.requires_grad:
            activation.requires_grad_(True)
        captured[layer_index] = activation
        return (activation,) + tuple(module_inputs[1:])
    return hook


def make_attenuation_hook(heads, alpha, head_dim, sequence_index=-1):
    def hook(module, module_inputs):
        if not module_inputs or module_inputs[0].ndim != 3:
            raise RuntimeError('o_proj input must have shape [batch, sequence, hidden]')
        value = module_inputs[0]
        patched = value.clone()
        position = sequence_index if sequence_index >= 0 else patched.shape[1] + sequence_index
        if position < 0 or position >= patched.shape[1]:
            raise RuntimeError('intervention position is outside sequence')
        for head in heads:
            start, end = head * head_dim, (head + 1) * head_dim
            if end > patched.shape[-1]:
                raise RuntimeError('head slice exceeds o_proj input')
            patched[:, position, start:end] *= alpha
        return (patched,) + tuple(module_inputs[1:])
    return hook


def get_logits(model, inputs):
    # Qwen2.5-VL versions with logits_to_keep can avoid materializing logits for
    # every video token, which is important for the gradient pass on 7B.
    kwargs = dict(inputs)
    kwargs['use_cache'] = False
    import inspect
    try:
        supports_logits_to_keep = 'logits_to_keep' in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        supports_logits_to_keep = False
    output = (model(**kwargs, logits_to_keep=1) if supports_logits_to_keep
              else model(**kwargs))
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f'unexpected logits shape: {tuple(logits.shape)}')
    return logits[0, -1, :]


def attribute_heads(model, inputs, choice_token_ids, layers, num_heads, head_dim):
    language_layers = model.model.language_model.layers
    captured, handles = {}, []
    expected = num_heads * head_dim
    try:
        model.train()
        # Keep checkpointed layers in training mode while disabling stochastic
        # dropout so the gradient pass matches the evaluation-mode interventions.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        for layer_index in layers:
            attention = language_layers[layer_index].self_attn
            handles.append(attention.o_proj.register_forward_pre_hook(
                make_capture_hook(captured, layer_index, expected)))
        with torch.enable_grad():
            logits = get_logits(model, inputs)
            margin, prediction, foil, scores = score_margin(logits, choice_token_ids)
            if not torch.isfinite(margin):
                raise RuntimeError('baseline choice margin is not finite')
            if set(captured) != set(layers):
                raise RuntimeError(f'missing head activations: expected={layers}, got={sorted(captured)}')
            gradients = torch.autograd.grad(margin, tuple(captured[layer] for layer in layers),
                                            allow_unused=False, retain_graph=False)
        attributions, activation_norms = [], {}
        for layer_index, gradient in zip(layers, gradients):
            activation = captured[layer_index]
            if activation.shape[0] != 1 or activation.shape[1] < 1:
                raise RuntimeError('expected a single unpadded prompt')
            z = activation[0, -1, :].float().reshape(num_heads, head_dim)
            grad = gradient[0, -1, :].float().reshape(num_heads, head_dim)
            contributions, norms = gradient_activation_scores(
                z.detach().cpu().tolist(), grad.detach().cpu().tolist())
            for head, (score, norm) in enumerate(zip(contributions, norms)):
                if not math.isfinite(score) or not math.isfinite(norm):
                    raise RuntimeError('non-finite head attribution or activation norm')
                attributions.append({'layer': layer_index, 'head': head, 'score': float(score)})
                activation_norms[(layer_index, head)] = float(norm)
        return (float(margin.detach().float().item()), prediction, foil,
                [float(value) for value in scores.detach().float().cpu().tolist()],
                attributions, activation_norms)
    finally:
        for handle in handles:
            handle.remove()
        model.eval()


def intervention_margin(model, inputs, choice_token_ids, prediction, layer_heads,
                        alpha, head_dim):
    language_layers = model.model.language_model.layers
    grouped = {}
    for item in layer_heads:
        grouped.setdefault(int(item['layer']), []).append(int(item['head']))
    handles = []
    try:
        for layer_index, heads in grouped.items():
            attention = language_layers[layer_index].self_attn
            handles.append(attention.o_proj.register_forward_pre_hook(
                make_attenuation_hook(heads, alpha, head_dim)))
        with torch.no_grad():
            logits = get_logits(model, inputs)
            margin, _, _, _ = score_margin(logits, choice_token_ids, prediction)
        value = float(margin.float().item())
        if not math.isfinite(value):
            raise RuntimeError('intervened margin is not finite')
        return value
    finally:
        for handle in handles:
            handle.remove()


def validate_choice_tokens(tokenizer):
    ids = []
    for label in LABELS:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f'choice label {label} is not one token')
        ids.append(int(token_ids[0]))
    if len(set(ids)) != len(ids):
        raise ValueError('choice labels A-E must map to distinct tokens')
    return ids


def sample_key(index):
    return f'sample_{index:03d}'


def summarize_primary(values, seed, draws=10000):
    if not values:
        return {
            'n': 0, 'mean': None, 'sd': None, 'ci95_t': None,
            'primary_threshold_supported': False,
            'status': 'not_estimable_no_completed_samples',
        }
    summary = mean_t_interval(values)
    if len(values) > 1:
        from scipy.stats import ttest_1samp
        test = ttest_1samp(values, popmean=0.25, alternative='two-sided')
        summary.update({
            'ci95_t': summary.pop('ci'),
            'test_null_mean': 0.25,
            't_statistic_vs_threshold': float(test.statistic),
            'p_two_sided_vs_threshold': float(test.pvalue),
            'bootstrap_ci95_pointwise': bootstrap_mean_interval(values, draws, seed),
            'signflip_p_two_sided_vs_threshold': signflip_p_against(
                values, null=0.25, draws=draws, seed=seed + 1),
            'positive_fraction': sum(value > 0 for value in values) / len(values),
            'fraction_above_practical_threshold': sum(value > 0.25 for value in values) / len(values),
        })
        summary['primary_threshold_supported'] = summary['ci95_t'][0] > 0.25
    else:
        summary['ci95_t'] = None
        summary['primary_threshold_supported'] = False
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--control-repeats', type=int, default=5)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.top_k != 4:
        parser.error('The preregistered primary uses --top-k 4; change only for labeled exploratory runs')
    if args.alpha != 0.5:
        parser.error('The preregistered primary uses --alpha 0.5; change only for labeled exploratory runs')
    if args.control_repeats != 5:
        parser.error('The preregistered primary uses exactly five matched control groups')
    if args.num_frames != 16:
        parser.error('The preregistered primary uses 16 frames')
    if args.n_samples is not None and args.n_samples < 1:
        parser.error('--n-samples must be positive')
    if args.bootstrap_draws < 100:
        parser.error('--bootstrap-draws must be at least 100')
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be in [0,1]')
    if os.path.exists(args.output):
        parser.error(f'refusing to overwrite existing output: {args.output}')
    try:
        import scipy  # noqa: F401
    except ImportError:
        parser.error('SciPy is required for the preregistered Student-t interval; install scripts/requirements-spike.txt')

    samples = json.loads(open(args.samples_file).read())
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    if not samples:
        parser.error('sample manifest is empty')
    if any(len(sample.get('choices') or []) != 5 for sample in samples):
        parser.error('each sample must contain exactly five choices')
    if any(sample.get('answer_choice_index') not in range(5) for sample in samples):
        parser.error('each sample must have answer_choice_index in 0..4')
    if any(not sample.get('question') or not sample.get('video') for sample in samples):
        parser.error('each sample must contain a question and a video path')
    video_ids = [str(sample['video_id']) for sample in samples if sample.get('video_id') is not None]
    video_paths = [os.path.realpath(os.path.expanduser(sample['video'])) for sample in samples]
    if len(video_ids) != len(set(video_ids)) or len(video_paths) != len(set(video_paths)):
        parser.error('each independent sample must use a unique video')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    choice_token_ids = validate_choice_tokens(processor.tokenizer)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=quant, device_map=device)
    model.eval()
    # Only gradients with respect to hook-captured activations are required.
    # Freezing parameters prevents autograd from retaining weight-gradient data.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, 'gradient_checkpointing_enable'):
        parser.error('this Transformers model does not support gradient checkpointing')
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    except TypeError as exc:
        parser.error(f'non-reentrant gradient checkpointing is required: {exc}')
    if not getattr(model, 'is_gradient_checkpointing', False):
        parser.error('gradient checkpointing did not activate')
    language_model = model.model.language_model
    layers = parse_layers(args.layers, len(language_model.layers))
    config = language_model.config
    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    if num_heads < args.top_k or hidden_size % num_heads:
        parser.error(f'invalid head layout: hidden_size={hidden_size}, num_heads={num_heads}')
    head_dim = hidden_size // num_heads
    print(f'device={device} quant={args.quant} layers={len(layers)} heads_per_layer={num_heads} '
          f'gradient_checkpointing=non_reentrant frozen_weights=True cap_pixels_per_frame=True '
          f'top_k={args.top_k} alpha={args.alpha} control_groups={args.control_repeats}', flush=True)

    records, failures = [], []
    run_started = time.perf_counter()
    termination_reason = 'completed'
    for index, sample in enumerate(samples):
        key = sample_key(index)
        temp = tempfile.mkdtemp(prefix='head_faithfulness_')
        stage = 'video_decode'
        started = time.perf_counter()
        frames = None
        inputs = None
        try:
            video_path = os.path.expanduser(sample['video'])
            total_frames, _ = decodable_video_info(video_path)
            indices = sorted(set(sample_indices(total_frames, args.num_frames)))
            if len(indices) != args.num_frames:
                raise RuntimeError(f'expected {args.num_frames} unique sampled frames')
            frames = read_sampled_frames(video_path, total_frames, indices, args.frame_size)
            frame_paths = write_frame_set([frames[position] for position in indices],
                                          os.path.join(temp, 'full'))
            stage = 'input_preparation'
            prompt = format_choice_prompt(sample['question'], sample['choices'])
            inputs = build_inputs(processor, device, frame_paths, prompt,
                                  cap_pixels_per_frame=True)
            stage = 'gradient_attribution'
            baseline_margin, prediction, foil, choice_scores, attribution_rows, norms = attribute_heads(
                model, inputs, choice_token_ids, layers, num_heads, head_dim)
            if sum(item['score'] for item in attribution_rows) <= 0:
                raise RuntimeError('all head attributions are zero; explanation ranking is undefined')
            selected = select_top_heads(attribution_rows, args.top_k)
            stage = 'matched_controls'
            controls, match_distances = build_matched_control_groups(
                selected, norms, num_heads=num_heads,
                repeats=args.control_repeats, seed=args.seed + index * 1009)
            stage = 'target_intervention'
            target_margin = intervention_margin(
                model, inputs, choice_token_ids, prediction, selected, args.alpha, head_dim)
            stage = 'control_interventions'
            control_margins = [intervention_margin(
                model, inputs, choice_token_ids, prediction, group, args.alpha, head_dim)
                for group in controls]
            target_drop = baseline_margin - target_margin
            control_drops = [baseline_margin - value for value in control_margins]
            mean_control_drop = sum(control_drops) / len(control_drops)
            selective = target_drop - mean_control_drop
            record = {
                'sample_key': key,
                'baseline_margin': baseline_margin,
                'baseline_prediction_index': prediction,
                'baseline_foil_index': foil,
                'baseline_choice_correct': prediction == int(sample['answer_choice_index']),
                'selected_heads': selected,
                'selected_attribution_sum': sum(item['score'] for item in selected),
                'target_margin_after_intervention': target_margin,
                'target_margin_drop': target_drop,
                'control_groups': controls,
                'control_margin_drops': control_drops,
                'mean_control_margin_drop': mean_control_drop,
                'selective_margin_drop': selective,
                'mean_abs_log_activation_norm_ratio': sum(match_distances) / len(match_distances),
                'elapsed_seconds': time.perf_counter() - started,
            }
            records.append(record)
            print(f'{key} baseline_margin={baseline_margin:+.4f} '
                  f'target_drop={target_drop:+.4f} control_drop={mean_control_drop:+.4f} '
                  f'selective={selective:+.4f} heads={[(h["layer"], h["head"]) for h in selected]}',
                  flush=True)
            del inputs
        except Exception as exc:
            is_oom = is_out_of_memory_error(exc)
            failures.append({'sample_key': key, 'stage': stage, 'error_type': type(exc).__name__,
                             'elapsed_seconds': time.perf_counter() - started})
            print(f'WARNING: {key} failed at {stage} ({type(exc).__name__}); '
                  'details withheld from published output',
                  flush=True)
            if is_oom:
                termination_reason = 'stopped_after_out_of_memory'
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for pending_index in range(index + 1, len(samples)):
                    failures.append({
                        'sample_key': sample_key(pending_index),
                        'stage': 'not_run_after_out_of_memory',
                        'error_type': 'NotRunAfterOutOfMemory',
                        'elapsed_seconds': 0.0,
                    })
                break
        finally:
            frames = None
            inputs = None
            shutil.rmtree(temp, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    values = [row['selective_margin_drop'] for row in records]
    primary = summarize_primary(values, args.seed, args.bootstrap_draws)
    result = {
        'schema_version': 1,
        'experiment': 'gradient_activation_head_faithfulness',
        'research_status': 'smoke' if len(samples) <= 3 else 'not_automatically_classified',
        'model': args.model,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'quant': args.quant,
        'dtype': 'bfloat16',
        'attention_implementation': 'sdpa',
        'gradient_checkpointing': 'non_reentrant',
        'model_parameters_trainable': False,
        'cap_pixels_per_frame': True,
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'opencv': cv2.__version__,
        'seed': args.seed,
        'layers': layers,
        'num_heads_per_layer': num_heads,
        'head_dim': head_dim,
        'num_frames': args.num_frames,
        'top_k': args.top_k,
        'alpha': args.alpha,
        'control_repeats': args.control_repeats,
        'control_matching': 'exact layer histogram; nearest activation L2 norm in log-ratio; disjoint groups',
        'score_definition': 'log p(baseline predicted option) minus strongest foil option; prediction frozen after baseline',
        'primary_estimand': 'mean(target margin drop minus mean matched-control margin drop), one independent video per sample',
        'practical_threshold_nat': 0.25,
        'primary_summary': primary,
        'secondary_descriptives': {
            'n_baseline_correct': sum(row['baseline_choice_correct'] for row in records),
            'baseline_choice_accuracy': sum(row['baseline_choice_correct'] for row in records) / len(records),
            'note': 'Descriptive only; primary estimand includes every successful sample regardless of baseline correctness.',
        },
        'n_requested': len(samples),
        'n_effective': len(records),
        'n_failures': len(failures),
        'runtime_seconds': time.perf_counter() - run_started,
        'termination_reason': termination_reason,
        'failures': failures,
        'records': records,
        'limitations': [
            'Gradient-times-activation is a local attribution score, not itself a causal effect.',
            'Internal attenuation may move activations off the model data manifold; matched controls reduce but do not eliminate this concern.',
            'Choice-letter margin is specific to the fixed five-choice prompt and is not free-generation accuracy.',
            'Temporal grounding attention analysis is not included in this primary-only implementation.',
            'Smoke samples are engineering checks and are excluded from confirmatory inference.',
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    mean_text = 'NA' if primary['mean'] is None else f'{primary["mean"]:+.4f}'
    print(f'n_requested={len(samples)} n_effective={len(records)} failures={len(failures)} '
          f'mean_selective={mean_text} ci95_t={primary.get("ci95_t")}', flush=True)
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
