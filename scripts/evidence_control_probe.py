"""Compare grounded-frame replacement with dose-matched non-evidence controls.

The primary estimand is the per-video paired contrast averaged across matched
single-frame interventions:
  mean(full score - evidence-frame-replaced score)
  - mean(full score - non-evidence-frame-replaced score).
Positive values mean the grounded-frame intervention reduced correct-answer
sequence log-probability more than an equally sized control perturbation.
This tests input-level temporal specificity, not head causality or general XAI
validity. Both arms replace one target frame with a non-grounded donor frame;
only the target location differs (grounded versus nearby non-grounded).
"""
import argparse
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from activation_patching_sweep import (
    build_inputs, read_sampled_frames, sequence_score, validate_alignment,
    write_frame_set,
)
from closed_research_branch import reject_inference
from necessity_mask_test import decodable_video_info, sample_indices


def strict_evidence_indices(indices, fps, intervals):
    return [index for index in indices
            if any(start <= index / fps <= end for start, end in intervals)]


def single_frame_replacement(frame_map, indices, target, donor):
    frames = [frame_map[index].copy() for index in indices]
    frames[indices.index(target)] = frame_map[donor].copy()
    return frames


def matched_single_frame_pairs(indices, evidence, repeats, seed):
    """Choose one evidence target/donor and one nearby control target/donor."""
    rng = random.Random(seed)
    controls = sorted(set(indices) - set(evidence))
    if not evidence or len(controls) < 2:
        return None
    evidence_order = list(evidence)
    rng.shuffle(evidence_order)
    pairs = []
    for repeat in range(repeats):
        evidence_target = evidence_order[repeat % len(evidence_order)]
        evidence_donor = min(controls, key=lambda idx: (abs(idx - evidence_target), idx))
        candidates = sorted(controls, key=lambda idx: (abs(idx - evidence_target), idx))
        # Randomize within the nearest few eligible controls while matching
        # the temporal distance to the evidence target as closely as possible.
        near = candidates[:min(4, len(candidates))]
        placebo_target = rng.choice(near)
        placebo_donors = [idx for idx in controls if idx != placebo_target]
        placebo_donor = min(placebo_donors,
                            key=lambda idx: (abs(idx - placebo_target), idx))
        pairs.append({'evidence_target': evidence_target,
                      'evidence_donor': evidence_donor,
                      'placebo_target': placebo_target,
                      'placebo_donor': placebo_donor})
    return pairs


def analyze_records(records, n_boot=10000, seed=0, n_permutations=10000):
    contrasts = [r['selective_evidence_drop'] for r in records]
    if len(contrasts) < 2 or len({str(r['sample_id']) for r in records}) != len(records):
        raise ValueError('Need at least two unique successful samples')
    if n_boot < 100 or n_permutations < 100:
        raise ValueError('Use at least 100 bootstrap and permutation draws')
    mean = sum(contrasts) / len(contrasts)
    rng = random.Random(seed)
    boot = sorted(sum(contrasts[rng.randrange(len(contrasts))]
                     for _ in contrasts) / len(contrasts) for _ in range(n_boot))
    def quantile(p):
        x = (len(boot) - 1) * p
        lo, hi = int(x), min(int(x) + 1, len(boot) - 1)
        return boot[lo] + (x - lo) * (boot[hi] - boot[lo])
    observed_sd = torch.tensor(contrasts, dtype=torch.float64).std(unbiased=True).item()
    observed_se = observed_sd / math.sqrt(len(contrasts))
    observed_t = (mean / observed_se if observed_se else
                  (math.copysign(float('inf'), mean) if mean else 0.0))
    extreme = 0
    for _ in range(n_permutations):
        signs = [rng.choice((-1, 1)) for _ in contrasts]
        perm = [x * sign for x, sign in zip(contrasts, signs)]
        sd = torch.tensor(perm, dtype=torch.float64).std(unbiased=True).item()
        perm_mean = sum(perm) / len(perm)
        perm_se = sd / math.sqrt(len(perm))
        stat = (perm_mean / perm_se if perm_se else
                (math.copysign(float('inf'), perm_mean) if perm_mean else 0.0))
        extreme += abs(stat) >= abs(observed_t)
    return {'n': len(contrasts), 'mean_selective_evidence_drop': mean,
            'sd': torch.tensor(contrasts, dtype=torch.float64).std(unbiased=True).item(),
            'bootstrap_ci95_pointwise': [quantile(.025), quantile(.975)],
            'positive_fraction': sum(x > 0 for x in contrasts) / len(contrasts),
            'paired_sign_flip_t': observed_t,
            'paired_sign_flip_p_two_sided': (1 + extreme) / (n_permutations + 1),
            'n_permutations': n_permutations,
            'assumption': 'paired sample-level contrast is symmetric under the null'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--placebo-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reject_inference(parser, "legacy_evidence_control",
                     "this exploratory single-frame protocol is not the current preregistered estimand")
    if args.placebo_repeats < 1 or args.num_frames < 2:
        parser.error('--placebo-repeats must be >=1 and --num-frames >=2')
    samples = json.loads(Path(args.samples_file).read_text())
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    for sample in samples:
        sample['video'] = os.path.expanduser(sample['video'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    records, failures = [], []
    for sample_index, sample in enumerate(samples):
        if sample.get('video_id') is not None and sample.get('qid') is not None:
            sample_id = f"{sample['video_id']}:{sample['qid']}"
        else:
            sample_id = str(sample.get('video_id', sample_index))
        temp_dir = tempfile.mkdtemp(prefix='evidence_control_probe_')
        try:
            total, fps = decodable_video_info(sample['video'])
            # Short clips can yield duplicate indices from evenly spaced
            # sampling. Deduplicate so a declared one-frame intervention
            # really changes one distinct temporal slot.
            indices = sorted(set(sample_indices(total, args.num_frames)))
            if len(indices) < 3:
                raise RuntimeError(f'Need at least three distinct sampled frames; got {indices}')
            evidence = strict_evidence_indices(indices, fps, sample['grounding'])
            pairs = matched_single_frame_pairs(
                indices, evidence, args.placebo_repeats,
                args.seed + sample_index * 1009)
            if pairs is None:
                raise RuntimeError(f'Cannot construct matched one-frame perturbations: '
                                   f'evidence={evidence}, non_evidence={len(set(indices)-set(evidence))}')
            frame_map = read_sampled_frames(sample['video'], total, indices, args.frame_size)
            full_images = [frame_map[i].copy() for i in indices]
            full_paths = write_frame_set(full_images, os.path.join(temp_dir, 'full'))
            full_inputs = build_inputs(processor, device, full_paths, sample['question'])
            answer_ids = processor.tokenizer(sample['answer'], add_special_tokens=False).input_ids
            if not answer_ids:
                raise RuntimeError('Empty answer tokenization')
            full_score = sequence_score(model, full_inputs, answer_ids)
            evidence_scores, placebo_scores = [], []
            for repeat, pair in enumerate(pairs):
                evidence_images = single_frame_replacement(
                    frame_map, indices, pair['evidence_target'], pair['evidence_donor'])
                placebo_images = single_frame_replacement(
                    frame_map, indices, pair['placebo_target'], pair['placebo_donor'])
                evidence_paths = write_frame_set(
                    evidence_images, os.path.join(temp_dir, f'evidence_{repeat}'))
                placebo_paths = write_frame_set(
                    placebo_images, os.path.join(temp_dir, f'placebo_{repeat}'))
                evidence_inputs = build_inputs(processor, device, evidence_paths, sample['question'])
                placebo_inputs = build_inputs(processor, device, placebo_paths, sample['question'])
                validate_alignment(full_inputs, evidence_inputs)
                validate_alignment(full_inputs, placebo_inputs)
                evidence_scores.append(sequence_score(model, evidence_inputs, answer_ids))
                placebo_scores.append(sequence_score(model, placebo_inputs, answer_ids))
            evidence_mean = sum(evidence_scores) / len(evidence_scores)
            placebo_mean = sum(placebo_scores) / len(placebo_scores)
            evidence_drop = full_score - evidence_mean
            placebo_drop = full_score - placebo_mean
            contrast = evidence_drop - placebo_drop
            if not all(math.isfinite(value) for value in
                       [full_score, *evidence_scores, *placebo_scores, contrast]):
                raise RuntimeError('Nonfinite model score')
            record = {'sample_id': sample_id, 'grounding': sample['grounding'],
                      'sampled_indices': indices, 'evidence_indices': evidence,
                      'matched_interventions': pairs, 'full_sequence_logprob': full_score,
                      'evidence_replaced_sequence_logprobs': evidence_scores,
                      'placebo_sequence_logprobs': placebo_scores,
                      'evidence_drop': evidence_drop,
                      'mean_placebo_drop': placebo_drop,
                      'selective_evidence_drop': contrast,
                      'edited_positions_per_intervention': 1}
            records.append(record)
            print(f'sample={sample_id} evidence_drop={record["evidence_drop"]:+.4f} '
                  f'placebo_drop={record["mean_placebo_drop"]:+.4f} '
                  f'selective={contrast:+.4f}', flush=True)
        except Exception as exc:
            failures.append({'id': sample_id, 'video': sample.get('video'), 'error': repr(exc)})
            print(f'WARNING: skipping sample {sample_id}: {exc}', flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    summary = analyze_records(records, args.n_boot, args.seed, args.n_permutations) if len(records) >= 2 else None
    result = {'schema_version': 1, 'script': 'evidence_control_probe.py',
              'samples_file': args.samples_file, 'model': args.model, 'quant': args.quant,
              'dtype': 'bfloat16', 'attn_implementation': 'eager',
              'n_requested': len(samples), 'n_effective': len(records),
              'paired_repeats_per_sample': args.placebo_repeats, 'num_frames': args.num_frames,
              'estimand': 'mean[(full-evidence_replaced) - (full-mean_placebo)]',
              'interpretation': 'Positive means one-frame evidence-position replacement reduced '
                                'correct-answer sequence log-probability more than the same replacement '
                                'at nearby non-evidence positions. '
                                'Input-level specificity only; not head causality. Correct-answer logprob '
                                'is not a choice margin; the current sample manifest has no foil choices.',
              'summary': summary, 'failures': failures, 'records': records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'n_requested={len(samples)} n_effective={len(records)} failures={len(failures)}')
    if summary:
        print(f"mean_selective={summary['mean_selective_evidence_drop']:+.4f} "
              f"ci95={summary['bootstrap_ci95_pointwise']} "
              f"signflip_p={summary['paired_sign_flip_p_two_sided']:.5f}")
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
