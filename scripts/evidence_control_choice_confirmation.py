"""Fresh-sample confirmation of evidence-vs-placebo effects on choice margin.

The preregistered primary cohort is videos/questions where the unperturbed
multiple-choice prompt predicts the dataset gold option. Outcomes are still
reported for all samples and baseline-incorrect samples as secondary results.
"""
import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import read_sampled_frames, write_frame_set, build_inputs
from evidence_control_probe import matched_single_frame_pairs, strict_evidence_indices
from necessity_mask_test import decodable_video_info, sample_indices

LABELS = list('ABCDE')


def boot_mean(values, rng):
    n = len(values)
    return sum(values[rng.randrange(n)] for _ in range(n)) / n


def bootstrap_ci(values, draws, seed):
    rng = random.Random(seed)
    estimates = sorted(boot_mean(values, rng) for _ in range(draws))
    return [estimates[int((draws - 1) * p)] for p in (.025, .975)]


def signflip_p(values, draws, seed):
    rng = random.Random(seed)
    observed = abs(sum(values) / len(values))
    extreme = sum(abs(sum(v * rng.choice((-1, 1)) for v in values) / len(values))
                  >= observed for _ in range(draws))
    return (1 + extreme) / (draws + 1)


def describe(values, draws, seed):
    if not values:
        return {'n': 0, 'mean': None, 'ci95_pointwise': None,
                'positive_fraction': None, 'signflip_p_two_sided': None}
    if len(values) == 1:
        return {'n': 1, 'mean': values[0], 'sd': None, 'ci95_pointwise': None,
                'positive_fraction': float(values[0] > 0), 'signflip_p_two_sided': None}
    sd = torch.tensor(values, dtype=torch.float64).std(unbiased=True).item()
    return {'n': len(values), 'mean': sum(values) / len(values), 'sd': sd,
            'ci95_pointwise': bootstrap_ci(values, draws, seed),
            'positive_fraction': sum(v > 0 for v in values) / len(values),
            'signflip_p_two_sided': signflip_p(values, draws, seed)}


def interaction_bootstrap(correct, incorrect, draws, seed):
    if len(correct) < 2 or len(incorrect) < 2:
        return {'mean_correct_minus_incorrect': None, 'ci95_pointwise': None}
    rng = random.Random(seed)
    estimates = sorted(boot_mean(correct, rng) - boot_mean(incorrect, rng)
                       for _ in range(draws))
    return {'mean_correct_minus_incorrect': sum(correct) / len(correct)
            - sum(incorrect) / len(incorrect),
            'ci95_pointwise': [estimates[int((draws - 1) * p)] for p in (.025, .975)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--placebo-repeats', type=int, default=5)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_frames < 3 or args.placebo_repeats < 1 or args.bootstrap_draws < 100:
        parser.error('Require num-frames >=3, placebo-repeats >=1, bootstrap-draws >=100')
    samples = json.loads(Path(args.samples_file).read_text())
    if not samples:
        parser.error('Sample file is empty')
    for i, sample in enumerate(samples):
        if len(sample.get('choices') or []) != 5 or sample.get('answer_choice_index') not in range(5):
            parser.error(f'Sample {i} must contain five choices and answer_choice_index 0..4')

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    token_ids = []
    for label in LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f'Choice label {label} is not a single token: {ids}')
        token_ids.append(ids[0])
    if len(set(token_ids)) != 5:
        parser.error('A-E choice labels must map to distinct tokens')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant, device_map=device)
    model.eval()

    def score(paths, question, choices):
        option_text = '\n'.join(f'{label}. {choice}'
                                 for label, choice in zip(LABELS, choices))
        prompt = (f'{question}\nOptions:\n{option_text}\n'
                  'Choose the best answer. Respond with one letter only.\nAnswer:')
        inputs = build_inputs(processor, device, paths, prompt)
        with torch.inference_mode():
            pos = inputs['input_ids'].shape[1] - 1
            logits = model(**inputs).logits[0, pos].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            return logprobs[torch.tensor(token_ids, device=logprobs.device)].cpu().tolist()

    rows, failures = [], []
    for sample_index, sample in enumerate(samples):
        sid = f"{sample.get('video_id', sample_index)}:{sample.get('qid', sample_index)}"
        temp = Path(tempfile.mkdtemp(prefix='choice_confirmation_'))
        try:
            video_path = os.path.expanduser(sample['video'])
            total, fps = decodable_video_info(video_path)
            indices = sorted(set(sample_indices(total, args.num_frames)))
            if len(indices) < 3:
                raise RuntimeError(f'Need >=3 distinct sampled frames; got {indices}')
            evidence = strict_evidence_indices(indices, fps, sample['grounding'])
            pairs = matched_single_frame_pairs(
                indices, evidence, args.placebo_repeats, args.seed + sample_index * 1009)
            if pairs is None:
                raise RuntimeError(f'Cannot construct matched interventions; evidence={evidence}, '
                                   f'non_evidence={len(set(indices)-set(evidence))}')
            frame_map = read_sampled_frames(video_path, total, indices, args.frame_size)
            full_paths = write_frame_set([frame_map[i].copy() for i in indices], str(temp / 'full'))
            choices = sample['choices']
            gold = int(sample['answer_choice_index'])
            full_scores = score(full_paths, sample['question'], choices)

            def margin(values):
                return values[gold] - max(v for i, v in enumerate(values) if i != gold)

            full_margin = margin(full_scores)
            baseline_correct = max(range(5), key=lambda i: full_scores[i]) == gold
            evidence_margins, placebo_margins = [], []
            for repeat, pair in enumerate(pairs):
                for arm, target_key, donor_key in (
                        ('evidence', 'evidence_target', 'evidence_donor'),
                        ('placebo', 'placebo_target', 'placebo_donor')):
                    images = [frame_map[i].copy() for i in indices]
                    target, donor = pair[target_key], pair[donor_key]
                    images[indices.index(target)] = frame_map[donor].copy()
                    paths = write_frame_set(images, str(temp / f'{arm}_{repeat}'))
                    value = margin(score(paths, sample['question'], choices))
                    (evidence_margins if arm == 'evidence' else placebo_margins).append(value)
            evidence_drop = full_margin - sum(evidence_margins) / len(evidence_margins)
            placebo_drop = full_margin - sum(placebo_margins) / len(placebo_margins)
            selective = evidence_drop - placebo_drop
            rows.append({'sample_id': sid, 'video_id': sample.get('video_id'),
                         'qid': sample.get('qid'), 'question_type': sample.get('question_type'),
                         'sampled_indices': indices, 'evidence_indices': evidence,
                         'baseline_choice_correct': baseline_correct,
                         'gold_choice_index': gold, 'predicted_choice_index': max(
                             range(5), key=lambda i: full_scores[i]),
                         'full_choice_logprobs': full_scores,
                         'full_correct_vs_best_foil_margin': full_margin,
                         'evidence_replaced_margins': evidence_margins,
                         'placebo_replaced_margins': placebo_margins,
                         'evidence_margin_drop': evidence_drop,
                         'mean_placebo_margin_drop': placebo_drop,
                         'selective_margin_drop': selective,
                         'matched_interventions': pairs})
            print(f'sample={sid} baseline_correct={baseline_correct} full_margin={full_margin:+.4f} '
                  f'evidence_drop={evidence_drop:+.4f} placebo_drop={placebo_drop:+.4f} '
                  f'selective={selective:+.4f}', flush=True)
        except Exception as exc:
            failures.append({'sample_id': sid, 'video': sample.get('video'), 'error': repr(exc)})
            print(f'WARNING: skipping sample={sid}: {exc}', flush=True)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
    correct = [r['selective_margin_drop'] for r in rows if r['baseline_choice_correct']]
    incorrect = [r['selective_margin_drop'] for r in rows if not r['baseline_choice_correct']]
    all_values = [r['selective_margin_drop'] for r in rows]
    if not rows:
        parser.error('No successful samples')
    result = {'schema_version': 1, 'script': 'evidence_control_choice_confirmation.py',
              'samples_file': args.samples_file, 'model': args.model, 'quant': args.quant,
              'dtype': 'bfloat16', 'seed': args.seed, 'num_frames': args.num_frames,
              'placebo_repeats': args.placebo_repeats,
              'prompt_protocol': 'five NExT-GQA options; gold A-E logprob minus best foil',
              'primary_cohort': 'unperturbed prompt predicts the dataset gold choice',
              'primary_estimand': 'mean selective margin drop among baseline-correct samples',
              'n_requested': len(samples), 'n_effective': len(rows), 'failures': failures,
              'summary': {'all': describe(all_values, args.bootstrap_draws, args.seed),
                          'baseline_correct_primary': describe(
                              correct, args.bootstrap_draws, args.seed + 1),
                          'baseline_incorrect_secondary': describe(
                              incorrect, args.bootstrap_draws, args.seed + 2),
                          'correct_minus_incorrect_interaction': interaction_bootstrap(
                              correct, incorrect, args.bootstrap_draws, args.seed + 3)},
              'multiplicity': 'One predeclared primary cohort; other cohorts and interaction are secondary; '
                              'no multiplicity correction.',
              'limitations': ['Fresh video IDs are required for independent confirmation.',
                              'Baseline-correct subgroup hypothesis was motivated by a prior exploratory '
                              'post-hoc split; this run must be interpreted as confirmation only if '
                              'sample IDs were not previously used.',
                              'Choice-letter scores depend on this prompt and are not official free-generation scores.',
                              'Single-frame replacements create hybrid/OOD inputs; this measures input-level '
                              'sensitivity, not head causality or general explanation validity.'],
              'records': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'n_requested={len(samples)} n_effective={len(rows)} failures={len(failures)} '
          f'n_baseline_correct={len(correct)} n_baseline_incorrect={len(incorrect)}')
    primary = result['summary']['baseline_correct_primary']
    print(f"primary_n={primary['n']} primary_mean={primary['mean']} "
          f"ci95={primary['ci95_pointwise']} "
          f"signflip_p={primary['signflip_p_two_sided']}")
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
