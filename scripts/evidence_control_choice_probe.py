"""Re-score stored evidence/placebo frame swaps with NExT-GQA choice margins.

Uses the exact sampled frames and one-frame replacement pairs from a completed
evidence_control_probe run. The primary outcome is the correct-choice letter
log-probability minus the best foil letter log-probability.
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import read_sampled_frames, write_frame_set, build_inputs
from necessity_mask_test import decodable_video_info
from closed_research_branch import reject_inference

LABELS = list('ABCDE')


def percentile_ci(values, draws=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(draws))
    return [boot[int((draws - 1) * q)] for q in (.025, .975)]


def summarize(values, draws, permutations, seed):
    n = len(values)
    avg = sum(values) / n
    rng = random.Random(seed)
    observed = abs(avg)
    extreme = sum(abs(sum(v * rng.choice((-1, 1)) for v in values) / n) >= observed
                  for _ in range(permutations))
    return {'n': n, 'mean_selective_margin_drop': avg,
            'sd': torch.tensor(values, dtype=torch.float64).std(unbiased=True).item(),
            'bootstrap_ci95_pointwise': percentile_ci(values, draws, seed),
            'positive_fraction': sum(v > 0 for v in values) / n,
            'paired_sign_flip_p_two_sided': (1 + extreme) / (permutations + 1),
            'n_permutations': permutations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--probe-results', required=True)
    parser.add_argument('--choice-audit', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reject_inference(parser, "legacy_evidence_control",
                     "this exploratory single-frame protocol is not the current preregistered estimand")
    if args.bootstrap_draws < 100 or args.permutations < 100:
        parser.error('Use at least 100 bootstrap draws and permutations')

    samples = json.loads(Path(args.samples_file).read_text())
    probe = json.loads(Path(args.probe_results).read_text())
    audit = json.loads(Path(args.choice_audit).read_text())
    samples_by_id = {str(i): sample for i, sample in enumerate(samples)}
    for sample in samples:
        if sample.get('video_id') is not None and sample.get('qid') is not None:
            samples_by_id[f"{sample['video_id']}:{sample['qid']}"] = sample
    choices_by_id = {str(row['sample_id']): row for row in audit.get('records', [])}
    probe_records = probe.get('records', [])
    if args.n_samples is not None:
        probe_records = probe_records[:args.n_samples]
    work = []
    for record in probe_records:
        sid = str(record['sample_id'])
        if sid not in samples_by_id or sid not in choices_by_id:
            parser.error(f'Missing sample or choice-audit record for sample_id={sid}')
        choice = choices_by_id[sid]
        opts, gold = choice['choices'], int(choice['answer_choice_index'])
        if len(opts) != 5 or gold not in range(5):
            parser.error(f'Invalid five-choice record for sample_id={sid}')
        work.append((samples_by_id[sid], record, choice, opts, gold))
    if not work:
        parser.error('No overlapping probe and choice-audit records')

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    choice_token_ids = []
    for label in LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f'{label} is not a single tokenizer token: {ids}')
        choice_token_ids.append(ids[0])
    if len(set(choice_token_ids)) != 5:
        parser.error('A-E do not map to distinct tokenizer tokens')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant, device_map=device)
    model.eval()
    rows, failures = [], []

    def score(paths, question, options):
        option_text = '\n'.join(f'{letter}. {text}'
                                 for letter, text in zip(LABELS, options))
        prompt = (f'{question}\nOptions:\n{option_text}\n'
                  'Choose the best answer. Respond with one letter only.\nAnswer:')
        inputs = build_inputs(processor, device, paths, prompt)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, inputs['input_ids'].shape[1] - 1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            return logprobs[torch.tensor(choice_token_ids, device=logprobs.device)].cpu().tolist()

    for sample, rec, choice, options, gold in work:
        sid = str(rec['sample_id'])
        temp = Path(tempfile.mkdtemp(prefix='choice_intervention_'))
        try:
            indices = rec['sampled_indices']
            video_path = os.path.expanduser(sample['video'])
            total, _fps = decodable_video_info(video_path)
            if max(indices) >= total:
                raise RuntimeError(f'Stored frame index {max(indices)} exceeds video length {total}')
            frame_map = read_sampled_frames(video_path, total, indices, args.frame_size)
            full = [frame_map[i].copy() for i in indices]
            full_paths = write_frame_set(full, str(temp / 'full'))
            full_scores = score(full_paths, sample['question'], options)

            def margin(scores):
                return scores[gold] - max(v for i, v in enumerate(scores) if i != gold)

            full_margin = margin(full_scores)
            evidence_scores, placebo_scores = [], []
            for j, pair in enumerate(rec['matched_interventions']):
                for arm, target_key in (('evidence', 'evidence_target'),
                                        ('placebo', 'placebo_target')):
                    images = [frame_map[i].copy() for i in indices]
                    target, donor = pair[target_key], pair[
                        'evidence_donor' if arm == 'evidence' else 'placebo_donor']
                    images[indices.index(target)] = frame_map[donor].copy()
                    paths = write_frame_set(images, str(temp / f'{arm}_{j}'))
                    scores = score(paths, sample['question'], options)
                    (evidence_scores if arm == 'evidence' else placebo_scores).append(scores)
            evidence_margins = [margin(v) for v in evidence_scores]
            placebo_margins = [margin(v) for v in placebo_scores]
            evidence_drop = full_margin - sum(evidence_margins) / len(evidence_margins)
            placebo_drop = full_margin - sum(placebo_margins) / len(placebo_margins)
            selective = evidence_drop - placebo_drop
            rows.append({
                'sample_id': sid, 'question_type': choice.get('question_type'),
                'baseline_choice_correct': bool(choice['correct']),
                'gold_choice_index': gold, 'full_choice_logprobs': full_scores,
                'full_correct_vs_best_foil_margin': full_margin,
                'evidence_replaced_margins': evidence_margins,
                'placebo_replaced_margins': placebo_margins,
                'evidence_margin_drop': evidence_drop,
                'mean_placebo_margin_drop': placebo_drop,
                'selective_margin_drop': selective,
                'matched_interventions': rec['matched_interventions']})
            print(f'sample={sid} baseline_correct={choice["correct"]} '
                  f'full_margin={full_margin:+.4f} evidence_drop={evidence_drop:+.4f} '
                  f'placebo_drop={placebo_drop:+.4f} selective={selective:+.4f}', flush=True)
        except Exception as exc:
            failures.append({'sample_id': sid, 'error': repr(exc)})
            print(f'WARNING: sample={sid} failed: {exc}', flush=True)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
    if len(rows) < 2:
        parser.error('Fewer than two successful records; cannot summarize')
    vals = [r['selective_margin_drop'] for r in rows]
    result = {'schema_version': 1, 'script': 'evidence_control_choice_probe.py',
              'samples_file': args.samples_file, 'probe_results': args.probe_results,
              'choice_audit': args.choice_audit, 'model': args.model,
              'quant': args.quant, 'dtype': 'bfloat16',
              'prompt_protocol': 'same five NExT-GQA choices; correct A-E logprob minus best foil',
              'estimand': 'mean[(full margin-evidence replaced margin) - '
                          '(full margin-placebo replaced margin)]',
              'interpretation': 'Positive means evidence-position frame replacement reduced '
                               'the correct-vs-best-foil margin more than matched placebo replacement.',
              'n_requested': len(work), 'n_effective': len(rows), 'failures': failures,
              'summary': summarize(vals, args.bootstrap_draws, args.permutations, args.seed),
              'by_baseline_correctness': {}, 'records': rows,
              'limitations': ['Exploratory outcome added after inspecting the free-answer probe.',
                              'Single-frame input replacement is a hybrid/OOD perturbation, not '
                              'a direct causal test of explanation validity.',
                              'Choice-letter margin depends on this prompt and does not equal '
                              'official free-generation accuracy.']}
    for correct in (True, False):
        subset = [r['selective_margin_drop'] for r in rows
                  if r['baseline_choice_correct'] is correct]
        result['by_baseline_correctness'][str(correct).lower()] = (
            summarize(subset, args.bootstrap_draws, args.permutations, args.seed)
            if len(subset) >= 2 else {'n': len(subset), 'summary': None})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"n_requested={len(work)} n_effective={len(rows)} failures={len(failures)}")
    print(f"mean_selective_margin_drop={result['summary']['mean_selective_margin_drop']:+.4f} "
          f"ci95={result['summary']['bootstrap_ci95_pointwise']} "
          f"signflip_p={result['summary']['paired_sign_flip_p_two_sided']:.5f}")
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
