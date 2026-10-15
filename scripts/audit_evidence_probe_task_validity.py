"""Audit base multiple-choice competence on existing evidence-probe samples.

This is a task-validity diagnostic, not a new faithfulness test. It scores the
five NExT-GQA option letters from one next-token distribution per original
video/question, then descriptively stratifies the already-computed free-answer
evidence/placebo contrasts by whether the model selected the correct option.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import build_inputs, read_sampled_frames, write_frame_set
from necessity_mask_test import decodable_video_info
from closed_research_branch import reject_inference


CHOICE_COLUMNS = [f'a{i}' for i in range(5)]
LABELS = list('ABCDE')


def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip().casefold()


def vidor_key(value):
    parts = [part for part in str(value).replace('\\', '/').split('/') if part]
    return '/'.join(parts[-2:]) if len(parts) >= 2 else str(value)


def grounding_key(intervals):
    return tuple(tuple(round(float(x), 6) for x in span) for span in intervals)


def load_source_rows(test_csv, video_map):
    with open(test_csv, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError('test.csv has no header')
        missing = set(['video_id', 'question', 'answer', 'qid', *CHOICE_COLUMNS]) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'test.csv missing required MCQA columns: {sorted(missing)}')
        rows = list(reader)
    mapping = json.loads(Path(video_map).read_text())
    rows_by_vidor = defaultdict(list)
    for row in rows:
        if all(row.get(column, '').strip() for column in CHOICE_COLUMNS):
            vidor = mapping.get(str(row['video_id']))
            if vidor is not None:
                rows_by_vidor[vidor_key(vidor)].append(row)
    return rows_by_vidor


def resolve_question(sample, record, record_index, rows_by_vidor, gsub):
    if sample.get('choices'):
        choices = [str(item).strip() for item in sample['choices']]
        answer_index = sample.get('answer_choice_index')
        if answer_index is None:
            matches = [i for i, item in enumerate(choices) if norm(item) == norm(sample['answer'])]
            if len(matches) != 1:
                raise ValueError(f'Correct answer does not uniquely match choices for sample {record["sample_id"]}')
            answer_index = matches[0]
        answer_index = int(answer_index)
        if len(choices) != 5 or answer_index not in range(5):
            raise ValueError(f'Expected five choices and a valid label for sample {record["sample_id"]}')
        return choices, answer_index, sample.get('question_type')

    parts = [part for part in os.path.expanduser(sample['video']).replace('\\', '/').split('/') if part]
    key = vidor_key(os.path.splitext('/'.join(parts[-2:]))[0])
    candidates = [row for row in rows_by_vidor.get(key, [])
                  if norm(row['question']) == norm(sample['question'])
                  and norm(row['answer']) == norm(sample['answer'])]
    candidates = [row for row in candidates
                  if grounding_key(gsub.get(str(row['video_id']), {})
                                   .get('location', {})
                                   .get(str(row['qid']), []))
                  == grounding_key(sample['grounding'])]
    if len(candidates) != 1:
        raise ValueError(f'Could not uniquely join sample_id={record["sample_id"]} '
                         f'to test.csv using video/question/answer/grounding; matches={len(candidates)}')
    row = candidates[0]
    choices = [row[column].strip() for column in CHOICE_COLUMNS]
    matches = [i for i, item in enumerate(choices) if norm(item) == norm(row['answer'])]
    if len(matches) != 1:
        raise ValueError(f'Answer must match exactly one a0-a4 choice for qid={row["qid"]}')
    return choices, matches[0], row.get('type')


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def mean_ci(values, draws=10000, seed=0):
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [samples[int((draws - 1) * q)] for q in (.025, .975)]


def summarize_predictions(records, seed=0):
    n = len(records)
    correct = sum(bool(row['correct']) for row in records)
    margins = [row['correct_vs_best_foil_logprob'] for row in records]
    types = defaultdict(list)
    for row in records:
        types[row['question_type'] or 'unknown'].append(row)
    by_correct = {}
    for key, rows in [('correct', [r for r in records if r['correct']]),
                      ('incorrect', [r for r in records if not r['correct']])]:
        old = [r['prior_selective_logprob_drop'] for r in rows
               if r['prior_selective_logprob_drop'] is not None]
        by_correct[key] = {'n': len(rows),
                           'prior_selective_logprob_drop_mean': sum(old) / len(old) if old else None,
                           'prior_selective_logprob_drop_ci95': mean_ci(old, seed=seed) if old else None}
    return {'n': n, 'multiple_choice_accuracy': correct / n if n else None,
            'accuracy_wilson_ci95': wilson_interval(correct, n),
            'correct_vs_best_foil_logprob_margin_mean': sum(margins) / n if n else None,
            'correct_vs_best_foil_logprob_margin_ci95': mean_ci(margins, seed=seed) if margins else None,
            'stratified_prior_free_answer_probe_contrast': by_correct,
            'by_question_type': {kind: {'n': len(rows),
                                        'accuracy': sum(r['correct'] for r in rows) / len(rows),
                                        'mean_correct_vs_best_foil_margin': sum(
                                            r['correct_vs_best_foil_logprob'] for r in rows) / len(rows)}
                                 for kind, rows in sorted(types.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--probe-results', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--map-json', required=True)
    parser.add_argument('--gsub-json', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reject_inference(parser, 'legacy_evidence_probe_task_validity',
                     'real-media QA scoring is outside the current gated protocol')

    samples = json.loads(Path(args.samples_file).read_text())
    probe = json.loads(Path(args.probe_results).read_text())
    gsub = json.loads(Path(args.gsub_json).read_text())
    rows_by_vidor = load_source_rows(args.test_csv, args.map_json)
    records = probe.get('records', [])
    if args.n_samples is not None:
        records = records[:args.n_samples]
    resolved = []
    for record in records:
        sid = str(record['sample_id'])
        if ':' in sid:
            video_id, qid = sid.rsplit(':', 1)
            match = next((s for s in samples if str(s.get('video_id')) == video_id
                          and str(s.get('qid')) == qid), None)
            index = samples.index(match) if match is not None else -1
        else:
            try:
                index = int(sid)
            except ValueError:
                index = -1
            match = samples[index] if 0 <= index < len(samples) else None
        if match is None:
            parser.error(f'Cannot match probe sample_id={sid} to --samples-file')
        choices, answer_index, question_type = resolve_question(
            match, record, index, rows_by_vidor, gsub)
        resolved.append((match, record, choices, answer_index, question_type))

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    token_ids = []
    for label in LABELS:
        # build_inputs adds the assistant generation prefix after the user
        # message. The first generated token is therefore the bare option
        # letter, not a space-prefixed continuation.
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f'Choice label {label} is not a single tokenizer token; got {ids}')
        token_ids.append(ids[0])
    if len(set(token_ids)) != 5:
        parser.error('A-E choice labels do not map to distinct tokens')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant, device_map=device)
    model.eval()
    result_records, failures = [], []
    for sample_index, (sample, probe_record, choices, answer_index, question_type) in enumerate(resolved):
        sid = str(probe_record['sample_id'])
        temp_dir = Path(tempfile.mkdtemp(prefix='task_validity_audit_'))
        try:
            indices = probe_record['sampled_indices']
            video_path = os.path.expanduser(sample['video'])
            total, _fps = decodable_video_info(video_path)
            if max(indices) >= total:
                raise RuntimeError(f'Stored frame index exceeds decoded video length: {max(indices)} >= {total}')
            frame_map = read_sampled_frames(video_path, total, indices, args.frame_size)
            images = [frame_map[index].copy() for index in indices]
            paths = write_frame_set(images, str(temp_dir / 'full'))
            options_text = '\n'.join(f'{label}. {choice}' for label, choice in zip(LABELS, choices))
            prompt = (f"{sample['question']}\nOptions:\n{options_text}\n"
                      'Choose the best answer. Respond with one letter only.\nAnswer:')
            inputs = build_inputs(processor, device, paths, prompt)
            with torch.inference_mode():
                outputs = model(**inputs)
                pos = inputs['input_ids'].shape[1] - 1
                all_logprobs = torch.log_softmax(outputs.logits[0, pos].float(), dim=-1)
                scores = all_logprobs[torch.tensor(token_ids, device=all_logprobs.device)].cpu().tolist()
            ranked = sorted(range(5), key=lambda i: scores[i], reverse=True)
            margin = scores[answer_index] - max(scores[i] for i in range(5) if i != answer_index)
            result_records.append({
                'sample_id': sid, 'video_id': sample.get('video_id'), 'qid': sample.get('qid'),
                'question_type': question_type, 'choices': choices,
                'answer_choice_index': answer_index, 'predicted_choice_index': ranked[0],
                'predicted_choice_label': LABELS[ranked[0]], 'correct': ranked[0] == answer_index,
                'choice_logprobs': scores, 'correct_vs_best_foil_logprob': margin,
                'top_vs_second_logprob': scores[ranked[0]] - scores[ranked[1]],
                'prior_selective_logprob_drop': probe_record.get('selective_evidence_drop'),
            })
            print(f"sample={sid} predicted={LABELS[ranked[0]]} "
                  f"gold={LABELS[answer_index]} correct={ranked[0] == answer_index} "
                  f"margin={margin:+.4f}", flush=True)
        except Exception as exc:
            failures.append({'sample_id': sid, 'error': repr(exc)})
            print(f'WARNING: sample={sid} failed: {exc}', flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    summary = summarize_predictions(result_records, args.seed)
    output = {'schema_version': 1, 'script': 'audit_evidence_probe_task_validity.py',
              'samples_file': args.samples_file, 'probe_results': args.probe_results,
              'model': args.model, 'quant': args.quant, 'dtype': 'bfloat16',
              'prompt_protocol': 'five NExT-GQA answer choices; next-token logprob of A-E; '
                                 'greedy-choice accuracy',
              'n_requested': len(resolved), 'n_effective': len(result_records),
              'failures': failures, 'summary': summary, 'records': result_records,
              'limitations': ['This is a post-hoc task-validity audit on the already-used probe set.',
                              'Choice-letter scoring is prompt-dependent and is not the dataset’s '
                              'free-form generation score.',
                              'Stratifying earlier free-answer contrasts by baseline accuracy is '
                              'descriptive and does not establish a causal interaction.']}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f"n_requested={len(resolved)} n_effective={len(result_records)} failures={len(failures)}")
    print(f"multiple_choice_accuracy={summary['multiple_choice_accuracy']} "
          f"wilson95={summary['accuracy_wilson_ci95']} "
          f"mean_correct_vs_foil_margin={summary['correct_vs_best_foil_logprob_margin_mean']}")
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
