"""Test grounding-frame necessity using matched black-mask controls and MC margin.

For each sample, compare the unmasked video against (1) black-masked sampled
frames strictly inside the annotated grounding interval and (2) an equal-size
set of nearest available non-grounding frames. The primary contrast is the
gold-vs-best-foil margin drop under evidence masking minus the drop under the
matched control mask. Black masking is OOD; this is an exploratory necessity
test, not proof of explanation validity.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from activation_patching_sweep import build_inputs, read_sampled_frames, write_frame_set
from necessity_mask_test import decodable_video_info, sample_indices
from necessity_choice_margin_utils import (apply_mask_fill, interaction_ci,
                                           nearest_control_indices, strict_evidence_indices,
                                           summarize)

LABELS = list('ABCDE')


def provenance(path: Path) -> dict:
    """Record reproducibility hashes without publishing the input contents."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], check=True, capture_output=True,
            text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'samples_file_sha256': digest, 'source_commit': commit}


def masked_paths(frame_map, indices, masked, directory, mask_fill, blur_radius):
    masked = set(masked)
    images = []
    for i in indices:
        source = frame_map[i]
        images.append(apply_mask_fill(source, mask_fill, blur_radius)
                      if i in masked else source.copy())
    return write_frame_set(images, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--mask-fill', choices=['black', 'blur'], default='black')
    parser.add_argument('--blur-radius', type=float, default=12.0)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--cap-pixels-per-frame', action='store_true',
                        help='apply the qwen-vl-utils per-frame pixel cap explicitly')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_frames < 3 or args.bootstrap_draws < 100:
        parser.error('Require --num-frames >=3 and --bootstrap-draws >=100')
    if args.mask_fill == 'blur' and args.blur_radius <= 0:
        parser.error('--blur-radius must be positive when --mask-fill=blur')
    samples = json.loads(Path(args.samples_file).read_text())
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    if not samples:
        parser.error('No samples selected')
    for j, sample in enumerate(samples):
        if len(sample.get('choices') or []) != 5 or sample.get('answer_choice_index') not in range(5):
            parser.error(f'Sample {j} must have five choices and answer_choice_index in 0..4')

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels,
        cap_pixels_per_frame=args.cap_pixels_per_frame)
    label_ids = []
    for label in LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f'Choice label {label} is not one token: {ids}')
        label_ids.append(ids[0])
    if len(set(label_ids)) != 5:
        parser.error('A-E must map to distinct tokens')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quant = None if args.quant == 'none' else BitsAndBytesConfig(
        **({'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
           if args.quant == '4bit' else {'load_in_8bit': True}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
        quantization_config=quant, device_map=device)
    model.eval()

    def score(paths, question, choices):
        options = '\n'.join(f'{label}. {choice}' for label, choice in zip(LABELS, choices))
        prompt = (f'{question}\nOptions:\n{options}\n'
                  'Choose the best answer. Respond with one letter only.\nAnswer:')
        inputs = build_inputs(processor, device, paths, prompt)
        input_meta = {
            'input_ids_shape': list(inputs['input_ids'].shape),
            'video_grid_thw': (inputs.get('video_grid_thw').detach().cpu().tolist()
                               if inputs.get('video_grid_thw') is not None else None),
        }
        with torch.inference_mode():
            position = inputs['input_ids'].shape[1] - 1
            logits = model(**inputs).logits[0, position].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            return logprobs[torch.tensor(label_ids, device=logprobs.device)].cpu().tolist(), input_meta

    rows, failures = [], []
    for sample_idx, sample in enumerate(samples):
        sid = f"{sample.get('video_id', sample_idx)}:{sample.get('qid', sample_idx)}"
        temp = Path(tempfile.mkdtemp(prefix='necessity_choice_margin_'))
        try:
            video = os.path.expanduser(sample['video'])
            total, fps = decodable_video_info(video)
            indices = sorted(set(sample_indices(total, args.num_frames)))
            if len(indices) < 3:
                raise RuntimeError(f'Need at least 3 distinct sampled frames; got {indices}')
            evidence = strict_evidence_indices(indices, fps, sample['grounding'])
            control = nearest_control_indices(indices, evidence)
            if not evidence:
                raise RuntimeError(f'No sampled frame strictly overlaps grounding; sampled={indices}')
            if not control:
                raise RuntimeError(f'Cannot match {len(evidence)} evidence frames to non-evidence controls')
            if len(control) != len(evidence) or set(control) & set(evidence):
                raise RuntimeError('Mask self-check failed: control/evidence dose or overlap mismatch')
            frame_map = read_sampled_frames(video, total, indices, args.frame_size)
            paths = {
                'full': masked_paths(frame_map, indices, [], str(temp / 'full'),
                                     args.mask_fill, args.blur_radius),
                'evidence_mask': masked_paths(frame_map, indices, evidence, str(temp / 'evidence'),
                                              args.mask_fill, args.blur_radius),
                'control_mask': masked_paths(frame_map, indices, control, str(temp / 'control'),
                                             args.mask_fill, args.blur_radius),
            }
            choices, gold = sample['choices'], int(sample['answer_choice_index'])
            scored = {condition: score(frame_paths, sample['question'], choices)
                      for condition, frame_paths in paths.items()}
            scores = {condition: value[0] for condition, value in scored.items()}
            input_metadata = {condition: value[1] for condition, value in scored.items()}
            margin = lambda condition: scores[condition][gold] - max(
                value for option, value in enumerate(scores[condition]) if option != gold)
            full_margin, evidence_margin, control_margin = (
                margin('full'), margin('evidence_mask'), margin('control_mask'))
            prediction = max(range(5), key=lambda i: scores['full'][i])
            evidence_drop = full_margin - evidence_margin
            control_drop = full_margin - control_margin
            selective = evidence_drop - control_drop
            rows.append({'sample_id': sid, 'video_id': sample.get('video_id'),
                         'qid': sample.get('qid'), 'question_type': sample.get('question_type'),
                         'sampled_indices': indices, 'evidence_mask_indices': evidence,
                         'control_mask_indices': control, 'masked_frame_count': len(evidence),
                         'baseline_choice_correct': prediction == gold,
                         'gold_choice_index': gold, 'predicted_choice_index': prediction,
                         'choice_logprobs': scores, 'margins': {
                             key: scores[key][gold] - max(v for i, v in enumerate(scores[key]) if i != gold)
                             for key in scores}, 'input_metadata': input_metadata,
                         'evidence_margin_drop': evidence_drop,
                         'control_margin_drop': control_drop,
                         'selective_necessity_margin_drop': selective})
            print(f'sample={sid} baseline_correct={prediction == gold} '
                  f'masked={len(evidence)} full={full_margin:+.4f} '
                  f'evidence={evidence_margin:+.4f} control={control_margin:+.4f} '
                  f'selective={selective:+.4f}', flush=True)
        except Exception as exc:
            failures.append({'sample_id': sid, 'video': sample.get('video'), 'error': repr(exc)})
            print(f'WARNING: skipping sample={sid}: {exc}', flush=True)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    all_effects = [r['selective_necessity_margin_drop'] for r in rows]
    correct = [r['selective_necessity_margin_drop'] for r in rows if r['baseline_choice_correct']]
    incorrect = [r['selective_necessity_margin_drop'] for r in rows if not r['baseline_choice_correct']]
    result = {'schema_version': 2, 'script': 'necessity_choice_margin_test.py',
              'samples_file': args.samples_file, 'model': args.model, 'quant': args.quant,
              **provenance(Path(args.samples_file)),
              'dtype': 'bfloat16', 'attn_implementation': 'eager',
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'device': device, 'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'seed': args.seed, 'num_frames': args.num_frames,
              'frame_size': args.frame_size, 'max_pixels': args.max_pixels,
              'cap_pixels_per_frame': args.cap_pixels_per_frame,
              'mask_intervention': {'fill': args.mask_fill,
                                    'blur_radius': args.blur_radius if args.mask_fill == 'blur' else None},
              'prompt_protocol': 'five NExT-GQA options; gold A-E logprob minus best foil',
              'primary_estimand': 'mean[(full margin - evidence-mask margin) - '
                                  '(full margin - matched-control-mask margin)] among baseline-correct samples',
              'control_protocol': 'same number of black-masked frames; non-grounding indices chosen '
                                 'to minimize temporal rank-distance from evidence indices',
              'n_requested': len(samples), 'n_effective': len(rows), 'failures': failures,
              'summary': {'all': summarize(all_effects, args.bootstrap_draws, args.seed),
                          'baseline_correct_primary': summarize(correct, args.bootstrap_draws, args.seed + 1),
                          'baseline_incorrect_secondary': summarize(incorrect, args.bootstrap_draws, args.seed + 2),
                          'correct_minus_incorrect_interaction': interaction_ci(
                              correct, incorrect, args.bootstrap_draws, args.seed + 3)},
              'multiplicity': 'One predeclared primary cohort (baseline-correct); all-sample, '
                              'baseline-incorrect, and interaction results are secondary and '
                              'not multiplicity-corrected.',
              'signflip_test': 'Two-sided sample-level random sign flips of the mean contrast; '
                               'assumes symmetric paired null effects.',
              'limitations': ['Black-frame masking is an out-of-distribution perturbation; equal-dose '
                              'control reduces but does not remove this concern.',
                              'The gold-vs-best-foil letter margin is prompt-dependent and is not official '
                              'free-generation accuracy.',
                              'Conditioning on baseline correctness is part of the prespecified primary '
                              'cohort; this does not establish general XAI faithfulness or head causality.'],
              'records': rows}
    if not rows:
        parser.error('No successful samples; no output summary can be produced')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    primary = result['summary']['baseline_correct_primary']
    print(f"n_requested={len(samples)} n_effective={len(rows)} failures={len(failures)} "
          f"n_primary={primary['n']} primary_mean={primary['mean']} "
          f"ci95={primary['ci95_pointwise']} p={primary['signflip_p_two_sided']}")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
