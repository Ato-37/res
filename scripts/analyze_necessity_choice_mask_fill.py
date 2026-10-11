"""Compare black and blurred evidence-mask results on exactly paired samples."""
import argparse
import json
from pathlib import Path

from necessity_choice_margin_utils import summarize


def compare_results(black, blur, draws=10000, seed=0):
    for key in ('model', 'quant', 'dtype', 'num_frames', 'primary_estimand', 'control_protocol'):
        if black.get(key) != blur.get(key):
            raise ValueError(f'Protocol mismatch for {key}: {black.get(key)!r} != {blur.get(key)!r}')
    black_fill = black.get('mask_intervention', {}).get('fill', 'black')
    if black_fill != 'black':
        raise ValueError('First input must use black fill')
    if blur.get('mask_intervention', {}).get('fill') != 'blur':
        raise ValueError('Second input must use blur fill')
    black_rows = {r['sample_id']: r for r in black.get('records', [])}
    blur_rows = {r['sample_id']: r for r in blur.get('records', [])}
    if not black_rows or set(black_rows) != set(blur_rows):
        raise ValueError('Black and blur results must have identical non-empty sample IDs')
    for sample_id, b in black_rows.items():
        u = blur_rows[sample_id]
        for key in ('baseline_choice_correct', 'gold_choice_index', 'sampled_indices',
                    'evidence_mask_indices', 'control_mask_indices'):
            if b.get(key) != u.get(key):
                raise ValueError(f'Paired condition mismatch for {sample_id}: {key}')

    cohorts = {
        'all': lambda row: True,
        'baseline_correct_primary': lambda row: row['baseline_choice_correct'],
        'baseline_incorrect_secondary': lambda row: not row['baseline_choice_correct'],
    }
    comparisons = {}
    for offset, (name, predicate) in enumerate(cohorts.items()):
        ids = [sid for sid in black_rows if predicate(black_rows[sid])]
        black_values = [black_rows[sid]['selective_necessity_margin_drop'] for sid in ids]
        blur_values = [blur_rows[sid]['selective_necessity_margin_drop'] for sid in ids]
        paired_delta = [u - b for b, u in zip(black_values, blur_values)]
        comparisons[name] = {
            'n': len(ids),
            'black': summarize(black_values, draws, seed + 3 * offset),
            'blur': summarize(blur_values, draws, seed + 3 * offset + 1),
            'blur_minus_black': summarize(paired_delta, draws, seed + 3 * offset + 2),
        }
    return {
        'script': 'analyze_necessity_choice_mask_fill.py',
        'black_input': black.get('output_path'),
        'blur_input': blur.get('output_path'),
        'n_paired': len(black_rows),
        'failures_black': black.get('failures', []),
        'failures_blur': blur.get('failures', []),
        'comparisons': comparisons,
        'inference_note': 'Pointwise bootstrap intervals and sample-level sign-flip tests; '
                          'the paired fill contrast is a robustness analysis, not an independent replication.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--black', required=True)
    parser.add_argument('--blur', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error('--bootstrap-draws must be >=100')
    output = Path(args.output)
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    black, blur = json.loads(Path(args.black).read_text()), json.loads(Path(args.blur).read_text())
    try:
        result = compare_results(black, blur, args.bootstrap_draws, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    result['black_input'] = args.black
    result['blur_input'] = args.blur
    result['bootstrap_draws'] = args.bootstrap_draws
    result['seed'] = args.seed
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for cohort, values in result['comparisons'].items():
        delta = values['blur_minus_black']
        print(f"{cohort}: n={values['n']} black={values['black']['mean']:+.6f} "
              f"blur={values['blur']['mean']:+.6f} delta={delta['mean']:+.6f} "
              f"CI={delta['ci95_pointwise']} p={delta['signflip_p_two_sided']}")
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
