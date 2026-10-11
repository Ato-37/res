"""Exploratorily decompose evidence-mask and matched-control margin drops.

This does not replace the pre-specified selective contrast. Component intervals
and sign-flip p-values are pointwise and unadjusted; they are descriptive probes
of which side contributes to that contrast, not confirmatory tests.
"""
import argparse
import json
import math
from pathlib import Path

from necessity_choice_margin_utils import summarize


PROTOCOL_KEYS = ('model', 'quant', 'dtype', 'num_frames', 'primary_estimand',
                 'control_protocol')
PAIR_FIELDS = ('baseline_choice_correct', 'gold_choice_index', 'sampled_indices',
               'evidence_mask_indices', 'control_mask_indices')
COHORTS = (
    ('all', lambda row: True),
    ('baseline_correct_primary', lambda row: row['baseline_choice_correct']),
    ('baseline_incorrect_secondary', lambda row: not row['baseline_choice_correct']),
)


def decompose_results(black, blur, draws=10000, seed=0):
    for key in PROTOCOL_KEYS:
        if black.get(key) != blur.get(key):
            raise ValueError(f'Protocol mismatch for {key}')
    if black.get('mask_intervention', {}).get('fill', 'black') != 'black':
        raise ValueError('First input must use black fill')
    if blur.get('mask_intervention', {}).get('fill') != 'blur':
        raise ValueError('Second input must use blur fill')

    def index_records(result, label):
        records = result.get('records', [])
        indexed = {row.get('sample_id'): row for row in records}
        if None in indexed or len(indexed) != len(records):
            raise ValueError(f'{label} records need unique sample_id values')
        if result.get('n_effective', len(records)) != len(records):
            raise ValueError(f'{label} n_effective does not match record count')
        return indexed

    black_rows = index_records(black, 'black')
    blur_rows = index_records(blur, 'blur')
    if not black_rows or black_rows.keys() != blur_rows.keys():
        raise ValueError('Inputs must have identical non-empty sample IDs')

    for sample_id, b_row in black_rows.items():
        u_row = blur_rows[sample_id]
        for field in PAIR_FIELDS:
            if b_row.get(field) != u_row.get(field):
                raise ValueError(f'Paired condition mismatch for {sample_id}: {field}')
        for label, row in (('black', b_row), ('blur', u_row)):
            evidence = row.get('evidence_margin_drop')
            control = row.get('control_margin_drop')
            selective = row.get('selective_necessity_margin_drop')
            if not all(isinstance(x, (int, float)) and math.isfinite(x)
                       for x in (evidence, control, selective)):
                raise ValueError(f'{label} record {sample_id} has missing/non-finite components')
            if not math.isclose(evidence - control, selective, rel_tol=0, abs_tol=1e-5):
                raise ValueError(f'{label} components do not reproduce selective contrast for {sample_id}')

    output = {
        'n_paired': len(black_rows),
        'failures_black': black.get('failures', []),
        'failures_blur': blur.get('failures', []),
        'inference_note': (
            'Exploratory component decomposition; pointwise percentile-bootstrap CIs and '
            'sample-level sign-flip tests, no multiplicity correction. Not confirmatory.'
        ),
        'cohorts': {},
    }
    fields = ('evidence_margin_drop', 'control_margin_drop')
    for cohort_index, (cohort, predicate) in enumerate(COHORTS):
        ids = [sid for sid, row in black_rows.items() if predicate(row)]
        condition_results = {}
        for condition_index, (condition, rows) in enumerate((('black', black_rows),
                                                              ('blur', blur_rows))):
            components = {}
            for component_index, field in enumerate(fields):
                values = [rows[sid][field] for sid in ids]
                component_seed = seed + 10 * cohort_index + 4 * condition_index + component_index
                components[field] = summarize(values, draws, component_seed)
            condition_results[condition] = components
        output['cohorts'][cohort] = {'n': len(ids), **condition_results}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--black', required=True)
    parser.add_argument('--blur', required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error('--bootstrap-draws must be >=100')
    try:
        black = json.loads(Path(args.black).read_text())
        blur = json.loads(Path(args.blur).read_text())
        result = decompose_results(black, blur, args.bootstrap_draws, args.seed)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print('Exploratory decomposition; pointwise 95% percentile-bootstrap CI; '
          'sample sign-flip p; 12 component tests, no multiplicity correction.')
    print('fill cohort n component mean sd ci95_low ci95_high signflip_p positive_fraction')
    for cohort, cohort_result in result['cohorts'].items():
        for fill in ('black', 'blur'):
            for field, summary in cohort_result[fill].items():
                ci = summary['ci95_pointwise'] or [None, None]
                print(f"{fill:5s} {cohort:29s} {summary['n']:2d} {field:21s} "
                      f"{summary['mean']:+.6f} {summary['sd'] if summary['sd'] is not None else float('nan'):.6f} "
                      f"{ci[0] if ci[0] is not None else float('nan'):+.6f} "
                      f"{ci[1] if ci[1] is not None else float('nan'):+.6f} "
                      f"{summary['signflip_p_two_sided'] if summary['signflip_p_two_sided'] is not None else float('nan'):.6f} "
                      f"{summary['positive_fraction']:.3f}")
    print(f"n_paired={result['n_paired']} failures_black={len(result['failures_black'])} "
          f"failures_blur={len(result['failures_blur'])}")


if __name__ == '__main__':
    main()
