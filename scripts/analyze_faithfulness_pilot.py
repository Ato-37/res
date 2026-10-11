"""Integrity-check and summarize a published head-faithfulness pilot JSON."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize_result(result):
    records = result.get('records')
    requested = result.get('n_requested')
    effective = result.get('n_effective')
    failures = result.get('n_failures')
    if not isinstance(records, list) or not isinstance(requested, int):
        raise ValueError('missing records or integer n_requested')
    if not isinstance(effective, int) or not isinstance(failures, int):
        raise ValueError('missing integer effective/failure counts')
    if len(records) != effective or effective + failures != requested:
        raise ValueError('sample accounting does not reconcile')
    keys = [row.get('sample_key') for row in records]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError('each record must have an anonymous sample_key')
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate sample_key')

    selective, target, control = [], [], []
    correct = 0
    for row in records:
        values = [row.get('selective_margin_drop'), row.get('target_margin_drop'),
                  row.get('mean_control_margin_drop')]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values):
            raise ValueError(f'non-finite or missing score in {row["sample_key"]}')
        delta, target_drop, control_drop = map(float, values)
        if not math.isclose(delta, target_drop - control_drop, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f'selective contrast mismatch in {row["sample_key"]}')
        selective.append(delta)
        target.append(target_drop)
        control.append(control_drop)
        correct += int(bool(row.get('baseline_choice_correct')))

    published = result.get('primary_summary', {})
    if selective:
        avg = statistics.mean(selective)
        sd = statistics.stdev(selective) if len(selective) > 1 else None
        if not math.isclose(float(published.get('mean')), avg, rel_tol=0, abs_tol=1e-5):
            raise ValueError('published primary mean does not match records')
        if sd is not None and not math.isclose(float(published.get('sd')), sd,
                                               rel_tol=0, abs_tol=1e-5):
            raise ValueError('published primary SD does not match records')
        loo = [(sum(selective) - value) / (len(selective) - 1)
               for value in selective] if len(selective) > 1 else []
        largest = max(range(len(selective)), key=lambda i: abs(selective[i]))
        result_summary = {
            'n_requested': requested,
            'n_effective': effective,
            'n_failures': failures,
            'n_unique_sample_keys': len(set(keys)),
            'baseline_correct_n': correct,
            'baseline_accuracy_descriptive': correct / effective,
            'target_margin_drop_mean': statistics.mean(target),
            'control_margin_drop_mean': statistics.mean(control),
            'selective_mean': avg,
            'selective_sd': sd,
            'selective_median': statistics.median(selective),
            'selective_positive_n': sum(value > 0 for value in selective),
            'selective_negative_n': sum(value < 0 for value in selective),
            'selective_min': min(selective),
            'selective_max': max(selective),
            'largest_abs_contrast_sample_key': keys[largest],
            'leave_one_out_mean_min': min(loo) if loo else None,
            'leave_one_out_mean_max': max(loo) if loo else None,
            'published_ci95_t': published.get('ci95_t'),
            'published_p_two_sided_vs_0.25': published.get('p_two_sided_vs_threshold'),
            'primary_threshold_supported': published.get('primary_threshold_supported'),
            'runtime_seconds': result.get('runtime_seconds'),
        }
    else:
        result_summary = {'n_requested': requested, 'n_effective': 0,
                          'n_failures': failures, 'status': 'no_completed_records'}
    return result_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    path = Path(args.input)
    result = json.loads(path.read_text())
    summary = summarize_result(result)
    print(f'input={path}')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
