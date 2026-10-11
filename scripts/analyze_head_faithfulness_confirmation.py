"""Recompute and integrity-check a published head-faithfulness confirmation."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from scipy.stats import ttest_1samp

from faithfulness_utils import (
    bootstrap_mean_interval,
    mean_t_interval,
    signflip_p_against,
)


EXPERIMENT = 'gradient_activation_head_faithfulness'
FORBIDDEN_KEYS = {'video', 'video_id', 'question', 'choices', 'answer', 'qid', 'prompt'}


def _assert_private(value, path='result'):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS or 'path' in str(key).lower():
                raise ValueError(f'private field at {path}.{key}')
            _assert_private(child, f'{path}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_private(child, f'{path}[{index}]')


def _validate_controls(record, top_k, repeats):
    selected = record.get('selected_heads')
    groups = record.get('control_groups')
    if not isinstance(selected, list) or len(selected) != top_k:
        raise ValueError('selected-head count mismatch')
    selected_keys = [(head.get('layer'), head.get('head')) for head in selected]
    if len(set(selected_keys)) != top_k:
        raise ValueError('selected heads are not unique')
    target_histogram = Counter(layer for layer, _ in selected_keys)
    if not isinstance(groups, list) or len(groups) != repeats:
        raise ValueError('control-group count mismatch')
    seen_controls = set()
    for group in groups:
        keys = [(head.get('layer'), head.get('head')) for head in group]
        if len(keys) != top_k or len(set(keys)) != top_k:
            raise ValueError('control-head count or uniqueness mismatch')
        if Counter(layer for layer, _ in keys) != target_histogram:
            raise ValueError('control layer histogram does not match selected heads')
        if set(keys) & set(selected_keys):
            raise ValueError('control group overlaps selected heads')
        if seen_controls & set(keys):
            raise ValueError('control heads are reused across repetitions')
        seen_controls.update(keys)


def analyze(result, manifest=None):
    _assert_private(result)
    if result.get('experiment') != EXPERIMENT:
        raise ValueError('unexpected experiment identifier')
    if result.get('primary_estimand') != (
            'mean(target margin drop minus mean matched-control margin drop), '
            'one independent video per sample'):
        raise ValueError('unexpected primary estimand')
    requested, effective, failures = (result.get(key) for key in
                                      ('n_requested', 'n_effective', 'n_failures'))
    records = result.get('records')
    top_k = result.get('top_k')
    repeats = result.get('control_repeats')
    if (not all(isinstance(value, int) for value in
                (requested, effective, failures, top_k, repeats))
            or not isinstance(records, list)):
        raise ValueError('missing integer counts or records')
    if len(records) != effective or effective + failures != requested or effective < 2:
        raise ValueError('sample accounting does not reconcile')
    if top_k != 4 or repeats != 5 or result.get('alpha') != 0.5:
        raise ValueError('primary intervention differs from the locked protocol')
    if result.get('layers') != list(range(28)) or result.get('num_frames') != 16:
        raise ValueError('layer or frame configuration differs from protocol')
    threshold = result.get('practical_threshold_nat')
    if threshold != 0.25:
        raise ValueError('practical threshold differs from the locked protocol')

    keys = [record.get('sample_key') for record in records]
    if (any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError('sample keys must be nonempty and unique')

    selective, target_drops, control_drops = [], [], []
    baseline_correct = 0
    for record in records:
        sample_key = record['sample_key']
        baseline_choice = record.get('baseline_prediction_index')
        foil = record.get('baseline_foil_index')
        if (not isinstance(baseline_choice, int) or not 0 <= baseline_choice < 5
                or not isinstance(foil, int) or not 0 <= foil < 5 or foil == baseline_choice):
            raise ValueError(f'invalid frozen prediction/foil in {sample_key}')
        _validate_controls(record, top_k, repeats)
        target, control, contrast = (record.get(field) for field in
                                     ('target_margin_drop', 'mean_control_margin_drop',
                                      'selective_margin_drop'))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (target, control, contrast)):
            raise ValueError(f'non-finite/missing contrast in {sample_key}')
        if not math.isclose(contrast, target - control, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f'selective contrast mismatch in {sample_key}')
        raw_controls = record.get('control_margin_drops')
        if (not isinstance(raw_controls, list) or len(raw_controls) != repeats
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in raw_controls)):
            raise ValueError(f'invalid raw control drops in {sample_key}')
        if not math.isclose(control, statistics.mean(raw_controls), rel_tol=0, abs_tol=1e-5):
            raise ValueError(f'control mean mismatch in {sample_key}')
        selective.append(float(contrast))
        target_drops.append(float(target))
        control_drops.append(float(control))
        baseline_correct += int(bool(record.get('baseline_choice_correct')))

    summary = mean_t_interval(selective)
    published = result.get('primary_summary', {})
    expected = {
        'mean': summary['mean'],
        'sd': summary['sd'],
        'ci95_t': summary['ci'],
        'test_null_mean': threshold,
        't_statistic_vs_threshold': float(ttest_1samp(selective, threshold).statistic),
        'p_two_sided_vs_threshold': float(ttest_1samp(selective, threshold).pvalue),
        'bootstrap_ci95_pointwise': bootstrap_mean_interval(
            selective, 10000, result['seed']),
        'signflip_p_two_sided_vs_threshold': signflip_p_against(
            selective, null=threshold, draws=10000, seed=result['seed'] + 1),
        'positive_fraction': sum(value > 0 for value in selective) / effective,
        'fraction_above_practical_threshold': sum(value > threshold for value in selective) / effective,
        'primary_threshold_supported': summary['ci'][0] > threshold,
    }
    for key, value in expected.items():
        actual = published.get(key)
        if isinstance(value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                actual, value, rel_tol=0, abs_tol=1e-8)
        elif key == 'ci95_t' or key == 'bootstrap_ci95_pointwise':
            matches = (isinstance(actual, list) and len(actual) == len(value)
                       and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
                               for a, b in zip(actual, value)))
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f'published primary summary mismatch for {key}')

    if manifest is not None:
        for key in ('experiment', 'model', 'quant', 'seed', 'top_k',
                    'alpha', 'control_repeats', 'num_frames', 'cap_pixels_per_frame',
                    'n_requested', 'n_effective', 'n_failures'):
            if result.get(key) != manifest.get(key):
                raise ValueError(f'result/run-manifest mismatch for {key}')
        if manifest.get('layers') != 'all':
            raise ValueError('run manifest layer configuration differs from the locked protocol')
        if manifest.get('max_pixels') != 200704:
            raise ValueError('run manifest max_pixels differs from the locked protocol')
        sample_hash = manifest.get('samples_sha256')
        if (not isinstance(sample_hash, str) or len(sample_hash) != 64
                or any(char not in '0123456789abcdef' for char in sample_hash.lower())):
            raise ValueError('run manifest has no valid sample-manifest SHA-256')

    zero_test = ttest_1samp(selective, 0.0)
    return {
        'interpretation_scope': 'one preregistered test-set confirmation; no secondary subgroup inference',
        'n_requested': requested,
        'n_effective': effective,
        'n_failures': failures,
        'n_unique_videos': len(set(keys)),
        'baseline_correct_n': baseline_correct,
        'baseline_accuracy_descriptive': baseline_correct / effective,
        'mean_target_margin_drop_nat': statistics.mean(target_drops),
        'mean_control_margin_drop_nat': statistics.mean(control_drops),
        'mean_selective_margin_drop_nat': summary['mean'],
        'sd_selective_margin_drop_nat': summary['sd'],
        'ci95_t': summary['ci'],
        'median_selective_margin_drop_nat': statistics.median(selective),
        'positive_n': sum(value > 0 for value in selective),
        'negative_n': sum(value < 0 for value in selective),
        'zero_n': sum(value == 0 for value in selective),
        'p_two_sided_vs_zero_descriptive': float(zero_test.pvalue),
        'p_two_sided_vs_threshold': expected['p_two_sided_vs_threshold'],
        'signflip_p_two_sided_vs_threshold': expected['signflip_p_two_sided_vs_threshold'],
        'primary_threshold_nat': threshold,
        'primary_threshold_supported': expected['primary_threshold_supported'],
        'runtime_seconds': result.get('runtime_seconds'),
        'seed': result.get('seed'),
        'model': result.get('model'),
        'model_revision': result.get('model_revision'),
        'gpu': result.get('gpu'),
        'quant': result.get('quant'),
        'samples_sha256': manifest.get('samples_sha256') if manifest else None,
        'sample_keys': keys,
        'no_multiplicity_adjustment': 'single primary contrast; secondary analyses excluded here',
    }


def format_concise(report):
    """Format the decision-relevant estimates without a per-sample data dump."""
    low, high = report['ci95_t']
    return '\n'.join((
        f"confirmation: {report['n_effective']}/{report['n_requested']} complete; "
        f"failures={report['n_failures']}; unique_videos={report['n_unique_videos']}",
        f"mean margin drop: target={report['mean_target_margin_drop_nat']:+.4f}, "
        f"matched control={report['mean_control_margin_drop_nat']:+.4f} nat",
        f"selective effect={report['mean_selective_margin_drop_nat']:+.4f} nat "
        f"(SD={report['sd_selective_margin_drop_nat']:.4f}; "
        f"95% t CI [{low:+.4f}, {high:+.4f}])",
        f"p vs 0={report['p_two_sided_vs_zero_descriptive']:.4f} (descriptive); "
        f"p vs +{report['primary_threshold_nat']:.2f} nat="
        f"{report['p_two_sided_vs_threshold']:.3g}",
        f"preregistered success rule met: {report['primary_threshold_supported']}",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--json', action='store_true',
                        help='print the full machine-readable report instead of a concise summary')
    args = parser.parse_args()
    result_path = Path(args.input)
    manifest_path = Path(args.manifest)
    report = analyze(json.loads(result_path.read_text()),
                     json.loads(manifest_path.read_text()))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(format_concise(report))


if __name__ == '__main__':
    main()
