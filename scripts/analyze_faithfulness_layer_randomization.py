"""Integrity-check and summarize cumulative decoder-layer randomization results."""

import argparse
import json
import math
from pathlib import Path
import statistics

from faithfulness_layer_randomization_utils import validate_accounting


EXPERIMENT = 'head_attribution_cascading_layer_randomization'
STATUS = 'exploratory_sanity_check_not_faithfulness_test'


def analyze(result, manifest=None):
    if result.get('experiment') != EXPERIMENT:
        raise ValueError('unexpected experiment identifier')
    if result.get('status') != STATUS:
        raise ValueError('result is not marked as an exploratory sanity check')
    order = result.get('randomized_layer_order')
    if not isinstance(order, list) or len(order) != result.get('num_layers_total'):
        raise ValueError('randomization order must cover all decoder layers')
    validate_accounting(result, result.get('n_requested'), len(order))
    if result.get('randomized_parameters') != (
            'all decoder-block linear weights/biases and RMS/LayerNorm affine parameters'):
        raise ValueError('unexpected randomized parameter scope')

    records = result['records']
    if not records:
        raise ValueError('no samples completed successfully')
    for record in records:
        baseline_prediction = record.get('baseline_prediction_index')
        if not isinstance(baseline_prediction, int) or not 0 <= baseline_prediction < 5:
            raise ValueError('invalid baseline prediction index')
        for condition in record['cumulative_conditions']:
            rho = condition.get('spearman_rho')
            overlap = condition.get('top4_overlap_fraction')
            overlap_count = condition.get('top4_overlap_count')
            randomized_prediction = condition.get('randomized_prediction_index')
            if not isinstance(rho, (int, float)) or not math.isfinite(rho) or not -1 <= rho <= 1:
                raise ValueError('invalid Spearman correlation')
            if (not isinstance(overlap, (int, float)) or not math.isfinite(overlap)
                    or not 0 <= overlap <= 1 or overlap_count not in range(5)):
                raise ValueError('invalid top-4 overlap')
            if not math.isclose(overlap, overlap_count / 4, rel_tol=0, abs_tol=1e-9):
                raise ValueError('top-4 overlap fraction/count mismatch')
            if not isinstance(randomized_prediction, int) or not 0 <= randomized_prediction < 5:
                raise ValueError('invalid randomized prediction index')

    summaries = result['descriptive_by_randomized_depth']
    first_nonpositive = next((row['cumulative_randomized_layers'] for row in summaries
                              if row['mean_sample_spearman_rho'] <= 0), None)
    first_near_zero = next((row['cumulative_randomized_layers'] for row in summaries
                            if row['mean_sample_spearman_rho'] <= 0.1), None)
    first_zero_top4 = next((row['cumulative_randomized_layers'] for row in summaries
                            if row['mean_sample_top4_overlap_fraction'] == 0), None)
    final = summaries[-1]

    if manifest is not None:
        for key in ('experiment', 'n_requested', 'n_effective', 'n_failures', 'seed',
                    'quant', 'model', 'randomized_layer_order', 'randomization_mode',
                    'num_frames'):
            if result.get(key) != manifest.get(key):
                raise ValueError(f'result/run-manifest mismatch for {key}')
        sample_hash = manifest.get('samples_sha256')
        if (not isinstance(sample_hash, str) or len(sample_hash) != 64
                or any(char not in '0123456789abcdef' for char in sample_hash.lower())):
            raise ValueError('run manifest has no valid sample-manifest SHA-256')

    return {
        'interpretation_scope': 'descriptive sanity check only; not a faithfulness test',
        'n_requested': result['n_requested'],
        'n_effective': result['n_effective'],
        'n_failures': result['n_failures'],
        'n_layers': len(order),
        'n_heads_per_ranking': result.get('heads_per_ranking'),
        'sample_keys': [record['sample_key'] for record in records],
        'layer_order': order,
        'mean_spearman_depth_1': summaries[0]['mean_sample_spearman_rho'],
        'mean_spearman_depth_28': final['mean_sample_spearman_rho'],
        'mean_top4_overlap_depth_1': summaries[0]['mean_sample_top4_overlap_fraction'],
        'mean_top4_overlap_depth_28': final['mean_sample_top4_overlap_fraction'],
        'prediction_change_fraction_depth_28': final['prediction_change_fraction'],
        'first_depth_mean_spearman_nonpositive': first_nonpositive,
        'first_depth_mean_spearman_at_most_0_1': first_near_zero,
        'first_depth_mean_top4_overlap_zero': first_zero_top4,
        'per_sample_final': [
            {
                'sample_key': record['sample_key'],
                'spearman_rho': record['cumulative_conditions'][-1]['spearman_rho'],
                'top4_overlap_count': record['cumulative_conditions'][-1]['top4_overlap_count'],
                'prediction_changed': record['cumulative_conditions'][-1]['prediction_changed'],
            }
            for record in records
        ],
        'runtime_seconds': result.get('runtime_seconds'),
        'seed': result.get('seed'),
        'model': result.get('model'),
        'model_revision': result.get('model_revision'),
        'gpu': result.get('gpu'),
        'quant': result.get('quant'),
        'manifest_samples_sha256': manifest.get('samples_sha256') if manifest else None,
        'inferential_tests': 'none; three validation-development videos are descriptive only',
        'limitations': result.get('limitations', []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--manifest')
    args = parser.parse_args()
    result_path = Path(args.input)
    result = json.loads(result_path.read_text())
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    report = analyze(result, manifest)
    print(f'input={result_path}')
    if args.manifest:
        print(f'manifest={args.manifest}')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
