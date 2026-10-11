"""Integrity-check and descriptively summarize readout-randomization results."""

import argparse
import json
import math
from pathlib import Path
import statistics


EXPERIMENT = 'head_attribution_output_readout_randomization_sanity'


def summarize_result(result, manifest=None):
    if result.get('experiment') != EXPERIMENT:
        raise ValueError('unexpected experiment identifier')
    if result.get('status') != 'exploratory_sanity_check_not_faithfulness_test':
        raise ValueError('result is not marked as an exploratory sanity check')
    requested, effective, failures = (result.get(key) for key in
                                      ('n_requested', 'n_effective', 'n_failures'))
    records = result.get('records')
    randomizations = result.get('randomizations_per_sample')
    if (not all(isinstance(value, int) for value in (requested, effective, failures,
                                                       randomizations))
            or not isinstance(records, list)):
        raise ValueError('missing integer counts or records')
    if (len(records) != effective or effective + failures != requested
            or randomizations < 1):
        raise ValueError('sample accounting does not reconcile')
    if result.get('randomized_parameter_scope') != (
            'five A-E output-readout rows only; transformer layers unchanged'):
        raise ValueError('unexpected randomized parameter scope')

    keys = [record.get('sample_key') for record in records]
    if (any(not isinstance(key, str) or not key for key in keys)
            or len(keys) != len(set(keys))):
        raise ValueError('sample keys must be nonempty and unique')

    per_sample, all_runs = [], []
    changed_predictions = 0
    for record in records:
        baseline_prediction = record.get('baseline_prediction_index')
        runs = record.get('randomized_readout_runs')
        if (not isinstance(baseline_prediction, int) or not 0 <= baseline_prediction < 5
                or not isinstance(runs, list) or len(runs) != randomizations):
            raise ValueError(f'invalid prediction or replicate count in {record["sample_key"]}')
        rho_values, overlap_values = [], []
        for run in runs:
            rho = run.get('spearman_rho')
            overlap = run.get('top_k_overlap_fraction')
            overlap_count = run.get('top_k_overlap_count')
            randomized_prediction = run.get('randomized_prediction_index')
            if (not isinstance(rho, (int, float)) or not math.isfinite(rho)
                    or not -1 <= rho <= 1):
                raise ValueError('invalid Spearman correlation')
            if (not isinstance(overlap, (int, float)) or not math.isfinite(overlap)
                    or not 0 <= overlap <= 1 or overlap_count not in range(5)):
                raise ValueError('invalid top-k overlap')
            if not math.isclose(overlap, overlap_count / 4, rel_tol=0, abs_tol=1e-9):
                raise ValueError('top-k overlap fraction/count mismatch')
            if (not isinstance(randomized_prediction, int)
                    or not 0 <= randomized_prediction < 5):
                raise ValueError('invalid randomized prediction index')
            rho_values.append(float(rho))
            overlap_values.append(float(overlap))
            changed_predictions += int(randomized_prediction != baseline_prediction)
            all_runs.append((float(rho), float(overlap)))

        row = {
            'sample_key': record['sample_key'],
            'randomizations': len(runs),
            'mean_spearman_rho': statistics.mean(rho_values),
            'mean_top4_overlap_fraction': statistics.mean(overlap_values),
        }
        if not math.isclose(row['mean_spearman_rho'], record.get('mean_spearman_rho', math.nan),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError(f'published per-sample Spearman mean mismatch in {record["sample_key"]}')
        if not math.isclose(row['mean_top4_overlap_fraction'],
                            record.get('mean_top4_overlap_fraction', math.nan),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError(f'published per-sample overlap mean mismatch in {record["sample_key"]}')
        per_sample.append(row)

    if effective:
        mean_rho = statistics.mean(row['mean_spearman_rho'] for row in per_sample)
        mean_overlap = statistics.mean(row['mean_top4_overlap_fraction'] for row in per_sample)
        if not math.isclose(mean_rho, result.get('mean_sample_spearman_rho', math.nan),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError('published aggregate Spearman mean mismatch')
        if not math.isclose(mean_overlap,
                            result.get('mean_sample_top4_overlap_fraction', math.nan),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError('published aggregate top-k overlap mismatch')
        run_mean_rho = statistics.mean(rho for rho, _ in all_runs)
        run_mean_overlap = statistics.mean(overlap for _, overlap in all_runs)
    else:
        mean_rho = mean_overlap = run_mean_rho = run_mean_overlap = None

    if manifest is not None:
        for key in ('experiment', 'n_requested', 'n_effective', 'n_failures',
                    'randomizations_per_sample', 'seed', 'quant'):
            result_key = key
            if result.get(result_key) != manifest.get(key):
                raise ValueError(f'result/run-manifest mismatch for {key}')
        if not isinstance(manifest.get('samples_sha256'), str):
            raise ValueError('run manifest has no sample-manifest hash')

    return {
        'experiment': EXPERIMENT,
        'interpretation_scope': 'descriptive only; not a faithfulness test',
        'n_requested': requested,
        'n_effective': effective,
        'n_failures': failures,
        'randomizations_per_sample': randomizations,
        'n_randomization_runs': len(all_runs),
        'mean_sample_spearman_rho': mean_rho,
        'mean_sample_top4_overlap_fraction': mean_overlap,
        'mean_run_spearman_rho': run_mean_rho,
        'mean_run_top4_overlap_fraction': run_mean_overlap,
        'prediction_change_count': changed_predictions,
        'prediction_change_fraction': (changed_predictions / len(all_runs)
                                       if all_runs else None),
        'per_sample': per_sample,
        'runtime_seconds': result.get('runtime_seconds'),
        'model_revision': result.get('model_revision'),
        'randomized_parameter_scope': result['randomized_parameter_scope'],
        'limitations': result.get('limitations', []),
        'manifest_sha256': manifest.get('samples_sha256') if manifest else None,
        'no_inferential_test': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--manifest')
    args = parser.parse_args()
    result_path = Path(args.input)
    result = json.loads(result_path.read_text())
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    summary = summarize_result(result, manifest)
    print(f'input={result_path}')
    if args.manifest:
        print(f'manifest={args.manifest}')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
