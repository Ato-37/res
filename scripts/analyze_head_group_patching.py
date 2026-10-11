"""Analyze singleton/group head patch results (CPU only, exploratory).

Usage:
    python scripts/analyze_head_group_patching.py \
        --input results/head_group_patch_smoke.json

Bootstrap intervals are pointwise. Singleton rescue p-values receive a
two-sided paired sign-flip max-|t| adjustment across all layer/head cells.
This randomization test assumes symmetric sample-level null differences.
The harmed subset is selected by observed swap damage, so its summaries are
descriptive only.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

from analyze_activation_patching import fmt, summarize


def group_key(group):
    return '+'.join(str(head) for head in group)


def max_t_sign_flip(records, layers, groups, n_permutations=10000, seed=0):
    """FWER-adjust singleton rescue tests by paired sample-level sign flips.

    The family is all layer x singleton-head cells. Sign flipping assumes the
    sample-level null differences are symmetric around zero; results are a
    screening correction, not a substitute for independent replication.
    """
    singleton_keys = [group_key(group) for group in groups if len(group) == 1]
    if not singleton_keys or len(records) < 2:
        return {}, {'method': 'paired sign-flip max-|t|', 'family_size': 0,
                    'n_permutations': n_permutations, 'seed': seed}
    columns = [(layer, key) for layer in layers for key in singleton_keys]
    matrix = np.asarray([
        [record['layers'][str(layer)][key]['patch_rescue'] for layer, key in columns]
        for record in records
    ], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite singleton rescue in max-T family')
    n = matrix.shape[0]
    means = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(n)
    observed = np.divide(means, standard_errors,
                         out=np.zeros_like(means), where=standard_errors > 0)
    observed[(standard_errors == 0) & (means != 0)] = np.inf
    rng = np.random.default_rng(seed)
    max_statistics = np.empty(n_permutations, dtype=np.float64)
    batch_size = 128
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, n))
        permuted_means = (signs @ matrix) / n
        permuted_statistics = np.divide(
            permuted_means, standard_errors,
            out=np.zeros_like(permuted_means), where=standard_errors > 0,
        )
        permuted_statistics[:, (standard_errors == 0) & (means != 0)] = np.inf
        max_statistics[start:stop] = np.max(np.abs(permuted_statistics), axis=1)
    adjusted = (1 + (max_statistics[:, None] >= np.abs(observed)[None, :]).sum(axis=0)) / (n_permutations + 1)
    return {column: (float(observed[index]), float(adjusted[index]))
            for index, column in enumerate(columns)}, {
                'method': 'paired sample-level sign-flip max-|t|, two-sided',
                'family_size': len(columns), 'n_permutations': n_permutations,
                'seed': seed,
                'assumption': 'sample-level null differences are symmetric around zero',
            }


def analyze(result, n_boot=10000, seed=0):
    if n_boot < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    records = result.get('records', [])
    layers = [int(layer) for layer in result.get('layers', [])]
    groups = [list(group) for group in result.get('groups', [])]
    keys = [group_key(group) for group in groups]
    if len(set(keys)) != len(keys) or not layers:
        raise ValueError('Invalid or duplicate layer/group metadata')
    ids = [str(record['sample_id']) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample_id')
    expected_checks = result.get('self_check_checks_per_record', 0)
    if result.get('self_check') and not expected_checks:
        # Results written before the reporting-field addition still contain
        # the individual checks, so infer the expected cardinality safely.
        expected_checks = 2 * len(layers) * len(keys)
    for record in records:
        if not math.isfinite(record['swap_drop']):
            raise ValueError('Nonfinite swap_drop')
        if any(str(layer) not in record.get('layers', {}) for layer in layers):
            raise ValueError('Incomplete layer coverage')
        for layer in layers:
            if set(record['layers'][str(layer)]) != set(keys):
                raise ValueError(f'Incomplete group coverage for sample {record["sample_id"]}')
        if result.get('self_check'):
            checks = record.get('self_checks', [])
            if len(checks) != expected_checks:
                raise ValueError(f'Incomplete self-check for sample {record["sample_id"]}')
            tolerance = result.get('self_check_atol')
            if tolerance is None or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Invalid self-check tolerance')
            for check in checks:
                error = check.get('absolute_error')
                if not math.isfinite(error) or error < 0 or error > tolerance:
                    raise ValueError(f'Failed self-check for sample {record["sample_id"]}')
    harmed = [record for record in records if record['swap_drop'] < 0]
    adjusted, multiplicity = max_t_sign_flip(records, layers, groups, n_boot, seed)
    rows = []
    for scope, subset in (('all', records), ('harmed', harmed)):
        for layer in layers:
            for key in keys:
                values = [record['layers'][str(layer)][key]['patch_rescue'] for record in subset]
                stats = summarize([str(record['sample_id']) for record in subset], values,
                                  n_boot=n_boot, seed=seed)
                if scope == 'all' and '+' not in key and key.isdigit() and adjusted:
                    t_stat, p_max_t = adjusted[(layer, key)]
                    stats.update(t_stat=t_stat, p_max_t=p_max_t,
                                 multiplicity_family_size=multiplicity['family_size'])
                rows.append(dict(scope=scope, layer=layer, group=key, **stats))
    # For fixed pairs, quantify deviation from the sum of the two singleton
    # rescues. This is an interaction on the log-probability score scale, not
    # proof that the heads are causally independent components.
    for scope, subset in (('all', records), ('harmed', harmed)):
        for layer in layers:
            singleton_keys = {key for key in keys if '+' not in key and key.isdigit()}
            for key in keys:
                heads = key.split('+')
                if len(heads) != 2 or any(head not in singleton_keys for head in heads):
                    continue
                values = []
                for record in subset:
                    layer_values = record['layers'][str(layer)]
                    values.append(layer_values[key]['patch_rescue']
                                  - sum(layer_values[head]['patch_rescue'] for head in heads))
                stats = summarize([str(record['sample_id']) for record in subset], values,
                                  n_boot=n_boot, seed=seed)
                rows.append(dict(scope=scope, layer=layer, group=key,
                                 metric='pair_excess_over_singletons', **stats))
    return {'schema_version': 1, 'n_requested': result.get('n_requested'),
            'n_effective': len(records), 'n_harmed': len(harmed), 'layers': layers,
            'groups': groups, 'failures': result.get('failures', []), 'seed': seed,
            'n_boot': n_boot,
            'singleton_multiplicity': multiplicity,
            'interpretation': 'Exploratory pointwise bootstrap; singleton p_max_t uses '
            'two-sided paired sample-level sign-flip max-|t| across layer x singleton-head '
            'tests and assumes symmetric null differences; '
            'harmed subset selected by observed damage. Head-group patching is diagnostic, '
            'not proof of independent head causality.', 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output')
    parser.add_argument('--n-boot', type=int, default=10000,
                        help='Bootstrap draws and paired sign-flip max-T permutations')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.input).resolve():
        parser.error('--output must differ from input')
    report = analyze(json.loads(Path(args.input).read_text()), args.n_boot, args.seed)
    report['input'] = args.input
    print(report['interpretation'])
    print(f"n_requested={report['n_requested']} n_effective={report['n_effective']} "
          f"n_harmed={report['n_harmed']} failures={len(report['failures'])}")
    print('scope layer group metric n mean ci95 p_max_t median positive_fraction')
    for row in report['rows']:
        print(f"{row['scope']} {row['layer']} {row['group']} {row.get('metric', 'rescue')} {row['n']} "
              f"{fmt(row['mean'])} {fmt(row['mean_ci95'])} {fmt(row.get('p_max_t'))} "
              f"{fmt(row['median'])} "
              f"{fmt(row['positive_fraction'])}")
    for failure in report['failures']:
        print(f'failure={failure}')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
