"""Compare adjacent-pair and fixed gap-2 pair interactions.

The base result must contain singleton and adjacent groups. The control result
contains ``skip1`` groups (head i with head i+2). Both runs must have identical
inputs, baselines, effective IDs, and self-check coverage.

Usage:
    python scripts/compare_head_group_pair_controls.py \
        --base results/head_group_patch_pairs_heldout.json \
        --control results/head_group_patch_gap2_heldout.json \
        --output results/head_group_pair_control_analysis.json
"""
import argparse
import json
import math
from pathlib import Path

from analyze_activation_patching import fmt, summarize


def key(group):
    return '+'.join(str(head) for head in group)


def validate_run(result):
    layers = [int(layer) for layer in result.get('layers', [])]
    groups = [list(group) for group in result.get('groups', [])]
    keys = [key(group) for group in groups]
    if len(keys) != len(set(keys)) or not layers:
        raise ValueError('Invalid layer/group metadata')
    if not result.get('self_check'):
        raise ValueError('Both runs require --self-check')
    expected = result.get('self_check_checks_per_record', 0)
    if not expected:
        expected = 2 * len(layers) * len(keys)
    tolerance = result.get('self_check_atol')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid self-check tolerance')
    indexed = {}
    for record in result.get('records', []):
        sample_id = str(record['sample_id'])
        if sample_id in indexed:
            raise ValueError('Duplicate sample_id')
        if len(record.get('self_checks', [])) != expected:
            raise ValueError(f'Incomplete self-check: {sample_id}')
        if any(not math.isfinite(c['absolute_error']) or c['absolute_error'] > tolerance
               for c in record['self_checks']):
            raise ValueError(f'Failed self-check: {sample_id}')
        for layer in layers:
            if set(record.get('layers', {}).get(str(layer), {})) != set(keys):
                raise ValueError(f'Incomplete groups: {sample_id}/{layer}')
        indexed[sample_id] = record
    return layers, groups, indexed


def compare(base, control, n_boot=10000, seed=0):
    base_layers, base_groups, base_records = validate_run(base)
    control_layers, control_groups, control_records = validate_run(control)
    if base_layers != control_layers:
        raise ValueError('Layer settings differ')
    for field in ('samples_file', 'model', 'quant', 'dtype', 'attn_implementation',
                  'num_frames', 'patch_scope', 'strict_grounding', 'n_requested'):
        if base.get(field) != control.get(field):
            raise ValueError(f'Run setting differs: {field}')
    if set(base_records) != set(control_records):
        raise ValueError('Effective sample IDs differ')
    ids = list(base_records)
    base_keys = {key(group): group for group in base_groups}
    control_keys = {key(group): group for group in control_groups}
    singleton_keys = {key(group) for group in base_groups if len(group) == 1}
    adjacent_keys = {key(group) for group in base_groups
                     if len(group) == 2 and group[1] == group[0] + 1}
    gap2_keys = {key(group) for group in control_groups
                 if len(group) == 2 and group[1] == group[0] + 2}
    if not singleton_keys or adjacent_keys != {key(group) for group in base_groups
                                               if len(group) == 2 and group[1] == group[0] + 1}:
        raise ValueError('Base run lacks singleton/adjacent metadata')
    if not gap2_keys:
        raise ValueError('Control run lacks gap-2 groups')
    for sample_id in ids:
        a, b = base_records[sample_id], control_records[sample_id]
        for field in ('full_sequence_logprob', 'swapped_sequence_logprob', 'swap_drop'):
            if not math.isclose(a[field], b[field], rel_tol=0, abs_tol=1e-4):
                raise ValueError(f'Baseline differs: {sample_id}/{field}')
        for field in ('grounding', 'sampled_indices', 'sampled_timestamps',
                      'evidence_indices', 'control_indices'):
            if a[field] != b[field]:
                raise ValueError(f'Input metadata differs: {sample_id}/{field}')
    rows = []
    contrasts = []
    for layer in base_layers:
        adjacent_by_sample, gap2_by_sample = {}, {}
        for sample_id in ids:
            record = base_records[sample_id]
            single = record['layers'][str(layer)]
            adjacent_values = [single[group]['patch_rescue']
                               - sum(single[head]['patch_rescue'] for head in base_keys[group])
                               for group in sorted(adjacent_keys)
                               for head in base_keys[group]
                               if head in singleton_keys]
            # The comprehension above repeats pair values once per head; use
            # explicit pair loops to keep exactly one value per pair.
            adjacent_values = [single[group]['patch_rescue']
                               - sum(single[str(head)]['patch_rescue']
                                     for head in base_keys[group])
                               for group in sorted(adjacent_keys)]
            control_layers_for_sample = control_records[sample_id]['layers'][str(layer)]
            gap2_values = [control_layers_for_sample[group]['patch_rescue']
                           - sum(single[str(head)]['patch_rescue']
                                 for head in control_keys[group])
                           for group in sorted(gap2_keys)]
            adjacent_by_sample[sample_id] = sum(adjacent_values) / len(adjacent_values)
            gap2_by_sample[sample_id] = sum(gap2_values) / len(gap2_values)
            contrasts.append(dict(sample_id=sample_id, layer=layer,
                                  adjacent_mean=adjacent_by_sample[sample_id],
                                  gap2_mean=gap2_by_sample[sample_id],
                                  adjacent_minus_gap2=(adjacent_by_sample[sample_id]
                                                       - gap2_by_sample[sample_id])))
        for label, values in (('adjacent_mean', adjacent_by_sample),
                              ('gap2_mean', gap2_by_sample)):
            rows.append(dict(layer=layer, metric=label,
                             **summarize(ids, [values[sample_id] for sample_id in ids],
                                         n_boot=n_boot, seed=seed)))
        difference = [adjacent_by_sample[sample_id] - gap2_by_sample[sample_id]
                      for sample_id in ids]
        rows.append(dict(layer=layer, metric='adjacent_minus_gap2',
                         **summarize(ids, difference, n_boot=n_boot, seed=seed)))
    return {'schema_version': 1, 'n_effective': len(ids), 'layers': base_layers,
            'seed': seed, 'n_boot': n_boot, 'rows': rows, 'per_sample': contrasts,
            'interpretation': 'Exploratory paired distance control; pointwise bootstrap; '
            'no multiplicity correction. Interaction is score nonadditivity, not proof '
            'of head causality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if Path(args.output).resolve() in {Path(args.base).resolve(), Path(args.control).resolve()}:
        parser.error('Output must differ from inputs')
    report = compare(json.loads(Path(args.base).read_text()),
                     json.loads(Path(args.control).read_text()), args.n_boot, args.seed)
    print(report['interpretation'])
    print(f"n_effective={report['n_effective']}")
    print('layer metric n mean ci95 median positive_fraction')
    for row in report['rows']:
        print(f"{row['layer']} {row['metric']} {row['n']} {fmt(row['mean'])} "
              f"{fmt(row['mean_ci95'])} {fmt(row['median'])} {fmt(row['positive_fraction'])}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
