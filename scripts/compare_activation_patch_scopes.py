"""Compare paired all/prompt/answer patch runs (CPU only, exploratory).

Usage: python scripts/compare_activation_patch_scopes.py --all all.json \
    --prompt prompt.json --answer answer.json --output comparison.json

Nonadditivity = rescue_all - rescue_prompt - rescue_answer. This is a
score-scale interaction between position interventions, not head synergy.
"""
import argparse
import json
import math
from pathlib import Path

from analyze_activation_patching import analyze, fmt, summarize


def compare(runs, n_boot=10000, seed=0):
    if set(runs) != {'all', 'prompt', 'answer'}:
        raise ValueError('Need all, prompt and answer runs')
    base = runs['all']
    indexed = {}
    audits = {}
    for scope in ('all', 'prompt', 'answer'):
        run = runs[scope]
        if run.get('patch_scope') != scope:
            raise ValueError(f'Wrong patch_scope for {scope}')
        for key in ('samples_file', 'model', 'quant', 'dtype', 'attn_implementation',
                    'num_frames', 'layers', 'n_requested', 'strict_grounding'):
            if key not in run or run[key] != base.get(key):
                raise ValueError(f'Incompatible setting: {scope}/{key}')
        if not run.get('strict_grounding') or not run.get('self_check'):
            raise ValueError('Require strict grounding and self-check in every run')
        audits[scope] = analyze(run, n_boot, seed)
        if (audits[scope]['outside_grounding_evidence_ids'] or
                audits[scope]['coverage_unknown_ids']):
            raise ValueError(f'Invalid grounding coverage: {scope}')
        indexed[scope] = {str(r['sample_id']): r for r in run['records']}
        if set(indexed[scope]) != set(indexed['all']):
            raise ValueError('Effective sample IDs differ; refusing silent intersection')
        for record in run['records']:
            expected = {(name, layer) for name in ('full', 'swapped') for layer in run['layers']}
            checks = record.get('self_checks', [])
            if (len(checks) != len(expected) or
                    {(c['input'], c['layer']) for c in checks} != expected):
                raise ValueError('Incomplete self-check coverage')
            tolerance = run.get('self_check_atol', -1)
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Invalid self-check tolerance')
            if any(not math.isfinite(c['absolute_error']) or
                   not 0 <= c['absolute_error'] <= tolerance for c in checks):
                raise ValueError('Failed self-check')
    ids = list(indexed['all'])
    if not ids:
        raise ValueError('No effective samples')
    for sid in ids:
        reference = indexed['all'][sid]
        for scope in ('prompt', 'answer'):
            record = indexed[scope][sid]
            for key in ('grounding', 'sampled_indices', 'sampled_timestamps',
                        'evidence_indices', 'control_indices'):
                if record[key] != reference[key]:
                    raise ValueError(f'Input mismatch: {sid}/{scope}/{key}')
            for key in ('full_sequence_logprob', 'swapped_sequence_logprob', 'swap_drop'):
                if not math.isclose(record[key], reference[key], rel_tol=0, abs_tol=1e-4):
                    raise ValueError(f'Baseline mismatch: {sid}/{scope}/{key}')
    rows, per_sample = [], []
    for layer in base['layers']:
        values = {}
        for sid in ids:
            r = {scope: indexed[scope][sid]['layers'][str(layer)]['patch_rescue']
                 for scope in runs}
            values[sid] = dict(r, prompt_minus_answer=r['prompt'] - r['answer'],
                              nonadditivity=r['all'] - r['prompt'] - r['answer'])
            per_sample.append(dict(sample_id=sid, layer=layer, **values[sid]))
        for subset in ('all', 'harmed'):
            selected = [sid for sid in ids if subset == 'all' or indexed['all'][sid]['swap_drop'] < 0]
            for metric in ('all', 'prompt', 'answer', 'prompt_minus_answer', 'nonadditivity'):
                rows.append(dict(subset=subset, layer=layer, metric=metric,
                                 **summarize(selected, [values[sid][metric] for sid in selected],
                                             n_boot=n_boot, seed=seed)))
    return dict(n_effective=len(ids), seed=seed, n_boot=n_boot, rows=rows,
                per_sample=per_sample, failures={s: r.get('failures', []) for s, r in runs.items()},
                interpretation='Exploratory pointwise bootstrap; no multiplicity correction; '
                'harmed subset selected by observed damage. Nonadditivity is not head synergy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for scope in ('all', 'prompt', 'answer'):
        parser.add_argument('--' + scope, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    paths = {s: Path(getattr(args, s)) for s in ('all', 'prompt', 'answer')}
    if Path(args.output).resolve() in {p.resolve() for p in paths.values()}:
        parser.error('Output must differ from inputs')
    report = compare({s: json.loads(p.read_text()) for s, p in paths.items()}, args.n_boot, args.seed)
    report['inputs'] = {s: str(p) for s, p in paths.items()}
    print(report['interpretation'])
    print('subset layer metric n mean ci95')
    for row in report['rows']:
        print(f"{row['subset']} {row['layer']} {row['metric']} {row['n']} "
              f"{fmt(row['mean'])} {fmt(row['mean_ci95'])}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
