"""Pure helpers for cumulative layer-randomization result integrity and summaries."""

import math
import statistics


def summarize_cumulative_records(records, expected_depths):
    """Aggregate by randomized depth; samples, not repeated conditions, are units."""
    if expected_depths < 1:
        raise ValueError('expected_depths must be positive')
    if not records:
        return []
    if any(len(record.get('cumulative_conditions', [])) != expected_depths
           for record in records):
        raise ValueError('records do not have the expected cumulative-depth coverage')
    keys = [record.get('sample_key') for record in records]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError('sample keys must be nonempty and unique')
    summaries = []
    for index in range(expected_depths):
        conditions = [record['cumulative_conditions'][index] for record in records]
        if any(condition.get('cumulative_randomized_layers') != index + 1
               for condition in conditions):
            raise ValueError('cumulative depth order mismatch')
        rho = [float(condition['spearman_rho']) for condition in conditions]
        overlap = [float(condition['top4_overlap_fraction']) for condition in conditions]
        changed = [bool(condition['prediction_changed']) for condition in conditions]
        if any(not math.isfinite(value) or not -1 <= value <= 1 for value in rho):
            raise ValueError('invalid Spearman correlation')
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in overlap):
            raise ValueError('invalid top-4 overlap')
        summaries.append({
            'cumulative_randomized_layers': index + 1,
            'n_samples': len(records),
            'mean_sample_spearman_rho': statistics.mean(rho),
            'mean_sample_top4_overlap_fraction': statistics.mean(overlap),
            'prediction_change_fraction': sum(changed) / len(changed),
        })
    return summaries


def validate_accounting(result, requested, max_randomized_layers=None):
    effective = result.get('n_effective')
    failures = result.get('n_failures')
    records = result.get('records')
    order = result.get('randomized_layer_order')
    if (result.get('n_requested') != requested or not isinstance(effective, int)
            or not isinstance(failures, int) or not isinstance(records, list)
            or not isinstance(order, list)):
        raise ValueError('missing or invalid result accounting')
    if len(records) != effective or effective + failures != requested:
        raise ValueError('sample accounting does not reconcile')
    if max_randomized_layers is not None and len(order) != max_randomized_layers:
        raise ValueError('randomized layer count differs from request')
    if order != sorted(order, reverse=True) or len(order) != len(set(order)):
        raise ValueError('randomized layer order must be unique and last-to-first')
    for record in records:
        if record.get('randomized_layer_order') != order:
            raise ValueError('per-sample layer order differs from result configuration')
        conditions = record.get('cumulative_conditions', [])
        if any(condition.get('randomized_layer') != order[index]
               for index, condition in enumerate(conditions)):
            raise ValueError('per-sample randomized layer sequence mismatch')
    summaries = summarize_cumulative_records(records, len(order)) if records else []
    if result.get('descriptive_by_randomized_depth') != summaries:
        raise ValueError('published cumulative-depth summaries do not match records')
    return True
