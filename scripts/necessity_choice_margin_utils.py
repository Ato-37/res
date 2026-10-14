"""Dependency-light utilities for temporal choice-margin necessity analyses."""
from math import inf, isfinite
from random import Random
from statistics import stdev


MIN_BOOTSTRAP_DRAWS = 100


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def uniform_sample_indices(total_frames, num_frames):
    """Match the repository's endpoint-inclusive rounded uniform sampling.

    The returned list may contain duplicate indices for clips shorter than the
    requested frame count. Callers that need distinct decoded frames must
    deduplicate and enforce their own minimum coverage.
    """
    for name, value in (("total_frames", total_frames), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return [round(i * (total_frames - 1) / max(num_frames - 1, 1))
            for i in range(num_frames)]


def _validate_bootstrap_inputs(values, draws):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < MIN_BOOTSTRAP_DRAWS:
        raise ValueError(f'draws must be an integer >= {MIN_BOOTSTRAP_DRAWS}')
    try:
        finite = all(isfinite(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        raise ValueError('values must contain only finite numbers')


def strict_evidence_indices(indices, fps, intervals):
    return [i for i in indices if any(start <= i / fps <= end for start, end in intervals)]


def nearest_control_indices(indices, evidence):
    """Choose equal-count non-evidence indices minimizing temporal rank-distance."""
    evidence = set(evidence)
    k = len(evidence)
    controls = [i for i in indices if i not in evidence]
    if k == 0 or len(controls) < k:
        return []
    evidence_positions = sorted(indices.index(i) for i in evidence)
    control_positions = [indices.index(i) for i in controls]

    # Sequence-alignment DP: each cell stores (minimum cost, selected ranks).
    # Comparing these pairs preserves the declared lexicographic tie-break.
    previous = [(0, ()) for _ in range(len(control_positions) + 1)]
    for evidence_rank, evidence_position in enumerate(evidence_positions, start=1):
        current = [(inf, ())]
        for control_count, control_position in enumerate(control_positions, start=1):
            skip = current[-1]
            prior = previous[control_count - 1]
            take = (prior[0] + abs(evidence_position - control_position),
                    prior[1] + (control_position,))
            current.append(min(skip, take))
        previous = current
    selected_positions = previous[-1][1]
    if len(selected_positions) != k:
        return []
    return [indices[position] for position in selected_positions]


def assign_evidence_control_frames(total_frames, num_frames, fps, intervals):
    """Build a deterministic pre-score evidence/control assignment.

    This composes the frozen endpoint-inclusive sampler, inclusive
    frame-index/FPS interval membership, and equal-count temporal control. It
    validates geometry only; it cannot authenticate or interpret intervals.
    """
    if not _finite_number(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(intervals, list):
        raise ValueError("intervals must be a list")
    parsed_intervals = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each interval must have two endpoints")
        start, end = interval
        if not _finite_number(start) or not _finite_number(end) or start < 0 or end < start:
            raise ValueError("interval endpoints must be finite nonnegative-duration times")
        parsed_intervals.append((start, end))
    if parsed_intervals != sorted(parsed_intervals) or len(set(parsed_intervals)) != len(parsed_intervals):
        raise ValueError("intervals must be sorted and unique")

    sampled = uniform_sample_indices(total_frames, num_frames)
    indices = sorted(set(sampled))
    base = {
        "requested_frame_count": num_frames,
        "sampled_frame_count": len(indices),
        "sampled_indices": indices,
        "evidence_indices": [],
        "control_indices": [],
        "matched_rank_distance": None,
    }
    if len(indices) != num_frames:
        return {"status": "insufficient_distinct_sample_frames", **base}

    evidence = strict_evidence_indices(indices, fps, parsed_intervals)
    if not evidence:
        return {"status": "no_evidence", **base}
    control = nearest_control_indices(indices, evidence)
    if len(control) != len(evidence):
        return {"status": "insufficient_control", **base,
                "evidence_indices": evidence}
    rank = {frame_index: position for position, frame_index in enumerate(indices)}
    cost = sum(abs(rank[a] - rank[b]) for a, b in zip(evidence, control))
    return {"status": "ok", **base, "evidence_indices": evidence,
            "control_indices": control, "matched_rank_distance": cost}


def assess_frame_coverage(indices, fps, intervals):
    """Return strict evidence/control indices and a machine-readable status."""
    evidence = strict_evidence_indices(indices, fps, intervals)
    if not evidence:
        return {'status': 'no_evidence', 'evidence': [], 'control': []}
    control = nearest_control_indices(indices, evidence)
    if len(control) != len(evidence):
        return {'status': 'insufficient_control', 'evidence': evidence, 'control': []}
    return {'status': 'ok', 'evidence': evidence, 'control': control}


def apply_mask_fill(image, fill, blur_radius=12.0):
    """Apply a spatially matched frame perturbation without changing dimensions."""
    if fill == 'black':
        from PIL import Image
        return Image.new('RGB', image.size, (0, 0, 0))
    if fill == 'blur':
        if blur_radius <= 0:
            raise ValueError('blur_radius must be positive for blur fill')
        from PIL import ImageFilter
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    raise ValueError(f"Unknown mask fill: {fill}")


def validate_choice_logprobs(values):
    """Reject malformed/non-finite A–E scores before they enter a run summary."""
    if len(values) != 5:
        raise ValueError(f'Expected five choice log-probabilities; got {len(values)}')
    if not all(isfinite(value) for value in values):
        raise ValueError('Choice log-probabilities must all be finite')
    return values


def _bootstrap_mean(values, rng):
    return sum(values[rng.randrange(len(values))] for _ in values) / len(values)


def summarize(values, draws, seed):
    _validate_bootstrap_inputs(values, draws)
    if not values:
        return {'n': 0, 'mean': None, 'sd': None, 'ci95_pointwise': None,
                'positive_fraction': None, 'signflip_p_two_sided': None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {'n': 1, 'mean': mean, 'sd': None, 'ci95_pointwise': None,
                'positive_fraction': float(values[0] > 0), 'signflip_p_two_sided': None}
    rng = Random(seed)
    boots = sorted(_bootstrap_mean(values, rng) for _ in range(draws))
    sd = stdev(values)
    ci = [boots[int((draws - 1) * q)] for q in (.025, .975)]
    observed = abs(mean)
    extreme = sum(abs(sum(v * rng.choice((-1, 1)) for v in values) / len(values)) >= observed
                  for _ in range(draws))
    return {'n': len(values), 'mean': mean, 'sd': sd, 'ci95_pointwise': ci,
            'positive_fraction': sum(v > 0 for v in values) / len(values),
            'signflip_p_two_sided': (1 + extreme) / (draws + 1)}


def interaction_ci(correct, incorrect, draws, seed):
    _validate_bootstrap_inputs(correct, draws)
    _validate_bootstrap_inputs(incorrect, draws)
    if len(correct) < 2 or len(incorrect) < 2:
        return {'mean_correct_minus_incorrect': None, 'ci95_pointwise': None}
    rng = Random(seed)
    boot = []
    for _ in range(draws):
        a = _bootstrap_mean(correct, rng)
        b = _bootstrap_mean(incorrect, rng)
        boot.append(a - b)
    boot.sort()
    return {'mean_correct_minus_incorrect': sum(correct) / len(correct)
            - sum(incorrect) / len(incorrect),
            'ci95_pointwise': [boot[int((draws - 1) * q)] for q in (.025, .975)]}
