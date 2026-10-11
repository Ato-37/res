"""Dependency-light utilities for temporal choice-margin necessity analyses."""
from itertools import combinations
from random import Random
from statistics import stdev


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
    best, best_cost = None, None
    for combo in combinations(controls, k):
        control_positions = [indices.index(i) for i in combo]
        key = (sum(abs(a - b) for a, b in zip(evidence_positions, control_positions)),
               tuple(control_positions))
        if best_cost is None or key < best_cost:
            best, best_cost = list(combo), key
    return best


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


def _bootstrap_mean(values, rng):
    return sum(values[rng.randrange(len(values))] for _ in values) / len(values)


def summarize(values, draws, seed):
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
