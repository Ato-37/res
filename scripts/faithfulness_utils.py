"""Dependency-light helpers for the preregistered head-faithfulness test."""

import math
import random


def select_top_heads(attributions, k):
    """Return the K largest finite per-head attribution rows deterministically."""
    if k < 1 or k > len(attributions):
        raise ValueError(f'k must be in 1..{len(attributions)}')
    rows = []
    seen = set()
    for item in attributions:
        layer, head, score = int(item['layer']), int(item['head']), float(item['score'])
        if not math.isfinite(score) or score < 0:
            raise ValueError(f'Invalid attribution for layer={layer}, head={head}: {score}')
        key = (layer, head)
        if key in seen:
            raise ValueError(f'Duplicate attribution for layer={layer}, head={head}')
        seen.add(key)
        rows.append({'layer': layer, 'head': head, 'score': score})
    rows.sort(key=lambda row: (-row['score'], row['layer'], row['head']))
    return rows[:k]


def format_choice_prompt(question, choices):
    labels = 'ABCDE'
    if len(choices) != len(labels):
        raise ValueError('exactly five choices are required')
    options = '\n'.join(f'{label}. {choice}' for label, choice in zip(labels, choices))
    return (f'{question}\nOptions:\n{options}\n'
            'Choose the best answer. Respond with one letter only.\nAnswer:')


def select_prediction_and_foil(choice_scores, prediction=None):
    """Freeze the highest-scoring option and return its strongest distinct foil."""
    scores = [float(value) for value in choice_scores]
    if len(scores) != 5 or any(not math.isfinite(value) for value in scores):
        raise ValueError('exactly five finite choice scores are required')
    if prediction is None:
        prediction = max(range(len(scores)), key=lambda index: scores[index])
    prediction = int(prediction)
    if prediction < 0 or prediction >= len(scores):
        raise ValueError('prediction index outside 0..4')
    foil = max((index for index in range(len(scores)) if index != prediction),
               key=lambda index: scores[index])
    return prediction, foil, scores[prediction] - scores[foil]


def gradient_activation_scores(activations_by_head, gradients_by_head):
    """Compute |dot(z_h, dmargin/dz_h)| and L2 norm for each head."""
    if len(activations_by_head) != len(gradients_by_head) or not activations_by_head:
        raise ValueError('activation and gradient head arrays must have equal nonzero length')
    scores, norms = [], []
    for head, (activation, gradient) in enumerate(zip(activations_by_head, gradients_by_head)):
        if len(activation) != len(gradient) or not activation:
            raise ValueError(f'head {head} activation/gradient dimensions do not match')
        if any(not math.isfinite(float(value)) for value in activation + gradient):
            raise ValueError(f'head {head} contains a non-finite value')
        score = abs(sum(float(a) * float(g) for a, g in zip(activation, gradient)))
        norm = math.sqrt(sum(float(a) ** 2 for a in activation))
        scores.append(score)
        norms.append(norm)
    return scores, norms


def build_matched_control_groups(selected, activation_norms, num_heads=28,
                                 repeats=5, seed=0):
    """Match control heads by exact layer counts and nearest output norm.

    Controls never include a selected head and are not reused across repeats.
    This makes the five control groups disjoint when the requested matching is
    feasible; groups remain at the same total size and layer histogram.
    """
    if repeats < 1 or num_heads < 1:
        raise ValueError('repeats and num_heads must be positive')
    selected_keys = [(int(row['layer']), int(row['head'])) for row in selected]
    if not selected_keys or len(set(selected_keys)) != len(selected_keys):
        raise ValueError('selected heads must be non-empty and unique')
    if any(head < 0 or head >= num_heads for _, head in selected_keys):
        raise ValueError('selected head outside configured head range')
    if any(not math.isfinite(float(activation_norms.get(key, float('nan'))))
           or float(activation_norms[key]) < 0 for key in selected_keys):
        raise ValueError('missing or invalid selected-head activation norm')

    selected_by_layer = {}
    for layer, head in selected_keys:
        selected_by_layer.setdefault(layer, []).append(head)
    rng = random.Random(seed)
    used = {layer: set(heads) for layer, heads in selected_by_layer.items()}
    groups = []
    match_distances = []
    for _ in range(repeats):
        group = []
        for layer in sorted(selected_by_layer):
            for target in sorted(selected_by_layer[layer]):
                target_norm = float(activation_norms[(layer, target)])
                candidates = [head for head in range(num_heads)
                              if head not in used.get(layer, set())]
                if not candidates:
                    raise ValueError(f'not enough disjoint controls in layer {layer}')
                rng.shuffle(candidates)
                candidates.sort(key=lambda head: abs(
                    math.log((float(activation_norms[(layer, head)]) + 1e-12)
                             / (target_norm + 1e-12))))
                control = candidates[0]
                norm = float(activation_norms[(layer, control)])
                used.setdefault(layer, set()).add(control)
                group.append({'layer': layer, 'head': control})
                match_distances.append(abs(math.log((norm + 1e-12)
                                                   / (target_norm + 1e-12))))
        groups.append(group)
    return groups, match_distances


def mean_t_interval(values, confidence=0.95):
    """Return mean, sample SD and two-sided Student-t interval for the mean."""
    if not values:
        raise ValueError('at least one observation is required')
    vals = [float(value) for value in values]
    if any(not math.isfinite(value) for value in vals):
        raise ValueError('observations must be finite')
    avg = sum(vals) / len(vals)
    if len(vals) < 2:
        return {'mean': avg, 'sd': None, 'ci': None}
    from scipy.stats import t
    n = len(vals)
    sd = math.sqrt(sum((value - avg) ** 2 for value in vals) / (n - 1))
    critical = float(t.ppf((1 + confidence) / 2, n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {'mean': avg, 'sd': sd, 'ci': [avg - half_width, avg + half_width]}


def bootstrap_mean_interval(values, draws=10000, seed=0, confidence=0.95):
    if not values or draws < 1:
        raise ValueError('values and positive draws are required')
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choice(values) for _ in values) / len(values)
                       for _ in range(draws))
    tail = (1 - confidence) / 2
    return [estimates[int((draws - 1) * tail)],
            estimates[int((draws - 1) * (1 - tail))]]


def signflip_p_against(values, null=0.0, draws=10000, seed=0):
    """Monte-Carlo two-sided sign-flip p-value for paired deviations from null."""
    if not values or draws < 1:
        raise ValueError('values and positive draws are required')
    centered = [float(value) - null for value in values]
    observed = abs(sum(centered) / len(centered))
    rng = random.Random(seed)
    extreme = sum(abs(sum(value * rng.choice((-1, 1)) for value in centered)
                      / len(centered)) >= observed for _ in range(draws))
    return (extreme + 1) / (draws + 1)
