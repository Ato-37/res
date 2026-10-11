"""Small, deterministic helpers for output-readout randomization diagnostics."""

import math


def compare_attribution_rankings(original, randomized, top_k=4):
    """Compare two complete head-score vectors using Spearman rho and top-K overlap."""
    if len(original) != len(randomized) or not original:
        raise ValueError('rankings must have equal nonzero length')
    if not 1 <= top_k <= len(original):
        raise ValueError('top_k outside ranking length')
    original = [float(value) for value in original]
    randomized = [float(value) for value in randomized]
    if any(not math.isfinite(value) or value < 0 for value in original + randomized):
        raise ValueError('attribution scores must be finite and nonnegative')
    from scipy.stats import spearmanr
    rho = float(spearmanr(original, randomized).statistic)
    if not math.isfinite(rho):
        raise ValueError('Spearman correlation is undefined for constant attribution scores')
    top_original = sorted(range(len(original)), key=lambda i: (-original[i], i))[:top_k]
    top_randomized = sorted(range(len(randomized)), key=lambda i: (-randomized[i], i))[:top_k]
    overlap = len(set(top_original).intersection(top_randomized))
    return {
        'spearman_rho': rho,
        'top_k_overlap_count': overlap,
        'top_k_overlap_fraction': overlap / top_k,
        'original_top_indices': top_original,
        'randomized_top_indices': top_randomized,
    }
