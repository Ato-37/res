"""ddmin (Zeller & Hildebrandt) for the smallest blocked head set that keeps >= frac of the all-heads effect. No main-effect assumption.
`effect(S)` -> effect of blocking the head set S (e.g. cohort-mean confidence loss, deterministic). Returns a 1-minimal set: removing any single head drops effect below the threshold.
Counterpart for AND-type interactions: the search never splits a required group permanently, because it refines granularity down to single heads."""
import math


def ddmin(items, effect, frac=0.8, max_calls=100000):
    full = effect(list(items)); thr = frac * full; calls = [1]; cache = {}
    def ok(s):
        k = tuple(sorted(s))
        if k not in cache:
            calls[0] += 1; cache[k] = effect(list(k)) >= thr
        return cache[k]
    cur = list(items); n = 2
    if full <= 0: return None, full, calls[0]
    while len(cur) >= 2 and calls[0] < max_calls:
        size = math.ceil(len(cur) / n); chunks = [cur[i:i + size] for i in range(0, len(cur), size)]; reduced = False
        for c in chunks:                       # try a single chunk alone (keep only c blocked)
            if ok(c): cur, n, reduced = c, 2, True; break
        if not reduced:
            for c in chunks:                   # try the complement of a chunk (remove c)
                comp = [x for x in cur if x not in set(c)]
                if comp and ok(comp): cur, n, reduced = comp, max(n - 1, 2), True; break
        if not reduced:
            if n >= len(cur): break
            n = min(len(cur), n * 2)
    return cur, full, calls[0]
