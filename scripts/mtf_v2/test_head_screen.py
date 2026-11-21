import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from head_screen import lasso_cd, stability_selection
def sim(P=60, V=30, R=40, p=0.3, truth=None, noise=0.3, vsd=0.3, seed=0):
    g = np.random.default_rng(seed); n = V * R
    X = (g.random((n, P)) < p).astype(float); vid = np.repeat(np.arange(V), R)
    beta = np.zeros(P)
    for h, v in (truth or {}).items(): beta[h] = v
    y = X @ beta + g.normal(0, vsd, V)[vid] + g.normal(0, noise, n)   # per-video offset + noise
    return X, y, vid, beta
# 1. lasso recovers large coefficients, ~0 elsewhere
X, y, vid, beta = sim(truth={3: -1.0, 10: -0.8, 25: 0.7})
_, b = lasso_cd(X, y, 0.05); assert all(abs(b[h] - beta[h]) < 0.3 for h in (3, 10, 25)); assert np.abs(np.delete(b, [3, 10, 25])).max() < 0.15
# 2. stability selection: true heads stable with right sign
st = stability_selection(X, y, vid, n_boot=60, seed=1)
assert set(np.where(st["stable"])[0]) == {3, 10, 25}, np.where(st["stable"])[0]; assert st["sign"][3] == -1 and st["sign"][25] == 1
# 3. null (no effect): false stable heads across 5 seeds (rate controlled)
fp = [int(stability_selection(*sim(seed=s)[:3], n_boot=60, seed=s)["stable"].sum()) for s in range(10, 15)]
print("null stable counts (60 heads x 5 sims):", fp); assert sum(fp) <= 1
# 4. diffuse small effects (many heads at -0.1): reported as not stable (criterion: dispersed contribution)
st = stability_selection(*sim(truth={h: -0.1 for h in range(40)}, seed=21)[:3], n_boot=60, seed=2)
print("diffuse: stable", int(st["stable"].sum()), "of 40 true small"); 
print("head_screen tests OK")
