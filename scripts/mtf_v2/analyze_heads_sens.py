"""EXPLORATORY sensitivity (post-hoc, not the prespecified analysis): within-video centring of X and y (video fixed effects) before the same lasso/stability rule; plus proportionality check mean(y_S) vs fraction of heads."""
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from head_screen import stability_selection
R = [json.loads(l) for l in open(sys.argv[1])]; meta = R[0]["meta"]; R = R[1:]; layers = meta["layers"]; H = meta["H"]
for task in sorted({r["task"] for r in R}):
    S = [r for r in R if r["task"] == task and r["cond"] != "all"]; A = {r["video"]: r["d_margin"] for r in R if r["task"] == task and r["cond"] == "all"}
    X = np.zeros((len(S), len(layers) * H))
    for i, r in enumerate(S):
        for l, hs in r["subset"].items():
            for h in hs: X[i, layers.index(int(l)) * H + h] = 1
    y = np.array([r["d_margin"] for r in S]); vid = np.array([r["video"] for r in S])
    for v in np.unique(vid): i = vid == v; X[i] -= X[i].mean(0); y[i] -= y[i].mean()
    st = stability_selection(X, y, vid, n_boot=100, seed=0)
    frac = np.array([r["n_heads"] for r in S]) / (len(layers) * H); yy = np.array([r["d_margin"] for r in S]); a = np.array([A[r["video"]] for r in S])
    print(f"{task}: [within-video centred] stable heads {int(st['stable'].sum())}; max freq {st['freq'].max():.2f}")
    print(f"   proportionality: corr(y_S, frac*all_heads_effect) = {np.corrcoef(yy, frac * a)[0, 1]:.3f}; slope {np.polyfit(frac * a, yy, 1)[0]:.2f} (1 = effect proportional to share of heads)")
