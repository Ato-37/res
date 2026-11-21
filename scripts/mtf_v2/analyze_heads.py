"""Analyze flow_heads_v2 output with head_screen (selection rule fixed in head_screen.py defaults: thr 0.8, sign 0.9, perm-null lambda)."""
import json, sys, numpy as np, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from head_screen import stability_selection, lasso_cd
f = sys.argv[1]; n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 200
R = [json.loads(l) for l in open(f)]; meta = R[0]["meta"]; R = R[1:]; layers = meta["layers"]; H = meta["H"]; col = lambda l, h: layers.index(l) * H + h
for task in sorted({r["task"] for r in R}):
    T = [r for r in R if r["task"] == task]; allr = [r for r in T if r["cond"] == "all"]; S = [r for r in T if r["cond"] != "all"]
    X = np.zeros((len(S), len(layers) * H)); 
    for i, r in enumerate(S):
        for l, hs in r["subset"].items():
            for h in hs: X[i, col(int(l), h)] = 1
    y = np.array([r["d_margin"] for r in S]); vid = np.array([r["video"] for r in S])
    # per-video centring is NOT applied (video offsets handled by resampling unit); report variance explained by additive fit on all data
    st = stability_selection(X, y, vid, n_boot=n_boot, seed=0); b0, b = lasso_cd(X, y, st["lam"]); res = y - (b0 + X @ b)
    print(f"{task}: videos {len(set(vid))}, rows {len(S)}, mean d_margin all-heads(range) {np.mean([r['d_margin'] for r in allr]):+.3f}, mean S {y.mean():+.3f} (mean n_heads {np.mean([r['n_heads'] for r in S]):.0f}), sd y {y.std():.3f}")
    print(f"  lambda {st['lam']:.4f}; lasso nonzero {int((b != 0).sum())}; R2(in-sample, lasso) {1 - res.var() / y.var():.3f}; stable heads {int(st['stable'].sum())}")
    for j in np.argsort(-st["freq"])[:8]: print(f"   L{layers[j // H]} h{j % H}: freq {st['freq'][j]:.2f} sign {int(st['sign'][j]):+d} agree {st['sign_agree'][j]:.2f} beta {b[j]:+.3f}")
