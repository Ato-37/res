"""Head screening analysis (plan 5): y_S = b0 + sum_h b_h x_h over random head subsets S; lasso + video-bootstrap stability selection.
numpy only (no sklearn in the isolated venv). Input arrays: X (rows x heads, 0/1), y (rows), vid (rows, video id = resampling unit).
Selection rule is fixed before outcomes: head h is 'stable' if it is selected (|b|>0) in >= `thr` of bootstrap resamples AND its sign agrees in >= `sign_thr`.
"""
import numpy as np


def lasso_cd(X, y, lam, n_iter=300, tol=1e-6):
    """Coordinate descent for (1/2n)||y - b0 - Xb||^2 + lam*||b||_1 on centred data. Returns (b0, b)."""
    n, p = X.shape; xm = X.mean(0); ym = y.mean(); Xc = X - xm; yc = y - ym
    sq = (Xc ** 2).sum(0) / n; b = np.zeros(p); r = yc.copy()
    for _ in range(n_iter):
        mx = 0.0
        for j in range(p):
            if sq[j] == 0: continue
            rho = (Xc[:, j] @ r) / n + sq[j] * b[j]
            nb = np.sign(rho) * max(abs(rho) - lam, 0.0) / sq[j]
            if nb != b[j]: r -= Xc[:, j] * (nb - b[j]); mx = max(mx, abs(nb - b[j])); b[j] = nb
        if mx < tol: break
    return ym - xm @ b, b


def lam_max(X, y):
    Xc = X - X.mean(0); return np.abs(Xc.T @ (y - y.mean())).max() / len(y)


def perm_lambda(X, y, vid, n_perm=50, q=0.95, seed=0):
    """Penalty calibrated to the null: q-quantile of lam_max after permuting y within each video (breaks the X-y link, keeps video offsets)."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_perm):
        yp = y.copy()
        for v in np.unique(vid):
            i = np.where(vid == v)[0]; yp[i] = y[rng.permutation(i)]
        out.append(lam_max(X, yp))
    return float(np.quantile(out, q))


def stability_selection(X, y, vid, mult=1.0, n_boot=200, thr=0.8, sign_thr=0.9, seed=0):
    """Bootstrap over videos (whole video resampled). lam = mult * permutation-null lambda (fixed before the bootstrap). Returns dict of arrays."""
    rng = np.random.default_rng(seed); uv = np.unique(vid); idx_by = {v: np.where(vid == v)[0] for v in uv}
    lam = mult * perm_lambda(X, y, vid, seed=seed); p = X.shape[1]; sel = np.zeros(p); pos = np.zeros(p); neg = np.zeros(p)
    for _ in range(n_boot):
        ids = np.concatenate([idx_by[v] for v in rng.choice(uv, len(uv), replace=True)])
        _, b = lasso_cd(X[ids], y[ids], lam); sel += b != 0; pos += b > 0; neg += b < 0
    freq = sel / n_boot; sgn = np.maximum(pos, neg) / np.maximum(sel, 1)
    return {"lam": lam, "freq": freq, "sign_agree": sgn, "stable": (freq >= thr) & (sgn >= sign_thr), "sign": np.sign(pos - neg)}
