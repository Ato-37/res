#!/usr/bin/env python3
"""Analysis of run_tvbench_logit_lens_v2.py (plan: experiments/2026-10-31_logit_lens_v2_and_fine_pathway_plan.md section B). Derived-only output.
Per layer and lens (raw/tuned): probe fidelity KL(final||lens) per video token, keyword top-1 rate, keyword top-5 rate, keyword probability mass (temporal/spatial),
last-token answer margin (gold-foil option log-prob), for baseline and cross-frame-blocked arms. Band means (early 1-4, middle 9-20, late 21-28) with bootstrap 95% CI (video unit, 10k, seed 20261031)."""
import argparse, json, random
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
d = json.load(open(a.receipt)); rng = random.Random(20261031)
words = d["words"]; ti = [i for i, w in enumerate(words) if w["kind"] == "temporal"]; si = [i for i, w in enumerate(words) if w["kind"] == "spatial"]
L = d["layer_count"]; recs = d["records"]; n = len(recs)
BANDS = {"early_1_4": range(1, 5), "middle_9_20": range(9, 21), "late_21_28": range(21, 29)}
def metric(rec, arm, layer, lens, what):
    row = rec[arm]["layers"][layer - 1]
    r = row.get(lens) or row["raw"]; nt = rec["n_video_tokens"]
    if what == "kl": return r["kl_sum"] / nt
    if what == "top1_temporal": return sum(r["top1_counts"][i] for i in ti) / nt * 100
    if what == "top1_spatial": return sum(r["top1_counts"][i] for i in si) / nt * 100
    if what == "mass_temporal": return r["mass_sum"]["temporal"] / nt * 100
    if what == "mass_spatial": return r["mass_sum"]["spatial"] / nt * 100
    if what == "top5_temporal": return r["top5_hits"]["temporal"] / nt * 100
    if what == "top5_spatial": return r["top5_hits"]["spatial"] / nt * 100
    if what == "margin": return r["answer_margin"]
def ci(v):
    v = np.array(v, float); n_ = len(v)
    m = sorted(float(v[[rng.randrange(n_) for _ in range(n_)]].mean()) for _ in range(2000))
    return [round(m[49], 4), round(m[1949], 4)]
out = {"n_test": n, "n_train": d["n_train"], "train_video_tokens": d["train_video_tokens"], "per_layer": {}, "bands": {}}
whats = ["kl", "top1_temporal", "top1_spatial", "mass_temporal", "mass_spatial", "top5_temporal", "top5_spatial", "margin"]
for arm in ("baseline", "crossframe_blocked"):
    for lens in ("raw", "tuned"):
        for w in whats:
            out["per_layer"][f"{arm}|{lens}|{w}"] = [round(float(np.mean([metric(r, arm, l, lens, w) for r in recs])), 4) for l in range(1, L + 1)]
            for b, rg in BANDS.items():
                per_video = [float(np.mean([metric(r, arm, l, lens, w) for l in rg])) for r in recs]
                out["bands"][f"{arm}|{lens}|{w}|{b}"] = {"mean": round(float(np.mean(per_video)), 4), "ci95": ci(per_video)}
# paired contrasts (video unit): tuned minus raw KL (middle band); middle minus early temporal mass (tuned); blocked minus baseline temporal mass in middle (tuned)
def band(arm, lens, w, b): return [float(np.mean([metric(r, arm, l, lens, w) for l in BANDS[b]])) for r in recs]
def paired(x, y): dlt = np.array(x) - np.array(y); return {"mean": round(float(dlt.mean()), 4), "ci95": ci(dlt)}
out["contrasts"] = {
 "kl_tuned_minus_raw_middle": paired(band("baseline", "tuned", "kl", "middle_9_20"), band("baseline", "raw", "kl", "middle_9_20")),
 "temporal_mass_middle_minus_early_tuned": paired(band("baseline", "tuned", "mass_temporal", "middle_9_20"), band("baseline", "tuned", "mass_temporal", "early_1_4")),
 "temporal_mass_middle_minus_early_raw": paired(band("baseline", "raw", "mass_temporal", "middle_9_20"), band("baseline", "raw", "mass_temporal", "early_1_4")),
 "temporal_mass_blocked_minus_baseline_middle_tuned": paired(band("crossframe_blocked", "tuned", "mass_temporal", "middle_9_20"), band("baseline", "tuned", "mass_temporal", "middle_9_20")),
 "spatial_mass_middle_minus_early_tuned": paired(band("baseline", "tuned", "mass_spatial", "middle_9_20"), band("baseline", "tuned", "mass_spatial", "early_1_4")),
}
json.dump(out, open(a.output, "w"), indent=1)
for k, v in out["contrasts"].items(): print(k, v)
for lens in ("raw", "tuned"):
    print(lens, "KL by layer", out["per_layer"][f"baseline|{lens}|kl"])
    print(lens, "temporal mass %", out["per_layer"][f"baseline|{lens}|mass_temporal"])
    print(lens, "answer margin", out["per_layer"][f"baseline|{lens}|margin"])
