"""WP1S (sliding window) in the base paper's metric: relative change of the predicted option's probability (%), baseline-correct videos.
Per-video least-squares slope of %p on window overlap fraction (effect = -slope, %p points per 0->100% overlap); window-pooled means by overlap bin.
Bootstrap 95% CI (video = unit, 10k, seed 0). Exploratory (WP1S's prespecified primary is the margin slope). Derived-only output."""
import json, random, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src)); rows = [r for r in d["records"] if not r.get("excluded") and r["baseline_correct"]]
slopes, pooled = [], {"0": [], "1-49": [], "50-99": [], "100": []}
for r in rows:
    pts = [(k["overlap_fraction"], k["relative_change_percent"]) for k in r["knockouts"] if k["arm"].startswith("v2t_dose")]
    for o, v in pts:
        b = "0" if o == 0 else "100" if o >= 1 else "1-49" if o < 0.5 else "50-99"
        pooled[b].append(v)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx > 0: slopes.append(sum((x - mx) * (y - my) for x, y in pts and zip(xs, ys)) / sxx)
eff = [-s for s in slopes]
rng = random.Random(0); m = sorted(sum(eff[rng.randrange(len(eff))] for _ in eff) / len(eff) for _ in range(10000))
res = {"n_baseline_correct": len(rows), "n_with_slope": len(eff), "effect_pct_points_per_0_to_100_overlap": {"mean": round(sum(eff) / len(eff), 2), "ci95": [round(m[250], 2), round(m[9749], 2)]},
       "pooled_mean_pct_by_overlap_bin": {k: {"n_windows": len(v), "mean": round(sum(v) / len(v), 2)} for k, v in pooled.items()}}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
