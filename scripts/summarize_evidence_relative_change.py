"""Paper-comparable metric for the evidence-interval knockout (WP1): mean relative change of answer probability (%)
on baseline-correct videos, per arm (V2T band layers 6-20), bootstrap 95% CI (video = unit, 10k, seed 0).
Reads the private per-video record file (local only) and writes a derived-only summary. Descriptive/exploratory:
the prespecified primary endpoint of WP1 is the margin contrast; this is the base paper's metric on the same data."""
import json, random, sys
src, out = sys.argv[1], sys.argv[2]
_d = json.load(open(src)); _lst = next(v for v in _d.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "knockouts" in v[0]); rows = [r for r in _lst if not r.get("excluded") and r["baseline_correct"]]
def arm(r, a): return next(k["relative_change_percent"] for k in r["knockouts"] if k["arm"] == a)
rng = random.Random(0)
def ci(v):
    m = sorted(sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v) for _ in range(10000))
    return round(m[250], 2), round(m[9749], 2)
res = {"n_baseline_correct": len(rows), "metric": "mean relative change of answer probability (%), V2T band layers 6-20", "bootstrap": "video unit, 10k, seed 0"}
vals = {}
for a in ("v2t_allvideo_band", "v2t_evidence_band", "v2t_sham_band", "v2l_evidence_band"):
    v = [arm(r, a) for r in rows]; vals[a] = v
    res[a] = {"mean_pct": round(sum(v) / len(v), 2), "ci95": ci(v)}
d = [e - s for e, s in zip(vals["v2t_evidence_band"], vals["v2t_sham_band"])]
res["evidence_minus_sham_pct"] = {"mean_pct": round(sum(d) / len(d), 2), "ci95": ci(d)}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
