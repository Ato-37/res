"""Qwen2.5-VL port, TVBench paper protocol (tvbpp): relative change of the answer probability (%) when cross-frame attention is blocked in layers 1-14 / 15-28,
baseline-correct videos, per task (same metric as the official-code Table 2/B reproduction). Bootstrap 95% CI (video unit, 10k, seed 0). Derived-only output."""
import json, random, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src)); rng = random.Random(0)
tasks = ["action_sequence", "moving_direction", "object_count", "scene_transition"]
res = {"source_videos": d.get("requested_videos"), "metric": "mean relative change of answer probability (%), cross-frame blocked, baseline-correct videos", "bootstrap": "video unit, 10k, seed 0", "tasks": {}}
def ci(v):
    m = sorted(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(10000)); return [round(m[250], 1), round(m[9749], 1)]
for t in tasks:
    row = {}
    for name, layers in (("layers_1_14", [1, 14]), ("layers_15_28", [15, 28])):
        v = [a["relative_change_percent"] for r in d["records"] if r["task"] == t and r["baseline_correct"] for a in r["arms"] if a["kind"] == "half" and a["target"] == "cross-frame" and a["layers"] == layers]
        row[name] = {"n": len(v), "mean_pct": round(sum(v) / len(v), 1), "ci95": ci(v)}
    res["tasks"][t] = row
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1, ensure_ascii=False))
