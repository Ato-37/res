"""Frozen dev/test split of TVBench (5 tasks) by video, hash-based. python make_split.py --out splits/tvbench_split_v1.json
Rule: key=sha256("mtf2-split-v1|task|video"); dev if int(key[:8],16) even else test. Order within split = hash order (pilot = first n)."""
import argparse, hashlib, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.expanduser("~/external/tvbench_root/json")); ap.add_argument("--out", required=True); a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"exists {a.out}")
TASKS = {"Action Antonym": "action_antonym.json", "Action Sequence": "action_sequence.json", "Moving Direction": "moving_direction.json", "Object Count": "object_count.json", "Scene Transition": "scene_transition.json"}
out = {"rule": "sha256(mtf2-split-v1|task|video) hex[:8] even -> dev else test; order = hash order", "tasks": {}}
for t, fn in TASKS.items():
    vids = sorted({x["video"] for x in json.load(open(os.path.join(a.root, fn)))})
    h = sorted(((hashlib.sha256(f"mtf2-split-v1|{t}|{v}".encode()).hexdigest(), v) for v in vids))
    out["tasks"][t] = {"dev": [v for k, v in h if int(k[:8], 16) % 2 == 0], "test": [v for k, v in h if int(k[:8], 16) % 2 == 1]}
    print(t, "videos", len(vids), "dev", len(out["tasks"][t]["dev"]), "test", len(out["tasks"][t]["test"]))
json.dump(out, open(a.out, "w"), indent=1)
