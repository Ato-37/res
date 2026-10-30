#!/usr/bin/env python3
"""Official logit-lens output (task 0, Action Sequence): per-layer share of video tokens whose top-1 is a temporal/spatial keyword.

Keyword lists are the official script's (spatial and temporal); first sub-word via the model tokenizer. Derived counts only.
Criterion (d), descriptive: temporal-keyword share in middle layers exceeds early layers.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

SPATIAL = ['bag', 'bed', 'blanket', 'book', 'box', 'cabinet', 'camera', 'clothes', 'cup', 'glass', 'bottle', 'dish', 'door', 'floor', 'food',
           'laptop', 'paper', 'person', 'phone', 'sandwich', 'table']
TEMPORAL = ['eat', 'close', 'do', 'down', 'drink', 'hold', 'on', 'open', 'put', 'sit', 'throw', 'tidy', 'take', 'up']


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsons", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    first = lambda ws: {tok.decode(tok(w, add_special_tokens=False)["input_ids"][0]) for w in ws}
    sets = {"spatial": first(SPATIAL), "temporal": first(TEMPORAL)}
    rows = [r for f in glob.glob(f"{a.jsons}/*.json") for r in json.loads(Path(f).read_text())]
    nl = max(int(k) for r in rows for k in r["video"]) + 1
    share = {k: [0.0] * nl for k in sets}
    for r in rows:
        for l, toks in r["video"].items():
            for k, s in sets.items():
                share[k][int(l)] += sum(t[0] in s for t in toks) / len(toks) / len(rows)
    bands = {"early": range(0, nl // 3), "mid": range(nl // 3, 2 * nl // 3), "late": range(2 * nl // 3, nl)}
    summ = {k: {b: round(sum(v[i] for i in r) / len(r), 5) for b, r in bands.items()} for k, v in share.items()}
    out = {"n_questions": len(rows), "n_layers": nl, "band_mean_top1_share": summ, "per_layer": {k: [round(x, 5) for x in v] for k, v in share.items()},
           "criterion_d_temporal_mid_gt_early": summ["temporal"]["mid"] > summ["temporal"]["early"]}
    path = Path(a.output)
    if path.exists():
        raise SystemExit("output_exists")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_questions", "n_layers", "band_mean_top1_share", "criterion_d_temporal_mid_gt_early")}, indent=1))


if __name__ == "__main__":
    main()
