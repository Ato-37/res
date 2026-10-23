#!/usr/bin/env python3
"""Forest plot (dependency-free SVG) of the multi-query key-bias results on two fresh NExT-GQA val cohorts.

Aggregated statistics only (mean, 95% t CI); reads the private receipts. Usage: plot_key_bias_forest.py [output.svg]
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "private_nextgqa"
RUNS = [
    ("-ln256 suppress", "cohort 1", "annotation_key_bias_valE149_multiquery_neg_ln256_full_20261023_02.json"),
    ("-ln256 suppress", "cohort 2", "annotation_key_bias_valRep2E196_A_neg_ln256_full_20261023_11.json"),
    ("+ln4 steer", "cohort 1", "annotation_key_bias_valE149_multiquery_ln4_full_20261023_06.json"),
    ("+ln4 steer", "cohort 2", "annotation_key_bias_valRep2E196_B_ln4_full_20261023_11.json"),
    ("+ln16 steer", "cohort 1", "annotation_key_bias_valE149_multiquery_ln16_full_20261023_06.json"),
    ("+ln16 steer", "cohort 2", "annotation_key_bias_valRep2E196_C_ln16_full_20261023_11.json"),
]
MEASURES = [("primary_annotation_minus_sham_nats", "annotation - sham (contrast)", "#1f4e79"),
            ("annotation_minus_baseline_nats", "annotation - baseline", "#c0504d"),
            ("sham_minus_baseline_nats", "sham - baseline", "#7f7f7f")]
XMIN, XMAX = -1.3, 1.3


def ci(values):
    n = len(values)
    m = st.fmean(values)
    h = stats.t.ppf(0.975, n - 1) * st.stdev(values) / math.sqrt(n)
    return m, m - h, m + h


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "figures" / "key_bias_forest.svg"
    left, right, top, rowh, gap = 210, 30, 50, 17, 12
    width = 760
    plot_w = width - left - right
    height = top + len(RUNS) * (3 * rowh + gap) + 60

    def x(v):
        return left + (min(max(v, XMIN), XMAX) - XMIN) / (XMAX - XMIN) * plot_w

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{left}" y="22" font-size="14" font-weight="bold">Multi-query key bias, NExT-GQA val, Qwen2.5-VL-7B (fp32 readout)</text>',
             f'<text x="{left}" y="40" fill="#444">answer-margin change in nat: mean and 95% t CI; dashed = practical threshold +/-0.3</text>']
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        parts.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{height - 40}" stroke="{"black" if t == 0 else "#dddddd"}" stroke-width="{1.2 if t == 0 else 0.8}"/>')
        parts.append(f'<text x="{x(t)}" y="{height - 24}" text-anchor="middle" fill="#444">{t:+.1f}</text>')
    for t in (-0.3, 0.3):
        parts.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{height - 40}" stroke="#2e8b57" stroke-dasharray="5,4"/>')
    y = top
    for label, cohort, fname in RUNS:
        rows = json.loads((D / fname).read_text())["records"]
        n = len(rows)
        parts.append(f'<text x="{left - 10}" y="{y + rowh + 4}" text-anchor="end" font-weight="bold">{label}</text>')
        parts.append(f'<text x="{left - 10}" y="{y + rowh + 19}" text-anchor="end" fill="#444">{cohort} (N={n})</text>')
        for key, _, color in MEASURES:
            m, lo, hi = ci([r[key] for r in rows])
            cy = y + rowh / 2
            parts.append(f'<line x1="{x(lo)}" y1="{cy}" x2="{x(hi)}" y2="{cy}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x(m)}" cy="{cy}" r="3.6" fill="{color}"/>')
            y += rowh
        y += gap
    ly = height - 8
    for i, (_, mlabel, color) in enumerate(MEASURES):
        lx = left + i * 190
        parts.append(f'<circle cx="{lx}" cy="{ly - 4}" r="4" fill="{color}"/><text x="{lx + 9}" y="{ly}" fill="#222">{mlabel}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
