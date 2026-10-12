"""Compare two completed necessity smoke artifacts without publishing record IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw.get("n_requested") != 3 or raw.get("n_effective") != 3 or raw.get("failures"):
        raise ValueError("both inputs must be complete three-sample smokes")
    rows = {r["sample_id"]: r for r in raw.get("records", [])}
    if len(rows) != 3:
        raise ValueError("records must contain three unique sample IDs")
    return raw | {"_rows": rows}


def compare(left_path: Path, right_path: Path, *, draws: int = 10000, seed: int = 20261013) -> dict:
    if left_path.resolve() == right_path.resolve():
        raise ValueError("conditions must be distinct")
    left, right = _load(left_path), _load(right_path)
    for key in ("model", "seed", "num_frames", "frame_size", "max_pixels"):
        if left.get(key) != right.get(key):
            raise ValueError(f"conditions disagree on shared setting: {key}")
    if set(left["_rows"]) != set(right["_rows"]):
        raise ValueError("conditions do not share the same cohort")
    pairs = []
    primary = []
    for sid in sorted(left["_rows"]):
        a, b = left["_rows"][sid], right["_rows"][sid]
        da = float(a["selective_necessity_margin_drop"])
        db = float(b["selective_necessity_margin_drop"])
        delta = da - db
        pairs.append(delta)
        if a.get("baseline_choice_correct") and b.get("baseline_choice_correct"):
            primary.append(delta)
    if not primary:
        raise ValueError("no paired baseline-correct records")
    rng = np.random.default_rng(seed)
    x = np.asarray(primary, dtype=float)
    boot = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return {
        "schema_version": 1,
        "analysis": "necessity_choice_paired_condition_comparison",
        "status": "engineering_smoke_only",
        "left_raw_artifact_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "right_raw_artifact_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        "n_paired": len(pairs),
        "n_primary_paired": len(primary),
        "paired_delta_definition": "left selective margin drop minus right selective margin drop",
        "all_mean_delta_nat": float(np.mean(pairs)),
        "primary_mean_delta_nat": float(np.mean(x)),
        "primary_bootstrap_ci95_nat": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_draws": draws,
        "seed": seed,
        "limitations": [
            "Raw questions/QIDs/video paths and sample IDs are intentionally excluded.",
            "This is a paired engineering smoke comparison, not direct explanation faithfulness evidence.",
            "No between-condition hypothesis was preregistered; the estimate is descriptive.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=10000)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(compare(args.left, args.right, draws=args.bootstrap_draws), indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
