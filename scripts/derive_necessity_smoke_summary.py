"""Derive a privacy-safe summary from a local necessity smoke JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def derive(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    if (raw.get("n_requested"), raw.get("n_effective"), len(raw.get("failures", []))) != (3, 3, 0):
        raise ValueError("expected complete three-sample smoke")
    for key in ("all", "baseline_correct_primary"):
        if key not in raw.get("summary", {}):
            raise ValueError(f"missing summary cohort: {key}")
    if raw.get("cap_pixels_per_frame") not in (True, False, None):
        raise ValueError("invalid cap flag")
    all_summary = raw["summary"]["all"]
    primary = raw["summary"]["baseline_correct_primary"]
    return {
        "schema_version": 1,
        "analysis": "necessity_choice_margin_smoke_derived",
        "status": "engineering_smoke_only",
        "raw_artifact_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "n_requested": 3,
        "n_effective": 3,
        "n_failures": 0,
        "cap_pixels_per_frame": raw.get("cap_pixels_per_frame"),
        "mask_fill": raw.get("mask_intervention", {}).get("fill"),
        "all_mean_nat": all_summary["mean"],
        "all_ci95_pointwise_nat": all_summary["ci95_pointwise"],
        "primary_n": primary["n"],
        "primary_mean_nat": primary["mean"],
        "primary_ci95_pointwise_nat": primary["ci95_pointwise"],
        "limitations": ["Raw questions/QIDs/video paths are intentionally not included.", "This is a preprocessing/engineering smoke, not direct explanation faithfulness evidence."],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(derive(args.raw), indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
