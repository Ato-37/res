#!/usr/bin/env python3
"""Outcome-blind screen: which ranked candidates admit an equal-count disjoint sham (annotation groups <= half).

Uses the base runner's own cohort freezer (no model, no scores). Writes a new mode-0600 exclusion manifest that adds
the media hashes of ineligible candidates to a prior exclusion manifest. Prints counts only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias as base

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-media-sha256", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--pool-n", type=int, required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--dataset-split", default="val")
    ap.add_argument("--output-manifest", type=Path, required=True)
    args = ap.parse_args(argv)
    private = (ROOT / "data/private_nextgqa").resolve(strict=True)
    payload, _ = base.read_private_json(args.exclude_media_sha256, private)
    exclusion = base.cohort.validate_private_media_exclusion(payload)
    selected, counts, meta, _ = base._freeze_exact(ROOT, video_root=args.video_root.resolve(strict=True),
        n=args.pool_n, frame_count=args.frames, exclusion=exclusion, lock_assignments=None,
        dataset_split=args.dataset_split)
    patch, groups_total = 2, args.frames // 2
    bad = []
    for s in selected:
        g = len({i // patch for i in s["evidence_positions"]})
        if not (0 < g and groups_total - g >= g):
            bad.append(s["decode"]["source_sha256"])
    out = {"media_sha256": sorted(set(exclusion) | set(bad)), "schema_version": 1}
    fd = os.open(args.output_manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(out, fh)
    print(json.dumps({"pool": len(selected), "ineligible_for_sham": len(bad), "eligible": len(selected) - len(bad)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
