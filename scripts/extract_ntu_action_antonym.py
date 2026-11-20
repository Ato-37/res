#!/usr/bin/env python3
"""Extract only the NTU RGB+D videos that TVBench Action Antonym needs from the downloaded NTU zip files (local-only; do not redistribute).

Usage:
  python scripts/extract_ntu_action_antonym.py --zip-dir ~/external/ntu_zips --out data/private_tvbench_extra/video/action_antonym
Reads the wanted file names from data/private_tvbench_extra/json/action_antonym.json (320 videos), scans every *.zip under --zip-dir,
extracts matching members by base name (any folder layout inside the zip), never overwrites, and prints which videos are still missing.
"""
from __future__ import annotations

import argparse, json, shutil, sys, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default="data/private_tvbench_extra/json/action_antonym.json")
    args = ap.parse_args()
    wanted = {q["video"] for q in json.loads(Path(args.json).read_text())}
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in out.iterdir()}
    todo = wanted - have
    zips = sorted(Path(args.zip_dir).expanduser().glob("**/*.zip"))
    if not zips:
        raise SystemExit("no_zip_files_found")
    extracted = 0
    for z in zips:
        if not todo:
            break
        try:
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    name = Path(info.filename).name
                    if name in todo:
                        target = out / name
                        if target.exists():
                            continue
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        todo.discard(name); extracted += 1
        except zipfile.BadZipFile:
            print(f"bad_zip:{z.name}", file=sys.stderr)
    print(json.dumps({"wanted": len(wanted), "already_present": len(wanted & have), "extracted": extracted, "missing": len(todo), "zips_scanned": len(zips)}))
    if todo:
        needed_setups = sorted({n[:4] for n in todo})
        print("missing_setups:", ",".join(needed_setups))
        print("missing_examples:", ",".join(sorted(todo)[:5]))
    return 0 if not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
