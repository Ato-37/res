#!/usr/bin/env python3
"""Verify fresh-cohort Minerva VISTA-style GPU receipts and write a privacy-safe summary.

Reads mode-0600 local receipts, rechecks the CPU-receipt/cohort/source-hash bindings and the
contrast arithmetic, and writes row-order-free derived numbers only (no IDs, text, row hashes,
paths). Descriptive and exploratory; not a faithfulness measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
TOL = 1e-6


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_private(path: Path) -> tuple[bytes, dict]:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("private_input_invalid")
    if path.stat().st_mode & 0o077:
        raise ValueError("private_input_mode_invalid")
    raw = path.read_bytes()
    return raw, json.loads(raw)


def verify_row(gpu_raw: bytes, gpu: dict, cpu_raw: bytes, cpu: dict, cohort_raw: bytes,
               preflight_raw: bytes, source_check: bool) -> dict:
    if cpu.get("status") != "completed_vista_residual_cpu_preflight" or cpu.get("generation_mode") != "fresh_in_run_greedy":
        raise ValueError("cpu_receipt_not_fresh_mode")
    for key, raw in (("cohort_sha256", cohort_raw), ("cohort_cpu_preflight_sha256", preflight_raw),
                     ("cpu_receipt_sha256", cpu_raw)):
        if gpu.get(key) != sha(raw):
            raise ValueError(f"binding_mismatch:{key}")
    for key in ("row_index", "row_key_sha256", "video_sha256", "time_map_sha256", "model_revision",
                "source_sha256", "runner_sha256", "dose_rms_fraction", "fixed_block_index_zero_based"):
        if gpu.get(key) != cpu.get(key):
            raise ValueError(f"gpu_cpu_mismatch:{key}")
    if source_check:
        for name, expected in gpu["source_sha256"].items():
            if sha((ROOT / "scripts" / name).read_bytes()) != expected:
                raise ValueError(f"source_drift:{name}")
    m = gpu["mean_object_name_logprob_nats_per_token"]
    base, anc, sham = (m["baseline"], m["anchor_steering"], m["temporal_sham_steering"])
    da, ds = anc - base, sham - base
    for got, want, label in ((gpu["delta_anchor_minus_baseline_nats_per_token"], da, "da"),
                             (gpu["delta_sham_minus_baseline_nats_per_token"], ds, "ds"),
                             (gpu["anchor_minus_sham_change_nats_per_token"], da - ds, "diff")):
        if not math.isclose(got, want, abs_tol=TOL):
            raise ValueError(f"arithmetic_mismatch:{label}")
    for hook in (gpu["capture_hook_audit"].values()):
        if hook["hook_call_count"] != 1 or hook["hook_registry_before"] != hook["hook_registry_after_removal"]:
            raise ValueError("hook_audit_failed")
    for hook in gpu["score_hook_audit"].values():
        if hook["hook_call_count"] != 1 or hook["hook_registry_before"] != hook["hook_registry_after_removal"]:
            raise ValueError("hook_audit_failed")
    eff = {k: v["effective_displacement_fraction_of_pre_rms"] for k, v in gpu["score_hook_audit"].items()}
    return {"object_name_token_count": gpu["object_name_token_count"],
            "target_token_count": gpu["target_token_count"],
            "baseline": base, "delta_anchor": da, "delta_sham": ds, "anchor_minus_sham": da - ds,
            "effective_dose_fraction": eff, "peak_vram_gib": gpu["peak_vram_bytes"] / 2**30}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--cohort-preflight", type=Path, required=True)
    ap.add_argument("--gpu", type=Path, nargs="+", required=True, help="one GPU receipt per requested row")
    ap.add_argument("--cpu", type=Path, nargs="+", required=True, help="matching CPU receipts, same order")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-source-hash-check", action="store_true")
    args = ap.parse_args(argv)
    if len(args.gpu) != len(args.cpu) or args.output.exists() or args.output.parent.resolve() != (ROOT / "results").resolve():
        raise ValueError("inputs_or_output_invalid")
    cohort_raw, cohort = load_private(args.cohort)
    preflight_raw, _ = load_private(args.cohort_preflight)
    rows, failures, cpus = [], [], []
    for g, c in zip(args.gpu, args.cpu):
        graw, gpu = load_private(g)
        craw, cpu = load_private(c)
        cpus.append(cpu)
        if gpu.get("status", "").startswith("failed_"):
            for key, raw in (("cohort_sha256", cohort_raw), ("cohort_cpu_preflight_sha256", preflight_raw),
                             ("cpu_receipt_sha256", craw)):
                if gpu.get(key) != sha(raw):
                    raise ValueError(f"failed_row_binding_mismatch:{key}")
            failures.append({"failure_code": gpu.get("failure_code"),
                             "completed_stages": [k for k, v in gpu["completed_stages"].items() if v]})
            continue
        rows.append(verify_row(graw, gpu, craw, cpu, cohort_raw, preflight_raw, not args.skip_source_hash_check))
    diffs = [r["anchor_minus_sham"] for r in rows]
    summary = {"schema_version": 1, "status": "verified_minerva_vista_fresh_cohort_exploratory",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {"requested_rows": len(args.gpu), "effective_rows": len(rows), "failed_rows": len(failures)},
        "failures": failures, "per_row": rows,
        "aggregate": {"mean_anchor_minus_sham_nats_per_token": statistics.fmean(diffs) if diffs else None,
            "n": len(diffs), "sign_counts": {"positive": sum(d > 0 for d in diffs), "negative": sum(d < 0 for d in diffs)},
            "inference": "none; n too small for a CI or test"},
        "experiment": {"dataset": "Minerva-Ego / HD-EPIC", "model_revision": cpus[0]["model_revision"],
            "precision": "4-bit weights; BF16 compute", "frames": 64, "block_1_based": cpus[0]["fixed_block_1_based"],
            "dose_rms_fraction": cpus[0]["dose_rms_fraction"], "selection_requested_rows": len(cohort["selection"])},
        "limitations": ["exploratory, tiny n", "object-name likelihood is not explanation faithfulness",
            "no human support labels",
            "effective rows are conditioned on the model's own sentence naming the metadata object (selection effect)",
            "each effective row scores a single object-name token", "30 fps proxy anchors; mean-fill OOD; farthest-endpoint sham"],
        "faithfulness_status": "補助診断のみ"}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["counts"]), json.dumps(summary["aggregate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
