"""Cross-check fixed synthetic attention backend/precision controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {
    "bf16": ("bfloat16", {"manual_eager", "sdpa"}),
    "bf16_fp32acc": ("bfloat16", {"manual_eager", "manual_fp32_accum", "sdpa"}),
    "fp32": ("float32", {"manual_eager", "sdpa"}),
}


def _load(path: Path, label: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or report.get("failure_count") != 0:
        raise ValueError(f"{label}: incomplete report")
    if report.get("seed") != 20261012 or report.get("lengths") != [584, 585, 586]:
        raise ValueError(f"{label}: fixed generator mismatch")
    if report.get("heads") != 8 or report.get("head_dim") != 64:
        raise ValueError(f"{label}: fixed shape mismatch")
    dtype, backends = EXPECTED[label]
    if report.get("dtype") != dtype:
        raise ValueError(f"{label}: dtype mismatch")
    records = report.get("records")
    if not isinstance(records, list) or {r.get("backend") for r in records} != backends:
        raise ValueError(f"{label}: backend set mismatch")
    for row in records:
        if (row.get("left_length"), row.get("right_length")) not in {(584, 585), (584, 586), (585, 586)}:
            raise ValueError(f"{label}: comparison shape mismatch")
        for key in ("raw_max_abs", "raw_mean_abs"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"{label}: invalid {key}")
        if row.get("finite") is not True:
            raise ValueError(f"{label}: nonfinite output flag")
    return report


def analyze(bf16_path: Path, bf16_fp32acc_path: Path, fp32_path: Path) -> dict:
    reports = {
        "bf16": _load(bf16_path, "bf16"),
        "bf16_fp32acc": _load(bf16_fp32acc_path, "bf16_fp32acc"),
        "fp32": _load(fp32_path, "fp32"),
    }
    keys = {(584, 585), (584, 586), (585, 586)}
    by_condition = {}
    for label, report in reports.items():
        rows = {(r["backend"], r["left_length"], r["right_length"]): r for r in report["records"]}
        if len(rows) != len(report["records"]):
            raise ValueError(f"{label}: duplicate record")
        by_condition[label] = rows
    summary = []
    for pair in sorted(keys):
        eager = by_condition["bf16"][("manual_eager", *pair)]["raw_max_abs"]
        accum = by_condition["bf16_fp32acc"][("manual_fp32_accum", *pair)]["raw_max_abs"]
        fp32 = by_condition["fp32"][("manual_eager", *pair)]["raw_max_abs"]
        sdpa = by_condition["bf16"][("sdpa", *pair)]["raw_max_abs"]
        accum_sdpa = by_condition["bf16_fp32acc"][("sdpa", *pair)]["raw_max_abs"]
        fp32_sdpa = by_condition["fp32"][("sdpa", *pair)]["raw_max_abs"]
        summary.append({"left_length": pair[0], "right_length": pair[1],
                        "bf16_manual_eager_raw_max_abs": eager,
                        "bf16_fp32_accum_raw_max_abs": accum,
                        "bf16_sdpa_raw_max_abs": sdpa,
                        "fp32_manual_eager_raw_max_abs": fp32,
                        "fp32_sdpa_raw_max_abs": fp32_sdpa,
                        "fp32_accum_equals_sdpa": accum == accum_sdpa,
                        "fp32_manual_equals_sdpa": fp32 == fp32_sdpa})
    return {
        "analysis": "synthetic_attention_backend_precision_control_crosscheck",
        "input_reports": {label: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for label, path in (("bf16", bf16_path), ("bf16_fp32acc", bf16_fp32acc_path), ("fp32", fp32_path))},
        "record_count": len(summary),
        "comparisons": summary,
        "all_bf16_accumulation_zero": all(x["bf16_fp32_accum_raw_max_abs"] == 0.0 for x in summary),
        "all_bf16_eager_nonzero": all(x["bf16_manual_eager_raw_max_abs"] > 0.0 for x in summary),
        "all_fp32_zero": all(x["fp32_manual_eager_raw_max_abs"] == 0.0 and x["fp32_sdpa_raw_max_abs"] == 0.0 for x in summary),
        "interpretation_limit": "fixed synthetic Q/K/V only; supports an intermediate-precision/backend mechanism hypothesis in this runtime, without identifying an attention sub-operation and with no Qwen, mask, semantic, or faithfulness claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16", type=Path)
    parser.add_argument("bf16_fp32acc", type=Path)
    parser.add_argument("fp32", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.bf16, args.bf16_fp32acc, args.fp32)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(text)
        args.output.chmod(0o600)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
