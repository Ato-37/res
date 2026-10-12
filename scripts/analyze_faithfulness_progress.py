"""Build a privacy-safe faithfulness progress snapshot from published artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load(path: Path) -> tuple[dict, str]:
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(head_path: Path, black_path: Path, blur_path: Path, noaa_path: Path,
            head_smoke_path: Path | None = None, mask_smoke_path: Path | None = None,
            mask_blur_smoke_path: Path | None = None,
            cap_black_smoke_path: Path | None = None, cap_blur_smoke_path: Path | None = None) -> dict:
    head, head_sha = _load(head_path)
    black, black_sha = _load(black_path)
    blur, blur_sha = _load(blur_path)
    noaa, noaa_sha = _load(noaa_path)

    if (head.get("n_requested"), head.get("n_effective"), head.get("n_failures")) != (79, 79, 0):
        raise ValueError("head confirmation accounting mismatch")
    primary = head.get("primary_summary", {})
    for key in ("mean", "sd", "ci95_t"):
        if key not in primary:
            raise ValueError("head primary summary incomplete")
    if len(primary["ci95_t"]) != 2:
        raise ValueError("head confidence interval malformed")
    for report, name in ((black, "black"), (blur, "blur")):
        if (report.get("n_requested"), report.get("n_effective"), len(report.get("failures", []))) != (60, 60, 0):
            raise ValueError(f"{name} mask accounting mismatch")
        if "summary" not in report or "all" not in report["summary"]:
            raise ValueError(f"{name} mask summary incomplete")
    if noaa.get("decision") != "no_go" or noaa.get("attestation_status") != "missing":
        raise ValueError("NOAA gate status changed unexpectedly")
    if "human_attestation_missing" not in noaa.get("failure_reasons", []):
        raise ValueError("NOAA missing-attestation reason absent")

    result = {
        "schema_version": 1,
        "analysis": "faithfulness_progress_snapshot",
        "direct_faithfulness": {
            "status": "直接評価済み",
            "artifact": str(head_path),
            "artifact_sha256": head_sha,
            "model": head.get("model"),
            "dataset_scope": "NExT-GQA independent confirmation cohort",
            "intervention": "top-4 gradient×activation heads attenuated at alpha=0.5",
            "metric": head.get("primary_estimand"),
            "claim_scope": "head-selective frozen-answer-margin dependence only; not generated-rationale or claim-semantic faithfulness",
            "n_requested": head["n_requested"],
            "n_effective": head["n_effective"],
            "n_failures": head["n_failures"],
            "estimate_nat": primary["mean"],
            "ci95_t_nat": primary["ci95_t"],
            "practical_threshold_nat": head.get("practical_threshold_nat"),
            "threshold_supported": primary["ci95_t"][0] > head.get("practical_threshold_nat", float("inf")),
        },
        "supporting_diagnostics": {
            "status": "補助診断のみ",
            "grounding_mask_black": {"artifact_sha256": black_sha, "n_effective": black["n_effective"], "all_mean_nat": black["summary"]["all"]["mean"], "ci95_pointwise_nat": black["summary"]["all"]["ci95_pointwise"]},
            "grounding_mask_blur": {"artifact_sha256": blur_sha, "n_effective": blur["n_effective"], "all_mean_nat": blur["summary"]["all"]["mean"], "ci95_pointwise_nat": blur["summary"]["all"]["ci95_pointwise"]},
            "interpretation": "mask sensitivity with imperfect content/control matching; not a direct explanation-faithfulness effect",
        },
        "unresolved_gates": {
            "noaa_human_attestation": {"status": "no_go", "artifact_sha256": noaa_sha, "items": noaa.get("ai_blind_review_items"), "reason": "human_attestation_missing"},
            "new_faithfulness_confirmation": "未実施",
        },
        "next_prespecified_action": "Do not reopen the closed head/layer/alpha search; if research continues, preregister a new intervention, matched control, and independent cohort before any GPU run.",
        "limitations": ["This snapshot summarizes existing artifacts and is not a new faithfulness experiment.", "It does not generalize beyond the tested model, cohort, intervention, and metrics."],
    }
    smokes = {}
    if head_smoke_path is not None:
        smoke, sha = _load(head_smoke_path)
        if (smoke.get("n_requested"), smoke.get("n_effective"), smoke.get("n_failures")) != (3, 3, 0):
            raise ValueError("head smoke accounting mismatch")
        smokes["head_intervention"] = {"status": "補助診断のみ", "artifact_sha256": sha, "n_effective": 3, "mean_selective_nat": smoke.get("primary_summary", {}).get("mean"), "interpretation": "engineering reproducibility smoke; not confirmatory"}
    if mask_smoke_path is not None:
        smoke, sha = _load(mask_smoke_path)
        if (smoke.get("n_requested"), smoke.get("n_effective"), smoke.get("n_failures")) != (3, 3, 0):
            raise ValueError("mask smoke accounting mismatch")
        smokes["grounding_mask"] = {"status": "補助診断のみ", "artifact_sha256": sha, "n_effective": 3, "all_mean_nat": smoke.get("all_mean_nat"), "interpretation": "engineering smoke; not direct explanation faithfulness"}
    if mask_blur_smoke_path is not None:
        smoke, sha = _load(mask_blur_smoke_path)
        if smoke.get("n_effective_blur") != 3 or smoke.get("n_effective_black") != 3:
            raise ValueError("paired mask smoke accounting mismatch")
        smokes["grounding_mask_black_blur_pair"] = {"status": "補助診断のみ", "artifact_sha256": sha, "n_effective_each": 3, "paired_blur_minus_black_primary_mean_nat": smoke.get("paired_blur_minus_black_primary_mean_nat"), "interpretation": "mask-fill engineering/OOD sensitivity; not direct explanation faithfulness"}
    for name, path in (("cap_black", cap_black_smoke_path), ("cap_blur", cap_blur_smoke_path)):
        if path is not None:
            smoke, sha = _load(path)
            if smoke.get("n_effective") != 3 or smoke.get("n_failures") != 0 or smoke.get("cap_pixels_per_frame") is not True:
                raise ValueError(f"{name} smoke accounting/config mismatch")
            smokes[name] = {"status": "補助診断のみ", "artifact_sha256": sha, "n_effective": 3, "cap_pixels_per_frame": True, "primary_mean_nat": smoke.get("primary_mean_nat"), "interpretation": "preprocessing reproducibility smoke; not direct explanation faithfulness"}
    if smokes:
        result["engineering_smokes"] = smokes
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--black", type=Path, required=True)
    p.add_argument("--blur", type=Path, required=True)
    p.add_argument("--noaa", type=Path, required=True)
    p.add_argument("--head-smoke", type=Path)
    p.add_argument("--mask-smoke", type=Path)
    p.add_argument("--mask-blur-smoke", type=Path)
    p.add_argument("--cap-black-smoke", type=Path)
    p.add_argument("--cap-blur-smoke", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    out = analyze(a.head, a.black, a.blur, a.noaa, a.head_smoke, a.mask_smoke, a.mask_blur_smoke, a.cap_black_smoke, a.cap_blur_smoke)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    a.output.chmod(0o600)
    print(json.dumps({"status": "complete", "output": str(a.output), "direct_status": out["direct_faithfulness"]["status"]}))


if __name__ == "__main__":
    main()
