"""Build a privacy-safe faithfulness progress snapshot from published artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_HEAD_ARTIFACT_SHA256 = "993acd16e62c1e8cf8aafe075674cc1e649761f2e1ca5bb05be83814a2e6ac36"
HEAD_PRACTICAL_THRESHOLD_NAT = 0.25


def _load(path: Path) -> tuple[dict, str]:
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(head_path: Path, black_path: Path, blur_path: Path, noaa_path: Path,
            head_smoke_path: Path | None = None, mask_smoke_path: Path | None = None,
            mask_blur_smoke_path: Path | None = None,
            cap_black_smoke_path: Path | None = None, cap_blur_smoke_path: Path | None = None,
            claim_reviewed_count: int = 0, claim_scored_count: int = 0) -> dict:
    if (type(claim_reviewed_count) is not int or not 0 <= claim_reviewed_count <= 3
            or type(claim_scored_count) is not int or not 0 <= claim_scored_count <= 3
            or claim_scored_count > 0 and claim_reviewed_count != 3):
        raise ValueError("generated_claim_progress_counts_invalid")
    head, head_sha = _load(head_path)
    black, black_sha = _load(black_path)
    blur, blur_sha = _load(blur_path)
    noaa, noaa_sha = _load(noaa_path)

    if (head.get("n_requested"), head.get("n_effective"), head.get("n_failures")) != (79, 79, 0):
        raise ValueError("head confirmation accounting mismatch")
    if head_sha != EXPECTED_HEAD_ARTIFACT_SHA256:
        raise ValueError("head confirmation artifact identity mismatch")
    primary = head.get("primary_summary", {})
    for key in ("mean", "sd", "ci95_t"):
        if key not in primary:
            raise ValueError("head primary summary incomplete")
    if len(primary["ci95_t"]) != 2:
        raise ValueError("head confidence interval malformed")
    threshold = head.get("practical_threshold_nat")
    if (type(threshold) not in (int, float) or threshold != HEAD_PRACTICAL_THRESHOLD_NAT
            or primary["ci95_t"][1] >= threshold):
        raise ValueError("head confirmation threshold decision mismatch")
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
        "schema_version": 2,
        "analysis": "faithfulness_progress_snapshot",
        "historical_head_confirmation": {
            "status": "closed_practical_threshold_not_met",
            "search_status": "closed_no_head_layer_alpha_reopening",
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
        "generated_claim_faithfulness": {
            "status": "not_evaluated_no_verified_direct_score_artifact",
            "count_source": "operator_declared_unverified",
            "human_reviewed_records_declared": claim_reviewed_count,
            "requested_human_review_records": 3,
            "scored_records_declared": claim_scored_count,
            "requested_scored_records": 3,
            "verified_score_artifact_supplied": False,
            "estimate_nat": None,
            "confidence_interval_nat": None,
            "p_value": None,
            "pending_gates": (["independent_human_review_of_all_three_claims"]
                              if claim_reviewed_count < 3 else []) + [
                "15_os_browser_to_decoder_frame_correspondences_and_clock_audit",
                "pre_outcome_joint_protocol_and_input_freeze",
                "production_all_record_atomic_preflight",
                "verified_direct_score_artifact",
            ],
        },
        "supporting_diagnostics": {
            "status": "補助診断のみ",
            "grounding_mask_black": {"artifact_sha256": black_sha, "n_effective": black["n_effective"], "all_mean_nat": black["summary"]["all"]["mean"], "ci95_pointwise_nat": black["summary"]["all"]["ci95_pointwise"]},
            "grounding_mask_blur": {"artifact_sha256": blur_sha, "n_effective": blur["n_effective"], "all_mean_nat": blur["summary"]["all"]["mean"], "ci95_pointwise_nat": blur["summary"]["all"]["ci95_pointwise"]},
            "interpretation": "mask sensitivity with imperfect content/control matching; not a direct explanation-faithfulness effect",
        },
        "unresolved_gates": {
            "noaa_human_attestation": {"status": "no_go", "artifact_sha256": noaa_sha, "items": noaa.get("ai_blind_review_items"), "reason": "human_attestation_missing"},
            "verified_generated_claim_score": False,
        },
        "next_prespecified_action": (
            "Complete the independent three-claim review and 15-anchor browser-to-decoder audit, then freeze all protocol inputs and build the production atomic preflight; keep the closed head/layer/alpha search closed."
            if claim_reviewed_count < 3 else
            "Complete the 15-anchor browser-to-decoder audit, freeze all protocol inputs, and build the production atomic preflight; keep the closed head/layer/alpha search closed."
        ),
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
    p.add_argument("--claim-reviewed-count", type=int, choices=range(4), required=True,
                   help="operator-declared current reviewed count; not authenticated")
    p.add_argument("--claim-scored-count", type=int, choices=range(4), required=True,
                   help="operator-declared current scored count; not authenticated")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    out = analyze(a.head, a.black, a.blur, a.noaa, a.head_smoke, a.mask_smoke,
                  a.mask_blur_smoke, a.cap_black_smoke, a.cap_blur_smoke,
                  a.claim_reviewed_count, a.claim_scored_count)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    a.output.chmod(0o600)
    print(json.dumps({"status": "complete", "output": str(a.output),
                      "head_status": out["historical_head_confirmation"]["status"],
                      "claim_status": out["generated_claim_faithfulness"]["status"]}))


if __name__ == "__main__":
    main()
