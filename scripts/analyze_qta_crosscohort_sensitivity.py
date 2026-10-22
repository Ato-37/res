#!/usr/bin/env python3
"""Recompute descriptive QTA sensitivity across nine frozen local cohorts.

This is a local-only synthesis of existing private receipts. It reports no
row-level QA or claim data and performs no population inference or hypothesis
test. The endpoint remains an auxiliary answer-margin proxy, not faithfulness.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

COHORTS = (16, 17, 18, 20, 21, 22, 23, 24, 26)
ZERO_TOLERANCE = 1e-7


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("receipt_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite_number_required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite_number_required")
    return result


def _recompute_qta_delta(record: dict) -> float:
    primary = record.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("qta_primary_missing")
    margins = primary.get("margins_nats")
    required = {"baseline_control_ablation", "baseline_evidence_ablation",
                "qta_control_ablation", "qta_evidence_ablation"}
    if not isinstance(margins, dict) or set(margins) != required:
        raise ValueError("qta_margin_contract_invalid")
    base = (_finite(margins["baseline_control_ablation"])
            - _finite(margins["baseline_evidence_ablation"]))
    qta = (_finite(margins["qta_control_ablation"])
           - _finite(margins["qta_evidence_ablation"]))
    observed = _finite(primary.get("delta_f_qta_minus_baseline_nats"))
    if not math.isclose(qta - base, observed, rel_tol=0, abs_tol=1e-6):
        raise ValueError("qta_delta_recomputation_mismatch")
    return qta - base


def _recompute_sham_delta(record: dict) -> float:
    qta = _finite(record.get("delta_f_qta_nats"))
    sham = _finite(record.get("delta_f_sham_nats"))
    observed = _finite(record.get("qta_minus_sham_specificity_nats"))
    if not math.isclose(qta - sham, observed, rel_tol=0, abs_tol=1e-6):
        raise ValueError("sham_delta_recomputation_mismatch")
    return qta - sham


def _describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("values_required")
    return {
        "mean_nats": statistics.mean(values),
        "median_nats": statistics.median(values),
        "minimum_nats": min(values),
        "maximum_nats": max(values),
        "positive_cohorts": sum(x > ZERO_TOLERANCE for x in values),
        "negative_cohorts": sum(x < -ZERO_TOLERANCE for x in values),
        "near_zero_cohorts": sum(abs(x) <= ZERO_TOLERANCE for x in values),
        "near_zero_tolerance_nats": ZERO_TOLERANCE,
    }


def analyze(repo: Path) -> dict:
    qta_means, sham_means = [], []
    qta_hashes, sham_hashes, video_hashes = [], [], set()
    qta_rows_total = sham_rows_total = 0
    qta_forwards = sham_forwards = 0
    for cohort in COHORTS:
        qta_path = repo / "data/private_nextgqa" / f"qta_fixed_prediction_key_ablation_cohort{cohort}.json"
        sham_path = repo / "data/private_nextgqa" / f"qta_hash_matched_sham_ablation_cohort{cohort}.json"
        qta, qta_hash = _read(qta_path)
        sham, sham_hash = _read(sham_path)
        qta_rows, sham_rows = qta.get("records"), sham.get("records")
        if (qta.get("status") != "completed_exploratory_internal_key_ablation"
                or sham.get("status") != "completed_exploratory_overlap_matched_sham_ablation"
                or qta.get("effective_records") != 3 or len(qta_rows or []) != 3
                or sham.get("effective_records") != 3 or len(sham_rows or []) != 3
                or qta.get("completed_forwards") != 15
                or sham.get("completed_forwards") != 6
                or qta.get("direct_faithfulness") is not False
                or sham.get("direct_faithfulness") is not False
                or not isinstance(qta.get("human_review"), dict)
                or qta["human_review"].get("review_actually_performed") is not False
                or not isinstance(sham.get("human_review"), dict)
                or sham["human_review"].get("review_actually_performed") is not False):
            raise ValueError("cohort_receipt_incomplete")
        qta_by_token = {r.get("row_token_sha256"): r for r in qta_rows}
        sham_by_token = {r.get("row_token_sha256"): r for r in sham_rows}
        if (None in qta_by_token or None in sham_by_token
                or len(qta_by_token) != 3 or set(qta_by_token) != set(sham_by_token)):
            raise ValueError("paired_row_identity_mismatch")
        row_qta, row_sham = [], []
        for token, qrow in qta_by_token.items():
            srow = sham_by_token[token]
            video_hash = qrow.get("source_video_sha256")
            qta_primary = qrow.get("primary")
            if not isinstance(qta_primary, dict):
                raise ValueError("qta_primary_missing")
            if (not isinstance(video_hash, str) or len(video_hash) != 64
                    or video_hash != srow.get("source_video_sha256")
                    or qta_primary.get("fixed_prediction_index") != srow.get("fixed_prediction_index")
                    or qta_primary.get("fixed_foil_index") != srow.get("fixed_foil_index")):
                raise ValueError("paired_row_binding_mismatch")
            video_hashes.add(video_hash)
            row_qta.append(_recompute_qta_delta(qrow))
            row_sham.append(_recompute_sham_delta(srow))
        qta_means.append(statistics.mean(row_qta))
        sham_means.append(statistics.mean(row_sham))
        qta_hashes.append({"cohort": cohort, "sha256": qta_hash})
        sham_hashes.append({"cohort": cohort, "sha256": sham_hash})
        qta_rows_total += len(qta_rows)
        sham_rows_total += len(sham_rows)
        qta_forwards += qta["completed_forwards"]
        sham_forwards += sham["completed_forwards"]
    if len(video_hashes) != 27:
        raise ValueError("source_video_uniqueness_mismatch")
    return {
        "schema_version": 1,
        "purpose": "descriptive_cross_cohort_internal_key_sensitivity_synthesis",
        "cohort_ids": list(COHORTS),
        "requested_effective_cohorts": [len(COHORTS), len(COHORTS)],
        "qta_rows": qta_rows_total,
        "matched_sham_rows": sham_rows_total,
        "unique_video_digests": len(video_hashes),
        "qta_completed_forwards": qta_forwards,
        "sham_completed_forwards": sham_forwards,
        "qta_minus_baseline_cohort_mean": _describe(qta_means),
        "qta_minus_sham_cohort_mean": _describe(sham_means),
        "qta_receipt_sha256": qta_hashes,
        "sham_receipt_sha256": sham_hashes,
        "inference": "descriptive only; post-hoc selected finite cohorts; no test, CI, multiplicity adjustment, or population inference",
        "construct_scope": "fixed-prediction internal key-ablation answer-margin proxy; not claim truth, reviewed evidence support, or direct faithfulness",
        "human_review_performed": False,
        "rights_status": "unverified; local-only",
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(analyze(repo), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
