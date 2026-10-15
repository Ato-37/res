"""Synthetic-only orchestration seam for testing claim-mask launch ordering.

This is not a production runner. It loads no media/model and cannot authorize
real scoring. The explicit synthetic fixture envelope and validator callback
exist only to test whole-cohort gate ordering. The fixed spy blocks accidental
instance-level callback replacement; Python monkeypatching remains possible,
so this module is not a security boundary.
"""
from __future__ import annotations

from copy import deepcopy

from preflight_claim_mask_timing import preflight_cohort


FIXTURE_TYPE = "synthetic_claim_mask_timing_fixture_v1"
_VALIDATOR_REPORT_KEYS = {"status", "failure_codes", "execution_authorization"}


class SyntheticScorerSpy:
    """A call-count probe; deliberately has no model or score implementation."""

    __slots__ = ("fail_on_call", "calls", "video_id_batches")

    def __init__(self, *, fail_on_call: bool = False):
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.video_id_batches: list[list[str]] = []

    def record_batch_call(self, records: list[dict]) -> None:
        self.calls += 1
        self.video_id_batches.append([record.get("video_id") for record in records])
        if self.fail_on_call:
            raise RuntimeError("synthetic spy failure")


def _blocked(code: str, *, timing_report=None) -> dict:
    result = {
        "status": "blocked",
        "failure_codes": [code],
        "denominator": 0,
        "score_calls": 0,
        "execution_authorization": "not_granted",
    }
    if timing_report is not None:
        result["timing_preflight"] = timing_report
    return result


def run_synthetic_gate_fixture_v1(
    fixture: dict,
    *,
    expected_video_ids: list[str],
    expected_num_frames: int,
    tolerance_seconds,
    structural_validator,
    scorer,
) -> dict:
    """Exercise whole-cohort-before-score ordering on an explicitly synthetic fixture.

    ``structural_validator`` must return exactly the small report schema tested
    below. ``scorer`` must be the fixed ``SyntheticScorerSpy`` and receives only
    a whole-cohort call-count probe, never model inputs for inference. The
    fixture marker is a scope guard, not authentication or evidence that records
    are synthetic.
    """
    if (not isinstance(fixture, dict)
            or set(fixture) != {"fixture_type", "records"}
            or fixture.get("fixture_type") != FIXTURE_TYPE
            or not isinstance(fixture.get("records"), list)):
        return _blocked("synthetic_fixture_schema_invalid")
    if not callable(structural_validator):
        return _blocked("structural_validator_missing")
    if type(scorer) is not SyntheticScorerSpy:
        return _blocked("synthetic_scorer_spy_required")
    try:
        records = deepcopy(fixture["records"])
        frozen_ids = deepcopy(expected_video_ids)
    except Exception:
        return _blocked("fixture_snapshot_failed")

    timing = preflight_cohort(
        records,
        expected_video_ids=frozen_ids,
        expected_num_frames=expected_num_frames,
        tolerance_seconds=tolerance_seconds,
    )
    if (timing.get("status") != "synthetic_timing_contract_passed"
            or timing.get("failure_codes") != []
            or timing.get("denominator") != len(frozen_ids)
            or timing.get("validated_videos") != len(frozen_ids)
            or timing.get("execution_authorization") != "not_granted"):
        return _blocked("timing_preflight_failed", timing_report=timing)

    try:
        structural = structural_validator(deepcopy(records))
    except Exception:
        return _blocked("structural_validator_failed", timing_report=timing)
    if (not isinstance(structural, dict)
            or set(structural) != _VALIDATOR_REPORT_KEYS
            or structural.get("status") != "synthetic_structure_passed"
            or structural.get("failure_codes") != []
            or structural.get("execution_authorization") != "not_granted"):
        return _blocked("structural_preflight_failed", timing_report=timing)

    by_id = {record["video_id"]: record for record in records}
    ordered = [deepcopy(by_id[video_id]) for video_id in frozen_ids]
    try:
        scorer.record_batch_call(ordered)
    except Exception:
        return {
            "status": "synthetic_scorer_failed",
            "failure_codes": ["scorer_callback_failed"],
            "denominator": len(frozen_ids),
            "score_calls": 1,
            "execution_authorization": "not_granted",
        }
    return {
        "status": "synthetic_scorer_callback_completed",
        "failure_codes": [],
        "denominator": len(frozen_ids),
        "validated_videos": len(frozen_ids),
        "score_calls": 1,
        "execution_authorization": "not_granted",
    }
