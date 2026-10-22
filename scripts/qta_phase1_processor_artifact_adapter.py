"""Attach an actual processor candidate to the pre-baseline-v2 artifact roles.

This composes bytes only. It does not sign a receipt, validate human evidence,
authenticate independent pins, or authorize model loading/scoring.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from qta_actual_processor_row_candidate import (
    ROW_COUNT, verify_actual_processor_row_candidate_self_consistency,
)
from qta_phase_freeze_receipts import PRE_BASELINE_V2_ARTIFACT_ROLES
from qta_processor_batch_digest_manifest_v1 import build_manifest

_GENERATED_ROLES = {"prompt_answer_token_bindings", "processor_batch_digest_manifest"}
_BASE_ROLES = set(PRE_BASELINE_V2_ARTIFACT_ROLES) - _GENERATED_ROLES


def build_qta_phase1_processor_artifacts(*, candidate: Any,
        base_artifact_bytes: Any, protocol_raw: bytes) -> dict[str, bytes]:
    """Compose a complete phase-1-v2 artifact mapping from verified CPU rows.

    ``base_artifact_bytes`` must supply every non-processor phase-1 role,
    including the caller's already-frozen source/review/runtime artifacts.
    The returned mapping remains a non-authorizing candidate for the caller's
    existing freeze-owner signing workflow.
    """
    verify_actual_processor_row_candidate_self_consistency(candidate)
    if (not isinstance(base_artifact_bytes, Mapping)
            or set(base_artifact_bytes) != _BASE_ROLES
            or not isinstance(protocol_raw, bytes) or not protocol_raw):
        raise ValueError("phase1_processor_artifact_inputs_invalid")
    base = {}
    for role in _BASE_ROLES:
        raw = base_artifact_bytes[role]
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("phase1_base_artifact_bytes_invalid")
        base[role] = raw

    rows = candidate["candidate_rows"]
    runtime_sha = hashlib.sha256(base["sampler_model_runtime"]).hexdigest()
    manifest_rows = []
    live_batches = []
    for row in rows:
        batch = row["batch"]
        live_batches.append(batch)
        manifest_rows.append({
            "record_token": hashlib.sha256(row["review_token"].encode("ascii")).hexdigest(),
            "base_processor_batch_sha256": row["typed_batch_sha256"],
            "source_video_sha256": row["source_video_sha256"],
            "prompt_sha256": row["rendered_prompt_sha256"],
            "input_ids_sha256": row["input_ids_sha256"],
            "processor_sha256": row["processor_fingerprint_sha256"],
            "tokenizer_sha256": row["tokenizer_sha256"],
            "runtime_sha256": runtime_sha,
            "sampler_metadata_sha256": row["sampler_row_sha256"],
            "video_grid_thw": row["video_grid_thw"],
        })
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    manifest_raw = build_manifest(
        rows=manifest_rows, live_batches=live_batches,
        expected_row_pins=manifest_rows, protocol_sha256=protocol_sha,
        expected_protocol_sha256=protocol_sha)
    artifacts = {
        **base,
        "prompt_answer_token_bindings": candidate["prompt_answer_token_bindings_raw"],
        "processor_batch_digest_manifest": manifest_raw,
    }
    if set(artifacts) != set(PRE_BASELINE_V2_ARTIFACT_ROLES):
        raise ValueError("phase1_v2_artifact_role_set_invalid")
    return artifacts


def audit_qta_phase1_processor_artifacts(artifacts: Any) -> dict[str, Any]:
    """Return counts and denial flags; this audit is not a receipt verifier."""
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != set(PRE_BASELINE_V2_ARTIFACT_ROLES)
            or any(not isinstance(raw, bytes) or not raw for raw in artifacts.values())):
        raise ValueError("phase1_v2_artifacts_invalid")
    return {
        "status": "complete_phase1_v2_artifact_mapping_candidate_only",
        "artifact_role_count": len(artifacts),
        "processor_rows": ROW_COUNT,
        "signature_created": False,
        "human_review_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }
