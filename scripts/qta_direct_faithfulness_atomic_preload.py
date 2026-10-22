"""Atomic all-record composition of the non-authorizing QTA pre-load checks.

This CPU-only wrapper binds the current candidate/sampler/clock bridge to the
signed phase-1/live-batch transaction for the same ordered three-row cohort.
Even a structurally consistent result is not execution authorization and must
never be used to load a model or score a faithfulness outcome.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from qta_actual_candidate_input_bridge import build_qta_actual_candidate_input_material
from qta_direct_faithfulness_preload_gate import evaluate_qta_direct_faithfulness_preload
from qta_prebaseline_transaction_preflight import preflight_qta_phase1_batch_transaction
from qta_stage1_source_pin_adapter import derive_stage1_source_pins
from qta_processor_batch_digest_manifest_v1 import validate_manifest
from qta_prompt_answer_token_bindings_v1 import validate_qta_prompt_answer_token_bindings

ROW_COUNT = 3
ANCHOR_COUNT = 15

PRELOAD_REQUIRED = frozenset({
    "sampler_receipt_raw", "candidate_inventory_raw", "media_sha256_by_role",
    "actual_source_inputs", "stage1_audit_inputs", "actual_qa_source_inputs",
    "clock_inputs_by_token", "processor_pins_by_token", "tokenizer_sha256",
    "video_duration_seconds_by_token",
})
PRELOAD_OPTIONAL = frozenset({"clock_frame_audit_inputs", "repo_root", "code_files"})
TRANSACTION_REQUIRED = frozenset({
    "actual_source_inputs", "review_bundle_inputs", "stage1_audit_inputs",
    "actual_qa_source_inputs", "phase1_receipt_raw", "phase1_artifact_bytes",
    "integrity_roster_raw", "release_roster_raw", "root_public_key_pem",
    "expected_root_public_der_sha256", "expected_external_cohort_pin_sha256",
    "live_processor_rows",
})
TRANSACTION_OPTIONAL = frozenset({
    "expected_processor_batch_row_pins",
    "expected_processor_batch_protocol_sha256",
})


def _blocked(code: str = "atomic_preload_blocked") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "qta_direct_faithfulness_atomic_preload_structure_only",
        "status": "blocked",
        "requested_rows": ROW_COUNT,
        "matched_rows": 0,
        "requested_clock_frame_anchors": ANCHOR_COUNT,
        "signature_bound_frame_pair_decisions": 0,
        "validated_clock_frame_anchors": 0,
        "cohort_binding": "blocked",
        "candidate_preload": "blocked",
        "phase1_batch_transaction": "blocked",
        "external_trust_root_origin_authenticated": False,
        "reviewer_identity_authenticated": False,
        "semantic_review_authenticated": False,
        "clock_frame_correspondence_authenticated": False,
        "direct_faithfulness_ready": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "forward_calls": 0,
        "score_calls": 0,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
        "failure_code": code,
    }


def _copy_plain(value: Any, *, depth: int = 0) -> Any:
    """Snapshot bytes and ordinary containers, rejecting live/custom objects."""
    if depth > 16:
        raise ValueError("input_nesting_too_deep")
    if type(value) in (bytes, str, int, bool) or value is None:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("input_nonfinite_number")
        return value
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("input_mapping_key_invalid")
        return {key: _copy_plain(child, depth=depth + 1)
                for key, child in value.items()}
    if type(value) is list:
        return [_copy_plain(child, depth=depth + 1) for child in value]
    if type(value) is tuple:
        return tuple(_copy_plain(child, depth=depth + 1) for child in value)
    if isinstance(value, Path):
        return Path(value)
    raise ValueError("input_value_type_invalid")


def _copy_bundle(value: Any, required: frozenset[str], optional=frozenset()) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("input_bundle_invalid")
    keys = set(value)
    if keys - required - optional or required - keys:
        raise ValueError("input_bundle_schema_invalid")
    # One read from each caller mapping; downstream sees only this private copy.
    return {key: _copy_plain(value[key]) for key in keys}


def _snapshot_live_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or len(value) != ROW_COUNT:
        raise ValueError("live_batch_inventory_invalid")
    result = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
                "review_token", "source_video_sha256", "batch_sha256", "batch"}:
            raise ValueError("live_batch_row_schema_invalid")
        # Read each dynamic/custom mapping field once. Keep the immutable batch
        # reference so the leaf transaction audits the actual tensors.
        token = row["review_token"]
        source = row["source_video_sha256"]
        digest = row["batch_sha256"]
        batch = row["batch"]
        if type(batch) is not FrozenBatch:
            raise ValueError("live_frozen_batch_required")
        if type(token) is not str or type(source) is not str or type(digest) is not str:
            raise ValueError("live_batch_row_value_invalid")
        result.append({"review_token": token, "source_video_sha256": source,
                       "batch_sha256": digest, "batch": batch})
    if len({row["review_token"] for row in result}) != ROW_COUNT:
        raise ValueError("live_batch_duplicate_token")
    if len({row["source_video_sha256"] for row in result}) != ROW_COUNT:
        raise ValueError("live_batch_duplicate_source")
    if len({id(row["batch"]) for row in result}) != ROW_COUNT:
        raise ValueError("live_batch_aliased_rows")
    return result


def _ordered_cohort(preload: dict, live_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rebuild private candidate pins and compare their ordered cohort to phase 1."""
    material = build_qta_actual_candidate_input_material(
        sampler_receipt_raw=preload["sampler_receipt_raw"],
        candidate_inventory_raw=preload["candidate_inventory_raw"],
        media_sha256_by_role=preload["media_sha256_by_role"],
        actual_source_inputs=preload["actual_source_inputs"],
        stage1_audit_inputs=preload["stage1_audit_inputs"],
        actual_qa_source_inputs=preload["actual_qa_source_inputs"],
        clock_inputs_by_token=preload["clock_inputs_by_token"],
        processor_pins_by_token=preload["processor_pins_by_token"],
        tokenizer_sha256=preload["tokenizer_sha256"],
        video_duration_seconds_by_token=preload["video_duration_seconds_by_token"],
        **({} if "repo_root" not in preload else {"repo_root": preload["repo_root"]}),
        **({} if "code_files" not in preload else {"code_files": preload["code_files"]}),
    )
    pins = derive_stage1_source_pins(
        stage1_audit_inputs=preload["stage1_audit_inputs"],
        actual_qa_source_inputs=preload["actual_qa_source_inputs"])
    expected = material.get("expected_rows") if isinstance(material, dict) else None
    source_rows = pins.get("source_rows") if isinstance(pins, dict) else None
    if (material.get("status") != "three_row_candidate_inputs_derived_unvalidated"
            or material.get("requested_records") != ROW_COUNT
            or material.get("effective_records") != ROW_COUNT
            or pins.get("status") != "three_row_source_pins_structurally_derived"
            or pins.get("effective_records") != ROW_COUNT
            or not isinstance(expected, list) or len(expected) != ROW_COUNT
            or not isinstance(source_rows, list) or len(source_rows) != ROW_COUNT):
        raise ValueError("candidate_cohort_derivation_incomplete")

    keys = ("review_token", "source_video_sha256")
    for candidate, source, live in zip(expected, source_rows, live_rows, strict=True):
        if (not isinstance(candidate, dict) or not isinstance(source, dict)
                or any(candidate.get(key) != source.get(key)
                       or candidate.get(key) != live.get(key)
                       for key in keys)):
            raise ValueError("ordered_candidate_phase1_cohort_mismatch")
    sampler_rows = material.get("sampler_rows")
    if not isinstance(sampler_rows, list) or len(sampler_rows) != ROW_COUNT:
        raise ValueError("candidate_sampler_rows_missing")
    return expected, source_rows, sampler_rows


def _validate_processor_manifest(preload: dict, transaction: dict,
        expected: list[dict], live_rows: list[dict], sampler_rows: list[dict],
        prompt_reconstruction_inputs: Any, processor: Any,
        tokenizer: Any) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Cross-bind the signed manifest's available pins to candidate rows/live tensors.

    Prompt/token pins are independently reconstructed from ephemeral source
    text, sampled frames, the supplied CPU processor/tokenizer and live batch.
    Processor/tokenizer artifact authenticity and runtime origin remain unverified.
    """
    artifacts = transaction["phase1_artifact_bytes"]
    raw = artifacts.get("processor_batch_digest_manifest")
    if not isinstance(raw, bytes):
        raise ValueError("processor_manifest_missing")
    runtime_raw = artifacts.get("sampler_model_runtime")
    if not isinstance(runtime_raw, bytes) or not runtime_raw:
        raise ValueError("signed_runtime_descriptor_missing")
    runtime_sha = hashlib.sha256(runtime_raw).hexdigest()
    manifest = json.loads(raw.decode("utf-8"))
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    protocol = preload["actual_source_inputs"].get("qta_protocol_raw")
    if not isinstance(protocol, bytes):
        raise ValueError("processor_protocol_bytes_missing")
    protocol_sha = hashlib.sha256(protocol).hexdigest()
    bindings = [{"record_token": hashlib.sha256(row["review_token"].encode("ascii")).hexdigest(),
                 "processor_batch_sha256": row["batch_sha256"]}
                for row in live_rows]
    validate_manifest(raw, bindings, expected_row_pins=rows,
                      expected_protocol_sha256=protocol_sha)
    prompt_raw = artifacts.get("prompt_answer_token_bindings")
    if not isinstance(prompt_raw, bytes):
        raise ValueError("prompt_answer_bindings_missing")
    prompt_binding_report = validate_qta_prompt_answer_token_bindings(
        prompt_raw, expected_rows=expected, processor_rows=rows,
        live_processor_rows=live_rows, sampler_rows=sampler_rows,
        prompt_reconstruction_inputs=prompt_reconstruction_inputs,
        processor=processor, tokenizer=tokenizer)
    for pin, manifest_row, live in zip(expected, rows, live_rows, strict=True):
        batch = live["batch"]
        input_ids = batch.get("input_ids")
        grid = batch.get("video_grid_thw")
        if input_ids is None or grid is None:
            raise ValueError("processor_live_batch_fields_missing")
        ids = input_ids.detach().cpu().tolist() if callable(getattr(input_ids, "detach", None)) else input_ids.tolist()
        grid_values = grid.detach().cpu().tolist() if callable(getattr(grid, "detach", None)) else grid.tolist()
        if len(ids) != 1 or len(grid_values) != 1:
            raise ValueError("processor_live_batch_row_shape_invalid")
        if (not isinstance(ids[0], list) or not ids[0]
                or any(type(value) is not int or value < 0 for value in ids[0])
                or not isinstance(grid_values[0], list) or len(grid_values[0]) != 3
                or any(type(value) is not int or value <= 0 for value in grid_values[0])):
            raise ValueError("processor_live_batch_integer_metadata_invalid")
        ids_sha = hashlib.sha256(json.dumps(ids[0], sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        if (manifest_row["record_token"] != hashlib.sha256(pin["review_token"].encode("ascii")).hexdigest()
                or manifest_row["base_processor_batch_sha256"] != live["batch_sha256"]
                or manifest_row["source_video_sha256"] != pin["source_video_sha256"]
                or manifest_row["processor_sha256"] != pin["processor_fingerprint_sha256"]
                or manifest_row["tokenizer_sha256"] != pin["tokenizer_sha256"]
                or manifest_row["sampler_metadata_sha256"] != pin["sampler_row_sha256"]
                or manifest_row["runtime_sha256"] != runtime_sha
                or manifest_row["input_ids_sha256"] != ids_sha
                or manifest_row["video_grid_thw"] != grid_values[0]):
            raise ValueError("processor_manifest_candidate_or_live_pin_mismatch")
    return rows, protocol_sha, prompt_binding_report


def preflight_qta_direct_faithfulness_atomic(*, candidate_preload_inputs: Mapping[str, Any],
        phase1_transaction_inputs: Mapping[str, Any],
        prompt_reconstruction_inputs: Any = None, processor: Any = None,
        tokenizer: Any = None) -> dict[str, Any]:
    """Compose all three-row raw-input preflights atomically, without authority.

    A pass means only that the candidate/sampler/clock-presence branch and the
    signed source/review/live-batch phase-1 branch derive the same ordered
    three-row cohort. Human identity, claim semantics, clock/frame truth, and
    external root origin remain unverified. No model or CUDA is loaded or used.
    """
    try:
        preload = _copy_bundle(candidate_preload_inputs, PRELOAD_REQUIRED, PRELOAD_OPTIONAL)
        if not isinstance(phase1_transaction_inputs, Mapping):
            raise ValueError("transaction_bundle_invalid")
        # Keep live tensors by reference while snapshotting every other field.
        transaction_raw = dict(phase1_transaction_inputs)
        raw_live_rows = transaction_raw.pop("live_processor_rows", None)
        transaction = _copy_bundle(
            transaction_raw, TRANSACTION_REQUIRED - {"live_processor_rows"},
            TRANSACTION_OPTIONAL)
        live_rows = _snapshot_live_rows(raw_live_rows)
        transaction["live_processor_rows"] = live_rows

        # The two APIs intentionally use different shapes for
        # ``actual_source_inputs``: the candidate bridge takes source/claim
        # lineage, while the phase-1 transaction takes the QA-source binding.
        # Their authoritative common inputs are the Stage-1 audit and QA
        # source bytes; the candidate bridge validates its separate lineage
        # against those pins itself.
        common = ("stage1_audit_inputs", "actual_qa_source_inputs")
        for key in common:
            if preload[key] != transaction[key]:
                raise ValueError("cross_branch_raw_input_mismatch")
        review_inputs = transaction["review_bundle_inputs"]
        stage1 = preload["stage1_audit_inputs"]
        for review_key, stage_key in (("worksheet_raw", "worksheet_raw"),
                ("lineage_raw", "lineage_raw"), ("review_raw", "review_raw"),
                ("review_index_raw", "review_index_raw"),
                ("expected_lineage_sha256", "expected_lineage_sha256")):
            if review_inputs.get(review_key) != stage1.get(stage_key):
                raise ValueError("cross_branch_review_bytes_mismatch")

        expected, source_rows, sampler_rows = _ordered_cohort(preload, live_rows)
        manifest_rows, protocol_sha, prompt_binding_report = _validate_processor_manifest(
            preload, transaction, expected, live_rows, sampler_rows,
            prompt_reconstruction_inputs, processor, tokenizer)
        transaction["expected_processor_batch_row_pins"] = manifest_rows
        transaction["expected_processor_batch_protocol_sha256"] = protocol_sha
        preload_report = evaluate_qta_direct_faithfulness_preload(**preload)
        transaction_report = preflight_qta_phase1_batch_transaction(**transaction)
        if (preload_report.get("status") != "structural_inputs_composed_unvalidated"
                or preload_report.get("structurally_composed_rows") != ROW_COUNT
                or preload_report.get("model_load_permitted") is not False
                or preload_report.get("execution_authorization") != "not_granted"):
            raise ValueError("candidate_preload_not_composed")
        if (transaction_report.get("status") != "structural_bindings_consistent_unvalidated"
                or transaction_report.get("structurally_bound_rows") != ROW_COUNT
                or transaction_report.get("model_load_permitted") is not False
                or transaction_report.get("scoring_authorized") is not False
                or transaction_report.get("execution_authorization") != "not_granted"):
            raise ValueError("phase1_transaction_not_composed")

        # Recheck that source rows, the candidate bridge, and the exact live
        # batch references still agree after every validator has run.
        for candidate, source, row in zip(expected, source_rows, live_rows, strict=True):
            if (candidate["review_token"] != row["review_token"]
                    or source["review_token"] != row["review_token"]
                    or candidate["source_video_sha256"] != row["source_video_sha256"]
                    or source["source_video_sha256"] != row["source_video_sha256"]
                    or batch_sha256(row["batch"]) != row["batch_sha256"]):
                raise ValueError("final_live_batch_binding_changed")

        return {
            "schema_version": 1,
            "purpose": "qta_direct_faithfulness_atomic_preload_structure_only",
            "status": "same_cohort_structurally_consistent_unvalidated",
            "requested_rows": ROW_COUNT,
            "matched_rows": ROW_COUNT,
            "requested_clock_frame_anchors": ANCHOR_COUNT,
            # The leaf report counts cryptographically checked decisions but
            # does not return their row tokens. This wrapper therefore cannot
            # safely attribute those decisions to this exact cohort.
            "signature_bound_frame_pair_decisions": 0,
            "validated_clock_frame_anchors": 0,
            "cohort_binding": "passed_unvalidated",
            "candidate_preload": "passed_unvalidated",
            "phase1_batch_transaction": "passed_unvalidated",
            "processor_manifest_binding": "candidate_source_and_live_tensor_pins_matched_unvalidated",
            "prompt_binding_status": prompt_binding_report["status"],
            "prompt_rendering_independently_reconstructed": True,
            "prompt_renderer_origin_authenticated": False,
            "tokenizer_authenticity_verified": False,
            "processor_authenticity_verified": False,
            "runtime_descriptor_origin_authenticated": False,
            "external_trust_root_origin_authenticated": False,
            "reviewer_identity_authenticated": False,
            "semantic_review_authenticated": False,
            "clock_frame_correspondence_authenticated": False,
            "direct_faithfulness_ready": False,
            "model_load_permitted": False,
            "scoring_authorized": False,
            "forward_calls": 0,
            "score_calls": 0,
            "faithfulness_scored": False,
            "execution_authorization": "not_granted",
            "failure_code": None,
        }
    except Exception:
        # Atomic failure: expose neither private exception content nor partial
        # row/anchor counts from whichever branch happened to pass.
        return _blocked()
