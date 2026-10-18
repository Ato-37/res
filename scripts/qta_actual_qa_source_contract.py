"""Fail-closed, CPU-only QTA binding of a fixed three-row claim cohort to QA.

This creates private source material only. It never authorizes model execution or
scoring. QTA protocol/config pins are supplied by the caller and are not inferred.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from nextgqa_actual_qa_source_adapter import build_actual_qa_source

SHA256 = re.compile(r"^[a-f0-9]{64}$")
DENIAL = {
    "model_load_permitted": False,
    "execution_authorized": False,
    "scoring_authorized": False,
    "score_calls": 0,
    "baseline_forwards": 0,
    "branch_forwards": 0,
}
MAX_INPUT_BYTES = 24 * 1024 * 1024


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json(raw: bytes) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("input_decode_invalid") from None


def _failure_code(exc: Exception) -> str:
    value = str(exc)
    return value if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value) else "binding_invalid"


def bind_qta_actual_qa_source(*, csv_raw: bytes, claim_manifest_raw: bytes,
                              selection_raw: bytes, target_manifest_raw: bytes,
                              lineage_manifest_raw: bytes, qta_protocol_raw: bytes,
                              expected_qta_protocol_sha256: str,
                              qta_config_raw: bytes,
                              expected_qta_config_sha256: str) -> dict[str, Any]:
    """Return aggregate-only public summary and a private QA bundle on success.

    The denominator is always the fixed three-row cohort. A failure yields no
    partial private bundle and validated_rows=0. The private QA bytes include
    questions, choices, and answers; callers must keep them local/private.
    """
    raw_inputs = {
        "csv": csv_raw, "claim_manifest": claim_manifest_raw,
        "selection": selection_raw, "target_manifest": target_manifest_raw,
        "lineage_manifest": lineage_manifest_raw,
        "qta_protocol": qta_protocol_raw, "qta_config": qta_config_raw,
    }
    hashes = {name: _sha(raw) if isinstance(raw, bytes) else None
              for name, raw in raw_inputs.items()}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "qta_actual_qa_source_binding_unvalidated",
        "status": "blocked",
        "requested_rows": 3,
        "validated_rows": 0,
        "protocol_pin_verified": False,
        "config_pin_verified": False,
        "primary_estimand": "baseline_full_video_frozen_p_over_q_prediction_link",
        "p_q_frozen": False,
        "gold_answer_index_use": "secondary_only_after_baseline_prediction_freeze",
        "rights_status": "unverified_local_only",
        "faithfulness_scored": False,
        "failure_code": "binding_invalid",
        **DENIAL,
    }
    result: dict[str, Any] = {"summary": summary, "private_bundle": None}
    try:
        if any(not isinstance(raw, bytes) for raw in raw_inputs.values()):
            raise ValueError("input_bytes_invalid")
        if any(len(raw) > MAX_INPUT_BYTES for raw in raw_inputs.values()):
            raise ValueError("input_byte_limit_exceeded")
        if (not isinstance(expected_qta_protocol_sha256, str)
                or not SHA256.fullmatch(expected_qta_protocol_sha256)
                or hashes["qta_protocol"] != expected_qta_protocol_sha256):
            raise ValueError("qta_protocol_digest_pin_mismatch")
        summary["protocol_pin_verified"] = True
        if (not isinstance(expected_qta_config_sha256, str)
                or not SHA256.fullmatch(expected_qta_config_sha256)
                or hashes["qta_config"] != expected_qta_config_sha256):
            raise ValueError("qta_config_digest_pin_mismatch")
        summary["config_pin_verified"] = True

        claim_doc = _json(claim_manifest_raw)
        selection = _json(selection_raw)
        lineage = _json(lineage_manifest_raw)
        target = _json(target_manifest_raw)
        if (not isinstance(claim_doc, dict) or not isinstance(claim_doc.get("records"), list)
                or len(claim_doc["records"]) != 3):
            raise ValueError("claim_manifest_rows_invalid")
        if (not isinstance(selection, dict) or set(selection) != {"records"}
                or not isinstance(selection["records"], list)
                or len(selection["records"]) != 3):
            raise ValueError("selection_rows_invalid")
        if (not isinstance(lineage, dict) or not isinstance(lineage.get("records"), list)
                or len(lineage["records"]) != 3
                or not isinstance(target, dict) or not isinstance(target.get("records"), list)
                or len(target["records"]) != 3):
            raise ValueError("upstream_rows_invalid")

        ordered_hashes = []
        source_selection_rows = []
        review_source_crosswalk = []
        seen_review_tokens = set()
        for claim, selected, line, target_row in zip(
                claim_doc["records"], selection["records"], lineage["records"],
                target["records"], strict=True):
            if not all(isinstance(row, dict) for row in (claim, selected, line, target_row)):
                raise ValueError("cohort_row_invalid")
            if set(selected) != {"review_token", "video_id", "claim_id",
                                 "dataset_video_id", "dataset_question_id"}:
                raise ValueError("selection_row_schema_invalid")
            claim_text = claim.get("claim_text")
            claim_id, video_id = claim.get("claim_id"), claim.get("video_id")
            if (not isinstance(claim_text, str) or not claim_text.strip()
                    or not isinstance(claim_id, str) or not isinstance(video_id, str)):
                raise ValueError("claim_fields_invalid")
            claim_sha = _sha(claim_text.encode("utf-8"))
            if claim.get("claim_text_sha256") != claim_sha:
                raise ValueError("claim_text_digest_mismatch")
            video_sha = line.get("video_sha256")
            if not isinstance(video_sha, str) or not SHA256.fullmatch(video_sha):
                raise ValueError("lineage_video_digest_invalid")
            if (claim_id != line.get("claim_id") or video_id != line.get("video_id")
                    or claim_id != target_row.get("claim_id")
                    or video_id != target_row.get("video_id")
                    or line.get("claim_text_sha256") != claim_sha
                    or target_row.get("claim_text_sha256") != claim_sha):
                raise ValueError("claim_lineage_target_mismatch")
            expected_claim_id = "claim-" + _sha(
                (video_sha + "\0"
                 + line.get("dataset_question_id", "") + "\0" + claim_sha).encode("utf-8"))
            if claim_id != expected_claim_id:
                raise ValueError("claim_identity_mismatch")
            review_token = selected.get("review_token")
            if (not isinstance(review_token, str) or not review_token.strip()
                    or review_token in seen_review_tokens):
                raise ValueError("review_token_identity_invalid")
            seen_review_tokens.add(review_token)
            if (selected.get("video_id") != video_id
                    or selected.get("claim_id") != claim_id
                    or selected.get("dataset_video_id") != line.get("dataset_video_id")
                    or selected.get("dataset_question_id") != line.get("dataset_question_id")):
                raise ValueError("selection_lineage_mismatch")
            source_selection_rows.append({
                "video_id": video_id, "claim_id": claim_id,
                "dataset_video_id": line["dataset_video_id"],
                "dataset_question_id": line["dataset_question_id"],
            })
            review_source_crosswalk.append({
                "review_token": review_token, **source_selection_rows[-1],
            })
            ordered_hashes.append(_sha(_canonical({
                "review_token_sha256": _sha(review_token.encode("utf-8")),
                "claim_text_sha256": claim_sha,
                "claim_id_sha256": _sha(claim_id.encode()),
                "video_id_sha256": _sha(video_id.encode()),
                "source_question_sha256": line.get("question_sha256"),
                "source_video_sha256": line.get("video_sha256"),
            })))

        qa_source, bound_manifest = build_actual_qa_source(
            csv_raw=csv_raw, selection={"records": source_selection_rows},
            target_manifest_raw=target_manifest_raw,
            lineage_manifest_raw=lineage_manifest_raw)
        # These source fields are frozen by the upstream target-candidate
        # builder. The adapter also computes them, so compare before accepting
        # its enriched output; otherwise a stale/tampered candidate would be
        # silently repaired and lose its provenance.
        source_fields = ("dataset_qid_sha256", "dataset_source_row_sha256",
                         "question_sha256", "ordered_choice_sha256", "gold_index")
        target_bound = bound_manifest["target_manifest"]["records"]
        if len(target_bound) != len(target["records"]):
            raise ValueError("target_source_binding_mismatch")
        for supplied, recomputed in zip(target["records"], target_bound, strict=True):
            if any(field not in supplied or supplied[field] != recomputed.get(field)
                   for field in source_fields):
                raise ValueError("target_source_binding_mismatch")
        # Adapter output carries identities/gold. Keep all of it only in the
        # explicitly private return channel; public summary contains aggregates.
        private_raw = _canonical({
            "input_sha256": hashes,
            "ordered_cohort_binding_sha256": _sha(_canonical(ordered_hashes)),
            "review_source_crosswalk": review_source_crosswalk,
            "qa_source": qa_source,
            "bound_manifest": bound_manifest,
        })
        summary.update({
            "status": "bound_unvalidated",
            "validated_rows": 3,
            "failure_code": None,
        })
        result["private_bundle"] = {
            "classification": "private_local_only_unvalidated_qa_source",
            "sha256": _sha(private_raw),
            "ordered_cohort_binding_sha256": _sha(_canonical(ordered_hashes)),
            "raw_json": private_raw,
        }
    except Exception as exc:  # Any unexpected condition must remain non-authorizing.
        summary["failure_code"] = _failure_code(exc)
    # Denial invariant is re-applied last, including all exception paths.
    summary.update(DENIAL)
    return result
