"""Sign and verify exact-byte adoption of candidate clock-capture artifacts.

This non-authorizing artifact attestation binds a candidate 15-anchor plan,
raw player events, screenshot bytes, OS capture-receipt rows, crop-map and crop
bytes to the distinct capture-operator key in the root-signed release roster.
It proves only that the rostered key signed those exact artifact hashes. It
does not prove who used the key, that capture actually occurred, or that a
screenshot matches a decoder frame.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any

from audit_nextgqa_review_release_order import _validate_release_roster
from crop_nextgqa_clock_screenshots import _plan_assignments, _validate_receipt
from nextgqa_pre_review_integrity_receipt import (
    ReceiptError, _canonical, _decode_signature, _public_der, _sha,
    _sign, _verify_signature,
)

DOMAIN = b"nextgqa_capture_operator_artifact_attestation_v1\0"
SIGNATURE_FIELD = "signature_ed25519_base64"
ANCHOR_ORDER = ("fit_begin", "heldout_quarter", "fit_midpoint",
                "heldout_three_quarter", "fit_end")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{24}$")
MAX_JSON = 2_000_000
MAX_ARTIFACT = 40_000_000


class CaptureOperatorAttestationError(ValueError):
    """Content-free failure from the capture-operator attestation boundary."""


def _fail(code: str) -> None:
    raise CaptureOperatorAttestationError(code)


def _parse(raw: bytes, code: str) -> Any:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_JSON:
        _fail(code)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("json_duplicate_key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=lambda _: _fail("json_nonfinite_value"))
    except CaptureOperatorAttestationError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, RecursionError):
        _fail(code)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _plan_order(plan_raw: bytes, expected_plan_sha256: str) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    if not _valid_sha(expected_plan_sha256) or _sha(plan_raw) != expected_plan_sha256:
        _fail("frozen_anchor_plan_hash_mismatch")
    try:
        assignments, _media = _plan_assignments(plan_raw, expected_plan_sha256)
    except ValueError as exc:
        _fail("frozen_anchor_plan_invalid:" + str(exc))
    plan = _parse(plan_raw, "anchor_plan_json_invalid")
    ordered = []
    seen_tokens = set()
    for record in plan["records"]:
        token = record["review_token"]
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token) or token in seen_tokens:
            _fail("frozen_anchor_plan_token_invalid_or_duplicate")
        seen_tokens.add(token)
        anchor_pts = {anchor["anchor_id"]: anchor["decoder_pts_seconds"]
                      for group in (record["fit_anchors"], record["heldout_anchors"])
                      for anchor in group}
        if set(anchor_pts) != set(ANCHOR_ORDER):
            _fail("frozen_anchor_plan_anchor_set_invalid")
        ordered.extend((token, anchor) for anchor in ANCHOR_ORDER)
    if len(ordered) != 15 or set(ordered) != assignments or len(set(ordered)) != 15:
        _fail("frozen_anchor_plan_coverage_invalid")
    return ordered, assignments


def _validate_inputs(*, plan_raw: bytes, expected_plan_sha256: str,
                     plan_provenance_raw: bytes, expected_plan_provenance_sha256: str,
                     observations_raw: bytes, observation_index_raw: bytes,
                     capture_receipt_raw: bytes, crop_map_raw: bytes,
                     crop_crosswalk_raw: bytes,
                     screenshot_files: Mapping[str, bytes],
                     crop_files: Mapping[str, bytes],
                     integrity_roster_raw: bytes, release_roster_raw: bytes,
                     root_public_key_pem: bytes,
                     expected_root_public_der_sha256: str) -> tuple[dict, list[dict], bytes]:
    ordered, expected = _plan_order(plan_raw, expected_plan_sha256)
    if (not _valid_sha(expected_plan_provenance_sha256)
            or _sha(plan_provenance_raw) != expected_plan_provenance_sha256):
        _fail("anchor_plan_provenance_hash_mismatch")
    provenance = _parse(plan_provenance_raw, "anchor_plan_provenance_invalid")
    if (not isinstance(provenance, dict)
            or provenance.get("purpose") != "private_clock_anchor_plan_candidate_provenance"
            or provenance.get("status") != "candidate_not_jointly_frozen"
            or provenance.get("execution_authorization") != "not_granted"
            or provenance.get("plan_sha256") != expected_plan_sha256):
        _fail("anchor_plan_provenance_binding_invalid")

    observations = _parse(observations_raw, "raw_observations_invalid")
    if (not isinstance(observations, dict)
            or type(observations.get("schema_version")) is not int
            or observations.get("schema_version") != 3
            or observations.get("purpose") != "local_unvalidated_html5_player_clock_observations"
            or observations.get("status") != "raw_player_clock_observations_only"
            or observations.get("execution_authorization") != "not_granted"
            or not isinstance(observations.get("records"), list)
            or len(observations["records"]) != 3):
        _fail("raw_observations_contract_invalid")
    observed_by_token = {}
    event_by_anchor: dict[tuple[str, str], dict] = {}
    for record in observations["records"]:
        if (not isinstance(record, dict) or set(record) != {"review_token", "events"}
                or not isinstance(record.get("review_token"), str)
                or not TOKEN_RE.fullmatch(record["review_token"])
                or record["review_token"] in observed_by_token
                or not isinstance(record.get("events"), list)
                or len(record["events"]) != 5):
            _fail("raw_observation_record_invalid")
        token = record["review_token"]
        observed_by_token[token] = record
        seen_anchors = set()
        for expected_sequence, event in enumerate(record["events"], 1):
            if (not isinstance(event, dict)
                    or not isinstance(event.get("anchor_id"), str)
                    or event.get("anchor_id") not in ANCHOR_ORDER
                    or event["anchor_id"] in seen_anchors
                    or type(event.get("sequence")) is not int
                    or event["sequence"] != expected_sequence):
                _fail("raw_observation_anchor_duplicate_or_invalid")
            anchor = event["anchor_id"]
            if anchor != ANCHOR_ORDER[expected_sequence - 1]:
                _fail("raw_observation_anchor_order_invalid")
            seen_anchors.add(anchor)
            event_by_anchor[(token, anchor)] = event
        if seen_anchors != set(ANCHOR_ORDER):
            _fail("raw_observation_anchor_coverage_invalid")
    if set(observed_by_token) != {token for token, _ in ordered}:
        _fail("raw_observation_plan_token_mismatch")

    observation_index = _parse(observation_index_raw, "observation_index_invalid")
    if (not isinstance(observation_index, dict)
            or set(observation_index) != {"schema_version", "purpose", "payload_sha256",
                                         "created_at_utc", "status", "execution_authorization"}
            or type(observation_index.get("schema_version")) is not int
            or observation_index.get("schema_version") != 1
            or observation_index.get("purpose") != "unvalidated_html5_clock_observation_integrity_index"
            or observation_index.get("payload_sha256") != _sha(observations_raw)
            or observation_index.get("status") != "raw_observations_only"
            or observation_index.get("execution_authorization") != "not_granted"):
        _fail("observation_index_binding_invalid")

    try:
        receipt_rows = _validate_receipt(capture_receipt_raw, _sha(capture_receipt_raw),
            expected_plan_sha256, expected_plan_provenance_sha256, expected)
    except ValueError as exc:
        _fail("capture_receipt_invalid:" + str(exc))
    receipt = _parse(capture_receipt_raw, "capture_receipt_invalid")
    if not _valid_sha(receipt.get("capture_helper_file_sha256")):
        _fail("capture_helper_hash_invalid")
    screenshot_names = {row["screenshot_file"] for row in receipt_rows.values()}
    if (not isinstance(screenshot_files, Mapping)
            or set(screenshot_files) != screenshot_names
            or len(screenshot_names) != 15):
        _fail("screenshot_artifact_set_mismatch")
    for key, row in receipt_rows.items():
        raw = screenshot_files.get(row["screenshot_file"])
        if (not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARTIFACT
                or _sha(raw) != row["screenshot_sha256"]):
            _fail("screenshot_artifact_hash_mismatch")

    crosswalk = _parse(crop_crosswalk_raw, "crop_crosswalk_invalid")
    crosswalk_keys = {"schema_version", "purpose", "status", "execution_authorization",
        "plan_sha256", "plan_provenance_sha256", "capture_receipt_sha256",
        "crop_map_sha256", "bbox_video_only_purity", "records"}
    if (not isinstance(crosswalk, dict) or set(crosswalk) != crosswalk_keys
            or type(crosswalk.get("schema_version")) is not int
            or crosswalk.get("schema_version") != 1
            or crosswalk.get("purpose") != "private_candidate_screenshot_crop_crosswalk"
            or crosswalk.get("status") != "candidate_only"
            or crosswalk.get("execution_authorization") != "not_granted"
            or crosswalk.get("plan_sha256") != expected_plan_sha256
            or crosswalk.get("plan_provenance_sha256") != expected_plan_provenance_sha256
            or crosswalk.get("capture_receipt_sha256") != _sha(capture_receipt_raw)
            or not _valid_sha(crosswalk.get("crop_map_sha256"))
            or crosswalk.get("bbox_video_only_purity") != "not_established_human_review_required"
            or not isinstance(crosswalk.get("records"), list)
            or len(crosswalk["records"]) != 15):
        _fail("crop_crosswalk_binding_invalid")
    crops_by_key = {}
    crop_names = set()
    for row in crosswalk["records"]:
        row_keys = {"review_token", "anchor_id", "screenshot_file", "screenshot_sha256",
                    "bbox_xywh", "crop_file", "crop_sha256", "crop_rgb_dimensions", "crop_purity"}
        if not isinstance(row, dict) or set(row) != row_keys:
            _fail("crop_crosswalk_record_invalid")
        token, anchor = row.get("review_token"), row.get("anchor_id")
        key = (token, anchor)
        bbox, dims = row.get("bbox_xywh"), row.get("crop_rgb_dimensions")
        filename = row.get("crop_file")
        if (key not in expected or key in crops_by_key
                or not isinstance(filename, str) or not filename or filename in crop_names
                or filename in {".", ".."} or Path(filename).name != filename
                or not _valid_sha(row.get("crop_sha256"))
                or not isinstance(row.get("screenshot_sha256"), str)
                or row.get("screenshot_sha256") != receipt_rows[key]["screenshot_sha256"]
                or row.get("screenshot_file") != receipt_rows[key]["screenshot_file"]
                or not isinstance(bbox, list) or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or not isinstance(dims, list) or len(dims) != 2
                or any(type(value) is not int or value < 1 for value in dims)
                or row.get("crop_purity") != "human_review_required"):
            _fail("crop_crosswalk_record_invalid")
        crops_by_key[key] = row
        crop_names.add(filename)
    if set(crops_by_key) != expected or not isinstance(crop_files, Mapping) or set(crop_files) != crop_names:
        _fail("crop_artifact_set_mismatch")
    for row in crops_by_key.values():
        raw = crop_files[row["crop_file"]]
        if (not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARTIFACT
                or _sha(raw) != row["crop_sha256"]):
            _fail("crop_artifact_hash_mismatch")

    try:
        _roster, public_by_role = _validate_release_roster(
            release_roster_raw, root_public_key_pem,
            expected_root_public_der_sha256, integrity_roster_raw)
    except ReceiptError as exc:
        _fail("release_roster_invalid:" + str(exc))
    if "capture_operator" not in public_by_role:
        _fail("capture_operator_role_missing")
    return {"receipt": receipt, "operator_public_key_pem": public_by_role["capture_operator"]}, [
        {"key": key, "event": event_by_anchor[key],
         "receipt_row": receipt_rows[key], "crop_row": crops_by_key[key]}
        for key in ordered], integrity_roster_raw


def _body(*, plan_raw: bytes, expected_plan_sha256: str,
          plan_provenance_raw: bytes, expected_plan_provenance_sha256: str,
          observations_raw: bytes, observation_index_raw: bytes,
          capture_receipt_raw: bytes, crop_map_raw: bytes,
          crop_crosswalk_raw: bytes, screenshot_files: Mapping[str, bytes],
          crop_files: Mapping[str, bytes], integrity_roster_raw: bytes,
          release_roster_raw: bytes, root_public_key_pem: bytes,
          expected_root_public_der_sha256: str) -> tuple[dict[str, Any], bytes]:
    if (not isinstance(crop_map_raw, bytes) or not 0 < len(crop_map_raw) <= MAX_JSON
            or _sha(crop_map_raw) != _parse(crop_crosswalk_raw, "crop_crosswalk_invalid").get("crop_map_sha256")):
        _fail("crop_map_hash_mismatch")
    validated, rows, _ = _validate_inputs(
        plan_raw=plan_raw, expected_plan_sha256=expected_plan_sha256,
        plan_provenance_raw=plan_provenance_raw,
        expected_plan_provenance_sha256=expected_plan_provenance_sha256,
        observations_raw=observations_raw, observation_index_raw=observation_index_raw,
        capture_receipt_raw=capture_receipt_raw, crop_map_raw=crop_map_raw,
        crop_crosswalk_raw=crop_crosswalk_raw,
        screenshot_files=screenshot_files, crop_files=crop_files,
        integrity_roster_raw=integrity_roster_raw, release_roster_raw=release_roster_raw,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256)
    roster, role_keys = _validate_release_roster(
        release_roster_raw, root_public_key_pem,
        expected_root_public_der_sha256, integrity_roster_raw)
    operator_entry = next((entry for entry in roster["keys"]
                           if entry.get("role") == "capture_operator"), None)
    if not isinstance(operator_entry, dict):
        _fail("capture_operator_role_missing")
    operator_public = role_keys["capture_operator"]
    operator_fingerprint = _sha(_public_der(operator_public))
    if operator_entry.get("public_key_der_sha256") != operator_fingerprint:
        _fail("capture_operator_roster_key_mismatch")
    anchors = []
    for slot, row in enumerate(rows):
        token, anchor = row["key"]
        anchors.append({
            "record_slot": slot // 5 + 1,
            "anchor_slot": slot % 5 + 1,
            "anchor_ref_sha256": _sha(_canonical(["capture-anchor-ref-v1", token, anchor])),
            "raw_event_sha256": _sha(_canonical(row["event"])),
            "screenshot_sha256": _sha(screenshot_files[row["receipt_row"]["screenshot_file"]]),
            "os_receipt_row_sha256": _sha(_canonical(row["receipt_row"])),
            "crop_sha256": _sha(crop_files[row["crop_row"]["crop_file"]]),
        })
    body = {
        "schema_version": 1,
        "purpose": "nextgqa_capture_operator_artifact_attestation_v1",
        "status": "capture_operator_key_attested_candidate_artifact_bytes",
        "plan_sha256": expected_plan_sha256,
        "plan_provenance_sha256": expected_plan_provenance_sha256,
        "observations_sha256": _sha(observations_raw),
        "observation_index_sha256": _sha(observation_index_raw),
        "capture_receipt_sha256": _sha(capture_receipt_raw),
        "crop_map_sha256": _sha(crop_map_raw),
        "crop_crosswalk_sha256": _sha(crop_crosswalk_raw),
        "integrity_roster_sha256": _sha(integrity_roster_raw),
        "release_roster_sha256": _sha(release_roster_raw),
        "trust_root_der_sha256": expected_root_public_der_sha256,
        "capture_helper_file_sha256": validated["receipt"]["capture_helper_file_sha256"],
        "capture_operator_role": "capture_operator",
        "capture_operator_key_id": operator_entry["key_id"],
        "capture_operator_key_der_sha256": operator_fingerprint,
        "statement_id": "operator_adopts_exact_candidate_capture_artifact_bytes_v1",
        "anchor_count": 15,
        "anchors": anchors,
        "human_identity_verified": False,
        "capture_process_authenticated": False,
        "frame_correspondence_reviewed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
    }
    return body, operator_public


def create_attestation(*, reviewer_private_key_pem: bytes,
                       reviewer_public_key_pem: bytes, **inputs: Any) -> bytes:
    """Sign exact candidate capture-artifact hashes with rostered operator key."""
    body, rostered_public = _body(**inputs)
    if _public_der(reviewer_public_key_pem) != _public_der(rostered_public):
        _fail("capture_operator_public_key_does_not_match_roster")
    try:
        signature = _sign(reviewer_private_key_pem, reviewer_public_key_pem,
                          DOMAIN + _canonical(body))
    except ReceiptError as exc:
        _fail("capture_operator_signing_key_invalid:" + str(exc))
    return (json.dumps({**body, SIGNATURE_FIELD: base64.b64encode(signature).decode("ascii")},
                       sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def verify_attestation(*, attestation_raw: bytes, **inputs: Any) -> dict[str, Any]:
    """Verify signature and recompute every plan and artifact binding."""
    expected, operator_public = _body(**inputs)
    envelope = _parse(attestation_raw, "capture_operator_attestation_json_invalid")
    expected_keys = set(expected) | {SIGNATURE_FIELD}
    if (not isinstance(envelope, dict) or set(envelope) != expected_keys
            or type(envelope.get("schema_version")) is not int
            or type(envelope.get("anchor_count")) is not int
            or envelope.get("anchor_count") != 15):
        _fail("capture_operator_attestation_schema_invalid")
    body = {key: value for key, value in envelope.items() if key != SIGNATURE_FIELD}
    if _canonical(body) != _canonical(expected):
        _fail("capture_operator_attestation_binding_mismatch")
    signature_value = envelope.get(SIGNATURE_FIELD)
    try:
        signature = base64.b64decode(signature_value, validate=True)
    except (TypeError, ValueError):
        _fail("capture_operator_signature_invalid")
    if len(signature) != 64 or not _verify_signature(
            operator_public, signature, DOMAIN + _canonical(body)):
        _fail("capture_operator_signature_invalid")
    return {
        "status": "capture_operator_key_signature_valid_candidate_artifact_bytes",
        "capture_operator_role": "capture_operator",
        "capture_operator_key_id": expected["capture_operator_key_id"],
        "anchor_count": 15,
        "all_exact_artifact_hashes_recomputed": True,
        "human_identity_verified": False,
        "capture_process_authenticated": False,
        "frame_correspondence_reviewed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
    }
