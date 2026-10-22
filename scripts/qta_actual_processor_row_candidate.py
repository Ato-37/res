"""Build a local, non-authorizing three-row QTA processor candidate.

This adapter binds supplied source rows, the repository's fixed QA prompt,
tokenizer continuation IDs, sampler metadata and the processor's CPU batch.
Its hashes are integrity pins only: it does not authenticate source media,
reviewers, clocks, human decisions, or authorize loading/scoring a model.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from nextgqa_atomic_batch_freeze import batch_sha256
from qta_choice_token_binding import build_choice_token_binding
from qta_prompt_answer_token_bindings_v1 import build_qta_prompt_answer_token_bindings
from qta_prescore_processor_row_adapter import processor_binding_row
from necessity_choice_margin_utils import uniform_sample_indices
from qwen_processor_batch_review_mapping import map_review_interval_from_processor_batch
from qwen_question_temporal_attention_selector import map_question_token_positions
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video

ROW_COUNT = 3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DENIAL = {
    "model_load_permitted": False,
    "scoring_authorized": False,
    "execution_authorization": "not_granted",
    "human_identity_verified": False,
    "semantic_review_authenticated": False,
    "source_authentication_status": "not_performed",
    "faithfulness_scored": False,
}
ROW_KEYS = {
    "review_token", "source_video_sha256", "question", "choices", "claim",
    "frames", "video_duration_seconds",
    "reviewed_intervals_seconds", "interval_boundary_guard_seconds",
    "processor_revision", "processor_fingerprint_sha256", "video_token_id",
    "spatial_merge_size", "temporal_patch_size",
}
PIN_KEYS = {
    "review_token", "source_video_sha256", "question_sha256",
    "options_sha256", "claim_sha256", "reviewed_intervals_sha256",
    "tokenizer_sha256", "processor_revision", "processor_fingerprint_sha256",
    "sampler_row_sha256",
    "video_duration_seconds", "interval_boundary_guard_seconds",
}
SAMPLER_REQUIRED = {
    "source_sha256", "decoded_frame_count", "sampled_frame_count",
    "sampled_indices", "sampled_pts", "effective_sample_fps",
    "sampled_rgb_sha256",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("candidate_digest_invalid")
    return value


def _source_pins(row: Mapping[str, Any]) -> dict[str, str]:
    question = row["question"]
    choices = row["choices"]
    claim = row["claim"]
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(claim, str) or not claim.strip()
            or not isinstance(choices, (list, tuple)) or len(choices) != 5
            or any(not isinstance(x, str) or not x.strip() for x in choices)):
        raise ValueError("candidate_source_text_invalid")
    return {
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "options_sha256": hashlib.sha256(_canonical(list(choices))).hexdigest(),
        "claim_sha256": hashlib.sha256(claim.encode("utf-8")).hexdigest(),
    }


def _candidate_row_commitment(row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical integrity projection; it is not a signature/authentication."""
    fields = (
        "review_token", "source_video_sha256", "typed_batch_sha256",
        "question_sha256", "options_sha256", "claim_sha256", "tokenizer_sha256",
        "processor_revision", "processor_fingerprint_sha256", "sampler_row_sha256",
        "video_duration_seconds", "interval_boundary_guard_seconds",
        "sampling_metadata_sha256", "video_grid_thw",
        "reviewed_intervals_rational_seconds", "interval_boundary_guard_rational_seconds",
        "processor_uniform_grid_assignment_check", "evidence_group_indices",
        "control_group_indices", "temporal_group_token_positions",
        "rendered_prompt_sha256", "prompt_token_ids_sha256",
        "choice_token_binding_sha256", "answer_token_ids", "question_positions",
        "input_ids_sha256",
    )
    return {field: row[field] for field in fields}


def _sampler_binding(sampler_row: Any, *, source_sha256: str) -> tuple[dict, list[int], list[Fraction], int, Fraction, str]:
    if not isinstance(sampler_row, Mapping) or not SAMPLER_REQUIRED.issubset(sampler_row):
        raise ValueError("candidate_external_sampler_row_invalid")
    source = _sha(sampler_row.get("source_sha256"))
    if source != source_sha256:
        raise ValueError("candidate_sampler_source_splice_rejected")
    decoded = sampler_row.get("decoded_frame_count")
    count = sampler_row.get("sampled_frame_count")
    raw_indices = sampler_row.get("sampled_indices")
    raw_pts = sampler_row.get("sampled_pts")
    raw_fps = sampler_row.get("effective_sample_fps")
    rgb_hashes = sampler_row.get("sampled_rgb_sha256")
    if (type(decoded) is not int or decoded < 32 or type(count) is not int or count != 32
            or not isinstance(raw_indices, list) or len(raw_indices) != 32
            or any(type(value) is not int for value in raw_indices)
            or raw_indices != uniform_sample_indices(decoded, 32)
            or not isinstance(raw_pts, list) or len(raw_pts) != 32
            or not isinstance(rgb_hashes, list) or len(rgb_hashes) != 32
            or any(not isinstance(value, str) or not SHA256.fullmatch(value)
                   for value in rgb_hashes)):
        raise ValueError("candidate_external_sampler_grid_invalid")
    pts = []
    for value in raw_pts:
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                if any(type(part) is not int for part in value) or value[1] <= 0:
                    raise ValueError
                parsed = Fraction(value[0], value[1])
            else:
                parsed = Fraction(str(value))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            raise ValueError("candidate_external_sampler_pts_invalid") from None
        pts.append(parsed)
    if pts[0] != 0 or any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("candidate_external_sampler_pts_invalid")
    try:
        if (not isinstance(raw_fps, (list, tuple)) or len(raw_fps) != 2
                or any(type(part) is not int for part in raw_fps) or raw_fps[1] <= 0):
            raise ValueError
        fps = Fraction(raw_fps[0], raw_fps[1])
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError("candidate_external_sampler_fps_invalid") from None
    inferred = Fraction(len(pts) - 1, 1) / (pts[-1] - pts[0])
    if fps <= 0 or abs(float(fps - inferred)) > 1e-7:
        raise ValueError("candidate_external_sampler_fps_pts_mismatch")
    try:
        sampler_sha = hashlib.sha256(_canonical(dict(sampler_row))).hexdigest()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("candidate_external_sampler_row_not_canonical") from None
    return dict(sampler_row), raw_indices, pts, decoded, fps, sampler_sha


def _validate_inventory(rows: Any, expected_rows: Any, sampler_rows: Any,
                        tokenizer_sha256: str) -> list[dict[str, Any]]:
    if (not isinstance(rows, (list, tuple)) or len(rows) != ROW_COUNT
            or not isinstance(expected_rows, (list, tuple))
            or len(expected_rows) != ROW_COUNT
            or not isinstance(sampler_rows, (list, tuple)) or len(sampler_rows) != ROW_COUNT):
        raise ValueError("candidate_three_ordered_rows_required")
    prepared = []
    tokens, sources = set(), set()
    for index, (row, expected, sampler_input) in enumerate(
            zip(rows, expected_rows, sampler_rows, strict=True), 1):
        if not isinstance(row, Mapping) or set(row) != ROW_KEYS:
            raise ValueError(f"candidate_row_{index}_schema_invalid")
        if not isinstance(expected, Mapping) or set(expected) != PIN_KEYS:
            raise ValueError(f"candidate_expected_row_{index}_schema_invalid")
        token, source = row["review_token"], _sha(row["source_video_sha256"])
        if (not isinstance(token, str) or not token or token in tokens
                or token != expected["review_token"] or source in sources
                or source != _sha(expected["source_video_sha256"])):
            raise ValueError("candidate_order_or_source_binding_mismatch")
        tokens.add(token)
        sources.add(source)
        observed = _source_pins(row)
        for field, digest in observed.items():
            if digest != _sha(expected[field]):
                raise ValueError("candidate_source_text_binding_mismatch")
        intervals = row["reviewed_intervals_seconds"]
        if (not isinstance(intervals, (list, tuple)) or not intervals
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       for pair in intervals)):
            raise ValueError("candidate_review_intervals_invalid")
        interval_rows = [[str(Fraction(str(x))) for x in pair] for pair in intervals]
        interval_digest = hashlib.sha256(_canonical(interval_rows)).hexdigest()
        if interval_digest != _sha(expected["reviewed_intervals_sha256"]):
            raise ValueError("candidate_review_interval_binding_mismatch")
        if not isinstance(row["frames"], (list, tuple)) or len(row["frames"]) != 32:
            raise ValueError("candidate_sampler_inventory_invalid")
        _sampler, indices, pts, decoded, fps, sampler_sha = _sampler_binding(
            sampler_input, source_sha256=source)
        observed_rgb = []
        try:
            for frame in row["frames"]:
                rgb_bytes = frame.convert("RGB").tobytes()
                if not isinstance(rgb_bytes, bytes) or not rgb_bytes:
                    raise ValueError
                observed_rgb.append(hashlib.sha256(rgb_bytes).hexdigest())
        except Exception:
            raise ValueError("candidate_rgb_frame_bytes_invalid") from None
        if observed_rgb != sampler_input["sampled_rgb_sha256"]:
            raise ValueError("candidate_rgb_frame_sampler_hash_mismatch")
        if sampler_sha != _sha(expected["sampler_row_sha256"]):
            raise ValueError("candidate_sampler_row_pin_mismatch")
        for field, value in (("tokenizer_sha256", tokenizer_sha256),
                             ("processor_revision", row["processor_revision"]),
                             ("processor_fingerprint_sha256", row["processor_fingerprint_sha256"])):
            expected_value = _sha(expected[field]) if field != "processor_revision" else expected[field]
            if value != expected_value:
                raise ValueError(f"candidate_{field}_pin_mismatch")
        for field in ("video_token_id", "spatial_merge_size",
                      "temporal_patch_size"):
            if type(row[field]) is not int or row[field] <= 0:
                raise ValueError("candidate_processor_geometry_invalid")
        if (not isinstance(row["processor_revision"], str) or not row["processor_revision"]
                or not isinstance(row["processor_fingerprint_sha256"], str)
                or not SHA256.fullmatch(row["processor_fingerprint_sha256"])):
            raise ValueError("candidate_processor_pin_invalid")
        try:
            duration = Fraction(str(row["video_duration_seconds"]))
            guard = Fraction(str(row["interval_boundary_guard_seconds"]))
            if duration <= 0 or guard < 0:
                raise ValueError
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            raise ValueError("candidate_duration_or_guard_invalid") from None
        if (str(duration) != expected["video_duration_seconds"]
                or str(guard) != expected["interval_boundary_guard_seconds"]):
            raise ValueError("candidate_duration_or_guard_pin_mismatch")
        prepared.append({**dict(row), "_reviewed_intervals_sha256": interval_digest,
                         "_video_duration_rational_seconds": str(duration),
                         "_guard_rational_seconds": str(guard),
                         "_sampler_indices": list(indices),
                         "_sampler_pts": pts, "_source_frame_count": decoded,
                         "_sample_fps": fps, "_sampler_row_sha256": sampler_sha})
    return prepared


def build_actual_processor_row_candidate(*, rows: Any, expected_rows: Any,
                                         sampler_rows: Any,
                                         processor: Any, tokenizer: Any,
                                         tokenizer_sha256: str,
                                         vision_info_fn: Any,
                                         torch_module: Any) -> dict[str, Any]:
    """Build exactly three ordered CPU processor-row candidates.

    Row frames are already-sampled decoded RGB objects supplied by the caller;
    no media is opened here. The actual runtime must separately verify their
    source provenance and every review/release/clock gate.
    """
    _sha(tokenizer_sha256)
    if (processor is None or not callable(processor) or tokenizer is None
            or not callable(vision_info_fn) or torch_module is None):
        raise ValueError("candidate_processor_dependencies_invalid")
    # build_qa_prompt invokes processor.apply_chat_template, which renders with
    # processor.tokenizer. Refuse a separately supplied tokenizer before any
    # processor/template call so token IDs and processor input cannot diverge.
    if getattr(processor, "tokenizer", None) is not tokenizer:
        raise ValueError("candidate_processor_tokenizer_identity_mismatch")
    prepared = _validate_inventory(rows, expected_rows, sampler_rows, tokenizer_sha256)

    output_rows = []
    for row in prepared:
        fps = row["_sample_fps"]
        text, messages = build_qa_prompt(
            processor, {"question": row["question"], "choices": list(row["choices"])},
            row["frames"], (fps.numerator, fps.denominator))
        choice = build_choice_token_binding(
            tokenizer=tokenizer, rendered_prompt=text,
            tokenizer_sha256=tokenizer_sha256)
        vision_result = vision_info_fn(messages, return_video_metadata=True)
        if not isinstance(vision_result, (list, tuple)) or len(vision_result) != 2:
            raise ValueError("candidate_vision_result_invalid")
        images, videos = vision_result
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            raise ValueError("candidate_video_payload_invalid")
        pair = videos[0]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("candidate_video_metadata_pair_invalid")
        pixels, video_metadata = pair
        batch = process_qa_video(processor, text, images, pixels, video_metadata)
        if not isinstance(batch, Mapping):
            raise ValueError("candidate_processor_batch_invalid")
        batch = {key: value.detach().clone() if torch_module.is_tensor(value) else value
                 for key, value in batch.items()}
        pts = row["_sampler_pts"]
        mapping = map_review_interval_from_processor_batch(
            batch=batch, torch_module=torch_module,
            video_token_id=row["video_token_id"],
            spatial_merge_size=row["spatial_merge_size"],
            temporal_patch_size=row["temporal_patch_size"],
            sampled_frame_indices=row["_sampler_indices"],
            sampled_pts_seconds=pts,
            source_frame_count=row["_source_frame_count"],
            video_duration_seconds=row["video_duration_seconds"],
            sample_fps=float(fps), processor_revision=row["processor_revision"],
            processor_fingerprint_sha256=row["processor_fingerprint_sha256"],
            reviewed_intervals_seconds=row["reviewed_intervals_seconds"],
            interval_boundary_guard_seconds=row["interval_boundary_guard_seconds"])
        # The mapper's returned digest commits the sampled indices, but its
        # result object currently omits that list. The row projection contract
        # requires the list for its consistency check, so reattach the exact
        # already-passed caller vector without altering any mapper output.
        mapping_for_row = {
            **mapping,
            "sampled_frame_indices": list(row["_sampler_indices"]),
            "video_grid_thw": [int(value) for value in
                               batch["video_grid_thw"][0].tolist()],
        }
        sampler_row = {
            "source_sha256": row["source_video_sha256"],
            "decoded_frame_count": row["_source_frame_count"],
            "sampled_frame_count": 32,
            "sampled_indices": list(row["_sampler_indices"]),
            "sampled_pts": [[Fraction(str(value)).numerator,
                             Fraction(str(value)).denominator] for value in pts],
        }
        # Keep the exact sampler row's optional metadata (including sampled RGB
        # digests) in the row hash pin supplied by the caller. The existing
        # projection consumes only the normalized structural fields above.
        processor_row = processor_binding_row(
            review_token=row["review_token"],
            source_video_sha256=row["source_video_sha256"],
            mapping=mapping_for_row, sampler_row=sampler_row)
        input_ids = batch.get("input_ids")
        if not torch_module.is_tensor(input_ids):
            raise ValueError("candidate_input_ids_missing")
        qpos = map_question_token_positions(
            tokenizer=tokenizer, rendered_text=text, question_text=row["question"],
            processed_input_ids=input_ids, video_token_id=row["video_token_id"])
        # Verify tokenizer.encode (used for A-E continuation binding) and the
        # offset-tokenization path agree, then check the processor's exact
        # processed IDs against that prompt with only the video placeholder
        # expanded. This costs no additional processor/model call.
        encoded_prompt = tokenizer.encode(text, add_special_tokens=False)
        tokenized_prompt = tokenizer(text, add_special_tokens=False,
                                     return_offsets_mapping=True).get("input_ids")
        if (not isinstance(encoded_prompt, list) or encoded_prompt != tokenized_prompt
                or encoded_prompt.count(row["video_token_id"]) != 1):
            raise ValueError("candidate_prompt_tokenizer_paths_disagree")
        placeholder = encoded_prompt.index(row["video_token_id"])
        processed_ids = input_ids[0].detach().cpu().tolist()
        visual_positions = [i for i, value in enumerate(processed_ids)
                            if value == row["video_token_id"]]
        if (not visual_positions or visual_positions != list(range(
                visual_positions[0], visual_positions[-1] + 1))
                or processed_ids != (encoded_prompt[:placeholder]
                    + [row["video_token_id"]] * len(visual_positions)
                    + encoded_prompt[placeholder + 1:])):
            raise ValueError("candidate_processed_prompt_suffix_mismatch")
        output_rows.append({
            **processor_row,
            **_source_pins(row),
            "reviewed_intervals_sha256": row["_reviewed_intervals_sha256"],
            "video_duration_seconds": row["_video_duration_rational_seconds"],
            "interval_boundary_guard_seconds": row["_guard_rational_seconds"],
            "tokenizer_sha256": tokenizer_sha256,
            "processor_revision": row["processor_revision"],
            "processor_fingerprint_sha256": row["processor_fingerprint_sha256"],
            "sampler_row_sha256": row["_sampler_row_sha256"],
            "rendered_prompt_sha256": choice["prompt_sha256"],
            "prompt_token_ids_sha256": choice["prompt_token_ids_sha256"],
            "choice_token_binding_sha256": choice["binding_sha256"],
            "answer_token_ids": list(choice["answer_token_ids"]),
            "question_positions": qpos,
            "input_ids_sha256": hashlib.sha256(_canonical(
                input_ids[0].detach().cpu().tolist())).hexdigest(),
            "batch": batch,
        })
    row_hashes = [hashlib.sha256(_canonical(_candidate_row_commitment(row))).hexdigest()
                  for row in output_rows]
    return {
        "schema_version": 1,
        "purpose": "qta_actual_processor_rows_candidate_only",
        "status": "three_processor_rows_structurally_bound_candidate",
        "requested_rows": ROW_COUNT,
        "candidate_rows": output_rows,
        "ordered_candidate_rows_sha256": hashlib.sha256(
            _canonical(row_hashes)).hexdigest(),
        "prompt_answer_token_bindings_raw": build_qta_prompt_answer_token_bindings(output_rows),
        "integrity_commitment_scope": "internal_self_consistency_not_authentication",
        "external_validation_required": True,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "source_authentication_status": "not_performed",
        "faithfulness_scored": False,
        "limitations": [
            "candidate structure only; source, sampler, reviewer, and clock provenance are not authenticated",
            "no model is loaded or called; this output is not scorer authorization",
        ],
    }


def verify_actual_processor_row_candidate_self_consistency(candidate: Any) -> None:
    """Check only internal integrity; this is not an external pin verifier."""
    if (not isinstance(candidate, Mapping)
            or type(candidate.get("schema_version")) is not int
            or candidate.get("schema_version") != 1
            or candidate.get("purpose") != "qta_actual_processor_rows_candidate_only"
            or candidate.get("status") != "three_processor_rows_structurally_bound_candidate"
            or candidate.get("model_load_permitted") is not False
            or candidate.get("scoring_authorized") is not False
            or candidate.get("execution_authorization") != "not_granted"
            or candidate.get("human_identity_verified") is not False
            or candidate.get("semantic_review_authenticated") is not False
            or candidate.get("source_authentication_status") != "not_performed"
            or candidate.get("faithfulness_scored") is not False
            or type(candidate.get("requested_rows")) is not int
            or candidate.get("requested_rows") != ROW_COUNT
            or candidate.get("integrity_commitment_scope")
            != "internal_self_consistency_not_authentication"
            or candidate.get("external_validation_required") is not True):
        raise ValueError("candidate_status_or_authority_invalid")
    rows = candidate.get("candidate_rows")
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ValueError("candidate_row_count_invalid")
    tokens, sources = set(), set()
    row_hashes = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate_row_invalid")
        token = row.get("review_token")
        source = _sha(row.get("source_video_sha256"))
        if not isinstance(token, str) or not token or token in tokens or source in sources:
            raise ValueError("candidate_order_or_source_binding_mismatch")
        tokens.add(token)
        sources.add(source)
        if batch_sha256(row.get("batch")) != row.get("typed_batch_sha256"):
            raise ValueError("candidate_batch_digest_mismatch")
        if len(row.get("answer_token_ids", [])) != 5 or len(set(row["answer_token_ids"])) != 5:
            raise ValueError("candidate_answer_token_binding_invalid")
        row_hashes.append(hashlib.sha256(_canonical(_candidate_row_commitment(row))).hexdigest())
    expected_commitment = hashlib.sha256(_canonical(row_hashes)).hexdigest()
    if candidate.get("ordered_candidate_rows_sha256") != expected_commitment:
        raise ValueError("candidate_order_or_row_commitment_mismatch")
    if candidate.get("prompt_answer_token_bindings_raw") != build_qta_prompt_answer_token_bindings(rows):
        raise ValueError("candidate_prompt_answer_binding_mismatch")


def verify_actual_processor_row_candidate(candidate: Any, *, expected_rows: Any) -> None:
    """Check candidate fields against an independently supplied ordered pin set.

    The caller must obtain these pins from an independently validated frozen
    source/sampler/protocol bundle. This function does not validate that bundle.
    """
    verify_actual_processor_row_candidate_self_consistency(candidate)
    if not isinstance(expected_rows, (list, tuple)) or len(expected_rows) != ROW_COUNT:
        raise ValueError("candidate_expected_rows_required")
    rows = candidate["candidate_rows"]
    for row, expected in zip(rows, expected_rows, strict=True):
        if not isinstance(expected, Mapping) or set(expected) != PIN_KEYS:
            raise ValueError("candidate_expected_row_schema_invalid")
        if any(row.get(field) != expected.get(field) for field in PIN_KEYS):
            raise ValueError("candidate_external_expected_pin_mismatch")
