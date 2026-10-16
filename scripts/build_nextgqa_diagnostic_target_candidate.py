#!/usr/bin/env python3
"""Build a private, diagnostic-only NExT-GQA target candidate without scoring.

This CPU-only command binds the current claim packet, lineage, worksheet, local
CSV, pinned processor, and production QA prompt path. It never loads model
weights, calls a model forward, computes a score, or grants authorization.
Every input path, including the output path, is required explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any

from audit_nextgqa_prompt_tokenizer import MODEL_REVISION, validate_answer_label_tokens
from build_nextgqa_v2_claim_input_bundle import (
    ALL_RELEVANT_OCCURRENCES, V2_COHORT_SHA256, V5_POLICY_ID, V5_POLICY_SHA256,
)
from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from private_output_paths import validate_private_output
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video

LABELS = list("ABCDE")
FRAME_COUNT = 32
UNIFORM_TOLERANCE = (1, 20)
SHA256 = re.compile(r"^[a-f0-9]{64}$")
GENERATION_PURPOSE = "private_local_nextgqa_model_generated_claim_review_packet"
LINEAGE_PURPOSE = "private_source_lineage_crosscheck_for_claim_review"
WORKSHEET_PURPOSE = "private_blinded_claim_to_evidence_mapping_worksheet"
# Exact current draft bytes for the diagnostic target candidate only. This is
# not a protocol freeze or scoring authorization; bump it when the draft text
# changes, and freeze the complete artifact set separately before any scoring.
PROTOCOL_V2_DRAFT_SHA256 = "370d4735951b8d648cac9584ff969805f5f47778d707fa5e4838534bb47f13d4"
PROTOCOL_V2_TITLE = b"# NExT-GQA finite-cohort claim-aligned input-dependence protocol v2"
PROTOCOL_V2_DRAFT_STATUS = b"Status: **design draft; not frozen, not scoreable, and not an execution"
PROTOCOL_V2_FRAME_RULE = b"Use 32 endpoint-inclusive uniformly sampled decoded-frame indices."
CLAIM_MANIFEST_KEYS = {"schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
                       "claim_policy_sha256", "occurrence_policy", "claims"}
CLAIM_KEYS = {"video_id", "claim_id", "claim_text", "claim_text_sha256",
              "model_revision", "generation_run_sha256"}
CLAIM_SIDECAR_KEYS = {
    "schema_version", "purpose", "status", "scope", "review_completed", "scoring_ready",
    "execution_authorization", "eligible_dataset_id", "cohort_sha256", "cohort_digest_check",
    "occurrence_policy", "policy_id", "policy_sha256", "model", "model_revision",
    "v1_generation_run_sha256_semantics", "worksheet_binding_check", "input_sha256",
    "claim_input_manifest_sha256", "records",
}
CLAIM_SIDECAR_INPUT_KEYS = {"eligible_manifest", "generation_packet", "lineage_manifest",
                            "policy", "test_csv", "video_map", "worksheet"}
CLAIM_SIDECAR_ROW_KEYS = {"video_id", "dataset_video_id", "dataset_question_id", "video_sha256",
                          "question_sha256", "claim_id", "claim_text_sha256"}

# Repo-relative source closure for this diagnostic candidate builder. This is
# deliberately not a frozen production scoring-code set.
DIAGNOSTIC_SOURCE_CODE_PATHS = (
    ("builder", "scripts/build_nextgqa_diagnostic_target_candidate.py"),
    ("runner", "scripts/run_nextgqa_gold_interval_margin_pilot.py"),
    ("decoder", "scripts/claim_mask_video_decoder.py"),
    ("frame_sampling_utils", "scripts/necessity_choice_margin_utils.py"),
    ("presentation_time_utils", "scripts/presentation_time_utils.py"),
    ("source_adapter", "scripts/nextgqa_actual_qa_source_adapter.py"),
    ("claim_bundle_validator", "scripts/build_nextgqa_v2_claim_input_bundle.py"),
    ("tokenizer_audit", "scripts/audit_nextgqa_prompt_tokenizer.py"),
)


class CandidateError(ValueError):
    """A sanitized, stable failure code safe for aggregate CLI output."""


def _fail(code: str) -> None:
    raise CandidateError(code)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(raw: bytes, code: str) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(code)
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda _: _fail(code))
    except CandidateError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        _fail(code)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _validate_protocol_v2(protocol_raw: bytes, frame_count: int = FRAME_COUNT) -> None:
    """Require the exact current v2 draft and its declared 32-frame rule."""
    if not isinstance(protocol_raw, bytes):
        _fail("protocol_bytes_invalid")
    if (PROTOCOL_V2_TITLE not in protocol_raw
            or PROTOCOL_V2_DRAFT_STATUS not in protocol_raw
            or PROTOCOL_V2_FRAME_RULE not in protocol_raw):
        _fail("protocol_v2_contract_invalid")
    if _sha(protocol_raw) != PROTOCOL_V2_DRAFT_SHA256:
        _fail("protocol_v2_identity_mismatch")
    if type(frame_count) is not int or frame_count != FRAME_COUNT:
        _fail("protocol_frame_count_mismatch")


def _validate_upstream(*, csv_raw: bytes, generation_raw: bytes,
                       lineage_raw: bytes, worksheet_raw: bytes,
                       video_root: Path) -> tuple[dict, dict, dict, dict]:
    """Verify exact candidate packet/lineage/worksheet joins before media decode."""
    generation = _strict_json(generation_raw, "generation_json_invalid")
    lineage = _strict_json(lineage_raw, "lineage_json_invalid")
    worksheet = _strict_json(worksheet_raw, "worksheet_json_invalid")
    if (not isinstance(generation, dict)
            or generation.get("purpose") != GENERATION_PURPOSE
            or generation.get("status") != "passed"
            or generation.get("requested_videos") != 3
            or generation.get("effective_videos") != 3
            or not isinstance(generation.get("records"), list)
            or len(generation["records"]) != 3):
        _fail("generation_packet_invalid")
    if (not isinstance(lineage, dict)
            or lineage.get("purpose") != LINEAGE_PURPOSE
            or lineage.get("status") != "metadata_lineage_crosschecked"
            or lineage.get("requested_records") != 3
            or lineage.get("effective_records") != 3
            or not isinstance(lineage.get("records"), list)
            or len(lineage["records"]) != 3):
        _fail("lineage_invalid")
    if (not isinstance(worksheet, dict)
            or worksheet.get("purpose") != WORKSHEET_PURPOSE
            or worksheet.get("requested_records") != 3
            or worksheet.get("effective_records") != 3
            or worksheet.get("claim_records_available") != 3
            or worksheet.get("execution_authorization") != "not_granted"
            or not isinstance(worksheet.get("review_completed"), bool)
            or not isinstance(worksheet.get("records"), list)
            or len(worksheet["records"]) != 3):
        _fail("worksheet_invalid")
    generation_sha, lineage_sha = _sha(generation_raw), _sha(lineage_raw)
    if (not _is_sha(generation.get("cohort_sha256"))
            or generation.get("cohort_sha256") != lineage.get("cohort_sha256")
            or lineage.get("cohort_sha256") != worksheet.get("cohort_sha256")
            or lineage.get("source_packet_sha256") != generation_sha
            or worksheet.get("source_packet_sha256") != generation_sha
            or generation.get("dataset_inputs_sha256") != lineage.get("dataset_inputs_sha256")
            or lineage.get("dataset_inputs_sha256") != worksheet.get("dataset_inputs_sha256")
            or lineage["dataset_inputs_sha256"].get("test.csv") != _sha(csv_raw)):
        _fail("cohort_or_source_manifest_mismatch")

    generated_rows, lineage_rows, worksheet_rows = (
        generation["records"], lineage["records"], worksheet["records"])
    selection_rows = []
    for generated, source, sheet in zip(generated_rows, lineage_rows, worksheet_rows, strict=True):
        if not all(isinstance(row, dict) for row in (generated, source, sheet)):
            _fail("row_schema_invalid")
        parsed = generated.get("generated_parsed")
        claim = parsed.get("claim") if isinstance(parsed, dict) else None
        qid = generated.get("dataset_question_id")
        video_sha = generated.get("video_sha256")
        question = generated.get("question")
        if (not isinstance(claim, str) or not claim.strip()
                or not isinstance(qid, str) or not qid.strip()
                or not isinstance(video_sha, str) or not _is_sha(video_sha)
                or not isinstance(question, str) or not question.strip()):
            _fail("generation_row_invalid")
        claim_sha = _sha(claim.encode("utf-8"))
        question_sha = _sha(question.encode("utf-8"))
        video_id = "video-" + video_sha
        claim_id = "claim-" + _sha(
            (video_sha + "\0" + qid + "\0" + claim_sha).encode("utf-8"))
        if (source.get("video_id") != video_id or sheet.get("video_id") != video_id
                or source.get("claim_id") != claim_id or sheet.get("claim_id") != claim_id
                or source.get("dataset_question_id") != qid
                or source.get("video_sha256") != video_sha
                or sheet.get("video_sha256") != video_sha
                or source.get("claim_text_sha256") != claim_sha
                or sheet.get("claim_text_sha256") != claim_sha
                or sheet.get("claim_text") != claim
                or source.get("question_sha256") != question_sha
                or sheet.get("question") != question
                or _sha(sheet.get("question", "").encode("utf-8")) != question_sha):
            _fail("generation_lineage_worksheet_row_mismatch")
        relative_text = source.get("relative_video_path")
        if not isinstance(relative_text, str) or "\\" in relative_text:
            _fail("lineage_media_path_invalid")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            _fail("lineage_media_path_invalid")
        media_path = video_root.joinpath(*relative.parts)
        if not media_path.is_file() or _file_sha(media_path) != video_sha:
            _fail("lineage_media_bytes_mismatch")
        packet_video = generated.get("local_video_path")
        if not isinstance(packet_video, str) or not Path(packet_video).is_file():
            _fail("generation_media_missing")
        if _file_sha(Path(packet_video)) != video_sha:
            _fail("generation_media_bytes_mismatch")
        selection_rows.append({
            "video_id": video_id, "claim_id": claim_id,
            "dataset_video_id": source.get("dataset_video_id"),
            "dataset_question_id": qid,
        })

    generation_order = [row.get("video_id") for row in worksheet_rows]
    if (len(set(generation_order)) != 3
            or [row.get("video_id") for row in lineage_rows] != generation_order
            or ["video-" + str(row.get("video_sha256")) for row in generated_rows]
            != generation_order):
        _fail("frozen_row_order_mismatch")
    selection = {"records": selection_rows}
    target = {"records": [{"video_id": row["video_id"], "claim_id": row["claim_id"]}
                           for row in selection_rows]}
    # `lineage_sha` is intentionally evaluated here so malformed/non-byte input
    # cannot be silently skipped by a future refactor.
    if not _is_sha(lineage_sha):
        _fail("lineage_digest_invalid")
    return generation, lineage, worksheet, {"selection": selection, "target": target}


def _validate_claim_input_bundle(*, manifest_raw: bytes, sidecar_raw: bytes,
                                 csv_raw: bytes, generation_raw: bytes,
                                 lineage_raw: bytes, worksheet_raw: bytes,
                                 generation: dict, lineage: dict, worksheet: dict) -> str:
    """Bind this diagnostic QA target to the exact candidate claim packet.

    This is structural byte binding only. It does not authenticate eligible
    cohort provenance, custody, semantic review, or execution permission.
    """
    manifest = _strict_json(manifest_raw, "claim_manifest_json_invalid")
    sidecar = _strict_json(sidecar_raw, "claim_sidecar_json_invalid")
    if (not isinstance(manifest, dict) or set(manifest) != CLAIM_MANIFEST_KEYS
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or not isinstance(manifest.get("eligible_dataset_id"), str)
            or not manifest["eligible_dataset_id"].strip()
            or not _is_sha(manifest.get("eligible_manifest_sha256"))
            or manifest.get("claim_policy_sha256") != V5_POLICY_SHA256
            or manifest.get("occurrence_policy") != ALL_RELEVANT_OCCURRENCES
            or not isinstance(manifest.get("claims"), list)
            or len(manifest["claims"]) != 3):
        _fail("claim_manifest_schema_invalid")
    if (not isinstance(sidecar, dict) or set(sidecar) != CLAIM_SIDECAR_KEYS
            or type(sidecar.get("schema_version")) is not int or sidecar["schema_version"] != 1
            or sidecar.get("purpose") != "private_nextgqa_v2_claim_input_bundle_provenance"
            or sidecar.get("status") != "candidate_unreviewed"
            or sidecar.get("scope") != "structural_only"
            or sidecar.get("review_completed") is not False
            or sidecar.get("scoring_ready") is not False
            or sidecar.get("execution_authorization") != "not_granted"
            or sidecar.get("eligible_dataset_id") != manifest.get("eligible_dataset_id")
            or sidecar.get("cohort_sha256") != V2_COHORT_SHA256
            or generation.get("cohort_sha256") != V2_COHORT_SHA256
            or lineage.get("cohort_sha256") != V2_COHORT_SHA256
            or worksheet.get("cohort_sha256") != V2_COHORT_SHA256
            or sidecar.get("cohort_digest_check") !=
                "pinned_value_and_input_declarations_match_not_recomputed_from_full_planner_inputs"
            or sidecar.get("occurrence_policy") != ALL_RELEVANT_OCCURRENCES
            or sidecar.get("policy_id") != V5_POLICY_ID
            or sidecar.get("policy_sha256") != V5_POLICY_SHA256
            or sidecar.get("model") != "Qwen/Qwen2.5-VL-7B-Instruct"
            or sidecar.get("model_revision") != MODEL_REVISION
            or sidecar.get("v1_generation_run_sha256_semantics") !=
                "generation_packet_container_byte_digest_not_unique_run_id"
            or sidecar.get("worksheet_binding_check") !=
                "claim_and_identity_hashes_matched_only_not_review_or_custody_validation"
            or sidecar.get("claim_input_manifest_sha256") != _sha(manifest_raw)):
        _fail("claim_sidecar_binding_invalid")
    inputs = sidecar.get("input_sha256")
    expected_inputs = {
        "eligible_manifest": manifest["eligible_manifest_sha256"],
        "generation_packet": _sha(generation_raw),
        "lineage_manifest": _sha(lineage_raw),
        "policy": V5_POLICY_SHA256,
        "test_csv": _sha(csv_raw),
        "video_map": lineage["dataset_inputs_sha256"].get("map_vid_vidorID.json"),
        "worksheet": _sha(worksheet_raw),
    }
    if (not isinstance(inputs, dict) or set(inputs) != CLAIM_SIDECAR_INPUT_KEYS
            or inputs != expected_inputs
            or not all(_is_sha(value) for value in inputs.values())):
        _fail("claim_sidecar_input_hashes_mismatch")
    generation_by_video = {
        "video-" + row["video_sha256"]: row for row in generation["records"]
        if isinstance(row, dict) and _is_sha(row.get("video_sha256"))
    }
    worksheet_by_video = {row["video_id"]: row for row in worksheet["records"]
                          if isinstance(row, dict) and isinstance(row.get("video_id"), str)}
    lineage_by_video = {row["video_id"]: row for row in lineage["records"]
                        if isinstance(row, dict) and isinstance(row.get("video_id"), str)}
    if (not isinstance(sidecar.get("records"), list) or len(sidecar["records"]) != 3
            or len(generation_by_video) != 3 or len(worksheet_by_video) != 3
            or len(lineage_by_video) != 3):
        _fail("claim_sidecar_record_count_invalid")
    claim_by_video = {}
    for claim in manifest["claims"]:
        if not isinstance(claim, dict) or set(claim) != CLAIM_KEYS:
            _fail("claim_manifest_row_schema_invalid")
        video_id = claim.get("video_id")
        generated = generation_by_video.get(video_id)
        sheet = worksheet_by_video.get(video_id)
        source = lineage_by_video.get(video_id)
        parsed = generated.get("generated_parsed") if isinstance(generated, dict) else None
        text = parsed.get("claim") if isinstance(parsed, dict) else None
        if (not isinstance(video_id, str) or video_id in claim_by_video
                or not isinstance(text, str)
                or claim.get("claim_id") != sheet.get("claim_id")
                or claim.get("claim_text") != text
                or claim.get("claim_text") != sheet.get("claim_text")
                or claim.get("claim_text_sha256") != _sha(text.encode("utf-8"))
                or claim.get("claim_text_sha256") != sheet.get("claim_text_sha256")
                or claim.get("model_revision") != "Qwen/Qwen2.5-VL-7B-Instruct@" + MODEL_REVISION
                or claim.get("generation_run_sha256") != _sha(generation_raw)
                or source is None or source.get("claim_id") != claim.get("claim_id")
                or source.get("claim_text_sha256") != claim.get("claim_text_sha256")):
            _fail("claim_manifest_lineage_join_mismatch")
        claim_by_video[video_id] = claim
    sidecar_by_video = {}
    for row in sidecar["records"]:
        if not isinstance(row, dict) or set(row) != CLAIM_SIDECAR_ROW_KEYS:
            _fail("claim_sidecar_record_schema_invalid")
        video_id = row.get("video_id")
        if video_id in sidecar_by_video:
            _fail("claim_sidecar_duplicate_video")
        sidecar_by_video[video_id] = row
        source = lineage_by_video.get(video_id)
        claim = claim_by_video.get(video_id)
        if (source is None or claim is None
                or row.get("dataset_video_id") != source.get("dataset_video_id")
                or row.get("dataset_question_id") != source.get("dataset_question_id")
                or row.get("video_sha256") != source.get("video_sha256")
                or row.get("question_sha256") != source.get("question_sha256")
                or row.get("claim_id") != claim.get("claim_id")
                or row.get("claim_text_sha256") != claim.get("claim_text_sha256")):
            _fail("claim_sidecar_lineage_join_mismatch")
    if set(claim_by_video) != set(lineage_by_video) or set(sidecar_by_video) != set(lineage_by_video):
        _fail("claim_bundle_cohort_mismatch")
    return _sha(manifest_raw)


def _checkpoint_fingerprint(model_path: Path) -> tuple[str, str]:
    """Hash exact indexed local weights and config, without loading any weights."""
    if model_path.name != MODEL_REVISION or not model_path.is_dir():
        _fail("pinned_model_snapshot_invalid")
    config = model_path / "config.json"
    index = model_path / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        _fail("checkpoint_identity_inputs_missing")
    config_sha = _file_sha(config)
    index_raw = index.read_bytes()
    index_doc = _strict_json(index_raw, "checkpoint_index_invalid")
    weight_map = index_doc.get("weight_map") if isinstance(index_doc, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        _fail("checkpoint_index_invalid")
    shards = sorted(set(weight_map.values()))
    if (not shards or any(not isinstance(name, str) or not name.endswith(".safetensors")
                          or PurePosixPath(name).name != name for name in shards)):
        _fail("checkpoint_shard_names_invalid")
    shard_rows = []
    for name in shards:
        path = model_path / name
        if not path.is_file():
            _fail("checkpoint_shard_missing")
        shard_rows.append({"name": name, "bytes": path.stat().st_size,
                           "sha256": _file_sha(path)})
    fingerprint = _sha(_canonical({
        "model_revision": MODEL_REVISION, "config_sha256": config_sha,
        "index_sha256": _sha(index_raw), "weight_shards": shard_rows,
    }))
    return fingerprint, config_sha


def _hash_file_set(root: Path, required: tuple[str, ...], optional: tuple[str, ...],
                   code: str) -> str:
    rows = []
    for name in required:
        path = root / name
        if not path.is_file():
            _fail(code)
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": _file_sha(path)})
    for name in optional:
        path = root / name
        if path.is_file():
            rows.append({"name": name, "bytes": path.stat().st_size, "sha256": _file_sha(path)})
    return _sha(_canonical(rows))


def _diagnostic_source_code_hashes(repo: Path) -> dict[str, dict[str, Any]]:
    """Hash the explicit local source closure used by this diagnostic builder."""
    result = {}
    try:
        repo_real = repo.resolve(strict=True)
    except OSError:
        _fail("diagnostic_source_repo_invalid")
    for role, relative_name in DIAGNOSTIC_SOURCE_CODE_PATHS:
        path = repo / relative_name
        try:
            expected = repo_real / relative_name
            resolved = path.resolve(strict=True)
            info = path.lstat()
        except OSError:
            _fail("diagnostic_source_file_missing")
        if (resolved != expected or stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)):
            _fail("diagnostic_source_file_invalid")
        fd = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            opened = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if ((opened.st_dev, opened.st_ino) != identity
                    or not stat.S_ISREG(opened.st_mode)
                    or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                    != (info.st_size, info.st_mtime_ns, info.st_ctime_ns)):
                _fail("diagnostic_source_file_changed")
            digest_state = hashlib.sha256()
            byte_count = 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                digest_state.update(block)
                byte_count += len(block)
            finished = os.fstat(fd)
            after_path = path.lstat()
            after_resolved = path.resolve(strict=True)
            stable = (finished.st_dev, finished.st_ino) == identity
            stable = stable and (after_path.st_dev, after_path.st_ino) == identity
            stable = stable and after_resolved == expected
            stable = stable and stat.S_ISREG(finished.st_mode) and stat.S_ISREG(after_path.st_mode)
            stable = stable and (finished.st_size, finished.st_mtime_ns, finished.st_ctime_ns) == (
                opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            stable = stable and (after_path.st_size, after_path.st_mtime_ns,
                                 after_path.st_ctime_ns) == (
                                     opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            stable = stable and byte_count == opened.st_size
            if not stable:
                _fail("diagnostic_source_file_changed")
            digest = digest_state.hexdigest()
        except OSError:
            _fail("diagnostic_source_file_unreadable")
        finally:
            if fd is not None:
                os.close(fd)
        result[role] = {
            "path": relative_name,
            "bytes": byte_count,
            "sha256": digest,
        }
    return result


def _pyav_library_versions() -> dict[str, list[int] | str] | None:
    """Return JSON-safe PyAV linked-library versions when the API exposes them."""
    import av

    library_versions = getattr(av, "library_versions", None)
    if not isinstance(library_versions, dict):
        return None
    result = {}
    for name, version in sorted(library_versions.items(), key=lambda item: str(item[0])):
        if (isinstance(version, (tuple, list))
                and all(type(component) is int for component in version)):
            result[str(name)] = list(version)
        else:
            result[str(name)] = str(version)
    return result


def _fingerprints(*, model_path: Path, protocol_raw: bytes,
                  dataset_source_inputs: dict, script_path: Path) -> dict:
    checkpoint_sha, model_config_sha = _checkpoint_fingerprint(model_path)
    tokenizer_sha = _hash_file_set(
        model_path, ("tokenizer.json", "tokenizer_config.json"),
        ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"),
        "tokenizer_files_missing")
    processor_sha = _hash_file_set(
        model_path, ("preprocessor_config.json",),
        ("processor_config.json", "chat_template.json"), "processor_files_missing")
    repo = Path(__file__).resolve().parents[1]
    expected_builder = (repo / DIAGNOSTIC_SOURCE_CODE_PATHS[0][1]).resolve()
    if script_path.resolve() != expected_builder:
        _fail("diagnostic_builder_source_path_mismatch")
    code_files = _diagnostic_source_code_hashes(repo)
    import inspect
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt as prompt_fn
    from run_nextgqa_gold_interval_margin_pilot import process_qa_video as process_fn
    prompt_sha = _sha((inspect.getsource(prompt_fn) + "\0" + inspect.getsource(process_fn)).encode())
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av",
                 "numpy", "Pillow")}
    pyav_libraries = _pyav_library_versions()
    return {
        "model_revision": MODEL_REVISION,
        "model_checkpoint_sha256": checkpoint_sha,
        "model_config_sha256": model_config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "processor_config_sha256": processor_sha,
        "prompt_builder_sha256": prompt_sha,
        "scoring_code_files_sha256": code_files,
        "scoring_code_sha256": _sha(_canonical({
            "files": code_files,
            "versions": versions,
            "pyav_library_versions": pyav_libraries,
        })),
        "scoring_code_hash_scope": (
            "diagnostic_builder_direct_repo_source_closure_not_frozen_production_set"),
        "scoring_protocol_sha256": _sha(protocol_raw),
        "dataset_source_manifest_sha256": _sha(_canonical(dataset_source_inputs)),
        "software_versions": versions,
        "pyav_library_versions": pyav_libraries,
    }


def build_candidate(*, csv_raw: bytes, generation_raw: bytes, lineage_raw: bytes,
                    worksheet_raw: bytes, protocol_raw: bytes, video_root: Path,
                    model_path: Path, processor, frame_count: int = FRAME_COUNT,
                    claim_input_manifest_raw: bytes | None = None,
                    claim_input_sidecar_raw: bytes | None = None,
                    script_path: Path | None = None, vision_info_fn=None,
                    decoder_fn=decode_sampled_video, verify_frames_fn=verify_sampled_frame_pixels) -> dict:
    """Create an unvalidated hash-only candidate payload; never score or authorize."""
    _validate_protocol_v2(protocol_raw, frame_count)
    generation, lineage, worksheet, upstream = _validate_upstream(
        csv_raw=csv_raw, generation_raw=generation_raw, lineage_raw=lineage_raw,
        worksheet_raw=worksheet_raw, video_root=video_root)
    if (claim_input_manifest_raw is None) != (claim_input_sidecar_raw is None):
        _fail("claim_bundle_pair_required")
    claim_input_sha = None
    if claim_input_manifest_raw is not None:
        claim_input_sha = _validate_claim_input_bundle(
            manifest_raw=claim_input_manifest_raw, sidecar_raw=claim_input_sidecar_raw,
            csv_raw=csv_raw, generation_raw=generation_raw, lineage_raw=lineage_raw,
            worksheet_raw=worksheet_raw, generation=generation, lineage=lineage,
            worksheet=worksheet)
    selection, target = upstream["selection"], upstream["target"]
    qa_source, binding = build_actual_qa_source(
        csv_raw=csv_raw, selection=selection,
        target_manifest_raw=_canonical(target), lineage_manifest_raw=lineage_raw)
    if vision_info_fn is None:
        from qwen_vl_utils import process_vision_info
        vision_info_fn = process_vision_info
    import torch
    import numpy as np
    from PIL import Image  # noqa: F401
    from fractions import Fraction
    if torch.cuda.is_available():
        _fail("cuda_visible_candidate_refused")

    token_ids = validate_answer_label_tokens(processor.tokenizer)
    if len(token_ids) != 5 or len(set(token_ids)) != 5:
        _fail("answer_label_tokens_invalid")
    lineage_by_video = {row["video_id"]: row for row in lineage["records"]}
    output_rows = []
    ordered_prompt_hashes, ordered_input_hashes = [], []
    for item in qa_source["records"]:
        lineage_row = lineage_by_video.get(item["video_id"])
        if not isinstance(lineage_row, dict):
            _fail("qa_lineage_row_missing")
        relative = PurePosixPath(lineage_row["relative_video_path"])
        media_path = video_root.joinpath(*relative.parts)
        frames, decode = decoder_fn(
            media_path, num_frames=frame_count,
            uniform_tolerance_seconds=Fraction(*UNIFORM_TOLERANCE))
        verify_frames_fn(frames, decode)
        if (len(frames) != frame_count or decode.get("source_sha256") != lineage_row["video_sha256"]):
            _fail("decoded_media_lineage_mismatch")
        sample_fps = decode["effective_sample_fps"]
        text, messages = build_qa_prompt(processor, item, frames, sample_fps)
        images, videos = vision_info_fn(messages, return_video_metadata=True)
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            _fail("qwen_video_utility_contract_failed")
        pixels, metadata = videos[0]
        if (not isinstance(pixels, torch.Tensor) or pixels.shape[0] != frame_count
                or metadata.get("frames_indices") != list(range(frame_count))
                or abs(float(metadata.get("fps", 0)) - sample_fps[0] / sample_fps[1]) > 1e-9):
            _fail("qwen_video_metadata_mismatch")
        source_signatures = torch.stack([
            torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
            for frame in frames])
        tensor_signatures = pixels.float().mean(dim=(-1, -2))
        if (tensor_signatures.shape != source_signatures.shape
                or not torch.allclose(tensor_signatures, source_signatures,
                                      rtol=0, atol=2.0)):
            _fail("qwen_frame_pixel_identity_or_order_mismatch")
        batch = process_qa_video(processor, text, images, pixels, metadata)
        input_ids = batch.get("input_ids")
        grid = batch.get("video_grid_thw")
        seconds_per_grid = batch.get("second_per_grid_ts")
        if input_ids is None or grid is None or seconds_per_grid is None:
            _fail("processor_output_missing")
        input_id_list = input_ids[0].tolist()
        grid_list = grid[0].tolist()
        second_value = float(seconds_per_grid[0])
        temporal_patch = int(processor.video_processor.temporal_patch_size)
        if (not input_id_list or len(grid_list) != 3 or temporal_patch <= 0
                or grid_list[0] != frame_count // temporal_patch
                or abs(second_value - 2 / (sample_fps[0] / sample_fps[1])) > 1e-6):
            _fail("processor_geometry_invalid")
        row = next((r for r in binding["source_row_bindings"]
                    if r["video_id"] == item["video_id"]), None)
        if not isinstance(row, dict):
            _fail("source_row_binding_missing")
        prefix_sha, ids_sha = _sha(text.encode("utf-8")), _sha(_canonical(input_id_list))
        ordered_prompt_hashes.append(prefix_sha)
        ordered_input_hashes.append(ids_sha)
        output_rows.append({
            "video_id": item["video_id"], "claim_id": lineage_row["claim_id"],
            "claim_text_sha256": lineage_row["claim_text_sha256"],
            "dataset_qid_sha256": _sha(item["dataset_question_id"].encode("utf-8")),
            "dataset_source_row_sha256": row["source_row_sha256"],
            "question_sha256": row["question_sha256"],
            "ordered_choice_sha256": row["ordered_choice_sha256"],
            "gold_index": row["gold_index"], "answer_label_order": LABELS,
            "answer_label_token_ids": token_ids,
            "answer_label_token_piece_counts": [1] * 5,
            "rendered_scoring_prefix_sha256": prefix_sha,
            "processor_input_ids_sha256": ids_sha,
            "processor_input_ids_hash_scope": "input_ids_only_not_pixels_or_video_metadata",
            "video_sha256": lineage_row["video_sha256"],
            "frame_count": len(frames), "video_grid_thw": grid_list,
            "second_per_grid_ts": second_value,
        })
        del batch, pixels, metadata, messages, images, videos, text, frames

    fingerprints = _fingerprints(
        model_path=model_path, protocol_raw=protocol_raw,
        dataset_source_inputs=lineage["dataset_inputs_sha256"],
        script_path=script_path or Path(__file__))
    return {
        "schema_version": 2,
        "purpose": "private_nextgqa_diagnostic_target_candidate_unvalidated",
        "status": "diagnostic_candidate_unvalidated",
        "diagnostic_only": True,
        "claim_input_manifest_present": claim_input_sha is not None,
        "claim_input_manifest_sha256": claim_input_sha,
        "claim_input_sidecar_sha256": (
            _sha(claim_input_sidecar_raw)
            if claim_input_sidecar_raw is not None else None),
        "source_packet_sha256": _sha(generation_raw),
        "lineage_manifest_sha256": _sha(lineage_raw),
        "worksheet_sha256": _sha(worksheet_raw),
        "dataset_csv_sha256": _sha(csv_raw),
        "cohort_sha256": generation["cohort_sha256"],
        "worksheet_review_completed": worksheet["review_completed"],
        "requested_records": 3,
        "effective_records": len(output_rows),
        "requested_frames_per_record": frame_count,
        "effective_frames_per_record": [row["frame_count"] for row in output_rows],
        "source_row_bindings": len(binding["source_row_bindings"]),
        "rendered_prefix_hashes_ordered_sha256": _sha(_canonical(ordered_prompt_hashes)),
        "processor_input_hashes_ordered_sha256": _sha(_canonical(ordered_input_hashes)),
        "answer_label_token_ids_sha256": _sha(_canonical(token_ids)),
        **fingerprints,
        "model_weights_loaded": False,
        "cuda_used": False,
        "network_downloads": 0,
        "model_forwards": 0,
        "score_calls": 0,
        "execution_authorization": "not_granted",
        "rights_status": "unverified_local_only",
        "records": output_rows,
        "limitations": ([
            "no claim-input manifest exists; its hash is explicitly absent",
        ] if claim_input_sha is None else []) + [
            "protocol is hashed as supplied and remains a design draft, not frozen",
            "only the full unmasked QA prompt/processor path was built",
            "processor_input_ids_sha256 covers input_ids only, not pixel tensors or video metadata",
            "human support review, HTML5 clock calibration and production atomic preflight are not established",
            "diagnostic source consistency does not authenticate upstream declarations or semantic claim support",
        ],
    }


def _load_local_processor(model_path: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    if torch.cuda.is_available():
        _fail("cuda_visible_candidate_refused")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)


def _write_exclusive_private(path: Path, payload: dict, repo_root: Path) -> str:
    final = validate_private_output(path, repo_root)
    private_dir = (repo_root / "data" / "private_nextgqa").resolve(strict=True)
    if final.parent != private_dir:
        _fail("output_must_be_repository_private_path")
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{final.name}.", dir=private_dir)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, final)
        os.chmod(final, 0o600)
    except FileExistsError:
        _fail("output_already_exists")
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return _file_sha(final)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--generation-packet", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--worksheet", required=True, type=Path)
    parser.add_argument("--claim-input-manifest", type=Path,
                        help="optional v2 claim manifest; must be paired with its sidecar")
    parser.add_argument("--claim-input-sidecar", type=Path,
                        help="optional v2 claim-input provenance sidecar")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.frames != FRAME_COUNT:
            _fail("protocol_frame_count_mismatch")
        if bool(args.claim_input_manifest) != bool(args.claim_input_sidecar):
            _fail("claim_bundle_pair_required")
        paths = (args.csv, args.generation_packet, args.lineage, args.worksheet,
                 args.protocol)
        if any(not path.is_file() for path in paths) or not args.video_root.is_dir():
            _fail("explicit_local_input_missing")
        if args.claim_input_manifest is not None:
            for private_input in (args.claim_input_manifest, args.claim_input_sidecar):
                info = private_input.lstat()
                if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                        or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                    _fail("claim_bundle_private_input_mode_invalid")
        csv_raw, generation_raw, lineage_raw, worksheet_raw, protocol_raw = (
            path.read_bytes() for path in paths)
        claim_input_manifest_raw = (args.claim_input_manifest.read_bytes()
                                    if args.claim_input_manifest is not None else None)
        claim_input_sidecar_raw = (args.claim_input_sidecar.read_bytes()
                                   if args.claim_input_sidecar is not None else None)
        _validate_protocol_v2(protocol_raw, args.frames)
        # Source/candidate identity must be sound before loading even the processor
        # or decoding any local video frames.
        _validate_upstream(csv_raw=csv_raw, generation_raw=generation_raw,
                           lineage_raw=lineage_raw, worksheet_raw=worksheet_raw,
                           video_root=args.video_root)
        processor = _load_local_processor(args.model_path)
        payload = build_candidate(
            csv_raw=csv_raw, generation_raw=generation_raw,
            lineage_raw=lineage_raw, worksheet_raw=worksheet_raw,
            claim_input_manifest_raw=claim_input_manifest_raw,
            claim_input_sidecar_raw=claim_input_sidecar_raw,
            protocol_raw=protocol_raw, video_root=args.video_root,
            model_path=args.model_path, processor=processor,
            frame_count=args.frames)
        repo_root = Path(__file__).resolve().parents[1]
        output_sha = _write_exclusive_private(args.output, payload, repo_root)
        print(json.dumps({
            "status": payload["status"], "diagnostic_only": True,
            "requested_records": payload["requested_records"],
            "effective_records": payload["effective_records"],
            "source_row_bindings": payload["source_row_bindings"],
            "output_written": True, "output_mode": "0600",
            "output_sha256": output_sha, "score_calls": 0,
            "execution_authorization": "not_granted",
        }, sort_keys=True))
        return 0
    except FileExistsError:
        code = "output_already_exists"
    except CandidateError as exc:
        code = str(exc)
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            ImportError, RuntimeError, csv.Error, OverflowError):
        code = "candidate_build_failed"
    print(json.dumps({"status": "blocked", "failure_code": code,
                      "diagnostic_only": True, "files_written": 0,
                      "score_calls": 0,
                      "execution_authorization": "not_granted"}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
