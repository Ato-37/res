"""Build a v2-bound, pre-review claim-input bundle without changing v1 schema.

The wrapper accepts exact local bytes for the eligible manifest, original
three-record generation packet, lineage manifest, and v5 review policy. It
cross-checks their structural identity, then delegates construction of the
private claim manifest to ``build_claim_input_manifest.build`` unchanged. The
sidecar is path-free and contains hashes/identities only, never claim text or
raw QA. Both products are candidate, unreviewed, structural-only artifacts;
neither authenticates lineage or permits scoring.

CLI usage: run this script with --help for the required eligible manifest,
generation packet, lineage, policy, test.csv, video map, video root, dataset
ID, and two output paths. Outputs must be new files; in-repository outputs are
restricted to data/private_nextgqa and created mode 0600. Pass
--user-authorized-local-only to confirm the already-authorized local operation.
An optional --worksheet checks claim/identity hashes in memory and adds only its
raw-byte digest to the sidecar. This does not establish custody, human review,
planner compatibility, or scoring authorization.
"""
from __future__ import annotations

import hashlib
import csv
import io
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from build_claim_input_manifest import build as build_v1_claim_input

V2_COHORT_SHA256 = "162fda51377768ae4e488fbfa1a8377f903ead3af7b277a17780b53e67b79a95"
V5_POLICY_SHA256 = "f33fd193fdcec7bd9200ab700c46616cbd7ee78535399ef193496ba52054260a"
V5_POLICY_ID = "nextgqa_claim_review_v5"
ALL_RELEVANT_OCCURRENCES = "all_relevant_occurrences"
GENERATION_PURPOSE = "private_local_nextgqa_model_generated_claim_review_packet"
LINEAGE_PURPOSE = "private_source_lineage_crosscheck_for_claim_review"
_SHA = re.compile(r"^[a-f0-9]{64}$")
_MODEL_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
_LINEAGE_KEYS = {
    "schema_version", "purpose", "created_by", "source_packet_sha256",
    "cohort_sha256", "dataset_inputs_sha256", "requested_records",
    "effective_records", "verification_scope", "status", "records",
}
_LINEAGE_ROW_KEYS = {
    "video_id", "claim_id", "dataset_question_id", "dataset_video_id",
    "video_key", "relative_video_path", "video_sha256", "question_sha256",
    "claim_text_sha256",
}
_DATASET_INPUT_NAMES = {
    "test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json",
}
_POLICY_KEYS = {
    "schema_version", "policy_id", "purpose", "occurrence_policy", "review_scope", "display_procedure",
    "permitted_evidence", "support_rule", "interval_rule", "failure_rule", "review_order",
}
_WORKSHEET_KEYS = {
    "schema_version", "purpose", "status", "created_at_utc", "source_packet_sha256",
    "cohort_sha256", "model_revision", "source_commit", "claim_packet_runner_sha256",
    "cohort_selector_runner_sha256", "dataset_inputs_sha256", "occurrence_policy",
    "review_procedure_status", "requested_records", "effective_records",
    "claim_records_available", "review_completed", "execution_authorization", "records",
    "claim_review_policy_id", "claim_review_policy_sha256",
}
_WORKSHEET_ROW_KEYS = {
    "video_path", "video_sha256", "video_id", "claim_id", "review_token",
    "question", "claim_text", "claim_text_sha256", "review",
}
_WORKSHEET_REVIEW_KEYS = {
    "source_identity_verified", "full_source_video_reviewed",
    "question_and_claim_only_context_shown", "gold_and_seed_hidden_until_mapping_frozen",
    "outcome_scores_not_accessed", "occurrence_policy_applied", "support_decision",
    "evidence_intervals_seconds", "rationale", "reviewer_role",
    "review_record_reference", "first_played_at_utc", "reviewed_at_utc", "review_status",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_confined_file(root: Path, relative_path: str) -> str:
    """Hash one regular file through no-follow directory descriptors."""
    parts = Path(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("source_video_path_invalid")
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, dir_flags)
    current_fd = root_fd
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, dir_flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("source_video_not_regular_file")
            digest = hashlib.sha256()
            with os.fdopen(file_fd, "rb", closefd=True) as stream:
                file_fd = -1
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _parse(raw: object, code: str) -> Any:
    if not isinstance(raw, bytes):
        raise ValueError(code)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("non_finite_json")))
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError,
            RecursionError) as exc:
        raise ValueError(code) from exc
    return value


def _text(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("input_json_not_canonicalizable") from exc


def _validate_policy(policy_raw: bytes) -> str:
    policy = _parse(policy_raw, "policy_json_invalid")
    if (not isinstance(policy, dict) or set(policy) != _POLICY_KEYS
            or type(policy.get("schema_version")) is not int
            or policy.get("schema_version") != 1
            or policy.get("policy_id") != V5_POLICY_ID
            or policy.get("purpose") != "pre_outcome_human_claim_to_video_mapping"
            or policy.get("occurrence_policy") != ALL_RELEVANT_OCCURRENCES):
        raise ValueError("policy_schema_or_identity_invalid")
    policy_sha = _sha(policy_raw)
    if policy_sha != V5_POLICY_SHA256:
        raise ValueError("policy_v5_hash_mismatch")
    if not all(_text(policy.get(key)) for key in _POLICY_KEYS - {"schema_version"}):
        raise ValueError("policy_fields_invalid")
    return policy_sha


def _validate_lineage(lineage_raw: bytes, generation_raw: bytes,
                      generation: dict) -> tuple[dict, dict[str, dict]]:
    lineage = _parse(lineage_raw, "lineage_json_invalid")
    if (not isinstance(lineage, dict) or set(lineage) != _LINEAGE_KEYS
            or type(lineage.get("schema_version")) is not int
            or lineage.get("schema_version") != 1
            or lineage.get("purpose") != LINEAGE_PURPOSE
            or lineage.get("status") != "metadata_lineage_crosschecked"
            or lineage.get("requested_records") != 3
            or lineage.get("effective_records") != 3
            or lineage.get("source_packet_sha256") != _sha(generation_raw)
            or lineage.get("cohort_sha256") != V2_COHORT_SHA256
            or generation.get("cohort_sha256") != V2_COHORT_SHA256
            or not isinstance(lineage.get("records"), list)
            or len(lineage["records"]) != 3):
        raise ValueError("lineage_manifest_binding_invalid")
    dataset_inputs = lineage.get("dataset_inputs_sha256")
    if (not isinstance(dataset_inputs, dict)
            or set(dataset_inputs) != _DATASET_INPUT_NAMES
            or not all(_is_sha(value) for value in dataset_inputs.values())
            or generation.get("dataset_inputs_sha256") != dataset_inputs):
        raise ValueError("lineage_dataset_inputs_invalid")
    if (not _is_sha(lineage.get("source_packet_sha256"))
            or not _text(lineage.get("created_by"))
            or not _text(lineage.get("verification_scope"))):
        raise ValueError("lineage_manifest_binding_invalid")
    by_video = {}
    for row in lineage["records"]:
        if not isinstance(row, dict) or set(row) != _LINEAGE_ROW_KEYS:
            raise ValueError("lineage_row_schema_invalid")
        video_id = row.get("video_id")
        if not _text(video_id) or video_id in by_video:
            raise ValueError("lineage_video_id_invalid_or_duplicate")
        for field in ("claim_id", "dataset_question_id", "dataset_video_id", "video_key",
                      "relative_video_path"):
            if not _text(row.get(field)):
                raise ValueError("lineage_row_identity_invalid")
        if ("\\" in row["relative_video_path"]
                or row["relative_video_path"].startswith("/")
                or ".." in row["relative_video_path"].split("/")):
            raise ValueError("lineage_media_path_invalid")
        for field in ("video_sha256", "question_sha256", "claim_text_sha256"):
            if not _is_sha(row.get(field)):
                raise ValueError("lineage_row_hash_invalid")
        if video_id != "video-" + row["video_sha256"]:
            raise ValueError("lineage_video_media_binding_invalid")
        by_video[video_id] = row
    return lineage, by_video


def _video_key(value: str) -> str:
    parts = [part for part in value.replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def _source_rows(test_csv_raw: bytes, video_map_raw: bytes) -> tuple[dict, dict[str, list[dict[str, str]]]]:
    if not isinstance(test_csv_raw, bytes) or not isinstance(video_map_raw, bytes):
        raise ValueError("dataset_source_bytes_invalid")
    try:
        text = test_csv_raw.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        video_map = _parse(video_map_raw, "video_map_json_invalid")
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("dataset_source_parse_invalid") from exc
    if not isinstance(video_map, dict) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("dataset_source_schema_invalid")
    by_qid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        qid = row.get("qid")
        if isinstance(qid, str):
            by_qid.setdefault(qid, []).append(row)
    return video_map, by_qid


def _validate_worksheet_binding(worksheet_raw: bytes, generation_raw: bytes,
                                generation: dict, manifest: dict,
                                lineage_by_video: dict[str, dict],
                                policy_sha: str, video_root: Path) -> str:
    worksheet = _parse(worksheet_raw, "worksheet_json_invalid")
    if (not isinstance(worksheet, dict) or set(worksheet) != _WORKSHEET_KEYS
            or type(worksheet.get("schema_version")) is not int
            or worksheet.get("schema_version") != 1
            or worksheet.get("status") != "draft_not_reviewed"
            or worksheet.get("review_completed") is not False
            or worksheet.get("execution_authorization") != "not_granted"
            or worksheet.get("review_procedure_status") != "frozen_pre_review"
            or worksheet.get("requested_records") != 3
            or worksheet.get("effective_records") != 3
            or worksheet.get("claim_records_available") != 3
            or worksheet.get("source_packet_sha256") != _sha(generation_raw)
            or worksheet.get("cohort_sha256") != V2_COHORT_SHA256
            or worksheet.get("model_revision") != generation.get("model_revision")
            or worksheet.get("dataset_inputs_sha256") != generation.get("dataset_inputs_sha256")
            or worksheet.get("occurrence_policy") != ALL_RELEVANT_OCCURRENCES
            or worksheet.get("claim_review_policy_id") != V5_POLICY_ID
            or worksheet.get("claim_review_policy_sha256") != policy_sha
            or not isinstance(worksheet.get("records"), list)
            or len(worksheet["records"]) != 3):
        raise ValueError("worksheet_manifest_binding_invalid")
    manifest_by_video = {row["video_id"]: row for row in manifest["claims"]}
    seen = set()
    for row in worksheet["records"]:
        if not isinstance(row, dict) or set(row) != _WORKSHEET_ROW_KEYS:
            raise ValueError("worksheet_record_schema_invalid")
        video_id = row.get("video_id")
        if video_id in seen or video_id not in manifest_by_video:
            raise ValueError("worksheet_cohort_binding_invalid")
        seen.add(video_id)
        claim_row = manifest_by_video[video_id]
        lineage_row = lineage_by_video[video_id]
        worksheet_video_path = row.get("video_path")
        if not _text(worksheet_video_path):
            raise ValueError("worksheet_video_path_binding_invalid")
        try:
            worksheet_media = Path(worksheet_video_path).resolve(strict=True)
            worksheet_relative = worksheet_media.relative_to(video_root.resolve(strict=True)).as_posix()
        except (OSError, RuntimeError, ValueError):
            raise ValueError("worksheet_video_path_binding_invalid") from None
        if worksheet_relative != lineage_row["relative_video_path"]:
            raise ValueError("worksheet_video_path_binding_invalid")
        if (row.get("video_sha256") != lineage_row["video_sha256"]
                or row.get("claim_id") != claim_row["claim_id"]
                or row.get("claim_text") != claim_row["claim_text"]
                or row.get("claim_text_sha256") != claim_row["claim_text_sha256"]
                or not isinstance(row.get("question"), str)
                or _sha(row["question"].encode("utf-8")) != lineage_row["question_sha256"]
                or not isinstance(row.get("review"), dict)
                or set(row["review"]) != _WORKSHEET_REVIEW_KEYS
                or row["review"].get("review_status") != "not_started"):
            raise ValueError("worksheet_claim_hash_binding_mismatch")
        if any(value is not None for key, value in row["review"].items()
               if key != "review_status"):
            raise ValueError("worksheet_not_unreviewed")
    if seen != set(manifest_by_video):
        raise ValueError("worksheet_cohort_binding_invalid")
    return _sha(worksheet_raw)


def build_v2_claim_input_bundle(*, eligible_manifest_raw: bytes,
                                generation_export_raw: bytes,
                                lineage_manifest_raw: bytes,
                                policy_raw: bytes,
                                dataset_id: str, test_csv_raw: bytes,
                                video_map_raw: bytes, video_root: Path,
                                worksheet_raw: bytes | None = None) -> tuple[dict, bytes, dict, bytes]:
    """Validate exact v2 inputs and return unchanged v1 manifest plus sidecar.

    The generation export is the raw three-row claim-generation packet. Its
    private QA/path fields remain only in the caller's input bytes. The sidecar
    includes no claim/question text, raw QA, or video path. The v1
    generation_run_sha256 compatibility field receives the generation packet
    byte digest only; it does not identify a unique run or attest to execution
provenance. Source files are hash-checked only, never decoded. The frozen
cohort digest is checked against the pinned value and both input declarations;
this wrapper does not recompute the planner's complete fingerprint, which also
depends on sampled PTS, interval/control positions, choices, and gold labels.
    """
    eligible = _parse(eligible_manifest_raw, "eligible_manifest_invalid")
    generation = _parse(generation_export_raw, "generation_export_invalid")
    if (not isinstance(eligible, list) or len(eligible) != 3
            or any(not isinstance(row, dict) or not _text(row.get("video_id"))
                   for row in eligible)):
        raise ValueError("eligible_cohort_must_have_three_rows")
    eligible_ids = [row["video_id"] for row in eligible]
    if len(set(eligible_ids)) != 3:
        raise ValueError("eligible_video_id_duplicate")
    if (not _text(dataset_id) or not isinstance(generation, dict)
            or type(generation.get("schema_version")) is not int
            or generation.get("schema_version") != 1
            or type(generation.get("requested_videos")) is not int
            or type(generation.get("effective_videos")) is not int
            or type(generation.get("fully_structured_claim_records")) is not int
            or generation.get("purpose") != GENERATION_PURPOSE
            or generation.get("status") != "passed"
            or generation.get("requested_videos") != 3
            or generation.get("effective_videos") != 3
            or generation.get("fully_structured_claim_records") != 3
            or not _is_sha(generation.get("cohort_sha256"))
            or generation.get("cohort_sha256") != V2_COHORT_SHA256
            or generation.get("model") != _MODEL_NAME
            or not isinstance(generation.get("model_revision"), str)
            or _MODEL_COMMIT.fullmatch(generation["model_revision"]) is None
            or generation.get("model_revision") != _MODEL_REVISION
            or not isinstance(generation.get("records"), list)
            or len(generation["records"]) != 3):
        raise ValueError("generation_export_scope_invalid")
    policy_sha = _validate_policy(policy_raw)
    video_map, csv_by_qid = _source_rows(test_csv_raw, video_map_raw)
    if (not isinstance(video_root, Path) or not video_root.exists()
            or not video_root.is_dir()):
        raise ValueError("video_root_invalid")
    root_resolved = video_root.resolve(strict=True)
    dataset_hashes = generation.get("dataset_inputs_sha256")
    if (not isinstance(dataset_hashes, dict)
            or dataset_hashes.get("test.csv") != _sha(test_csv_raw)
            or dataset_hashes.get("map_vid_vidorID.json") != _sha(video_map_raw)):
        raise ValueError("dataset_source_hash_mismatch")
    lineage, lineage_by_video = _validate_lineage(
        lineage_manifest_raw, generation_export_raw, generation)
    if set(lineage_by_video) != set(eligible_ids):
        raise ValueError("eligible_lineage_cohort_mismatch")
    generation_by_video = {}
    sidecar_rows = []
    for row in generation["records"]:
        if not isinstance(row, dict):
            raise ValueError("generation_record_invalid")
        qid = row.get("dataset_question_id")
        question = row.get("question")
        video_sha = row.get("video_sha256")
        parsed = row.get("generated_parsed")
        local_path = row.get("local_video_path")
        if (not _text(qid) or not _text(question) or not _is_sha(video_sha)
                or not _text(local_path)
                or not isinstance(parsed, dict) or not _text(parsed.get("claim"))):
            raise ValueError("generation_record_binding_invalid")
        video_id = "video-" + video_sha
        if video_id in generation_by_video:
            raise ValueError("generation_video_duplicate")
        lineage_row = lineage_by_video.get(video_id)
        if lineage_row is None:
            raise ValueError("generation_lineage_row_missing")
        claim = parsed["claim"]
        try:
            media_path = Path(local_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("source_video_unavailable") from exc
        try:
            relative_path = media_path.relative_to(root_resolved).as_posix()
        except ValueError:
            raise ValueError("source_video_outside_frozen_root") from None
        try:
            media_sha = _sha_confined_file(root_resolved, relative_path)
        except OSError as exc:
            raise ValueError("source_video_unavailable") from exc
        if (relative_path != lineage_row["relative_video_path"] or media_sha != video_sha):
            raise ValueError("source_video_hash_or_path_mismatch")
        qid_rows = csv_by_qid.get(qid, [])
        mapped_matches = []
        for candidate in qid_rows:
            dataset_video_id = candidate.get("video_id")
            if (not isinstance(dataset_video_id, str) or dataset_video_id not in video_map
                    or not isinstance(video_map[dataset_video_id], str)):
                continue
            if _video_key(video_map[dataset_video_id]) == _video_key(relative_path):
                mapped_matches.append(candidate)
        if (len(mapped_matches) != 1
                or mapped_matches[0].get("video_id") != lineage_row["dataset_video_id"]
                or mapped_matches[0].get("question", "").strip() != question
                or _video_key(relative_path) != lineage_row["video_key"]):
            raise ValueError("dataset_video_question_binding_mismatch")
        claim_sha = _sha(claim.encode("utf-8"))
        question_sha = _sha(question.encode("utf-8"))
        expected_claim_id = "claim-" + _sha(
            (video_sha + "\0" + qid + "\0" + claim_sha).encode("utf-8"))
        if (lineage_row["dataset_question_id"] != qid
                or lineage_row["video_sha256"] != video_sha
                or lineage_row["question_sha256"] != question_sha
                or lineage_row["claim_text_sha256"] != claim_sha
                or lineage_row["claim_id"] != expected_claim_id):
            raise ValueError("generation_lineage_binding_mismatch")
        generation_by_video[video_id] = {
            "video_id": video_id,
            "claim_id": expected_claim_id,
            "claim_text": claim,
            "model_revision": generation["model"] + "@" + generation["model_revision"],
            # Compatibility slot required by the unchanged v1 builder. This
            # is the packet-container byte digest, not a unique execution ID.
            "generation_run_sha256": _sha(generation_export_raw),
        }
        sidecar_rows.append({
            "video_id": video_id,
            "dataset_video_id": lineage_row["dataset_video_id"],
            "dataset_question_id": qid,
            "video_sha256": video_sha,
            "question_sha256": question_sha,
            "claim_id": expected_claim_id,
            "claim_text_sha256": claim_sha,
        })
    if set(generation_by_video) != set(eligible_ids):
        raise ValueError("generation_eligible_cohort_mismatch")
    projected_generation = {
        "schema_version": 1,
        "claims": [generation_by_video[video_id] for video_id in eligible_ids],
    }
    manifest, manifest_raw = build_v1_claim_input(
        eligible, eligible_manifest_raw, projected_generation, dataset_id,
        policy_sha, ALL_RELEVANT_OCCURRENCES)
    worksheet_sha = None
    if worksheet_raw is not None:
        worksheet_sha = _validate_worksheet_binding(
            worksheet_raw, generation_export_raw, generation, manifest,
            lineage_by_video, policy_sha, root_resolved)
    sidecar = {
        "schema_version": 1,
        "purpose": "private_nextgqa_v2_claim_input_bundle_provenance",
        "status": "candidate_unreviewed",
        "scope": "structural_only",
        "review_completed": False,
        "scoring_ready": False,
        "execution_authorization": "not_granted",
        "eligible_dataset_id": dataset_id,
        "cohort_sha256": V2_COHORT_SHA256,
        "cohort_digest_check": "pinned_value_and_input_declarations_match_not_recomputed_from_full_planner_inputs",
        "occurrence_policy": ALL_RELEVANT_OCCURRENCES,
        "policy_id": V5_POLICY_ID,
        "policy_sha256": policy_sha,
        "model": generation["model"],
        "model_revision": generation["model_revision"],
        "v1_generation_run_sha256_semantics": "generation_packet_container_byte_digest_not_unique_run_id",
        "worksheet_binding_check": (
            "claim_and_identity_hashes_matched_only_not_review_or_custody_validation"
            if worksheet_sha else "not_supplied"),
        "input_sha256": {
            "eligible_manifest": _sha(eligible_manifest_raw),
            "generation_packet": _sha(generation_export_raw),
            "lineage_manifest": _sha(lineage_manifest_raw),
            "policy": policy_sha,
            "test_csv": _sha(test_csv_raw),
            "video_map": _sha(video_map_raw),
            **({"worksheet": worksheet_sha} if worksheet_sha else {}),
        },
        "claim_input_manifest_sha256": _sha(manifest_raw),
        "records": [next(row for row in sidecar_rows if row["video_id"] == video_id)
                    for video_id in eligible_ids],
    }
    sidecar_raw = (_canonical(sidecar) + b"\n")
    return manifest, manifest_raw, sidecar, sidecar_raw


def _read_input(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Read a regular local input without following a final-component symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > max_bytes:
        raise ValueError("input_file_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)):
            raise ValueError("input_file_changed_or_invalid")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("input_file_too_large")
        after = os.fstat(fd)
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (opened.st_dev, opened.st_ino, total, opened.st_mtime_ns, opened.st_ctime_ns)):
            raise ValueError("input_file_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_private_pair_exclusive(*, manifest_path: Path, manifest_raw: bytes,
                                 sidecar_path: Path, sidecar_raw: bytes,
                                 repo_root: Path) -> None:
    """Atomically stage each private file and roll back pair on second-link failure.

    The two links cannot be committed as one filesystem transaction; a failure
    during the second publication removes the first link before returning.
    Both destinations must be new files in the same directory.
    """
    from private_output_paths import validate_private_output

    manifest_output = validate_private_output(manifest_path, repo_root)
    sidecar_output = validate_private_output(sidecar_path, repo_root)
    if (manifest_output == sidecar_output or manifest_output.parent != sidecar_output.parent
            or not manifest_output.parent.is_dir()):
        raise ValueError("private_output_pair_paths_invalid")
    temp_paths = []
    published = []
    try:
        for output, raw in ((manifest_output, manifest_raw), (sidecar_output, sidecar_raw)):
            fd = os.open(output.parent / f".{output.name}.{os.getpid()}.tmp",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            temp = output.parent / f".{output.name}.{os.getpid()}.tmp"
            temp_paths.append(temp)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
        for temp, output in zip(temp_paths, (manifest_output, sidecar_output), strict=True):
            os.link(temp, output, follow_symlinks=False)
            published.append(output)
        dir_fd = os.open(manifest_output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        for output in reversed(published):
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temp in temp_paths:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--generation-packet", required=True, type=Path)
    parser.add_argument("--lineage-manifest", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--video-map", required=True, type=Path)
    parser.add_argument("--worksheet", type=Path,
                        help="optional pre-review worksheet; checked in memory, never copied to sidecar")
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--sidecar-output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("local-only authorization flag required")
    try:
        result = build_v2_claim_input_bundle(
            eligible_manifest_raw=_read_input(args.eligible_manifest),
            generation_export_raw=_read_input(args.generation_packet),
            lineage_manifest_raw=_read_input(args.lineage_manifest),
            policy_raw=_read_input(args.policy),
            dataset_id=args.dataset_id,
            test_csv_raw=_read_input(args.test_csv),
            video_map_raw=_read_input(args.video_map),
            video_root=args.video_root,
            worksheet_raw=_read_input(args.worksheet) if args.worksheet else None,
        )
        _, manifest_raw, sidecar, sidecar_raw = result
        write_private_pair_exclusive(
            manifest_path=args.manifest_output, manifest_raw=manifest_raw,
            sidecar_path=args.sidecar_output, sidecar_raw=sidecar_raw,
            repo_root=args.repo_root,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "bundle_build_failed")
    print("Private claim-input bundle written; status=candidate_unreviewed; "
          "execution_authorization=not_granted; "
          f"manifest_sha256={_sha(manifest_raw)}; sidecar_sha256={_sha(sidecar_raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
