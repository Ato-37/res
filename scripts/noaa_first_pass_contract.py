"""Structural-only validation for a private human first-pass choice record.

This helper checks that the first-pass record is complete and bound to the
caller-verified canonical source hashes and QID set. It does not read gold labels and
cannot establish correctness, chronology, reviewer identity, or authorization.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping


SOURCE_HASH_FIELDS = frozenset({
    "media_sha256",
    "samples_sha256",
    "gold_sha256",
    "audit_sha256",
    "rights_sha256",
})
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
UTC_MILLIS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "signed", "exported_at_utc", "attestor",
    "source_sha256", "review_media", "groups",
})
ATTESTOR_FIELDS = frozenset({"name", "role", "human"})
GROUP_FIELDS = frozenset({"group_id", "source_sha256", "choices"})
CHOICE_FIELDS = frozenset({"qid", "choice_index", "choice_label"})
REVIEW_MEDIA_FIELDS = frozenset({
    "group_id",
    "original_media_sha256",
    "silent_copy_sha256",
    "source_video_packet_sha256",
    "copy_video_packet_sha256",
    "source_video_start_time",
    "copy_video_start_time",
    "source_video_duration",
    "copy_video_duration",
    "source_video_nb_frames",
    "copy_video_nb_frames",
    "video_pts_count",
    "source_audio_streams_removed",
    "copy_audio_streams",
})
REVIEW_MEDIA_HASH_FIELDS = frozenset({
    "original_media_sha256", "silent_copy_sha256",
    "source_video_packet_sha256", "copy_video_packet_sha256",
})
REVIEW_MEDIA_TIME_FIELDS = frozenset({
    "source_video_start_time", "copy_video_start_time",
    "source_video_duration", "copy_video_duration",
})
REVIEW_MEDIA_COUNT_FIELDS = frozenset({
    "source_video_nb_frames", "copy_video_nb_frames", "video_pts_count",
    "source_audio_streams_removed", "copy_audio_streams",
})


def _strict_qid(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value:
        return value
    if type(value) is int:
        return str(value)
    return None


def _valid_hash_map(value: Any, expected: Mapping[str, str]) -> bool:
    if not isinstance(value, dict) or set(value) != SOURCE_HASH_FIELDS:
        return False
    if set(expected) != SOURCE_HASH_FIELDS:
        return False
    return all(
        isinstance(value.get(field), str)
        and SHA256_RE.fullmatch(value[field]) is not None
        and value[field] == expected[field]
        for field in SOURCE_HASH_FIELDS
    )


def _normalize_expected(
    expected_hashes_by_group: Mapping[str, Mapping[str, str]],
    expected_qids_by_group: Mapping[str, set[str] | list[str] | tuple[str, ...]],
    choice_counts_by_group: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]], dict[str, dict[str, int]]]:
    if not all(isinstance(value, Mapping) for value in (
        expected_hashes_by_group, expected_qids_by_group, choice_counts_by_group
    )):
        raise ValueError("expected contract inputs must be mappings")
    group_ids = set(expected_hashes_by_group)
    if (not group_ids or group_ids != set(expected_qids_by_group)
            or group_ids != set(choice_counts_by_group)
            or any(not isinstance(group, str) or not group for group in group_ids)):
        raise ValueError("expected group sets must be identical nonempty strings")

    hashes_out: dict[str, dict[str, str]] = {}
    qids_out: dict[str, set[str]] = {}
    counts_out: dict[str, dict[str, int]] = {}
    for group_id in group_ids:
        hashes = expected_hashes_by_group[group_id]
        if not isinstance(hashes, Mapping) or set(hashes) != SOURCE_HASH_FIELDS:
            raise ValueError("expected group hash map is malformed")
        if any(not isinstance(hashes[field], str) or not SHA256_RE.fullmatch(hashes[field])
               for field in SOURCE_HASH_FIELDS):
            raise ValueError("expected group hash is malformed")
        qids = expected_qids_by_group[group_id]
        if not isinstance(qids, (set, list, tuple)):
            raise ValueError("expected QID collection is malformed")
        normalized_qids = [_strict_qid(qid) for qid in qids]
        if any(qid is None for qid in normalized_qids) or len(set(normalized_qids)) != len(normalized_qids):
            raise ValueError("expected QIDs must be unique strings or strict integers")
        count_map = choice_counts_by_group[group_id]
        if not isinstance(count_map, Mapping):
            raise ValueError("choice count map is malformed")
        normalized_counts: dict[str, int] = {}
        for qid, count in count_map.items():
            normalized_qid = _strict_qid(qid)
            if normalized_qid is None or normalized_qid in normalized_counts:
                raise ValueError("choice count QIDs are malformed")
            if type(count) is not int or count != 5:
                raise ValueError("this NOAA contract requires exactly five choices per QID")
            normalized_counts[normalized_qid] = count
        if not normalized_qids:
            raise ValueError("each expected group must contain at least one QID")
        if set(normalized_counts) != set(normalized_qids):
            raise ValueError("choice count QIDs must exactly match expected QIDs")
        hashes_out[group_id] = dict(hashes)
        qids_out[group_id] = set(normalized_qids)
        counts_out[group_id] = normalized_counts
    return hashes_out, qids_out, counts_out


def _normalize_expected_review_media(
    expected_review_media_by_group: Mapping[str, Mapping[str, Any]],
    expected_hashes_by_group: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(expected_review_media_by_group, Mapping):
        raise ValueError("expected review-media metadata must be a mapping")
    expected_group_ids = set(expected_hashes_by_group)
    if set(expected_review_media_by_group) != expected_group_ids:
        raise ValueError("expected review-media groups must match source groups")
    normalized: dict[str, dict[str, Any]] = {}
    for group_id, raw in expected_review_media_by_group.items():
        if not isinstance(raw, Mapping) or set(raw) != REVIEW_MEDIA_FIELDS:
            raise ValueError("expected review-media record is malformed")
        if raw.get("group_id") != group_id:
            raise ValueError("expected review-media group identity is malformed")
        for field in REVIEW_MEDIA_HASH_FIELDS:
            value = raw.get(field)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ValueError("expected review-media hash is malformed")
        if raw["original_media_sha256"] != expected_hashes_by_group[group_id]["media_sha256"]:
            raise ValueError("expected review media does not bind to the source media hash")
        for field in REVIEW_MEDIA_TIME_FIELDS:
            value = raw.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not _is_finite_number(value)):
                raise ValueError("expected review-media timing is malformed")
        for field in REVIEW_MEDIA_COUNT_FIELDS:
            value = raw.get(field)
            if type(value) is not int or value < 0:
                raise ValueError("expected review-media count is malformed")
        normalized[group_id] = dict(raw)
    return normalized


def _is_finite_number(value: int | float) -> bool:
    # Import locally so the module has no heavyweight or data dependencies.
    import math

    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _review_media_matches(observed: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(observed, dict) or set(observed) != REVIEW_MEDIA_FIELDS:
        return False
    for field in REVIEW_MEDIA_HASH_FIELDS:
        value = observed.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            return False
    for field in REVIEW_MEDIA_TIME_FIELDS:
        value = observed.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not _is_finite_number(value)):
            return False
    for field in REVIEW_MEDIA_COUNT_FIELDS:
        value = observed.get(field)
        if type(value) is not int or value < 0:
            return False
    for field in REVIEW_MEDIA_TIME_FIELDS:
        if float(observed[field]) != float(expected[field]):
            return False
    for field in REVIEW_MEDIA_FIELDS - REVIEW_MEDIA_TIME_FIELDS:
        if type(observed[field]) is not type(expected[field]) or observed[field] != expected[field]:
            return False
    return True


def audit_first_pass_structure(
    first_pass: Any,
    *,
    expected_hashes_by_group: Mapping[str, Mapping[str, str]],
    expected_qids_by_group: Mapping[str, set[str] | list[str] | tuple[str, ...]],
    choice_counts_by_group: Mapping[str, Mapping[str, int]],
    expected_review_media_by_group: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a counts-only structural audit; never compares choices with gold.

    The expected mappings are supplied by a caller that has independently
    checked canonical source files and review-media metadata. This function
    does not inspect those files.
    Every response keeps execution authorization false, including structurally
    valid records.
    """
    hashes, qids, choice_counts = _normalize_expected(
        expected_hashes_by_group, expected_qids_by_group, choice_counts_by_group
    )
    review_media_expected = _normalize_expected_review_media(
        expected_review_media_by_group, hashes
    )
    expected_group_ids = set(hashes)
    expected_questions = sum(len(values) for values in qids.values())
    effective_groups = 0
    effective_choices = 0
    review_media_verified = False
    failures: set[str] = set()

    def result() -> dict[str, Any]:
        valid = not failures
        return {
            "validity": "structural_only" if valid else "invalid",
            "structurally_valid": valid,
            "failure_codes": sorted(failures),
            "expected_groups": len(expected_group_ids),
            "effective_groups": effective_groups,
            "expected_questions": expected_questions,
            "effective_choices": effective_choices,
            "review_media_matches_expected_metadata": review_media_verified,
            "semantic_correctness_verified": False,
            "chronology_verified": False,
            "reviewer_identity_verified": False,
            "execution_authorization": "not_granted",
        }

    if not isinstance(first_pass, dict):
        failures.add("first_pass_not_object")
        return result()
    if set(first_pass) != TOP_LEVEL_FIELDS:
        failures.add("first_pass_fields_invalid")
    if type(first_pass.get("schema_version")) is not int or first_pass.get("schema_version") != 1:
        failures.add("first_pass_schema_version_invalid")
    if first_pass.get("signed") is not False:
        failures.add("first_pass_signed_flag_invalid")
    timestamp = first_pass.get("exported_at_utc")
    if not isinstance(timestamp, str) or not UTC_MILLIS_RE.fullmatch(timestamp):
        failures.add("first_pass_timestamp_invalid")
    else:
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError:
            failures.add("first_pass_timestamp_invalid")

    attestor = first_pass.get("attestor")
    if (not isinstance(attestor, dict) or set(attestor) != ATTESTOR_FIELDS
            or attestor.get("name") != "" or attestor.get("role") != ""
            or attestor.get("human") is not False):
        failures.add("first_pass_attestor_template_invalid")

    source_hashes = first_pass.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_group_ids:
        failures.add("first_pass_source_hashes_invalid")
    else:
        for group_id, expected in hashes.items():
            if not _valid_hash_map(source_hashes.get(group_id), expected):
                failures.add("first_pass_source_hashes_invalid")
                break

    review_media = first_pass.get("review_media")
    if (not isinstance(review_media, list)
            or any(not isinstance(item, dict) for item in review_media)):
        failures.add("first_pass_review_media_invalid")
    else:
        media_group_ids = [item.get("group_id") for item in review_media]
        if (any(not isinstance(group_id, str) or not group_id for group_id in media_group_ids)
                or len(set(media_group_ids)) != len(media_group_ids)
                or set(media_group_ids) != expected_group_ids):
            failures.add("first_pass_review_media_invalid")
        else:
            media_by_group = {item["group_id"]: item for item in review_media}
            review_media_verified = all(
                _review_media_matches(media_by_group[group_id], expected)
                for group_id, expected in review_media_expected.items()
            )
            if not review_media_verified:
                failures.add("first_pass_review_media_mismatch")

    groups = first_pass.get("groups")
    if not isinstance(groups, list) or any(not isinstance(item, dict) for item in groups):
        failures.add("first_pass_groups_invalid")
        return result()
    group_ids = [item.get("group_id") for item in groups]
    if (any(not isinstance(group_id, str) or not group_id for group_id in group_ids)
            or len(set(group_ids)) != len(group_ids)
            or set(group_ids) != expected_group_ids):
        failures.add("first_pass_group_set_invalid")
    else:
        effective_groups = len(groups)

    for group in groups:
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or group_id not in expected_group_ids:
            continue
        if set(group) != GROUP_FIELDS:
            failures.add("first_pass_group_fields_invalid")
        expected_hash_map = hashes[group_id]
        if not _valid_hash_map(group.get("source_sha256"), expected_hash_map):
            failures.add("first_pass_group_hash_mismatch")
        choices = group.get("choices")
        if not isinstance(choices, list) or any(not isinstance(choice, dict) for choice in choices):
            failures.add("first_pass_choices_invalid")
            continue
        effective_choices += len(choices)
        choice_qids: list[str] = []
        for choice in choices:
            if set(choice) != CHOICE_FIELDS:
                failures.add("first_pass_choice_fields_invalid")
            qid = _strict_qid(choice.get("qid"))
            if qid is None:
                failures.add("first_pass_choice_qid_invalid")
                continue
            choice_qids.append(qid)
            count = choice_counts[group_id].get(qid)
            index = choice.get("choice_index")
            if count is None:
                failures.add("first_pass_choice_qid_invalid")
                continue
            if type(index) is not int or not 0 <= index < count:
                failures.add("first_pass_choice_index_invalid")
                continue
            expected_label = chr(ord("A") + index)
            if choice.get("choice_label") != expected_label:
                failures.add("first_pass_choice_label_invalid")
        if (len(choice_qids) != len(set(choice_qids))
                or set(choice_qids) != qids[group_id]):
            failures.add("first_pass_choice_qid_set_invalid")
    return result()
