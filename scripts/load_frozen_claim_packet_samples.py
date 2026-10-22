"""Reconstruct and rebind an exact frozen claim-generation cohort."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
from run_nextgqa_gold_interval_margin_pilot import (
    _hash, candidate_pair_sha256, freeze_cohort,
    validate_candidate_pair_filter,
    validate_private_media_exclusion,
)

PRIVATE = Path("data/private_nextgqa")
SPLIT_DATA_FILES = {
    "val": {"val.csv", "gsub_val.json", "map_vid_vidorID.json", "frame2time_val.json"},
    "test": {"test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json"},
}
LOCK_KEYS = {
    "assignments", "candidate_pair_filter_sha256", "candidate_pairs_available",
    "candidate_pool_summary_sha256", "candidate_videos_scanned", "cohort_sha256",
    "dataset_input_sha256", "dataset_split", "effective_videos", "full_split_candidate_video_count",
    "limitations", "local_claim_exposure_manifest_sha256", "local_claim_exposure_media_hash_count",
    "local_claim_exposure_source_file_count", "local_claim_exposure_source_inventory_sha256",
    "local_sample_exclusion_manifest_count", "local_sample_exclusion_manifest_digest",
    "local_scored_claim_overlap_count", "prior_media_exclusion_count", "prior_media_exclusion_sha256",
    "prior_scored_claim_digest_count", "prior_scored_claim_digest_set_sha256", "purpose",
    "requested_videos", "schema_version", "selected_pair_filter_sha256", "selection_rule",
    "source_sha256", "status",
}
PACKET_REQUIRED = {
    "authorization_scope", "candidate_pair_filter_digest", "candidate_pair_filter_entries",
    "candidate_pair_filter_sha256", "candidate_video_count", "candidate_videos_scanned",
    "candidate_videos_unscanned", "claim_review_completed", "cohort_sha256", "compute_dtype",
    "dataset_inputs_sha256", "dataset_split", "effective_videos", "elapsed_seconds",
    "faithfulness_effect_estimate", "frames_per_video", "fully_structured_claim_records",
    "generation_prompt_version", "gpu", "gpu_free_before_load_gib", "gpu_total_gib",
    "host_available_before_load_gib", "model", "model_revision", "peak_allocated_gib",
    "pilot_runner_sha256", "privacy", "private_exclusion_manifest_sha256", "purpose",
    "quantization", "records", "requested_videos", "runner_sha256", "schema_version",
    "seed", "source_commit", "started_utc", "status", "uniform_grid_tolerance_seconds",
    "versions",
}
PACKET_OPTIONAL = {"frozen_development_replay", "replay_loader_sha256"}
RECORD_KEYS = {
    "candidate_pair_sha256", "choices", "dataset_question_id", "generated_parsed",
    "generated_raw_text", "generation_prompt_version", "gold_answer", "gold_interval_seconds",
    "human_review", "local_video_path", "question", "review_index",
    "sampled_frame_relative_pts_seconds", "video_sha256",
}


def _private_payload(path: Path, root: Path) -> tuple[bytes, dict]:
    root = root.resolve(strict=True)
    path = Path(path)
    resolved_parent = path.parent.resolve(strict=True)
    if (not resolved_parent.is_relative_to(root) or resolved_parent != root
            or path.is_symlink()):
        raise ValueError("frozen_input_path_invalid")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                      | getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                     dir_fd=root_fd)
    finally:
        os.close(root_fd)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("frozen_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("frozen_input_duplicate_json_key")
            result[key] = value
        return result
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen_input_json_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen_input_json_shape_invalid")
    return raw, payload


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _equal_span_lists(actual, expected) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    try:
        return all(isinstance(pair, list) and len(pair) == 2
                   and abs(float(pair[0]) - span[0]) <= 1e-9
                   and abs(float(pair[1]) - span[1]) <= 1e-9
                   for pair, span in zip(actual, expected, strict=True))
    except (TypeError, ValueError, OverflowError):
        return False


def reconstruct_samples(*, packet_path: Path, lock_path: Path,
                        pair_filter_path: Path, exclusion_path: Path,
                        video_root: Path | None, repo_root: Path, frame_count: int = 16
                        ) -> tuple[list[dict], dict]:
    """Re-read dataset rows and prove each private packet row matches its frozen lock."""
    repo_root = Path(repo_root).resolve(strict=True)
    private_root = repo_root / PRIVATE
    packet_raw, packet = _private_payload(packet_path, private_root)
    lock_raw, lock = _private_payload(lock_path, private_root)
    pair_raw, pair_payload = _private_payload(pair_filter_path, private_root)
    exclusion_raw, exclusion_payload = _private_payload(exclusion_path, private_root)
    packet_sha, lock_sha = hashlib.sha256(packet_raw).hexdigest(), hashlib.sha256(lock_raw).hexdigest()
    pair_sha, exclusion_sha = hashlib.sha256(pair_raw).hexdigest(), hashlib.sha256(exclusion_raw).hexdigest()

    if (not PACKET_REQUIRED.issubset(packet)
            or not set(packet).issubset(PACKET_REQUIRED | PACKET_OPTIONAL)
            or type(packet.get("schema_version")) is not int
            or packet.get("schema_version") != 1
            or packet.get("purpose") != "private_local_nextgqa_model_generated_claim_review_packet"
            or packet.get("dataset_split") not in SPLIT_DATA_FILES
            or packet.get("status") not in {"complete", "partial", "passed"}
            or packet.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
            or packet.get("requested_videos") != 3 or packet.get("effective_videos") != 3
            or packet.get("frames_per_video") != frame_count):
        raise ValueError("frozen_packet_schema_or_scope_invalid")
    from generate_nextgqa_claim_review_pilot import (
        PROMPT_SUFFIX_BY_VERSION, parse_claim_output,
    )
    if packet.get("generation_prompt_version") not in PROMPT_SUFFIX_BY_VERSION:
        raise ValueError("frozen_packet_prompt_version_invalid")
    split = packet["dataset_split"]
    if (set(lock) != LOCK_KEYS or type(lock.get("schema_version")) is not int
            or lock.get("schema_version") != 1
            or lock.get("dataset_split") != split
            or lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("purpose") != "exploratory_claim_likelihood_selected_window_prediction_dependence"
            or lock.get("requested_videos") != 3 or lock.get("effective_videos") != 3
            or type(lock.get("local_scored_claim_overlap_count")) is not int
            or lock.get("local_scored_claim_overlap_count") != 0):
        raise ValueError("frozen_lock_schema_or_scope_invalid")
    try:
        allowed_pairs = validate_candidate_pair_filter(pair_payload)
        excluded_media = validate_private_media_exclusion(exclusion_payload)
    except ValueError as exc:
        raise ValueError("frozen_filter_manifest_invalid") from exc
    if (len(allowed_pairs) != 3 or len(excluded_media) != len(set(excluded_media))
            or packet.get("candidate_pair_filter_sha256") != pair_sha
            or packet.get("candidate_pair_filter_digest") != _hash(sorted(allowed_pairs))
            or packet.get("candidate_pair_filter_entries") != 3
            or lock.get("selected_pair_filter_sha256") != pair_sha
            or not _is_sha256(lock.get("candidate_pair_filter_sha256"))
            or packet.get("private_exclusion_manifest_sha256") != exclusion_sha):
        raise ValueError("frozen_packet_lock_manifest_binding_invalid")

    records, assignments = packet.get("records"), lock.get("assignments")
    if (not isinstance(records, list) or len(records) != 3
            or not isinstance(assignments, list) or len(assignments) != 3
            or any(not isinstance(row, dict) or set(row) != RECORD_KEYS for row in records)
            or any(not isinstance(a, dict) or set(a) != {"media_sha256", "qid_sha256"}
                   or not _is_sha256(a.get("media_sha256")) or not _is_sha256(a.get("qid_sha256"))
                   for a in assignments)):
        raise ValueError("frozen_packet_rows_or_assignments_invalid")
    if (len({a["media_sha256"] for a in assignments}) != 3
            or len({a["qid_sha256"] for a in assignments}) != 3
            or len(set(allowed_pairs)) != 3):
        raise ValueError("frozen_packet_duplicate_assignment")

    dataset_dir = repo_root / "NExT-GQA/datasets/nextgqa"
    expected_files = SPLIT_DATA_FILES[split]
    packet_data_hashes = packet.get("dataset_inputs_sha256")
    lock_data_hashes = lock.get("dataset_input_sha256")
    if (not isinstance(packet_data_hashes, dict) or set(packet_data_hashes) != expected_files
            or not isinstance(lock_data_hashes, dict)
            or set(lock_data_hashes) != expected_files - {f"frame2time_{split}.json"}):
        raise ValueError("frozen_dataset_hash_manifest_schema_invalid")
    verified_dataset_hashes = {}
    for name in expected_files:
        path = dataset_dir / name
        if not _is_sha256(packet_data_hashes.get(name)) or not path.is_file() or path.is_symlink():
            raise ValueError("frozen_dataset_input_missing_or_invalid")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != packet_data_hashes[name]:
            raise ValueError("frozen_dataset_input_hash_mismatch")
        if name in lock_data_hashes and digest != lock_data_hashes[name]:
            raise ValueError("frozen_lock_dataset_hash_mismatch")
        verified_dataset_hashes[name] = digest

    parsed_records = []
    for row in records:
        raw_text = row.get("generated_raw_text")
        if (not isinstance(raw_text, str)
                or row.get("generation_prompt_version") != packet["generation_prompt_version"]):
            raise ValueError("frozen_packet_generation_binding_invalid")
        parsed = parse_claim_output(raw_text, frame_count)
        if row.get("generated_parsed") != parsed:
            raise ValueError("frozen_packet_parsed_claim_binding_invalid")
        parsed_records.append(parsed)
    structured_count = sum(item["valid"] is True for item in parsed_records)
    if (type(packet.get("fully_structured_claim_records")) is not int
            or packet["fully_structured_claim_records"] != structured_count
            or (packet.get("status") in {"complete", "passed"}
                and structured_count != len(records))
            or (packet.get("status") == "partial" and structured_count >= len(records))):
        raise ValueError("frozen_packet_structured_count_invalid")
    paths = [Path(row["local_video_path"]) for row in records]
    if any(not p.is_absolute() or p.is_symlink() for p in paths):
        raise ValueError("frozen_video_paths_invalid")
    roots = {p.resolve(strict=True).parent.parent for p in paths}
    if len(roots) != 1:
        raise ValueError("frozen_video_roots_disagree")
    inferred_root = next(iter(roots))
    if video_root is not None:
        if video_root.is_symlink() or video_root.resolve(strict=True) != inferred_root:
            raise ValueError("frozen_video_root_source_mismatch")

    # Re-select from current CSV/gsub/map and re-decode; this recomputes all
    # row semantics, frame PTS, evidence/control assignments, and cohort digest.
    import run_nextgqa_gold_interval_margin_pilot as pilot
    old_cap = pilot.MAX_VIDEOS
    pilot.MAX_VIDEOS = 3
    try:
        selected, counts, meta, cohort_sha = freeze_cohort(
            repo_root, inferred_root, excluded_media, frame_count,
            locked_assignments=assignments, allowed_pair_sha256=allowed_pairs,
            dataset_split=split)
    finally:
        pilot.MAX_VIDEOS = old_cap
    if (len(selected) != 3 or cohort_sha != packet.get("cohort_sha256")
            or cohort_sha != lock.get("cohort_sha256")):
        raise ValueError("frozen_cohort_recomputation_mismatch")

    samples = []
    for index, (row, assignment, sample) in enumerate(zip(records, assignments, selected, strict=True)):
        media_sha = sample["decode"]["source_sha256"]
        qid_sha = hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest()
        pair_sha_value = candidate_pair_sha256(sample["video_id"], sample["qid"])
        sampled = row.get("sampled_frame_relative_pts_seconds")
        if (row.get("review_index") != index or row.get("video_sha256") != media_sha
                or row.get("dataset_question_id") != sample["qid"]
                or row.get("question") != sample["question"]
                or row.get("choices") != sample["choices"]
                or row.get("gold_answer") != sample["choices"][sample["gold"]]
                or not _equal_span_lists(row.get("gold_interval_seconds"), sample["spans"])
                or row.get("candidate_pair_sha256") != pair_sha_value
                or assignment != {"media_sha256": media_sha, "qid_sha256": qid_sha}
                or pair_sha_value != allowed_pairs[index]
                or media_sha in excluded_media
                or Path(row["local_video_path"]).resolve(strict=True) != Path(sample["video_path"]).resolve(strict=True)
                or not isinstance(sampled, list) or len(sampled) != frame_count):
            raise ValueError("frozen_packet_dataset_row_binding_invalid")
        frames, decode = sample["frames"], sample["decode"]
        decoded_times = [Fraction(*record["relative_presentation_time"])
                         for record in decode["sampled_frames"]]
        try:
            if any(abs(float(a) - float(b)) > 1e-6
                   for a, b in zip(sampled, decoded_times, strict=True)):
                raise ValueError("frozen_video_pts_binding_invalid")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frozen_video_pts_binding_invalid") from exc
        verify_sampled_frame_pixels(frames, decode)
        samples.append({
            "review_index": index, "video_path": str(sample["video_path"]),
            "video_id": sample["video_id"], "qid": sample["qid"],
            "question": sample["question"], "choices": sample["choices"],
            "gold": sample["gold"], "spans": sample["spans"],
            "sampled_times": [Fraction(str(x)) for x in sampled],
            "frames": frames, "decode": decode,
            "candidate_pair_sha256": pair_sha_value,
        })
    return samples, {
        "source_packet_sha256": packet_sha, "source_lock_sha256": lock_sha,
        "pair_filter_sha256": pair_sha, "exclusion_manifest_sha256": exclusion_sha,
        "cohort_sha256": cohort_sha, "requested_videos": 3,
        "effective_videos": len(samples), "dataset_split": split,
        "candidate_scan_count": counts["candidate_scan_count"],
        "dataset_inputs_sha256": verified_dataset_hashes,
        "source_generation_prompt_version": packet.get("generation_prompt_version"),
    }
