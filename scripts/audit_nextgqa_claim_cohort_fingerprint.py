#!/usr/bin/env python3
"""Recompute a local three-row NExT-GQA cohort fingerprint from source bytes.

This audit verifies the fingerprint payload for the exact rows in an existing
claim-generation packet. It does not replay the full candidate scan/selection
procedure and does not authenticate human review, custody, protocol freeze,
or execution permission. Output is aggregate and contains no QA/claim text or
stable row identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from fractions import Fraction
import math
import os
from pathlib import Path
import re
import stat
import unicodedata
from typing import Any

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels

UNIFORM_TOLERANCE = Fraction(1, 20)
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_DIR = Path("data") / "private_nextgqa"
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
MAX_PRIVATE_INPUT_BYTES = 64 * 1024 * 1024


class AuditError(ValueError):
    """Stable sanitized failure code."""


def _read_private(path: Path) -> bytes:
    root_fd = data_fd = private_fd = file_fd = None
    try:
        requested = Path(path)
        if requested.is_absolute():
            relative = requested.relative_to(REPO_ROOT)
        else:
            if Path.cwd().resolve(strict=True) != REPO_ROOT:
                raise AuditError("private_input_path_invalid")
            relative = requested
        if (len(relative.parts) != 3
                or relative.parts[:2] != PRIVATE_DATA_DIR.parts
                or relative.name in {"", ".", ".."}):
            raise AuditError("private_input_path_invalid")

        root_fd = os.open(REPO_ROOT, _DIR_FLAGS)
        data_fd = os.open("data", _DIR_FLAGS, dir_fd=root_fd)
        data_info = os.fstat(data_fd)
        if not stat.S_ISDIR(data_info.st_mode) or data_info.st_uid != os.geteuid():
            raise AuditError("private_input_parent_invalid")
        private_fd = os.open("private_nextgqa", _DIR_FLAGS, dir_fd=data_fd)
        parent_info = os.fstat(private_fd)
        if (not stat.S_ISDIR(parent_info.st_mode)
                or parent_info.st_uid != os.geteuid()
                or stat.S_IMODE(parent_info.st_mode) != 0o700):
            raise AuditError("private_input_parent_invalid")
        file_fd = os.open(relative.name, _FILE_FLAGS, dir_fd=private_fd)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size > MAX_PRIVATE_INPUT_BYTES):
            raise AuditError("private_input_mode_or_type_invalid")
        chunks = []
        size = 0
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > MAX_PRIVATE_INPUT_BYTES:
                raise AuditError("private_input_too_large")
            chunks.append(block)
        after = os.fstat(file_fd)
        if (size != before.st_size or after.st_size != before.st_size
                or after.st_ino != before.st_ino or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise AuditError("private_input_changed")
        return b"".join(chunks)
    except (OSError, ValueError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError("private_input_unavailable") from exc
    finally:
        for fd in (file_fd, private_fd, data_fd, root_fd):
            if fd is not None:
                os.close(fd)


def _parse(raw: bytes) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite")))
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AuditError("private_input_json_invalid") from exc


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AuditError("source_video_unavailable") from exc
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _video_key(value: str) -> str:
    parts = [part for part in value.replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def _normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _evidence_indices(times: list[Fraction], intervals: list) -> list[int]:
    parsed = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise AuditError("interval_shape_invalid")
        bounds = []
        for value in interval:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or (isinstance(value, float) and not math.isfinite(value)) or value < 0):
                raise AuditError("interval_endpoint_invalid")
            bounds.append(Fraction(str(value)))
        if bounds[1] < bounds[0]:
            raise AuditError("interval_order_invalid")
        parsed.append(tuple(bounds))
    return [index for index, timestamp in enumerate(times)
            if any(start <= timestamp <= end for start, end in parsed)]


def _matched_controls(evidence: list[int], size: int) -> list[int]:
    """Exhaustively apply rank-distance minimization and lexical tie-break."""
    k = len(evidence)
    candidates = [index for index in range(size) if index not in set(evidence)]
    if k == 0 or len(candidates) < k:
        return []
    ranked = []
    for choice in combinations(candidates, k):
        cost = sum(abs(left - right) for left, right in zip(sorted(evidence), choice, strict=True))
        ranked.append((cost, choice))
    return list(min(ranked)[1])


def recompute(*, generation_raw: bytes, lineage_raw: bytes,
              video_root: Path) -> dict[str, Any]:
    generation = _parse(generation_raw)
    lineage = _parse(lineage_raw)
    if (not isinstance(generation, dict)
            or generation.get("purpose") != "private_local_nextgqa_model_generated_claim_review_packet"
            or generation.get("status") != "passed"
            or generation.get("requested_videos") != 3
            or generation.get("effective_videos") != 3
            or not isinstance(generation.get("records"), list)
            or len(generation["records"]) != 3):
        raise AuditError("generation_packet_scope_invalid")
    if (not isinstance(lineage, dict)
            or lineage.get("status") != "metadata_lineage_crosschecked"
            or lineage.get("source_packet_sha256") != hashlib.sha256(generation_raw).hexdigest()
            or lineage.get("cohort_sha256") != generation.get("cohort_sha256")
            or lineage.get("dataset_inputs_sha256") != generation.get("dataset_inputs_sha256")
            or not isinstance(lineage.get("records"), list)
            or len(lineage["records"]) != 3):
        raise AuditError("lineage_scope_invalid")
    if not isinstance(video_root, Path) or not video_root.is_dir():
        raise AuditError("video_root_invalid")
    root = video_root.resolve(strict=True)
    lineage_by_sha = {row.get("video_sha256"): row for row in lineage["records"]
                      if isinstance(row, dict)}
    if (len(lineage_by_sha) != 3
            or len({row.get("dataset_question_id") for row in lineage["records"]
                    if isinstance(row, dict)}) != 3):
        raise AuditError("lineage_video_set_invalid")

    fingerprint_rows = []
    for row in generation["records"]:
        if not isinstance(row, dict):
            raise AuditError("generation_record_invalid")
        qid, local_path, video_sha = (row.get("dataset_question_id"),
                                      row.get("local_video_path"),
                                      row.get("video_sha256"))
        choices, gold_answer = row.get("choices"), row.get("gold_answer")
        spans = row.get("gold_interval_seconds")
        if (not isinstance(qid, str) or not qid or not isinstance(local_path, str)
                or not isinstance(video_sha, str) or len(video_sha) != 64
                or any(char not in "0123456789abcdef" for char in video_sha)
                or not isinstance(choices, list) or len(choices) != 5
                or any(not isinstance(x, str) or not x.strip() for x in choices)
                or not isinstance(gold_answer, str)
                or not isinstance(spans, list) or not spans):
            raise AuditError("generation_record_fields_invalid")
        matching_gold = [i for i, choice in enumerate(choices)
                         if _normalize_answer(choice) == _normalize_answer(gold_answer)]
        if len(matching_gold) != 1:
            raise AuditError("gold_choice_binding_ambiguous")
        local = Path(local_path).resolve(strict=True)
        try:
            local.relative_to(root)
        except ValueError as exc:
            raise AuditError("source_video_outside_root") from exc
        if _sha_file(local) != video_sha:
            raise AuditError("source_video_hash_mismatch")
        frame_count = generation.get("frames_per_video")
        if type(frame_count) is not int or frame_count < 2:
            raise AuditError("frame_count_invalid")
        try:
            frames, decoded = decode_sampled_video(
                local, num_frames=frame_count,
                uniform_tolerance_seconds=UNIFORM_TOLERANCE)
            verify_sampled_frame_pixels(frames, decoded)
        except Exception as exc:
            raise AuditError("source_video_decode_or_pixel_check_failed") from exc
        decoded_rows = decoded.get("sampled_frames")
        if not isinstance(decoded_rows, list) or len(decoded_rows) != frame_count:
            raise AuditError("sampled_pts_count_invalid")
        if decoded.get("source_sha256") != video_sha:
            raise AuditError("decoded_source_hash_mismatch")
        sampled_pts = []
        for item in decoded_rows:
            pair = item.get("relative_presentation_time") if isinstance(item, dict) else None
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(v) is not int for v in pair) or pair[1] <= 0):
                raise AuditError("sampled_pts_invalid")
            sampled_pts.append((pair[0], pair[1]))
        from fractions import Fraction
        pts = [Fraction(*pair) for pair in sampled_pts]
        evidence = _evidence_indices(pts, spans)
        control = _matched_controls(evidence, frame_count)
        if not evidence or len(control) != len(evidence) or set(control) & set(evidence):
            raise AuditError("evidence_control_assignment_invalid")
        question = row.get("question")
        if not isinstance(question, str):
            raise AuditError("question_field_invalid")
        source = lineage_by_sha.get(video_sha)
        if (source is None or source.get("video_id") != "video-" + video_sha
                or source.get("dataset_question_id") != qid
                or source.get("question_sha256") != hashlib.sha256(question.encode("utf-8")).hexdigest()
                or source.get("relative_video_path") != local.relative_to(root).as_posix()):
            raise AuditError("lineage_row_binding_invalid")
        fingerprint_rows.append({
            "qid": qid,
            "video_key": _video_key(local.relative_to(root).as_posix()),
            "video_sha256": decoded.get("source_sha256"),
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "choices_sha256": _canonical_sha(choices),
            "gold": matching_gold[0],
            "spans": spans,
            "sampled_pts": [[value.numerator, value.denominator] for value in pts],
            "evidence_positions": evidence,
            "control_positions": control,
        })
    observed = _canonical_sha(fingerprint_rows)
    declared = generation.get("cohort_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise AuditError("declared_cohort_digest_invalid")
    return {
        "status": "passed" if observed == declared else "mismatch",
        "requested_records": 3,
        "decoded_records": len(fingerprint_rows),
        "frame_count_per_record": generation["frames_per_video"],
        "recomputed_cohort_sha256": observed,
        "declared_cohort_sha256": declared,
        "digest_match": observed == declared,
        "full_candidate_scan_replayed": False,
        "human_review_or_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-packet", required=True, type=Path)
    parser.add_argument("--lineage-manifest", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = recompute(generation_raw=_read_private(args.generation_packet),
                           lineage_raw=_read_private(args.lineage_manifest),
                           video_root=args.video_root)
    except AuditError as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["digest_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
