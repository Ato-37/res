#!/usr/bin/env python3
"""Aggregate-only comparison of a NExT-GQA split with supplied prior HMAC indexes.

This checks exact dataset video-ID and video/QID linkage against explicitly
supplied local exposure indexes. It does not infer freshness from a non-match,
read media, authenticate index completeness, or authorize model execution.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
from typing import Any

try:
    from build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _loads, _read_secret, _read_stable_regular_file,
        _token,
    )
except ImportError:  # package import under unittest discovery
    from scripts.build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _loads, _read_secret, _read_stable_regular_file,
        _token,
    )

MAX_INDEX_COUNT = 64
TOKEN_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_split(raw: bytes) -> tuple[set[str], set[tuple[str, str]]]:
    try:
        table = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="strict"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("split_csv_invalid") from exc
    if not table:
        raise ValueError("split_csv_empty")
    header = table[0]
    if (len(header) != len(set(header)) or "video_id" not in header or "qid" not in header
            or any(not name for name in header)):
        raise ValueError("split_csv_header_invalid")
    video_column, qid_column = header.index("video_id"), header.index("qid")
    video_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for row in table[1:]:
        if len(row) != len(header):
            raise ValueError("split_csv_row_width_invalid")
        video_id, qid = row[video_column], row[qid_column]
        if (not video_id or video_id != video_id.strip()
                or not qid or qid != qid.strip()):
            raise ValueError("split_csv_identity_invalid")
        video_ids.add(video_id)
        pairs.add((video_id, qid))
    if not video_ids or not pairs:
        raise ValueError("split_csv_has_no_identities")
    return video_ids, pairs


def _read_index(raw: bytes) -> tuple[set[str], set[str]]:
    try:
        value: Any = _loads(raw)
    except ValueError as exc:
        raise ValueError("prior_index_invalid") from exc
    if (not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("purpose") != "outcome_blind_prior_exposure_index"
            or value.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(value.get("coverage"), dict)
            or value["coverage"].get("historical_completeness_authenticated") is not False
            or not isinstance(value.get("identities"), dict)):
        raise ValueError("prior_index_schema_invalid")
    identities = value["identities"]
    videos = identities.get("dataset_video_hmac_sha256")
    pairs = identities.get("video_question_pair_hmac_sha256")
    if (not isinstance(videos, list) or not isinstance(pairs, list)
            or any(not isinstance(token, str) or not TOKEN_SHA256.fullmatch(token)
                   for token in videos + pairs)
            or len(videos) != len(set(videos)) or len(pairs) != len(set(pairs))):
        raise ValueError("prior_index_identity_lists_invalid")
    return set(videos), set(pairs)


def audit_split_identity_exposure(*, split_raw: bytes, index_raws: list[bytes],
                                  hmac_key: bytes) -> dict[str, Any]:
    if not isinstance(split_raw, bytes) or len(split_raw) > MAX_SOURCE_BYTES:
        raise ValueError("split_size_invalid")
    if (not isinstance(index_raws, list) or not 1 <= len(index_raws) <= MAX_INDEX_COUNT
            or any(not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES
                   for raw in index_raws)
            or sum(map(len, index_raws)) > MAX_SOURCE_BYTES):
        raise ValueError("prior_index_count_or_size_invalid")
    if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
        raise ValueError("hmac_key_invalid")
    video_ids, qid_pairs = _read_split(split_raw)
    video_tokens = {_token(hmac_key, "dataset-video-id", ("nextgqa", value))
                    for value in video_ids}
    pair_tokens = {
        _token(hmac_key, "video-question-pair",
               ("nextgqa", "dataset-video-id", "nextgqa", video_id, qid))
        for video_id, qid in qid_pairs
    }
    index_sets = [_read_index(raw) for raw in index_raws]
    matched_video_by_index = [len(video_tokens & known_video)
                              for known_video, _ in index_sets]
    matched_pair_by_index = [len(pair_tokens & known_pair)
                             for _, known_pair in index_sets]
    known_videos = set().union(*(item[0] for item in index_sets))
    known_pairs = set().union(*(item[1] for item in index_sets))
    return {
        "schema_version": 1,
        "purpose": "nextgqa_split_prior_identity_exposure_audit",
        "status": "completed",
        "scope": "exact_video_id_and_video_qid_hmac_matches_only",
        "split_unique_video_ids": len(video_ids),
        "split_unique_video_qid_pairs": len(qid_pairs),
        "matched_video_ids_by_prior_index": matched_video_by_index,
        "matched_video_qid_pairs_by_prior_index": matched_pair_by_index,
        "matched_video_ids_in_prior_index_union": len(video_tokens & known_videos),
        "matched_video_qid_pairs_in_prior_index_union": len(pair_tokens & known_pairs),
        "hmac_key_to_index_binding_authenticated": False,
        "historical_completeness_authenticated": False,
        "freshness_conclusion": "not_made",
        "media_byte_match_checked": False,
        "faithfulness_scored": False,
        "gpu_used": False,
        "execution_authorization": "not_granted",
    }


def _read_input(path: Path, *, required_mode: int | None = None) -> bytes:
    try:
        return _read_stable_regular_file(path, max_bytes=MAX_SOURCE_BYTES,
                                         required_mode=required_mode)
    except (OSError, ValueError) as exc:
        raise ValueError("input_file_invalid") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path,
                        help="split CSV with explicit video_id and qid columns")
    parser.add_argument("--prior-index", required=True, action="append", type=Path,
                        help="owner-only prior HMAC index; repeat to union known history")
    parser.add_argument("--hmac-key", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if len(args.prior_index) > MAX_INDEX_COUNT:
            raise ValueError("prior_index_count_invalid")
        split_raw = _read_input(args.split_csv)
        index_raws = [_read_input(path, required_mode=0o600)
                      for path in args.prior_index]
        hmac_key = _read_secret(args.hmac_key)
        report = audit_split_identity_exposure(
            split_raw=split_raw, index_raws=index_raws, hmac_key=hmac_key)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "audit_failed")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
