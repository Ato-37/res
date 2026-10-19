#!/usr/bin/env python3
"""Aggregate-only prior-history audit for a frozen NExT-GQA cohort.

Join frozen candidate-pair digests to the split CSV, then bind packet rows to
those pairs by packet-supplied pair digest (or unambiguous QID for legacy
packets) and compare video, video/QID, and packet-claimed media HMACs against
the explicitly supplied partial prior indexes. This does not authenticate the
packet's pair-to-media association. Nonmatches are never freshness
certificates and this tool does not authorize scoring or model execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

try:
    from build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _read_secret, _read_stable_regular_file, _token,
    )
    from build_nextgqa_zip_gold_interval_candidate_pool import (
        SHA, _load_index, candidate_pair_sha256,
    )
except ImportError:  # package import during unittest discovery
    from scripts.build_nextgqa_prior_exposure_manifest import (
        MAX_SOURCE_BYTES, _read_secret, _read_stable_regular_file, _token,
    )
    from scripts.build_nextgqa_zip_gold_interval_candidate_pool import (
        SHA, _load_index, candidate_pair_sha256,
    )

MAX_INDEX_COUNT = 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _json_object(raw: bytes, code: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(code)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8", errors="strict"),
                           object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _split_pairs(raw: bytes) -> set[tuple[str, str]]:
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="strict"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("split_csv_invalid") from exc
    if not rows or len(rows[0]) != len(set(rows[0])) or not {"video_id", "qid"}.issubset(rows[0]):
        raise ValueError("split_csv_header_invalid")
    video_i, qid_i = rows[0].index("video_id"), rows[0].index("qid")
    pairs: set[tuple[str, str]] = set()
    for row in rows[1:]:
        if len(row) != len(rows[0]):
            raise ValueError("split_csv_row_width_invalid")
        video_id, qid = row[video_i], row[qid_i]
        if not video_id or video_id != video_id.strip() or not qid or qid != qid.strip():
            raise ValueError("split_csv_identity_invalid")
        pair = (video_id, qid)
        if pair in pairs:
            raise ValueError("split_csv_duplicate_pair")
        pairs.add(pair)
    if not pairs:
        raise ValueError("split_csv_empty")
    return pairs


def _frozen_hashes(raw: bytes) -> set[str]:
    obj = _json_object(raw, "frozen_pairs_invalid")
    values = obj.get("pair_sha256")
    if (type(obj.get("schema_version")) is not int or obj["schema_version"] != 1
            or set(obj) != {"schema_version", "pair_sha256"}
            or not isinstance(values, list) or not values
            or any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in values)
            or len(values) != len(set(values))):
        raise ValueError("frozen_pairs_invalid")
    return set(values)


def audit_frozen_candidate_exposure(*, split_raw: bytes, frozen_pairs_raw: bytes,
                                    packet_raw: bytes, index_raws: list[bytes],
                                    hmac_key: bytes) -> dict[str, Any]:
    """Compare one frozen cohort to supplied prior indexes without echoing IDs."""
    sources = (split_raw, frozen_pairs_raw, packet_raw, *index_raws)
    if any(not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES for raw in sources):
        raise ValueError("input_size_invalid")
    if (not 1 <= len(index_raws) <= MAX_INDEX_COUNT
            or sum(len(raw) for raw in index_raws) > MAX_SOURCE_BYTES
            or not isinstance(hmac_key, bytes) or len(hmac_key) < 32):
        raise ValueError("prior_index_or_key_invalid")

    frozen = _frozen_hashes(frozen_pairs_raw)
    split_pairs = _split_pairs(split_raw)
    pair_map: dict[str, tuple[str, str]] = {}
    for video_id, qid in split_pairs:
        digest = candidate_pair_sha256(video_id, qid)
        if digest in frozen:
            if digest in pair_map and pair_map[digest] != (video_id, qid):
                raise ValueError("frozen_pair_digest_collision")
            pair_map[digest] = (video_id, qid)
    if set(pair_map) != frozen:
        raise ValueError("frozen_pair_not_in_split")

    packet = _json_object(packet_raw, "run_packet_invalid")
    records = packet.get("records")
    if (packet.get("dataset_split") != "val" or not isinstance(records, list)
            or len(records) != len(frozen) or not records):
        raise ValueError("run_packet_cohort_invalid")
    packet_by_pair: dict[str, dict[str, Any]] = {}
    packet_qids: list[str] = []
    packet_pair_binding: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("run_packet_record_invalid")
        qid, media_sha = record.get("dataset_question_id"), record.get("video_sha256")
        if (not isinstance(qid, str) or not qid.strip()
                or not isinstance(media_sha, str) or HEX64.fullmatch(media_sha) is None):
            raise ValueError("run_packet_identity_invalid")
        packet_qids.append(qid)
        pair_digest = record.get("candidate_pair_sha256")
        if pair_digest is not None:
            if not isinstance(pair_digest, str) or HEX64.fullmatch(pair_digest) is None:
                raise ValueError("run_packet_pair_binding_invalid")
            if pair_digest in packet_pair_binding:
                raise ValueError("run_packet_pair_binding_duplicate")
            packet_pair_binding[pair_digest] = qid

    selected_qids = [qid for _video_id, qid in pair_map.values()]
    if packet_pair_binding:
        if (len(packet_pair_binding) != len(records) or set(packet_pair_binding) != frozen
                or any(packet_pair_binding[digest] != pair_map[digest][1]
                       for digest in packet_pair_binding)):
            raise ValueError("packet_to_frozen_pair_join_invalid")
        for record in records:
            packet_by_pair[record["candidate_pair_sha256"]] = record
        packet_pair_binding_present = True
    else:
        if (len(selected_qids) != len(set(selected_qids))
                or len(packet_qids) != len(set(packet_qids))
                or set(selected_qids) != set(packet_qids)):
            raise ValueError("packet_to_frozen_pair_join_invalid")
        qid_to_digest = {qid: digest for digest, (_video_id, qid) in pair_map.items()}
        packet_by_pair = {qid_to_digest[record["dataset_question_id"]]: record for record in records}
        packet_pair_binding_present = False

    indexes = [_load_index(raw) for raw in index_raws]
    prior_video = set().union(*(index["video"] for index in indexes))
    prior_pair = set().union(*(index["pair"] for index in indexes))
    prior_media = set().union(*(index["media"] for index in indexes))
    video_tokens = {_token(hmac_key, "dataset-video-id", ("nextgqa", video_id))
                    for video_id, _qid in pair_map.values()}
    pair_tokens = {_token(hmac_key, "video-question-pair",
                          ("nextgqa", "dataset-video-id", "nextgqa", video_id, qid))
                   for video_id, qid in pair_map.values()}
    media_tokens = {_token(hmac_key, "media-sha256", (record["video_sha256"],))
                    for record in packet_by_pair.values()}

    return {
        "schema_version": 1,
        "purpose": "frozen_nextgqa_cohort_prior_exposure_audit",
        "status": "completed",
        "requested_frozen_pairs": len(frozen),
        "mapped_frozen_pairs": len(pair_map),
        "packet_records_mapped": len(packet_by_pair),
        "known_video_id_matches": len(video_tokens & prior_video),
        "known_video_qid_pair_matches": len(pair_tokens & prior_pair),
        "known_packet_claimed_media_sha256_matches": len(media_tokens & prior_media),
        "packet_pair_digest_binding_present": packet_pair_binding_present,
        "packet_pair_digest_binding_authenticated": False,
        "expected_pair_to_media_crosswalk_verified": False,
        "packet_media_matches_are_packet_claims_only": True,
        "input_sha256": {
            "split_csv": hashlib.sha256(split_raw).hexdigest(),
            "frozen_pairs": hashlib.sha256(frozen_pairs_raw).hexdigest(),
            "run_packet": hashlib.sha256(packet_raw).hexdigest(),
            "prior_indexes": [hashlib.sha256(raw).hexdigest() for raw in index_raws],
        },
        "hmac_key_to_index_binding_authenticated": False,
        "historical_completeness_authenticated": False,
        "freshness_conclusion": "not_made",
        "audit_scored_faithfulness": False,
        "audit_used_gpu": False,
        "execution_authorization": "not_granted",
    }


def _read(path: Path, *, private: bool = False) -> bytes:
    try:
        return _read_stable_regular_file(path, max_bytes=MAX_SOURCE_BYTES,
                                         required_mode=0o600 if private else None)
    except (OSError, ValueError) as exc:
        raise ValueError("input_file_invalid") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--frozen-pairs", required=True, type=Path)
    parser.add_argument("--run-packet", required=True, type=Path,
                        help="private aggregate generation packet with QID and media SHA rows")
    parser.add_argument("--prior-index", required=True, action="append", type=Path)
    parser.add_argument("--hmac-key", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if len(args.prior_index) > MAX_INDEX_COUNT:
            raise ValueError("prior_index_count_invalid")
        report = audit_frozen_candidate_exposure(
            split_raw=_read(args.split_csv),
            frozen_pairs_raw=_read(args.frozen_pairs, private=True),
            packet_raw=_read(args.run_packet, private=True),
            index_raws=[_read(path, private=True) for path in args.prior_index],
            hmac_key=_read_secret(args.hmac_key),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "audit_failed")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
