#!/usr/bin/env python3
"""Aggregate-only canonical video-ID match against a private source ledger.

Optional exact source-byte exclusions remove self-sources from match counts;
ledger HMAC, schema, token-list, and source-union checks still run first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any

try:
    from audit_nextgqa_split_prior_identity import _read_split
    from build_nextgqa_prior_exposure_manifest import (
        FAMILIES, MAX_SOURCE_BYTES, NAMESPACE, _read_secret,
        _read_stable_regular_file, _token,
    )
    from build_nextgqa_prior_exposure_source_ledger import LEDGER_DOMAIN, _canonical
except ImportError:
    from scripts.audit_nextgqa_split_prior_identity import _read_split
    from scripts.build_nextgqa_prior_exposure_manifest import (
        FAMILIES, MAX_SOURCE_BYTES, NAMESPACE, _read_secret,
        _read_stable_regular_file, _token,
    )
    from scripts.build_nextgqa_prior_exposure_source_ledger import LEDGER_DOMAIN, _canonical

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_SOURCES = 64


def audit_split_against_source_ledger(*, split_raw: bytes, ledger_raw: bytes,
                                      secret: bytes,
                                      exclude_source_raws: list[bytes] | tuple[bytes, ...] = ()) -> dict[str, Any]:
    if not isinstance(split_raw, bytes) or len(split_raw) > MAX_SOURCE_BYTES:
        raise ValueError("split_size_invalid")
    if not isinstance(ledger_raw, bytes) or len(ledger_raw) > MAX_SOURCE_BYTES:
        raise ValueError("ledger_size_invalid")
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("secret_invalid")
    if (not isinstance(exclude_source_raws, (list, tuple))
            or len(exclude_source_raws) > MAX_SOURCES
            or sum(len(raw) for raw in exclude_source_raws if isinstance(raw, bytes))
            > MAX_SOURCE_BYTES
            or any(not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES
                   for raw in exclude_source_raws)):
        raise ValueError("excluded_source_bytes_invalid")
    try:
        ledger = json.loads(ledger_raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("ledger_json_invalid") from exc
    if not isinstance(ledger, dict):
        raise ValueError("ledger_schema_invalid")
    provided_mac = ledger.pop("ledger_integrity_hmac_sha256", None)
    if not isinstance(provided_mac, str) or not HEX64.fullmatch(provided_mac):
        raise ValueError("ledger_authentication_missing")
    expected_mac = hmac.new(secret, LEDGER_DOMAIN + _canonical(ledger), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided_mac, expected_mac):
        raise ValueError("ledger_authentication_failed")
    if (not isinstance(ledger, dict) or type(ledger.get("schema_version")) is not int
            or ledger.get("schema_version") != 1
            or ledger.get("purpose") != "source_attributed_prior_exposure_hmac_ledger"
            or ledger.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(ledger.get("sources"), list)
            or not 1 <= len(ledger["sources"]) <= MAX_SOURCES
            or not isinstance(ledger.get("known_identity_unions"), dict)
            or not isinstance(ledger.get("coverage"), dict)):
        raise ValueError("ledger_schema_invalid")
    excluded_fingerprints = {
        _token(secret, "source-file-sha256", (hashlib.sha256(raw).hexdigest(),))
        for raw in exclude_source_raws
    }
    if len(excluded_fingerprints) != len(exclude_source_raws):
        raise ValueError("excluded_source_duplicate")
    union = ledger["known_identity_unions"].get("dataset_video_hmac_sha256")
    media_union = ledger["known_identity_unions"].get("media_sha256_hmac_sha256")
    pair_union = ledger["known_identity_unions"].get("video_question_pair_hmac_sha256")
    all_fields = (union, media_union, pair_union)
    if any(not isinstance(values, list)
           or any(not isinstance(token, str) or not HEX64.fullmatch(token) for token in values)
           or len(values) != len(set(values)) for values in all_fields):
        raise ValueError("ledger_union_invalid")
    seen_sources = set()
    source_unions = {name: set() for name in
                     ("dataset_video_hmac_sha256", "media_sha256_hmac_sha256",
                      "video_question_pair_hmac_sha256")}
    canonical_video_union: set[str] = set()
    by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"source_count": 0, "sources_with_video_id_match": 0,
                 "matched_unique_video_ids": 0})
    canonical_source_count = 0
    compared_source_count = 0
    found_exclusions: set[str] = set()
    video_ids, _ = _read_split(split_raw)
    candidates = {_token(secret, "dataset-video-id", ("nextgqa", value))
                  for value in video_ids}
    for row in ledger["sources"]:
        if (not isinstance(row, dict)
                or set(row) != {"source_ordinal", "source_id_hmac_sha256", "source_class",
                                "identity_namespace", "status", "rows_total",
                                "rows_with_any_identity", "rows_with_video_id",
                                "rows_with_media_sha256", "rows_with_video_question_pair",
                                "video_identity_hmac_sha256", "media_identity_hmac_sha256",
                                "video_question_pair_hmac_sha256", "unresolved_reason_counts"}
                or not isinstance(row.get("source_class"), str)
                or row["source_class"] not in FAMILIES
                or len(row["source_class"]) > 32
                or not isinstance(row.get("identity_namespace"), str)
                or len(row["identity_namespace"]) > 64
                or NAMESPACE.fullmatch(row["identity_namespace"]) is None
                or not row["identity_namespace"]
                or not isinstance(row.get("source_class"), str)
                or not isinstance(row.get("source_id_hmac_sha256"), str)
                or not HEX64.fullmatch(row["source_id_hmac_sha256"])
                or row["source_id_hmac_sha256"] in seen_sources):
            raise ValueError("ledger_source_invalid")
        if (type(row.get("source_ordinal")) is not int or row["source_ordinal"] <= 0
                or row.get("status") not in {"parsed", "parsed_provenance_unverified",
                                              "opaque_unresolved", "unresolved_invalid_source"}
                or (row.get("rows_with_any_identity") is not None
                    and (type(row["rows_with_any_identity"]) is not int
                         or row["rows_with_any_identity"] < 0))):
            raise ValueError("ledger_source_metadata_invalid")
        seen_sources.add(row["source_id_hmac_sha256"])
        if row["source_id_hmac_sha256"] in excluded_fingerprints:
            found_exclusions.add(row["source_id_hmac_sha256"])
        tokens = row.get("video_identity_hmac_sha256")
        if (not isinstance(tokens, list)
                or any(not isinstance(token, str) or not HEX64.fullmatch(token) for token in tokens)
                or len(tokens) != len(set(tokens))):
            raise ValueError("ledger_source_video_tokens_invalid")
        media_tokens = row.get("media_identity_hmac_sha256")
        pair_tokens = row.get("video_question_pair_hmac_sha256")
        for values, error in ((media_tokens, "ledger_source_media_tokens_invalid"),
                              (pair_tokens, "ledger_source_pair_tokens_invalid")):
            if (not isinstance(values, list)
                    or any(not isinstance(token, str) or not HEX64.fullmatch(token)
                           for token in values)
                    or len(values) != len(set(values))):
                raise ValueError(error)
        source_unions["dataset_video_hmac_sha256"].update(tokens)
        source_unions["media_sha256_hmac_sha256"].update(media_tokens)
        source_unions["video_question_pair_hmac_sha256"].update(pair_tokens)
        if row["identity_namespace"] != "nextgqa":
            continue
        canonical_source_count += 1
        if row["source_id_hmac_sha256"] in excluded_fingerprints:
            continue
        compared_source_count += 1
        canonical_video_union.update(tokens)
        matches = len(candidates & set(tokens))
        aggregate = by_class[row["source_class"]]
        aggregate["source_count"] += 1
        aggregate["sources_with_video_id_match"] += int(matches > 0)
        aggregate["matched_unique_video_ids"] += matches
    if found_exclusions != excluded_fingerprints:
        raise ValueError("excluded_source_not_in_ledger")
    for values, name in zip(all_fields, source_unions, strict=True):
        if source_unions[name] != set(values):
            raise ValueError("ledger_source_union_mismatch")
    coverage = ledger["coverage"]
    expected_identity_sources = sum(
        type(row.get("rows_with_any_identity")) is int
        and row["rows_with_any_identity"] > 0 for row in ledger["sources"])
    if (coverage.get("historical_completeness_authenticated") is not False
            or coverage.get("hmac_key_to_source_binding_authenticated") is not False
            or coverage.get("freshness_conclusion") != "not_made"
            or coverage.get("execution_authorization") != "not_granted"
            or type(coverage.get("source_count")) is not int
            or coverage.get("source_count") != len(ledger["sources"])
            or type(coverage.get("sources_with_identity_rows")) is not int
            or coverage.get("sources_with_identity_rows") != expected_identity_sources):
        raise ValueError("ledger_coverage_flags_invalid")
    matched_union = candidates & canonical_video_union
    return {
        "schema_version": 1,
        "purpose": "nextgqa_split_source_attributed_prior_identity_audit",
        "status": "completed",
        "scope": "exact_nextgqa_video_id_hmac_matches_against_explicit_source_records",
        "aggregation_scope": ("explicit_sources_excluding_exact_source_bytes"
                               if found_exclusions else "all_explicit_sources"),
        "split_unique_video_ids": len(video_ids),
        "source_record_count": len(ledger["sources"]),
        "canonical_nextgqa_source_record_count": canonical_source_count,
        "compared_nextgqa_source_record_count": compared_source_count,
        "excluded_source_record_count": len(found_exclusions),
        "prior_unique_nextgqa_video_id_tokens": len(canonical_video_union),
        "matched_video_ids_in_source_union": len(matched_union),
        "matched_source_records_by_class": dict(sorted(by_class.items())),
        "ledger_integrity_verified": True,
        "hmac_key_to_source_binding_authenticated": False,
        "historical_completeness_authenticated": False,
        "freshness_conclusion": "not_made",
        "media_byte_identity_checked": False,
        "faithfulness_scored": False,
        "gpu_used": False,
        "execution_authorization": "not_granted",
    }


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--hmac-key", required=True, type=Path)
    parser.add_argument("--exclude-source", action="append", default=[], type=Path,
                        help="exact source file included in the ledger; exclude it from match counts (repeatable)")
    args = parser.parse_args(argv)
    try:
        split = _read_stable_regular_file(args.split_csv, max_bytes=MAX_SOURCE_BYTES)
        ledger = _read_stable_regular_file(args.source_ledger, max_bytes=MAX_SOURCE_BYTES,
                                           required_mode=0o600)
        key = _read_secret(args.hmac_key)
        excluded = [_read_stable_regular_file(path, max_bytes=MAX_SOURCE_BYTES)
                    for path in args.exclude_source]
        report = audit_split_against_source_ledger(split_raw=split, ledger_raw=ledger,
                                                   secret=key,
                                                   exclude_source_raws=excluded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "audit_failed")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
