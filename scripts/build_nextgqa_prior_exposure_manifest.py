#!/usr/bin/env python3
"""Build a private, outcome-blind prior-exposure index from explicit inputs.

This CPU-only tool accepts only paths named on the command line. Repeat
``--input FAMILY@NAMESPACE=PATH`` for each source. Supported families are:

* ``sample-json``: JSON array of rows with explicit ``video_id`` and optional
  ``qid``. A path-only ``video`` field is never canonicalized.
* ``lineage-json``: object with ``records``; rows may contain
  ``dataset_video_id``, ``dataset_question_id``, ``video_sha256``.
* ``lock-json``: version-1 object with ``assignments`` containing
  ``media_sha256`` and ``qid_sha256``. Only media identity is usable; the
  pre-hashed QID is opaque and is not treated as a recoverable pair.
* ``media-set-json``: object with exactly one ``media_sha256`` array of
  lowercase SHA-256 strings. Hash identities are usable, while source purpose
  and completeness remain unverified.
* ``review-custody-json``: object with a top-level ``records`` array. Each row
  must contain ``source_media_sha256`` as lowercase SHA-256. Other envelope and
  row fields are ignored; custody metadata does not prove human viewing.
* ``result-json``: JSON object/list with explicit per-record ``video_id``,
  ``qid`` and/or ``video_sha256`` fields.
* ``result-csv``: CSV with the same explicit column names.
* ``opaque-legacy``: arbitrary legacy exclusion data. The file is fingerprinted
  and its rows are counted when safely parseable, but no values are interpreted.

Every source is explicitly allowlisted by the caller; there is no repository
crawl. Identity values are converted to keyed HMAC-SHA256 tokens. Raw IDs,
paths, questions, answers, and source filenames are never written to output.
The output is private, created exclusively with mode 0600, and is not a cohort
selection or an assertion that historical coverage is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA_VERSION = 1
FAMILIES = {"sample-json", "lineage-json", "lock-json", "media-set-json",
            "review-custody-json", "result-json", "result-csv", "opaque-legacy"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _loads(raw: bytes) -> Any:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json") from exc
    return value


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _token(key: bytes, domain: str, parts: tuple[str, ...]) -> str:
    # JSON provides an unambiguous, versioned tuple encoding (including embedded separators).
    payload = json.dumps(["nextgqa-exposure-v1", domain, *parts],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _rows_for(family: str, raw: bytes) -> tuple[list[dict], list[str]]:
    """Return identity-bearing row objects plus any nonfatal row issues."""
    if family == "opaque-legacy":
        # Count only a top-level array; never inspect or interpret its values.
        try:
            value = _loads(raw)
        except ValueError:
            return [], ["opaque_legacy_row_count_unknown"]
        if isinstance(value, list):
            count = len(value)
        else:
            return [], ["opaque_legacy_row_count_unknown", "opaque_legacy_values_not_interpreted"]
        return [{} for _ in range(count)], ["opaque_legacy_values_not_interpreted"]
    if family == "result-csv":
        try:
            table = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="strict"), newline="")))
            if not table:
                raise ValueError("csv_header_missing")
            header = table[0]
            if (not header or any(not name for name in header)
                    or len(header) != len(set(header))):
                raise ValueError("csv_header_invalid")
            if any(len(row) != len(header) for row in table[1:]):
                raise ValueError("csv_row_width_mismatch")
            return [dict(zip(header, row, strict=True)) for row in table[1:]], []
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("invalid_csv") from exc
    value = _loads(raw)
    if family == "sample-json":
        if not isinstance(value, list):
            raise ValueError("sample_json_must_be_array")
        rows = value
    elif family == "lineage-json":
        if not isinstance(value, dict) or not isinstance(value.get("records"), list):
            raise ValueError("lineage_records_missing")
        rows = value["records"]
    elif family == "lock-json":
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value.get("schema_version") != 1
                or not isinstance(value.get("assignments"), list)):
            raise ValueError("lock_assignments_missing")
        rows = value["assignments"]
    elif family == "media-set-json":
        allowed_shapes = ({"media_sha256"}, {"media_sha256", "schema_version"},
                          {"media_sha256", "purpose"})
        if not isinstance(value, dict) or set(value) not in allowed_shapes:
            raise ValueError("media_set_shape_invalid")
        if ("schema_version" in value
                and (type(value["schema_version"]) is not int or value["schema_version"] != 1)):
            raise ValueError("media_set_schema_version_invalid")
        if ("purpose" in value
                and (not isinstance(value["purpose"], str) or not value["purpose"].strip())):
            raise ValueError("media_set_purpose_invalid")
        hashes = value.get("media_sha256")
        if (not isinstance(hashes, list) or any(not _sha(item) for item in hashes)):
            raise ValueError("media_set_hashes_invalid")
        return [{"media_sha256": item} for item in hashes], []
    elif family == "review-custody-json":
        if not isinstance(value, dict) or not isinstance(value.get("records"), list):
            raise ValueError("review_custody_records_missing")
        rows = value["records"]
        if any(not isinstance(row, dict) or not _sha(row.get("source_media_sha256"))
               for row in rows):
            raise ValueError("review_custody_source_media_sha256_invalid")
        return [{"source_media_sha256": row["source_media_sha256"]} for row in rows], []
    elif family == "result-json":
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            # Only known row containers; aggregate-only objects remain unresolved.
            candidates = [value[k] for k in ("records", "rows", "samples", "results")
                          if isinstance(value.get(k), list)]
            if len(candidates) != 1:
                raise ValueError("result_json_row_container_missing_or_ambiguous")
            rows = candidates[0]
        else:
            raise ValueError("result_json_shape_invalid")
    else:
        raise ValueError("source_family_invalid")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("source_row_not_object")
    return rows, []


def _extract(family: str, namespace: str, rows: list[dict], key: bytes) -> tuple[set[str], set[str], set[str], dict[str, int], dict[str, int]]:
    videos: set[str] = set()
    media: set[str] = set()
    pairs: set[str] = set()
    row_counts = {"rows_with_any_identity": 0, "rows_with_video_id": 0,
                  "rows_with_media_sha256": 0, "rows_with_video_question_pair": 0}
    issues: dict[str, int] = {}

    for row in rows:
        got_video = got_media = got_pair = False
        if family in {"lock-json", "media-set-json", "review-custody-json"}:
            media_sha = (row.get("source_media_sha256") if family == "review-custody-json"
                         else row.get("media_sha256"))
            if _sha(media_sha):
                media.add(_token(key, "media-sha256", (media_sha,)))
                got_media = True
            else:
                issues["invalid_or_missing_media_sha256"] = issues.get("invalid_or_missing_media_sha256", 0) + 1
            if family == "lock-json":
                issues["qid_hash_is_opaque_pair_unresolved"] = issues.get("qid_hash_is_opaque_pair_unresolved", 0) + 1
        else:
            video_id = row.get("dataset_video_id") if family == "lineage-json" else row.get("video_id")
            qid = row.get("dataset_question_id") if family == "lineage-json" else row.get("qid")
            media_sha = row.get("video_sha256")
            # Explicit dataset IDs are accepted as opaque source identifiers. Never derive from paths.
            if _text(video_id):
                videos.add(_token(key, "dataset-video-id", (namespace, video_id)))
                got_video = True
            if _sha(media_sha):
                media.add(_token(key, "media-sha256", (media_sha,)))
                got_media = True
            if _text(qid):
                pair_anchor = ("dataset-video-id", namespace, video_id) if _text(video_id) else (
                    ("media-sha256", media_sha) if _sha(media_sha) else None)
                if pair_anchor:
                    pairs.add(_token(key, "video-question-pair", (namespace, *pair_anchor, qid)))
                    got_pair = True
                else:
                    issues["qid_without_video_identity"] = issues.get("qid_without_video_identity", 0) + 1
            if family == "sample-json" and not got_video and isinstance(row.get("video"), str):
                issues["path_only_video_unresolved"] = issues.get("path_only_video_unresolved", 0) + 1
            if family == "lineage-json" and row.get("video_id") and not got_video:
                issues["lineage_video_id_requires_dataset_video_id"] = issues.get("lineage_video_id_requires_dataset_video_id", 0) + 1
        if got_video:
            row_counts["rows_with_video_id"] += 1
        if got_media:
            row_counts["rows_with_media_sha256"] += 1
        if got_pair:
            row_counts["rows_with_video_question_pair"] += 1
        if got_video or got_media or got_pair:
            row_counts["rows_with_any_identity"] += 1
        elif family != "lock-json":
            issues["row_without_usable_identity"] = issues.get("row_without_usable_identity", 0) + 1
    return videos, media, pairs, row_counts, issues


def build(inputs: list[tuple[str, str, str, bytes]], secret: bytes) -> tuple[dict, bytes]:
    """Build from `(family, namespace, private_locator, raw_bytes)` tuples.

    Callers must validate that video IDs use the declared namespace; no
    crosswalk or canonical-ID validation is performed here.
    """
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("secret_must_be_at_least_32_bytes")
    seen_locators = set()
    seen_fingerprints = set()
    summaries = []
    all_video: set[str] = set()
    all_media: set[str] = set()
    all_pairs: set[str] = set()
    for family, namespace, locator, raw in inputs:
        if family not in FAMILIES:
            raise ValueError("source_family_invalid")
        if not isinstance(namespace, str) or NAMESPACE.fullmatch(namespace) is None:
            raise ValueError("source_namespace_invalid")
        if not isinstance(raw, bytes):
            raise ValueError("source_bytes_invalid")
        if locator in seen_locators:
            raise ValueError("duplicate_source_locator")
        seen_locators.add(locator)
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash in seen_fingerprints:
            raise ValueError("duplicate_source_content")
        seen_fingerprints.add(source_hash)
        issues: dict[str, int] = {}
        try:
            rows, parse_issues = _rows_for(family, raw)
            for issue in parse_issues:
                issues[issue] = issues.get(issue, 0) + 1
            total = len(rows)
            videos, media, pairs, row_counts, extraction_issues = _extract(family, namespace, rows, secret)
            if family == "opaque-legacy":
                row_counts = {name: None for name in row_counts}
            for issue, count in extraction_issues.items():
                issues[issue] = issues.get(issue, 0) + count
            if family == "opaque-legacy":
                status = "opaque_unresolved"
            elif family in {"media-set-json", "review-custody-json"}:
                status = "parsed_provenance_unverified"
            else:
                status = "parsed"
        except (ValueError, TypeError, KeyError) as exc:
            total, row_counts = None, {name: None for name in
                                       ("rows_with_any_identity", "rows_with_video_id",
                                        "rows_with_media_sha256", "rows_with_video_question_pair")}
            videos, media, pairs = set(), set(), set()
            reason = str(exc)
            if reason in {"result_json_row_container_missing_or_ambiguous",
                          "csv_row_width_mismatch", "csv_header_missing", "csv_header_invalid"}:
                issues[reason] = 1
            else:
                issues["source_parse_failed"] = 1
            status = "unresolved_invalid_source"
        all_video.update(videos); all_media.update(media); all_pairs.update(pairs)
        summaries.append({
            "source_class": family,
            "identity_namespace": namespace,
            "source_fingerprint_hmac_sha256": _token(secret, "source-file-sha256", (source_hash,)),
            "status": status,
            "rows_total": (None if total is None or
                           "opaque_legacy_row_count_unknown" in issues else total),
            **row_counts,
            "rows_without_any_identity": (None if total is None or
                                            "opaque_legacy_row_count_unknown" in issues or
                                            family == "opaque-legacy" else
                                            max(0, total - row_counts["rows_with_any_identity"])),
            "video_identity_count": len(videos),
            "media_identity_count": len(media),
            "video_question_pair_count": len(pairs),
            "unresolved_reason_counts": dict(sorted(issues.items())),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "outcome_blind_prior_exposure_index",
        "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
        "sources": summaries,
        "identities": {
            "dataset_video_hmac_sha256": sorted(all_video),
            "media_sha256_hmac_sha256": sorted(all_media),
            "video_question_pair_hmac_sha256": sorted(all_pairs),
        },
        "coverage": {
            "source_count": len(summaries),
            "sources_with_identity_rows": sum(
                (s["rows_with_any_identity"] or 0) > 0 for s in summaries),
            "aggregate_only_history_recoverable": False,
            "historical_completeness_authenticated": False,
        },
        "limitations": [
            "explicit inputs only; no repository crawl",
            "video IDs are not crosswalk-validated; caller must ensure the declared namespace is correct",
            "path-only video values are unresolved",
            "opaque legacy exclusion values are not interpreted",
            "aggregate-only, missing, deleted, or identity-redacted history cannot be reconstructed",
            "HMAC equality supports linkage only under the same secret and canonicalization version",
            "video identity and media-byte identity are separate; transcoded derivatives may not match",
        ],
    }
    encoded = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return manifest, encoded


def _read_secret(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError("secret_file_unavailable") from exc
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("secret_file_must_be_regular_mode_0600")
    raw = path.read_bytes()
    if len(raw) < 32:
        raise ValueError("secret_must_be_at_least_32_bytes")
    return raw


def _write_exclusive_private(path: Path, raw: bytes) -> None:
    parent = path.parent
    try:
        parent_info = parent.lstat()
    except OSError as exc:
        raise ValueError("output_parent_unavailable") from exc
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ValueError("output_parent_must_be_real_directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise ValueError("output_already_exists") from None
    except OSError as exc:
        raise ValueError("output_unavailable") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        raise ValueError("output_write_failed") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="FAMILY@NAMESPACE=PATH",
                        help="explicit allowlisted source; repeat as needed")
    parser.add_argument("--secret-file", required=True, type=Path,
                        help="local key file, regular file with exact mode 0600 and at least 32 bytes")
    parser.add_argument("--output", required=True, type=Path,
                        help="new private output path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        secret = _read_secret(args.secret_file)
        inputs = []
        for item in args.input:
            family_namespace, sep, raw_path = item.partition("=")
            family, at, namespace = family_namespace.partition("@")
            if (not sep or not at or family not in FAMILIES or not raw_path
                    or NAMESPACE.fullmatch(namespace) is None):
                raise ValueError("input_must_be_supported_family_at_namespace_equals_path")
            path = Path(raw_path)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("input_must_be_regular_file")
            inputs.append((family, namespace, str(path), path.read_bytes()))
        _manifest, encoded = build(inputs, secret)
        _write_exclusive_private(args.output, encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "manifest_build_failed")
    print("Prior-exposure manifest written privately; source and identity values were not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
