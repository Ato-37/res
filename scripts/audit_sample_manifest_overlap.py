"""Compare a proposed sample manifest with historical manifests by video identity.

This CPU-only audit prints counts and manifest hashes, never sample identifiers,
paths, questions, or answers. It does not compare decoded content or establish
statistical independence, annotation validity, data rights, or permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load(raw: bytes) -> Any:
    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))

    def check(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(parsed)
    return parsed


def _identity(row: dict) -> str:
    video_id = row.get("video_id")
    video = row.get("video")
    if "video_id" in row and (not isinstance(video_id, str) or not video_id.strip()):
        raise ValueError("video_id must be a non-empty string when supplied")
    if "video_id" in row and video_id != video_id.strip():
        raise ValueError("video_id must not have surrounding whitespace")
    if "video" in row and (not isinstance(video, str) or not video.strip()):
        raise ValueError("video must be a non-empty string when supplied")
    # Treat an explicit dataset ID as opaque: dotted IDs must not be truncated
    # as if their last segment were a filename extension.
    id_value = video_id if "video_id" in row else None
    path_value = PurePosixPath(video.strip().replace("\\", "/")).stem if "video" in row else None
    if id_value is not None and path_value is not None and id_value != path_value:
        raise ValueError("video_id and video stem disagree")
    identity = id_value or path_value
    if not identity or identity in {".", ".."}:
        raise ValueError("sample has no valid video identity")
    return identity


def _parse_manifest(raw: bytes) -> tuple[list[str], str]:
    rows = _load(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("sample manifest must be a non-empty JSON array")
    ids = [_identity(row) for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows):
        raise ValueError("sample manifest rows must be objects")
    return ids, hashlib.sha256(raw).hexdigest()


def audit(candidate_raw: bytes, historical_raws: list[bytes]) -> dict:
    candidate_ids, candidate_hash = _parse_manifest(candidate_raw)
    if not historical_raws:
        raise ValueError("at least one historical manifest is required")
    candidate_set = set(candidate_ids)
    per_manifest = []
    historical_union: set[str] = set()
    historical_duplicate_rows_within_manifest = 0
    historical_manifest_membership_counts: dict[str, int] = {}
    for index, raw in enumerate(historical_raws, start=1):
        ids, digest = _parse_manifest(raw)
        id_set = set(ids)
        historical_union.update(id_set)
        historical_duplicate_rows_within_manifest += len(ids) - len(id_set)
        for identity in id_set:
            historical_manifest_membership_counts[identity] = historical_manifest_membership_counts.get(identity, 0) + 1
        per_manifest.append({
            "manifest_label": f"historical_{index:02d}",
            "requested_rows": len(ids),
            "unique_video_identities": len(id_set),
            "duplicate_identity_rows": len(ids) - len(id_set),
            "candidate_identity_overlap": len(candidate_set & id_set),
            "manifest_sha256": digest,
        })
    overlap = candidate_set & historical_union
    return {
        "schema_version": 1,
        "status": "audit_complete",
        "candidate_identity_status": "unique" if len(candidate_ids) == len(candidate_set) else "duplicates_present",
        "candidate_requested_rows": len(candidate_ids),
        "candidate_unique_video_identities": len(candidate_set),
        "candidate_duplicate_identity_rows": len(candidate_ids) - len(candidate_set),
        "historical_manifest_count": len(historical_raws),
        "historical_unique_video_identities": len(historical_union),
        "historical_duplicate_identity_rows_within_manifests": historical_duplicate_rows_within_manifest,
        "historical_identities_shared_across_manifests": sum(count > 1 for count in historical_manifest_membership_counts.values()),
        "candidate_identities_seen_historically": len(overlap),
        "candidate_identities_unseen_historically": len(candidate_set - historical_union),
        "candidate_manifest_sha256": candidate_hash,
        "per_historical_manifest": per_manifest,
        "comparison_scope": "opaque video_id equality or video filename-stem equality; possible false overlap for repeated basenames; no content or statistical-independence audit",
        "data_rights_or_execution_authorization": "not_granted",
        "hash_privacy_note": "SHA-256 values are unsalted and may be linkable or guessable",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    try:
        result = audit(args.candidate.read_bytes(), [p.read_bytes() for p in args.historical])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.chmod(0o600)
    except Exception as exc:
        parser.error(f"audit failed: {type(exc).__name__}")
    print(json.dumps({k: v for k, v in result.items() if k != "per_historical_manifest"}, sort_keys=True))
    return 0 if result["candidate_identity_status"] == "unique" else 2


if __name__ == "__main__":
    raise SystemExit(main())
