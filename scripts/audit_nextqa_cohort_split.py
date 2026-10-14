"""Privacy-safe cohort membership check against local NExT-QA video splits.

This checks filename-stem membership only. It cannot authenticate split
provenance or establish license, rights, source-media identity, or permission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def _stem(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).stem


def audit(cohort: Any, cohort_raw: bytes, split_rows: dict[str, Any], split_raw: dict[str, bytes]) -> dict:
    failures: set[str] = set()
    if not isinstance(cohort, list) or not cohort:
        failures.add("cohort_manifest_invalid")
        cohort = []
    cohort_ids = []
    for row in cohort:
        if not isinstance(row, dict) or not isinstance(row.get("video"), str) or not row["video"].strip():
            failures.add("cohort_video_field_invalid")
            continue
        video_id = _stem(row["video"])
        if not video_id:
            failures.add("cohort_video_stem_invalid")
            continue
        cohort_ids.append(video_id)
    if len(cohort_ids) != len(cohort):
        failures.add("cohort_video_rows_unresolved")
    if len(cohort_ids) != len(set(cohort_ids)):
        failures.add("cohort_video_stems_duplicate")

    sets = {}
    for split in ("train", "val", "test"):
        rows = split_rows.get(split)
        if not isinstance(rows, list) or not rows:
            failures.add(f"{split}_split_invalid")
            sets[split] = set()
            continue
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("video_id"), str) or not row["video_id"].strip():
                failures.add(f"{split}_split_video_field_invalid")
                continue
            value = _stem(row["video_id"])
            if not value:
                failures.add(f"{split}_split_video_stem_invalid")
                continue
            ids.add(value)
        sets[split] = ids

    overlaps = {f"{a}_{b}": len(sets[a] & sets[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    memberships = {split: sum(video_id in ids for video_id in cohort_ids)
                   for split, ids in sets.items()}
    multi_split = sum(sum(video_id in ids for ids in sets.values()) > 1 for video_id in cohort_ids)
    matched = sum(video_id in set.union(*sets.values()) for video_id in cohort_ids) if sets else 0
    unmatched = len(cohort_ids) - matched
    if multi_split:
        failures.add("source_split_membership_ambiguous")
    if unmatched:
        failures.add("cohort_video_unmatched_to_source_splits")
    return {
        "schema_version": 1,
        "status": "mapped_unambiguously" if not failures else "incomplete_or_invalid",
        "cohort_rows": len(cohort),
        "cohort_unique_video_ids": len(set(cohort_ids)),
        "cohort_matches_by_split": memberships,
        "source_unique_video_ids_by_split": {split: len(ids) for split, ids in sets.items()},
        "source_split_pairwise_overlaps": overlaps,
        "cohort_ids_in_multiple_splits": multi_split,
        "cohort_ids_unmatched": unmatched,
        "failure_codes": sorted(failures),
        "cohort_manifest_sha256": hashlib.sha256(cohort_raw).hexdigest(),
        "source_split_sha256": {split: hashlib.sha256(split_raw[split]).hexdigest()
                                for split in ("train", "val", "test") if split in split_raw},
        "semantic_validation": "not_machine_verifiable",
        "rights_or_inference_authorization": "not_granted",
    }


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return _decode(raw), raw


def _read_split(path: Path) -> tuple[list[dict], bytes]:
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8", errors="strict"), newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("split CSV is empty") from exc
    if not header or len(header) != len(set(header)) or "video_id" not in header:
        raise ValueError("split CSV header is invalid")
    rows = []
    try:
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError("split CSV row width is invalid")
            rows.append(dict(zip(header, values)))
    except csv.Error as exc:
        raise ValueError("split CSV is malformed") from exc
    return rows, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--nextqa-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        cohort, cohort_raw = _read_json(args.cohort_manifest)
        split_rows, split_raw = {}, {}
        for split in ("train", "val", "test"):
            split_rows[split], split_raw[split] = _read_split(args.nextqa_dir / f"{split}.csv")
        result = audit(cohort, cohort_raw, split_rows, split_raw)
    except Exception:
        result = {
            "schema_version": 1, "status": "input_error", "cohort_rows": 0,
            "cohort_unique_video_ids": 0, "cohort_matches_by_split": {},
            "source_unique_video_ids_by_split": {}, "source_split_pairwise_overlaps": {},
            "cohort_ids_in_multiple_splits": 0, "cohort_ids_unmatched": 0,
            "failure_codes": ["input_unreadable_or_malformed"],
            "cohort_manifest_sha256": None, "source_split_sha256": {},
            "semantic_validation": "not_machine_verifiable",
            "rights_or_inference_authorization": "not_granted",
        }
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "mapped_unambiguously" else 2


if __name__ == "__main__":
    raise SystemExit(main())
