"""Privacy-safe audit of NExT-GQA video-name to VidOR-ID map coverage.

This checks only a local filename-stem join. It does not establish source-media
license, mapping evidence authenticity, or permission to run inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _decode(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def _read(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return _decode(raw), raw


def audit(rows: Any, rows_raw: bytes, mapping: Any, mapping_raw: bytes) -> dict:
    failures: set[str] = set()
    if not isinstance(rows, list) or not rows:
        failures.add("sample_manifest_invalid")
        rows = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("video"), str) or not row["video"].strip():
            failures.add("sample_video_field_invalid")
            continue
        # Dataset map keys are video filename stems; discard any local path and
        # extension so this audit never places private paths in the summary.
        video_value = row["video"].strip().replace("\\", "/")
        video_path = PurePosixPath(video_value)
        if (video_value.endswith("/") or not video_path.name
                or not video_path.suffix or video_path.suffix == "."):
            failures.add("sample_video_stem_invalid")
            continue
        stem = video_path.stem
        if not stem:
            failures.add("sample_video_stem_invalid")
            continue
        names.append(stem)
    if len(names) != len(rows):
        failures.add("sample_video_rows_unresolved")
    if len(names) != len(set(names)):
        failures.add("sample_video_stems_duplicate")
    if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in mapping.items()):
        failures.add("video_mapping_invalid")
        mapping = {}
    targets = [posixpath.normpath(mapping[name].strip().replace("\\", "/"))
               for name in names if name in mapping]
    if any(target in {"", "."} for target in targets):
        failures.add("video_mapping_target_invalid")
    missing = sum(name not in mapping for name in names)
    if len(targets) != len(set(targets)):
        failures.add("mapped_video_paths_duplicate")
    return {
        "schema_version": 1,
        "status": "mapped" if not failures and not missing else "incomplete_or_invalid",
        "requested_rows": len(rows),
        "valid_video_stems": len(names),
        "mapped_rows": len(targets),
        "unmapped_rows": missing,
        "unique_mapped_video_paths": len(set(targets)),
        "failure_codes": sorted(failures),
        "sample_manifest_sha256": hashlib.sha256(rows_raw).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "semantic_validation": "not_machine_verifiable",
        "rights_or_inference_authorization": "not_granted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--video-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        rows, rows_raw = _read(args.sample_manifest)
        mapping, mapping_raw = _read(args.video_map)
        result = audit(rows, rows_raw, mapping, mapping_raw)
    except Exception:
        result = {
            "schema_version": 1, "status": "input_error", "requested_rows": 0,
            "valid_video_stems": 0, "mapped_rows": 0, "unmapped_rows": 0,
            "unique_mapped_video_paths": 0, "failure_codes": ["input_unreadable_or_malformed"],
            "sample_manifest_sha256": None, "mapping_sha256": None,
            "semantic_validation": "not_machine_verifiable",
            "rights_or_inference_authorization": "not_granted",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "mapped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
