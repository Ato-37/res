"""Privacy-safe audit of NExT-GQA video-name to VidOR-ID map coverage.

This checks only a local filename-stem join. It does not establish source-media
license, mapping evidence authenticity, or permission to run inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath, PureWindowsPath
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


def audit(rows: Any, rows_raw: bytes, mapping: Any, mapping_raw: bytes,
          video_root: Path | None = None) -> dict:
    failures: set[str] = set()
    if not isinstance(rows, list) or not rows:
        failures.add("sample_manifest_invalid")
        rows = []
    names = []
    row_paths = []
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
        row_paths.append(row["video"].strip())
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
    local_path_audit = None
    if video_root is not None:
        root = video_root.expanduser().resolve()
        matched = present = outside = invalid_target = 0
        for name, raw_path in zip(names, row_paths):
            source = Path(raw_path).expanduser()
            source = (source if source.is_absolute() else root / source).resolve(strict=False)
            try:
                relative = PurePosixPath(*source.relative_to(root).parts)
            except ValueError:
                outside += 1
                continue
            if source.is_file():
                present += 1
            mapped = mapping.get(name) if isinstance(mapping, dict) else None
            if not isinstance(mapped, str) or not mapped.strip():
                continue
            target = PurePosixPath(mapped.strip().replace("\\", "/"))
            windows_target = PureWindowsPath(mapped.strip())
            if (target.is_absolute() or windows_target.is_absolute() or windows_target.drive
                    or ".." in target.parts or not target.name):
                invalid_target += 1
                continue
            if target.suffix:
                if target.suffix.lower() != ".mp4":
                    invalid_target += 1
                    continue
            else:
                target = target.with_suffix(".mp4")
            if relative == target:
                matched += 1
        if outside:
            failures.add("local_video_path_outside_root")
        if invalid_target:
            failures.add("local_mapping_path_invalid")
        if matched != len(row_paths):
            failures.add("local_video_path_mapping_mismatch")
        if present != len(row_paths):
            failures.add("local_video_files_missing")
        local_path_audit = {
            "status": "matched" if matched == len(rows) and present == len(rows)
                      and outside == 0 and invalid_target == 0 else "incomplete_or_invalid",
            "requested_rows": len(rows),
            "valid_rows_checked": len(row_paths),
            "unresolved_rows": len(rows) - len(row_paths),
            "mapping_relative_path_matches": matched,
            "files_present": present,
            "paths_outside_root": outside,
            "invalid_mapping_paths": invalid_target,
            "root_path_or_file_names_emitted": False,
        }
    return {
        "schema_version": 1,
        "status": "mapped" if not failures and not missing else "incomplete_or_invalid",
        "requested_rows": len(rows),
        "valid_video_stems": len(names),
        "mapped_rows": len(targets),
        "unmapped_rows": missing,
        "unique_mapped_video_paths": len(set(targets)),
        "local_path_mapping_audit": local_path_audit,
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
    parser.add_argument("--video-root", type=Path,
                        help="optional local NExTVideo root for path-to-map verification")
    args = parser.parse_args(argv)
    try:
        rows, rows_raw = _read(args.sample_manifest)
        mapping, mapping_raw = _read(args.video_map)
        result = audit(rows, rows_raw, mapping, mapping_raw, args.video_root)
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
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        parser.error("output already exists; choose a new path")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "mapped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
