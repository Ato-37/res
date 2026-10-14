"""Create a private, incomplete v2 item-rights review ledger template.

The output contains eligible video IDs and intentionally unresolved declarations.
It is created mode 0600, refuses to overwrite, and never grants inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

ALLOWED_USES = {"local_inference", "local_retention", "aggregate_publication", "redistribution"}


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_manifest(path: Path, video_field: str = "video_id") -> tuple[list[str], bytes]:
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("eligible manifest must be a non-empty JSON array")
    ids = []
    for row in manifest:
        if not isinstance(row, dict) or not isinstance(row.get(video_field), str):
            raise ValueError("cohort video field is invalid")
        value = row[video_field]
        if not value.strip() or value != value.strip():
            raise ValueError("cohort video field is empty or has surrounding whitespace")
        if video_field == "video":
            normalized = value.replace("\\", "/")
            path_value = PurePosixPath(normalized)
            if normalized.endswith("/") or not path_value.suffix or not path_value.stem:
                raise ValueError("video field must be a file-like path with an extension")
            value = path_value.stem
            if value != value.strip():
                raise ValueError("video filename stem has surrounding whitespace")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError("eligible manifest video_id values are not unique")
    return ids, raw


def build_template(ids: list[str], eligible_raw: bytes, dataset_id: str,
                   intended_use_scope: str, intended_uses: list[str]) -> dict:
    if not dataset_id.strip() or not intended_use_scope.strip():
        raise ValueError("dataset ID and intended-use scope must be non-empty")
    if (not intended_uses or "local_inference" not in intended_uses
            or len(intended_uses) != len(set(intended_uses))
            or any(use not in ALLOWED_USES for use in intended_uses)):
        raise ValueError("intended uses must be unique allowed values and include local_inference")
    unresolved = {use: "unresolved" for use in intended_uses}
    items = []
    for video_id in ids:
        items.append({
            "video_id": video_id,
            "source_dataset_id": "",
            "source_video_id": "",
            "source_lineage": [],
            "media_license_id": "",
            "media_evidence_locator": "",
            "media_evidence_sha256": "",
            "media_review_record_sha256": "",
            "media_reviewer_role": "",
            "media_review_status": "not_reviewed",
            "media_reviewed_at_utc": "",
            "conditions_record_sha256": "",
            "conditions_status": "unreviewed",
            "decisions": unresolved.copy(),
        })
    return {
        "schema_version": 2,
        "eligible_dataset_id": dataset_id.strip(),
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "intended_use_scope": intended_use_scope.strip(),
        "intended_uses": intended_uses,
        "annotation_rights": {
            "dataset_id": "", "license_id": "", "evidence_locator": "",
            "evidence_sha256": "", "review_record_sha256": "", "reviewer_role": "",
            "review_status": "not_reviewed", "reviewed_at_utc": "", "decisions": unresolved.copy(),
        },
        "items": items,
    }


def write_private_bundle(path: Path, eligible: list[dict], ledger: dict) -> tuple[bytes, bytes]:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    eligible_bytes = (json.dumps(eligible, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Bind the ledger to the exact bytes written next to it, even for callers
    # that constructed the ledger from a different JSON serialization.
    ledger = {**ledger, "eligible_manifest_sha256": hashlib.sha256(eligible_bytes).hexdigest()}
    ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        for name, content in (("eligible_manifest.json", eligible_bytes),
                              ("rights_ledger_template.json", ledger_bytes)):
            descriptor = os.open(path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
    except Exception:
        try:
            for item in path.iterdir():
                item.unlink()
            path.rmdir()
        except OSError:
            pass
        raise
    return eligible_bytes, ledger_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--video-field", choices=("video_id", "video"), default="video_id")
    parser.add_argument("--eligible-dataset-id", required=True)
    parser.add_argument("--intended-use-scope", required=True)
    parser.add_argument("--intended-uses", required=True, nargs="+", choices=sorted(ALLOWED_USES))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        ids, _ = _read_manifest(args.cohort_manifest, args.video_field)
        eligible = [{"video_id": video_id} for video_id in ids]
        eligible_raw = (json.dumps(eligible, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ledger = build_template(ids, eligible_raw, args.eligible_dataset_id,
                                args.intended_use_scope, args.intended_uses)
        eligible_bytes, ledger_bytes = write_private_bundle(args.output_dir, eligible, ledger)
    except FileExistsError:
        parser.error("output already exists; choose a new private path")
    except Exception as exc:
        parser.error(f"template not created: {type(exc).__name__}")
    print(json.dumps({
        "status": "incomplete_review_template_created",
        "eligible_video_count": len(ids),
        "eligible_manifest_sha256": hashlib.sha256(eligible_bytes).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "execution_authorization": "not_granted",
        "reviewer_attestation_created": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
