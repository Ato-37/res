"""Reconcile a head-confirmation result with a privacy-safe identity audit.

This checks only aggregate provenance: ordinal coverage, exact sample-manifest
hash agreement, and the identity/content counts in a separately recorded audit.
It does not prove the old run manifest cryptographically binds the result, does
not inspect or reveal sample IDs, and does not establish independence. A passing
status is specifically `cohort_identity_verified`, not a full run-provenance pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

EXPERIMENT = "gradient_activation_head_faithfulness"
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reconcile(result: Any, manifest: Any, identity: Any, *, result_sha256: str,
              manifest_sha256: str, identity_sha256: str,
              audited_runner_sha256: str, source_runner_sha256: str,
              source_commit_available: bool) -> dict:
    failures = set()
    if not isinstance(result, dict) or result.get("experiment") != EXPERIMENT:
        failures.add("result_schema_invalid")
    if not isinstance(manifest, dict):
        failures.add("run_manifest_schema_invalid")
    if not isinstance(identity, dict):
        failures.add("identity_audit_schema_invalid")

    requested = result.get("n_requested") if isinstance(result, dict) else None
    effective = result.get("n_effective") if isinstance(result, dict) else None
    failed = result.get("n_failures") if isinstance(result, dict) else None
    records = result.get("records") if isinstance(result, dict) else None
    if (any(type(x) is not int or x < 0 for x in (requested, effective, failed))
            or not isinstance(records, list) or effective != len(records)
            or requested != effective + failed):
        failures.add("result_accounting_invalid")
        requested = effective = failed = None

    ordinals_complete = False
    if isinstance(result, dict) and isinstance(manifest, dict):
        for key in ("experiment", "n_requested", "n_effective", "n_failures"):
            if (result.get(key) != manifest.get(key)
                    or (key != "experiment" and type(manifest.get(key)) is not int)):
                failures.add("run_manifest_result_mismatch")
                break
    if type(requested) is int and isinstance(records, list):
        ordinals = [record.get("sample_key") for record in records
                    if isinstance(record, dict)]
        expected = [f"sample_{index:03d}" for index in range(requested)]
        ordinals_complete = (len(ordinals) == len(records)
                             and all(isinstance(key, str) for key in ordinals)
                             and sorted(ordinals) == expected)
        if not ordinals_complete:
            failures.add("result_sample_ordinals_incomplete_or_duplicate")

    sample_hash = manifest.get("samples_sha256") if isinstance(manifest, dict) else None
    if not isinstance(sample_hash, str) or SHA.fullmatch(sample_hash) is None:
        failures.add("sample_manifest_hash_invalid")
        sample_hash = None
    runner_source_match = False
    manifest_hash_matches = False
    if isinstance(identity, dict):
        manifest_hash_matches = (sample_hash is not None
                                 and identity.get("input_manifest_sha256") == sample_hash)
        if not manifest_hash_matches:
            failures.add("identity_audit_manifest_hash_mismatch")
        expected_counts = {
            "status": "complete",
            "requested_sample_records": requested,
            "effective_sample_records_with_hashed_video": effective,
            "unique_nonempty_normalized_video_ids": effective,
            "effective_unique_nonempty_normalized_video_ids": effective,
            "duplicate_video_id_excess_records": 0,
            "effective_duplicate_video_id_excess_records": 0,
            "unique_resolved_video_paths": effective,
            "files_hashed_successfully": effective,
            "missing_video_path_records": 0,
            "missing_files": 0,
            "read_failed_files": 0,
            "duplicate_resolved_path_excess_records": 0,
            "duplicate_content_excess_files": 0,
        }
        if any(identity.get(key) != value
               or (type(value) is int and type(identity.get(key)) is not int)
               for key, value in expected_counts.items()):
            failures.add("identity_audit_counts_incomplete_or_nonunique")

        source_commit = identity.get("source_commit")
        runner_hash = identity.get("runner_sha256")
        if not isinstance(source_commit, str) or COMMIT.fullmatch(source_commit) is None:
            failures.add("identity_audit_source_commit_invalid")
        runner_source_match = (
            isinstance(runner_hash, str) and SHA.fullmatch(runner_hash) is not None
            and runner_hash == audited_runner_sha256
            and runner_hash == source_runner_sha256 and source_commit_available
        )
        if not runner_source_match:
            failures.add("identity_audit_runner_provenance_unverified")
    else:
        source_commit = None
        runner_hash = None

    for digest in (result_sha256, manifest_sha256, identity_sha256):
        if not isinstance(digest, str) or SHA.fullmatch(digest) is None:
            failures.add("artifact_hash_invalid")

    passed = not failures
    result_manifest_binding = (
        isinstance(manifest, dict)
        and manifest.get("result_sha256") == result_sha256
        and isinstance(manifest.get("result_sha256"), str)
        and SHA.fullmatch(manifest["result_sha256"]) is not None
    )
    return {
        "schema_version": 1,
        "status": "cohort_identity_verified" if passed else "failed",
        "failure_codes": sorted(failures),
        "result_sha256": result_sha256,
        "run_manifest_sha256": manifest_sha256,
        "identity_audit_sha256": identity_sha256,
        "sample_manifest_sha256": sample_hash,
        "identity_audit_source_commit": source_commit,
        "identity_audit_runner_sha256": runner_hash,
        "identity_audit_runner_source_match": runner_source_match,
        "result_run_manifest_crosshash_binding": result_manifest_binding,
        "n_requested": requested,
        "n_effective": effective,
        "n_failures": failed,
        "result_sample_ordinals_complete": ordinals_complete,
        "manifest_hash_matches_identity_audit": manifest_hash_matches,
        "n_unique_video_ids": identity.get("unique_nonempty_normalized_video_ids") if isinstance(identity, dict) else None,
        "duplicate_video_id_excess_records": identity.get("duplicate_video_id_excess_records") if isinstance(identity, dict) else None,
        "n_files_hashed": identity.get("files_hashed_successfully") if isinstance(identity, dict) else None,
        "duplicate_content_excess_files": identity.get("duplicate_content_excess_files") if isinstance(identity, dict) else None,
        "independence_established": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--identity-audit", required=True)
    parser.add_argument("--output", required=True, help="new JSON path directly under results/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(args.result), Path(args.run_manifest), Path(args.identity_audit)]
    raw = [path.read_bytes() for path in paths]
    values = [json.loads(blob.decode("utf-8")) for blob in raw]
    runner_path = repo / "scripts/audit_sample_video_identity.py"
    runner_raw = runner_path.read_bytes()
    source_commit = values[2].get("source_commit") if isinstance(values[2], dict) else None
    source = subprocess.run(
        ["git", "show", f"{source_commit}:scripts/audit_sample_video_identity.py"],
        cwd=repo, capture_output=True, check=False,
    ) if isinstance(source_commit, str) and COMMIT.fullmatch(source_commit) else None
    report = reconcile(
        *values, result_sha256=_sha(raw[0]), manifest_sha256=_sha(raw[1]),
        identity_sha256=_sha(raw[2]),
        audited_runner_sha256=values[2].get("runner_sha256", "") if isinstance(values[2], dict) else "",
        source_runner_sha256=_sha(source.stdout) if source is not None and source.returncode == 0 else "",
        source_commit_available=source is not None and source.returncode == 0,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = repo / output
    if output.parent.resolve() != (repo / "results").resolve() or output.suffix.lower() != ".json":
        parser.error("output must be a JSON file directly under repository results/")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "failure_codes": report["failure_codes"],
                      "n_effective": report["n_effective"],
                      "n_unique_video_ids": report["n_unique_video_ids"]}, sort_keys=True))
    return 0 if report["status"] == "cohort_identity_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
