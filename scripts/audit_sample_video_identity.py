"""Audit sample-manifest video identity and file-content duplication (CPU only).

This metadata audit verifies that the exact requested manifest bytes match a
caller-supplied SHA-256 before it reads any referenced video path. It reports
aggregate counts only: never IDs, paths, per-file hashes, or annotation text.
Distinct IDs or file hashes do not establish statistical independence; files
may share scenes, transcodes, or source campaigns.

Example:
    python scripts/audit_sample_video_identity.py \
        --samples-file /private/samples.json \
        --expected-sha256 2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata
import platform
import time


def _expected_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value.lower()))


def _normalized_video_id(value):
    if not isinstance(value, str):
        raise ValueError("sample manifest contains a non-string or empty video ID")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError("sample manifest contains a non-string or empty video ID")
    return normalized


def _resolve_video_path(value, base_dir):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(os.path.expanduser(value.strip()))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def audit_manifest_bytes(samples_raw, expected_sha256, n_requested=None, base_dir=None,
                        file_opener=None):
    """Return privacy-safe aggregate counts for a raw JSON sample manifest.

    The raw SHA-256 is checked before JSON parsing and before any referenced
    video path is resolved, statted, or opened. `base_dir` only affects relative
    video paths; it is not included in the returned report.
    """
    if not isinstance(samples_raw, bytes):
        raise ValueError("sample manifest must be supplied as raw bytes")
    if not _expected_hash(expected_sha256):
        raise ValueError("expected SHA-256 must be 64 hexadecimal characters")
    manifest_hash = hashlib.sha256(samples_raw).hexdigest()
    if manifest_hash != expected_sha256.lower():
        raise ValueError("sample manifest SHA-256 mismatch")

    try:
        samples = json.loads(samples_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("hashed sample manifest is not valid UTF-8 JSON") from None
    if not isinstance(samples, list):
        raise ValueError("sample manifest must be a JSON list")
    if n_requested is None:
        requested = len(samples)
    else:
        if isinstance(n_requested, bool) or not isinstance(n_requested, int) or n_requested < 1:
            raise ValueError("requested prefix count must be a positive integer")
        if n_requested > len(samples):
            raise ValueError("requested prefix exceeds sample manifest length")
        requested = n_requested
    prefix = samples[:requested]
    if not prefix:
        raise ValueError("sample manifest prefix must not be empty")
    if any(not isinstance(sample, dict) for sample in prefix):
        raise ValueError("requested sample manifest rows must be objects")

    video_ids = [_normalized_video_id(sample.get("video_id")) for sample in prefix]
    base = Path.cwd() if base_dir is None else Path(base_dir)
    resolved_paths = [_resolve_video_path(sample.get("video"), base) for sample in prefix]
    path_counts = Counter(path for path in resolved_paths if path is not None)
    id_counts = Counter(video_ids)

    opener = open if file_opener is None else file_opener
    content_counts = Counter()
    hashed_by_path = {}
    present_paths = set()
    missing_paths = set()
    failed_paths = set()
    bytes_hashed = 0
    for path in path_counts:
        if not path.is_file():
            missing_paths.add(path)
            continue
        present_paths.add(path)
        digest = hashlib.sha256()
        try:
            with opener(path, "rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    bytes_hashed += len(chunk)
        except OSError:
            failed_paths.add(path)
            continue
        file_hash = digest.hexdigest()
        hashed_by_path[path] = file_hash
        content_counts[file_hash] += 1

    successful_paths = set(hashed_by_path)
    effective_ids = [video_id for video_id, path in zip(video_ids, resolved_paths)
                     if path in successful_paths]
    effective_records = len(effective_ids)
    effective_id_counts = Counter(effective_ids)
    duplicate_id_excess = sum(count - 1 for count in id_counts.values() if count > 1)
    duplicate_path_excess = sum(count - 1 for count in path_counts.values() if count > 1)
    duplicate_content_excess = sum(
        count - 1 for count in content_counts.values() if count > 1)
    missing_path_records = sum(path is None for path in resolved_paths)

    return {
        "schema_version": 1,
        "purpose": "sample_video_identity_and_exact_content_audit",
        "status": "complete" if effective_records == requested else "partial",
        "content_comparison_complete": effective_records == requested,
        "content_duplicate_scope": "successfully hashed unique resolved paths only",
        "input_manifest_sha256": manifest_hash,
        "requested_sample_records": requested,
        "effective_sample_records_with_hashed_video": effective_records,
        "unique_nonempty_normalized_video_ids": len(id_counts),
        "duplicate_video_id_excess_records": duplicate_id_excess,
        "effective_unique_nonempty_normalized_video_ids": len(effective_id_counts),
        "effective_duplicate_video_id_excess_records": sum(
            count - 1 for count in effective_id_counts.values() if count > 1),
        "unique_resolved_video_paths": len(path_counts),
        "duplicate_resolved_path_excess_records": duplicate_path_excess,
        "files_present": len(present_paths),
        "files_hashed_successfully": len(successful_paths),
        "missing_video_path_records": missing_path_records,
        "missing_files": len(missing_paths),
        "read_failed_files": len(failed_paths),
        "duplicate_content_excess_files": duplicate_content_excess,
        "total_bytes_hashed": bytes_hashed,
        "identity_scope": (
            "distinct IDs or file contents do not establish statistical independence; "
            "overlapping scenes, transcodes, and shared campaigns remain possible"),
    }


def _source_metadata(script_path):
    repo_root = Path(script_path).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    runner_hash = hashlib.sha256(Path(script_path).read_bytes()).hexdigest()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "runner_sha256": runner_hash,
        "python_version": platform.python_version(),
    }


def _write_exclusive(output_path, report, repo_root=None):
    repo = Path(__file__).resolve().parents[1] if repo_root is None else Path(repo_root).resolve()
    output_path = (repo / output_path).resolve()
    if output_path.parent != (repo / 'results').resolve() or output_path.suffix.lower() != '.json':
        raise ValueError('output must be a JSON file directly under repository results/')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _concise(report):
    return (
        f"manifest records: requested={report['requested_sample_records']}; "
        f"effective_hashed={report['effective_sample_records_with_hashed_video']}\n"
        f"video IDs: requested_unique={report['unique_nonempty_normalized_video_ids']}; "
        f"effective_unique={report['effective_unique_nonempty_normalized_video_ids']}; "
        f"duplicate_excess={report['duplicate_video_id_excess_records']}\n"
        f"files: paths={report['unique_resolved_video_paths']}; "
        f"present={report['files_present']}; hashed={report['files_hashed_successfully']}; "
        f"missing={report['missing_files']}; read_failed={report['read_failed_files']}\n"
        f"duplicate paths={report['duplicate_resolved_path_excess_records']}; "
        f"duplicate contents={report['duplicate_content_excess_files']}; "
        f"bytes_hashed={report['total_bytes_hashed']}\n"
        "identity check does not establish statistical independence"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--n-requested", type=int,
                        help="optional prefix length; default is the full manifest")
    parser.add_argument("--out", help="unique JSON output path (existing files are refused)")
    args = parser.parse_args()
    if not _expected_hash(args.expected_sha256):
        parser.error("--expected-sha256 must be 64 hexadecimal characters")
    samples_path = Path(args.samples_file).expanduser()
    try:
        samples_raw = samples_path.read_bytes()
    except OSError:
        parser.error("could not read sample manifest")
    try:
        started = time.monotonic()
        report = audit_manifest_bytes(
            samples_raw, args.expected_sha256, args.n_requested, Path.cwd())
    except ValueError as error:
        parser.error(str(error))
    report.update(_source_metadata(__file__))
    report['elapsed_seconds'] = time.monotonic() - started
    if args.out:
        output_path = Path(args.out).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("results") / f"sample_video_identity_audit_{stamp}.json"
    try:
        _write_exclusive(output_path, report)
    except ValueError as error:
        parser.error(str(error))
    except FileExistsError:
        parser.error("output file already exists; choose a new --out path")
    print(_concise(report))


if __name__ == "__main__":
    main()
