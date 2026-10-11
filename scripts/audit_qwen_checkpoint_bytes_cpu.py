#!/usr/bin/env python3
"""Verify cached Qwen2.5-VL shard bytes against pinned Hub metadata (CPU only).

This audit streams the five local safetensors files and compares their SHA-256
and Git blob SHA-1 digests with the official, fixed-revision metadata artifact.
It never imports model libraries, opens tensor contents, loads weights, or uses
the GPU. The report contains no local filesystem paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
API_METADATA_RELATIVE = Path(
    "datasets/checkpoint_source_audit/private/"
    "qwen_cc594898_model_metadata_20261012_01.json"
)
API_METADATA_SHA256 = "4b62c781f1f1e70ff315d119853e65f59f7103db52fbf11e0bd8eed3242ea06d"
SHARD_NAMES = tuple(f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6))
INDEX_NAME = "model.safetensors.index.json"
EXPECTED_INDEX_ENTRIES = 729
EXPECTED_INDEX_TENSOR_BYTES = 16_584_333_312
CHUNK_BYTES = 8 * 1024 * 1024
SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _lfs_pointer_bytes(sha256: str, size: int) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha256}\n"
        f"size {size}\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _safe_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _stable_stat(path: Path) -> os.stat_result:
    result = path.stat()
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("not_regular_file")
    return result


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns, left.st_ctime_ns
    ) == (
        right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns, right.st_ctime_ns
    )


def _cache_blob_roots(snapshot_dir: Path) -> tuple[Path, Path]:
    try:
        # Hugging Face uses both per-repo blobs and the shared Hub Xet blob store.
        return snapshot_dir.parents[1] / "blobs", snapshot_dir.parents[2] / "blobs"
    except IndexError as exc:
        raise ValueError("snapshot_path_shape_invalid") from exc


def _resolved_cache_blob(snapshot_dir: Path, path: Path, expected_link_name: str) -> Path:
    if not path.is_symlink() or Path(os.readlink(path)).name != expected_link_name:
        raise ValueError("cache_link_mismatch")
    resolved = path.resolve(strict=True)
    allowed_roots = tuple(root.resolve(strict=True) for root in _cache_blob_roots(snapshot_dir))
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError("cache_target_outside_blob_store")
    if not resolved.is_file():
        raise ValueError("cache_target_not_regular_file")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _metadata_shards(metadata: dict) -> tuple[dict[str, dict], dict]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("api_siblings_invalid")
    by_name: dict[str, dict] = {}
    for sibling in siblings:
        if not isinstance(sibling, dict):
            raise ValueError("api_sibling_invalid")
        name = sibling.get("rfilename")
        if name in SHARD_NAMES or name == INDEX_NAME:
            if not isinstance(name, str) or name in by_name:
                raise ValueError("api_duplicate_or_invalid_expected_file")
            by_name[name] = sibling
    if set(by_name) != set(SHARD_NAMES) | {INDEX_NAME}:
        raise ValueError("api_expected_file_set_mismatch")
    for name in (*SHARD_NAMES, INDEX_NAME):
        item = by_name[name]
        if not _safe_int(item.get("size")) or item["size"] < 1:
            raise ValueError("api_file_size_invalid")
        blob_id = item.get("blobId")
        if not isinstance(blob_id, str) or len(blob_id) != 40 or any(
            char not in "0123456789abcdef" for char in blob_id
        ):
            raise ValueError("api_git_blob_id_invalid")
        if name in SHARD_NAMES:
            lfs = item.get("lfs")
            if not isinstance(lfs, dict):
                raise ValueError("api_lfs_metadata_missing")
            lfs_sha = lfs.get("sha256")
            if not isinstance(lfs_sha, str) or len(lfs_sha) != 64 or any(
                char not in "0123456789abcdef" for char in lfs_sha
            ):
                raise ValueError("api_lfs_sha256_invalid")
            if not _safe_int(lfs.get("size")) or lfs["size"] < 1 or lfs.get("size") != item["size"]:
                raise ValueError("api_lfs_size_mismatch")
            if not _safe_int(lfs.get("pointerSize")):
                raise ValueError("api_lfs_pointer_size_invalid")
            pointer = _lfs_pointer_bytes(lfs_sha, lfs["size"])
            if lfs.get("pointerSize") != len(pointer):
                raise ValueError("api_lfs_pointer_size_mismatch")
            if _git_blob_sha1(pointer) != blob_id:
                raise ValueError("api_lfs_pointer_git_blob_sha1_mismatch")
    return {name: by_name[name] for name in SHARD_NAMES}, by_name[INDEX_NAME]


def _check_index(snapshot_dir: Path, api_index: dict, expected_entries: int,
                 expected_tensor_bytes: int) -> dict:
    index_path = snapshot_dir / INDEX_NAME
    resolved_path = _resolved_cache_blob(snapshot_dir, index_path, api_index["blobId"])
    before = _stable_stat(resolved_path)
    if before.st_size != api_index["size"]:
        raise ValueError("index_size_mismatch")
    with resolved_path.open("rb") as stream:
        fd_before = os.fstat(stream.fileno())
        data = stream.read()
        fd_after = os.fstat(stream.fileno())
    after = _stable_stat(resolved_path)
    if not _same_stat(before, fd_before) or not _same_stat(fd_before, fd_after) or not _same_stat(fd_after, after):
        raise ValueError("index_changed_during_read")
    if index_path.resolve(strict=True) != resolved_path:
        raise ValueError("index_cache_link_changed")
    observed_blob_id = _git_blob_sha1(data)
    if observed_blob_id != api_index["blobId"]:
        raise ValueError("index_git_blob_sha1_mismatch")
    index = json.loads(data)
    if not isinstance(index, dict):
        raise ValueError("index_root_invalid")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not _safe_int(metadata.get("total_size")):
        raise ValueError("index_metadata_invalid")
    if not isinstance(weight_map, dict) or len(weight_map) != expected_entries:
        raise ValueError("index_weight_map_count_mismatch")
    if metadata["total_size"] != expected_tensor_bytes:
        raise ValueError("index_tensor_bytes_mismatch")
    values = list(weight_map.values())
    if any(not isinstance(value, str) or value not in SHARD_NAMES for value in values):
        raise ValueError("index_shard_name_invalid")
    if set(values) != set(SHARD_NAMES):
        raise ValueError("index_shard_coverage_mismatch")
    return {
        "observed_size_bytes": len(data),
        "expected_size_bytes": api_index["size"],
        "expected_git_blob_sha1": api_index["blobId"],
        "observed_git_blob_sha1": observed_blob_id,
        "git_blob_sha1_matches": True,
        "weight_map_entries": len(weight_map),
        "tensor_payload_bytes": metadata["total_size"],
    }


def _stream_verify_shard(snapshot_dir: Path, path: Path, expected: dict) -> dict:
    resolved_path = _resolved_cache_blob(snapshot_dir, path, expected["lfs"]["sha256"])
    before = _stable_stat(resolved_path)
    if before.st_size != expected["size"]:
        return {
            "observed_size_bytes": before.st_size,
            "expected_size_bytes": expected["size"],
            "expected_sha256": expected["lfs"]["sha256"],
            "observed_sha256": None,
            "expected_lfs_pointer_git_blob_sha1": expected["blobId"],
            "reconstructed_lfs_pointer_git_blob_sha1": _git_blob_sha1(
                _lfs_pointer_bytes(expected["lfs"]["sha256"], expected["size"])
            ),
            "lfs_pointer_git_blob_sha1_matches": True,
            "sha256_matches": False,
            "stable_during_read": None,
            "failure_code": "shard_size_mismatch",
            "bytes_read": 0,
        }
    sha256 = hashlib.sha256()
    bytes_read = 0
    try:
        with resolved_path.open("rb") as stream:
            fd_before = os.fstat(stream.fileno())
            if not _same_stat(before, fd_before):
                return {
                    "observed_size_bytes": before.st_size,
                    "expected_size_bytes": expected["size"],
                    "expected_sha256": expected["lfs"]["sha256"],
                    "observed_sha256": None,
                    "expected_lfs_pointer_git_blob_sha1": expected["blobId"],
                    "reconstructed_lfs_pointer_git_blob_sha1": _git_blob_sha1(
                        _lfs_pointer_bytes(expected["lfs"]["sha256"], expected["size"])
                    ),
                    "lfs_pointer_git_blob_sha1_matches": True,
                    "sha256_matches": False,
                    "stable_during_read": False,
                    "failure_code": "shard_changed_before_read",
                    "bytes_read": 0,
                }
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                sha256.update(block)
                bytes_read += len(block)
            fd_after = os.fstat(stream.fileno())
    except OSError:
        return {
            "observed_size_bytes": before.st_size,
            "expected_size_bytes": expected["size"],
            "expected_sha256": expected["lfs"]["sha256"],
            "observed_sha256": None,
            "expected_lfs_pointer_git_blob_sha1": expected["blobId"],
            "reconstructed_lfs_pointer_git_blob_sha1": _git_blob_sha1(
                _lfs_pointer_bytes(expected["lfs"]["sha256"], expected["size"])
            ),
            "lfs_pointer_git_blob_sha1_matches": True,
            "sha256_matches": False,
            "stable_during_read": None,
            "failure_code": "shard_read_error",
            "bytes_read": bytes_read,
        }
    try:
        after = _stable_stat(resolved_path)
        same_target = path.resolve(strict=True) == resolved_path
    except (OSError, ValueError):
        after = None
        same_target = False
    stable = (
        after is not None and _same_stat(before, fd_before) and _same_stat(fd_before, fd_after)
        and _same_stat(fd_after, after) and same_target
    )
    sha_matches = sha256.hexdigest() == expected["lfs"]["sha256"]
    failure = None
    if bytes_read != expected["size"]:
        failure = "shard_bytes_read_mismatch"
    elif not stable:
        failure = "shard_changed_during_read"
    elif not sha_matches:
        failure = "shard_sha256_mismatch"
    return {
        "observed_size_bytes": before.st_size,
        "expected_size_bytes": expected["size"],
        "expected_sha256": expected["lfs"]["sha256"],
        "observed_sha256": sha256.hexdigest(),
        "expected_lfs_pointer_git_blob_sha1": expected["blobId"],
        "reconstructed_lfs_pointer_git_blob_sha1": _git_blob_sha1(
            _lfs_pointer_bytes(expected["lfs"]["sha256"], expected["size"])
        ),
        "lfs_pointer_git_blob_sha1_matches": True,
        "sha256_matches": sha_matches,
        "stable_during_read": stable,
        "failure_code": failure,
        "bytes_read": bytes_read,
    }


def audit_checkpoint(
    *, metadata_bytes: bytes, snapshot_dir: Path, expected_metadata_sha256: str,
    expected_index_entries: int = EXPECTED_INDEX_ENTRIES,
    expected_index_tensor_bytes: int = EXPECTED_INDEX_TENSOR_BYTES,
    repo_root: Path | None = None,
) -> dict:
    """Audit a snapshot. Hash override exists for small synthetic unit fixtures only."""
    started = time.monotonic()
    created = datetime.now(timezone.utc).isoformat()
    metadata_sha = _sha256_bytes(metadata_bytes)
    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "qwen_checkpoint_byte_authentication_audit",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "created_at_utc": created,
        "source_commit": _source_commit(repo_root) if repo_root else None,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "device": "cpu",
        "api_metadata_sha256": metadata_sha,
        "api_metadata_hash_matches_pinned_artifact": metadata_sha == expected_metadata_sha256,
        "requested_shards": len(SHARD_NAMES),
        "effective_shards": 0,
        "expected_shard_bytes": None,
        "bytes_read": 0,
        "index_weight_map_entries": None,
        "index_tensor_payload_bytes": None,
        "index": None,
        "shards": [],
        "failure_codes": [],
        "failure_count": 0,
        "byte_authentication_status": "not_verified",
        "historical_gpu_runs_byte_verified": False,
        "elapsed_seconds": None,
        "limitations": [
            "Authenticates only the local checkpoint bytes read during this run; it does not retroactively authenticate prior model executions.",
            "The index total_size describes tensor payload size; safetensors shard files also contain format headers, so their byte sum may be larger.",
            "API lfs.sha256 is compared to streamed shard SHA-256; for LFS shards, sibling blobId is checked against the canonical 135-byte Git LFS pointer object, not against the downloaded shard bytes. The index sibling blobId is checked against the index file. Xet identifiers are not treated as byte digests.",
            "No model code, model libraries, tensor metadata, inference, or GPU was used.",
        ],
    }
    if metadata_sha != expected_metadata_sha256:
        report["failure_codes"].append("api_metadata_sha256_mismatch")
        report["failure_count"] = len(report["failure_codes"])
        report["elapsed_seconds"] = time.monotonic() - started
        return report
    try:
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError("api_metadata_root_invalid")
        if metadata.get("id") != MODEL_ID or metadata.get("modelId") != MODEL_ID:
            raise ValueError("api_model_id_mismatch")
        if metadata.get("sha") != REVISION:
            raise ValueError("api_revision_mismatch")
        api_shards, api_index = _metadata_shards(metadata)
        report["expected_shard_bytes"] = sum(item["size"] for item in api_shards.values())
        report["index"] = {
            "expected_size_bytes": api_index["size"],
            "expected_git_blob_sha1": api_index["blobId"],
            "observed_size_bytes": None,
            "observed_git_blob_sha1": None,
            "git_blob_sha1_matches": False,
            "weight_map_entries": None,
            "tensor_payload_bytes": None,
        }
        index_result = _check_index(
            snapshot_dir, api_index, expected_index_entries, expected_index_tensor_bytes
        )
        index_result["failure_code"] = None
        report["index"] = index_result
        report["index_weight_map_entries"] = index_result["weight_map_entries"]
        report["index_tensor_payload_bytes"] = index_result["tensor_payload_bytes"]
    except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
        code = str(exc) if str(exc).isidentifier() else type(exc).__name__
        report["failure_codes"].append(code)
        if isinstance(report.get("index"), dict):
            report["index"]["failure_code"] = code
        report["elapsed_seconds"] = time.monotonic() - started
        report["failure_count"] = len(report["failure_codes"])
        return report

    for name in SHARD_NAMES:
        try:
            result = _stream_verify_shard(snapshot_dir, snapshot_dir / name, api_shards[name])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            code = str(exc) if str(exc).isidentifier() else type(exc).__name__
            result = {
                "observed_size_bytes": None,
                "expected_size_bytes": api_shards[name]["size"],
                "expected_sha256": api_shards[name]["lfs"]["sha256"],
                "observed_sha256": None,
                "expected_lfs_pointer_git_blob_sha1": api_shards[name]["blobId"],
                "reconstructed_lfs_pointer_git_blob_sha1": _git_blob_sha1(
                    _lfs_pointer_bytes(api_shards[name]["lfs"]["sha256"], api_shards[name]["size"])
                ),
                "lfs_pointer_git_blob_sha1_matches": True,
                "sha256_matches": False,
                "stable_during_read": None,
                "failure_code": code,
                "bytes_read": 0,
            }
        report["shards"].append({"filename": name, **result})
        report["effective_shards"] += int(result["bytes_read"] > 0)
        report["bytes_read"] += result["bytes_read"]
        if result["failure_code"]:
            report["failure_codes"].append(f"{name}:{result['failure_code']}")
    complete = (
        report["effective_shards"] == len(SHARD_NAMES)
        and all(
            item["sha256_matches"] and item["lfs_pointer_git_blob_sha1_matches"]
            and item["stable_during_read"] is True and item["failure_code"] is None
            for item in report["shards"]
        )
        and not report["failure_codes"]
    )
    if complete:
        report["byte_authentication_status"] = "verified_current_local_bytes"
    report["failure_count"] = len(report["failure_codes"])
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def write_exclusive(output: Path, repo_root: Path, report: dict) -> Path:
    output = validate_output_path(output, repo_root)
    results_dir = (repo_root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def validate_output_path(output: Path, repo_root: Path) -> Path:
    output = output if output.is_absolute() else repo_root / output
    output = output.resolve()
    results_dir = (repo_root / "results").resolve()
    if output.parent != results_dir or output.suffix.lower() != ".json":
        raise ValueError("output_must_be_json_directly_in_results")
    return output


def _default_snapshot_dir() -> Path:
    return (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / REVISION
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="unique JSON path directly under repository results/")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        output = validate_output_path(args.output, repo_root)
    except ValueError:
        print("checkpoint byte audit failed: invalid_output_path", file=sys.stderr)
        return 2
    # Refuse an existing artifact before touching the metadata or checkpoint.
    if output.exists():
        print("checkpoint byte audit failed: output_already_exists", file=sys.stderr)
        return 2
    try:
        metadata_bytes = (repo_root / API_METADATA_RELATIVE).read_bytes()
        report = audit_checkpoint(
            metadata_bytes=metadata_bytes,
            snapshot_dir=_default_snapshot_dir(),
            expected_metadata_sha256=API_METADATA_SHA256,
            repo_root=repo_root,
        )
        out = write_exclusive(args.output, repo_root, report)
    except (OSError, ValueError, TypeError) as exc:
        # Keep error detail private: no paths, exception messages, or input data.
        print(f"checkpoint byte audit failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    matched = report["byte_authentication_status"] == "verified_current_local_bytes"
    print(
        f"checkpoint bytes: {report['effective_shards']}/{report['requested_shards']} shards read; "
        f"{report['bytes_read']} bytes; status={report['byte_authentication_status']}; "
        f"failures={len(report['failure_codes'])}; artifact={out.name}"
    )
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
