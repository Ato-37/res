#!/usr/bin/env python3
"""Build an owner-only union of media hashes from explicit claim/scoring JSONs.

Only caller-listed JSON files under ``data/private_nextgqa`` are considered.
Generation packets are identified from their purpose/schema content, while
scoring receipts require an explicit completed-forward or scoring marker.
The output contains only hashes, relative filenames, roles, and counts; stdout
contains counts and hashes only. This is a local exposure heuristic, not proof
that the historical inventory is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable

PRIVATE_REL = Path("data/private_nextgqa")
OUTPUT_PURPOSE = "local_claim_exposure_union_manifest"
MAX_SOURCE_BYTES = 128 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_KEYS = frozenset({"video_sha256", "source_video_sha256", "media_sha256"})
_SCORING_MARKERS = frozenset({
    "scoring_performed", "target_scoring_performed", "faithfulness_scored",
})


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("source_json_duplicate_key")
        result[key] = value
    return result


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=lambda _value: (_ for _ in ()).throw(
                              ValueError("source_json_nonfinite")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("source_json_invalid") from None


def _safe_private_root(repo_root: Path) -> Path:
    repo = repo_root.resolve(strict=True)
    private = repo / PRIVATE_REL
    try:
        info = private.lstat()
    except OSError:
        raise ValueError("private_root_unavailable") from None
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_root_permissions_invalid")
    return private


def _relative_private_path(value: str | Path) -> Path:
    path = Path(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts
            or path.parts[:2] != PRIVATE_REL.parts or path.suffix.lower() != ".json"):
        raise ValueError("source_path_invalid")
    return path


def _read_private_source(repo: Path, private_root: Path, relative: Path) -> bytes:
    """Read a private JSON via no-follow directory traversal and verify identity."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("safe_open_unavailable")
    components = relative.parts
    current = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=current)
            os.close(current)
            current = next_fd
        fd = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=current)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_size > MAX_SOURCE_BYTES):
                raise ValueError("source_permissions_or_size_invalid")
            # Verify the opened object is still the path entry and remains beneath
            # the owner-only private root. The descriptor walk itself rejects links.
            path = repo.joinpath(*components)
            path_info = path.lstat()
            if (stat.S_ISLNK(path_info.st_mode) or path_info.st_dev != info.st_dev
                    or path_info.st_ino != info.st_ino):
                raise ValueError("source_path_changed")
            if not path.resolve(strict=True).is_relative_to(private_root.resolve(strict=True)):
                raise ValueError("source_path_invalid")
            chunks = []
            total = 0
            while True:
                block = os.read(fd, min(1024 * 1024, MAX_SOURCE_BYTES + 1 - total))
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError("source_permissions_or_size_invalid")
            after = os.fstat(fd)
            if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or total != info.st_size):
                raise ValueError("source_changed_during_read")
            return b"".join(chunks)
        finally:
            os.close(fd)
    except OSError:
        raise ValueError("source_open_failed") from None
    finally:
        os.close(current)


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk(item)


def _is_claim_generation_packet(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    purpose = payload.get("purpose")
    if isinstance(purpose, str):
        normalized = re.sub(r"[^a-z0-9]+", "_", purpose.lower()).strip("_")
        # Accept names such as temporal_order_claim_generation and
        # claim_review_generation while avoiding generic candidate inventories.
        has_claim = "claim" in normalized.split("_")
        has_generation = any(token in normalized.split("_")
                             for token in ("generation", "generated", "generator"))
        if has_claim and has_generation:
            return True
    schema = payload.get("schema")
    schema_version = payload.get("schema_version")
    schema_text = " ".join(str(x).lower() for x in (schema, payload.get("artifact_type"))
                            if isinstance(x, str))
    schema_tokens = re.sub(r"[^a-z0-9]+", "_", schema_text).split("_")
    records = payload.get("records")
    if ("claim" in schema_tokens and any(x in schema_tokens
            for x in ("generation", "generated", "generator"))
            and (type(schema_version) is int or isinstance(records, list))):
        return True
    return False


def _has_scoring_evidence(payload: Any) -> bool:
    for key, value in _walk(payload):
        if key == "completed_forwards" and type(value) is int and value > 0:
            return True
        if key in _SCORING_MARKERS and value is True:
            return True
    return False


def _source_role(payload: Any) -> str | None:
    if _is_claim_generation_packet(payload):
        return "claim_generation"
    if _has_scoring_evidence(payload):
        return "scoring_receipt"
    return None


def _collect_media_hashes(payload: Any) -> list[str]:
    found: set[str] = set()
    for key, value in _walk(payload):
        if key not in _DIGEST_KEYS:
            continue
        values = value if isinstance(value, list) else [value]
        for candidate in values:
            if isinstance(candidate, str) and _SHA256.fullmatch(candidate):
                found.add(candidate)
    return sorted(found)


def build_manifest(sources: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Build deterministic manifest from `(private relative path, raw JSON)` pairs."""
    entries = []
    media_hashes: set[str] = set()
    seen_paths = set()
    for relative_name, raw in sources:
        relative = _relative_private_path(relative_name)
        rel = relative.as_posix()
        if rel in seen_paths:
            raise ValueError("source_path_duplicate")
        seen_paths.add(rel)
        payload = _parse_json(raw)
        role = _source_role(payload)
        if role is None:
            continue
        hashes = _collect_media_hashes(payload)
        media_hashes.update(hashes)
        entries.append({"relative_filename": rel, "source_sha256": _sha256(raw),
                        "role": role})
    entries.sort(key=lambda item: (item["relative_filename"], item["role"],
                                   item["source_sha256"]))
    inventory_digest = _sha256(_canonical(entries))
    union = sorted(media_hashes)
    return {
        "schema_version": 1,
        "purpose": OUTPUT_PURPOSE,
        "source_inventory_sha256": inventory_digest,
        "source_file_count": len(entries),
        "unique_media_count": len(union),
        "media_sha256": union,
        "sources": entries,
        "coverage": {
            "explicit_input_count": len(sources),
            "historical_completeness_authenticated": False,
            "exposure_is_heuristic": True,
        },
        "limitations": [
            "only explicitly listed, successfully parsed local JSON inputs are scanned",
            "generation/scoring classification uses schema, purpose, and explicit markers",
            "missing or redacted historical artifacts can cause unobserved exposure",
            "media hashes identify bytes, not semantic claim/question identity",
        ],
    }


def _write_exclusive(private_root: Path, output_name: str, raw: bytes) -> None:
    if (Path(output_name).name != output_name or not output_name.endswith(".json")):
        raise ValueError("output_name_invalid")
    parent_fd = os.open(private_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(output_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.unlink(output_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", action="append", required=True,
                        help="repository-relative JSON path under data/private_nextgqa; repeatable")
    parser.add_argument("--output-name", required=True,
                        help="new JSON filename within data/private_nextgqa")
    args = parser.parse_args(argv)
    try:
        repo = args.repo_root.resolve(strict=True)
        private_root = _safe_private_root(repo)
        inputs = []
        seen = set()
        for item in args.input:
            relative = _relative_private_path(item)
            if relative.as_posix() in seen:
                raise ValueError("source_path_duplicate")
            seen.add(relative.as_posix())
            raw = _read_private_source(repo, private_root, relative)
            inputs.append((relative.as_posix(), raw))
        manifest = build_manifest(inputs)
        encoded = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n").encode("utf-8")
        _write_exclusive(private_root, args.output_name, encoded)
        print(json.dumps({"status": "written", "source_file_count": manifest["source_file_count"],
                          "unique_media_count": manifest["unique_media_count"],
                          "source_inventory_sha256": manifest["source_inventory_sha256"],
                          "manifest_sha256": _sha256(encoded)}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(
            r"[a-z0-9_]+", str(exc)) else "manifest_build_failed"
        print(json.dumps({"status": "failed", "failure_code": code}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
