#!/usr/bin/env python3
"""Range-download only frozen cohort MP4/time-map files with MD5 checks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"
EXPECTED_MANIFEST_SHA256 = "4b2bc0f8bae6c27d3521d7f319b3a9df16652da1f0492be3ff01261fd8197c06"
CHUNK_BYTES = 8 * 1024 * 1024
WORKERS = 24


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("official_manifest_hash_mismatch")
    result = {}
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, relative = parts[0], " ".join(parts[1:]).strip()
        if len(digest) != 32 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("official_manifest_md5_invalid")
        result[relative] = digest
    return result


def _file_size(url: str) -> int:
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=60) as response:
        size = response.headers.get("Content-Length")
        if response.status != 200 or size is None or not size.isdigit():
            raise ValueError("remote_file_size_missing")
        return int(size)


def build_ranges(size: int, chunk_bytes: int = CHUNK_BYTES) -> list[tuple[int, int]]:
    if type(size) is not int or size < 1 or type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("range_geometry_invalid")
    return [(start, min(start + chunk_bytes, size) - 1)
            for start in range(0, size, chunk_bytes)]


def _fetch_range(url: str, start: int, end: int, expected_size: int,
                 total_size: int) -> bytes:
    for attempt in range(3):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=90) as response:
                raw = response.read(expected_size + 1)
                if (response.status != 206
                        or response.headers.get("Content-Range")
                           != f"bytes {start}-{end}/{total_size}"
                        or len(raw) != expected_size):
                    raise ValueError("http_range_contract_failed")
                return raw
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def download_file(relative: str, expected_md5: str, root: Path) -> tuple[int, bool]:
    url = DATA_URL + quote(relative, safe="/")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    if destination.exists():
        if md5_file(destination) == expected_md5:
            return destination.stat().st_size, True
        raise ValueError("refuse_existing_checksum_mismatch")
    size = _file_size(url)
    ranges = build_ranges(size)
    if not ranges:
        raise ValueError("remote_file_empty")
    with tempfile.TemporaryDirectory(prefix="hd-epic-ranges-",
                                     dir=str(destination.parent)) as tmp:
        temp_dir = Path(tmp)
        def write_range(pair):
            index, (start, end) = pair
            data = _fetch_range(url, start, end, end - start + 1, size)
            chunk = temp_dir / f"{index:06d}.part"
            chunk.write_bytes(data)
            os.chmod(chunk, 0o600)
            return chunk
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            chunks = list(pool.map(write_range, enumerate(ranges)))
        fd, partial_name = tempfile.mkstemp(prefix=destination.name + ".assembled.",
            suffix=".partial", dir=str(destination.parent))
        partial = Path(partial_name)
        with os.fdopen(fd, "wb") as output:
            for chunk in chunks:
                with chunk.open("rb") as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        output.write(block)
        os.chmod(partial, 0o600)
        if partial.stat().st_size != size or md5_file(partial) != expected_md5:
            partial.unlink()
            raise ValueError("assembled_file_checksum_mismatch")
        os.replace(partial, destination)
    return size, False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
        default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--output-root", type=Path,
        default=ROOT / "data/private_minerva/range_download/HD-EPIC")
    args = parser.parse_args(argv)
    private = ROOT / "data/private_minerva"
    if (args.cohort.parent.resolve() != private.resolve()
            or not private.is_dir() or private.stat().st_mode & 0o077
            or not args.output_root.resolve().is_relative_to(private.resolve())):
        raise ValueError("private_cohort_contract_failed")
    if not args.cohort.is_file():
        raise ValueError("cohort_missing")
    cohort = json.loads(args.cohort.read_text())
    video_ids = [entry.get("video_id") for entry in cohort.get("selection", [])]
    if not video_ids or any(not isinstance(v, str) for v in video_ids):
        raise ValueError("cohort_selection_invalid")
    manifest = parse_manifest(args.manifest)
    paths = sorted(path for path in manifest
                   if "/Videos/" in path and any(v in path for v in video_ids)
                   and (path.endswith(".mp4")
                        or path.endswith("_mp4_to_vrs_time_ns.csv")))
    expected = 2 * len(video_ids)
    if len(paths) != expected:
        raise ValueError("cohort_media_manifest_count_mismatch")
    args.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_root, 0o700)
    started = time.monotonic()
    sizes, reused = 0, 0
    for relative in paths:
        size, was_reused = download_file(relative, manifest[relative], args.output_root)
        sizes += size
        reused += was_reused
    print(json.dumps({"status": "selected_media_downloaded_and_md5_verified",
        "selected_video_count": len(video_ids), "verified_file_count": len(paths),
        "verified_bytes": sizes, "reused_file_count": reused,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "rights_status": "unverified; files remain local-only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
