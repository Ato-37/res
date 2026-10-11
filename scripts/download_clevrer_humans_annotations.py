#!/usr/bin/env python3
"""Download and safely extract the small public CLEVRER-Humans annotation package.

This deliberately downloads annotations only; it does not fetch CLEVRER videos.
The archive is accepted only when the expected files are present and every ZIP
member has a safe relative path. Existing output directories are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


DATASET_URL = (
    "https://datadryad.org/api/v2/datasets/"
    "doi%3A10.5061%2Fdryad.5tb2rbp7c/download"
)
EXPECTED_FILES = {
    "README.md",
    "causal_cloze.json",
    "train_ceg_data.p",
    "train_question.json",
    "valid_ceg_data.p",
    "valid_question.json",
}
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


def _validate_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    basenames = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            raise ValueError(f"Unsafe path in dataset archive: {member.filename!r}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symbolic link in dataset archive: {member.filename!r}")
        if not member.is_dir():
            basenames.add(path.name)

    missing = EXPECTED_FILES - basenames
    if missing:
        raise ValueError(
            "Archive is missing expected CLEVRER-Humans files: "
            + ", ".join(sorted(missing))
        )
    return members


def _extract_members(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], root: Path) -> None:
    for member in members:
        if member.is_dir():
            continue
        relative_path = PurePosixPath(member.filename)
        destination = root.joinpath(*relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)


def _extract_archive(archive_path: Path, output_dir: Path) -> tuple[int, str, list[str]]:
    total = archive_path.stat().st_size
    if total > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Archive exceeded safety limit ({MAX_ARCHIVE_BYTES} bytes); "
            "no extraction was performed"
        )
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)

    with zipfile.ZipFile(archive_path) as archive:
        members = _validate_members(archive)
        expanded_bytes = sum(member.file_size for member in members if not member.is_dir())
        if expanded_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"Expanded archive exceeded safety limit ({MAX_ARCHIVE_BYTES} bytes); "
                "no extraction was performed"
            )
        with tempfile.TemporaryDirectory(
            prefix="clevrer_humans_extract_", dir=output_dir.parent
        ) as temp_name:
            staging = Path(temp_name) / "extracted"
            staging.mkdir()
            _extract_members(archive, members, staging)
            extracted_files = sorted(
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
            )
            os.replace(staging, output_dir)

    return total, digest.hexdigest(), extracted_files


def extract_local_archive(archive_path: Path, output_dir: Path) -> tuple[int, str, list[str]]:
    """Safely extract a user-downloaded official Dryad ZIP."""
    output_dir = output_dir.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists; refusing to overwrite: {output_dir}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Annotation archive does not exist: {archive_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return _extract_archive(archive_path, output_dir)


def download_and_extract(output_dir: Path, url: str = DATASET_URL) -> tuple[int, str, list[str]]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": "res-research-data-preflight/1.0"})
    with tempfile.TemporaryDirectory(prefix="clevrer_humans_", dir=output_dir.parent) as temp_name:
        temp_root = Path(temp_name)
        archive_path = temp_root / "annotations.zip"
        total = 0
        try:
            with urlopen(request, timeout=60) as response, archive_path.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError(
                            f"Archive exceeded safety limit ({MAX_ARCHIVE_BYTES} bytes); "
                            "no extraction was performed"
                        )
                    output.write(chunk)
        except HTTPError as error:
            if error.code == 401:
                raise RuntimeError(
                    "Dryad's automated download endpoint returned HTTP 401. "
                    "Download the latest ZIP in a browser from "
                    "https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp7c, "
                    "then rerun with --archive-file PATH. Do not put a Dryad token "
                    "or login credentials in this repository."
                ) from error
            raise

        return _extract_archive(archive_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for annotation files; must not already exist",
    )
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--archive-file",
        type=Path,
        help=(
            "Safely extract a CLEVRER-Humans ZIP downloaded from the official Dryad page; "
            "use this when automated download is unavailable"
        ),
    )
    args = parser.parse_args()

    if args.archive_file:
        size, sha256, files = extract_local_archive(args.archive_file, args.output_dir)
        source = args.archive_file.expanduser().resolve()
    else:
        size, sha256, files = download_and_extract(args.output_dir, args.url)
        source = args.url
    print(f"archive_bytes={size}")
    print(f"archive_sha256={sha256}")
    print(f"source={source}")
    print(f"output_dir={args.output_dir.expanduser().resolve()}")
    print("extracted_files:")
    for filename in files:
        print(f"  {filename}")
    print("Note: pickle files are not loaded or executed by this script.")


if __name__ == "__main__":
    main()
