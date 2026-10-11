from __future__ import annotations

import tempfile
import unittest
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError
from pathlib import Path

from download_clevrer_humans_annotations import (
    EXPECTED_FILES,
    _validate_members,
    download_and_extract,
    extract_local_archive,
)


def make_annotation_zip(path: Path, *, extra_name: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(EXPECTED_FILES):
            archive.writestr(f"dataset/{name}", f"fixture:{name}")
        if extra_name:
            archive.writestr(extra_name, "unsafe fixture")


class AnnotationDownloadTests(unittest.TestCase):
    def test_extracts_expected_files_from_local_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "source.zip"
            make_annotation_zip(archive_path)
            output_dir = root / "out"

            size, digest, filenames = download_and_extract(
                output_dir,
                url=archive_path.as_uri(),
            )

            self.assertEqual(size, archive_path.stat().st_size)
            self.assertEqual(len(digest), 64)
            self.assertEqual(set(Path(name).name for name in filenames), EXPECTED_FILES)
            self.assertTrue((output_dir / "dataset" / "valid_ceg_data.p").is_file())

    def test_extracts_user_downloaded_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "official-download.zip"
            make_annotation_zip(archive_path)

            size, digest, filenames = extract_local_archive(archive_path, root / "out")

            self.assertEqual(size, archive_path.stat().st_size)
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(filenames), len(EXPECTED_FILES))

    def test_http_401_explains_manual_official_download_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            error = HTTPError("https://example.test", 401, "Unauthorized", {}, None)
            with patch("download_clevrer_humans_annotations.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "--archive-file PATH"):
                    download_and_extract(root / "out")
            self.assertFalse((root / "out").exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "source.zip"
            make_annotation_zip(archive_path)
            output_dir = root / "out"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                download_and_extract(output_dir, url=archive_path.as_uri())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_rejects_path_traversal_before_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "source.zip"
            make_annotation_zip(archive_path, extra_name="../escape.txt")
            output_dir = root / "out"

            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                download_and_extract(output_dir, url=archive_path.as_uri())
            self.assertFalse(output_dir.exists())
            self.assertFalse((root / "escape.txt").exists())

    def test_rejects_archive_missing_required_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "incomplete.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("README.md", "incomplete")

            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "missing expected"):
                    _validate_members(archive)


if __name__ == "__main__":
    unittest.main()
