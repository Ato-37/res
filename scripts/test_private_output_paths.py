from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import unittest

from private_output_paths import validate_private_output


class PrivateOutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        (self.root / "data").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_creates_only_owner_private_allowlisted_directory(self):
        output = self.root / "data" / "private_nextgqa" / "packet.json"
        self.assertEqual(validate_private_output(output, self.root), output)
        self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
        self.assertEqual(output.parent.stat().st_uid, os.geteuid())

    def test_external_path_remains_allowed(self):
        output = Path(self.temp.name) / "outside" / "packet.json"
        self.assertEqual(validate_private_output(output, self.root), output)

    def test_rejects_other_repository_paths(self):
        with self.assertRaisesRegex(ValueError, "repository_output_not_private"):
            validate_private_output(self.root / "results" / "private.json", self.root)

    def test_rejects_nested_private_paths(self):
        with self.assertRaisesRegex(ValueError, "repository_output_not_private"):
            validate_private_output(self.root / "data" / "private_nextgqa" / "nested" / "x.json", self.root)

    def test_rejects_existing_file_and_dangling_symlink(self):
        store = self.root / "data" / "private_nextgqa"
        store.mkdir(mode=0o700)
        os.chmod(store, 0o700)
        existing = store / "existing.json"
        existing.write_text("x")
        with self.assertRaises(FileExistsError):
            validate_private_output(existing, self.root)
        dangling = store / "dangling.json"
        dangling.symlink_to(store / "missing.json")
        with self.assertRaises(FileExistsError):
            validate_private_output(dangling, self.root)

    def test_rejects_wrong_private_directory_mode(self):
        store = self.root / "data" / "private_nextgqa"
        store.mkdir(mode=0o700)
        os.chmod(store, 0o755)
        with self.assertRaisesRegex(ValueError, "private_output_directory_mode_invalid"):
            validate_private_output(store / "x.json", self.root)

    def test_rejects_symlinked_private_directory(self):
        actual = Path(self.temp.name) / "actual"
        actual.mkdir(mode=0o700)
        os.chmod(actual, 0o700)
        (self.root / "data" / "private_nextgqa").symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "private_output_directory_invalid"):
            validate_private_output(self.root / "data" / "private_nextgqa" / "x.json", self.root)

    def test_rejects_external_symlink_alias_into_repository(self):
        private = self.root / "data" / "private_nextgqa"
        private.mkdir(mode=0o700)
        os.chmod(private, 0o700)
        alias = Path(self.temp.name) / "alias.json"
        alias.symlink_to(private / "missing.json")
        with self.assertRaisesRegex(ValueError, "private_output_symlink_into_repository"):
            validate_private_output(alias, self.root)


if __name__ == "__main__":
    unittest.main()
