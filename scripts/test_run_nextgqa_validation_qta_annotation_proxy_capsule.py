from __future__ import annotations

import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_nextgqa_validation_qta_annotation_proxy as runner


class SourceCapsuleRunnerGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir(mode=0o700)
        (self.repo / ".git" / "objects").mkdir(parents=True)
        (self.repo / ".git" / "refs").mkdir()
        (self.repo / ".git" / "HEAD").write_text("ref: refs/heads/test\n")
        (self.repo / "scripts").mkdir()
        self.paths = ("scripts/runner.py", "scripts/qta_source_capsule.py")
        self.contents = {self.paths[0]: b"synthetic runner\n",
                         self.paths[1]: b"synthetic capsule utility\n"}
        for relative, data in self.contents.items():
            target = self.repo / relative
            target.write_bytes(data)
            target.chmod(0o644)
        self.capsules = self.base / "private-capsules"
        self.capsules.mkdir(mode=0o700)
        self.capsules.chmod(0o700)
        self.old_paths = runner.SOURCE_FILES
        runner.SOURCE_FILES = self.paths

    def tearDown(self):
        runner.SOURCE_FILES = self.old_paths
        self.temp.cleanup()

    def expected_map(self):
        return {path: hashlib.sha256(self.contents[path]).hexdigest()
                for path in self.paths}

    def test_capture_and_recovery_bind_exact_source_map_and_safe_metadata(self):
        metadata = runner.capture_verified_source_capsule(
            self.repo, self.capsules, "smoke-001", self.expected_map())
        self.assertEqual(metadata["basename"], "smoke-001.source-capsule.zip")
        self.assertNotIn(str(self.base), repr(metadata))
        self.assertEqual(metadata["verified_source_sha256"], self.expected_map())
        self.assertTrue(metadata["verified_before_data_access"])
        self.assertFalse(metadata["provenance_authenticated"])
        capsule = self.capsules / metadata["basename"]
        self.assertEqual(capsule.stat().st_mode & 0o777, 0o600)

    def test_hash_mismatch_fails_closed(self):
        wrong = self.expected_map()
        wrong[self.paths[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch") as raised:
            runner.capture_verified_source_capsule(
                self.repo, self.capsules, "mismatch", wrong)
        metadata = raised.exception.source_capsule
        failed = runner.failure_receipt(
            requested_n=1, stage="source_capsule_capture_verify",
            error_type="SourceCapsuleMismatch", preflight_attempted=0,
            preflight_completed=0, branch_counts={}, source_capsule=metadata)
        self.assertEqual(failed["source_capsule"], metadata)
        self.assertEqual(metadata["verified_source_sha256"], self.expected_map())

    def test_capsule_name_collision_does_not_overwrite(self):
        first = runner.capture_verified_source_capsule(
            self.repo, self.capsules, "same-name", self.expected_map())
        capsule = self.capsules / first["basename"]
        original = capsule.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.capture_verified_source_capsule(
                self.repo, self.capsules, "same-name", self.expected_map())
        self.assertEqual(capsule.read_bytes(), original)

    def test_cli_requires_explicit_capsule_directory_before_work(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                runner.main(["--n", "1", "--output-name", "synthetic",
                             "--user-authorized-local-only"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--source-capsule-dir", stderr.getvalue())

    def test_private_receipt_collision_is_detected_without_following_links(self):
        (self.repo / "data").mkdir()
        private = self.repo / "data/private_nextgqa"
        private.mkdir(mode=0o700)
        private.chmod(0o700)
        runner.assert_private_output_slot_free(self.repo, "fresh-001.json")
        existing = private / "existing.json"
        existing.write_bytes(b"preserve")
        with self.assertRaisesRegex(FileExistsError, "collision"):
            runner.assert_private_output_slot_free(self.repo, existing.name)
        self.assertEqual(existing.read_bytes(), b"preserve")
        existing.unlink()
        dangling = private / "dangling.json"
        dangling.symlink_to(private / "missing.json")
        with self.assertRaisesRegex(FileExistsError, "collision"):
            runner.assert_private_output_slot_free(self.repo, dangling.name)
        self.assertTrue(dangling.is_symlink())

    def test_private_receipt_name_is_rejected_before_work(self):
        with self.assertRaisesRegex(ValueError, "filename_invalid"):
            runner.assert_private_output_slot_free(self.repo, "../escape.json")


if __name__ == "__main__":
    unittest.main()
