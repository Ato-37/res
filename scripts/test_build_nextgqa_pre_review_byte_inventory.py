from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from build_nextgqa_pre_review_byte_inventory import (
    CODE_FILES,
    PACKAGE_DIR,
    PRIVATE_FILES,
    PRIVATE_ROOT,
    PUBLIC_OR_SOURCE_FILES,
    STAGED_MEDIA_DIR,
    build_inventory,
)


class SyntheticPreReviewInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir(mode=0o700)
        private_root = self.root / PRIVATE_ROOT
        private_root.mkdir(mode=0o700)
        for relative in PRIVATE_FILES.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(("synthetic private bytes " + relative.name).encode())
            path.chmod(0o600)
        for role, relative in PUBLIC_OR_SOURCE_FILES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic public bytes " + role).encode())
        for relative in CODE_FILES.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic source " + relative).encode())

        self.package = self.root / PACKAGE_DIR
        self.staged = self.root / STAGED_MEDIA_DIR
        self.package.mkdir(mode=0o700)
        self.staged.mkdir(mode=0o700)
        (self.package / "index.html").write_bytes(b"synthetic review page")
        (self.package / "index.html").chmod(0o600)
        self.media_names = ["synthetic-token-a.mp4", "synthetic-token-b.mp4",
                            "synthetic-token-c.mp4"]
        for index, name in enumerate(self.media_names):
            data = f"synthetic video bytes {index}".encode()
            for directory in (self.package, self.staged):
                path = directory / name
                path.write_bytes(data)
                path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_synthetic_inventory_is_candidate_only_and_redacted(self):
        output = self.root / PRIVATE_ROOT / "inventory-unique"
        result = build_inventory(self.root, output)
        raw = (output / "pre_review_candidate_byte_inventory.json").read_bytes()
        parsed = json.loads(raw)

        self.assertEqual(parsed, result)
        self.assertEqual(result["purpose"], "pre_review_candidate_byte_inventory")
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["review_started"])
        self.assertFalse(result["review_completed"])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["rights_status"], "unverified_local_only")
        self.assertEqual(result["missing_required_slots"], [])
        self.assertTrue(all(set(row) == {"role", "size_bytes", "sha256"}
                            for row in result["artifacts"]))
        encoded = raw.decode("ascii")
        for sensitive in (*self.media_names, str(self.root), "synthetic video bytes",
                          "synthetic private bytes", "synthetic-token"):
            self.assertNotIn(sensitive, encoded)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual((output / "pre_review_candidate_byte_inventory.json").stat().st_mode & 0o777,
                         0o600)

    def test_media_hash_or_basename_difference_is_missing_slot_not_pass(self):
        (self.staged / self.media_names[0]).write_bytes(b"changed synthetic video bytes")
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-hash-mismatch")
        self.assertIn("package_staged_media_byte_match", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_media_basename_difference_is_missing_slot_not_pass(self):
        old_path = self.staged / self.media_names[0]
        new_path = self.staged / "synthetic-other-token.mp4"
        old_path.rename(new_path)
        new_path.chmod(0o600)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-name-mismatch")
        self.assertIn("package_staged_media_basename_set_match", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])

    def test_private_input_symlink_or_wrong_mode_is_missing(self):
        symlink_path = self.root / PRIVATE_FILES["claim_input_manifest"]
        original = symlink_path.read_bytes()
        symlink_path.unlink()
        target = self.root / "synthetic-target"
        target.write_bytes(original)
        symlink_path.symlink_to(target)
        wrong_mode_path = self.root / PRIVATE_FILES["review_custody"]
        wrong_mode_path.chmod(0o644)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-invalid-inputs")
        self.assertIn("claim_input_manifest", result["missing_required_slots"])
        self.assertIn("review_custody", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["score_ready"])

    def test_extra_package_file_or_wrong_private_directory_mode_is_missing(self):
        extra = self.package / "extra.txt"
        extra.write_bytes(b"unplanned synthetic file")
        extra.chmod(0o600)
        self.package.chmod(0o755)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-extra-file")
        self.assertIn("review_package_exact_three_media_and_index",
                      result["missing_required_slots"])
        self.assertIn("review_package_contents", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_output_symlink_is_rejected_without_touching_target(self):
        target = self.root / PRIVATE_ROOT / "inventory-symlink-target"
        target.mkdir(mode=0o700)
        marker = target / "preserve.txt"
        marker.write_text("synthetic")
        link = self.root / PRIVATE_ROOT / "inventory-output-link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "output_directory_already_exists|output_directory_create_failed"):
            build_inventory(self.root, link)
        self.assertEqual(marker.read_text(), "synthetic")

    def test_missing_input_is_reported_without_authorizing(self):
        (self.root / PRIVATE_FILES["clock_anchor_plan"]).unlink()
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-missing")
        self.assertIn("clock_anchor_plan", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / PRIVATE_ROOT / "inventory-existing"
        output.mkdir(mode=0o700)
        marker = output / "preserve.txt"
        marker.write_text("synthetic")
        with self.assertRaisesRegex(ValueError, "output_directory_already_exists"):
            build_inventory(self.root, output)
        self.assertEqual(marker.read_text(), "synthetic")


if __name__ == "__main__":
    unittest.main()
