import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.freeze_candidate_pool_manifest import freeze


class FreezeCandidatePoolManifestTests(unittest.TestCase):
    def make_pool(self, root: Path, count: int = 490):
        root.mkdir(mode=0o700)
        for i in range(count):
            (root / f"{i:03}.mp4").write_bytes(bytes([i % 256]))
            os.chmod(root / f"{i:03}.mp4", 0o600)

    def test_freezes_exact_set_with_exclusive_private_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, output = base / "pool", base / "manifest.json"
            self.make_pool(root)
            result = freeze(root, output, 490)
            self.assertEqual(result["video_count"], 490)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(json.loads(output.read_text())["entries"]), 490)
            with self.assertRaises(FileExistsError):
                freeze(root, output, 490)

    def test_rejects_symlink_video_without_creating_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, output = base / "pool", base / "manifest.json"
            self.make_pool(root, 489)
            outside = base / "outside.mp4"
            outside.write_bytes(b"x")
            (root / "link.mp4").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "candidate_tree_symlink_rejected"):
                freeze(root, output, 490)
            self.assertFalse(output.exists())

    def test_rejects_changed_tree_during_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, output = base / "pool", base / "manifest.json"
            self.make_pool(root)
            from scripts import freeze_candidate_pool_manifest as module
            original = module._hash_stable_file
            changed = False

            def racing_hash(root_fd, relative, snapshot, *, max_bytes):
                nonlocal changed
                result = original(root_fd, relative, snapshot, max_bytes=max_bytes)
                if not changed:
                    (root / relative).write_bytes(b"changed")
                    changed = True
                return result

            with mock.patch.object(module, "_hash_stable_file", side_effect=racing_hash):
                with self.assertRaisesRegex(ValueError, "candidate_tree_changed_during_freeze"):
                    freeze(root, output, 490)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
