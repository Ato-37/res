"""Tests for private output handling in the synthetic six-branch runner."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_qwen_visual_attention_boost_six_condition import (  # noqa: E402
    _write_private_json,
)


class PrivateOutputTests(unittest.TestCase):
    def test_writes_mode_0600_json_in_mode_0700_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "private"
            output = output_dir / "summary.json"
            _write_private_json(output, {"status": "passed"})
            self.assertEqual(output_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})

    def test_refuses_existing_nonprivate_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "shared"
            output_dir.mkdir(mode=0o755)
            os.chmod(output_dir, 0o755)
            with self.assertRaisesRegex(
                    RuntimeError, "private_output_directory_permissions_invalid"):
                _write_private_json(output_dir / "summary.json", {"status": "blocked"})
            self.assertFalse((output_dir / "summary.json").exists())

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "private"
            output = output_dir / "summary.json"
            _write_private_json(output, {"status": "first"})
            with self.assertRaises(FileExistsError):
                _write_private_json(output, {"status": "second"})
            self.assertEqual(json.loads(output.read_text()), {"status": "first"})

    def test_removes_partial_file_after_nonfinite_json_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "private" / "summary.json"
            with self.assertRaises(ValueError):
                _write_private_json(output, {"value": float("nan")})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
