from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from smoke_qwen_real_video_attention_boost import _write_private_json


class RealVideoAttentionBoostOutputTests(unittest.TestCase):
    def test_output_is_exclusive_private_and_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "private-run"
            output = parent / "summary.json"
            expected = {"status": "passed", "faithfulness_result": False}
            _write_private_json(output, expected)
            self.assertEqual(parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
            with self.assertRaises(FileExistsError):
                _write_private_json(output, {"status": "overwrite"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)

    def test_refuses_a_preexisting_public_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "shared"
            parent.mkdir(mode=0o755)
            os.chmod(parent, 0o755)
            output = parent / "summary.json"
            with self.assertRaisesRegex(
                    RuntimeError, "private_output_directory_permissions_invalid"):
                _write_private_json(output, {"status": "blocked"})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
