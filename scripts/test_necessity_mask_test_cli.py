"""Fail-fast argument checks for the legacy necessity mask runner."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import necessity_mask_test


class NecessityMaskCliTests(unittest.TestCase):
    def test_one_frame_is_rejected_before_model_or_sample_loading(self):
        stderr = io.StringIO()
        with patch.object(sys, "argv", [
                "necessity_mask_test.py", "--samples-file", "/missing/private.json",
                "--num-frames", "1", "--output", "/tmp/unused.json"]), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                necessity_mask_test.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--num-frames must be >=2", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
