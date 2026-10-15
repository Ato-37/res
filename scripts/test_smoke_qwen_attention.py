import ast
from pathlib import Path
import tempfile
import unittest

import cv2

from smoke_qwen_attention import make_synthetic_video


class SyntheticAttentionSmokeTests(unittest.TestCase):
    def test_fixture_is_a_closed_synthetic_video_with_expected_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.mp4"
            digest = make_synthetic_video(path)
            self.assertEqual(len(digest), 64)
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                count = 0
                while True:
                    ok, _ = capture.read()
                    if not ok:
                        break
                    count += 1
                self.assertEqual(count, 24)
            finally:
                capture.release()

    def test_smoke_no_longer_invokes_retired_real_media_scorer(self):
        path = Path(__file__).with_name("smoke_qwen_attention.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            for node in calls
        }
        self.assertNotIn("run", names)
        self.assertNotIn("Popen", names)
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("grounding_check.py\"),", source)
        self.assertIn('from grounding_check import extract_frames', source)


if __name__ == "__main__":
    unittest.main()
