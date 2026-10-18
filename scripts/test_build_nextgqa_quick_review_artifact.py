import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_nextgqa_quick_review_artifact import (
    DEFAULT_OUTPUT, browser_compatible_bytes, exploratory_item_id, private_output_path,
    serialize_inline_json,
)


class QuickReviewArtifactSafetyTests(unittest.TestCase):
    def test_default_output_targets_current_traceable_v2_artifact(self):
        self.assertEqual(
            DEFAULT_OUTPUT.name,
            "quick_review_ja_video_only_20261019_itemid_v2.html",
        )
        self.assertTrue(DEFAULT_OUTPUT.is_relative_to(
            Path(__file__).resolve().parents[1] / "data/private_nextgqa"
        ))

    def test_item_id_is_stable_opaque_and_does_not_echo_review_token(self):
        token = "private-review-token-123"
        first = exploratory_item_id(token)
        self.assertEqual(first, exploratory_item_id(token))
        self.assertNotEqual(first, exploratory_item_id(token + "x"))
        self.assertEqual(len(first), 20)
        self.assertNotIn(token, first)

    def test_inline_json_cannot_close_script_element(self):
        payload = {"question": "</script><script>alert('x')</script>"}
        serialized = serialize_inline_json(payload)
        self.assertNotIn("<", serialized)
        self.assertEqual(json.loads(serialized), payload)

    def test_output_must_stay_inside_private_data_directory(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):
                private_output_path(Path(outside) / "review.html")
        accepted = private_output_path(
            Path(__file__).resolve().parents[1]
            / "data/private_nextgqa/quick_review_test.html"
        )
        self.assertTrue(accepted.is_relative_to(Path(__file__).resolve().parents[1] / "data/private_nextgqa"))

    def test_browser_video_conversion_removes_audio_and_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")

            probe_calls = 0

            def fake_run(args, **kwargs):
                nonlocal probe_calls
                if args[0] == "ffprobe":
                    probe_calls += 1
                if args[0] == "ffprobe" and probe_calls == 1:
                    return type("Result", (), {"stdout": json.dumps({"streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {"codec_type": "audio", "codec_name": "aac"},
                        {"codec_type": "subtitle", "codec_name": "mov_text"},
                    ]})})()
                if args[0] == "ffmpeg":
                    self.assertIn("-an", args)
                    self.assertIn("-sn", args)
                    self.assertEqual(args[args.index("-map") + 1], "0:v:0")
                    Path(args[-1]).write_bytes(b"video-only")
                    return type("Result", (), {"stdout": ""})()
                return type("Result", (), {"stdout": json.dumps({"streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                ]})})()

            with patch("build_nextgqa_quick_review_artifact.subprocess.run", side_effect=fake_run):
                self.assertEqual(browser_compatible_bytes(source), b"video-only")


if __name__ == "__main__":
    unittest.main()
