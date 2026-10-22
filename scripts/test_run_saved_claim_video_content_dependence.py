from __future__ import annotations

import unittest

from PIL import Image

from run_saved_claim_video_content_dependence import (
    content_logprob_contrast,
    make_constant_gray_frames,
    resolve_claim_prompt_suffix,
)


class SavedClaimVideoContentDependenceTests(unittest.TestCase):
    def test_claim_prompt_version_is_resolved_and_unknown_versions_rejected(self):
        suffixes = {"v1": " old ", "v2_short": " short "}
        self.assertEqual(resolve_claim_prompt_suffix({}, suffixes), ("v1", " old "))
        self.assertEqual(resolve_claim_prompt_suffix(
            {"generation_prompt_version": "v2_short"}, suffixes),
            ("v2_short", " short "))
        with self.assertRaisesRegex(ValueError, "prompt_version_invalid"):
            resolve_claim_prompt_suffix({"generation_prompt_version": "future"}, suffixes)

    def test_gray_transform_preserves_frame_count_dimensions_and_source(self):
        frames = [Image.new("RGB", (13, 7), (10 + i, 20, 30)) for i in range(3)]
        before = [frame.getpixel((0, 0)) for frame in frames]
        gray = make_constant_gray_frames(frames)
        self.assertEqual(len(gray), len(frames))
        self.assertEqual([frame.size for frame in gray], [frame.size for frame in frames])
        self.assertEqual([frame.mode for frame in gray], ["RGB"] * 3)
        for frame in gray:
            self.assertEqual(frame.getpixel((0, 0)), (127, 127, 127))
            self.assertEqual(frame.getpixel((12, 6)), (127, 127, 127))
        self.assertEqual([frame.getpixel((0, 0)) for frame in frames], before)

    def test_gray_value_bounds_and_empty_inputs_fail_closed(self):
        frame = Image.new("RGB", (1, 1))
        for gray in (-1, 256, True, 1.5):
            with self.subTest(gray=gray), self.assertRaisesRegex(ValueError, "gray_value"):
                make_constant_gray_frames([frame], gray)
        with self.assertRaisesRegex(ValueError, "frames_invalid"):
            make_constant_gray_frames([])

    def test_contrast_direction_and_finite_input_validation(self):
        self.assertAlmostEqual(content_logprob_contrast(-1.0, -1.4), 0.4)
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                content_logprob_contrast(value, -1.0)


if __name__ == "__main__":
    unittest.main()
