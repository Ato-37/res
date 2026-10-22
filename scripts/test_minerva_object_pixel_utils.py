import unittest

import numpy as np
from PIL import Image

from minerva_object_pixel_utils import (
    clip_bbox_xyxy,
    mean_fill_box,
    select_spatial_sham_box,
)


class ClipBboxTests(unittest.TestCase):
    def test_clips_and_rasterizes_half_open_integer_box(self):
        result = clip_bbox_xyxy((-0.25, 1.2, 3.1, 4.0), 5, 6,
                                max_clipped_fraction=0.1)
        self.assertEqual(result["box_xyxy"], (0, 1, 4, 4))
        self.assertEqual(result["clipped_bbox_xyxy"], (0.0, 1.2, 3.1, 4.0))
        self.assertAlmostEqual(result["clipped_area_fraction"], 0.07462686567164178)

    def test_rejects_materially_oversized_bbox_by_clipping_fraction(self):
        with self.assertRaisesRegex(ValueError, "bbox_clipping_fraction_exceeded"):
            clip_bbox_xyxy((-2, 1, 8, 5), 6, 6, max_clipped_fraction=0.2)

    def test_accepts_oversized_bbox_when_threshold_explicitly_allows_it(self):
        result = clip_bbox_xyxy((-2, -1, 8, 5), 6, 6,
                                max_clipped_fraction=0.5)
        self.assertEqual(result["box_xyxy"], (0, 0, 6, 5))
        self.assertAlmostEqual(result["clipped_area_fraction"], 0.5)

    def test_rejects_malformed_boxes_and_invalid_dimensions(self):
        for bbox in ([0, 0, 0, 1], [1, 0, 0, 1], [0, 0, float("nan"), 1],
                     [0, False, 2, 3], [0, 0, 1]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                clip_bbox_xyxy(bbox, 4, 4)
        with self.assertRaisesRegex(ValueError, "frame_width_invalid"):
            clip_bbox_xyxy([0, 0, 1, 1], True, 4)
        with self.assertRaisesRegex(ValueError, "bbox_outside_frame"):
            clip_bbox_xyxy([5, 5, 6, 6], 4, 4)


class SpatialShamTests(unittest.TestCase):
    def test_sham_is_deterministic_equal_size_and_disjoint(self):
        args = dict(support_box=(2, 2, 4, 4), width=8, height=8,
                    forbidden_boxes=[(0, 0, 2, 2)], seed="pilot-row-a-v1")
        first = select_spatial_sham_box(**args)
        self.assertEqual(first, select_spatial_sham_box(**args))
        self.assertEqual(first[2] - first[0], 2)
        self.assertEqual(first[3] - first[1], 2)
        for forbidden in [args["support_box"], *args["forbidden_boxes"]]:
            self.assertFalse(_overlap(first, forbidden))
        self.assertGreaterEqual(first[0], 0)
        self.assertGreaterEqual(first[1], 0)
        self.assertLessEqual(first[2], args["width"])
        self.assertLessEqual(first[3], args["height"])

    def test_sham_fails_when_forbidden_boxes_cover_all_positions(self):
        with self.assertRaisesRegex(ValueError, "spatial_sham_unavailable"):
            select_spatial_sham_box((0, 0, 2, 2), 2, 2, [], "seed")

    def test_sham_rejects_malformed_forbidden_boxes(self):
        with self.assertRaisesRegex(ValueError, "excluded_box_out_of_bounds"):
            select_spatial_sham_box((0, 0, 1, 1), 3, 3,
                                    [(2, 2, 4, 4)], "seed")


class MeanFillTests(unittest.TestCase):
    def test_fills_box_with_original_frame_mean_and_preserves_other_pixels(self):
        pixels = np.array([
            [[0, 10, 20], [10, 20, 30], [20, 30, 40], [30, 40, 50]],
            [[40, 50, 60], [50, 60, 70], [60, 70, 80], [70, 80, 90]],
            [[80, 90, 100], [90, 100, 110], [100, 110, 120], [110, 120, 130]],
        ], dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        result = np.asarray(mean_fill_box(image, (1, 1, 3, 3)))
        mean_rgb = np.rint(pixels.mean(axis=(0, 1), dtype=np.float64)).astype(np.uint8)
        np.testing.assert_array_equal(result[1:3, 1:3],
                                      np.broadcast_to(mean_rgb, (2, 2, 3)))
        unchanged = np.ones(pixels.shape[:2], dtype=bool)
        unchanged[1:3, 1:3] = False
        np.testing.assert_array_equal(result[unchanged], pixels[unchanged])
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_mean_fill_rejects_invalid_inputs_and_out_of_bounds_boxes(self):
        image = Image.new("RGB", (3, 3), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "integer_box_out_of_bounds"):
            mean_fill_box(image, (0, 0, 4, 2))
        with self.assertRaisesRegex(ValueError, "image_must_be_rgb_pil"):
            mean_fill_box(Image.new("L", (3, 3)), (0, 0, 1, 1))


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


if __name__ == "__main__":
    unittest.main()
