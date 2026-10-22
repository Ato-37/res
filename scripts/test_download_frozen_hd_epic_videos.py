import unittest

from download_frozen_hd_epic_videos import build_ranges


class RangeGeometryTests(unittest.TestCase):
    def test_ranges_cover_file_without_gaps_or_overlap(self):
        ranges = build_ranges(23, 8)
        self.assertEqual(ranges, [(0, 7), (8, 15), (16, 22)])
        self.assertEqual(sum(end - start + 1 for start, end in ranges), 23)

    def test_exact_multiple_has_no_empty_trailing_range(self):
        self.assertEqual(build_ranges(16, 8), [(0, 7), (8, 15)])

    def test_invalid_range_geometry_fails_closed(self):
        for size, chunk in ((0, 4), (-1, 4), (4, 0)):
            with self.subTest(size=size, chunk=chunk):
                with self.assertRaisesRegex(ValueError, "range_geometry_invalid"):
                    build_ranges(size, chunk)


if __name__ == "__main__":
    unittest.main()
