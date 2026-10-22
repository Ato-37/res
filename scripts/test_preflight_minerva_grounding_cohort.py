import unittest

from pathlib import Path
import tempfile

from preflight_minerva_grounding_cohort import (
    load_time_map_seconds, normalize_manifest_path)


class ManifestPathTests(unittest.TestCase):
    def test_normalizes_official_dot_prefix(self):
        self.assertEqual(normalize_manifest_path("./Videos/P05/video.mp4"),
                         "Videos/P05/video.mp4")

    def test_normalizes_hash_manifest_marker(self):
        self.assertEqual(normalize_manifest_path("*./Videos/P05/video.mp4"),
                         "Videos/P05/video.mp4")

    def test_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "manifest_path_traversal"):
            normalize_manifest_path("../../private/file")

    def test_loads_float_encoded_nanosecond_times(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "time.csv"
            path.write_text("mp4_time_ns\n0.0\n33333333.0\n")
            self.assertEqual(load_time_map_seconds(path), [0.0, 0.033333333])

    def test_rejects_non_increasing_time_map(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "time.csv"
            path.write_text("mp4_time_ns\n0\n0\n")
            with self.assertRaisesRegex(ValueError, "time_map_invalid"):
                load_time_map_seconds(path)


if __name__ == "__main__":
    unittest.main()
