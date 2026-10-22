import unittest

from analyze_noaa_qta_annotation_time_alignment import (
    nearest_temporal_group,
    validate_manifest_hashes,
    validate_group_indices,
    validate_run_rows,
)


class NoaaQtaAnnotationTimeAlignmentTests(unittest.TestCase):
    def test_nearest_frame_maps_to_adjacent_temporal_group(self):
        pts = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0]
        self.assertEqual(nearest_temporal_group(0.0, pts, 4), 0)
        self.assertEqual(nearest_temporal_group(119.0, pts, 4), 1)
        self.assertEqual(nearest_temporal_group(160.0, pts, 4), 2)
        self.assertEqual(nearest_temporal_group(280.0, pts, 4), 3)

    def test_exact_tie_uses_earlier_frame_deterministically(self):
        self.assertEqual(nearest_temporal_group(20.0,
            [0.0, 40.0, 80.0, 120.0], 2), 0)

    def test_rejects_invalid_geometry_or_timestamp(self):
        for timestamp, pts, groups in [
            (-1.0, [0.0, 1.0], 1),
            (float("nan"), [0.0, 1.0], 1),
            (0.0, [0.0, 1.0, 2.0], 2),
            (0.0, [1.0, 0.0], 1),
        ]:
            with self.subTest(timestamp=timestamp, pts=pts, groups=groups):
                with self.assertRaisesRegex(ValueError, "temporal_mapping_inputs_invalid"):
                    nearest_temporal_group(timestamp, pts, groups)

    def test_manifest_hashes_require_exact_paths_and_values(self):
        expected = {"samples.json"}
        validate_manifest_hashes({"samples.json": "abc"}, {"samples.json": "abc"}, expected)
        for receipt, actual in [
            ({}, {"samples.json": "abc"}),
            ({"samples.json": "wrong"}, {"samples.json": "abc"}),
            ({"samples.json": "abc", "extra.json": "x"}, {"samples.json": "abc"}),
        ]:
            with self.subTest(receipt=receipt):
                with self.assertRaisesRegex(ValueError, "sample_manifest_receipt_hash_mismatch"):
                    validate_manifest_hashes(receipt, actual, expected)

    def test_run_rows_require_exact_unique_joins(self):
        samples = {}
        rows = []
        videos = {"v1", "v2", "v3"}
        for video_i in range(3):
            for q_i in range(2):
                key = (f"v{video_i + 1}", f"q{q_i}")
                token = f"t{video_i}{q_i}"
                samples[key] = ([], "method", token)
                rows.append({"source_video_sha256": key[0], "qid_sha256": key[1],
                             "row_token_sha256": token})
        self.assertEqual(validate_run_rows(rows, samples, videos), rows)
        for malformed in (rows[:-1], rows + [rows[0]], [*rows[:-1], rows[0]]):
            with self.subTest(count=len(malformed)):
                with self.assertRaises(ValueError):
                    validate_run_rows(malformed, samples, videos)

    def test_group_indices_reject_duplicates_and_overlap(self):
        self.assertEqual(validate_group_indices([1], [3], 4), ({1}, {3}))
        for qta, sham in [([1, 1], [3]), ([1], [1]), ([4], [2])]:
            with self.subTest(qta=qta, sham=sham):
                with self.assertRaisesRegex(ValueError, "qta_temporal_group_contract_invalid"):
                    validate_group_indices(qta, sham, 4)


if __name__ == "__main__":
    unittest.main()
