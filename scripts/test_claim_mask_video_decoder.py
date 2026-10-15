"""Synthetic tests for the fail-closed two-pass PyAV decoder adapter."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import patch

from audit_qwen_pts_to_processor_cpu import encode_fixture
from claim_mask_video_decoder import (
    decode_sampled_video, decode_sampled_video_sets, verify_sampled_frame_pixels,
)


class ClaimMaskVideoDecoderTests(unittest.TestCase):
    def make_video(self, directory, pts):
        path = Path(directory) / "synthetic.mkv"
        encode_fixture(path, pts=tuple(pts), time_base=Fraction(1, 2))
        return path

    def test_two_pass_cfr_decode_binds_pts_order_pixels_and_exact_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            frames, meta = decode_sampled_video(
                path, num_frames=4, uniform_tolerance_seconds=Fraction(0))
        self.assertEqual(meta["decoded_frame_count"], 7)
        self.assertEqual([r["decoded_frame_index"] for r in meta["sampled_frames"]], [0, 2, 4, 6])
        self.assertEqual([r["relative_presentation_time"] for r in meta["sampled_frames"]],
                         [[0, 1], [1, 1], [2, 1], [3, 1]])
        self.assertEqual(meta["effective_sample_fps"], [1, 1])
        self.assertEqual(meta["max_sampled_grid_deviation_seconds"], [0, 1])
        self.assertEqual(meta["annotation_clock_calibrated"], False)
        self.assertEqual(meta["execution_authorization"], "not_granted")
        verify_sampled_frame_pixels(frames, meta)

    def test_irregular_sampled_pts_fail_frozen_uniform_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, [0, 1, 3, 4, 6])
            with self.assertRaisesRegex(ValueError, "uniform-grid tolerance"):
                decode_sampled_video(path, num_frames=4,
                                     uniform_tolerance_seconds=Fraction(0))

    def test_too_short_video_fails_instead_of_repeating_sample_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(3))
            with self.assertRaisesRegex(ValueError, "insufficient_distinct"):
                decode_sampled_video(path, num_frames=4,
                                     uniform_tolerance_seconds=Fraction(1))

    def test_selected_pixels_are_compared_across_decode_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            import claim_mask_video_decoder as decoder
            original = decoder._stream_and_decode

            def mutate_expected_pixel_digest(source):
                result = list(original(source))
                result[4] = list(result[4])
                result[4][0] = "0" * 64
                return tuple(result)

            with patch.object(decoder, "_stream_and_decode",
                              side_effect=mutate_expected_pixel_digest):
                with self.assertRaisesRegex(ValueError, "pixels_changed_between_decodes"):
                    decoder.decode_sampled_video(
                        path, num_frames=4, uniform_tolerance_seconds=Fraction(0))

    def test_frame_mutation_after_decode_is_rejected_at_processor_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            frames, meta = decode_sampled_video(
                path, num_frames=4, uniform_tolerance_seconds=Fraction(0))
        frames[0].putpixel((0, 0), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "pixel_hash_mismatch"):
            verify_sampled_frame_pixels(frames, meta)

    def test_source_hash_change_between_passes_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            import claim_mask_video_decoder as decoder
            with path.open("rb") as source:
                original = decoder._file_sha256(source)
            with patch.object(decoder, "_file_sha256", side_effect=[original, "0" * 64]):
                with self.assertRaisesRegex(ValueError, "input_changed_during_decode"):
                    decoder.decode_sampled_video(
                        path, num_frames=4, uniform_tolerance_seconds=Fraction(0))

    def test_path_replacement_during_first_pass_keeps_pinned_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            original_dir = Path(temp) / "original"
            replacement_dir = Path(temp) / "replacement"
            original_dir.mkdir()
            replacement_dir.mkdir()
            path = self.make_video(original_dir, range(7))
            replacement = self.make_video(replacement_dir, [0, 2, 4, 6, 8, 10, 12])
            original_hash = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            import claim_mask_video_decoder as decoder
            real_first_pass = decoder._stream_and_decode

            def replace_path_after_first_pass(source):
                result = real_first_pass(source)
                os.replace(replacement, path)
                return result

            with patch.object(decoder, "_stream_and_decode", side_effect=replace_path_after_first_pass):
                frames, meta = decoder.decode_sampled_video(
                    path, num_frames=4, uniform_tolerance_seconds=Fraction(0))
            self.assertEqual(meta["source_sha256"], original_hash)
            self.assertEqual(meta["decoded_frame_count"], 7)
            verify_sampled_frame_pixels(frames, meta)

    def test_multiple_sample_grids_share_the_same_two_verified_decode_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            import claim_mask_video_decoder as decoder
            first_pass = decoder._stream_and_decode
            second_pass = decoder._decode_selected_frames
            with patch.object(decoder, "_stream_and_decode", wraps=first_pass) as pass1, \
                    patch.object(decoder, "_decode_selected_frames", wraps=second_pass) as pass2:
                frame_sets, metadata_sets = decode_sampled_video_sets(
                    path, frame_counts=(4, 6), uniform_tolerance_seconds=Fraction(1, 2))
        self.assertEqual(pass1.call_count, 1)
        self.assertEqual(pass2.call_count, 1)
        self.assertEqual(set(frame_sets), {4, 6})
        self.assertEqual(len(frame_sets[4]), 4)
        self.assertEqual(len(frame_sets[6]), 6)
        self.assertEqual(metadata_sets[4]["source_sha256"], metadata_sets[6]["source_sha256"])
        self.assertEqual(metadata_sets[4]["decoded_pts_sequence_sha256"],
                         metadata_sets[6]["decoded_pts_sequence_sha256"])
        self.assertEqual(len(set(row["decoded_frame_index"] for row in
                                 metadata_sets[4]["sampled_frames"]
                                 + metadata_sets[6]["sampled_frames"])), 6)
        for count in (4, 6):
            verify_sampled_frame_pixels(frame_sets[count], metadata_sets[count])

    def test_multi_sample_grids_reject_duplicate_or_invalid_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            for counts in ((4, 4), (), (1, 4), (4, True)):
                with self.subTest(counts=counts), self.assertRaisesRegex(
                        ValueError, "frame_counts_must_be_ordered"):
                    decode_sampled_video_sets(
                        path, frame_counts=counts,
                        uniform_tolerance_seconds=Fraction(0))

    def test_each_sample_grid_is_checked_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_video(temp, range(7))
            import claim_mask_video_decoder as decoder
            original_check = decoder.uniform_sampling_rate_from_times
            with patch.object(decoder, "uniform_sampling_rate_from_times",
                              wraps=original_check) as grid_check:
                with self.assertRaisesRegex(ValueError, "uniform-grid tolerance"):
                    decode_sampled_video_sets(
                        path, frame_counts=(4, 6),
                        uniform_tolerance_seconds=Fraction(0))
            self.assertEqual(grid_check.call_count, 2)


if __name__ == "__main__":
    unittest.main()
