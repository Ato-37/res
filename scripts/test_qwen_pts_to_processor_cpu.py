"""Codec-backed tests for the synthetic PTS-to-Qwen fixture boundaries."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from audit_qwen_pts_to_processor_cpu import (
    decode_fixture, encode_fixture,
)
from necessity_choice_margin_utils import uniform_sample_indices
from presentation_time_utils import uniform_sampling_rate_from_times


class QwenPtsProcessorBoundaryTests(unittest.TestCase):
    def test_sampled_cfr_indices_retain_exact_pixel_and_time_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cfr.mkv"
            encode_fixture(path, pts=tuple(range(7)), time_base=Fraction(1, 2))
            _pts, _tb, times, frames = decode_fixture(path)
        indices = uniform_sample_indices(len(frames), 4)
        self.assertEqual(indices, [0, 2, 4, 6])
        sampled_times = [times[index] for index in indices]
        self.assertEqual(sampled_times, [Fraction(0), Fraction(1), Fraction(2), Fraction(3)])
        fps, deviation = uniform_sampling_rate_from_times(sampled_times, Fraction(0))
        self.assertEqual(fps, Fraction(1))
        self.assertEqual(deviation, Fraction(0))

    def test_endpoint_sampling_that_is_not_uniform_fails_qwen_fps_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cfr.mkv"
            encode_fixture(path, pts=tuple(range(8)), time_base=Fraction(1, 2))
            _pts, _tb, times, frames = decode_fixture(path)
        indices = uniform_sample_indices(len(frames), 4)
        self.assertEqual(indices, [0, 2, 5, 7])
        sampled_times = [times[index] for index in indices]
        with self.assertRaisesRegex(ValueError, "uniform-grid tolerance"):
            uniform_sampling_rate_from_times(sampled_times, Fraction(0))


if __name__ == "__main__":
    unittest.main()
