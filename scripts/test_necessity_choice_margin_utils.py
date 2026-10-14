"""CPU tests for necessity choice-margin matching and summary utilities."""
import unittest
from math import inf, nan

from analyze_necessity_choice_mask_fill import compare_results
from necessity_choice_margin_utils import (apply_mask_fill, assess_frame_coverage,
                                           interaction_ci, nearest_control_indices,
                                           strict_evidence_indices, summarize,
                                           uniform_sample_indices,
                                           validate_choice_logprobs)


class NecessityChoiceMarginUtilsTests(unittest.TestCase):
    def test_uniform_sample_indices_preserve_legacy_rounding_and_short_clip_duplicates(self):
        self.assertEqual(uniform_sample_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(uniform_sample_indices(2, 4), [0, 0, 1, 1])
        self.assertEqual(uniform_sample_indices(7, 1), [0])

    def test_uniform_sample_indices_reject_invalid_dimensions(self):
        for total_frames, num_frames in ((0, 4), (4, 0), (-1, 4), (4, -1),
                                         (True, 4), (4, 2.5)):
            with self.subTest(total_frames=total_frames, num_frames=num_frames):
                with self.assertRaises(ValueError):
                    uniform_sample_indices(total_frames, num_frames)

    def test_video_preflight_uses_shared_sampling_implementation(self):
        from preflight_necessity_choice_manifest import sample_indices
        for total_frames, num_frames in ((1, 16), (7, 4), (101, 16)):
            with self.subTest(total_frames=total_frames, num_frames=num_frames):
                self.assertEqual(sample_indices(total_frames, num_frames),
                                 uniform_sample_indices(total_frames, num_frames))

    @staticmethod
    def _fill_result(fill, effects):
        return {
            'model': 'test-model', 'quant': '4bit', 'dtype': 'bfloat16', 'num_frames': 16,
            'primary_estimand': 'fixed', 'control_protocol': 'matched',
            'mask_intervention': {'fill': fill},
            'records': [
                {'sample_id': 'v1:q1', 'baseline_choice_correct': True,
                 'gold_choice_index': 0, 'sampled_indices': [0, 1, 2],
                 'evidence_mask_indices': [1], 'control_mask_indices': [2],
                 'selective_necessity_margin_drop': effects[0]},
                {'sample_id': 'v2:q2', 'baseline_choice_correct': False,
                 'gold_choice_index': 1, 'sampled_indices': [0, 1, 2],
                 'evidence_mask_indices': [0], 'control_mask_indices': [2],
                 'selective_necessity_margin_drop': effects[1]},
            ],
            'failures': [],
        }

    def test_fill_comparison_requires_exactly_paired_protocol_and_masks(self):
        black = self._fill_result('black', [0.2, -0.1])
        blur = self._fill_result('blur', [0.5, 0.1])
        result = compare_results(black, blur, draws=500, seed=12)
        self.assertEqual(result['n_paired'], 2)
        self.assertAlmostEqual(result['comparisons']['all']['blur_minus_black']['mean'], 0.25)
        blur['records'][0]['control_mask_indices'] = [0]
        with self.assertRaises(ValueError):
            compare_results(black, blur)

    def test_fill_comparison_rejects_different_sample_sets(self):
        black = self._fill_result('black', [0.2, -0.1])
        blur = self._fill_result('blur', [0.5, 0.1])
        blur['records'].pop()
        with self.assertRaises(ValueError):
            compare_results(black, blur)

    def test_strict_evidence_uses_sampled_timestamps_and_inclusive_bounds(self):
        self.assertEqual(strict_evidence_indices([0, 10, 20, 30], 10, [[1.0, 2.0]]),
                         [10, 20])

    def test_control_mask_is_equal_size_disjoint_and_nearest(self):
        indices = list(range(8))
        evidence = [3, 4]
        controls = nearest_control_indices(indices, evidence)
        self.assertEqual(len(controls), len(evidence))
        self.assertFalse(set(controls) & set(evidence))
        self.assertEqual(controls, [2, 5])

    def test_control_matching_ties_choose_lexicographically_lower_rank(self):
        self.assertEqual(nearest_control_indices([0, 1, 2, 3, 4], [2]), [1])

    def test_no_evidence_or_insufficient_controls_returns_empty(self):
        self.assertEqual(nearest_control_indices([0, 1, 2], []), [])
        self.assertEqual(nearest_control_indices([0, 1, 2], [0, 1]), [])

    def test_coverage_audit_distinguishes_missing_evidence_and_control(self):
        self.assertEqual(assess_frame_coverage([0, 10, 20], 10, [[0.2, 0.8]])['status'],
                         'no_evidence')
        broad = assess_frame_coverage(list(range(16)), 1, [[0, 13]])
        self.assertEqual(broad['status'], 'insufficient_control')
        self.assertEqual(len(broad['evidence']), 14)
        matched = assess_frame_coverage(list(range(8)), 1, [[2, 3]])
        self.assertEqual(matched['status'], 'ok')
        self.assertEqual(len(matched['evidence']), len(matched['control']))

    def test_mask_fill_preserves_frame_size_and_blur_preserves_nonblack_content(self):
        from PIL import Image
        image = Image.new('RGB', (21, 21), (0, 0, 0))
        for y in range(21):
            for x in range(10, 21):
                image.putpixel((x, y), (255, 255, 255))
        black = apply_mask_fill(image, 'black')
        blurred = apply_mask_fill(image, 'blur', 3.0)
        self.assertEqual(black.size, image.size)
        self.assertEqual(black.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(blurred.size, image.size)
        self.assertNotEqual(blurred.getpixel((10, 10)), image.getpixel((10, 10)))

    def test_blur_fill_rejects_nonpositive_radius_and_unknown_fill(self):
        from PIL import Image
        image = Image.new('RGB', (2, 2))
        with self.assertRaises(ValueError):
            apply_mask_fill(image, 'blur', 0)
        with self.assertRaises(ValueError):
            apply_mask_fill(image, 'pixelate')

    def test_choice_logprobs_reject_nonfinite_or_wrong_choice_count(self):
        valid = [-3.0, -2.0, -1.0, -4.0, -5.0]
        self.assertEqual(validate_choice_logprobs(valid), valid)
        for invalid in ([0.0] * 4, [0.0, 0.0, nan, 0.0, 0.0],
                        [0.0, 0.0, inf, 0.0, 0.0], [0.0, 0.0, -inf, 0.0, 0.0]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_choice_logprobs(invalid)

    def test_summary_utilities_reject_nonfinite_values_and_too_few_draws(self):
        for values in ([1.0, nan], [1.0, inf], [1.0, -inf], [1.0, 'bad'],
                       [10**1000]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    summarize(values, draws=100, seed=1)
        with self.assertRaises(ValueError):
            interaction_ci([1.0, nan], [0.0, 1.0], draws=100, seed=1)
        for draws in (0, 1, 99, True, 100.0):
            with self.subTest(draws=draws):
                with self.assertRaises(ValueError):
                    summarize([1.0, 2.0], draws=draws, seed=1)
                with self.assertRaises(ValueError):
                    interaction_ci([1.0, 2.0], [0.0, 1.0], draws=draws, seed=1)

    def test_singleton_summary_does_not_claim_inferential_uncertainty(self):
        result = summarize([0.25], draws=100, seed=7)
        self.assertEqual(result['n'], 1)
        self.assertIsNone(result['ci95_pointwise'])
        self.assertIsNone(result['signflip_p_two_sided'])

    def test_bootstrap_and_interaction_are_reproducible(self):
        first = summarize([0.1, 0.3, -0.2], draws=500, seed=17)
        second = summarize([0.1, 0.3, -0.2], draws=500, seed=17)
        self.assertEqual(first, second)
        interaction = interaction_ci([0.3, 0.4], [-0.1, 0.0], 500, 17)
        self.assertGreater(interaction['mean_correct_minus_incorrect'], 0)
        self.assertEqual(len(interaction['ci95_pointwise']), 2)


if __name__ == '__main__':
    unittest.main()
