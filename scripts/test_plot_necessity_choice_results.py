import unittest

from plot_necessity_choice_results import csv_text, svg_text, validate_pairing


def result(sample_id='v:q', baseline_correct=True):
    record = {
        'sample_id': sample_id,
        'video_id': 'v',
        'qid': 'q',
        'sampled_indices': [1, 2],
        'evidence_mask_indices': [1],
        'control_mask_indices': [2],
        'masked_frame_count': 1,
        'baseline_choice_correct': baseline_correct,
        'gold_choice_index': 0,
    }
    summary = {
        'n': 1,
        'mean': 0.2,
        'sd': 0.0,
        'ci95_pointwise': [0.2, 0.2],
        'positive_fraction': 1.0,
        'signflip_p_two_sided': 1.0,
    }
    return {
        'records': [record],
        'summary': {
            'all': summary,
            'baseline_correct_primary': summary,
            'baseline_incorrect_secondary': {**summary, 'n': 0, 'ci95_pointwise': None},
        },
    }


def comparison():
    summary = {
        'n': 1,
        'mean': 0.0,
        'sd': 0.0,
        'ci95_pointwise': [0.0, 0.0],
        'positive_fraction': 0.0,
        'signflip_p_two_sided': 1.0,
    }
    return {
        'n_paired': 1,
        'comparisons': {
            cohort: {'blur_minus_black': summary}
            for cohort in ('all', 'baseline_correct_primary', 'baseline_incorrect_secondary')
        },
    }


class PlotNecessityResultsTests(unittest.TestCase):
    def test_accepts_exact_paired_ids_and_protocol(self):
        self.assertEqual(validate_pairing(result(), result(), comparison()), 1)

    def test_rejects_different_ids(self):
        with self.assertRaisesRegex(ValueError, 'identical sample IDs'):
            validate_pairing(result('a'), result('b'), comparison())

    def test_rejects_protocol_mismatch(self):
        blur = result()
        blur['records'][0]['control_mask_indices'] = [3]
        with self.assertRaisesRegex(ValueError, 'control_mask_indices'):
            validate_pairing(result(), blur, comparison())

    def test_rejects_incomplete_comparison(self):
        comp = comparison()
        comp['n_paired'] = 2
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            validate_pairing(result(), result(), comp)

    def test_emits_csv_and_svg(self):
        black, blur, comp = result(), result(), comparison()
        csv_output = csv_text(black, blur, comp)
        self.assertIn('blur_minus_black', csv_output)
        self.assertNotIn('\r', csv_output)
        self.assertTrue(csv_output.endswith('\n'))
        svg = svg_text(black, blur, comp)
        self.assertIn('<svg', svg)
        self.assertIn('Baseline correct (primary)', svg)
        self.assertIn('pointwise percentile bootstrap', svg)


if __name__ == '__main__':
    unittest.main()
