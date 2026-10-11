"""CPU tests for cumulative layer-randomization summaries and accounting."""

import unittest

from faithfulness_layer_randomization_utils import (
    summarize_cumulative_records,
    validate_accounting,
)


def result_fixture():
    records = [
        {'sample_key': 'sample_000', 'randomized_layer_order': [27, 26],
         'cumulative_conditions': [
            {'randomized_layer': 27, 'cumulative_randomized_layers': 1, 'spearman_rho': 0.4,
             'top4_overlap_fraction': 0.25, 'prediction_changed': True},
            {'randomized_layer': 26, 'cumulative_randomized_layers': 2, 'spearman_rho': 0.2,
             'top4_overlap_fraction': 0.0, 'prediction_changed': True},
        ]},
        {'sample_key': 'sample_001', 'randomized_layer_order': [27, 26],
         'cumulative_conditions': [
            {'randomized_layer': 27, 'cumulative_randomized_layers': 1, 'spearman_rho': 0.6,
             'top4_overlap_fraction': 0.0, 'prediction_changed': False},
            {'randomized_layer': 26, 'cumulative_randomized_layers': 2, 'spearman_rho': 0.0,
             'top4_overlap_fraction': 0.0, 'prediction_changed': True},
        ]},
    ]
    return {
        'n_requested': 2, 'n_effective': 2, 'n_failures': 0,
        'records': records, 'randomized_layer_order': [27, 26],
        'descriptive_by_randomized_depth': summarize_cumulative_records(records, 2),
    }


class LayerRandomizationUtilsTests(unittest.TestCase):
    def test_summarizes_by_sample_and_cumulative_depth(self):
        summary = result_fixture()['descriptive_by_randomized_depth']
        self.assertAlmostEqual(summary[0]['mean_sample_spearman_rho'], 0.5)
        self.assertAlmostEqual(summary[0]['mean_sample_top4_overlap_fraction'], 0.125)
        self.assertAlmostEqual(summary[0]['prediction_change_fraction'], 0.5)
        self.assertAlmostEqual(summary[1]['mean_sample_spearman_rho'], 0.1)

    def test_accepts_consistent_result_accounting(self):
        self.assertTrue(validate_accounting(result_fixture(), 2, 2))

    def test_rejects_accounting_or_order_mismatch(self):
        result = result_fixture()
        result['n_failures'] = 1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            validate_accounting(result, 2)
        result = result_fixture()
        result['randomized_layer_order'] = [26, 27]
        with self.assertRaisesRegex(ValueError, 'last-to-first'):
            validate_accounting(result, 2)

    def test_rejects_missing_depth(self):
        records = result_fixture()['records'][:1]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            summarize_cumulative_records(records, 3)


if __name__ == '__main__':
    unittest.main()
