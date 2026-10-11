"""CPU-only integrity tests for the faithfulness pilot analyzer."""

import unittest

from analyze_faithfulness_pilot import summarize_result


def example_result():
    rows = [
        {'sample_key': 'sample_000', 'target_margin_drop': 0.5,
         'mean_control_margin_drop': 0.1, 'selective_margin_drop': 0.4,
         'baseline_choice_correct': True},
        {'sample_key': 'sample_001', 'target_margin_drop': -0.1,
         'mean_control_margin_drop': 0.0, 'selective_margin_drop': -0.1,
         'baseline_choice_correct': False},
    ]
    return {
        'n_requested': 2, 'n_effective': 2, 'n_failures': 0,
        'records': rows,
        'primary_summary': {'mean': 0.15, 'sd': 0.3535533905932738,
                            'ci95_t': [-3.027, 3.327],
                            'p_two_sided_vs_threshold': 0.8,
                            'primary_threshold_supported': False},
        'runtime_seconds': 12.0,
    }


class FaithfulnessPilotAnalysisTests(unittest.TestCase):
    def test_summary_recomputes_contrasts_and_leave_one_out(self):
        summary = summarize_result(example_result())
        self.assertEqual(summary['n_unique_sample_keys'], 2)
        self.assertAlmostEqual(summary['selective_mean'], 0.15)
        self.assertEqual(summary['selective_positive_n'], 1)
        self.assertAlmostEqual(summary['target_margin_drop_mean'], 0.2)
        self.assertAlmostEqual(summary['control_margin_drop_mean'], 0.05)
        self.assertAlmostEqual(summary['leave_one_out_mean_min'], -0.1)
        self.assertAlmostEqual(summary['leave_one_out_mean_max'], 0.4)

    def test_rejects_accounting_errors(self):
        result = example_result()
        result['n_failures'] = 1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            summarize_result(result)

    def test_rejects_contrast_mismatch(self):
        result = example_result()
        result['records'][0]['selective_margin_drop'] = 0.9
        with self.assertRaisesRegex(ValueError, 'contrast mismatch'):
            summarize_result(result)

    def test_rejects_duplicate_keys(self):
        result = example_result()
        result['records'][1]['sample_key'] = 'sample_000'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize_result(result)


if __name__ == '__main__':
    unittest.main()
