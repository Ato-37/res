"""CPU-only integrity tests for readout-randomization analysis."""

import unittest

from analyze_faithfulness_readout_randomization import summarize_result


def example_result():
    runs = [
        {'spearman_rho': 0.4, 'top_k_overlap_count': 0,
         'top_k_overlap_fraction': 0.0, 'randomized_prediction_index': 1},
        {'spearman_rho': 0.6, 'top_k_overlap_count': 2,
         'top_k_overlap_fraction': 0.5, 'randomized_prediction_index': 0},
    ]
    return {
        'experiment': 'head_attribution_output_readout_randomization_sanity',
        'status': 'exploratory_sanity_check_not_faithfulness_test',
        'n_requested': 1, 'n_effective': 1, 'n_failures': 0,
        'randomizations_per_sample': 2,
        'randomized_parameter_scope':
            'five A-E output-readout rows only; transformer layers unchanged',
        'records': [{
            'sample_key': 'sample_000', 'baseline_prediction_index': 0,
            'randomized_readout_runs': runs,
            'mean_spearman_rho': 0.5,
            'mean_top4_overlap_fraction': 0.25,
        }],
        'mean_sample_spearman_rho': 0.5,
        'mean_sample_top4_overlap_fraction': 0.25,
        'randomizations_per_sample': 2,
        'runtime_seconds': 10.0,
        'model_revision': 'abc',
        'limitations': [],
        'seed': 3,
        'quant': '4bit',
    }


class ReadoutRandomizationAnalysisTests(unittest.TestCase):
    def test_recomputes_descriptive_metrics(self):
        result = example_result()
        summary = summarize_result(result, {
            'experiment': result['experiment'], 'n_requested': 1,
            'n_effective': 1, 'n_failures': 0,
            'randomizations_per_sample': 2, 'seed': 3,
            'quant': '4bit', 'samples_sha256': 'a' * 64,
        })
        self.assertEqual(summary['n_randomization_runs'], 2)
        self.assertAlmostEqual(summary['mean_sample_spearman_rho'], 0.5)
        self.assertAlmostEqual(summary['mean_sample_top4_overlap_fraction'], 0.25)
        self.assertEqual(summary['prediction_change_count'], 1)
        self.assertTrue(summary['no_inferential_test'])

    def test_rejects_inconsistent_counts(self):
        result = example_result()
        result['n_failures'] = 1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            summarize_result(result)

    def test_rejects_published_mean_mismatch(self):
        result = example_result()
        result['records'][0]['mean_spearman_rho'] = 0.9
        with self.assertRaisesRegex(ValueError, 'per-sample Spearman mean mismatch'):
            summarize_result(result)

    def test_rejects_run_manifest_mismatch(self):
        result = example_result()
        with self.assertRaisesRegex(ValueError, 'mismatch for seed'):
            summarize_result(result, {
                'experiment': result['experiment'], 'n_requested': 1,
                'n_effective': 1, 'n_failures': 0,
                'randomizations_per_sample': 2, 'seed': 99,
                'quant': '4bit', 'samples_sha256': 'a' * 64,
            })


if __name__ == '__main__':
    unittest.main()
