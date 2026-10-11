"""CPU tests for the cumulative layer-randomization analyzer."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from analyze_faithfulness_layer_randomization import analyze
from faithfulness_layer_randomization_utils import summarize_cumulative_records


def make_result():
    order = [1, 0]
    records = []
    for key, rhos, overlaps, predictions in [
            ('sample_000', [0.4, -0.2], [0.25, 0.0], [0, 1]),
            ('sample_001', [0.6, 0.0], [0.0, 0.0], [2, 2])]:
        conditions = []
        for i, layer in enumerate(order):
            conditions.append({
                'randomized_layer': layer,
                'cumulative_randomized_layers': i + 1,
                'spearman_rho': rhos[i],
                'top4_overlap_count': round(overlaps[i] * 4),
                'top4_overlap_fraction': overlaps[i],
                'randomized_prediction_index': predictions[i],
                'prediction_changed': predictions[i] != (0 if key == 'sample_000' else 2),
            })
        records.append({'sample_key': key,
                        'baseline_prediction_index': 0 if key == 'sample_000' else 2,
                        'randomized_layer_order': order,
                        'cumulative_conditions': conditions})
    return {
        'experiment': 'head_attribution_cascading_layer_randomization',
        'status': 'exploratory_sanity_check_not_faithfulness_test',
        'n_requested': 2, 'n_effective': 2, 'n_failures': 0,
        'records': records, 'randomized_layer_order': order,
        'num_layers_total': 2, 'heads_per_ranking': 784,
        'randomized_parameters': (
            'all decoder-block linear weights/biases and RMS/LayerNorm affine parameters'),
        'descriptive_by_randomized_depth': summarize_cumulative_records(records, 2),
        'randomization_mode': 'cumulative decoder-block reinitialization, last-to-first',
        'num_frames': 16, 'seed': 9, 'model': 'model', 'model_revision': 'rev',
        'gpu': 'gpu', 'quant': '4bit', 'runtime_seconds': 2.0, 'limitations': [],
    }


def make_manifest(result):
    return {key: result[key] for key in ('experiment', 'n_requested', 'n_effective',
                                           'n_failures', 'seed', 'quant', 'model',
                                           'randomized_layer_order', 'randomization_mode',
                                           'num_frames')} | {'samples_sha256': 'a' * 64}


class LayerRandomizationAnalyzerTests(unittest.TestCase):
    def test_validates_and_summarizes_result_and_manifest(self):
        result = make_result()
        report = analyze(result, make_manifest(result))
        self.assertEqual(report['n_effective'], 2)
        self.assertEqual(report['first_depth_mean_spearman_nonpositive'], 2)
        self.assertEqual(report['first_depth_mean_top4_overlap_zero'], 2)
        self.assertEqual(report['per_sample_final'][0]['top4_overlap_count'], 0)
        self.assertIn('descriptive only', report['inferential_tests'])

    def test_rejects_summary_mismatch(self):
        result = make_result()
        result['descriptive_by_randomized_depth'][0]['mean_sample_spearman_rho'] = 0.99
        with self.assertRaisesRegex(ValueError, 'summaries do not match'):
            analyze(result)

    def test_rejects_wrong_top4_fraction(self):
        result = make_result()
        result['records'][0]['cumulative_conditions'][0]['top4_overlap_fraction'] = 0.5
        result['descriptive_by_randomized_depth'] = summarize_cumulative_records(
            result['records'], 2)
        with self.assertRaisesRegex(ValueError, 'fraction/count mismatch'):
            analyze(result)

    def test_rejects_manifest_mismatch(self):
        result = make_result()
        manifest = make_manifest(result)
        manifest['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'mismatch for seed'):
            analyze(result, manifest)

    def test_rejects_invalid_randomized_prediction(self):
        result = make_result()
        result['records'][0]['cumulative_conditions'][0]['randomized_prediction_index'] = 5
        with self.assertRaisesRegex(ValueError, 'prediction index'):
            analyze(result)


if __name__ == '__main__':
    unittest.main()
