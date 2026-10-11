"""CPU-only integrity tests for the independent confirmation analyzer."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_head_faithfulness_confirmation import analyze, format_concise  # noqa: E402
from faithfulness_utils import (  # noqa: E402
    bootstrap_mean_interval,
    mean_t_interval,
    signflip_p_against,
)
from scipy.stats import ttest_1samp  # noqa: E402


THRESHOLD = 0.25


def example_data():
    records = []
    for index, target_drop in enumerate((0.2, -0.1, 0.4, 0.05, -0.2)):
        controls = [target_drop - offset + index * 0.03 for offset in
                    (0.1, -0.05, 0.02, -0.03, 0.04)]
        control_mean = sum(controls) / len(controls)
        selected = [
            {'layer': 0, 'head': 0, 'score': 1.0},
            {'layer': 0, 'head': 1, 'score': 0.9},
            {'layer': 1, 'head': 0, 'score': 0.8},
            {'layer': 1, 'head': 1, 'score': 0.7},
        ]
        groups = []
        for repeat in range(5):
            groups.append([
                {'layer': 0, 'head': 2 + repeat * 2},
                {'layer': 0, 'head': 3 + repeat * 2},
                {'layer': 1, 'head': 2 + repeat * 2},
                {'layer': 1, 'head': 3 + repeat * 2},
            ])
        records.append({
            'sample_key': f'sample_{index:03d}',
            'baseline_margin': 1.0,
            'baseline_prediction_index': index % 5,
            'baseline_foil_index': (index + 1) % 5,
            'baseline_choice_correct': index % 2 == 0,
            'selected_heads': selected,
            'control_groups': groups,
            'target_margin_drop': target_drop,
            'control_margin_drops': controls,
            'mean_control_margin_drop': control_mean,
            'selective_margin_drop': target_drop - control_mean,
        })

    values = [row['selective_margin_drop'] for row in records]
    interval = mean_t_interval(values)
    test = ttest_1samp(values, THRESHOLD)
    summary = {
        'n': len(values),
        'mean': interval['mean'],
        'sd': interval['sd'],
        'ci95_t': interval['ci'],
        'test_null_mean': THRESHOLD,
        't_statistic_vs_threshold': float(test.statistic),
        'p_two_sided_vs_threshold': float(test.pvalue),
        'bootstrap_ci95_pointwise': bootstrap_mean_interval(values, 10000, 20261010),
        'signflip_p_two_sided_vs_threshold': signflip_p_against(
            values, null=THRESHOLD, draws=10000, seed=20261011),
        'positive_fraction': sum(value > 0 for value in values) / len(values),
        'fraction_above_practical_threshold': sum(
            value > THRESHOLD for value in values) / len(values),
        'primary_threshold_supported': interval['ci'][0] > THRESHOLD,
    }
    result = {
        'experiment': 'gradient_activation_head_faithfulness',
        'primary_estimand': (
            'mean(target margin drop minus mean matched-control margin drop), '
            'one independent video per sample'),
        'n_requested': 5,
        'n_effective': 5,
        'n_failures': 0,
        'records': records,
        'top_k': 4,
        'control_repeats': 5,
        'alpha': 0.5,
        'layers': list(range(28)),
        'num_frames': 16,
        'practical_threshold_nat': THRESHOLD,
        'seed': 20261010,
        'primary_summary': summary,
        'runtime_seconds': 10.0,
        'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'quant': '4bit',
        'cap_pixels_per_frame': True,
    }
    manifest = {
        'experiment': result['experiment'],
        'model': result['model'],
        'quant': result['quant'],
        'seed': result['seed'],
        'layers': 'all',
        'top_k': result['top_k'],
        'alpha': result['alpha'],
        'control_repeats': result['control_repeats'],
        'num_frames': result['num_frames'],
        'cap_pixels_per_frame': True,
        'max_pixels': 200704,
        'n_requested': 5,
        'n_effective': 5,
        'n_failures': 0,
        'samples_sha256': 'a' * 64,
    }
    return result, manifest


class HeadFaithfulnessConfirmationTests(unittest.TestCase):
    def test_recomputes_primary_summary_and_validates_manifest(self):
        result, manifest = example_data()
        report = analyze(result, manifest)
        self.assertEqual(report['n_effective'], 5)
        self.assertEqual(report['n_unique_videos'], 5)
        self.assertAlmostEqual(
            report['mean_selective_margin_drop_nat'],
            result['primary_summary']['mean'])
        self.assertFalse(report['primary_threshold_supported'])

    def test_concise_format_omits_verbose_record_dump(self):
        result, manifest = example_data()
        output = format_concise(analyze(result, manifest))
        self.assertIn('5/5 complete; failures=0', output)
        self.assertIn('95% t CI', output)
        self.assertNotIn('sample_000', output)
        self.assertEqual(len(output.splitlines()), 5)

    def test_rejects_accounting_mismatch(self):
        result, manifest = example_data()
        result['n_failures'] = 1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            analyze(result, manifest)

    def test_rejects_selective_contrast_mismatch(self):
        result, manifest = example_data()
        result['records'][0]['selective_margin_drop'] += 0.1
        with self.assertRaisesRegex(ValueError, 'contrast mismatch'):
            analyze(result, manifest)

    def test_rejects_control_layer_histogram_mismatch(self):
        result, manifest = example_data()
        result['records'][0]['control_groups'][0][0].update(layer=1, head=20)
        with self.assertRaisesRegex(ValueError, 'layer histogram'):
            analyze(result, manifest)

    def test_rejects_manifest_mismatch(self):
        result, manifest = example_data()
        manifest['alpha'] = 0.25
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            analyze(result, manifest)

    def test_rejects_private_fields(self):
        result, manifest = example_data()
        result['records'][0]['video_path'] = '/private/video.mp4'
        with self.assertRaisesRegex(ValueError, 'private field'):
            analyze(result, manifest)


if __name__ == '__main__':
    unittest.main()
