"""CPU tests for exploratory evidence/control component decomposition."""
import unittest

from analyze_necessity_margin_components import decompose_results


def make_result(fill, evidence, control, correct):
    records = []
    for index, (e_drop, c_drop, is_correct) in enumerate(zip(evidence, control, correct)):
        records.append({
            'sample_id': f'video-{index}:q1',
            'baseline_choice_correct': is_correct,
            'gold_choice_index': 0,
            'sampled_indices': [0, 1, 2, 3],
            'evidence_mask_indices': [1],
            'control_mask_indices': [2],
            'evidence_margin_drop': e_drop,
            'control_margin_drop': c_drop,
            'selective_necessity_margin_drop': e_drop - c_drop,
        })
    return {
        'model': 'test', 'quant': '4bit', 'dtype': 'bfloat16', 'num_frames': 4,
        'primary_estimand': 'fixed', 'control_protocol': 'matched',
        'mask_intervention': {'fill': fill}, 'n_effective': len(records),
        'records': records, 'failures': [],
    }


class ComponentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.black = make_result('black', [0.4, 0.2, -0.1], [-0.2, 0.1, 0.0],
                                 [True, True, False])
        self.blur = make_result('blur', [0.5, 0.3, 0.1], [-0.1, -0.2, -0.1],
                                [True, True, False])

    def test_decomposes_raw_components_by_prespecified_cohorts(self):
        result = decompose_results(self.black, self.blur, draws=500, seed=7)
        self.assertEqual(result['n_paired'], 3)
        primary = result['cohorts']['baseline_correct_primary']
        self.assertEqual(primary['n'], 2)
        self.assertAlmostEqual(primary['black']['evidence_margin_drop']['mean'], 0.3)
        self.assertAlmostEqual(primary['black']['control_margin_drop']['mean'], -0.05)
        self.assertAlmostEqual(primary['black']['evidence_margin_drop']['mean']
                               - primary['black']['control_margin_drop']['mean'], 0.35)

    def test_rejects_protocol_and_pairing_mismatches(self):
        mismatch = make_result('blur', [0.5, 0.3, 0.1], [-0.1, -0.2, -0.1],
                               [True, True, False])
        mismatch['records'][0]['control_mask_indices'] = [3]
        with self.assertRaises(ValueError):
            decompose_results(self.black, mismatch, draws=500, seed=7)

    def test_rejects_component_that_does_not_reproduce_selective_effect(self):
        bad = make_result('black', [0.4, 0.2, -0.1], [-0.2, 0.1, 0.0],
                          [True, True, False])
        bad['records'][0]['selective_necessity_margin_drop'] += 0.1
        with self.assertRaises(ValueError):
            decompose_results(bad, self.blur, draws=500, seed=7)

    def test_rejects_duplicate_ids(self):
        bad = make_result('black', [0.4, 0.2, -0.1], [-0.2, 0.1, 0.0],
                          [True, True, False])
        bad['records'][1]['sample_id'] = bad['records'][0]['sample_id']
        with self.assertRaises(ValueError):
            decompose_results(bad, self.blur, draws=500, seed=7)


if __name__ == '__main__':
    unittest.main()
