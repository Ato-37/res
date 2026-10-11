"""Dependency-free tests for head-group result validation and summaries."""
import copy
import unittest

from analyze_head_group_patching import analyze


def fixture():
    records = []
    for sample_id, drop, offset in [('a', -1.0, 0.2), ('b', 0.5, -0.1)]:
        records.append({'sample_id': sample_id, 'swap_drop': drop,
                        'layers': {'6': {'0': {'patch_rescue': offset},
                                         '1': {'patch_rescue': offset + 0.1},
                                         '0+1': {'patch_rescue': offset + 0.3}}},
                        'self_checks': [
                            {'input': name, 'layer': 6, 'heads': [0], 'absolute_error': 0.0}
                            for name in ('full', 'swapped') for _ in range(3)]})
    return {'layers': [6], 'groups': [[0], [1], [0, 1]], 'self_check': True,
            'self_check_atol': 1e-4, 'self_check_checks_per_record': 6,
            'n_requested': 2, 'records': records, 'failures': []}


class HeadAnalysisTests(unittest.TestCase):
    def test_summary_and_harmed_subset(self):
        report = analyze(fixture(), n_boot=100)
        self.assertEqual(report['n_effective'], 2)
        self.assertEqual(report['n_harmed'], 1)
        self.assertEqual(len(report['rows']), 8)

    def test_pair_excess_is_reported(self):
        report = analyze(fixture(), n_boot=100)
        interactions = [row for row in report['rows']
                        if row.get('metric') == 'pair_excess_over_singletons'
                        and row['scope'] == 'all']
        self.assertEqual(len(interactions), 1)
        self.assertAlmostEqual(interactions[0]['mean'], 0.15)

    def test_rejects_missing_group_or_check(self):
        broken = fixture()
        del broken['records'][0]['layers']['6']['1']
        with self.assertRaises(ValueError):
            analyze(broken, n_boot=100)
        broken = fixture()
        broken['records'][0]['self_checks'] = []
        with self.assertRaises(ValueError):
            analyze(broken, n_boot=100)

    def test_rejects_duplicate_id_and_failed_check(self):
        broken = fixture()
        broken['records'][1]['sample_id'] = 'a'
        with self.assertRaises(ValueError):
            analyze(broken, n_boot=100)

    def test_accepts_legacy_check_count_metadata(self):
        legacy = fixture()
        del legacy['self_check_checks_per_record']
        report = analyze(legacy, n_boot=100)
        self.assertEqual(report['n_effective'], 2)
        broken = fixture()
        broken['records'][0]['self_checks'][0]['absolute_error'] = 1.0
        with self.assertRaises(ValueError):
            analyze(broken, n_boot=100)


if __name__ == '__main__':
    unittest.main()
