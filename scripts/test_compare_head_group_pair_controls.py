"""Dependency-free tests for distance-matched pair comparison."""
import copy
import unittest

from compare_head_group_pair_controls import compare


def run(groups, values, sample_ids=('a', 'b')):
    records = []
    for index, sample_id in enumerate(sample_ids):
        layers = {'6': {key: {'patch_rescue': values[index][key]}
                         for key in values[index]}}
        records.append({'sample_id': sample_id, 'full_sequence_logprob': -2,
                        'swapped_sequence_logprob': -3, 'swap_drop': -1,
                        'grounding': [[0, 1]], 'sampled_indices': [0, 2],
                        'sampled_timestamps': [0, 2], 'evidence_indices': [0],
                        'control_indices': [2], 'layers': layers,
                        'self_checks': [{'absolute_error': 0.0} for _ in range(2 * len(groups))]})
    return {'layers': [6], 'groups': groups, 'samples_file': 'x', 'model': 'm',
            'quant': '4bit', 'dtype': 'bfloat16', 'attn_implementation': 'eager',
            'num_frames': 2, 'patch_scope': 'all', 'strict_grounding': True,
            'n_requested': 2, 'self_check': True, 'self_check_atol': 1e-4,
            'self_check_checks_per_record': 2 * len(groups), 'records': records}


class PairControlTests(unittest.TestCase):
    def test_paired_distance_contrast(self):
        base_groups = [[0], [1], [2], [0, 1], [1, 2]]
        control_groups = [[0, 2]]
        base = run(base_groups, [
            {'0': .1, '1': .2, '2': .1, '0+1': .5, '1+2': .1},
            {'0': .0, '1': .1, '2': .0, '0+1': .1, '1+2': .4}],)
        control = run(control_groups, [
            {'0+2': .4}, {'0+2': .2}],)
        report = compare(base, control, n_boot=100)
        row = next(row for row in report['rows'] if row['metric'] == 'adjacent_minus_gap2')
        self.assertAlmostEqual(row['mean'], -.125)

    def test_rejects_mismatch(self):
        base_groups = [[0], [1], [0, 1]]
        control_groups = [[0, 2]]
        base = run(base_groups, [{'0': 0, '1': 0, '0+1': 0}, {'0': 0, '1': 0, '0+1': 0}])
        control = run(control_groups, [{'0+2': 0}, {'0+2': 0}])
        control['records'][1]['swap_drop'] = -2
        with self.assertRaises(ValueError):
            compare(base, control, n_boot=100)


if __name__ == '__main__':
    unittest.main()
