"""Usage: python -m unittest discover -s scripts -p 'test_compare_activation_patch_scopes.py'."""
import copy
import unittest
from compare_activation_patch_scopes import compare


def fixture():
    runs = {}
    for scope, value in [('all', 0.3), ('prompt', 1.4), ('answer', 1.1)]:
        runs[scope] = dict(patch_scope=scope, samples_file='samples.json', model='test',
            quant='4bit', dtype='bfloat16', attn_implementation='eager', num_frames=2,
            layers=[6], n_requested=1, n_effective=1, strict_grounding=True,
            self_check=True, self_check_atol=1e-4,
            records=[dict(sample_id='0', grounding=[[0, 1]], sampled_indices=[0, 2],
                sampled_timestamps=[0, 2], evidence_indices=[0], control_indices=[2],
                full_sequence_logprob=-2, swapped_sequence_logprob=-3, swap_drop=-1,
                layers={'6': {'patch_rescue': value}}, self_checks=[
                    dict(input=name, layer=6, absolute_error=0) for name in ('full', 'swapped')])])
    return runs


class CompareTests(unittest.TestCase):
    def test_nonadditivity_and_singleton(self):
        report = compare(fixture(), 100)
        self.assertAlmostEqual(report['per_sample'][0]['nonadditivity'], -2.2)
        self.assertIsNone(report['rows'][0]['mean_ci95'])

    def test_reject_mismatch(self):
        for key, value in [('sample_id', 'other'), ('full_sequence_logprob', -9),
                           ('control_indices', [3]), ('self_checks', [])]:
            runs = fixture()
            runs['answer']['records'][0][key] = value
            with self.assertRaises(ValueError):
                compare(runs, 100)

    def test_duplicate_and_failed_checks(self):
        runs = fixture()
        runs['prompt']['records'].append(copy.deepcopy(runs['prompt']['records'][0]))
        with self.assertRaises(ValueError):
            compare(runs, 100)
        runs = fixture()
        runs['all']['records'][0]['self_checks'][0]['absolute_error'] = 1
        with self.assertRaises(ValueError):
            compare(runs, 100)


if __name__ == '__main__':
    unittest.main()
