import unittest
import json
from math import inf, nan
import torch

from necessity_choice_margin_test import (failure_code, sample_accounting_fields,
                                          score_conditions_with_accounting,
                                          summary_seed_schedule)


class FailureCodeTests(unittest.TestCase):
    def test_summary_seed_schedule_is_stable_and_distinct(self):
        self.assertEqual(summary_seed_schedule(2026), {
            'all': 2026,
            'baseline_correct_primary': 2027,
            'baseline_incorrect_secondary': 2028,
            'correct_minus_incorrect_interaction': 2029,
        })

    def test_oom_is_generic(self):
        self.assertEqual(failure_code(torch.cuda.OutOfMemoryError('secret path /x')), 'cuda_oom')

    def test_io_and_validation_codes(self):
        self.assertEqual(failure_code(FileNotFoundError('/private/video.mp4')), 'media_io_error')
        self.assertEqual(failure_code(ValueError('question text')), 'input_validation_error')

    def test_unknown_is_runtime(self):
        self.assertEqual(failure_code(RuntimeError('/private/video.mp4')), 'runtime_error')

    def test_nonfinite_sample_is_counted_once_without_partial_record(self):
        paths = {'full': 'full', 'evidence_mask': 'evidence', 'control_mask': 'control'}
        valid_scores = [-1.0, -2.0, -3.0, -4.0, -5.0]
        records, failures = [], []

        def run_fake_sample(index, invalid_value=None):
            def score(path, _question, _choices):
                values = list(valid_scores)
                if path == 'evidence' and invalid_value == 'short':
                    values = values[:4]
                elif path == 'evidence' and invalid_value is not None:
                    values[2] = invalid_value
                return values, {'synthetic': True}

            result, failure = score_conditions_with_accounting(
                index, score, paths, 'synthetic question', list('ABCDE'))
            if failure is not None:
                failures.append(failure)
            else:
                records.append(result)

        for index, invalid in enumerate((nan, inf, -inf, 'short')):
            with self.subTest(invalid=invalid):
                run_fake_sample(index, invalid)
        self.assertEqual(len(records), 0)
        self.assertEqual(failures, [
            {'sample_index': i, 'failure_code': 'input_validation_error'} for i in range(4)])

        run_fake_sample(4)
        self.assertEqual(len(records), 1)
        output = {**sample_accounting_fields(5, records, failures), 'records': records}
        self.assertEqual(output['n_requested'], output['n_effective'] + len(output['failures']))
        json.dumps(output, allow_nan=False)
