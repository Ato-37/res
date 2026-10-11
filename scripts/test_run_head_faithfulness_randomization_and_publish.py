"""CPU-only privacy and completeness tests for randomized-readout publishing."""

import unittest

from run_head_faithfulness_randomization_and_publish import (
    reject_private_fields,
    validate_counts,
)


class ReadoutRandomizationPublisherTests(unittest.TestCase):
    def test_anonymous_diagnostic_is_publishable(self):
        reject_private_fields({
            'records': [{'sample_key': 'sample_000', 'mean_spearman_rho': 0.2}],
            'failures': [],
        })
        validate_counts({'n_requested': 1, 'n_effective': 1, 'n_failures': 0,
                         'records': [{}]}, 1)

    def test_private_fields_are_rejected(self):
        for payload in ({'question': 'secret'}, {'video_path': '/private/x.mp4'}):
            with self.subTest(payload=payload), self.assertRaises(SystemExit):
                reject_private_fields(payload)

    def test_incomplete_counts_are_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'completeness'):
            validate_counts({'n_requested': 2, 'n_effective': 1, 'n_failures': 0,
                             'records': [{}]}, 2)

    def test_accounted_failure_counts_are_valid_for_publication(self):
        validate_counts({'n_requested': 2, 'n_effective': 1, 'n_failures': 1,
                         'records': [{}]}, 2)


if __name__ == '__main__':
    unittest.main()
