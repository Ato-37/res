"""CPU-only guards for the validation smoke→pilot runner."""

import unittest

from run_faithfulness_validation_pilot import validate_effective_result


class ValidationPilotTests(unittest.TestCase):
    def test_complete_stage_is_accepted(self):
        validate_effective_result(
            {'n_requested': 3, 'n_effective': 3, 'n_failures': 0}, 3, 'smoke')

    def test_incomplete_stage_stops_the_pipeline(self):
        with self.assertRaisesRegex(SystemExit, 'will not start'):
            validate_effective_result(
                {'n_requested': 3, 'n_effective': 2, 'n_failures': 1}, 3, 'smoke')

    def test_bad_accounting_stops_the_pipeline(self):
        with self.assertRaisesRegex(SystemExit, 'accounting mismatch'):
            validate_effective_result(
                {'n_requested': 3, 'n_effective': 2, 'n_failures': 0}, 3, 'smoke')


if __name__ == '__main__':
    unittest.main()
