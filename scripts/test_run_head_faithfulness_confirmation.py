"""CPU tests for the guarded head-faithfulness confirmation runner."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from run_head_faithfulness_confirmation import validate_result


class ConfirmationRunnerTests(unittest.TestCase):
    def test_accepts_complete_smoke(self):
        self.assertIsNone(validate_result(
            {'n_requested': 3, 'n_effective': 3, 'n_failures': 0},
            3, 'smoke'))

    def test_stops_after_smoke_failure(self):
        with self.assertRaisesRegex(SystemExit, 'stopping without replacement'):
            validate_result(
                {'n_requested': 3, 'n_effective': 2, 'n_failures': 1},
                3, 'smoke')

    def test_accepts_confirmation_above_minimum_effective(self):
        self.assertIsNone(validate_result(
            {'n_requested': 79, 'n_effective': 72, 'n_failures': 7},
            79, 'confirmation', 71))

    def test_stops_if_confirmation_is_underpowered(self):
        with self.assertRaisesRegex(SystemExit, 'required effective n >= 71'):
            validate_result(
                {'n_requested': 79, 'n_effective': 70, 'n_failures': 9},
                79, 'confirmation', 71)

    def test_rejects_inconsistent_accounting(self):
        with self.assertRaisesRegex(SystemExit, 'accounting mismatch'):
            validate_result(
                {'n_requested': 79, 'n_effective': 70, 'n_failures': 8},
                79, 'confirmation', 71)


if __name__ == '__main__':
    unittest.main()
