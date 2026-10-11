"""Dependency-free tests for existing-result path safety."""
from pathlib import Path
import tempfile
import unittest

from publish_existing_head_group_result import git_identity, result_path


class ExistingPublishTests(unittest.TestCase):
    def test_result_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            self.assertEqual(result_path('results/a.json', repo),
                             (repo / 'results/a.json').resolve())

    def test_rejects_outside(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            with self.assertRaises(ValueError):
                result_path('../a.json', repo)

    def test_missing_identity_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaises(SystemExit) as context:
                git_identity(repo)
            self.assertIn('Missing local Git config', str(context.exception))


if __name__ == '__main__':
    unittest.main()
