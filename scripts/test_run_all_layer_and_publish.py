"""Dependency-free tests for publish path safety."""
from pathlib import Path
import tempfile
import unittest

from run_all_layer_and_publish import git_identity, repo_path


class PublishTests(unittest.TestCase):
    def test_accepts_results_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            self.assertEqual(repo_path('results/out.json', repo),
                             (repo / 'results/out.json').resolve())

    def test_rejects_outside_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            for value in ('out.json', '../out.json', 'results/../out.json'):
                with self.assertRaises(ValueError):
                    repo_path(value, repo)

    def test_missing_identity_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaises(SystemExit) as context:
                git_identity(repo)
            self.assertIn('Missing local Git config', str(context.exception))


if __name__ == '__main__':
    unittest.main()
