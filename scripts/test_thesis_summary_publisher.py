import tempfile
import unittest
from pathlib import Path

from run_thesis_summary_and_publish import (
    output_paths, recoverable_staged_batch, require_no_tracked_changes,
    test_environment,
)


class ThesisSummaryPublisherTests(unittest.TestCase):
    def test_allows_untracked_dataset_but_rejects_tracked_changes(self):
        require_no_tracked_changes('?? NExT-GQA/')
        require_no_tracked_changes('')
        with self.assertRaisesRegex(ValueError, 'Tracked worktree/index changes'):
            require_no_tracked_changes(' M docs/notes.md')

    def test_output_paths_are_inside_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            paths = output_paths(repo, 'results/run_1')
            self.assertEqual([p.name for p in paths], [
                'run_1_estimates.csv', 'run_1_forest.svg', 'run_1_metadata.json'])

    def test_rejects_path_outside_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            with self.assertRaisesRegex(ValueError, 'inside results'):
                output_paths(repo, 'docs/run')

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'results').mkdir()
            (repo / 'results/run_1_forest.svg').write_text('keep')
            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite'):
                output_paths(repo, 'results/run_1')

    def test_sets_scripts_path_for_unittest_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            env = test_environment(Path(directory), {'PATH': '/usr/bin'})
            self.assertEqual(env['PYTHONPATH'], str(Path(directory) / 'scripts'))
            self.assertEqual(env['PATH'], '/usr/bin')

    def test_recovers_only_complete_staged_runner_batch(self):
        prefix = 'results/necessity_choice_summary_ubuntu_20261006T123456Z'
        status = '\n'.join((
            f'A  {prefix}_estimates.csv',
            f'A  {prefix}_forest.svg',
            f'A  {prefix}_metadata.json',
            '?? NExT-GQA/',
        ))
        self.assertEqual(set(recoverable_staged_batch(status)), {
            f'{prefix}_estimates.csv', f'{prefix}_forest.svg', f'{prefix}_metadata.json'})

    def test_does_not_recover_partial_or_unrelated_staging(self):
        prefix = 'results/necessity_choice_summary_ubuntu_20261006T123456Z'
        self.assertIsNone(recoverable_staged_batch(f'A  {prefix}_forest.svg'))
        self.assertIsNone(recoverable_staged_batch('M  docs/notes.md'))


if __name__ == '__main__':
    unittest.main()
