"""CPU-only command-construction checks for the choice-margin publisher."""
from argparse import Namespace
from unittest.mock import patch
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

from run_necessity_choice_margin_test_and_publish import (
    analysis_command, run_analysis, validate_sample_units)


class AnalysisCommandTests(unittest.TestCase):
    def test_bootstrap_draw_count_is_forwarded(self):
        args = Namespace(quant='4bit', seed=17, bootstrap_draws=2500,
                         num_frames=16, mask_fill='black', blur_radius=12.0,
                         n_samples=5, preflight_report=Path('/private/gate.json'))
        command = analysis_command(Path('/repo'), Path('/repo/data.json'),
                                   Path('/repo/results/out.json'), args)
        self.assertIn('--bootstrap-draws', command)
        self.assertEqual(command[command.index('--preflight-report') + 1],
                         '/private/gate.json')
        index = command.index('--bootstrap-draws')
        self.assertEqual(command[index + 1], '2500')
        self.assertEqual(command[command.index('--n-samples') + 1], '5')

    def test_sample_units_require_unique_video_ids(self):
        rows = [{'video_id': 'v1'}, {'video_id': 'v2'}, {'video_id': 'v3'}]
        self.assertEqual(validate_sample_units(rows, 2), 2)
        with self.assertRaisesRegex(ValueError, 'one analysis unit per video'):
            validate_sample_units([{'video_id': 'v1'}, {'video_id': 'v1'}])
        with self.assertRaisesRegex(ValueError, 'video analysis unit'):
            validate_sample_units([{'qid': 'q1'}])
        with self.assertRaisesRegex(ValueError, 'requested sample count'):
            validate_sample_units(rows, 4)

    def test_failed_child_does_not_expose_private_command(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'private.log'
            secret_path = str(Path(directory) / 'private-samples.json')
            with patch('run_necessity_choice_margin_test_and_publish.subprocess.run',
                       side_effect=subprocess.CalledProcessError(2, ['python', secret_path])):
                with self.assertRaises(SystemExit) as raised:
                    run_analysis(['python', secret_path], directory, log_path)
            self.assertNotIn(secret_path, str(raised.exception))
            self.assertIn('exit status 2', str(raised.exception))
            self.assertEqual(log_path.stat().st_mode & 0o777, 0o600)

    def test_publisher_stops_on_v12_report_before_child_or_git_checks(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix='.test_preflight_') as directory:
            directory = Path(directory)
            samples = directory / 'samples.json'
            samples.write_text('[]', encoding='utf-8')
            report = directory / 'preflight.json'
            report.write_text('{"schema_version":1,"purpose":'
                              '"claim_mask_structural_cross_gate_preflight",'
                              '"overall_status":"blocked",'
                              '"execution_authorization":"not_granted"}',
                              encoding='utf-8')
            result = subprocess.run(
                [sys.executable, 'scripts/run_necessity_choice_margin_test_and_publish.py',
                 '--samples-file', str(samples.relative_to(repo)),
                 '--preflight-report', str(report.relative_to(repo)),
                 '--output', 'results/test_preflight_never_publish.json'],
                cwd=repo, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn('execution preflight blocked', result.stderr)
            self.assertNotIn('worktree and index must be clean', result.stderr)


if __name__ == '__main__':
    unittest.main()
