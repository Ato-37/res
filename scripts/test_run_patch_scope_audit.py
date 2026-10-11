"""Usage: python -m unittest discover -s scripts -p 'test_run_patch_scope_audit.py'."""
from pathlib import Path
import unittest
from run_patch_scope_audit import commands


class RunnerTests(unittest.TestCase):
    def test_fixed_protocol_and_analysis_inputs(self):
        plan = commands(Path('samples with spaces.json'), Path('output'), 25)
        self.assertEqual(len(plan), 4)
        for cmd, scope in zip(plan[:3], ('all', 'prompt', 'answer')):
            self.assertEqual(cmd[cmd.index('--patch-scope') + 1], scope)
            self.assertEqual(cmd[cmd.index('--samples-file') + 1], 'samples with spaces.json')
            self.assertEqual(cmd[cmd.index('--n-samples') + 1], '25')
            self.assertIn('--self-check', cmd)
            self.assertIn('--strict-grounding', cmd)
            self.assertEqual(cmd[cmd.index('--output') + 1],
                             plan[3][plan[3].index('--' + scope) + 1])


if __name__ == '__main__':
    unittest.main()
