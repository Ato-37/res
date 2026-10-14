"""CPU-only command-construction checks for the choice-margin publisher."""
from argparse import Namespace
from pathlib import Path
import unittest

from run_necessity_choice_margin_test_and_publish import analysis_command


class AnalysisCommandTests(unittest.TestCase):
    def test_bootstrap_draw_count_is_forwarded(self):
        args = Namespace(quant='4bit', seed=17, bootstrap_draws=2500,
                         num_frames=16, mask_fill='black', blur_radius=12.0,
                         n_samples=5)
        command = analysis_command(Path('/repo'), Path('/repo/data.json'),
                                   Path('/repo/results/out.json'), args)
        self.assertIn('--bootstrap-draws', command)
        index = command.index('--bootstrap-draws')
        self.assertEqual(command[index + 1], '2500')
        self.assertEqual(command[command.index('--n-samples') + 1], '5')


if __name__ == '__main__':
    unittest.main()
