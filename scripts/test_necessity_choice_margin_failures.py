import unittest
import torch

from necessity_choice_margin_test import failure_code


class FailureCodeTests(unittest.TestCase):
    def test_oom_is_generic(self):
        self.assertEqual(failure_code(torch.cuda.OutOfMemoryError('secret path /x')), 'cuda_oom')

    def test_io_and_validation_codes(self):
        self.assertEqual(failure_code(FileNotFoundError('/private/video.mp4')), 'media_io_error')
        self.assertEqual(failure_code(ValueError('question text')), 'input_validation_error')

    def test_unknown_is_runtime(self):
        self.assertEqual(failure_code(RuntimeError('/private/video.mp4')), 'runtime_error')
