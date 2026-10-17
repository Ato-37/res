"""CPU checks for the synthetic-only internal-ablation GPU smoke fixture."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_qwen_internal_attention_ablation import synthetic_records  # noqa: E402


class InternalAblationSyntheticFixtureTests(unittest.TestCase):
    def test_three_records_are_deterministic_and_have_complete_video_metadata(self):
        import numpy as np
        from PIL import Image

        first = synthetic_records(np, Image)
        second = synthetic_records(np, Image)
        self.assertEqual(len(first), 3)
        for row_index, (left, right) in enumerate(zip(first, second, strict=True)):
            self.assertEqual(len(left["frames"]), 32)
            self.assertEqual(left["presentation_times"], list(range(32)))
            self.assertEqual(left["sampled_indices"], list(range(32)))
            self.assertEqual(left["intervals"], [[5, 8], [24, 27]])
            self.assertEqual(left["sample_fps"], 1)
            self.assertEqual(left["qa_item"]["gold"], row_index % 3)
            self.assertEqual(len(left["qa_item"]["choices"]), 5)
            for image_left, image_right in zip(left["frames"], right["frames"], strict=True):
                self.assertEqual(image_left.size, (224, 224))
                self.assertTrue(np.array_equal(np.asarray(image_left), np.asarray(image_right)))


if __name__ == "__main__":
    unittest.main()
