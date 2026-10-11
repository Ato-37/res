"""CPU-only privacy checks for the layer-randomization publisher."""

import unittest

from run_head_faithfulness_layer_randomization_and_publish import reject_private_fields


class LayerRandomizationPublisherTests(unittest.TestCase):
    def test_anonymous_layer_diagnostic_is_publishable(self):
        reject_private_fields({
            'records': [{'sample_key': 'sample_000', 'cumulative_conditions': []}],
            'layer_reinitialization_counts': [{'layer': 27, 'linear_modules': 7}],
        })

    def test_video_and_text_fields_are_rejected(self):
        for payload in ({'video': '/private/video.mp4'},
                        {'question': 'raw text'},
                        {'nested': {'video_id': 'private-id'}}):
            with self.subTest(payload=payload), self.assertRaises(SystemExit):
                reject_private_fields(payload)


if __name__ == '__main__':
    unittest.main()
