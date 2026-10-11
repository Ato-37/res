"""CPU-only publication privacy tests."""

import unittest

from run_head_faithfulness_and_publish import (
    reject_sensitive_keys,
    validate_result_completeness,
)


class PublicationPrivacyTests(unittest.TestCase):
    def test_derived_anonymous_artifact_is_allowed(self):
        reject_sensitive_keys({
            'records': [{'sample_key': 'sample_000', 'selective_margin_drop': 0.4}],
            'failures': [{'sample_key': 'sample_001', 'error_type': 'RuntimeError'}],
        })

    def test_raw_question_and_path_fields_are_rejected(self):
        for payload in (
            {'records': [{'question': 'private'}]},
            {'records': [{'video_path': '/private/video.mp4'}]},
            {'records': [{'metadata': {'qid': '123'}}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(SystemExit):
                reject_sensitive_keys(payload)

    def test_failure_only_smoke_result_is_complete_and_publishable(self):
        validate_result_completeness({
            'n_requested': 3, 'n_effective': 0, 'n_failures': 3,
        }, 3)

    def test_incomplete_failure_accounting_is_rejected(self):
        with self.assertRaises(SystemExit):
            validate_result_completeness({
                'n_requested': 3, 'n_effective': 0, 'n_failures': 2,
            }, 3)


if __name__ == '__main__':
    unittest.main()
