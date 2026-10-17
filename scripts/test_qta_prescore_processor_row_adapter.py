from __future__ import annotations

import hashlib
import unittest

from qta_prescore_processor_row_adapter import processor_binding_row


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class QtaPrescoreProcessorRowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "status": "processor_batch_snapshot_and_caller_metadata_bound",
            "typed_batch_sha256": sha("typed-batch"),
            "sampling_metadata_sha256": sha("sampling"),
            "video_grid_thw": [5, 2, 2],
            "reviewed_intervals_rational_seconds": [["1/10", "2/5"]],
            "interval_boundary_guard_rational_seconds": "1/20",
            "evidence_group_indices": [1, 3],
            "control_group_indices": [0, 4],
            "_processor_batch_snapshot": {"private": "must not copy"},
        }

    def test_projects_exact_join_fields_and_omits_private_tensor_snapshot(self):
        result = processor_binding_row(review_token="opaque-token-0",
                                       source_video_sha256=sha("video"),
                                       mapping=self.mapping)
        self.assertEqual(set(result), {
            "review_token", "source_video_sha256", "typed_batch_sha256",
            "sampling_metadata_sha256", "video_grid_thw",
            "reviewed_intervals_rational_seconds",
            "interval_boundary_guard_rational_seconds",
            "evidence_group_indices", "control_group_indices",
        })
        self.assertEqual(result["reviewed_intervals_rational_seconds"],
                         [["1/10", "2/5"]])
        self.assertNotIn("_processor_batch_snapshot", result)

    def test_rejects_failed_mapping_or_malformed_digest(self):
        with self.assertRaisesRegex(ValueError, "processor_mapping_status_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "status": "blocked"})
        with self.assertRaisesRegex(ValueError, "processor_row_typed_batch_sha256_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "typed_batch_sha256": "bad"})

    def test_rejects_invalid_group_index_order_or_grid(self):
        with self.assertRaisesRegex(ValueError, "processor_row_evidence_groups_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "evidence_group_indices": [3, 1]})
        with self.assertRaisesRegex(ValueError, "processor_video_grid_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "video_grid_thw": [0, 2, 2]})


if __name__ == "__main__":
    unittest.main()
