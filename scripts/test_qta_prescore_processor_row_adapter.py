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
            "processor_uniform_grid_assignment_check": (
                "passed_actual_pts_and_scalar_fps_group_sets_equal"),
            "evidence_group_indices": [1, 3],
            "control_group_indices": [0, 4],
            "groups": [{"token_positions": [i]} for i in range(5)],
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
            "processor_uniform_grid_assignment_check",
            "evidence_group_indices", "control_group_indices",
            "temporal_group_token_positions",
        })
        self.assertEqual(result["reviewed_intervals_rational_seconds"],
                         [["1/10", "2/5"]])
        self.assertNotIn("_processor_batch_snapshot", result)
        self.assertEqual(result["temporal_group_token_positions"],
                         [[0], [1], [2], [3], [4]])

    def test_rejects_failed_mapping_or_malformed_digest(self):
        with self.assertRaisesRegex(ValueError, "processor_mapping_status_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "status": "blocked"})
        with self.assertRaisesRegex(ValueError, "processor_row_typed_batch_sha256_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "typed_batch_sha256": "bad"})

    def test_requires_explicit_actual_pts_fps_group_equality_pass(self):
        for value in (None, "not_requested", "actual_pts_and_fps_group_sets_mismatch"):
            mapping = dict(self.mapping)
            if value is None:
                mapping.pop("processor_uniform_grid_assignment_check")
            else:
                mapping["processor_uniform_grid_assignment_check"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "processor_uniform_grid_assignment_check_invalid"):
                processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                      mapping=mapping)

    def test_rejects_invalid_group_index_order_or_grid(self):
        with self.assertRaisesRegex(ValueError, "processor_row_evidence_groups_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "evidence_group_indices": [3, 1]})
        with self.assertRaisesRegex(ValueError, "processor_video_grid_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "video_grid_thw": [0, 2, 2]})

    def test_rejects_noncontiguous_or_unequal_temporal_token_groups(self):
        for groups in (
                [{"token_positions": [0]}, {"token_positions": [2]},
                 {"token_positions": [3]}, {"token_positions": [4]},
                 {"token_positions": [5]}],
                [{"token_positions": [0, 1]}, {"token_positions": [2]},
                 {"token_positions": [3, 4]}, {"token_positions": [5]},
                 {"token_positions": [6, 7]}]):
            with self.subTest(groups=groups), self.assertRaisesRegex(
                    ValueError, "processor_row_temporal_groups_geometry_invalid"):
                processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                      mapping={**self.mapping, "groups": groups})


if __name__ == "__main__":
    unittest.main()
