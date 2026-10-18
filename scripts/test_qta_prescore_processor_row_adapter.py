from __future__ import annotations

import hashlib
import unittest
from fractions import Fraction

from necessity_choice_margin_utils import uniform_sample_indices
from qta_prescore_processor_row_adapter import processor_binding_row


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class QtaPrescoreProcessorRowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.source_video_sha256 = sha("video")
        sample_indices = uniform_sample_indices(64, 32)
        sample_pts = [[index, 1] for index in range(32)]
        self.sampler_row = {
            "source_sha256": self.source_video_sha256,
            "decoded_frame_count": 64,
            "sampled_frame_count": 32,
            "sampled_indices": sample_indices,
            "sampled_pts": sample_pts,
        }
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
            "sampled_frame_indices": sample_indices,
            "sampled_pts_rational_seconds": [str(Fraction(*pair))
                                              for pair in sample_pts],
            "_processor_batch_snapshot": {"private": "must not copy"},
        }

    def test_projects_exact_join_fields_and_omits_private_tensor_snapshot(self):
        result = processor_binding_row(review_token="opaque-token-0",
                                       source_video_sha256=self.source_video_sha256,
                                       mapping=self.mapping,
                                       sampler_row=self.sampler_row)
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
                                  mapping={**self.mapping, "status": "blocked"},
                                  sampler_row=self.sampler_row)
        with self.assertRaisesRegex(ValueError, "processor_row_typed_batch_sha256_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "typed_batch_sha256": "bad"},
                                  sampler_row=self.sampler_row)

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
                                      mapping=mapping, sampler_row=self.sampler_row)

    def test_rejects_invalid_group_index_order_or_grid(self):
        with self.assertRaisesRegex(ValueError, "processor_row_evidence_groups_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "evidence_group_indices": [3, 1]},
                                  sampler_row=self.sampler_row)
        with self.assertRaisesRegex(ValueError, "processor_video_grid_invalid"):
            processor_binding_row(review_token="token", source_video_sha256=sha("video"),
                                  mapping={**self.mapping, "video_grid_thw": [0, 2, 2]},
                                  sampler_row=self.sampler_row)

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
                                      mapping={**self.mapping, "groups": groups},
                                      sampler_row=self.sampler_row)

    def test_requires_matching_source_and_fixed_32_frame_endpoint_sample(self):
        for sampler_row, expected_error in (
                ({**self.sampler_row, "source_sha256": sha("other-video")},
                 "processor_sampler_source_hash_mismatch"),
                ({**self.sampler_row, "sampled_frame_count": 31},
                 "processor_sampler_row_shape_invalid"),
                ({**self.sampler_row, "sampled_indices": [1, *self.sampler_row[
                    "sampled_indices"][1:]]},
                 "processor_sampler_indices_not_frozen_endpoint_grid")):
            with self.subTest(expected_error=expected_error), self.assertRaisesRegex(
                    ValueError, expected_error):
                processor_binding_row(
                    review_token="token", source_video_sha256=self.source_video_sha256,
                    mapping=self.mapping, sampler_row=sampler_row)

    def test_compares_sampler_pts_as_exact_rationals(self):
        mapping = dict(self.mapping)
        changed_pts = list(mapping["sampled_pts_rational_seconds"])
        changed_pts[10] = "10000000000000001/10000000000000000"
        mapping["sampled_pts_rational_seconds"] = changed_pts
        with self.assertRaisesRegex(ValueError, "processor_sampler_pts_mismatch"):
            processor_binding_row(
                review_token="token", source_video_sha256=self.source_video_sha256,
                mapping=mapping, sampler_row=self.sampler_row)

        # A malformed sampler pair cannot be coerced into an approximate float.
        sampler = {**self.sampler_row, "sampled_pts": [*self.sampler_row[
            "sampled_pts"]]}
        sampler["sampled_pts"][10] = [10, 0]
        with self.assertRaisesRegex(ValueError, "processor_sampler_pts_invalid"):
            processor_binding_row(
                review_token="token", source_video_sha256=self.source_video_sha256,
                mapping=self.mapping, sampler_row=sampler)


if __name__ == "__main__":
    unittest.main()
