"""CPU contract tests for binding interval maps to processor batch payloads."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nextgqa_atomic_batch_freeze import batch_sha256  # noqa: E402
from annotation_clock_calibration import map_annotation_intervals_to_pts  # noqa: E402
from qwen_processor_batch_review_mapping import (  # noqa: E402
    map_review_interval_from_processor_batch,
)


class ProcessorBatchReviewMappingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.batch = {
            "input_ids": torch.tensor([[10] + [99] * 16 + [10]], dtype=torch.long),
            "attention_mask": torch.ones((1, 18), dtype=torch.long),
            "pixel_values_videos": torch.zeros((16, 3), dtype=torch.float32),
            "video_grid_thw": torch.tensor([[4, 2, 2]], dtype=torch.long),
            "second_per_grid_ts": torch.tensor([1.0], dtype=torch.float32),
        }
        self.kwargs = {
            "batch": self.batch,
            "torch_module": torch,
            "video_token_id": 99,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_frame_indices": [0, 1, 2, 3],
            "sampled_pts_seconds": [0.0, 1.0, 2.0, 3.0],
            "source_frame_count": 4,
            "video_duration_seconds": 4.0,
            "sample_fps": 1.0,
            "processor_revision": "test-revision",
            "processor_fingerprint_sha256": "a" * 64,
            "reviewed_interval_seconds": [0.1, 0.4],
            "interval_boundary_guard_seconds": 0.0,
        }

    def test_binds_mapping_to_exact_complete_cpu_processor_batch(self):
        result = map_review_interval_from_processor_batch(**self.kwargs)
        self.assertEqual(result["status"],
                         "processor_batch_snapshot_and_caller_metadata_bound")
        self.assertEqual(result["evidence_group_indices"], [0])
        self.assertEqual(result["control_group_indices"], [1])
        self.assertEqual(len(result["typed_batch_sha256"]), 64)
        self.assertEqual(len(result["sampling_metadata_sha256"]), 64)
        self.assertEqual(result["source_authentication_status"],
                         "not_performed_by_this_helper")
        self.assertEqual(result["max_pts_cadence_deviation_from_nominal_seconds"], 0)
        expected_hash = batch_sha256(self.batch)
        self.assertEqual(result["typed_batch_sha256"], expected_hash)
        self.batch["input_ids"][0, 0] = 11
        self.assertEqual(result["_processor_batch_snapshot"]["input_ids"][0, 0].item(), 10)
        self.assertEqual(batch_sha256(result["_processor_batch_snapshot"]), expected_hash)

    def test_one_shot_review_interval_is_bound_once_in_mapping_and_digest(self):
        expected = map_review_interval_from_processor_batch(**self.kwargs)
        actual = map_review_interval_from_processor_batch(**{
            **self.kwargs, "reviewed_interval_seconds": iter([0.1, 0.4])})
        self.assertEqual(actual["evidence_group_indices"], expected["evidence_group_indices"])
        self.assertEqual(actual["sampling_metadata_sha256"],
                         expected["sampling_metadata_sha256"])
        changed = map_review_interval_from_processor_batch(**{
            **self.kwargs, "processor_revision": "different-revision"})
        self.assertNotEqual(actual["sampling_metadata_sha256"],
                            changed["sampling_metadata_sha256"])
        changed_interval = map_review_interval_from_processor_batch(**{
            **self.kwargs, "reviewed_interval_seconds": [0.6, 0.8]})
        self.assertNotEqual(actual["sampling_metadata_sha256"],
                            changed_interval["sampling_metadata_sha256"])
        changed_pts = map_review_interval_from_processor_batch(**{
            **self.kwargs, "sampled_pts_seconds": [0.0, 1.05, 2.0, 3.0]})
        self.assertNotEqual(actual["sampling_metadata_sha256"],
                            changed_pts["sampling_metadata_sha256"])
        changed_processor = map_review_interval_from_processor_batch(**{
            **self.kwargs, "processor_fingerprint_sha256": "b" * 64})
        self.assertNotEqual(actual["sampling_metadata_sha256"],
                            changed_processor["sampling_metadata_sha256"])

    def test_binds_every_disjoint_review_occurrence_and_union_control(self):
        result = map_review_interval_from_processor_batch(**{
            **self.kwargs,
            "reviewed_interval_seconds": None,
            "reviewed_intervals_seconds": [[0.1, 0.4], [2.6, 2.9]],
        })
        self.assertEqual(result["reviewed_interval_count"], 2)
        self.assertEqual(result["reviewed_intervals_seconds"],
                         [[0.1, 0.4], [2.6, 2.9]])
        self.assertEqual(result["evidence_group_indices"], [0, 3])
        self.assertEqual(result["reviewed_occurrence_group_indices"], [[0], [3]])
        self.assertEqual(result["control_group_indices"], [1, 2])
        self.assertEqual(result["token_budget_per_arm"], 8)
        self.assertEqual(len(result["sampling_metadata_sha256"]), 64)
        changed_one = map_review_interval_from_processor_batch(**{
            **self.kwargs,
            "reviewed_interval_seconds": None,
            "reviewed_intervals_seconds": [[0.2, 0.4], [2.6, 2.9]],
        })
        reversed_order = map_review_interval_from_processor_batch(**{
            **self.kwargs,
            "reviewed_interval_seconds": None,
            "reviewed_intervals_seconds": [[2.6, 2.9], [0.1, 0.4]],
        })
        self.assertNotEqual(result["sampling_metadata_sha256"],
                            changed_one["sampling_metadata_sha256"])
        self.assertNotEqual(result["sampling_metadata_sha256"],
                            reversed_order["sampling_metadata_sha256"])

    def test_rejects_missing_or_ambiguous_interval_arguments(self):
        with self.assertRaisesRegex(ValueError, "reviewed_intervals_argument_invalid"):
            map_review_interval_from_processor_batch(**{
                **self.kwargs, "reviewed_interval_seconds": None})
        with self.assertRaisesRegex(ValueError, "reviewed_intervals_argument_invalid"):
            map_review_interval_from_processor_batch(**{
                **self.kwargs,
                "reviewed_intervals_seconds": [[0.1, 0.4]],
            })

    def test_requires_and_enforces_calibrated_boundary_guard(self):
        with self.assertRaisesRegex(ValueError, "clock_guard_required"):
            map_review_interval_from_processor_batch(**{
                key: value for key, value in self.kwargs.items()
                if key != "interval_boundary_guard_seconds"})
        stable = map_review_interval_from_processor_batch(**{
            **self.kwargs, "interval_boundary_guard_seconds": 0.05})
        self.assertEqual(stable["evidence_group_indices"], [0])
        self.assertEqual(stable["interval_boundary_guard_seconds"], 0.05)
        self.assertEqual(stable["reviewed_intervals_rational_seconds"], [["1/10", "2/5"]])
        with self.assertRaisesRegex(ValueError,
                                    "calibration_guard_changes_group_assignment"):
            map_review_interval_from_processor_batch(**{
                **self.kwargs, "interval_boundary_guard_seconds": 0.2})

    def test_exact_interval_converter_output_flows_directly_to_batch_mapper(self):
        mapped = map_annotation_intervals_to_pts([[0.1, 0.4]], "1", "0")
        result = map_review_interval_from_processor_batch(**{
            **self.kwargs,
            "reviewed_interval_seconds": mapped[0],
            "interval_boundary_guard_seconds": "0",
        })
        self.assertEqual(result["reviewed_intervals_rational_seconds"],
                         [["1/10", "2/5"]])

    def test_rejects_padding_pixel_grid_and_sampling_disagreement(self):
        changes = [
            ({**self.batch, "attention_mask": self.torch.cat((
                self.torch.tensor([[1] * 17 + [0]]),), dim=0)},
             self.kwargs, "single_unpadded"),
            ({**self.batch, "pixel_values_videos": self.torch.zeros((15, 3))},
             self.kwargs, "pixel_grid_geometry_mismatch"),
            (self.batch, {**self.kwargs, "sample_fps": 2.0},
             "temporal_spacing_fps_mismatch"),
            (self.batch, {**self.kwargs, "sampled_frame_indices": [0, 1, 1, 3]},
             "sampled_frame_metadata_grid_mismatch"),
            ({**self.batch, "pixel_values_videos": self.torch.zeros((16, 3),
                                                                    dtype=self.torch.int64)},
             self.kwargs, "pixels_shape_or_dtype"),
            ({**self.batch, "pixel_values_videos": self.torch.zeros((16, 0))},
             self.kwargs, "pixels_shape_or_dtype"),
            ({**self.batch, "second_per_grid_ts": self.torch.tensor([1],
                                                                      dtype=self.torch.int64)},
             self.kwargs, "temporal_spacing_invalid"),
            (self.batch, {**self.kwargs, "sample_fps": "1"}, "sample_fps_invalid"),
        ]
        for batch, kwargs, expected in changes:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                map_review_interval_from_processor_batch(**{**kwargs, "batch": batch})


if __name__ == "__main__":
    unittest.main()
