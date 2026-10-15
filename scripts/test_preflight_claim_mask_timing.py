import copy
from fractions import Fraction
import unittest

from preflight_claim_mask_timing import preflight_cohort, sha256_frame_buffer


def record(video_id):
    hashes = [sha256_frame_buffer(bytes([i + len(video_id)]) * (64 * 64 * 3)) for i in range(4)]
    timing = sha256_frame_buffer(f"{video_id}:timing".encode())
    return {
        "video_id": video_id,
        "pts": list(range(7)),
        "time_base": [1, 2],
        "num_frames": 4,
        "evidence_intervals": [[0, 0]],
        "sample_indices": [0, 2, 4, 6],
        "sampled_pixel_sha256": hashes,
        "processor_pixel_sha256": hashes.copy(),
        "processor_fps": 1.0,
        "processor_frame_indices": [0, 1, 2, 3],
        "condition_timing_sha256": [timing, timing, timing],
    }


class ClaimMaskTimingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.records = [record("fixture-a"), record("fixture-b")]
        self.kwargs = {"expected_video_ids": ["fixture-a", "fixture-b"],
                       "expected_num_frames": 4,
                       "tolerance_seconds": Fraction(0)}

    def test_entire_synthetic_cohort_passes_without_authorizing_scoring(self):
        result = preflight_cohort(self.records, **self.kwargs)
        self.assertEqual(result["status"], "synthetic_timing_contract_passed")
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["validated_videos"], 2)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_one_member_failure_blocks_whole_denominator(self):
        bad = copy.deepcopy(self.records)
        bad[1]["processor_fps"] = 2.0
        result = preflight_cohort(bad, **self.kwargs)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("video_timing_invalid", result["failure_codes"])
        self.assertEqual(result["video_failures"]["fixture-b"], ["processor_fps_mismatch"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["validated_videos"], 1)
        self.assertEqual(result["score_calls"], 0)

    def test_pixel_order_mismatch_blocks_all(self):
        bad = copy.deepcopy(self.records)
        bad[0]["processor_pixel_sha256"][0] = sha256_frame_buffer(bytes([255]) * (64 * 64 * 3))
        result = preflight_cohort(bad, **self.kwargs)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("pixel_identity_mismatch", result["video_failures"]["fixture-a"])
        self.assertEqual(result["score_calls"], 0)

    def test_non_json_processor_index_container_fails_closed(self):
        malformed = copy.deepcopy(self.records)
        malformed[0]["processor_frame_indices"] = (0, 1, 2, 3)
        result = preflight_cohort(malformed, **self.kwargs)
        self.assertIn("processor_frame_order_mismatch",
                      result["video_failures"]["fixture-a"])
        self.assertEqual(result["status"], "blocked")

    def test_full_selected_control_timing_must_match(self):
        bad = copy.deepcopy(self.records)
        bad[0]["condition_timing_sha256"][2] = sha256_frame_buffer(b"control-timing-differs")
        result = preflight_cohort(bad, **self.kwargs)
        self.assertIn("condition_timing_binding_mismatch", result["video_failures"]["fixture-a"])
        self.assertEqual(result["score_calls"], 0)

    def test_nonuniform_sampled_pts_fails_and_cohort_is_not_shrunk(self):
        bad = copy.deepcopy(self.records)
        bad[1]["pts"] = [0, 1, 2, 3, 4, 6, 7]
        result = preflight_cohort(bad, **self.kwargs)
        self.assertIn("sampled_pts_not_uniform", result["video_failures"]["fixture-b"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["score_calls"], 0)

    def test_nonmonotonic_pts_and_missing_evidence_assignment_fail(self):
        bad_pts = copy.deepcopy(self.records)
        bad_pts[1]["pts"] = [0, 1, 2, 2, 4, 5, 6]
        result = preflight_cohort(bad_pts, **self.kwargs)
        self.assertIn("pts_map_invalid", result["video_failures"]["fixture-b"])

        no_evidence = copy.deepcopy(self.records)
        no_evidence[1]["evidence_intervals"] = [[100, 101]]
        result = preflight_cohort(no_evidence, **self.kwargs)
        self.assertIn("evidence_or_control_assignment_failed",
                      result["video_failures"]["fixture-b"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["score_calls"], 0)

    def test_unrepresentable_extreme_rate_fails_closed_without_crashing(self):
        extreme = copy.deepcopy(self.records)
        extreme[0]["time_base"] = [1, 10**400]
        result = preflight_cohort(extreme, **self.kwargs)
        self.assertIn("processor_fps_mismatch", result["video_failures"]["fixture-a"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["score_calls"], 0)

        tiny = copy.deepcopy(self.records)
        tiny[0]["time_base"] = [10**400, 1]
        tiny[0]["processor_fps"] = 0.0
        result = preflight_cohort(tiny, **self.kwargs)
        self.assertIn("processor_fps_mismatch", result["video_failures"]["fixture-a"])
        self.assertEqual(result["status"], "blocked")

    def test_mixed_frame_counts_cannot_pass_a_frozen_cohort(self):
        three = record("fixture-b")
        three.update(pts=list(range(5)), num_frames=3, sample_indices=[0, 2, 4],
                     sampled_pixel_sha256=three["sampled_pixel_sha256"][:3],
                     processor_pixel_sha256=three["processor_pixel_sha256"][:3])
        result = preflight_cohort([self.records[0], three], **self.kwargs)
        self.assertIn("cohort_sampling_count_mismatch",
                      result["video_failures"]["fixture-b"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["score_calls"], 0)

    def test_mismatched_huge_frame_count_is_rejected_before_assignment(self):
        huge = copy.deepcopy(self.records)
        huge[1]["num_frames"] = 10**8
        result = preflight_cohort(huge, **self.kwargs)
        self.assertIn("cohort_sampling_count_mismatch",
                      result["video_failures"]["fixture-b"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["score_calls"], 0)

    def test_missing_or_duplicate_cohort_member_blocks(self):
        for rows in (self.records[:1], [self.records[0], self.records[0]]):
            with self.subTest(count=len(rows)):
                result = preflight_cohort(rows, **self.kwargs)
                self.assertIn("cohort_membership_mismatch", result["failure_codes"])
                self.assertEqual(result["score_calls"], 0)

    def test_empty_or_singleton_analysis_cohort_is_invalid(self):
        for video_ids in ([], ["fixture-a"]):
            with self.subTest(video_ids=video_ids):
                result = preflight_cohort([], expected_video_ids=video_ids,
                                          expected_num_frames=4,
                                          tolerance_seconds=Fraction(0))
                self.assertEqual(result["status"], "blocked")
                self.assertIn("expected_cohort_invalid", result["failure_codes"])
                self.assertEqual(result["score_calls"], 0)


if __name__ == "__main__":
    unittest.main()
