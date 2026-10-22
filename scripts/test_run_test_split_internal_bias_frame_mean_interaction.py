import unittest
from pathlib import Path

from run_test_split_internal_bias_frame_mean_interaction import interaction, verify_locked_assignment
from analyze_test_split_internal_bias_frame_mean_interaction import summarize


class InteractionTests(unittest.TestCase):
    def test_interaction_reduces_to_difference_in_differences(self):
        result = interaction(2.0, 1.0, 2.5, 1.25)
        self.assertEqual(result["baseline_original_minus_frame_mean_nats_per_token"], 1.0)
        self.assertEqual(result["biased_original_minus_frame_mean_nats_per_token"], 1.25)
        self.assertEqual(result["bias_by_frame_mean_interaction_nats_per_token"], 0.25)

    def test_equal_bias_shift_cancels(self):
        result = interaction(-0.4, -0.9, -0.2, -0.7)
        self.assertAlmostEqual(result["bias_by_frame_mean_interaction_nats_per_token"], 0.0)

    def test_nonfinite_and_boolean_scores_rejected(self):
        for values in ((0.0, 0.0, float("nan"), 0.0), (0.0, 0.0, True, 0.0)):
            with self.assertRaises(ValueError):
                interaction(*values)

    def test_locked_row_assignment_is_bound(self):
        lock = {"status": "frozen_before_claim_generation_and_scoring", "dataset_split": "test",
            "requested_videos": 3, "effective_videos": 3,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [{"media_sha256": "b" * 64, "qid_sha256": "c" * 64}, {}, {}]}
        row = {"video_sha256": "b" * 64, "dataset_question_id": "qid"}
        import hashlib
        lock["assignments"][0]["qid_sha256"] = hashlib.sha256(b"qid").hexdigest()
        verify_locked_assignment(lock, "a" * 64, 0, row, "a" * 64)
        row["video_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            verify_locked_assignment(lock, "a" * 64, 0, row, "a" * 64)

    def test_completed_local_receipts_reconcile(self):
        result = summarize(Path(__file__).resolve().parents[1])
        self.assertEqual(result["effective_rows"], 3)
        self.assertEqual(result["requested_new_forwards"], 6)
        self.assertEqual(result["completed_new_forwards"], 6)
        self.assertEqual(result["positive_rows"], 3)
        self.assertAlmostEqual(result["mean_interaction"], 0.039273870106, places=10)


if __name__ == "__main__":
    unittest.main()
