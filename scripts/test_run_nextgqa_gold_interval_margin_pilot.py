import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from run_nextgqa_gold_interval_margin_pilot import (
    control_minus_evidence_margin,
    gold_vs_best_foil_margin,
    helper_source_hashes,
    prepare_conditions,
    validate_cohort_cardinality,
    validate_cohort_lock_payload,
    validate_preflight_cardinality,
)


class GoldIntervalAnswerMarginTests(unittest.TestCase):
    def test_grid_preparation_discards_batches_and_score_can_retain_one_video(self):
        utility = ModuleType("qwen_vl_utils")

        def process_vision_info(messages, return_video_metadata=True):
            return None, [(torch.zeros((1, 3, 2, 2)), {
                "frames_indices": [0], "fps": 1.0,
            })]

        utility.process_vision_info = process_vision_info

        class Processor:
            video_processor = SimpleNamespace(temporal_patch_size=1)

            def apply_chat_template(self, *args, **kwargs):
                return "synthetic prompt"

            def __call__(self, **kwargs):
                return {"video_grid_thw": torch.tensor([[1, 1, 1]]),
                        "second_per_grid_ts": torch.tensor([2.0]),
                        "input_ids": torch.tensor([[1]])}

        item = {
            "frames": [Image.new("RGB", (2, 2), "black")],
            "evidence_positions": [], "control_positions": [],
            "choices": ["a", "b", "c", "d", "e"],
            "question": "synthetic", "decode": {"effective_sample_fps": [1, 1]},
        }
        with patch.dict("sys.modules", {"qwen_vl_utils": utility}):
            grid_only = prepare_conditions([item], Processor(), torch, 1,
                                           retain_batches=False)
            self.assertIsNone(grid_only[0]["batches"])
            self.assertEqual(grid_only[0]["grid"], [1, 1, 1])
            retained = prepare_conditions([item], Processor(), torch, 1,
                                          retain_batches=True)
        self.assertEqual(set(retained[0]["batches"]), {
            "full", "black_gold_interval_mask", "black_equal_count_control_mask",
            "blur_gold_interval_mask", "blur_equal_count_control_mask"})

    def test_hash_only_cohort_lock_requires_unique_valid_exact_assignments(self):
        h1, h2 = "a" * 64, "b" * 64
        payload = {"schema_version": 1, "assignments": [
            {"media_sha256": h1, "qid_sha256": h2},
            {"media_sha256": h2, "qid_sha256": h1},
        ]}
        self.assertEqual(validate_cohort_lock_payload(payload, 2), payload["assignments"])
        with self.assertRaisesRegex(ValueError, "count_must_equal"):
            validate_cohort_lock_payload(payload, 3)
        with self.assertRaisesRegex(ValueError, "lock_invalid"):
            validate_cohort_lock_payload({"schema_version": 1, "assignments": [
                {"media_sha256": h1, "qid_sha256": h2},
                {"media_sha256": h1, "qid_sha256": h2},
            ]}, 2)
        with self.assertRaisesRegex(ValueError, "lock_invalid"):
            validate_cohort_lock_payload({"schema_version": 1, "assignments": [
                {"media_sha256": "private-id", "qid_sha256": h2},
            ]}, 1)
        with self.assertRaisesRegex(ValueError, "lock_invalid"):
            validate_cohort_lock_payload({"schema_version": True, "assignments": [
                {"media_sha256": h1, "qid_sha256": h2},
            ]}, 1)

    def test_helper_hashes_bind_the_imported_protocol_utilities(self):
        repo = Path(__file__).resolve().parents[1]
        hashes = helper_source_hashes(repo)
        self.assertEqual(set(hashes), {
            "necessity_choice_margin_utils.py", "presentation_time_utils.py"})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_cohort_must_match_requested_size_for_all_supported_n(self):
        for requested in (1, 3, 10):
            with self.subTest(requested=requested):
                validate_cohort_cardinality(requested, requested)
                with self.assertRaisesRegex(ValueError, f"requested_{requested}_eligible_samples"):
                    validate_cohort_cardinality(requested, requested - 1)

    def test_preflight_cannot_bind_an_undersized_cohort(self):
        gate = {"requested_videos": 3, "effective_videos": 2}
        with self.assertRaisesRegex(ValueError, "requested_3_eligible_samples"):
            validate_preflight_cardinality(gate, requested=3, selected_count=2)
        with self.assertRaisesRegex(ValueError, "preflight_requested_or_effective_count_mismatch"):
            validate_preflight_cardinality(
                {"requested_videos": 1, "effective_videos": 3}, requested=3, selected_count=3)

    def test_gold_vs_best_foil_margin_uses_fixed_gold_index(self):
        scores = [0.5, 0.1, 0.2, -0.1, 0.3]
        self.assertAlmostEqual(gold_vs_best_foil_margin(scores, 0), 0.2)
        self.assertAlmostEqual(gold_vs_best_foil_margin(scores, 3), -0.6)

    def test_positive_control_minus_evidence_means_evidence_mask_hurts_more(self):
        self.assertAlmostEqual(control_minus_evidence_margin(0.25, -0.5), 0.75)
        self.assertAlmostEqual(control_minus_evidence_margin(-0.2, 0.4), -0.6)

    def test_rejects_malformed_and_nonfinite_scores(self):
        for scores, gold in (([1.0] * 4, 0), ([1.0] * 5, -1),
                             ([1.0, 1.0, float("nan"), 1.0, 1.0], 0)):
            with self.subTest(scores=scores, gold=gold):
                with self.assertRaises(ValueError):
                    gold_vs_best_foil_margin(scores, gold)
        with self.assertRaisesRegex(ValueError, "intervention_margin_nonfinite"):
            control_minus_evidence_margin(float("inf"), 0.0)


if __name__ == "__main__":
    unittest.main()
