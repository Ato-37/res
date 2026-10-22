from __future__ import annotations

import unittest

from run_claim_text_qa_interval_key_bias import (
    BRANCHES, align_interval_and_control_groups, compute_text_dependence,
    map_groups_to_visual_keys, masked_group_frame_indices,
    select_disjoint_sham_groups, validate_packet, validate_smoke_receipt,
)


class QATemporalGroupDesignTests(unittest.TestCase):
    @staticmethod
    def valid_smoke():
        means = {name: -1.0 for name in BRANCHES}
        target = [-1.0, -1.0]
        calls = {}
        for branch in BRANCHES:
            biased = not branch.startswith("baseline_")
            calls[branch] = {
                "target_token_count": 2, "query_token_count": 2,
                "biased_key_count": 8 if biased else 0,
                "surviving_bias_key_count": 8 if biased else 0,
                "ablated_key_count": 0,
                "hook_calls_by_layer": {str(layer): int(biased) for layer in range(5, 17)},
            }
        return {
            "status": "completed_exploratory_claim_text_qa_interval_key_bias_smoke",
            "mode": "smoke", "model_revision": "revision",
            "model_weight_sha256": {"shard": "c"},
            "input_sha256": {"input": "a"}, "source_sha256": {"source": "b"},
            "requested_records": 1, "effective_records": 1,
            "requested_forwards": 6, "attempted_forwards": 6,
            "completed_forwards": 6,
            "records": [{
                "row_token_sha256": "row", "source_video_sha256": "video",
                "saved_claim_sha256": "claim", "target_token_ids_sha256": "tokens",
                "branch_mean_logprob_nats_per_token": means,
                "branch_token_logprobs_nats": {name: target[:] for name in BRANCHES},
                "calls": calls, "forward_count": {"attempted": 6, "completed": 6},
                "target_token_count": 2, "qa_bias_key_count": 8,
                "primary": compute_text_dependence(means),
            }],
            "direct_faithfulness": False, "faithfulness_result": False,
            "human_review": {"review_actually_performed": False},
        }

    def test_smoke_receipt_validator_recomputes_all_six_branches(self):
        receipt = self.valid_smoke()
        validate_smoke_receipt(receipt, input_hashes={"input": "a"},
            source_hashes={"source": "b"}, model_revision="revision",
            weight_sha256={"shard": "c"},
            expected_row={"row_token_sha256": "row", "source_video_sha256": "video",
                "claim_hash": "claim", "target_hash": "tokens"})
        receipt["attempted_forwards"] = 5
        with self.assertRaisesRegex(ValueError, "successful_matching_smoke_receipt_required"):
            validate_smoke_receipt(receipt, input_hashes={"input": "a"},
                source_hashes={"source": "b"}, model_revision="revision",
                weight_sha256={"shard": "c"},
                expected_row={"row_token_sha256": "row", "source_video_sha256": "video",
                    "claim_hash": "claim", "target_hash": "tokens"})

    def test_smoke_receipt_validator_rejects_corrupt_arithmetic_and_hooks(self):
        receipt = self.valid_smoke()
        receipt["records"][0]["primary"]["delta_qa_bias_minus_baseline_nats"] = 4.0
        with self.assertRaisesRegex(ValueError, "smoke_receipt_primary_arithmetic_invalid"):
            validate_smoke_receipt(receipt, input_hashes={"input": "a"},
                source_hashes={"source": "b"}, model_revision="revision",
                weight_sha256={"shard": "c"},
                expected_row={"row_token_sha256": "row", "source_video_sha256": "video",
                    "claim_hash": "claim", "target_hash": "tokens"})
        receipt = self.valid_smoke()
        receipt["records"][0]["calls"]["qa_bias_qa_pixel_mask"]["hook_calls_by_layer"]["5"] = 0
        with self.assertRaisesRegex(ValueError, "smoke_receipt_hook_observation_invalid"):
            validate_smoke_receipt(receipt, input_hashes={"input": "a"},
                source_hashes={"source": "b"}, model_revision="revision",
                weight_sha256={"shard": "c"},
                expected_row={"row_token_sha256": "row", "source_video_sha256": "video",
                    "claim_hash": "claim", "target_hash": "tokens"})

    def test_outward_resolution_and_nearest_control(self):
        pts = list(map(float, range(32)))
        row_a = align_interval_and_control_groups(
            sampled_pts=pts, intervals=[[19, 20]])
        self.assertEqual(row_a["sampled_qa_frame_indices"], [19, 20])
        self.assertEqual(row_a["evidence_group_indices"], [9, 10])
        self.assertEqual(row_a["control_group_indices"], [7, 8])
        self.assertEqual(row_a["evidence_frame_window"], [18, 22])
        self.assertEqual(row_a["control_frame_window"], [14, 18])
        self.assertEqual(row_a["expansion_frames"], 2)

        row_b = align_interval_and_control_groups(
            sampled_pts=pts, intervals=[[2, 4]])
        self.assertEqual(row_b["sampled_qa_frame_indices"], [2, 3, 4])
        self.assertEqual(row_b["evidence_group_indices"], [1, 2])
        self.assertEqual(row_b["control_group_indices"], [3, 4])
        self.assertEqual(row_b["evidence_frame_window"], [2, 6])
        self.assertEqual(row_b["control_frame_window"], [6, 10])
        self.assertEqual(row_b["expansion_frames"], 1)

        row_16 = align_interval_and_control_groups(
            sampled_pts=list(map(float, range(16))), intervals=[[4, 4.1]], frame_count=16)
        self.assertEqual(row_16["evidence_group_indices"], [2])
        self.assertEqual(row_16["control_group_indices"], [1])
        self.assertEqual(row_16["evidence_frame_window"], [4, 6])

    def test_temporal_groups_map_to_equal_disjoint_visual_key_sets(self):
        video_token = 99
        ids = [1, 2] + [video_token] * (16 * 4) + [3]
        result = map_groups_to_visual_keys(input_ids=ids, video_token_id=video_token,
            video_grid_thw=(16, 4, 4), spatial_merge_size=2,
            temporal_patch_size=2, frame_count=32,
            evidence_group_indices=(9, 10), control_group_indices=(7, 8))
        self.assertEqual(result["temporal_group_count"], 16)
        self.assertEqual(result["tokens_per_temporal_group"], 4)
        self.assertEqual(len(result["evidence_key_positions"]), 8)
        self.assertEqual(len(result["control_key_positions"]), 8)
        self.assertFalse(set(result["evidence_key_positions"]) & set(result["control_key_positions"]))
        sham = select_disjoint_sham_groups(group_count=16, group_budget=2,
            evidence_groups=(9, 10), control_groups=(7, 8), row_token="a" * 64)
        self.assertEqual(len(sham), 2)
        self.assertEqual(sham, select_disjoint_sham_groups(group_count=16,
            group_budget=2, evidence_groups=(9, 10), control_groups=(7, 8),
            row_token="a" * 64))
        self.assertFalse(set(sham) & {7, 8, 9, 10})
        sham_keys = tuple(sorted(p for group in sham for p in result["temporal_groups"][group]))
        self.assertEqual(len(sham_keys), len(result["evidence_key_positions"]))
        self.assertFalse(set(sham_keys) & set(result["evidence_key_positions"]))
        self.assertFalse(set(sham_keys) & set(result["control_key_positions"]))

    def test_sham_group_set_uses_all_remaining_positions_when_no_window_fits(self):
        sham = select_disjoint_sham_groups(group_count=8, group_budget=4,
            evidence_groups=(0, 1), control_groups=(2, 3), row_token="b" * 64)
        self.assertEqual(sham, (4, 5, 6, 7))
        with self.assertRaisesRegex(ValueError, "disjoint_sham_group_set_unavailable"):
            select_disjoint_sham_groups(group_count=7, group_budget=4,
                evidence_groups=(0, 1), control_groups=(2, 3), row_token="b" * 64)

    def test_pixel_mask_frame_indices_follow_whole_groups(self):
        self.assertEqual(masked_group_frame_indices((1, 2), temporal_patch_size=2,
                                                    frame_count=32), (2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "masked_group_indices_invalid"):
            masked_group_frame_indices((2, 1), temporal_patch_size=2, frame_count=32)

    def test_six_branch_arithmetic_and_descriptive_deltas(self):
        scores = dict(zip(BRANCHES, (-2.0, -1.0, -2.1, -0.8, -1.9, -1.1)))
        result = compute_text_dependence(scores)
        self.assertAlmostEqual(result["f_baseline_nats"], 1.0)
        self.assertAlmostEqual(result["f_qa_bias_nats"], 1.3)
        self.assertAlmostEqual(result["f_sham_bias_nats"], 0.8)
        self.assertAlmostEqual(result["delta_qa_bias_minus_baseline_nats"], 0.3)
        self.assertAlmostEqual(result["specificity_qa_minus_sham_bias_nats"], 0.5)
        with self.assertRaisesRegex(ValueError, "branch_score_set_invalid"):
            compute_text_dependence({name: float("nan") for name in BRANCHES})

    def test_invalid_intervals_and_unavailable_controls_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "interval_group_mapping_inputs_invalid"):
            align_interval_and_control_groups(sampled_pts=list(map(float, range(32))),
                                              intervals=[])
        with self.assertRaisesRegex(ValueError, "interval_group_mapping_inputs_invalid"):
            align_interval_and_control_groups(sampled_pts=list(map(float, range(32))),
                                              intervals=[[2, 3], [19, 20]])
        with self.assertRaisesRegex(ValueError, "equal_count_group_control_unavailable"):
            align_interval_and_control_groups(sampled_pts=list(map(float, range(32))),
                intervals=[[0, 31]])

    def test_frozen_packet_frame_count_and_review_gate_are_explicit(self):
        # Contract test intentionally checks only public schema values, not private text.
        packet = {"status": "passed", "dataset_split": "test",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "generation_prompt_version": "v2_short", "frames_per_video": 32,
            "claim_review_completed": False, "requested_videos": 3,
            "effective_videos": 3, "fully_structured_claim_records": 3,
            "records": []}
        with self.assertRaisesRegex(ValueError, "frozen_claim_packet_record_count_invalid"):
            validate_packet(packet)

    def test_val_packet_requires_explicit_supported_split(self):
        packet = {"status": "passed", "dataset_split": "val",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "generation_prompt_version": "v2_short", "frames_per_video": 16,
            "claim_review_completed": False, "requested_videos": 3,
            "effective_videos": 3, "fully_structured_claim_records": 3,
            "records": []}
        with self.assertRaisesRegex(ValueError, "frozen_claim_packet_record_count_invalid"):
            validate_packet(packet, expected_split="val")
        with self.assertRaisesRegex(ValueError, "frozen_claim_packet_contract_invalid"):
            validate_packet(packet, expected_split="train")
        with self.assertRaisesRegex(ValueError, "frozen_claim_packet_contract_invalid"):
            validate_packet(packet, expected_split="test")

    def test_partial_generation_failures_are_counted_without_row_replacement(self):
        def row(video_hash, parsed):
            return {"generated_parsed": parsed, "generation_prompt_version": "v2_short",
                "dataset_question_id": "qid", "sampled_frame_relative_pts_seconds": list(range(16)),
                "gold_interval_seconds": [[1, 2]], "question": "question",
                "choices": ["a", "b", "c", "d", "e"],
                "local_video_path": "/private/video.mp4", "video_sha256": video_hash}
        valid = row("a" * 64, {"valid": True, "claim": "A person walks.",
                                "claim_text_quality_failure": None})
        failed = row("b" * 64, {"valid": False, "claim": None,
                                 "claim_text_quality_failure": "claim_missing"})
        multiple_interval = row("c" * 64, {"valid": True, "claim": "A person sits.",
                                            "claim_text_quality_failure": None})
        multiple_interval["gold_interval_seconds"] = [[1, 2], [6, 7]]
        packet = {"status": "partial", "dataset_split": "val",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "generation_prompt_version": "v2_short", "frames_per_video": 16,
            "claim_review_completed": False, "requested_videos": 3,
            "effective_videos": 3, "fully_structured_claim_records": 2,
            "records": [valid, failed, multiple_interval]}
        with self.assertRaisesRegex(ValueError, "frozen_claim_packet_contract_invalid"):
            validate_packet(packet, expected_split="val")
        selected = validate_packet(packet, expected_split="val",
                                   allow_ineligible_rows=True)
        self.assertEqual(selected, [valid])
        failed["generated_parsed"]["claim_text_quality_failure"] = ""
        with self.assertRaisesRegex(ValueError, "frozen_claim_record_invalid"):
            validate_packet(packet, expected_split="val", allow_ineligible_rows=True)


if __name__ == "__main__":
    unittest.main()
