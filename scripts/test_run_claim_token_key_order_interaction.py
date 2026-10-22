from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_claim_token_key_order_interaction import (
    BRANCHES, LOGIT_BIAS, build_claim_key_plan, build_preflight_receipt,
    build_six_forward_schedule, canonical_json_bytes, compute_interactions,
    deterministic_reverse_group_indices, map_group_keys_to_permuted_positions,
    select_disjoint_hash_sham_groups, validate_locked_claim_packets,
    _argument_parser, _processor_temporal_groups, _safe_failure_code,
    _run_gpu, require_gpu_execution_authorization,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ClaimTokenKeyOrderTests(unittest.TestCase):
    def test_smoke_row_token_uses_nested_receipt_schema(self):
        row_token = "a" * 64
        receipt_row = {"row": {"row_token_sha256": row_token}}
        self.assertEqual(receipt_row["row"].get("row_token_sha256"), row_token)

    def test_argparse_custom_packet_does_not_append_legacy_default(self):
        args = _argument_parser().parse_args([
            "--mode", "smoke", "--output", "data/private_nextgqa/out.json",
            "--claim-packet", "data/private_nextgqa/packet08.json"])
        self.assertEqual(args.claim_packet,
            ["data/private_nextgqa/packet08.json"])

    def test_processor_group_geometry_matches_temporal_patch_and_merge(self):
        class Row:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        class Tensor:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), len(rows[0]))
            def __getitem__(self, index): return Row(self.rows[index])
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(merge_size=2),
            video_processor=SimpleNamespace(temporal_patch_size=2))
        ids = [7] + [99] * 16 + [8]
        groups = _processor_temporal_groups({
            "video_grid_thw": Tensor([[4, 4, 4]]), "input_ids": Tensor([ids])},
            processor, 99, 8)
        self.assertEqual(groups, tuple(tuple(range(1 + i*4, 5 + i*4)) for i in range(4)))
        with self.assertRaisesRegex(ValueError, "frame_contract"):
            _processor_temporal_groups({"video_grid_thw": Tensor([[3, 4, 4]]),
                "input_ids": Tensor([ids])}, processor, 99, 8)
        with self.assertRaisesRegex(ValueError, "token_block"):
            _processor_temporal_groups({"video_grid_thw": Tensor([[4, 4, 4]]),
                "input_ids": Tensor([[7] + [99] * 15 + [8]])}, processor, 99, 8)

    def test_os_error_failure_code_never_contains_path(self):
        code = _safe_failure_code(OSError("missing /private/user/video.mp4"))
        self.assertEqual(code, "operation_failed_oserror")
        self.assertNotIn("/private", code)

    def test_fixed_top_ten_percent_and_disjoint_equal_budget_sham(self):
        groups = [(i,) for i in range(10)]
        scores = [0.0] * 10
        scores[6] = 0.9
        scores[2] = 0.9
        plan = build_claim_key_plan(attention_scores=scores,
            temporal_groups=groups, query_positions=(20, 21, 22),
            opaque_row_token="a" * 64)
        self.assertEqual(plan["layers_1_based"], list(range(5, 17)))
        self.assertEqual(plan["logit_bias_requested"], LOGIT_BIAS)
        self.assertEqual(plan["selected_group_indices"], [2])
        self.assertEqual(len(plan["sham_group_indices"]), 1)
        self.assertNotEqual(plan["selected_group_indices"], plan["sham_group_indices"])
        self.assertEqual(plan["selected_key_positions"], [2])
        self.assertEqual(len(plan["sham_key_positions"]), 1)

    def test_hash_sham_is_deterministic_and_excludes_selected(self):
        first = select_disjoint_hash_sham_groups(group_count=8,
            selected_groups=(1, 4), opaque_row_token="b" * 64)
        second = select_disjoint_hash_sham_groups(group_count=8,
            selected_groups=(1, 4), opaque_row_token="b" * 64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertFalse(set(first) & {1, 4})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_disjoint_hash_sham_groups(group_count=3,
                selected_groups=(0, 1), opaque_row_token="b" * 64)

    def test_rejects_noncausal_or_malformed_attention_groups(self):
        with self.assertRaisesRegex(ValueError, "causal"):
            build_claim_key_plan(attention_scores=[1, 2],
                temporal_groups=((0, 1), (2, 3)), query_positions=(3, 4),
                opaque_row_token="c" * 64)
        with self.assertRaisesRegex(ValueError, "partition"):
            build_claim_key_plan(attention_scores=[1, 2],
                temporal_groups=((0, 1), (3, 4)), query_positions=(8, 9),
                opaque_row_token="c" * 64)
        with self.assertRaisesRegex(ValueError, "score_count"):
            build_claim_key_plan(attention_scores=[1],
                temporal_groups=((0,), (1,)), query_positions=(8, 9),
                opaque_row_token="c" * 64)

    def test_reversal_maps_same_content_groups_not_same_positions(self):
        self.assertEqual(deterministic_reverse_group_indices(4), (3, 2, 1, 0))
        original = ((5, 6), (7, 8), (9, 10), (11, 12))
        permuted = ((5, 6), (7, 8), (9, 10), (11, 12))
        positions = map_group_keys_to_permuted_positions(
            selected_group_indices=(0, 3), original_groups=original,
            permuted_groups=permuted,
            permuted_group_for_original=deterministic_reverse_group_indices(4))
        self.assertEqual(positions, (5, 6, 11, 12))

    def test_forward_schedule_has_capture_plus_five_comparison_branches(self):
        schedule = build_six_forward_schedule()
        self.assertEqual({row["branch"] for row in schedule}, BRANCHES)
        self.assertEqual(sum(row["capture_claim_query_attention"] == "yes"
                             for row in schedule), 1)
        self.assertEqual(len(schedule), 6)

    def test_difference_in_differences_and_sham_specificity(self):
        scores = {
            "natural_baseline": -2.0, "permuted_baseline": -2.5,
            "natural_claim_key": -1.5, "permuted_claim_key": -2.0,
            "natural_hash_sham": -1.7, "permuted_hash_sham": -2.4,
        }
        result = compute_interactions(scores)
        self.assertAlmostEqual(result["baseline_natural_minus_permuted_nats_per_token"], .5)
        self.assertAlmostEqual(result["claim_key_order_interaction_nats_per_token"], 0)
        self.assertAlmostEqual(result["hash_sham_order_interaction_nats_per_token"], .2)
        self.assertAlmostEqual(result["claim_key_minus_sham_specificity_nats_per_token"], -.2)
        with self.assertRaisesRegex(ValueError, "branch_score_set"):
            compute_interactions({"natural_baseline": 0})

    def _fixture(self):
        pairs = {"schema_version": 1, "pair_sha256": [sha("p0"), sha("p1"), sha("p2")]}
        lock = {"status": "frozen_before_claim_generation_and_scoring",
            "dataset_split": "test", "requested_videos": 3, "effective_videos": 3,
            "cohort_sha256": sha("cohort"),
            "selected_pair_filter_sha256": hashlib.sha256(canonical_json_bytes(pairs)).hexdigest(),
            "assignments": [{"media_sha256": sha(f"media{i}"),
                             "qid_sha256": sha(str(i))} for i in range(3)]}
        records = [{"video_sha256": sha(f"media{i}"), "dataset_question_id": i,
            "candidate_pair_sha256": pairs["pair_sha256"][i],
            "generated_parsed": {"valid": True, "claim": f"claim {i}"}}
            for i in range(3)]
        packet = {"status": "passed", "dataset_split": "test",
            "cohort_sha256": lock["cohort_sha256"], "requested_videos": 3,
            "effective_videos": 3, "fully_structured_claim_records": 3,
            "records": records}
        return lock, pairs, packet

    def test_validates_exact_three_locked_claim_rows(self):
        lock, pairs, packet = self._fixture()
        rows = validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                               packet_payloads=(packet,))
        self.assertEqual(len(rows), 3)

    def test_accepts_split_packets_but_restores_frozen_row_order(self):
        lock, pairs, packet = self._fixture()
        records = packet["records"]
        first = {**packet, "requested_videos": 1, "effective_videos": 1,
            "fully_structured_claim_records": 1, "records": [records[2]]}
        second = {**packet, "requested_videos": 2, "effective_videos": 2,
            "fully_structured_claim_records": 2, "records": [records[1], records[0]]}
        rows = validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                               packet_payloads=(first, second))
        self.assertEqual([row["dataset_question_id"] for row in rows], [0, 1, 2])

    def test_rejects_wrong_pair_order_duplicate_row_and_missing_claim(self):
        lock, pairs, packet = self._fixture()
        mutated = {**packet, "records": [dict(row) for row in packet["records"]]}
        mutated["records"][0] = {**mutated["records"][0],
            "candidate_pair_sha256": pairs["pair_sha256"][1]}
        with self.assertRaisesRegex(ValueError, "not_in_frozen"):
            validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                           packet_payloads=(mutated,))
        duplicate = {**packet, "records": [packet["records"][0],
            packet["records"][0], packet["records"][2]]}
        with self.assertRaisesRegex(ValueError, "duplicate_packet_row"):
            validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                           packet_payloads=(duplicate,))
        invalid = {**packet, "status": "partial", "fully_structured_claim_records": 2,
            "records": [packet["records"][0],
            {**packet["records"][1], "generated_parsed": {"valid": False}},
            packet["records"][2]]}
        with self.assertRaisesRegex(ValueError, "all_frozen_claims"):
            validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                           packet_payloads=(invalid,))

    def test_preflight_receipt_is_explicitly_nonexecuting_and_source_bound(self):
        digest = "d" * 64
        receipt = build_preflight_receipt(lock_hash=digest,
            pair_filter_hash="e" * 64, packet_hashes={"packet.json": "f" * 64},
            row_count=3, source_hashes={"runner.py": digest},
            packet_declared_video_hashes=["1" * 64, "2" * 64, "3" * 64])
        self.assertEqual(receipt["status"], "inputs_validated_gpu_not_run")
        self.assertFalse(receipt["gpu_execution_performed"])
        self.assertFalse(receipt["direct_faithfulness"])
        self.assertEqual(receipt["planned_forward_count_full"], 18)
        self.assertFalse(receipt["video_bytes_rehashed"])

    def test_current_capture_protocol_denies_gpu_scoring_before_input_or_model_load(self):
        args = SimpleNamespace(mode="smoke",
            output="data/private_nextgqa/test_non_authorizing_runner_guard.json")
        with patch("run_claim_token_key_order_interaction.write_private_json_exclusive") as write:
            with redirect_stdout(StringIO()):
                status = _run_gpu(args)
        self.assertEqual(status, 2)
        write.assert_called_once()
        failure = write.call_args.args[1]
        self.assertEqual(failure["stage"], "capture_protocol_preload_authorization")
        self.assertEqual(failure["failure_code"], "capture_protocol_authorization_not_granted")
        self.assertEqual(failure["attempted_forwards"], 0)
        self.assertEqual(failure["completed_forwards"], 0)
        self.assertEqual(failure["effective_rows"], 0)
        with self.assertRaisesRegex(ValueError, "authorization_not_granted"):
            require_gpu_execution_authorization()

    def test_preflight_receipt_rejects_duplicate_or_malformed_video_hashes(self):
        args = dict(lock_hash="d" * 64, pair_filter_hash="e" * 64,
            packet_hashes={"packet.json": "f" * 64}, row_count=3,
            source_hashes={"runner.py": "d" * 64})
        with self.assertRaisesRegex(ValueError, "binding_invalid"):
            build_preflight_receipt(**args, packet_declared_video_hashes=["a"*64]*3)
        with self.assertRaisesRegex(ValueError, "binding_invalid"):
            build_preflight_receipt(**args, packet_declared_video_hashes=["bad"]*3)

    def test_source_checkpoint_and_execution_are_not_falsely_claimed_in_preflight(self):
        receipt = build_preflight_receipt(lock_hash="d"*64,
            pair_filter_hash="e"*64, packet_hashes={"p": "f"*64}, row_count=3,
            source_hashes={"s": "d"*64},
            packet_declared_video_hashes=["1"*64, "2"*64, "3"*64])
        self.assertEqual(receipt["status"], "inputs_validated_gpu_not_run")
        self.assertFalse(receipt["gpu_execution_performed"])

    def test_forward_schedule_has_exact_branch_protocol(self):
        schedule = build_six_forward_schedule()
        self.assertEqual([r["branch"] for r in schedule], [
            "natural_baseline", "permuted_baseline", "natural_claim_key",
            "permuted_claim_key", "natural_hash_sham", "permuted_hash_sham"])
        self.assertIn("same_content_selected_claim_key_ln2", schedule[3]["intervention"])


if __name__ == "__main__":
    unittest.main()
