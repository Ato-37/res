from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

from run_frozen_frame_mean_attention_bias_interaction import (
    compute_interaction,
    DIRECT_FAITHFULNESS,
    group_score_receipts,
    LOCK_PURPOSE,
    PACKET_PURPOSE,
    PROMPT_VERSION,
    read_private_json,
    validate_frozen_membership,
    validate_packet_filter_binding,
    validate_packet_payload,
    _validate_score,
)
from run_saved_claim_frame_permutation_attention_smoke import (
    LAYERS_1_BASED, MODEL_REVISION, _write_receipt_exclusive,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class InteractionTests(unittest.TestCase):
    def test_direct_faithfulness_is_fixed_false(self):
        self.assertIs(DIRECT_FAITHFULNESS, False)

    def test_difference_in_differences(self):
        result = compute_interaction(2.0, 1.0, 2.5, 1.25)
        self.assertEqual(result["baseline_original_minus_frame_mean_nats_per_token"], 1.0)
        self.assertEqual(result["biased_original_minus_frame_mean_nats_per_token"], 1.25)
        self.assertEqual(result["bias_by_frame_mean_interaction_nats_per_token"], 0.25)

    def test_common_bias_shift_cancels(self):
        result = compute_interaction(-0.4, -0.9, -0.2, -0.7)
        self.assertAlmostEqual(result["bias_by_frame_mean_interaction_nats_per_token"], 0.0)

    def test_invalid_logprobs_rejected(self):
        for values in ((float("nan"), 0.0, 0.0, 0.0),
                       (0.0, float("inf"), 0.0, 0.0),
                       (0.0, 0.0, True, 0.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_interaction(*values)

    def test_lock_packet_pair_filter_binding_for_ten_rows(self):
        media = digest("media-7")
        qid = "qid-7"
        pair = digest("pair-7")
        pair_filter = {"schema_version": 1, "pair_sha256": [digest(f"pair-{i}") for i in range(10)]}
        pair_sha = "f" * 64
        lock = {"status": "frozen_before_claim_generation_and_scoring", "purpose": LOCK_PURPOSE,
            "dataset_split": "test", "requested_videos": 10, "effective_videos": 10,
            "selected_pair_filter_sha256": pair_sha,
            "assignments": [{"media_sha256": digest(f"media-{i}"),
                "qid_sha256": digest(f"qid-{i}")} for i in range(10)]}
        row = {"dataset_question_id": qid, "video_sha256": media,
            "candidate_pair_sha256": digest("pair-7"), "review_index": 7}
        actual = validate_frozen_membership(lock, "e" * 64, pair_filter,
            pair_sha, row, 7, 7)
        self.assertEqual(actual["row_ordinal"], 7)
        self.assertEqual(actual["cohort_size"], 10)
        self.assertEqual(actual["row_token_sha256"], digest(media + qid))

    def test_rejects_wrong_assignment_pair_or_ordinal(self):
        pair_filter = {"schema_version": 1, "pair_sha256": [digest("pair") ]}
        lock = {"status": "frozen_before_claim_generation_and_scoring", "purpose": LOCK_PURPOSE,
            "dataset_split": "test", "requested_videos": 1, "effective_videos": 1,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [{"media_sha256": digest("media"),
                "qid_sha256": digest("qid")}]}
        row = {"dataset_question_id": "qid", "video_sha256": digest("media"),
            "candidate_pair_sha256": digest("pair"), "review_index": 0}
        validate_frozen_membership(lock, "b" * 64, pair_filter, "a" * 64, row, 0, 0)
        for bad_row, bad_ordinal in (
                ({**row, "video_sha256": digest("other")}, 0),
                ({**row, "dataset_question_id": "other"}, 0),
                ({**row, "candidate_pair_sha256": digest("absent")}, 0),
                (row, 1)):
            with self.subTest(row=bad_row, ordinal=bad_ordinal), self.assertRaises(ValueError):
                validate_frozen_membership(lock, "b" * 64, pair_filter,
                    "a" * 64, bad_row, bad_ordinal, 0)

    def test_pair_filter_order_is_bound_to_lock_ordinal(self):
        pair0, pair1 = digest("pair-0"), digest("pair-1")
        pair_filter = {"schema_version": 1, "pair_sha256": [pair0, pair1]}
        lock = {"status": "frozen_before_claim_generation_and_scoring", "purpose": LOCK_PURPOSE,
            "dataset_split": "test", "requested_videos": 2, "effective_videos": 2,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [
                {"media_sha256": digest("media-0"), "qid_sha256": digest("qid-0")},
                {"media_sha256": digest("media-1"), "qid_sha256": digest("qid-1")} ]}
        row = {"dataset_question_id": "qid-1", "video_sha256": digest("media-1"),
            "candidate_pair_sha256": pair0, "review_index": 1}
        with self.assertRaisesRegex(ValueError, "packet_row_not_in_frozen_cohort"):
            validate_frozen_membership(lock, "b" * 64, pair_filter,
                "a" * 64, row, 1, 1)

    def test_pair_filter_length_must_equal_locked_cohort_size(self):
        lock = {"status": "frozen_before_claim_generation_and_scoring", "purpose": LOCK_PURPOSE,
            "dataset_split": "test", "requested_videos": 2, "effective_videos": 2,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [
                {"media_sha256": digest("media-0"), "qid_sha256": digest("qid-0")},
                {"media_sha256": digest("media-1"), "qid_sha256": digest("qid-1")} ]}
        row = {"dataset_question_id": "qid-0", "video_sha256": digest("media-0"),
            "candidate_pair_sha256": digest("pair-0"), "review_index": 0}
        with self.assertRaisesRegex(ValueError, "frozen_lock_pair_filter_contract_invalid"):
            validate_frozen_membership(lock, "b" * 64,
                {"schema_version": 1, "pair_sha256": [digest("pair-0")]},
                "a" * 64, row, 0, 0)

    def test_partial_packet_returns_valid_and_invalid_rows_for_bound_handling(self):
        valid = {"review_index": 0, "generation_prompt_version": PROMPT_VERSION,
            "generated_parsed": {"valid": True, "claim": "a short claim"}}
        invalid = {"review_index": 1, "generation_prompt_version": PROMPT_VERSION,
            "generated_parsed": {"valid": False, "claim": ""}}
        packet = {"status": "partial", "purpose": PACKET_PURPOSE,
            "dataset_split": "test", "generation_prompt_version": PROMPT_VERSION,
            "model_revision": MODEL_REVISION, "claim_review_completed": False,
            "fully_structured_claim_records": 1, "records": [valid, invalid]}
        self.assertIs(validate_packet_payload(packet, 0), valid)
        selected_invalid = validate_packet_payload(packet, 1)
        self.assertIs(selected_invalid, invalid)
        self.assertFalse(selected_invalid["generated_parsed"]["valid"])

    def test_partial_packet_rejects_valid_count_and_record_order_mismatch(self):
        packet = {"status": "partial", "purpose": PACKET_PURPOSE,
            "dataset_split": "test", "generation_prompt_version": PROMPT_VERSION,
            "model_revision": MODEL_REVISION, "claim_review_completed": False,
            "fully_structured_claim_records": 2,
            "records": [
                {"review_index": 0, "generation_prompt_version": PROMPT_VERSION,
                    "generated_parsed": {"valid": True, "claim": "x"}},
                {"review_index": 8, "generation_prompt_version": PROMPT_VERSION,
                    "generated_parsed": {"valid": False, "claim": ""}}]}
        with self.assertRaisesRegex(ValueError, "claim_packet_valid_count_mismatch"):
            validate_packet_payload(packet, 0)
        packet["fully_structured_claim_records"] = 1
        with self.assertRaisesRegex(ValueError, "packet_record_order_binding_invalid"):
            validate_packet_payload(packet, 1)

    def test_packet_purpose_and_prompt_version_are_frozen(self):
        row = {"review_index": 0, "generation_prompt_version": PROMPT_VERSION,
            "generated_parsed": {"valid": True, "claim": "x"}}
        packet = {"status": "passed", "purpose": PACKET_PURPOSE,
            "dataset_split": "test", "generation_prompt_version": PROMPT_VERSION,
            "model_revision": MODEL_REVISION, "claim_review_completed": False,
            "fully_structured_claim_records": 1, "records": [row]}
        validate_packet_payload(packet, 0)
        with self.assertRaisesRegex(ValueError, "claim_packet_contract_invalid"):
            validate_packet_payload({**packet, "purpose": "different"}, 0)
        with self.assertRaisesRegex(ValueError, "claim_packet_contract_invalid"):
            validate_packet_payload({**packet, "generation_prompt_version": "v1"}, 0)
        with self.assertRaisesRegex(ValueError, "packet_record_prompt_version_mismatch"):
            validate_packet_payload({**packet, "records": [{**row,
                "generation_prompt_version": "v1"}]}, 0)

    def test_batch_packet_pair_filter_binding(self):
        selected_pair = digest("pair-selected")
        other_pair = digest("pair-other")
        pair_filter = {"schema_version": 1, "pair_sha256": [selected_pair, other_pair]}
        raw = (json.dumps(pair_filter, sort_keys=True, separators=(",", ":")) + "\n").encode()
        file_sha = hashlib.sha256(raw).hexdigest()
        canonical = hashlib.sha256(json.dumps(sorted(pair_filter["pair_sha256"]),
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        packet = {"candidate_pair_filter_sha256": file_sha,
            "candidate_pair_filter_digest": canonical,
            "candidate_pair_filter_entries": 2}
        row = {"candidate_pair_sha256": selected_pair}
        binding = validate_packet_filter_binding(packet, pair_filter, file_sha, row)
        self.assertEqual(binding["packet_pair_filter_sha256"], file_sha)
        self.assertEqual(binding["packet_pair_filter_entries"], 2)
        with self.assertRaisesRegex(ValueError, "packet_pair_filter_binding_invalid"):
            validate_packet_filter_binding({**packet,
                "candidate_pair_filter_sha256": "0" * 64}, pair_filter, file_sha, row)
        with self.assertRaisesRegex(ValueError, "packet_pair_filter_binding_invalid"):
            validate_packet_filter_binding({**packet,
                "candidate_pair_filter_digest": "0" * 64}, pair_filter, file_sha, row)
        with self.assertRaisesRegex(ValueError, "packet_pair_filter_binding_invalid"):
            validate_packet_filter_binding({**packet,
                "candidate_pair_filter_entries": 1}, pair_filter, file_sha, row)
        with self.assertRaisesRegex(ValueError, "packet_pair_filter_binding_invalid"):
            validate_packet_filter_binding(packet, pair_filter, file_sha,
                {"candidate_pair_sha256": digest("not-selected")})

    def test_lock_purpose_is_fixed_to_frame_mean_protocol(self):
        lock = {"status": "frozen_before_claim_generation_and_scoring",
            "purpose": "other_protocol", "dataset_split": "test",
            "requested_videos": 1, "effective_videos": 1,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [{"media_sha256": digest("media"),
                "qid_sha256": digest("qid")}]}
        with self.assertRaisesRegex(ValueError, "frozen_lock_pair_filter_contract_invalid"):
            validate_frozen_membership(lock, "b" * 64,
                {"schema_version": 1, "pair_sha256": [digest("pair")]},
                "a" * 64, {"dataset_question_id": "qid",
                    "video_sha256": digest("media"),
                    "candidate_pair_sha256": digest("pair"), "review_index": 0}, 0, 0)

    def test_rejects_bad_lock_contract_and_duplicate_filter(self):
        lock = {"status": "frozen_before_claim_generation_and_scoring", "purpose": LOCK_PURPOSE,
            "dataset_split": "test", "requested_videos": 1, "effective_videos": 1,
            "selected_pair_filter_sha256": "a" * 64,
            "assignments": [{"media_sha256": digest("media"), "qid_sha256": digest("qid")}]}
        row = {"dataset_question_id": "qid", "video_sha256": digest("media"),
            "candidate_pair_sha256": digest("pair"), "review_index": 0}
        with self.assertRaisesRegex(ValueError, "frozen_lock_pair_filter_contract_invalid"):
            validate_frozen_membership({**lock, "effective_videos": 2}, "b" * 64,
                {"schema_version": 1, "pair_sha256": [digest("pair")]}, "a" * 64, row, 0, 0)
        with self.assertRaisesRegex(ValueError, "pair_filter_entries_invalid"):
            duplicate_lock = {**lock, "requested_videos": 2, "effective_videos": 2,
                "assignments": lock["assignments"] * 2}
            validate_frozen_membership(duplicate_lock, "b" * 64,
                {"schema_version": 1, "pair_sha256": [digest("pair"), digest("pair")]},
                "a" * 64, row, 0, 0)

    def test_private_reader_requires_private_mode_and_containment(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            private = repo / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            good = private / "input.json"
            good.write_text('{"ok":true}', encoding="utf-8")
            good.chmod(0o600)
            value, sha, path = read_private_json(repo, Path("data/private_nextgqa/input.json"))
            self.assertEqual(value, {"ok": True})
            self.assertEqual(sha, hashlib.sha256(good.read_bytes()).hexdigest())
            self.assertEqual(path, good)
            bad = private / "loose.json"
            bad.write_text("{}", encoding="utf-8")
            bad.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "private_input_permissions_invalid"):
                read_private_json(repo, Path("data/private_nextgqa/loose.json"))
            with self.assertRaisesRegex(ValueError, "private_input_path_invalid"):
                read_private_json(repo, Path("../outside.json"))

    def test_receipt_writer_is_exclusive_and_owner_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            _write_receipt_exclusive(path, {"status": "failed", "direct_faithfulness": False})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()),
                {"direct_faithfulness": False, "status": "failed"})
            with self.assertRaises(FileExistsError):
                _write_receipt_exclusive(path, {"status": "overwrite"})

    def test_score_receipt_requires_finite_target_score_and_exact_hooks(self):
        base = {"score_semantics": "mean conditional logprob of frozen target token IDs",
            "target_token_count": 5, "target_logprob_mean": -0.4,
            "visual_key_position_count": 12,
            "intervention_hook_invocations_by_layer_1_based": {
                layer: 1 for layer in LAYERS_1_BASED}}
        _validate_score(base, target_count=5, biased=True)
        for invalid in ({**base, "target_logprob_mean": float("nan")},
                        {**base, "target_token_count": 4},
                        {**base, "intervention_hook_invocations_by_layer_1_based": {}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_score(invalid, target_count=5, biased=True)
        unbias = {**base, "visual_key_position_count": 12,
            "intervention_hook_invocations_by_layer_1_based": {
                layer: 0 for layer in LAYERS_1_BASED}}
        _validate_score(unbias, target_count=5, biased=False)

    def test_score_grouping_is_explicit_and_preserves_only_sanitized_branches(self):
        self.assertEqual(group_score_receipts({}), {"baseline": {}, "biased": {}})
        score = {"target_token_count": 3, "first_query_position": 10,
            "last_target_query_position": 12, "target_logprob_sum": -1.5,
            "target_logprob_mean": -0.5,
            "target_logprob_token_values": [-0.5, -0.4, -0.6],
            "score_semantics": "mean conditional logprob of frozen target token IDs",
            "intervention_hook_invocations_by_layer_1_based": {5: 0, 6: 0}}
        grouped = group_score_receipts({
            ("baseline", "original_video"): score,
            ("biased", "per_frame_rgb_mean_video"): {**score,
                "intervention_hook_invocations_by_layer_1_based": {5: 1, 6: 1}},
        })
        self.assertEqual(set(grouped), {"baseline", "biased"})
        self.assertIn("original_video", grouped["baseline"])
        self.assertIn("per_frame_rgb_mean_video", grouped["biased"])
        self.assertNotIn("target_logprob_token_values", grouped["baseline"]["original_video"])
        self.assertEqual(grouped["biased"]["per_frame_rgb_mean_video"]
            ["intervention_hook_invocations_by_layer_1_based"], {"5": 1, "6": 1})
        partial = group_score_receipts({("baseline", "original_video"): score})
        self.assertEqual(set(partial["baseline"]), {"original_video"})
        self.assertEqual(partial["biased"], {})


if __name__ == "__main__":
    unittest.main()
