import json
import hashlib
import stat
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from run_nextgqa_gold_interval_margin_pilot import (
    control_minus_evidence_margin,
    candidate_pair_sha256,
    build_cohort_lock_assignments,
    filter_candidate_groups,
    finalize_cuda_scoring,
    gold_vs_best_foil_margin,
    helper_source_hashes,
    prepare_conditions,
    validate_cohort_cardinality,
    validate_cohort_lock_payload,
    validate_locked_selection_assignments,
    validate_candidate_filter_binding,
    validate_candidate_pair_filter,
    validate_private_media_exclusion,
    validate_preflight_cardinality,
    read_private_json_snapshot,
    validate_new_output_path,
    write_private_json_exclusive,
)


class GoldIntervalAnswerMarginTests(unittest.TestCase):
    def test_private_json_snapshot_checks_schema_permissions_and_same_bytes_hash(self):
        payload = {"schema_version": 1, "pair_sha256": ["a" * 64]}
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filter.json"
            path.write_bytes(raw)
            path.chmod(0o600)
            parsed, digest = read_private_json_snapshot(path)
            self.assertEqual(parsed, payload)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            validate_candidate_filter_binding(
                {"candidate_pair_filter_sha256": digest}, digest)
            path.write_text('{"schema_version":1,"pair_sha256":[]}', encoding="utf-8")
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError, "duplicate_key"):
                path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
                path.chmod(0o600)
                read_private_json_snapshot(path)
            with self.assertRaisesRegex(ValueError, "permissions_invalid"):
                path.write_text(json.dumps(payload), encoding="utf-8")
                path.chmod(0o644)
                read_private_json_snapshot(path)

    def test_candidate_filter_schema_rejects_extra_keys(self):
        with self.assertRaisesRegex(ValueError, "private_candidate_pair_filter_invalid"):
            validate_candidate_pair_filter({"schema_version": 1,
                                            "pair_sha256": ["a" * 64],
                                            "comment": "extra"})

    def test_cohort_lock_private_snapshot_and_strict_top_level_schema(self):
        raw_payload = {"schema_version": 1, "assignments": [
            {"media_sha256": "a" * 64, "qid_sha256": "b" * 64}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            path.write_text(json.dumps(raw_payload), encoding="utf-8")
            path.chmod(0o600)
            parsed, digest = read_private_json_snapshot(path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(validate_cohort_lock_payload(parsed, 1), raw_payload["assignments"])
            parsed["extra"] = "reject"
            with self.assertRaisesRegex(ValueError, "private_cohort_lock_invalid"):
                validate_cohort_lock_payload(parsed, 1)

    def test_locked_selection_must_consume_each_media_qid_pair_once(self):
        def row(media, qid):
            return {"decode": {"source_sha256": media}, "qid": qid}
        media_a, media_b = "a" * 64, "b" * 64
        qid_a = hashlib.sha256(b"q-a").hexdigest()
        qid_b = hashlib.sha256(b"q-b").hexdigest()
        lock = [{"media_sha256": media_a, "qid_sha256": qid_a},
                {"media_sha256": media_b, "qid_sha256": qid_b}]
        validate_locked_selection_assignments([row(media_a, "q-a"), row(media_b, "q-b")], lock)
        with self.assertRaisesRegex(ValueError, "selected_cohort_lock_assignments_invalid"):
            validate_locked_selection_assignments(
                [row(media_a, "q-a"), row(media_a, "q-a")], lock)
        with self.assertRaisesRegex(ValueError, "cohort_lock_selected_assignment_mismatch"):
            validate_locked_selection_assignments([row(media_a, "q-a"), row(media_b, "q-other")], lock)

    def test_final_cuda_sync_failure_discards_unverified_metrics(self):
        class BrokenCuda:
            @staticmethod
            def synchronize():
                raise RuntimeError("synthetic_cuda_sync_failure")
        fake_torch = SimpleNamespace(cuda=BrokenCuda)
        records = [{"primary_effect": 1.25}]
        failures = []
        passed, discarded = finalize_cuda_scoring(fake_torch, records, failures)
        self.assertFalse(passed)
        self.assertEqual(discarded, 1)
        self.assertEqual(records, [])
        self.assertEqual(failures, [{
            "video_index": None,
            "failure_type": "RuntimeError",
            "failed_condition": "final_cuda_synchronize",
            "completed_conditions": None,
        }])

    def test_lock_output_path_rejects_existing_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "exists"
            existing.write_text("", encoding="utf-8")
            dangling_link = root / "dangling"
            dangling_link.symlink_to(root / "missing")
            for path in (existing, dangling_link):
                with self.subTest(path=path), self.assertRaisesRegex(
                        ValueError, "refusing_to_overwrite_output"):
                    validate_new_output_path(path)
            with self.assertRaisesRegex(ValueError, "output_paths_must_be_different"):
                validate_new_output_path(root / "same", root / "same")

    def test_cohort_lock_output_contains_selected_hashes_and_is_private_exclusive(self):
        media_sha = "a" * 64
        selected = [{"decode": {"source_sha256": media_sha}, "qid": "qid-private"}]
        assignments = build_cohort_lock_assignments(selected)
        self.assertEqual(assignments, [{
            "media_sha256": media_sha,
            "qid_sha256": hashlib.sha256(b"qid-private").hexdigest(),
        }])
        payload = {"schema_version": 1, "assignments": assignments}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cohort-lock.json"
            write_private_json_exclusive(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_private_json_exclusive(output, payload)

    def test_private_pair_filter_keeps_only_exact_video_qid_pairs(self):
        q_a = candidate_pair_sha256("video-a", "q1")
        q_b = candidate_pair_sha256("video-a", "q2")
        grouped = {"video-a": [{"qid": "q1"}, {"qid": "q2"}],
                   "video-b": [{"qid": "q1"}]}
        filtered = filter_candidate_groups(grouped, [q_a])
        self.assertEqual(filtered, {"video-a": [{"qid": "q1"}]})
        self.assertEqual(filter_candidate_groups(grouped, None), grouped)
        self.assertNotEqual(q_a, candidate_pair_sha256("video-b", "q1"))
        self.assertNotEqual(q_a, q_b)

    def test_pair_filter_rejects_malformed_or_duplicate_hashes(self):
        valid = "a" * 64
        self.assertEqual(validate_candidate_pair_filter(
            {"schema_version": 1, "pair_sha256": [valid]}), [valid])
        for payload in (
                {"schema_version": 1, "pair_sha256": [valid, valid]},
                {"schema_version": 1, "pair_sha256": ["private-id"]},
                {"schema_version": True, "pair_sha256": [valid]},
                {"schema_version": 1, "pair_sha256": []},
                {"schema_version": 1, "pair_sha256": ["A" * 64]}):
            with self.subTest(payload=payload), self.assertRaisesRegex(
                    ValueError, "private_candidate_pair_filter_invalid"):
                validate_candidate_pair_filter(payload)

    def test_media_exclusion_manifest_schema_and_uniqueness(self):
        valid = "a" * 64
        self.assertEqual(validate_private_media_exclusion(
            {"schema_version": 1, "media_sha256": [valid]}), [valid])
        for payload in (
                {"schema_version": 1, "media_sha256": [valid, valid]},
                {"schema_version": 1, "media_sha256": []},
                {"schema_version": True, "media_sha256": [valid]},
                {"schema_version": 1, "media_sha256": ["private-id"]},
                {"schema_version": 1, "media_sha256": [valid], "extra": 1}):
            with self.subTest(payload=payload), self.assertRaisesRegex(
                    ValueError, "private_media_exclusion_manifest_invalid"):
                validate_private_media_exclusion(payload)

    def test_preflight_binds_exact_candidate_filter_file_hash(self):
        validate_candidate_filter_binding(
            {"candidate_pair_filter_sha256": "a" * 64}, "a" * 64)
        for gate, expected in (({"candidate_pair_filter_sha256": "a" * 64}, "b" * 64),
                               ({}, "a" * 64)):
            with self.subTest(gate=gate, expected=expected), self.assertRaisesRegex(
                    ValueError, "candidate_pair_filter_binding_mismatch"):
                validate_candidate_filter_binding(gate, expected)

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
