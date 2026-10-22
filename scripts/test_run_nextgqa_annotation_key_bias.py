from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_nextgqa_annotation_key_bias as runner
import torch


class AnnotationKeyBiasGeometryTests(unittest.TestCase):
    def test_annotation_frames_map_to_unique_complete_temporal_groups(self):
        self.assertEqual(
            runner.map_annotation_frames_to_groups(
                evidence_frame_indices=(0, 1, 4), frame_count=8,
                temporal_patch_size=2, temporal_group_count=4),
            (0, 2))

    def test_zero_hit_and_all_group_annotations_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "annotation_frame_group_inputs_invalid"):
            runner.map_annotation_frames_to_groups(evidence_frame_indices=(),
                frame_count=8, temporal_patch_size=2, temporal_group_count=4)
        with self.assertRaisesRegex(ValueError, "annotation_temporal_group_budget_invalid"):
            runner.map_annotation_frames_to_groups(evidence_frame_indices=(0, 2, 4, 6),
                frame_count=8, temporal_patch_size=2, temporal_group_count=4)

    def test_temporal_key_positions_require_exact_contiguous_video_run(self):
        ids = (1, 99, 99, 99, 99, 99, 99, 99, 99, 2)
        groups = runner.video_group_key_positions(input_ids=ids, video_token_id=99,
            video_grid_thw=(2, 4, 4), spatial_merge_size=2, temporal_patch_size=1)
        self.assertEqual(groups, ((1, 2, 3, 4), (5, 6, 7, 8)))
        with self.assertRaisesRegex(ValueError, "video_placeholder_geometry_mismatch"):
            runner.video_group_key_positions(input_ids=(1, 99, 2, 99),
                video_token_id=99, video_grid_thw=(1, 2, 2),
                spatial_merge_size=1, temporal_patch_size=1)

    def test_hash_sham_is_deterministic_equal_count_and_disjoint(self):
        args = dict(temporal_group_count=8, annotation_groups=(1, 4),
                    opaque_row_token="a" * 64)
        first = runner.choose_disjoint_hash_sham(**args)
        self.assertEqual(first, runner.choose_disjoint_hash_sham(**args))
        self.assertEqual(len(first), 2)
        self.assertFalse(set(first) & set(args["annotation_groups"]))

    def test_no_equal_budget_sham_fails_instead_of_shrinking(self):
        with self.assertRaisesRegex(ValueError, "annotation_sham_inputs_invalid"):
            runner.choose_disjoint_hash_sham(temporal_group_count=4,
                annotation_groups=(0, 1, 2), opaque_row_token="b" * 64)

    def test_flattening_checks_exact_video_key_budget_and_disjointness(self):
        groups = ((10, 11), (12, 13), (14, 15), (16, 17))
        treatment, sham = runner.flatten_selected_key_positions(
            temporal_groups=groups, annotation_group_indices=(0, 2),
            sham_group_indices=(1, 3))
        self.assertEqual(len(treatment), len(sham))
        self.assertEqual(treatment, (10, 11, 14, 15))
        self.assertEqual(sham, (12, 13, 16, 17))
        with self.assertRaisesRegex(ValueError, "annotation_sham_group_selection_invalid"):
            runner.flatten_selected_key_positions(temporal_groups=groups,
                annotation_group_indices=(0,), sham_group_indices=(0,))


class AnnotationKeyBiasScoringTests(unittest.TestCase):
    def test_weight_shard_manifest_requires_all_five_and_tracks_bytes(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            repo_cache = hub / "models--Qwen--Qwen2.5-VL-7B-Instruct"
            snapshot = repo_cache / "snapshots" / "revision"
            blobs = hub / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            (repo_cache / "blobs").symlink_to(blobs, target_is_directory=True)
            for index, name in enumerate(runner.WEIGHT_SHARD_NAMES):
                prefix = blobs / f"prefix-{index}"
                prefix.mkdir()
                target = prefix / f"blob-{index}"
                target.write_bytes(f"shard-{index}".encode())
                # First hop is relative to the repo-local blobs symlink; its
                # target then resolves through that symlink into hub/blobs.
                (snapshot / name).symlink_to(
                    Path("../../blobs") / prefix.name / target.name)
            original = runner.hash_weight_shards(snapshot)
            self.assertEqual(original["shard_count"], 5)
            self.assertEqual(tuple(original["shards_sha256"]),
                             runner.WEIGHT_SHARD_NAMES)
            target = blobs / "prefix-2" / "blob-2"
            target.write_bytes(b"mutated")
            mutated = runner.hash_weight_shards(snapshot)
            self.assertNotEqual(original["shards_sha256"], mutated["shards_sha256"])
            self.assertNotEqual(original["shards_digest_sha256"],
                                mutated["shards_digest_sha256"])
            (snapshot / runner.WEIGHT_SHARD_NAMES[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "expected_five_weight_shards_not_present"):
                runner.hash_weight_shards(snapshot)

    def test_split_file_names_are_genuine_and_distinct(self):
        self.assertEqual(runner.dataset_filenames("val"), {
            "csv": "val.csv", "gsub": "gsub_val.json",
            "mapping": "map_vid_vidorID.json"})
        self.assertEqual(runner.dataset_filenames("test"), {
            "csv": "test.csv", "gsub": "gsub_test.json",
            "mapping": "map_vid_vidorID.json"})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, text in (("val.csv", "real-val"),
                               ("gsub_val.json", "val-annotations"),
                               ("test.csv", "real-test"),
                               ("gsub_test.json", "test-annotations"),
                               ("map_vid_vidorID.json", "map")):
                (root / name).write_text(text)
            val = runner._data_hashes(root, "val")
            test = runner._data_hashes(root, "test")
            self.assertEqual(val["csv_sha256"], runner.sha256_file(root / "val.csv"))
            self.assertEqual(val["gsub_sha256"], runner.sha256_file(root / "gsub_val.json"))
            self.assertNotEqual(val["csv_sha256"], test["csv_sha256"])
            self.assertNotEqual(val["gsub_sha256"], test["gsub_sha256"])
            self.assertEqual(val["dataset_split"], "val")
            self.assertEqual(test["dataset_split"], "test")

    def test_lock_is_split_bound_and_rejects_test_lock_for_validation(self):
        lock = {"schema_version": 2, "dataset_split": "val",
            "assignments": [{"media_sha256": "a" * 64, "qid_sha256": "b" * 64}]}
        self.assertEqual(runner.validate_split_bound_lock(
            lock, requested=1, dataset_split="val"), lock["assignments"])
        with self.assertRaisesRegex(ValueError, "cohort_lock_dataset_split_mismatch"):
            runner.validate_split_bound_lock(lock, requested=1, dataset_split="test")
        legacy = {"schema_version": 1, "assignments": lock["assignments"]}
        with self.assertRaisesRegex(ValueError, "cohort_lock_dataset_split_mismatch"):
            runner.validate_split_bound_lock(legacy, requested=1, dataset_split="val")

    def test_freezer_receives_requested_split(self):
        chosen = [{"row": 1}]
        with patch.object(runner.cohort, "freeze_cohort",
                          return_value=(chosen, {}, {}, "fingerprint")) as freeze:
            result = runner._freeze_exact(Path("."), video_root=Path("videos"),
                n=1, frame_count=16, exclusion=[], lock_assignments=None,
                dataset_split="val")
        self.assertEqual(result[0], chosen)
        self.assertEqual(freeze.call_args.kwargs["dataset_split"], "val")

    def test_baseline_best_foil_is_fixed_for_all_condition_margins(self):
        baseline = (-2.0, -1.0, -3.0, -4.0, -5.0)
        gold = 0
        foil = runner.baseline_best_foil(baseline, gold)
        self.assertEqual(foil, 1)
        treated = (-1.2, -2.5, -0.5, -3.0, -4.0)
        margin = runner.fixed_gold_foil_margin(treated, gold_index=gold,
                                               baseline_foil_index=foil)
        self.assertAlmostEqual(margin, 1.3)
        # The treated-condition best non-gold is C, but the contract keeps foil B.
        self.assertNotEqual(foil, max((i for i in range(5) if i != gold),
                                      key=lambda i: treated[i]))

    def test_nonfinite_scores_and_gold_as_foil_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline_choice_scores_invalid"):
            runner.baseline_best_foil((-2.0, float("nan"), -3.0, -4.0, -5.0), 0)
        with self.assertRaisesRegex(ValueError, "fixed_gold_foil_inputs_invalid"):
            runner.fixed_gold_foil_margin((-1.0,) * 5, gold_index=2,
                                          baseline_foil_index=2)

    def test_full_run_requires_complete_matching_smoke(self):
        first_row = {"row_token_sha256": "4" * 64, "media_sha256": "5" * 64,
            "qid_sha256": "6" * 64, "annotation_group_indices": [0, 1],
            "sham_group_indices": [2, 3], "annotation_group_count": 2,
            "temporal_group_count": 8, "sampled_frame_count": 16,
            "video_grid_thw": [8, 4, 4], "input_batch_sha256": "7" * 64,
            "video_keys_per_temporal_group": 2}
        expected_bias = float(torch.tensor(runner.BIAS_NATS, dtype=torch.float32).item())
        per_layer = {str(i): {"attention_mask_dtype": "torch.float32",
            "requested_bias": runner.BIAS_NATS, "effective_bias": expected_bias,
            "intended_changed_cells": 4, "observed_changed_cells": 4,
            "all_and_only_intended_cells_changed": True}
            for i in runner.LAYERS_1_BASED}
        audit = {"hook_calls_by_layer": {str(i): 1 for i in runner.LAYERS_1_BASED},
            "biased_key_count": 4, "requested_bias": runner.BIAS_NATS,
            "intended_changed_cells": 4, "observed_changed_cells": 4,
            "all_and_only_intended_cells_changed": True,
            "attention_mask_dtype": "torch.float32", "effective_bias": expected_bias,
            "effective_bias_per_layer": expected_bias, "per_layer": per_layer}
        values = {"baseline": [-1.0, -0.2, -1.2, -2.0, -3.0],
                  "annotation_interval_key_bias": [-0.5, -0.4, -1.4, -2.0, -3.0],
                  "hash_sham_key_bias": [-0.7, -0.3, -1.5, -2.0, -3.0]}
        margins = {name: scores[0] - scores[1] for name, scores in values.items()}
        record = {name: first_row[name] for name in (
            "row_token_sha256", "media_sha256", "qid_sha256",
            "annotation_group_indices", "sham_group_indices", "video_grid_thw",
            "input_batch_sha256")}
        record.update({"gold_choice_index": 0, "baseline_best_foil_index": 1,
            "conditions": {name: {"choice_logprobs_nats": scores,
                "gold_choice_logprob_nats": scores[0],
                "baseline_best_foil_logprob_nats": scores[1],
                "fixed_gold_minus_baseline_foil_margin_nats": margins[name]}
                for name, scores in values.items()},
            "primary_annotation_minus_sham_nats": margins["annotation_interval_key_bias"] - margins["hash_sham_key_bias"],
            "annotation_minus_baseline_nats": margins["annotation_interval_key_bias"] - margins["baseline"],
            "sham_minus_baseline_nats": margins["hash_sham_key_bias"] - margins["baseline"],
            "annotation_mask_audit": audit, "sham_mask_audit": audit})
        smoke = {"status": "smoke_passed", "purpose": runner.PURPOSE,
            "mode": "smoke", "dataset_split": "val",
            "requested_videos": 12, "effective_videos": 1,
            "requested_forwards": 3, "attempted_forwards": 3,
            "completed_forwards": 3, "cohort_lock_sha256": "1" * 64,
            "preflight_receipt_sha256": "2" * 64, "source_snapshot": {"x": 1},
            "dataset_sha256": {"y": "z"}, "model_config_sha256": "3" * 64,
            "candidate_manifest_digest_sha256": "8" * 64,
            "candidate_manifest_count": 9,
            "weight_shards_sha256": {name: str(i) * 64 for i, name in
                                      enumerate(runner.WEIGHT_SHARD_NAMES, 1)},
            "weight_shards_digest_sha256": "9" * 64,
            "weight_shard_count": 5, "records": [record]}
        weight_manifest = {"shards_sha256": smoke["weight_shards_sha256"],
            "shards_digest_sha256": "9" * 64, "shard_count": 5}
        kwargs = dict(args_n=12, lock_sha256="1" * 64,
            preflight_sha256="2" * 64, source={"x": 1},
            data_hashes={"y": "z"}, model_config_sha256="3" * 64,
            first_row_binding=first_row, dataset_split="val",
            candidate_manifest_digest="8" * 64,
            candidate_manifest_count=9, weight_shard_manifest=weight_manifest,
            torch_module=torch)
        runner.validate_smoke_receipt(smoke, **kwargs)
        for mutate in (
            lambda value: value["records"][0].update(input_batch_sha256="e" * 64),
            lambda value: value["records"][0]["annotation_mask_audit"].update(
                all_and_only_intended_cells_changed=False),
            lambda value: value.update(candidate_manifest_digest_sha256="f" * 64),
            lambda value: value.update(weight_shards_digest_sha256="f" * 64),
            lambda value: value["records"][0]["conditions"][
                "annotation_interval_key_bias"].update(
                    fixed_gold_minus_baseline_foil_margin_nats=9.0),
            lambda value: value["records"][0]["annotation_mask_audit"].update(
                biased_key_count=3)):
            corrupted = copy.deepcopy(smoke)
            mutate(corrupted)
            with self.assertRaisesRegex(ValueError,
                    "smoke_receipt_binding_or_completion_invalid"):
                runner.validate_smoke_receipt(corrupted, **kwargs)

    def test_attention_mask_delta_audit_checks_only_intended_cells_and_dtype(self):
        original = torch.zeros((1, 1, 4, 4), dtype=torch.bfloat16)
        original[0, 0, 0, 1:] = float("-inf")
        actual = original.clone()
        actual[0, 0, 3, 0] += runner.BIAS_NATS
        actual[0, 0, 3, 1] += runner.BIAS_NATS
        result = runner.audit_attention_mask_delta(original_mask=original,
            actual_mask=actual, query_position=3, key_positions=(0, 1),
            logit_bias=runner.BIAS_NATS, torch_module=torch)
        self.assertEqual(result["attention_mask_dtype"], "torch.bfloat16")
        self.assertEqual(result["effective_bias"],
                         float(torch.tensor(runner.BIAS_NATS, dtype=torch.bfloat16).item()))
        self.assertEqual(result["observed_changed_cells"], 2)
        self.assertTrue(result["all_and_only_intended_cells_changed"])
        altered = actual.clone()
        altered[0, 0, 2, 1] = 0.5
        with self.assertRaisesRegex(ValueError,
                "attention_mask_changed_outside_or_wrong_at_intended_cells"):
            runner.audit_attention_mask_delta(original_mask=original,
                actual_mask=altered, query_position=3, key_positions=(0, 1),
                logit_bias=runner.BIAS_NATS, torch_module=torch)

    def test_run_binding_rejects_shrunken_or_changed_preflight(self):
        preflight = {"status": "passed", "purpose": runner.PURPOSE,
            "mode": "outcome_blind_preflight", "dataset_split": "val",
            "requested_videos": 12,
            "effective_videos": 12, "cohort_sha256": "a", "cohort_lock_sha256": "b",
            "source_snapshot": {"x": 1}, "dataset_sha256": {"d": "e"},
            "exclusion_manifest_sha256": "f", "model_config_sha256": "g",
            "package_versions": {"torch": "x"},
            "candidate_manifest_digest_sha256": "h",
            "candidate_manifest_count": 4,
            "weight_shards_sha256": {name: str(i) * 64 for i, name in
                                      enumerate(runner.WEIGHT_SHARD_NAMES, 1)},
            "weight_shards_digest_sha256": "i",
            "weight_shard_count": 5}
        kwargs = dict(args_n=12, cohort_hash="a", lock_sha256="b", source={"x": 1},
            data_hashes={"d": "e"}, exclusion_sha256="f",
            model_config_sha256="g", versions={"torch": "x"},
            dataset_split="val", mode="smoke",
            candidate_manifest_digest="h", candidate_manifest_count=4,
            weight_shard_manifest={"shards_sha256": preflight["weight_shards_sha256"],
                "shards_digest_sha256": "i", "shard_count": 5})
        runner.validate_run_binding(preflight, **kwargs)
        with self.assertRaisesRegex(ValueError, "outcome_blind_preflight_binding_mismatch"):
            runner.validate_run_binding(preflight, **{**kwargs, "dataset_split": "test"})
        preflight["effective_videos"] = 11
        with self.assertRaisesRegex(ValueError, "outcome_blind_preflight_binding_mismatch"):
            runner.validate_run_binding(preflight, **kwargs)

    def test_summary_is_descriptive_and_preserves_every_pair(self):
        self.assertEqual(runner.summarize((-1.0, 0.0, 2.0)),
                         {"n": 3, "mean": 1 / 3, "median": 0.0,
                          "min": -1.0, "max": 2.0})

    def test_batch_digest_is_stable_and_handles_bfloat16_tensors(self):
        batch = {"input_ids": torch.tensor([[1, 2, 3]]),
                 "pixel_values_videos": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
        first = runner._batch_digest(batch, torch)
        self.assertEqual(first, runner._batch_digest(batch, torch))
        batch["input_ids"][0, 0] = 9
        self.assertNotEqual(first, runner._batch_digest(batch, torch))


if __name__ == "__main__":
    unittest.main()
