from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
import platform

from analyze_qta_synthetic_event_construct import (
    AnalysisError, _estimate, _validate_full_receipt, analyze,
)
from qta_synthetic_event_construct_v1 import assignment_manifest, sha256_canonical
from run_qta_synthetic_event_construct_gpu import (
    MODEL_REVISION, _checkpoint_digest, _preflight_expected_source_subset, source_hashes,
)


PROTOCOL = Path(__file__).resolve().parents[1] / (
    "experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md")


def _receipt(*, constant: bool = False):
    groups = {}
    for onset in (4, 10, 18, 24):
        values = [2.0] * 10 if constant else [float(index + onset / 10) for index in range(10)]
        groups[str(onset)] = {
            "n_pairs": 10,
            "qta_delta_sum": sum(values),
            "qta_delta_sum_squares": sum(value * value for value in values),
            "sham_delta_sum": -sum(values),
            "sham_delta_sum_squares": sum(value * value for value in values),
        }
    shards = {f"shard-{index}.safetensors": {"size_bytes": index,
                                              "sha256": f"{index:064x}"}
              for index in range(1, 6)}
    sources = source_hashes(Path(__file__).resolve().parents[1])
    assignment_sha = sha256_canonical(
        assignment_manifest(python_version=platform.python_version()))
    return {
        "schema_version": 1,
        "status": "passed_full_synthetic_construct_cohort", "mode": "full",
        "purpose": "synthetic_qta_construct_check_not_faithfulness",
        "requested_pairs": 40, "effective_pairs": 40,
        "requested_clips": 80, "effective_clips": 80,
        "requested_forwards": 720, "attempted_forwards": 720,
        "completed_forwards": 720, "failure_count": 0,
        "cohort_included": True, "faithfulness_result": False,
        "model_revision": MODEL_REVISION,
        "precision": "4bit_bf16_compute_bf16_eager",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "source_sha256": sources,
        "assignment_manifest_sha256": assignment_sha,
        "external_smoke_seed_sha256": "a" * 64,
        "preflight_receipt_sha256": "b" * 64,
        "smoke_receipt_sha256": "c" * 64,
        "checkpoint_shards": shards,
        "checkpoint_sha256": _checkpoint_digest(shards),
        "model_config_sha256": "d" * 64,
        "source_commit": "e" * 40,
        "processor_sha256": "f" * 64,
        "tokenizer_sha256": "1" * 64,
        "runtime": {"python": platform.python_version()},
        "analysis_sufficient_statistics_by_onset": groups,
    }


def _upstream_receipts(receipt):
    expected_sources = source_hashes(Path(__file__).resolve().parents[1])
    preflight = {
        "schema_version": 1,
        "status": "passed_cpu_processor_preflight_no_model_forward",
        "purpose": "qta_synthetic_event_construct_processor_preflight",
        "requested_pairs": 40, "effective_pairs": 40,
        "requested_clips": 80, "effective_clips": 80,
        "model_forwards": 0, "model_weights_loaded": False, "gpu_used": False,
        "protocol_sha256": receipt["protocol_sha256"],
        "source_sha256": _preflight_expected_source_subset(expected_sources),
        "assignment_manifest_sha256": receipt["assignment_manifest_sha256"],
        "processor_sha256": receipt["processor_sha256"],
        "tokenizer_sha256": receipt["tokenizer_sha256"],
        "scope": "synthetic_processor_geometry_only_no_qta_overlap",
        "model_revision": MODEL_REVISION,
        "qta_selector_overlap_measured": False,
        "faithfulness_result": False,
        "authorization": "processor_preflight_only_no_scoring_authorization",
        "checks": {
            "paired_rendered_qa_equal": 40,
            "paired_choice_lists_equal": 40,
            "paired_processor_geometry_equal": 40,
            "clips_with_exactly_two_event_groups": 80,
            "clips_with_two_disjoint_nearest_controls": 80,
        },
        "design_balance": {
            "onset_pair_counts": [
                {"start_frame": onset, "pairs": 10} for onset in (4, 10, 18, 24)],
            "orientation_pair_counts_by_onset": [
                {"start_frame": onset, "circle_left": 5, "circle_right": 5}
                for onset in (4, 10, 18, 24)],
            "sampled_cohort_seeds_unique": 40,
            "orientation_rotation_combinations_per_onset": 10,
            "circle_target_correct_label_counts": {key: 8 for key in "ABCDE"},
            "triangle_target_correct_label_counts": {key: 8 for key in "ABCDE"},
        },
        "aggregate_processor_geometry": {
            "grid_counts": [{"video_grid_thw": [16, 26, 42], "clips": 80}],
        },
    }
    smoke = {
        "schema_version": 1,
        "status": "passed_external_engineering_smoke_only", "mode": "smoke",
        "requested_pairs": 1, "effective_pairs": 1,
        "requested_clips": 2, "effective_clips": 2,
        "requested_forwards": 18, "attempted_forwards": 18,
        "completed_forwards": 18, "failure_count": 0,
        "model_weights_loaded": True, "gpu_used": True,
        "cohort_included": False, "faithfulness_result": False,
        "model_revision": receipt["model_revision"],
        "source_sha256": expected_sources,
        "protocol_sha256": receipt["protocol_sha256"],
        "assignment_manifest_sha256": receipt["assignment_manifest_sha256"],
        "external_smoke_seed_sha256": receipt["external_smoke_seed_sha256"],
        "preflight_receipt_sha256": "b" * 64,
        "source_commit": receipt["source_commit"],
        "checkpoint_shards": receipt["checkpoint_shards"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "model_config_sha256": receipt["model_config_sha256"],
        "processor_sha256": receipt["processor_sha256"],
        "tokenizer_sha256": receipt["tokenizer_sha256"],
    }
    return preflight, smoke


class SyntheticQTAAnalysisTests(unittest.TestCase):
    def test_complete_receipt_produces_weighted_mean_and_stratified_welch_inference(self):
        receipt = _receipt()
        preflight, smoke = _upstream_receipts(receipt)
        groups = _validate_full_receipt(receipt, preflight=preflight,
                                        preflight_sha256="b" * 64,
                                        smoke=smoke, smoke_sha256="c" * 64)
        result = _estimate(groups, "qta")
        self.assertEqual(result["inference_status"],
                         "approximate_stratified_welch_satterthwaite_t")
        self.assertAlmostEqual(result["estimate"], sum(result["stratum_means"]) / 4)
        self.assertEqual(len(result["ci95"]), 2)
        self.assertLess(result["ci95"][0], result["estimate"])
        self.assertGreater(result["ci95"][1], result["estimate"])
        self.assertGreaterEqual(result["two_sided_p"], 0)
        self.assertLessEqual(result["two_sided_p"], 1)
        self.assertGreater(result["df"], 0)

    def test_zero_variance_reports_estimate_without_inventing_test(self):
        receipt = _receipt(constant=True)
        preflight, smoke = _upstream_receipts(receipt)
        result = _estimate(_validate_full_receipt(
            receipt, preflight=preflight, preflight_sha256="b" * 64,
            smoke=smoke, smoke_sha256="c" * 64), "qta")
        self.assertEqual(result["estimate"], 2.0)
        self.assertIsNone(result["ci95"])
        self.assertIsNone(result["two_sided_p"])
        self.assertIsNone(result["df"])

    def test_rejects_smoke_incomplete_denominator_and_inconsistent_squares(self):
        for mutate in (
            lambda row: row.update(schema_version=True),
            lambda row: row.update(mode="smoke"),
            lambda row: row.update(completed_forwards=719),
            lambda row: row["analysis_sufficient_statistics_by_onset"]["4"].update(n_pairs=9),
            lambda row: row["analysis_sufficient_statistics_by_onset"]["4"].update(
                qta_delta_sum_squares=0.0),
        ):
            candidate = _receipt()
            mutate(candidate)
            preflight, smoke = _upstream_receipts(candidate)
            with self.assertRaises(AnalysisError):
                _validate_full_receipt(candidate, preflight=preflight,
                                       preflight_sha256="b" * 64,
                                       smoke=smoke, smoke_sha256="c" * 64)

    def test_analysis_keeps_scope_and_result_hash(self):
        digest = hashlib.sha256(b"fixture receipt").hexdigest()
        receipt = _receipt()
        preflight, smoke = _upstream_receipts(receipt)
        report = analyze(receipt, digest, preflight=preflight,
                         preflight_sha256="b" * 64, smoke=smoke,
                         smoke_sha256="c" * 64)
        self.assertEqual(report["input_receipt_sha256"], digest)
        self.assertEqual(report["n_scene_pairs"], 40)
        self.assertFalse(report["faithfulness_result"])
        self.assertEqual(len(report["limitations"]), 3)
        self.assertEqual(report["secondary_sham_descriptive"]["inference"],
                         "not_performed_by_frozen_protocol")
        self.assertNotIn("ci95", report["secondary_sham_descriptive"])

    def test_rejects_unbound_source_and_overflowing_moments(self):
        candidate = _receipt()
        candidate["source_sha256"] = {"runner": "f" * 64}
        preflight, smoke = _upstream_receipts(candidate)
        with self.assertRaisesRegex(AnalysisError, "source_hash"):
            _validate_full_receipt(candidate, preflight=preflight,
                                   preflight_sha256="b" * 64,
                                   smoke=smoke, smoke_sha256="c" * 64)
        candidate = _receipt()
        candidate["analysis_sufficient_statistics_by_onset"]["4"].update(
            qta_delta_sum=1e308, qta_delta_sum_squares=1e308)
        preflight, smoke = _upstream_receipts(candidate)
        with self.assertRaisesRegex(AnalysisError, "moment_arithmetic_overflow"):
            _validate_full_receipt(candidate, preflight=preflight,
                                   preflight_sha256="b" * 64,
                                   smoke=smoke, smoke_sha256="c" * 64)

    def test_rejects_upstream_digest_or_smoke_source_commit_mismatch(self):
        receipt = _receipt()
        preflight, smoke = _upstream_receipts(receipt)
        with self.assertRaisesRegex(AnalysisError, "upstream_receipt_digest"):
            analyze(receipt, "a" * 64, preflight=preflight,
                    preflight_sha256="9" * 64, smoke=smoke, smoke_sha256="c" * 64)
        smoke["source_commit"] = "0" * 40
        with self.assertRaisesRegex(AnalysisError, "smoke_receipt_binding"):
            analyze(receipt, "a" * 64, preflight=preflight,
                    preflight_sha256="b" * 64, smoke=smoke, smoke_sha256="c" * 64)

    def test_rejects_smoke_without_model_or_gpu_execution(self):
        receipt = _receipt()
        preflight, smoke = _upstream_receipts(receipt)
        for field in ("model_weights_loaded", "gpu_used"):
            candidate = dict(smoke)
            candidate[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(
                    AnalysisError, "smoke_receipt_binding"):
                analyze(receipt, "a" * 64, preflight=preflight,
                        preflight_sha256="b" * 64, smoke=candidate,
                        smoke_sha256="c" * 64)


if __name__ == "__main__":
    unittest.main()
