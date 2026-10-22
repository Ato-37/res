import unittest
from pathlib import Path

import run_nextgqa_validation_qta_annotation_proxy as kernel
from run_nextgqa_qta_v4_union_scorer import (
    CHOICE_NAMES, EXPECTED_N, EXPECTED_FORWARDS, assert_forward_denominator,
    frozen_media_sha256, validate_preflight, validate_smoke_receipt,
)
from run_nextgqa_qta_v4_union_geometry_census import canonical_sha256
from run_nextgqa_qta_v4_union_geometry_census import SOURCE_FILES as CENSUS_SOURCES
from run_nextgqa_qta_v4_union_scorer import SOURCE_FILES as SCORER_SOURCES


class QtaV4ScorerGateTests(unittest.TestCase):
    def test_media_hash_comes_from_frozen_plan_not_derived_item(self):
        digest = "a" * 64
        plan = {"evaluation": [{"video_id": "vid", "media_sha256": digest,
                                "media_path": "/tmp/vid.mp4"}]}
        derived_item = {"video_path": Path("/tmp/vid.mp4")}
        self.assertNotIn("media_sha256", derived_item)
        self.assertEqual(frozen_media_sha256(plan, "vid"), digest)
        with self.assertRaisesRegex(ValueError, "v4_frozen_media_hash_missing_or_duplicate"):
            frozen_media_sha256({"evaluation": []}, "vid")
        with self.assertRaisesRegex(ValueError, "v4_frozen_media_hash_missing_or_duplicate"):
            frozen_media_sha256({"evaluation": [
                {"video_id": "vid", "media_sha256": digest},
                {"video_id": "vid", "media_sha256": digest}]}, "vid")

    def test_transitive_runtime_sources_are_hash_bound(self):
        source_names = {Path(name).stem for name in SCORER_SOURCES}
        required = {
            "run_nextgqa_gold_interval_margin_pilot", "claim_mask_video_decoder",
            "qwen_review_interval_token_groups", "qwen_question_temporal_attention_selector",
            "qwen_internal_attention_ablation", "audit_qwen_checkpoint_bytes_cpu",
            "qwen_visual_attention_boost",
            "fixed_prediction_selective_dependence", "presentation_time_utils",
            "necessity_choice_margin_utils", "nextgqa_atomic_batch_freeze",
        }
        self.assertTrue(required <= source_names)
        self.assertTrue(set(CENSUS_SOURCES) <= set(SCORER_SOURCES))

    def test_exact_588_forward_contract_and_all_four_branch_denominators(self):
        counts = {name: {"requested": EXPECTED_N, "attempted": EXPECTED_N,
                         "completed": EXPECTED_N} for name in CHOICE_NAMES}
        assert_forward_denominator(counts, EXPECTED_N)
        self.assertEqual(sum(row["completed"] for row in counts.values()), EXPECTED_FORWARDS)
        counts["intervention_union_mask"]["completed"] -= 1
        with self.assertRaisesRegex(ValueError, "v4_exact_four_forward_denominator_failed"):
            assert_forward_denominator(counts, EXPECTED_N)

    def test_cpu_receipt_must_match_frozen_provenance_and_full_unique_denominator(self):
        expected = {"status": "complete", "plan_sha256": "a" * 64}
        rows = [{"video_id": f"v{i}", "qid": f"q{i}"} for i in range(EXPECTED_N)]
        validate_preflight({**expected, "geometry": rows}, expected)
        with self.assertRaisesRegex(ValueError, "v4_cpu_preflight_receipt_mismatch"):
            validate_preflight({**expected, "plan_sha256": "b" * 64,
                                "geometry": rows}, expected)
        duplicate = [*rows[:-1], rows[0]]
        with self.assertRaisesRegex(ValueError, "v4_cpu_preflight_geometry_denominator_invalid"):
            validate_preflight({**expected, "geometry": duplicate}, expected)

    def test_gpu_smoke_binds_processor_data_and_first_row_geometry(self):
        source = {"code.py": "a" * 64}; data = {"test.csv": "b" * 64}
        processor = {"processor_fingerprint_sha256": "c" * 64}
        geometry = {"video_id": "v", "qid": "q", "geometry_sha256": "d" * 64,
                    "row_binding_sha256": "e" * 64}
        full = {"status": "preflight", "mode": "cohort_geometry", "requested_n": 147,
            "completed_n": 147, "plan_sha256": "f" * 64, "census_sha256": "1" * 64,
            "source_sha256": source, "data_sha256": data,
            "processor_manifest": processor, "library_versions": {}, "scope": "geometry",
            "geometry": [geometry]}
        smoke_preflight = {key: value for key, value in full.items() if key != "geometry"}
        smoke_preflight.update({"requested_n": 1, "completed_n": 1, "geometry": [geometry]})
        smoke = {"status": "v4_one_row_four_forward_gpu_smoke_complete",
            "plan_sha256": full["plan_sha256"], "source_sha256": source,
            "census_sha256": full["census_sha256"],
            "data_sha256": data, "processor_manifest": processor,
            "checkpoint_byte_audit": {"revision": kernel.MODEL_REVISION,
                "status": "verified_current_local_bytes"},
            "model_revision": kernel.MODEL_REVISION, "precision": "4bit_bf16_compute_bf16_eager",
            "seed": 20261018, "requested_n": 1, "effective_n": 1,
            "requested_forwards": 4, "attempted_forwards": 4, "completed_forwards": 4,
            "preflight_requested": 1, "preflight_completed": 1,
            "branch_counts": {name: {"requested": 1, "attempted": 1, "completed": 1}
                for name in CHOICE_NAMES},
            "preflight_sha256": canonical_sha256(smoke_preflight),
            "records": [{**geometry, "baseline_full_correct": 1}]}
        validate_smoke_receipt(smoke, plan_sha256=full["plan_sha256"], sources=source,
            data_hashes=data, processor_manifest=processor, full_preflight=full,
            checkpoint_byte_audit=smoke["checkpoint_byte_audit"],
            first_plan_row={**geometry, "selected_row_binding_sha256": geometry["row_binding_sha256"]})
        first_row = {**geometry, "selected_row_binding_sha256": geometry["row_binding_sha256"]}
        tampered = [
            None,
            {**smoke, "requested_n": 2},
            {**smoke, "effective_n": 2},
            {**smoke, "requested_forwards": 5},
            {**smoke, "attempted_forwards": 3},
            {**smoke, "completed_forwards": 3},
            {**smoke, "preflight_requested": 2},
            {**smoke, "preflight_completed": 0},
            {**smoke, "branch_counts": {**smoke["branch_counts"],
                "baseline_full": {"requested": 1, "attempted": 1, "completed": 0}}},
            {**smoke, "checkpoint_byte_audit": {"revision": "different"}},
            {**smoke, "census_sha256": "0" * 64},
            {key: value for key, value in smoke.items() if key != "census_sha256"},
            {**smoke, "data_sha256": {"test.csv": "x" * 64}},
            {**smoke, "records": [{**smoke["records"][0], "row_binding_sha256": "0" * 64}]},
            {**smoke, "records": [None]},
        ]
        for key in ("requested_n", "effective_n", "requested_forwards",
                    "attempted_forwards", "completed_forwards",
                    "preflight_requested", "preflight_completed"):
            tampered.append({**smoke, key: True})
        for name in CHOICE_NAMES:
            for field in ("requested", "attempted", "completed"):
                bad_branches = {key: dict(value) for key, value in
                                smoke["branch_counts"].items()}
                bad_branches[name][field] = True
                tampered.append({**smoke, "branch_counts": bad_branches})
        tampered.extend([
            {**smoke, "branch_counts": {name: counts for name, counts in
                smoke["branch_counts"].items() if name != "baseline_full"}},
            {**smoke, "branch_counts": {**smoke["branch_counts"], "extra": {
                "requested": 1, "attempted": 1, "completed": 1}}},
        ])
        for bad in tampered:
            with self.subTest(bad=bad is not None), self.assertRaisesRegex(
                    ValueError, "v4_gpu_smoke_receipt_missing_or_mismatched"):
                validate_smoke_receipt(bad, plan_sha256=full["plan_sha256"], sources=source,
                    data_hashes=data, processor_manifest=processor,
                    checkpoint_byte_audit=smoke["checkpoint_byte_audit"],
                    full_preflight=full, first_plan_row=first_row)


if __name__ == "__main__":
    unittest.main()
