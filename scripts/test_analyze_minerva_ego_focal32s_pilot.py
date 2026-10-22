import unittest

from analyze_minerva_ego_focal32s_pilot import summarize_results


def _result(key, support, baseline, sham):
    layers = {str(i): 1 for i in range(5, 17)}
    audit = {
        "baseline": {"biased_key_count": 0, "deleted_key_count": 0,
                     "layer_hook_calls": layers, "target_token_ids_sha256": "a" * 64},
        "object_time_support_bias": {"biased_key_count": 121, "deleted_key_count": 0,
                     "layer_hook_calls": layers, "target_token_ids_sha256": "a" * 64},
        "equal_dose_temporal_sham_bias": {"biased_key_count": 121, "deleted_key_count": 0,
                     "layer_hook_calls": layers, "target_token_ids_sha256": "a" * 64},
    }
    return {"row_key_sha256": key * 64, "cohort_sha256": "c" * 64,
        "runtime_source_sha256": {"run_minerva_ego_anchor_gpu_pilot.py": "r" * 64,
                                  "minerva_ego_anchor_utils.py": "h" * 64},
        "model_revision": "pinned", "status": "completed_single_video_exploratory_reference_likelihood_pilot",
        "forward_count": 3, "direct_generated_explanation_faithfulness": False,
        "human_review_performed": False, "scoring_audit": audit,
        "target_mean_logprob_nats_per_token": {"baseline": baseline,
            "object_time_support_bias": support, "equal_dose_temporal_sham_bias": sham},
        "support_minus_baseline": support-baseline, "support_minus_sham": support-sham,
        "elapsed_scoring_seconds": 1.0, "peak_vram_bytes": 10}


class FocalPilotAnalysisTests(unittest.TestCase):
    def test_recomputes_paired_descriptive_means(self):
        rows = [_result(c, 1.0, 0.5, 0.75) for c in "abc"]
        result = summarize_results(rows, ["d" * 64] * 3)
        self.assertEqual(result["forward_count"], 9)
        self.assertEqual(result["support_minus_baseline_mean_nats_per_token"], 0.5)
        self.assertEqual(result["support_minus_sham_mean_nats_per_token"], 0.25)
        self.assertIsNone(result["statistical_test"])

    def test_rejects_wrong_control_dose(self):
        rows = [_result(c, 1.0, 0.5, 0.75) for c in "abc"]
        rows[0]["scoring_audit"]["equal_dose_temporal_sham_bias"]["biased_key_count"] = 120
        with self.assertRaisesRegex(ValueError, "intervention_audit_invalid"):
            summarize_results(rows)


if __name__ == "__main__":
    unittest.main()
