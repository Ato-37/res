import importlib.util
import copy
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("summarize_minerva_vista_residual_smoke.py")
SPEC = importlib.util.spec_from_file_location("vista_summary", MODULE_PATH)
vista = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vista)


class SummaryVerifierTests(unittest.TestCase):
    def test_score_contrasts_are_recomputed_and_tampering_fails(self):
        receipt = {
            "mean_object_name_logprob_nats_per_token": {
                "baseline": -0.55, "anchor_steering": -0.53,
                "temporal_sham_steering": -0.49,
            },
            "delta_anchor_minus_baseline_nats_per_token": 0.02,
            "delta_sham_minus_baseline_nats_per_token": 0.06,
            "anchor_minus_sham_change_nats_per_token": -0.04,
        }
        result = vista._verify_score_arithmetic(receipt)
        self.assertAlmostEqual(result["contrasts"]["anchor_change_minus_sham_change"], -0.04)
        receipt["anchor_minus_sham_change_nats_per_token"] = 0.04
        with self.assertRaisesRegex(vista.VerificationError, "anchor_minus_sham_arithmetic_mismatch"):
            vista._verify_score_arithmetic(receipt)

    def test_hook_audit_rejects_leaked_or_repeated_hook(self):
        clean = {
            "capture_hook_audit": {
                key: {"hook_call_count": 1, "hook_registry_before": 4,
                      "hook_registry_after_removal": 4}
                for key in ("baseline", "anchor_pair_mean_fill", "temporal_sham_pair_mean_fill")
            },
            "score_hook_audit": {
                key: {"hook_call_count": 1, "hook_registry_before": 5,
                      "hook_registry_after_removal": 5}
                for key in ("anchor_steering", "temporal_sham_steering")
            },
        }
        vista._verify_hook_audit(clean)
        clean["score_hook_audit"]["anchor_steering"]["hook_call_count"] = 2
        with self.assertRaisesRegex(vista.VerificationError, "score_hook_count_or_cleanup_invalid"):
            vista._verify_hook_audit(clean)

    def test_summary_privacy_fence_rejects_raw_identifier(self):
        with self.assertRaisesRegex(vista.VerificationError, "privacy_literal_leak_detected"):
            vista._assert_privacy({"unsafe": "private-video-id"}, ("private-video-id",))

    def test_effective_dose_fraction_must_match_measured_rms_ratio(self):
        dose = {key: {"full_residual_rms_fp32": 1.0,
            "raw_full_minus_filled_rms_fp32": 0.02,
            "normalized_delta_rms_fp32": 0.1,
            "dose_fraction_of_full_residual_rms": 0.1,
            "target_delta_rms_fp32": 0.1}
            for key in ("anchor_steering", "temporal_sham_steering")}
        dose["anchor_sham_cosine_raw"] = 0.4
        hooks = {key: {"pre_update_rms_fp32": 1.0,
            "effective_displacement_rms_fp32": 0.1,
            "effective_displacement_fraction_of_pre_rms": 0.1}
            for key in ("anchor_steering", "temporal_sham_steering")}
        receipt = {"vector_dose_audit": dose, "score_hook_audit": hooks}
        vista._verify_dose_audit(receipt)
        changed = copy.deepcopy(receipt)
        changed["score_hook_audit"]["anchor_steering"][
            "effective_displacement_fraction_of_pre_rms"] = 0.101
        with self.assertRaisesRegex(vista.VerificationError,
                                    "effective_bf16_fraction_arithmetic_mismatch"):
            vista._verify_dose_audit(changed)


if __name__ == "__main__":
    unittest.main()
