from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_minerva_vista_fresh_cohort import sha, verify_row


def _hook():
    return {"hook_call_count": 1, "hook_registry_before": 0, "hook_registry_after_removal": 0,
            "effective_displacement_fraction_of_pre_rms": 0.0998}


def _receipts(cohort=b"c", preflight=b"p"):
    cpu = {"status": "completed_vista_residual_cpu_preflight", "generation_mode": "fresh_in_run_greedy",
           "row_index": 1, "row_key_sha256": "r", "video_sha256": "v", "time_map_sha256": "t",
           "model_revision": "m", "source_sha256": {}, "runner_sha256": "x",
           "dose_rms_fraction": 0.1, "fixed_block_index_zero_based": 13}
    import json
    cpu_raw = json.dumps(cpu).encode()
    gpu = dict(cpu, cohort_sha256=sha(cohort), cohort_cpu_preflight_sha256=sha(preflight),
               cpu_receipt_sha256=sha(cpu_raw), object_name_token_count=1, target_token_count=12,
               mean_object_name_logprob_nats_per_token={"baseline": -0.5, "anchor_steering": -0.4,
                                                        "temporal_sham_steering": -0.45},
               delta_anchor_minus_baseline_nats_per_token=0.1,
               delta_sham_minus_baseline_nats_per_token=0.05,
               anchor_minus_sham_change_nats_per_token=0.05,
               capture_hook_audit={"baseline": _hook()}, score_hook_audit={"anchor_steering": _hook()},
               peak_vram_bytes=2**30)
    return gpu, cpu, cpu_raw


class VerifyRowTests(unittest.TestCase):
    def test_valid_row_recomputes_contrast(self):
        gpu, cpu, cpu_raw = _receipts()
        out = verify_row(b"", gpu, cpu_raw, cpu, b"c", b"p", False)
        self.assertAlmostEqual(out["anchor_minus_sham"], 0.05)
        self.assertAlmostEqual(out["delta_anchor"], 0.1)

    def test_tampered_contrast_rejected(self):
        gpu, cpu, cpu_raw = _receipts()
        gpu["anchor_minus_sham_change_nats_per_token"] = 0.2
        with self.assertRaisesRegex(ValueError, "arithmetic_mismatch"):
            verify_row(b"", gpu, cpu_raw, cpu, b"c", b"p", False)

    def test_wrong_cohort_binding_rejected(self):
        gpu, cpu, cpu_raw = _receipts()
        with self.assertRaisesRegex(ValueError, "binding_mismatch"):
            verify_row(b"", gpu, cpu_raw, cpu, b"other", b"p", False)

    def test_non_fresh_cpu_receipt_rejected(self):
        gpu, cpu, cpu_raw = _receipts()
        cpu["generation_mode"] = "prior_receipt_bound"
        with self.assertRaisesRegex(ValueError, "not_fresh_mode"):
            verify_row(b"", gpu, cpu_raw, cpu, b"c", b"p", False)

    def test_hook_cleanup_failure_rejected(self):
        gpu, cpu, cpu_raw = _receipts()
        gpu["score_hook_audit"]["anchor_steering"]["hook_registry_after_removal"] = 1
        with self.assertRaisesRegex(ValueError, "hook_audit_failed"):
            verify_row(b"", gpu, cpu_raw, cpu, b"c", b"p", False)


if __name__ == "__main__":
    unittest.main()
