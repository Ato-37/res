import unittest

from qta_phase_schedule_diagnostic import (
    BASELINE, POST_BASELINE, diagnose_qta_phase_schedule,
)
from qta_nine_branch_plan import BRANCH_ORDER


def fixture():
    p1, p2 = [], []
    for index in range(3):
        token = f"opaque_record_{index}"
        batch = f"{index + 1:064x}"
        p1.append({"record_token": token, "processor_batch_sha256": batch,
                   "branches": list(BASELINE)})
        p2.append({"record_token": token, "processor_batch_sha256": batch,
                   "branches": list(POST_BASELINE)})
    pins = {"phase1_receipt_sha256": "a" * 64,
            "baseline_manifest_sha256": "b" * 64,
            "stage2_review_receipt_sha256": "c" * 64}
    return p1, p2, pins


class QtaPhaseScheduleDiagnosticTests(unittest.TestCase):
    def assert_denied(self, report):
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["scoring_authorized"], False)
        self.assertIs(report["phase_authorized"], False)
        self.assertEqual(report["authorized_forwards"], 0)
        self.assertIs(report["signatures_verified"], False)
        self.assertIs(report["artifact_bytes_verified"], False)
        self.assertIs(report["execution_performed"], False)
        self.assertIs(report["counts_are_schedule_shape_only"], True)

    def test_exact_three_plus_twenty_four_schedule_passes_shape_only(self):
        self.assertEqual(BASELINE, BRANCH_ORDER[:1])
        self.assertEqual(POST_BASELINE, BRANCH_ORDER[1:])
        p1, p2, pins = fixture()
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["status"], "shape_valid_non_authorizing")
        self.assertEqual(report["requested_rows"], 3)
        self.assertEqual(report["phase1_schedule_branches"], 3)
        self.assertEqual(report["phase2_schedule_branches"], 24)
        self.assertEqual(report["total_schedule_branches"], 27)
        self.assertTrue(all(g["status"] == "passed_shape_only"
                            for g in report["gates"].values()))
        self.assert_denied(report)

    def test_missing_or_partial_rows_block_entire_schedule(self):
        p1, p2, pins = fixture()
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1[:2], phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["phase1_schedule_branches"], 0)
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assertEqual(report["gates"]["cross_phase_batch_binding"]["status"], "blocked")
        self.assert_denied(report)

    def test_baseline_cannot_be_smuggled_into_phase_two(self):
        p1, p2, pins = fixture()
        p2[0]["branches"][0] = "baseline_full"
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["phase2_branch_schedule"]["status"], "blocked")
        self.assertEqual(report["phase2_schedule_branches"], 0)
        self.assert_denied(report)

    def test_missing_or_duplicate_branch_blocks_phase(self):
        p1, p2, pins = fixture()
        p2[1]["branches"][-1] = p2[1]["branches"][0]
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["phase2_branch_schedule"]["failure_code"],
                         "branch_set_invalid")
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assert_denied(report)

    def test_reordered_phase2_branches_fail_the_frozen_sequence(self):
        p1, p2, pins = fixture()
        p2[0]["branches"][0], p2[0]["branches"][1] = (
            p2[0]["branches"][1], p2[0]["branches"][0])
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["phase2_branch_schedule"]["failure_code"],
                         "branch_set_invalid")
        self.assertEqual(report["phase2_schedule_branches"], 0)
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assert_denied(report)

    def test_processor_batch_must_be_identical_across_phases(self):
        p1, p2, pins = fixture()
        p2[2]["processor_batch_sha256"] = "f" * 64
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["cross_phase_batch_binding"]["failure_code"],
                         "phase_row_or_batch_binding_mismatch")
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assert_denied(report)

    def test_phase2_requires_three_well_formed_artifact_hashes(self):
        p1, p2, pins = fixture()
        pins["stage2_review_receipt_sha256"] = "not-a-digest"
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["phase2_artifact_binding_shape"]["status"], "blocked")
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assert_denied(report)

    def test_duplicate_record_token_blocks_phase(self):
        p1, p2, pins = fixture()
        p1[1]["record_token"] = p1[0]["record_token"]
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        self.assertEqual(report["gates"]["phase1_baseline_schedule"]["failure_code"],
                         "record_token_invalid_or_duplicate")
        self.assertEqual(report["total_schedule_branches"], 0)
        self.assert_denied(report)

    def test_report_never_echoes_record_tokens_or_input_digests(self):
        p1, p2, pins = fixture()
        report = diagnose_qta_phase_schedule(
            phase1_rows=p1, phase2_rows=p2, phase2_bindings=pins)
        serialized = repr(report)
        for value in [row["record_token"] for row in p1]:
            self.assertNotIn(value, serialized)
        for row in p1 + p2:
            self.assertNotIn(row["processor_batch_sha256"], serialized)
        for value in pins.values():
            self.assertNotIn(value, serialized)
        self.assert_denied(report)


if __name__ == "__main__":
    unittest.main()
