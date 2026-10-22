import json
import contextlib
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import run_nextgqa_qta_exploratory_proxy_v2 as runner

from run_nextgqa_qta_exploratory_proxy_v2 import (
    aggregate_summary, baseline_accuracy, choice_gold_from_row, finite_cohort_delta_f_summary,
    failure_payload, geometry_preflight_receipt, validate_geometry_preflight_receipt,
    validate_preflight_plans, validate_scored_denominator, write_private_output,
    BRANCH_NAMES,
)


class ExploratoryRunnerV2Tests(unittest.TestCase):
    def test_exact_five_choice_gold_mapping(self):
        row = {f"a{i}": f"choice{i}" for i in range(5)}
        row["answer"] = "choice3"
        self.assertEqual(choice_gold_from_row(row), ([f"choice{i}" for i in range(5)], 3))

    def test_duplicate_choice_and_unmapped_gold_rejected(self):
        row = {f"a{i}": f"choice{i}" for i in range(5)}
        row["answer"] = "choice0"
        row["a4"] = "choice0"
        with self.assertRaisesRegex(ValueError, "evaluation_AE_choices_invalid"):
            choice_gold_from_row(row)
        row["a4"] = "choice4"; row["answer"] = "missing"
        with self.assertRaisesRegex(ValueError, "evaluation_gold_choice_mapping_invalid"):
            choice_gold_from_row(row)

    def test_preflight_requires_exact_requested_denominator_and_single_anchor(self):
        row = {"geometry_sha256": "a" * 64, "group_count": 5,
               "evidence_count": 1, "control_count": 1}
        validate_preflight_plans([row], 1)
        with self.assertRaisesRegex(ValueError, "single_anchor_preflight_denominator_invalid"):
            validate_preflight_plans([], 1)
        with self.assertRaisesRegex(ValueError, "single_anchor_preflight_denominator_invalid"):
            validate_preflight_plans([{**row, "control_count": 2}], 1)

    def test_scored_branch_counts_are_all_or_none(self):
        counts = {name: {"attempted": 1, "completed": 1} for name in BRANCH_NAMES}
        outputs = [{"delta_f": -0.1}]
        validate_scored_denominator(outputs, counts, 1)
        counts[BRANCH_NAMES[-1]]["completed"] = 0
        with self.assertRaisesRegex(ValueError, "all_or_none_six_branch_denominator_failed"):
            validate_scored_denominator(outputs, counts, 1)

    def test_accuracy_is_descriptive_and_denominator_bound(self):
        result = baseline_accuracy([{"baseline_full_correct": 1},
                                   {"baseline_full_correct": 0}], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertIn("not faithfulness", result["interpretation"])
        with self.assertRaisesRegex(ValueError, "baseline_accuracy_denominator_invalid"):
            baseline_accuracy([{"baseline_full_correct": 1}], 2)

    def test_finite_cohort_delta_summary_has_only_descriptive_sign_counts(self):
        summary = finite_cohort_delta_f_summary(
            [{"delta_f": 2.0}, {"delta_f": -1.0}, {"delta_f": 0.0}])
        self.assertEqual(summary["sign_counts"], {"positive": 1, "negative": 1, "zero": 1})
        self.assertEqual(summary["n"], 3)
        self.assertNotIn("p_value", summary)
        with self.assertRaisesRegex(ValueError, "finite_cohort_delta_f_invalid"):
            finite_cohort_delta_f_summary([{"delta_f": float("nan")}])

    def test_stdout_aggregate_excludes_raw_score_rows(self):
        payload = {"status": "ok", "requested_n": 1, "effective_n": 1,
                   "failures": 0, "preflight_completed": 1,
                   "completed_forwards": 6, "records": [{"question": "secret"}],
                   "plan_sha256": "private-fingerprint", "preflight_receipt_sha256": "secret-hash"}
        summary = aggregate_summary(payload, "out.json")
        serialized = json.dumps(summary)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("records", summary)
        self.assertNotIn("private-fingerprint", serialized)
        self.assertNotIn("secret-hash", serialized)

    def test_private_output_is_exclusive_and_mode_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); private = repo / "data/private_nextgqa"
            private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
            write_private_output(repo, "receipt.json", {"status": "test"})
            output = private / "receipt.json"
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_private_output(repo, "receipt.json", {"status": "replacement"})

    def test_negative_preflight_and_runtime_failures_write_private_incomplete_receipts(self):
        scenarios = (
            ("pinned_checkpoint_presence_check", FileNotFoundError),
            ("cuda_capability_check", ValueError),
            ("complete_requested_cpu_geometry_preflight", ValueError),
            ("source_and_receipt_preflight", ValueError),
        )
        for index, (stage, error_type) in enumerate(scenarios):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp); private = repo / "data/private_nextgqa"
                private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
                branches = {name: {"requested": 1, "attempted": 0, "completed": 0}
                            for name in BRANCH_NAMES}
                payload = failure_payload(requested_n=1, stage=stage,
                    exc=error_type("simulated_gate_failure"),
                    preflight_counts={"attempted": 1, "completed": 0},
                    branch_counts=branches, sources={}, plan_sha256=None, receipt_sha256=None)
                name = f"failure_{index}.json"
                write_private_output(repo, name, payload)
                saved = json.loads((private / name).read_text())
                self.assertEqual(saved["status"], "failed_incomplete_no_cohort_estimate")
                self.assertEqual(saved["effective_n"], 0)
                self.assertIsNone(saved["estimate"])
                self.assertEqual(stat.S_IMODE((private / name).stat().st_mode), 0o600)

    def test_full_cpu_geometry_receipt_is_non_authorizing_exact_and_private(self):
        gate = {"plan_sha256": "p" * 64, "preflight_receipt_sha256": "r" * 64,
                "v1_plan_sha256": "v" * 64, "v1_freeze_receipt_sha256": "f" * 64,
                "source_sha256": {"source.py": "s" * 64},
                "data_sha256": {"test.csv": "d" * 64}}
        plans = [{"geometry_sha256": "a" * 64, "group_count": 5,
                  "evidence_count": 1, "control_count": 1}]
        receipt = geometry_preflight_receipt(input_gate=gate, plans=plans, requested_n=1)
        self.assertIn("not_authorization_or_model_result", receipt["status"])
        validate_geometry_preflight_receipt(receipt, receipt.copy())
        with self.assertRaisesRegex(ValueError, "geometry_preflight_receipt_mismatch"):
            validate_geometry_preflight_receipt({**receipt, "geometry_plans": []}, receipt)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); private = repo / "data/private_nextgqa"
            private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
            write_private_output(repo, "geometry.json", receipt)
            saved = json.loads((private / "geometry.json").read_text())
            self.assertEqual(saved, receipt)
            self.assertEqual(stat.S_IMODE((private / "geometry.json").stat().st_mode), 0o600)

    def test_run_gate_failure_writes_private_receipt_and_aggregate_only_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); (repo / "scripts").mkdir()
            private = repo / "data/private_nextgqa"
            private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
            fake_script = repo / "scripts/run_nextgqa_qta_exploratory_proxy_v2.py"
            output = io.StringIO()
            with patch.object(runner, "__file__", str(fake_script)), \
                 patch.object(runner, "source_manifest", return_value=("test-commit", {})), \
                 patch.object(runner, "load_verified_plan_and_receipt",
                               side_effect=ValueError("private source gate mismatch")), \
                 contextlib.redirect_stdout(output):
                code = runner.run(["--n", "1", "--output-name", "failure.json",
                                   "--user-authorized-local-only"])
            self.assertEqual(code, 2)
            receipt_path = private / "failure.json"
            saved = json.loads(receipt_path.read_text())
            self.assertEqual(saved["status"], "failed_incomplete_no_cohort_estimate")
            self.assertEqual(saved["failure_stage"], "source_and_receipt_preflight")
            self.assertEqual(saved["failure_type"], "ValueError")
            self.assertEqual(saved["effective_n"], 0)
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
            self.assertIn('"failures": 1', output.getvalue())
            self.assertNotIn("private source gate mismatch", output.getvalue())
            self.assertNotIn("records", output.getvalue())


if __name__ == "__main__":
    unittest.main()
