import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from require_execution_preflight import require_execution_preflight


def passing_report():
    return {
        "schema_version": 1,
        "purpose": "claim_mask_structural_cross_gate_preflight",
        "structural_checks_status": "passed",
        "overall_status": "ready",
        "execution_authorization": "granted",
        "blocker_codes": [],
        "components": {name: {"status": "passed", "failure_codes": []}
                       for name in ("protocol", "claim_review", "item_rights",
                                    "data_rights_evidence")},
    }


class RequireExecutionPreflightTests(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(text, encoding="utf-8")
            return require_execution_preflight(path)

    def test_refuses_fabricated_ready_report_until_authenticated_gate_exists(self):
        encoded = json.dumps(passing_report(), sort_keys=True) + "\n"
        with self.assertRaisesRegex(ValueError, "never grants execution"):
            self.check(encoded)

    def test_current_blocked_not_granted_cross_gate_is_rejected(self):
        report = passing_report()
        report["overall_status"] = "blocked"
        report["execution_authorization"] = "not_granted"
        report["blocker_codes"] = ["execution_authorization_not_granted"]
        with self.assertRaisesRegex(ValueError, "never grants execution"):
            self.check(json.dumps(report))

    def test_rejects_duplicate_keys_nonfinite_and_wrong_purpose(self):
        bad_documents = (
            '{"schema_version":1,"schema_version":1}',
            '{"value":NaN}',
            json.dumps({**passing_report(), "purpose": "unrelated"}),
        )
        for document in bad_documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.check(document)

    def test_rejects_unsupported_future_schema_until_implemented(self):
        report = passing_report()
        report["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported execution preflight schema"):
            self.check(json.dumps(report))

    def test_direct_scorer_stops_on_blocked_report_before_reading_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report = passing_report()
            report["overall_status"] = "blocked"
            report["execution_authorization"] = "not_granted"
            report["blocker_codes"] = ["execution_authorization_not_granted"]
            report_path = directory / "blocked.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "scripts/necessity_choice_margin_test.py",
                 "--samples-file", str(directory / "does_not_exist.json"),
                 "--preflight-report", str(report_path), "--output",
                 str(directory / "out.json")],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("execution preflight blocked", result.stderr)
            self.assertNotIn("does_not_exist.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
