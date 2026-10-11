"""Ensure the CLEVRER audit's publishable report contains counts, not source text."""

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_clevrer_counterfactual_programs import audit_payload, safe_report  # noqa: E402


class SafeClevrerAuditReportTests(unittest.TestCase):
    def test_report_excludes_raw_content_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.json"
            source.write_text("[]", encoding="utf-8")
            payload = [{
                "scene_index": 3,
                "questions": [{
                    "question": "PRIVATE QUESTION",
                    "question_type": "counterfactual",
                    "choices": [{"program": [
                        {"module_name": "PRIVATE PROGRAM", "value_inputs": ["PRIVATE ARGUMENT"]}
                    ]}],
                    "answer": "PRIVATE ANSWER",
                }],
            }]
            report = safe_report([(source, audit_payload(payload))])
        rendered = str(report)
        self.assertIn("counterfactual_count_by_type_field", rendered)
        self.assertIn("counterfactual_program_schema", rendered)
        self.assertIn("module_name:str", rendered)
        self.assertIn("sha256", rendered)
        for secret in (
            "PRIVATE QUESTION", "PRIVATE PROGRAM", "PRIVATE ARGUMENT",
            "PRIVATE ANSWER", directory,
        ):
            self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
