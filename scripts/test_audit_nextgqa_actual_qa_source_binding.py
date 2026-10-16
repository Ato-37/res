import json
from pathlib import Path
import tempfile
import unittest

from audit_nextgqa_actual_qa_source_binding import summarize
from test_nextgqa_actual_qa_source_adapter import make_inputs


class ActualSourceBindingAuditTests(unittest.TestCase):
    def test_aggregate_summary_omits_private_rows(self):
        raw, selection, _, lineage, _ = make_inputs()
        worksheet_rows = []
        for row in selection["records"]:
            index = int(row["dataset_question_id"])
            worksheet_rows.append({"video_id": row["video_id"],
                                   "question": f"question {index}?"})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "test.csv"
            lineage_path = root / "lineage.json"
            worksheet_path = root / "worksheet.json"
            csv_path.write_bytes(raw)
            lineage_path.write_bytes(lineage)
            worksheet_path.write_text(json.dumps({"records": worksheet_rows}))
            result = summarize(csv_path, lineage_path, worksheet_path)
        self.assertEqual(result["status"], "local_source_binding_passed_unvalidated")
        self.assertEqual(result["csv_rows"], 3)
        self.assertEqual(result["question_matches_worksheet"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        encoded = json.dumps(result)
        self.assertNotIn("ds-0", encoded)
        self.assertNotIn("claim-0", encoded)


if __name__ == "__main__":
    unittest.main()
