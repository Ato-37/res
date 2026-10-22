import csv
import json
import tempfile
import unittest
from pathlib import Path

from audit_nextgqa_native_interval_coverage import summarize, summarize_packet_frame_overlap


class NativeIntervalCoverageTests(unittest.TestCase):
    def test_exact_join_and_invalid_boundaries_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, json_path = root / "val.csv", root / "gsub_val.json"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["video_id", "qid"])
                writer.writeheader()
                writer.writerows([
                    {"video_id": "v1", "qid": "1"},
                    {"video_id": "v1", "qid": "2"},
                ])
            json_path.write_text(json.dumps({"v1": {
                "duration": 10,
                "location": {"1": [[1, 3], [8, 11]], "2": [[-1, 2]]},
            }}))
            result = summarize(csv_path, json_path)
            self.assertEqual(result["exact_csv_qa_rows_with_intervals"], 2)
            self.assertEqual(result["csv_qa_rows_without_intervals"], 0)
            self.assertEqual(result["alternative_interval_count"], 3)
            self.assertEqual(result["interval_validation_counts"], {
                "valid_positive_interval": 1,
                "negative_start": 1,
                "end_after_video_duration": 1,
            })

    def test_duplicate_csv_qa_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, json_path = root / "val.csv", root / "gsub_val.json"
            csv_path.write_text("video_id,qid\nv1,1\nv1,1\n")
            json_path.write_text(json.dumps({"v1": {"duration": 10, "location": {"1": [[1, 2]]}}}))
            with self.assertRaisesRegex(ValueError, "duplicate_split_video_qid"):
                summarize(csv_path, json_path)

    def test_boolean_is_not_accepted_as_a_numeric_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, json_path = root / "val.csv", root / "gsub_val.json"
            csv_path.write_text("video_id,qid\nv1,1\n")
            json_path.write_text(json.dumps({"v1": {"duration": 10, "location": {"1": [[True, 2]]}}}))
            result = summarize(csv_path, json_path)
            self.assertEqual(result["interval_validation_counts"], {"malformed": 1})

    def test_packet_frame_overlap_returns_only_counts(self):
        result = summarize_packet_frame_overlap([{
            "gold_interval_seconds": [[1, 2], [4, 5]],
            "sampled_frame_relative_pts_seconds": [0, 1, 2.5, 4.5],
        }])
        self.assertEqual(result, {
            "review_items": 1,
            "qa_reference_intervals": 2,
            "sampled_frame_timestamps": 4,
            "timestamps_inside_qa_interval_union": 2,
            "items_with_at_least_one_inside_timestamp": 1,
        })


if __name__ == "__main__":
    unittest.main()
