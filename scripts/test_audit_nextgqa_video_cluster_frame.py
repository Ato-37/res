from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from audit_nextgqa_video_cluster_frame import audit_split


class NextgqaVideoClusterFrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv_path = self.root / "test.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid", "question", "answer"])
            writer.writeheader()
            writer.writerows([
                {"video_id": "v1", "qid": "q1", "question": "PRIVATE QUESTION", "answer": "PRIVATE ANSWER"},
                {"video_id": "v1", "qid": "q2", "question": "PRIVATE QUESTION 2", "answer": "PRIVATE ANSWER 2"},
                {"video_id": "v2", "qid": "q1", "question": "PRIVATE QUESTION 3", "answer": "PRIVATE ANSWER 3"},
            ])
        self.gsub_path = self.root / "gsub.json"
        self.map_path = self.root / "map.json"
        self.gsub_path.write_text(json.dumps({
            "v1": {"location": {"q1": [[0, 1]], "q2": [[2, 3], [4, 5]]}},
            "v2": {"location": {"q1": [[1, 2]]}},
            "extra": {"location": {"q3": [[9, 10]]}},
        }))
        self.map_path.write_text(json.dumps({"v1": "a/video1.mp4", "v2": "b/video2.mp4"}))

    def test_aggregate_retains_video_cluster_denominator_and_only_counts(self):
        report = audit_split(self.csv_path, self.gsub_path, self.map_path)
        self.assertEqual(report["qa_rows"], 3)
        self.assertEqual(report["unique_test_video_ids"], 2)
        self.assertEqual(report["unique_test_video_question_pairs"], 3)
        self.assertEqual(report["test_video_ids_with_gsub"], 2)
        self.assertEqual(report["gsub_video_ids_outside_test"], 1)
        self.assertEqual(report["test_video_ids_missing_video_map"], 0)
        self.assertEqual(report["distinct_mapped_video_paths_for_test_ids"], 2)
        self.assertEqual(report["gsub_question_ids_for_test_pairs"], 3)
        self.assertEqual(report["gsub_intervals_all_ids"], 5)
        self.assertEqual(report["gsub_intervals_for_test_pairs"], 4)
        self.assertEqual(report["qa_rows_per_video_histogram"], {"1": 1, "2": 1})
        self.assertFalse(report["source_cluster_independence_verified"])
        self.assertFalse(report["review_eligibility_verified"])
        serialized = json.dumps(report)
        self.assertNotIn("PRIVATE QUESTION", serialized)
        self.assertNotIn("PRIVATE ANSWER", serialized)
        self.assertNotIn("v1", serialized)
        self.assertNotIn("video1.mp4", serialized)

    def test_duplicate_video_question_pair_is_counted_not_expanded(self):
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid", "question", "answer"])
            writer.writerow({"video_id": "v1", "qid": "q1", "question": "x", "answer": "y"})
        report = audit_split(self.csv_path, self.gsub_path, self.map_path)
        self.assertEqual(report["qa_rows"], 4)
        self.assertEqual(report["unique_test_video_question_pairs"], 3)
        self.assertEqual(report["duplicate_video_question_pairs"], 1)

    def test_missing_required_columns_rejected(self):
        self.csv_path.write_text("video_id,question\nv1,PRIVATE QUESTION\n")
        with self.assertRaisesRegex(ValueError, "test_csv_required_columns_missing"):
            audit_split(self.csv_path, self.gsub_path, self.map_path)


if __name__ == "__main__":
    unittest.main()
