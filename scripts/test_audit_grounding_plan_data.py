import json
from pathlib import Path
import tempfile
import unittest

from audit_grounding_plan_data import (
    audit_split, audit_time_grid, canonical_video_key, normalize_intervals, interval_value_shape, _shape_summary,
)


class GroundingPlanAuditTests(unittest.TestCase):
    def test_canonical_video_key_hides_prefix_and_extension(self):
        self.assertEqual(canonical_video_key(r"/private/root/123/abc.mp4"), "123/abc")
        self.assertEqual(canonical_video_key("abc.mp4"), "abc")

    def test_interval_validation(self):
        intervals, invalid = normalize_intervals([[0, 1.5], [2, 2], [-1, 3], [4, 3], ["x", 8], [1]])
        self.assertEqual(intervals, [(0.0, 1.5), (2.0, 2.0)])
        self.assertEqual(interval_value_shape([[1, 2], [3, 4]]), "list_of_pairs")
        self.assertEqual(interval_value_shape([1, 2]), "flat_pair")
        self.assertEqual(normalize_intervals([1, 2]), ([], 2))
        self.assertEqual(invalid, 4)

    def test_split_counts_and_video_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "split.csv"
            gsub_path = root / "gsub.json"
            csv_path.write_text(
                "video_id,qid,question,answer,type\n"
                "v1,q1,private question,private answer,action\n"
                "v1,q2,another private question,another answer,action\n"
                "v2,q3,q,a,unknown\n", encoding="utf-8"
            )
            gsub_path.write_text(json.dumps({
                "v1": {"duration": 10, "location": {"q1": [[1, 2]], "q2": [[3, 4], ["bad", 5]]}},
                "v2": {"duration": 0.5, "location": {"q3": [[0, 1]]}},
            }), encoding="utf-8")
            result = audit_split(csv_path, gsub_path, {"v1": "10/a", "v2": "10/b"}, {"10/a"})
            self.assertEqual(result["csv_rows"], 3)
            self.assertEqual(result["rows_with_gsub_key"], 3)
            self.assertEqual(result["rows_with_structurally_valid_intervals"], 2)
            self.assertEqual(result["rows_with_valid_intervals"], 1)
            self.assertEqual(result["rows_with_any_malformed_interval"], 1)
            self.assertEqual(result["rows_on_previously_used_videos"], 2)
            self.assertEqual(result["unique_eligible_annotated_videos_after_exclusion"], 0)
            self.assertNotIn("private question", json.dumps(result))

            self.assertEqual(result["rows_with_interval_ending_after_video_duration"], 1)
    def test_generic_mapping_shape_does_not_emit_values(self):
        summary = _shape_summary({"video-secret": [0.0, 0.5, 1.0]})
        self.assertEqual(summary["top_level_entries"], 1)
        self.assertEqual(summary["finite_numeric_value_count"], 3)
        self.assertEqual(summary["numeric_time_vector_count"], 1)
        self.assertEqual(summary["time_vector_length"]["min"], 3)
        self.assertEqual(summary["non_monotonic_time_vector_count"], 0)
        summary = _shape_summary({"v": [0.0, 0.5, 0.5, 0.4]})
        self.assertEqual(summary["non_monotonic_time_vector_count"], 1)
        self.assertEqual(summary["duplicate_adjacent_time_vector_count"], 1)
        self.assertNotIn("video-secret", json.dumps(summary))

    def test_time_grid_alignment_uses_exact_keys_and_hides_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "split.csv"
            gsub_path = root / "gsub.json"
            csv_path.write_text(
                "video_id,qid,question,answer,type\nv1,q1,private question,private answer,action\n",
                encoding="utf-8",
            )
            gsub_path.write_text(json.dumps({
                "v1": {"duration": 5, "location": {"q1": [[1, 2]]}},
            }), encoding="utf-8")
            result = audit_time_grid(csv_path, gsub_path, {"v1": [0, 1.5, 3]}, {"v1": "10/a"})
            self.assertEqual(result["csv_video_ids_with_all_three_metadata_keys"], 1)
            self.assertEqual(result["valid_gold_interval_count"], 1)
            self.assertEqual(result["intervals_with_any_sampled_timestamp"], 1)
            self.assertEqual(result["nearest_interval_midpoint_seconds"]["median"], 0)
            self.assertNotIn("v1", json.dumps(result))
            self.assertNotIn("private question", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
