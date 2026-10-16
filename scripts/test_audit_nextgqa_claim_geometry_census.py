import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audit_nextgqa_claim_geometry_census import build_summary
import run_nextgqa_gold_interval_margin_pilot as pilot


class ClaimGeometryCensusTests(unittest.TestCase):
    def test_summary_is_aggregate_only_and_binds_inputs_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
            dataset.mkdir(parents=True)
            for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json"):
                (dataset / name).write_text("synthetic", encoding="utf-8")
            (repo / "data").mkdir()
            (repo / "data" / "nextgqa_samples_fixture.json").write_text("[]", encoding="utf-8")
            exclusion = repo / "exclude.json"
            exclusion.write_text(json.dumps({"schema_version": 1, "media_sha256": ["a" * 64]}))
            selected = [{"decode": {"source_sha256": "b" * 64}} for _ in range(2)]
            counts = {
                "candidate_scan_count": 3, "timing_pass_videos": 2,
                "timing_rejected_videos": 0, "decode_failures": 0,
                "videos_without_interval_control_pair": 0,
                "prior_cohort_media_excluded": 1,
                "questions_tested_for_assignment": 4,
            }
            with mock.patch.object(pilot, "freeze_cohort", return_value=(
                    selected, counts, {"candidate_video_count": 3}, "c" * 64)):
                result = build_summary(repo, Path(tmp), exclusion)
            self.assertEqual(result["candidate_count"], 3)
            self.assertEqual(result["candidate_rows_examined"], 3)
            self.assertEqual(result["candidates_excluded_before_decode"], 1)
            self.assertEqual(result["decode_attempts"], 2)
            self.assertEqual(result["unexamined_candidate_rows"], 0)
            self.assertEqual(result["eligible_geometry_cohort"], 2)
            self.assertEqual(result["prior_claim_media_excluded"], 1)
            self.assertEqual(result["distinct_selected_media"], 1)
            self.assertEqual(result["faithfulness_scores"], 0)
            self.assertEqual(len(result["dataset_inputs_sha256"]), 4)
            self.assertEqual(result["selector_sha256"], hashlib.sha256(
                Path(pilot.__file__).read_bytes()).hexdigest())
            raw = json.dumps(result)
            self.assertNotIn("source_sha256\": \"b", raw)
            self.assertNotIn("video_path", raw)
            self.assertNotIn("question_text", raw)
            self.assertNotIn("qid", raw)

    def test_rejects_inconsistent_scan_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exclusion = root / "exclude.json"
            exclusion.write_text(json.dumps({"schema_version": 1, "media_sha256": []}))
            counts = {
                "candidate_scan_count": 2, "timing_pass_videos": 1,
                "timing_rejected_videos": 0, "decode_failures": 0,
                "videos_without_interval_control_pair": 0,
                "prior_cohort_media_excluded": 0,
                "questions_tested_for_assignment": 1,
            }
            with mock.patch.object(pilot, "freeze_cohort", return_value=(
                    [], counts, {"candidate_video_count": 2}, "c" * 64)):
                with self.assertRaisesRegex(ValueError, "candidate_scan_accounting_inconsistent"):
                    build_summary(root, root, exclusion)

    def test_rejects_duplicate_or_malformed_private_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exclusion = root / "exclude.json"
            exclusion.write_text(json.dumps({"schema_version": 1,
                                             "media_sha256": ["a" * 64, "a" * 64]}))
            with self.assertRaisesRegex(ValueError, "private_media_exclusion_manifest_invalid"):
                build_summary(root, root, exclusion)


if __name__ == "__main__":
    unittest.main()
