from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from freeze_qta_question_swap_cohort import (
    build_plan, csv_question_rows, write_private_exclusive,
)


class QuestionSwapFreezeTests(unittest.TestCase):
    def _fixture(self, root: Path):
        repo = root / "repo"
        dataset = repo / "NExT-GQA/datasets/nextgqa"
        data = repo / "data"
        private = data / "private_nextgqa"
        video_root = root / "videos"
        dataset.mkdir(parents=True)
        data.mkdir()
        private.mkdir()
        video_root.mkdir()
        (dataset / "map_vid_vidorID.json").write_text(
            json.dumps({"v1": "foo/v1.mp4", "v2": "foo/v2.mp4"}))
        with (dataset / "test.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["video_id", "qid", "question", "answer", "gsub"])
            writer.writerows([
                ["v1", "q1", "What moves?", "gold-a", "secret-a"],
                ["v1", "q2", "Where is it?", "gold-b", "secret-b"],
                ["v1", "q3", "What moves?", "gold-c", "secret-c"],
                ["v2", "q4", "What color?", "gold-d", "secret-d"],
                ["v2", "q5", "What is nearby?", "gold-e", "secret-e"],
            ])
        (video_root / "foo").mkdir()
        (video_root / "foo/v1.mp4").write_bytes(b"video-one")
        (video_root / "foo/v2.mp4").write_bytes(b"video-two")
        fake_hashes = [f"{index:064x}" for index in range(1, 638)]
        for name in ("nextgqa_tlvs_exploratory_split_v2_20261018_05.json",
                     "nextgqa_tlvs_exploratory_split_v4_union_20261018_03.json"):
            (private / name).write_text(json.dumps({"evaluation": [
                {"media_sha256": value} for value in fake_hashes[:147]
            ]}))
        (private / "qta_visual_overlap_pool_freeze_20261018.json").write_text(
            json.dumps({"expected_videos": 490, "entries": [
                {"sha256": value} for value in fake_hashes[147:637]
            ]}))
        return repo, dataset, video_root

    def test_csv_reader_hashes_only_declared_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = root / "a.csv"
            b = root / "b.csv"
            for path, answer in ((a, "gold-one"), (b, "gold-two")):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["video_id", "qid", "question", "answer"])
                    writer.writerow(["v", "q", "Where now?", answer])
            rows_a, digest_a = csv_question_rows(a)
            rows_b, digest_b = csv_question_rows(b)
            self.assertEqual(rows_a, rows_b)
            self.assertEqual(digest_a, digest_b)
            self.assertNotIn("answer", str(rows_a))

    def test_plan_selects_distinct_question_text_without_answer_or_gsub(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, video_root = self._fixture(Path(temp))
            plan = build_plan(repo=repo, dataset=dataset, video_root=video_root,
                              requested_videos=2)
            self.assertEqual(plan["effective_videos"], 2)
            self.assertTrue(plan["new_question_eligibility_and_pair_selection_use_only_question_columns"])
            self.assertFalse(plan["historical_repeatability_exclusion_used_annotation_validity"])
            for row in plan["fixed_records"]:
                self.assertNotEqual(row["question_a"]["question"],
                                    row["question_b"]["question"])
                self.assertNotIn("answer", str(row))
                self.assertNotIn("gsub", str(row))

    def test_explicit_exclusion_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, video_root = self._fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "eligible_video_count_below_requested_no_replacement"):
                build_plan(repo=repo, dataset=dataset, video_root=video_root,
                           requested_videos=2, exclude_keys={"foo/v1"})

    def test_same_questions_under_distinct_qids_do_not_make_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, video_root = self._fixture(Path(temp))
            path = dataset / "test.csv"
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["v2", "q6", "What color?", "gold-f", "secret-f"])
            with self.assertRaisesRegex(ValueError, "eligible_video_count_below_requested_no_replacement"):
                build_plan(repo=repo, dataset=dataset, video_root=video_root,
                           requested_videos=2, exclude_keys={"foo/v1"})

    def test_private_writer_uses_0600_and_refuses_overwrite_or_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            output = root / "candidate.json"
            write_private_exclusive(output, root, {"ok": True})
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_exclusive(output, root, {"ok": False})
            target = root / "target.json"
            target.write_text("{}")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                write_private_exclusive(link, root, {"ok": False})


if __name__ == "__main__":
    unittest.main()
