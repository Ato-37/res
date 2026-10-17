from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import csv

from smoke_qwen_real_video_attention_boost import (
    _temporal_visual_groups,
    _write_private_json,
    question_for_video,
)


class RealVideoAttentionBoostOutputTests(unittest.TestCase):
    def test_temporal_visual_groups_follow_processor_grid_and_fail_closed(self):
        token_ids = [99, 1, 1, 1, 1, 1, 1, 1, 1, 7, 2]
        groups = _temporal_visual_groups(
            token_ids=token_ids, video_token_id=1, temporal_steps=2,
            grid_h=4, grid_w=4, merge_size=2)
        self.assertEqual(groups, [(1, 2, 3, 4), (5, 6, 7, 8)])
        with self.assertRaisesRegex(ValueError, "geometry_mismatch"):
            _temporal_visual_groups(
                token_ids=token_ids, video_token_id=1, temporal_steps=3,
                grid_h=4, grid_w=1, merge_size=2)
        with self.assertRaisesRegex(ValueError, "not_contiguous"):
            _temporal_visual_groups(
                token_ids=[1, 9, 1, 1, 1, 1, 1], video_token_id=1,
                temporal_steps=1, grid_h=2, grid_w=3, merge_size=1)

    def test_question_selection_is_deterministic_and_requires_eligible_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, video_root = root / "dataset", root / "videos"
            dataset.mkdir()
            (video_root / "foo").mkdir(parents=True)
            video = video_root / "foo" / "a.mp4"
            video.touch()
            (dataset / "map_vid_vidorID.json").write_text(
                json.dumps({"v1": "foo/a.mp4"}), encoding="utf-8")
            (dataset / "gsub_test.json").write_text(json.dumps({
                "v1": {"duration": 10, "location": {
                    "q1": [[0, 2]], "q2": [[2, 4]], "q3": []}},
            }), encoding="utf-8")
            with (dataset / "test.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "video_id", "qid", "question", "answer"))
                writer.writeheader()
                writer.writerow({"video_id": "v1", "qid": "q2",
                                 "question": "Later question?", "answer": "secret"})
                writer.writerow({"video_id": "v1", "qid": "q3",
                                 "question": "No interval?", "answer": "secret"})
                writer.writerow({"video_id": "v1", "qid": "q1",
                                 "question": "First question?", "answer": "secret"})
            self.assertEqual(question_for_video(dataset, video, video_root),
                             "First question?")

    def test_refuses_a_video_without_an_eligible_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, video_root = root / "dataset", root / "videos"
            dataset.mkdir()
            video_root.mkdir()
            video = video_root / "a.mp4"
            video.touch()
            (dataset / "map_vid_vidorID.json").write_text(
                json.dumps({"v1": "a.mp4"}), encoding="utf-8")
            (dataset / "gsub_test.json").write_text(
                json.dumps({"v1": {"duration": 1, "location": {}}}),
                encoding="utf-8")
            with (dataset / "test.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("video_id", "qid", "question"))
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "eligible_question_unavailable"):
                question_for_video(dataset, video, video_root)

    def test_output_is_exclusive_private_and_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "private-run"
            output = parent / "summary.json"
            expected = {"status": "passed", "faithfulness_result": False}
            _write_private_json(output, expected)
            self.assertEqual(parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
            with self.assertRaises(FileExistsError):
                _write_private_json(output, {"status": "overwrite"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)

    def test_refuses_a_preexisting_public_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "shared"
            parent.mkdir(mode=0o755)
            os.chmod(parent, 0o755)
            output = parent / "summary.json"
            with self.assertRaisesRegex(
                    RuntimeError, "private_output_directory_permissions_invalid"):
                _write_private_json(output, {"status": "blocked"})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
