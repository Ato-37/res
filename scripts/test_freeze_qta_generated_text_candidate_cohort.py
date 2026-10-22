from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from freeze_qta_generated_text_candidate_cohort import (
    build_candidate_lock,
    collect_known_local_media_hashes,
    write_private_exclusive,
)


class CandidateCohortFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        self.data = self.root / "dataset"
        self.data.mkdir()
        self.media = self.root / "videos"
        self.media.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def _inputs(self):
        (self.private / "past.json").write_text(json.dumps({
            "rows": [{"video_sha256": "a" * 64}, {"model_sha256": "b" * 64}]}))
        os.chmod(self.private / "past.json", 0o600)
        with (self.data / "test.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid"])
            writer.writeheader()
            writer.writerows([{"video_id": "v1", "qid": "q1"},
                              {"video_id": "v2", "qid": "q2"}])
        (self.data / "gsub_test.json").write_text(json.dumps({
            "v1": {"location": {"q1": [[0.1, 0.2]]}},
            "v2": {"location": {"q2": [[0.2, 0.3]]}}}))
        (self.data / "map_vid_vidorID.json").write_text(json.dumps({
            "v1": "a/v1.mp4", "v2": "b/v2.mp4"}))
        (self.data / "frame2time_test.json").write_text("{}")
        (self.media / "a").mkdir(); (self.media / "b").mkdir()
        (self.media / "a/v1.mp4").write_bytes(b"prior-video")
        (self.media / "b/v2.mp4").write_bytes(b"new-video")
        import hashlib
        prior_digest = hashlib.sha256(b"prior-video").hexdigest()
        payload = json.loads((self.private / "past.json").read_text())
        payload["rows"][0]["video_sha256"] = prior_digest
        (self.private / "past.json").write_text(json.dumps(payload))

    def test_known_hash_exclusion_and_outcome_blind_candidate_order(self):
        self._inputs()
        payload = build_candidate_lock(dataset_root=self.data, video_root=self.media,
                                       private_root=self.private)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["excluded_known_media_hash_count"], 1)
        self.assertEqual(payload["candidates"][0]["video_id"], "v2")
        self.assertEqual(payload["history_limit"].split(";")[0],
                         "Excludes media hashes present in current local mode-0600 JSON only")

    def test_val_split_uses_val_annotations_and_hashes(self):
        self._inputs()
        with (self.data / "val.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid"])
            writer.writeheader()
            writer.writerow({"video_id": "v2", "qid": "q2"})
        (self.data / "gsub_val.json").write_text(json.dumps({
            "v2": {"location": {"q2": [[0.2, 0.3]]}}}))
        (self.data / "frame2time_val.json").write_text("{}")
        payload = build_candidate_lock(dataset_root=self.data, video_root=self.media,
                                       private_root=self.private, split="val")
        self.assertEqual(payload["split"], "val")
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["dataset_sha256"]["val.csv"],
                         hashlib.sha256((self.data / "val.csv").read_bytes()).hexdigest())

    def test_unsupported_split_is_rejected(self):
        self._inputs()
        with self.assertRaisesRegex(ValueError, "candidate_split_invalid"):
            build_candidate_lock(dataset_root=self.data, video_root=self.media,
                                 private_root=self.private, split="train")

    def test_private_inventory_rejects_bad_permissions_and_invalid_json(self):
        self._inputs()
        os.chmod(self.private / "past.json", 0o644)
        with self.assertRaisesRegex(ValueError, "mode_or_type_invalid"):
            collect_known_local_media_hashes(self.private)
        os.chmod(self.private / "past.json", 0o600)
        (self.private / "bad.json").write_text("{")
        os.chmod(self.private / "bad.json", 0o600)
        with self.assertRaisesRegex(ValueError, "json_invalid"):
            collect_known_local_media_hashes(self.private)

    def test_output_is_exclusive_and_mode_0600(self):
        self._inputs()
        path = self.private / "lock.json"
        write_private_exclusive(path, {"x": 1}, private_root=self.private)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "exists"):
            write_private_exclusive(path, {"x": 2}, private_root=self.private)

    def test_unscored_candidate_lock_does_not_self_exclude_its_rows(self):
        self._inputs()
        lock = build_candidate_lock(dataset_root=self.data, video_root=self.media,
                                    private_root=self.private)
        lock_path = self.private / "frozen_candidates.json"
        write_private_exclusive(lock_path, lock, private_root=self.private)
        known, _files = collect_known_local_media_hashes(self.private)
        candidate_hashes = {row["video_sha256"] for row in lock["candidates"]}
        self.assertTrue(candidate_hashes)
        self.assertFalse(candidate_hashes & known)
        self.assertEqual(hashlib.sha256(b"prior-video").hexdigest() in known, True)


if __name__ == "__main__":
    unittest.main()
