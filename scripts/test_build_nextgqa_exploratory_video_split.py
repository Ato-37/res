"""CPU tests for private deterministic exploratory video allocation."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nextgqa_exploratory_video_split import build_split, write_private_exclusive


class ExploratorySplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video_root = self.root / "videos"
        self.video_root.mkdir()
        self.csv_path = self.root / "test.csv"
        self.manifest = self.root / "prior.json"
        self.prior_media = self.root / "old_media"
        self.prior_media.mkdir()
        rows = []
        media = {
            "v1": b"exact-copy",
            "v2": b"exact-copy",
            "v3": b"video-three",
            "v4": b"video-four",
            "v5": b"prior-id",
            "v6": b"prior-stem",
        }
        for video_id, payload in media.items():
            (self.video_root / f"{video_id}.mp4").write_bytes(payload)
            rows.append({"video_id": video_id, "qid": f"q-{video_id}"})
        rows.append({"video_id": "v1", "qid": "q-v1-extra"})
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "qid"])
            writer.writeheader()
            writer.writerows(rows)
        self.manifest.write_text(json.dumps([
            {"video_id": "v5", "video": "/old/path/v5.mp4"},
            {"video": "/old/path/v6.mp4"},
        ]), encoding="utf-8")
        (self.prior_media / "v3-reencoded-name.mp4").write_bytes(b"video-three")

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_is_repeatable_excludes_previous_ids_and_keeps_exact_copies_together(self):
        kwargs = dict(test_csv=self.csv_path, video_root=self.video_root,
                      prior_manifests=[self.manifest],
                      prior_video_roots=[self.prior_media], calibration_fraction=0.67)
        first = build_split(**kwargs)
        second = build_split(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["candidate_media_file_count_before_exclusions"], 6)
        self.assertEqual(first["provenance"]["excluded_by_prior_video_id"], 1)
        self.assertEqual(first["provenance"]["excluded_by_prior_path_stem"], 1)
        self.assertEqual(first["provenance"]["excluded_by_prior_exact_media_hash"], 1)
        self.assertEqual(first["provenance"]["eligible_candidate_video_count_after_exclusions"], 3)
        self.assertEqual(first["split"]["eligible_qid_count"], 4)
        self.assertEqual(first["split"]["selected_qid_count"], 3)
        self.assertEqual(first["split"]["unselected_qid_count"], 1)
        self.assertEqual(first["split"]["effective_calibration_videos"]
                         + first["split"]["effective_evaluation_videos"], 3)
        self.assertEqual(first["split"]["exact_media_sha256_overlap"], 0)
        self.assertEqual(first["split"]["video_id_overlap"], 0)
        assignments = {}
        for split in ("calibration", "evaluation"):
            for row in first[split]:
                assignments[row["video_id"]] = (split, row["media_sha256"])
        self.assertNotIn("v5", assignments)
        self.assertNotIn("v6", assignments)
        self.assertEqual(assignments["v1"][0], assignments["v2"][0])
        self.assertEqual(assignments["v1"][1], assignments["v2"][1])

    def test_private_writer_is_owner_only_and_refuses_overwrite(self):
        target = self.root / "private" / "plan.json"
        private_root = self.root / "private"
        write_private_exclusive(target, {"ok": True}, private_root=private_root)
        self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_private_exclusive(target, {"ok": False}, private_root=private_root)
        with self.assertRaisesRegex(ValueError, "private_root"):
            write_private_exclusive(self.root / "public.json", {"secret": True},
                                    private_root=private_root)

    def test_split_rejects_invalid_fraction_and_duplicate_media_mapping(self):
        with self.assertRaisesRegex(ValueError, "fraction_out_of_range"):
            build_split(test_csv=self.csv_path, video_root=self.video_root,
                        prior_manifests=[], calibration_fraction=1.0)
        (self.video_root / "v1.avi").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "duplicate_media_for_video_id"):
            build_split(test_csv=self.csv_path, video_root=self.video_root,
                        prior_manifests=[], calibration_fraction=0.67)


if __name__ == "__main__":
    unittest.main()
