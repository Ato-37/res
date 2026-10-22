import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from build_nextgqa_exploratory_video_split_v2 import derive_v2, valid_spans, write_exclusive


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SplitV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.private = self.root / "private"; self.private.mkdir(mode=0o700); os.chmod(self.private, 0o700)
        self.data = self.root / "data"; self.data.mkdir()
        self.csv = self.data / "test.csv"
        with self.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "qid", "question", "answer"]); w.writeheader()
            for vid in ("v1", "v2"):
                w.writerow({"video_id": vid, "qid": vid+"bad", "question": "private", "answer": "secret"})
                w.writerow({"video_id": vid, "qid": vid+"ok", "question": "private2", "answer": "secret2"})
        self.gsub = self.data / "gsub.json"
        self.gsub.write_text(json.dumps({vid: {"duration": 10, "location": {vid+"bad": [[-1, 2]],
            vid+"ok": [[2, 4]]}} for vid in ("v1", "v2")}))
        self.f2t = self.data / "frame2time.json"
        # The existing array is bound as given; no unsupported monotonicity claim.
        self.f2t.write_text(json.dumps({vid: [2, 1] + [float(i) for i in range(30)] for vid in ("v1", "v2")}))
        self.mapping = self.data / "mapping.json"; self.mapping.write_text(json.dumps({"v1": "v1.mp4", "v2": "v2.mp4"}))
        self.media = {}
        for vid in ("v1", "v2"):
            p = self.data / (vid+".mp4"); p.write_bytes(vid.encode()); self.media[vid] = p
        self.v1src = self.root / "v1.py"; self.v1src.write_text("v1 frozen")
        self.v2src = Path(__file__).with_name("build_nextgqa_exploratory_video_split_v2.py")
        self.plan = self.private / "v1.json"
        self.plan.write_text(json.dumps({"status": "exploratory_split_plan_not_evaluation",
            "provenance": {"source_script_sha256": sha(self.v1src), "test_csv_sha256": sha(self.csv)},
            "calibration": [{"video_id": "v1", "qid": "v1bad", "media_sha256": sha(self.media["v1"]),
                             "media_path": str(self.media["v1"])}],
            "evaluation": [{"video_id": "v2", "qid": "v2bad", "media_sha256": sha(self.media["v2"]),
                            "media_path": str(self.media["v2"])}]}))

    def tearDown(self): self.tmp.cleanup()

    def build(self):
        return derive_v2(base_plan=self.plan, test_csv=self.csv, gsub_json=self.gsub,
            frame2time_json=self.f2t, mapping_json=self.mapping, v1_splitter=self.v1src,
            v2_source=self.v2src, expected_v1_plan_sha256=sha(self.plan),
            v1_freeze_receipt_sha256="f" * 64,
            expected_partition_sizes=(1, 1))

    def test_preserves_video_split_and_chooses_only_valid_qid(self):
        plan = self.build()
        self.assertEqual(plan["calibration"][0]["video_id"], "v1")
        self.assertEqual(plan["evaluation"][0]["video_id"], "v2")
        self.assertEqual(plan["evaluation"][0]["qid"], "v2ok")
        self.assertEqual(plan["counts"]["evaluation_videos_with_valid_qid"], 1)
        self.assertEqual(len(plan["evaluation"][0]["selected_row_binding_sha256"]), 64)
        dumped = json.dumps(plan)
        for value in ("secret", "private", "answer"):
            self.assertNotIn(value, dumped)
        self.assertIn("faithfulness evidence", plan["limitations"][1].lower())

    def test_invalid_interval_predicate(self):
        self.assertFalse(valid_spans({"duration": 5, "location": {"q": [[-0.1, 1]]}}, "q"))
        self.assertTrue(valid_spans({"duration": 5, "location": {"q": [[1, 2]]}}, "q"))

    def test_all_or_none_when_video_has_no_valid_qid(self):
        self.gsub.write_text(json.dumps({vid: {"duration": 10, "location": {vid+"bad": [[-2, 3]]}}
                                         for vid in ("v1", "v2")}))
        with self.assertRaisesRegex(ValueError, "video_has_no_valid_test_qid"):
            self.build()

    def test_pinned_source_and_input_mutations_rejected(self):
        self.v1src.write_text("changed")
        with self.assertRaisesRegex(ValueError, "v1_splitter_source_binding_mismatch"):
            self.build()

    def test_malformed_frame2time_rejected(self):
        self.f2t.write_text(json.dumps({"v1": [False] * 32, "v2": list(range(32))}))
        with self.assertRaisesRegex(ValueError, "frame2time_record_invalid"):
            self.build()

    def test_map_target_must_match_video_identity(self):
        self.mapping.write_text(json.dumps({"v1": "wrong.mp4", "v2": "v2.mp4"}))
        with self.assertRaisesRegex(ValueError, "video_map_media_identity_mismatch"):
            self.build()

    def test_duplicate_video_partition_rejected(self):
        plan = json.loads(self.plan.read_text())
        plan["calibration"].append(dict(plan["calibration"][0]))
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "split_overlap_detected"):
            derive_v2(base_plan=self.plan, test_csv=self.csv, gsub_json=self.gsub,
                frame2time_json=self.f2t, mapping_json=self.mapping, v1_splitter=self.v1src,
                v2_source=self.v2src, expected_v1_plan_sha256=sha(self.plan),
                v1_freeze_receipt_sha256="f" * 64,
                expected_partition_sizes=(2, 1))

    def test_exclusive_private_writer(self):
        plan = self.build(); dest = self.private / "v2.json"
        write_exclusive(dest, plan, self.private)
        self.assertEqual(dest.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_exclusive(dest, plan, self.private)


if __name__ == "__main__": unittest.main()
