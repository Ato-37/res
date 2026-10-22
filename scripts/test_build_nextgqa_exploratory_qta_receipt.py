import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from build_nextgqa_exploratory_qta_receipt import (build_receipt, canonical_sha256,
    public_summary, sha256_file, write_private_exclusive)
from build_nextgqa_exploratory_video_split_v2 import derive_v2


class ExploratoryReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        os.chmod(self.private, 0o700)
        self.csv = self.dataset / "test.csv"
        with self.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "qid", "question", "answer"])
            w.writeheader()
            w.writerow({"video_id": "v1", "qid": "q1", "question": "private Q", "answer": "private A"})
            w.writerow({"video_id": "v2", "qid": "q2", "question": "private Q2", "answer": "private A2"})
        self.gsub = self.dataset / "gsub.json"
        self.gsub.write_text(json.dumps({"v1": {"duration": 5, "location": {"q1": [[1, 2]]}},
                                         "v2": {"duration": 6, "location": {"q2": [[2, 3]]}}}))
        self.f2t = self.dataset / "frame2time.json"
        self.f2t.write_text(json.dumps({"v1": list(range(32)), "v2": list(range(32))}))
        self.mapping = self.dataset / "mapping.json"
        self.mapping.write_text(json.dumps({"v1": "v1.mp4", "v2": "v2.mp4"}))
        self.media = self.dataset / "v1.mp4"
        self.media.write_bytes(b"video bytes")
        self.media2 = self.dataset / "v2.mp4"
        self.media2.write_bytes(b"video bytes 2")
        h1 = hashlib.sha256(self.media.read_bytes()).hexdigest()
        h2 = hashlib.sha256(self.media2.read_bytes()).hexdigest()
        self.plan = self.private / "plan.json"
        self.splitter = Path(__file__).with_name("build_nextgqa_exploratory_video_split_v2.py")
        self.v1_splitter = Path(__file__).with_name("build_nextgqa_exploratory_video_split.py")
        self.base_plan = self.private / "v1.json"
        self.base_plan.write_text(json.dumps({"status": "exploratory_split_plan_not_evaluation",
            "provenance": {"source_script_sha256": sha256_file(self.v1_splitter),
                           "test_csv_sha256": sha256_file(self.csv)},
            "calibration": [{"video_id": "v2", "qid": "q2", "media_sha256": h2,
                             "media_path": str(self.media2)}],
            "evaluation": [{"video_id": "v1", "qid": "q1", "media_sha256": h1,
                            "media_path": str(self.media)}]}))
        self.plan.write_text(json.dumps(derive_v2(base_plan=self.base_plan, test_csv=self.csv,
            gsub_json=self.gsub, frame2time_json=self.f2t, mapping_json=self.mapping,
            v1_splitter=self.v1_splitter, v2_source=self.splitter,
            expected_v1_plan_sha256=sha256_file(self.base_plan),
            v1_freeze_receipt_sha256="f" * 64,
            expected_partition_sizes=(1, 1))))

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return build_receipt(test_csv=self.csv, gsub_json=self.gsub, frame2time_json=self.f2t,
            mapping_json=self.mapping, split_plan=self.plan, splitter_source=self.splitter,
            base_plan=self.base_plan, v1_splitter=self.v1_splitter,
            expected_v1_plan_sha256=sha256_file(self.base_plan),
            v1_freeze_receipt_sha256="f" * 64,
            expected_partition_sizes=(1, 1))

    def test_receipt_binds_inputs_and_redacts_raw_content(self):
        receipt = self.build()
        self.assertEqual(receipt["counts"]["evaluation_rows_bound"], 1)
        self.assertEqual(set(receipt["input_sha256"]), {"test_csv", "gsub_test_json",
            "frame2time_test_json", "map_vid_vidorID_json", "split_plan", "splitter_source",
            "v1_plan", "v1_splitter_source", "v1_freeze_receipt"})
        serialized = json.dumps(receipt)
        for secret in ("private Q", "private A", str(self.media), "q1"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(receipt["faithfulness_status"], "not_measured")
        self.assertIn("not clean, fresh, confirmatory", receipt["scope"])

    def test_public_stdout_summary_contains_only_counts_and_hashes(self):
        summary = json.dumps(public_summary(self.build(), "a" * 64))
        for secret in ("private Q", "private A", str(self.media)):
            self.assertNotIn(secret, summary)

    def test_content_mutation_changes_binding(self):
        first = self.build()["evaluation_row_bindings"][0]["row_binding_sha256"]
        self.gsub.write_text(json.dumps({"v1": {"duration": 5, "location": {"q1": [[1, 4]]}},
                                         "v2": {"duration": 6, "location": {"q2": [[2, 3]]}}}))
        self.assertEqual(len(first), 64)
        with self.assertRaisesRegex(ValueError, "v2_plan_input_hash_mismatch"):
            self.build()

    def test_missing_join_rejected(self):
        self.gsub.write_text(json.dumps({"v1": {"duration": 5, "location": {}}}))
        plan = json.loads(self.plan.read_text())
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "v2_plan_input_hash_mismatch"):
            self.build()

    def test_split_overlap_rejected(self):
        plan = json.loads(self.plan.read_text())
        plan["calibration"].append(dict(plan["evaluation"][0]))
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "v2_plan_reconstruction_mismatch"):
            self.build()

    def test_self_consistent_edited_plan_rejected_by_reconstruction(self):
        plan = json.loads(self.plan.read_text())
        row = {"video_id": "v1", "qid": "q1", "question": "private Q", "answer": "private A"}
        plan["evaluation"][0]["media_sha256"] = sha256_file(self.media2)
        plan["evaluation"][0]["media_path"] = str(self.media2)
        plan["evaluation"][0]["selected_row_binding_sha256"] = canonical_sha256({
            "csv_row": row, "gsub": {"duration": 5, "spans": [[1, 2]]},
            "frame2time": list(range(32)), "map_target": "v1.mp4",
            "media_sha256": sha256_file(self.media2)})
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "v2_plan_reconstruction_mismatch"):
            self.build()

    def test_exclusive_private_mode_and_no_overwrite(self):
        receipt = self.build()
        out = self.private / "receipt.json"
        write_private_exclusive(out, receipt, private_root=self.private)
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            write_private_exclusive(out, receipt, private_root=self.private)

    def test_output_outside_private_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside_private_root"):
            write_private_exclusive(self.root / "leak.json", self.build(), private_root=self.private)


if __name__ == "__main__":
    unittest.main()
