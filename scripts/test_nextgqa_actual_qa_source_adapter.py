import csv
import hashlib
import io
import json
import unittest

from nextgqa_actual_qa_source_adapter import build_actual_qa_source


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_inputs():
    fields = ["video_id", "qid", "question", "answer", "a0", "a1", "a2", "a3", "a4", "type"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    rows = []
    selection = []
    targets = []
    for i in range(3):
        row = {"video_id": f"ds-{i}", "qid": str(i), "question": f"question {i}?",
               "answer": f"answer {i}", "a0": f"answer {i}", "a1": f"foil {i}1",
               "a2": f"foil {i}2", "a3": f"foil {i}3", "a4": f"foil {i}4", "type": "TN"}
        rows.append(row)
        writer.writerow(row)
        selection.append({"video_id": f"target-{i}", "claim_id": f"claim-{i}",
                          "dataset_video_id": f"ds-{i}", "dataset_question_id": str(i)})
        targets.append({"video_id": f"target-{i}", "claim_id": f"claim-{i}",
                        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest()})
    raw = stream.getvalue().encode()
    lineage_rows = [{"video_id": f"target-{i}", "claim_id": f"claim-{i}",
                     "dataset_question_id": str(i), "dataset_video_id": f"ds-{i}",
                     "video_key": f"video-{i}", "relative_video_path": f"video-{i}.mp4",
                     "video_sha256": h(f"video-{i}"),
                     "question_sha256": hashlib.sha256(rows[i]["question"].encode()).hexdigest(),
                     "claim_text_sha256": h(f"claim text {i}")} for i in range(3)]
    lineage = {"schema_version": 1,
               "purpose": "private_source_lineage_crosscheck_for_claim_review",
               "created_by": "synthetic fixture", "source_packet_sha256": h("packet"),
               "cohort_sha256": h("cohort"),
               "dataset_inputs_sha256": {"test.csv": hashlib.sha256(raw).hexdigest(),
                                          "frame2time_test.json": h("frame2time"),
                                          "gsub_test.json": h("gsub"),
                                          "map_vid_vidorID.json": h("map")},
               "requested_records": 3, "effective_records": 3,
               "verification_scope": "synthetic", "status": "metadata_lineage_crosschecked",
               "records": lineage_rows}
    return raw, {"records": selection}, json.dumps({"records": targets}).encode(), json.dumps(lineage).encode(), rows


class ActualSourceAdapterTests(unittest.TestCase):
    def test_binds_real_csv_fields_including_choice_order_gold_and_source_row(self):
        raw, selection, target, lineage, source_rows = make_inputs()
        qa, manifest = build_actual_qa_source(csv_raw=raw, selection=selection,
                                              target_manifest_raw=target,
                                              lineage_manifest_raw=lineage)
        self.assertEqual(qa["purpose"], "private_actual_nextgqa_qa_source_rows_unvalidated")
        self.assertEqual(qa["records"][1]["choices"], ["answer 1", "foil 11", "foil 12", "foil 13", "foil 14"])
        self.assertEqual(manifest["target_manifest"]["records"][1]["gold_index"], 0)
        expected = hashlib.sha256(json.dumps(source_rows[1], sort_keys=True,
                                             separators=(",", ":"),
                                             ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(manifest["source_row_bindings"][1]["source_row_sha256"], expected)
        self.assertEqual(manifest["target_manifest"]["records"][1]["dataset_source_row_sha256"], expected)
        self.assertEqual(manifest["target_manifest"]["records"][1]["ordered_choice_sha256"],
                         [hashlib.sha256(c.encode()).hexdigest()
                          for c in qa["records"][1]["choices"]])

    def test_fails_closed_on_csv_hash_mismatch_and_duplicate_pair(self):
        raw, selection, target, lineage, _ = make_inputs()
        wrong = json.loads(lineage)
        wrong["dataset_inputs_sha256"]["test.csv"] = h("different")
        with self.assertRaisesRegex(ValueError, "lineage_test_csv_hash_mismatch"):
            build_actual_qa_source(csv_raw=raw, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(wrong).encode())
        duplicate = raw + raw.splitlines(keepends=True)[1]
        fixed_lineage = json.loads(lineage)
        fixed_lineage["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(duplicate).hexdigest()
        with self.assertRaisesRegex(ValueError, "dataset_video_qid_not_unique"):
            build_actual_qa_source(csv_raw=duplicate, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(fixed_lineage).encode())

    def test_rejects_repeated_canonical_video_id_with_distinct_video_hashes(self):
        raw, selection, target, lineage, _ = make_inputs()
        repeated = json.loads(lineage)
        repeated["records"][1]["dataset_video_id"] = "ds-0"
        selection["records"][1]["dataset_video_id"] = "ds-0"
        # The three claims still carry distinct media hashes; identity reuse is
        # rejected on the canonical dataset ID before any CSV-row lookup.
        self.assertEqual(len({row["video_sha256"] for row in repeated["records"]}), 3)
        with self.assertRaisesRegex(ValueError, "dataset_video_id_duplicate"):
            build_actual_qa_source(
                csv_raw=raw, selection=selection, target_manifest_raw=target,
                lineage_manifest_raw=json.dumps(repeated).encode())

    def test_ambiguous_gold_is_rejected(self):
        raw, selection, target, lineage, _ = make_inputs()
        altered = raw.replace(b"foil 01", b"answer 0")
        fixed = json.loads(lineage)
        fixed["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(altered).hexdigest()
        with self.assertRaisesRegex(ValueError, "dataset_gold_ambiguous_or_missing"):
            build_actual_qa_source(csv_raw=altered, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(fixed).encode())

    def test_short_csv_rows_and_nonstring_selection_fail_closed(self):
        raw, selection, target, lineage, _ = make_inputs()
        short = (b"video_id,qid,question,answer,a0,a1,a2,a3,a4,type\n"
                 b"ds-0,0,q,a,a,b,c,d,e,TN\nds-1,1,q,a,a,b,c,d,e,TN\n"
                 b"ds-2,2,q,a,a,b,c,d,e\n")
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(short).hexdigest()
        with self.assertRaisesRegex(ValueError, "csv_row_width_invalid"):
            build_actual_qa_source(csv_raw=short, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(source).encode())
        selection["records"][0]["dataset_video_id"] = None
        with self.assertRaisesRegex(ValueError, "selection_identity_invalid"):
            build_actual_qa_source(csv_raw=raw, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=lineage)

    def test_selection_and_csv_question_must_match_frozen_lineage(self):
        raw, selection, target, lineage, _ = make_inputs()
        selection["records"][0]["dataset_question_id"] = "attacker-qid"
        with self.assertRaisesRegex(ValueError, "selection_lineage_binding_mismatch"):
            build_actual_qa_source(csv_raw=raw, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=lineage)

        raw, selection, target, lineage, _ = make_inputs()
        altered = raw.replace(b"question 0?", b"question x?")
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(altered).hexdigest()
        with self.assertRaisesRegex(ValueError, "source_question_lineage_mismatch"):
            build_actual_qa_source(csv_raw=altered, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(source).encode())

    def test_duplicate_headers_are_rejected_and_nfkc_gold_matches(self):
        raw, selection, target, lineage, _ = make_inputs()
        duplicate_header = raw.replace(b"video_id,qid,", b"video_id,video_id,")
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(duplicate_header).hexdigest()
        with self.assertRaisesRegex(ValueError, "csv_columns_invalid"):
            build_actual_qa_source(csv_raw=duplicate_header, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(source).encode())

        # Full-width Latin A and ASCII A normalize to the same gold.
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["video_id", "qid", "question", "answer",
                                                      "a0", "a1", "a2", "a3", "a4", "type"],
                                lineterminator="\n")
        writer.writeheader()
        for i in range(3):
            row = {"video_id": f"ds-{i}", "qid": str(i), "question": f"question {i}?",
                   "answer": "A" if i == 0 else f"answer {i}",
                   "a0": "Ａ" if i == 0 else f"answer {i}",
                   "a1": f"foil {i}1", "a2": f"foil {i}2", "a3": f"foil {i}3",
                   "a4": f"foil {i}4", "type": "TN"}
            writer.writerow(row)
        rows = stream.getvalue().encode()
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(rows).hexdigest()
        qa, manifest = build_actual_qa_source(csv_raw=rows, selection=selection,
                                              target_manifest_raw=target,
                                              lineage_manifest_raw=json.dumps(source).encode())
        self.assertEqual(manifest["target_manifest"]["records"][0]["gold_index"], 0)
        self.assertEqual(qa["records"][0]["choices"][0], "Ａ")

    def test_source_question_hash_must_match_frozen_lineage(self):
        raw, selection, target, lineage, _ = make_inputs()
        altered = raw.replace(b"question 0?", b"question x?")
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(altered).hexdigest()
        with self.assertRaisesRegex(ValueError, "source_question_lineage_mismatch"):
            build_actual_qa_source(csv_raw=altered, selection=selection,
                                   target_manifest_raw=target,
                                   lineage_manifest_raw=json.dumps(source).encode())

    def test_full_csv_row_digest_covers_nonrequired_metadata_and_frozen_order(self):
        raw, selection, target, lineage, _ = make_inputs()
        changed = raw.replace(b",TN\n", b",OTHER_METADATA\n", 1)
        source = json.loads(lineage)
        source["dataset_inputs_sha256"]["test.csv"] = hashlib.sha256(changed).hexdigest()
        _, original_manifest = build_actual_qa_source(
            csv_raw=raw, selection=selection, target_manifest_raw=target,
            lineage_manifest_raw=lineage)
        _, changed_manifest = build_actual_qa_source(
            csv_raw=changed, selection=selection, target_manifest_raw=target,
            lineage_manifest_raw=json.dumps(source).encode())
        self.assertNotEqual(
            original_manifest["source_row_bindings"][0]["source_row_sha256"],
            changed_manifest["source_row_bindings"][0]["source_row_sha256"])

        reordered_target = json.loads(target)
        reordered_target["records"].reverse()
        with self.assertRaisesRegex(ValueError, "frozen_target_order_mismatch"):
            build_actual_qa_source(csv_raw=raw, selection=selection,
                                   target_manifest_raw=json.dumps(reordered_target).encode(),
                                   lineage_manifest_raw=lineage)


if __name__ == "__main__":
    unittest.main()
