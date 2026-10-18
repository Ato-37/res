import csv
import hashlib
import io
import json
import unittest

from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from qta_actual_prebaseline_diagnostic import diagnose_qta_prebaseline
from test_audit_nextgqa_review_handoff import fixture as review_fixture
from test_nextgqa_actual_qa_source_adapter import make_inputs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def enc(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def combined_inputs():
    worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = review_fixture()
    _, _, starter_target, source_lineage_raw, csv_rows = make_inputs()
    source_lineage = json.loads(source_lineage_raw)
    source_rows = []
    for i, (review_line, original) in enumerate(zip(lineage["records"], csv_rows, strict=True)):
        row = dict(original)
        row["video_id"] = source_lineage["records"][i]["dataset_video_id"]
        row["qid"] = review_line["dataset_question_id"]
        row["answer"] = f"answer {i}"
        row["a0"] = row["answer"]
        source_rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(source_rows)
    csv_raw = stream.getvalue().encode()
    source_lineage["dataset_inputs_sha256"]["test.csv"] = sha(csv_raw)
    for i, (source_line, worksheet_row, review_line) in enumerate(zip(
            source_lineage["records"], worksheet["records"], lineage["records"], strict=True)):
        source_line.update({
            "video_id": worksheet_row["video_id"], "claim_id": worksheet_row["claim_id"],
            "video_sha256": worksheet_row["video_sha256"],
            "claim_text_sha256": worksheet_row["claim_text_sha256"],
            "dataset_question_id": review_line["dataset_question_id"],
            "question_sha256": sha(source_rows[i]["question"].encode()),
        })
    source_lineage_raw = enc(source_lineage)
    claims = {"schema_version": 1, "records": [
        {"video_id": row["video_id"], "claim_id": row["claim_id"],
         "claim_text": row["claim_text"], "claim_text_sha256": row["claim_text_sha256"]}
        for row in worksheet["records"]]}
    selection = {"records": [{
        "review_token": worksheet_row["review_token"],
        "video_id": source_line["video_id"], "claim_id": source_line["claim_id"],
        "dataset_video_id": source_line["dataset_video_id"],
        "dataset_question_id": source_line["dataset_question_id"],
    } for worksheet_row, source_line in zip(worksheet["records"], source_lineage["records"], strict=True)]}
    target = {"records": [{
        "video_id": row["video_id"], "claim_id": row["claim_id"],
        "claim_text_sha256": row["claim_text_sha256"],
        "question_sha256": line["question_sha256"],
    } for row, line in zip(worksheet["records"], source_lineage["records"], strict=True)]}
    _, bound = build_actual_qa_source(
        csv_raw=csv_raw,
        selection={"records": [{k: v for k, v in row.items() if k != "review_token"}
                               for row in selection["records"]]},
        target_manifest_raw=enc(target), lineage_manifest_raw=source_lineage_raw)
    protocol = b'{"qta":"synthetic protocol fixture"}'
    config = b'{"qta":"synthetic config fixture"}'
    source_inputs = {
        "csv_raw": csv_raw, "claim_manifest_raw": enc(claims),
        "selection_raw": enc(selection),
        "target_manifest_raw": enc(bound["target_manifest"]),
        "lineage_manifest_raw": source_lineage_raw,
        "qta_protocol_raw": protocol,
        "expected_qta_protocol_sha256": sha(protocol),
        "qta_config_raw": config,
        "expected_qta_config_sha256": sha(config),
    }
    review_inputs = {
        "worksheet_raw": worksheet_raw,
        "lineage_raw": lineage_raw,
        "review_raw": review_raw,
        "review_index_raw": index_raw,
        "expected_lineage_sha256": sha(lineage_raw),
    }
    private_values = [row["claim_text"] for row in worksheet["records"]]
    private_values += [row["question"] for row in worksheet["records"]]
    return source_inputs, review_inputs, private_values


def changed_source_claims(source):
    source = dict(source)
    claims = json.loads(source["claim_manifest_raw"])
    lineage = json.loads(source["lineage_manifest_raw"])
    selection = json.loads(source["selection_raw"])
    target = json.loads(source["target_manifest_raw"])
    for claim, line, selected, target_row in zip(
            claims["records"], lineage["records"], selection["records"],
            target["records"], strict=True):
        text = "different source claim text for " + line["video_id"]
        claim_sha = sha(text.encode())
        claim_id = "claim-" + sha((line["video_sha256"] + "\0"
                                    + line["dataset_question_id"] + "\0"
                                    + claim_sha).encode())
        claim.update(claim_text=text, claim_text_sha256=claim_sha, claim_id=claim_id)
        line.update(claim_text_sha256=claim_sha, claim_id=claim_id)
        selected["claim_id"] = claim_id
        target_row.update(claim_text_sha256=claim_sha, claim_id=claim_id)
    source["claim_manifest_raw"] = enc(claims)
    source["lineage_manifest_raw"] = enc(lineage)
    source["selection_raw"] = enc(selection)
    _, bound = build_actual_qa_source(
        csv_raw=source["csv_raw"],
        selection={"records": [{key: row[key] for key in (
            "video_id", "claim_id", "dataset_video_id", "dataset_question_id")}
                               for row in selection["records"]]},
        target_manifest_raw=enc(target), lineage_manifest_raw=source["lineage_manifest_raw"])
    source["target_manifest_raw"] = enc(bound["target_manifest"])
    return source


class QtaActualPrebaselineDiagnosticTests(unittest.TestCase):
    def test_byte_validators_rerun_and_joint_three_row_order_is_consistent(self):
        source, review, private_values = combined_inputs()
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        self.assertEqual(report["status"], "diagnostic_incomplete")
        self.assertEqual(report["requested_rows"], 3)
        for name in ("actual_qa_source_binding", "review_bundle_byte_structure",
                     "source_to_review_ordered_join"):
            self.assertEqual(report["gates"][name]["status"], "passed")
            self.assertEqual(report["gates"][name]["validated_rows"], 3)
            self.assertEqual(report["gates"][name]["provenance_level"],
                             "caller_pinned_unverified")
        self.assertEqual(report["gates"]["baseline_full_video_pq_freeze"]["status"],
                         "not_supplied")
        self.assertEqual(report["gates"]["human_identity_and_semantic_review_proof"]["status"],
                         "not_supplied")
        public = json.dumps(report, ensure_ascii=False)
        for secret in private_values + ["claim-", "video-", "gold_index", "answer 0"]:
            self.assertNotIn(secret, public)
        self.assertNotIn("sha256", public)
        for inputs in (source, review):
            for value in inputs.values():
                if isinstance(value, bytes):
                    self.assertNotIn(sha(value), public)
                elif isinstance(value, str) and len(value) == 64:
                    self.assertNotIn(value, public)
        self.assertIs(report["human_identity_verified"], False)
        self.assertIs(report["semantic_review_authenticated"], False)
        self.assertIs(report["baseline_pq_frozen"], False)
        self.assert_denied(report)

    def test_missing_inputs_are_not_supplied_and_denial_is_unconditional(self):
        report = diagnose_qta_prebaseline(actual_source_inputs=None,
                                          review_bundle_inputs=None)
        self.assertEqual(report["gates"]["actual_qa_source_binding"]["status"],
                         "not_supplied")
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["status"],
                         "not_supplied")
        self.assertEqual(report["gates"]["source_to_review_ordered_join"]["status"],
                         "blocked")
        self.assertTrue(all(g["requested_rows"] == 3 for g in report["gates"].values()))
        self.assert_denied(report)

    def test_tampered_pin_blocks_source_and_no_partial_rows_are_counted(self):
        source, review, _ = combined_inputs()
        source["expected_qta_config_sha256"] = "0" * 64
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        gate = report["gates"]["actual_qa_source_binding"]
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["validated_rows"], 0)
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["status"], "passed")
        self.assertEqual(report["gates"]["source_to_review_ordered_join"]["status"], "blocked")
        self.assert_denied(report)

    def test_wrong_review_expected_hash_blocks_without_leaking_identifiers(self):
        source, review, _ = combined_inputs()
        review["expected_lineage_sha256"] = "f" * 64
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["status"], "blocked")
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["validated_rows"], 0)
        self.assertEqual(report["gates"]["source_to_review_ordered_join"]["status"], "blocked")
        self.assert_denied(report)

    def test_valid_source_and_review_bytes_with_different_claims_do_not_join(self):
        source, review, _ = combined_inputs()
        source = changed_source_claims(source)
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        self.assertEqual(report["gates"]["actual_qa_source_binding"]["status"], "passed")
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["status"], "passed")
        join = report["gates"]["source_to_review_ordered_join"]
        self.assertEqual(join["status"], "blocked")
        self.assertEqual(join["failure_code"], "ordered_source_review_claim_video_mismatch")
        self.assertEqual(join["validated_rows"], 0)
        self.assert_denied(report)

    def test_oversized_raw_input_is_rejected_before_private_binding(self):
        source, review, _ = combined_inputs()
        source["csv_raw"] = b"x" * 24_000_001
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        self.assertEqual(report["gates"]["actual_qa_source_binding"]["failure_code"],
                         "source_raw_input_too_large")
        self.assertEqual(report["gates"]["actual_qa_source_binding"]["validated_rows"], 0)
        self.assertEqual(report["gates"]["source_to_review_ordered_join"]["status"], "blocked")
        self.assert_denied(report)

    def test_oversized_review_input_is_rejected_before_review_validation(self):
        source, review, _ = combined_inputs()
        review["worksheet_raw"] = b"x" * 24_000_001
        report = diagnose_qta_prebaseline(actual_source_inputs=source,
                                          review_bundle_inputs=review)
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["failure_code"],
                         "review_raw_input_too_large")
        self.assertEqual(report["gates"]["review_bundle_byte_structure"]["validated_rows"], 0)
        self.assertEqual(report["gates"]["source_to_review_ordered_join"]["status"], "blocked")
        self.assert_denied(report)

    def assert_denied(self, report):
        for field in ("model_load_permitted", "scoring_authorized", "phase_authorized"):
            self.assertIs(report[field], False)
        for field in ("forwards", "score_calls"):
            self.assertEqual(report[field], 0)


if __name__ == "__main__":
    unittest.main()
