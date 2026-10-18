from __future__ import annotations

import csv
import hashlib
import io
import json
import unittest

from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from qta_stage1_source_pin_adapter import (
    Stage1SourcePinError,
    derive_stage1_source_pins,
)
from qta_prescore_mapping_binding import interval_list_sha256
from test_qta_stage1_bundle_audit import QtaStage1BundleAuditTests


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class QtaStage1SourcePinAdapterTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing all-synthetic stage-1 fixture; it creates no
        # protected review artifact or production result.
        self.stage1_fixture = QtaStage1BundleAuditTests()
        self.stage1_fixture.setUp()
        self.addCleanup(self.stage1_fixture.doCleanups)
        self.stage1_inputs = {
            "worksheet_raw": self.stage1_fixture.worksheet_raw,
            "lineage_raw": self.stage1_fixture.lineage_raw,
            "review_raw": self.stage1_fixture.review_raw,
            "review_index_raw": self.stage1_fixture.review_index_raw,
            "custody_raw": self.stage1_fixture.custody_raw,
            "pre_review_receipt_raw": self.stage1_fixture.receipt_raw,
            "integrity_roster_raw": self.stage1_fixture.f.integrity_roster_raw,
            "release_roster_raw": self.stage1_fixture.f.release_roster_raw,
            "inventory_raw": self.stage1_fixture.f.inventory_raw,
            "release_witness_raw": self.stage1_fixture.release_witness_raw,
            "reviewer_attestation_raw": self.stage1_fixture.reviewer_attestation_raw,
            "media_dir": self.stage1_fixture.media_dir,
            "expected_lineage_sha256": self.stage1_fixture.expected_lineage_sha256,
            "expected_custody_sha256": self.stage1_fixture.expected_custody_sha256,
            "root_public_key_pem": self.stage1_fixture.f.keys["root"][1],
            "expected_root_public_der_sha256": self.stage1_fixture.f.root_pin,
            "expected_ceremony_id": self.stage1_fixture.f.ceremony,
            "artifact_sources": self.stage1_fixture.f.artifacts,
        }
        self.qa_inputs = self._make_qa_inputs()

    def _make_qa_inputs(self):
        worksheet = json.loads(self.stage1_fixture.worksheet_raw)
        lineage = json.loads(self.stage1_fixture.lineage_raw)
        stream = io.StringIO(newline="")
        fields = ["video_id", "qid", "question", "answer", *(f"a{i}" for i in range(5))]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        selection_records = []
        claims = []
        initial_targets = []
        for index, (row, line) in enumerate(zip(worksheet["records"], lineage["records"],
                                                strict=True)):
            options = [f"answer {index}", *(f"foil {index}-{j}" for j in range(1, 5))]
            writer.writerow({"video_id": line["dataset_video_id"],
                             "qid": line["dataset_question_id"],
                             "question": row["question"], "answer": options[0],
                             **{f"a{j}": option for j, option in enumerate(options)}})
            selection_records.append({
                "review_token": row["review_token"], "video_id": row["video_id"],
                "claim_id": row["claim_id"], "dataset_video_id": line["dataset_video_id"],
                "dataset_question_id": line["dataset_question_id"],
            })
            claims.append({"video_id": row["video_id"], "claim_id": row["claim_id"],
                           "claim_text": row["claim_text"],
                           "claim_text_sha256": row["claim_text_sha256"]})
            initial_targets.append({"video_id": row["video_id"], "claim_id": row["claim_id"],
                                    "claim_text_sha256": row["claim_text_sha256"]})
        csv_raw = stream.getvalue().encode()
        lineage["dataset_inputs_sha256"]["test.csv"] = sha(csv_raw)
        lineage_raw = encode(lineage)
        selection_raw = encode({"records": selection_records})
        claim_manifest_raw = encode({"records": claims})
        source_selection = {"records": [{key: record[key] for key in (
            "video_id", "claim_id", "dataset_video_id", "dataset_question_id")}
            for record in selection_records]}
        _, bound = build_actual_qa_source(
            csv_raw=csv_raw, selection=source_selection,
            target_manifest_raw=encode({"records": initial_targets}),
            lineage_manifest_raw=lineage_raw)
        protocol = b'{"protocol":"synthetic protocol"}'
        config = b'{"config":"synthetic config"}'
        return {
            "csv_raw": csv_raw,
            "claim_manifest_raw": claim_manifest_raw,
            "selection_raw": selection_raw,
            "target_manifest_raw": encode(bound["target_manifest"]),
            "lineage_manifest_raw": lineage_raw,
            "qta_protocol_raw": protocol,
            "expected_qta_protocol_sha256": sha(protocol),
            "qta_config_raw": config,
            "expected_qta_config_sha256": sha(config),
        }

    def invoke(self, *, stage1=None, qa=None):
        return derive_stage1_source_pins(
            stage1_audit_inputs=dict(self.stage1_inputs if stage1 is None else stage1),
            actual_qa_source_inputs=dict(self.qa_inputs if qa is None else qa),
        )

    def test_derives_exact_three_ordered_non_authorizing_source_pins(self):
        result = self.invoke()
        self.assertEqual(result["status"], "three_row_source_pins_structurally_derived")
        self.assertEqual(result["classification"],
                         "private_local_only_structural_source_pins")
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertEqual(result["stage1_audit_status"],
                         "stage1_structurally_valid_non_authorizing")
        self.assertEqual(result["actual_qa_source_status"], "bound_unvalidated")
        self.assertEqual(len(result["source_rows"]), 3)
        self.assertEqual(result["expected_cohort_tokens"], [
            row["review_token"] for row in json.loads(self.stage1_fixture.worksheet_raw)["records"]])
        self.assertTrue(all(row["video_support_decision"] == "supports_claim"
                            for row in result["source_rows"]))
        self.assertEqual(set(result["expected_baseline_source_sha256_by_token"]),
                         set(result["expected_cohort_tokens"]))
        self.assertEqual(set(result["expected_baseline_candidate_source_bindings_by_token"]),
                         set(result["expected_cohort_tokens"]))
        self.assertEqual(set(result["expected_baseline_candidate_source_bindings_by_token"][
            result["expected_cohort_tokens"][0]]), {
                "question_sha256", "options_sha256", "claim_sha256",
                "source_video_sha256", "reviewed_intervals_sha256",
                "video_support_decision"})
        token = result["expected_cohort_tokens"][0]
        self.assertEqual(
            result["expected_baseline_candidate_source_bindings_by_token"][token],
            {**result["expected_baseline_source_sha256_by_token"][token],
             **result["cohort_review_source_pins_by_token"][token]})
        self.assertEqual(result["expected_baseline_source_sha256_by_token"][
            result["expected_cohort_tokens"][0]], {
                "question_sha256": result["source_rows"][0]["question_sha256"],
                "options_sha256": result["source_rows"][0]["options_sha256"],
                "claim_sha256": result["source_rows"][0]["claim_sha256"],
            })
        self.assertEqual(result["cohort_review_source_pins_by_token"][
            result["expected_cohort_tokens"][0]]["reviewed_intervals_sha256"],
            result["source_rows"][0]["reviewed_intervals_sha256"])
        review_payload = json.loads(self.stage1_fixture.review_raw)
        first_intervals = review_payload["records"][0]["review"]["evidence_intervals_seconds"]
        self.assertEqual(result["source_rows"][0]["reviewed_intervals_sha256"],
                         interval_list_sha256(first_intervals))
        for field in ("model_load_permitted", "scoring_authorized",
                      "reviewer_identity_verified", "semantic_review_authenticated",
                      "clock_frame_correspondence_verified"):
            self.assertIs(result[field], False)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_caller_forged_stage1_summary_is_not_an_accepted_input(self):
        with self.assertRaises(TypeError):
            derive_stage1_source_pins(stage1_audit_inputs=self.stage1_inputs,
                                      actual_qa_source_inputs=self.qa_inputs,
                                      stage1_report={"status": "stage1_structurally_valid_non_authorizing"})

    def test_stage1_mutation_or_incomplete_stage1_audit_blocks(self):
        changed = dict(self.stage1_inputs)
        changed["review_raw"] += b" "
        with self.assertRaisesRegex(Stage1SourcePinError, "stage1_audit_not_complete"):
            self.invoke(stage1=changed)
        missing = dict(self.stage1_inputs)
        missing.pop("custody_raw")
        with self.assertRaisesRegex(Stage1SourcePinError, "stage1_audit_call_failed"):
            self.invoke(stage1=missing)

    def test_non_support_decision_and_interval_mutation_fail_closed(self):
        changed = dict(self.stage1_inputs)
        review = json.loads(changed["review_raw"])
        review["records"][0]["review"]["support_decision"] = "partial"
        changed["review_raw"] = encode(review)
        with self.assertRaisesRegex(Stage1SourcePinError, "stage1_audit_not_complete"):
            self.invoke(stage1=changed)

    def test_qa_question_option_claim_mismatch_and_wrong_order_block(self):
        for mutate in (
            lambda q: q.update(csv_raw=q["csv_raw"].replace(b"answer 0", b"changed 0", 1)),
            lambda q: q.update(selection_raw=encode({"records": list(
                reversed(json.loads(q["selection_raw"])["records"]))})),
        ):
            qa = dict(self.qa_inputs)
            mutate(qa)
            with self.assertRaises(Stage1SourcePinError):
                self.invoke(qa=qa)

    def test_partial_or_extra_cohort_rows_fail_closed(self):
        qa = dict(self.qa_inputs)
        selection = json.loads(qa["selection_raw"])
        selection["records"].pop()
        qa["selection_raw"] = encode(selection)
        with self.assertRaises(Stage1SourcePinError):
            self.invoke(qa=qa)

    def test_duplicate_json_key_in_stage1_bytes_fails_closed(self):
        changed = dict(self.stage1_inputs)
        changed["worksheet_raw"] = b'{"records":[],"records":[]}'
        with self.assertRaises(Stage1SourcePinError):
            self.invoke(stage1=changed)


if __name__ == "__main__":
    unittest.main()
