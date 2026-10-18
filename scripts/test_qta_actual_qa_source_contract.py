import hashlib
import json
import unittest
from unittest.mock import patch

from qta_actual_qa_source_contract import MAX_INPUT_BYTES, bind_qta_actual_qa_source
from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from test_nextgqa_actual_qa_source_adapter import make_inputs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class QtaActualQaSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.csv, self.selection, self.target, self.lineage, _ = make_inputs()
        line_doc = json.loads(self.lineage)
        target_doc = json.loads(self.target)
        selection_doc = self.selection
        lines = line_doc["records"]
        for i, selected in enumerate(selection_doc["records"]):
            selected["review_token"] = f"opaque-token-{i}"
        claims = [{"video_id": line["video_id"], "claim_id": line["claim_id"],
                   "claim_text": f"private claim phrase {i}",
                   "claim_text_sha256": sha(f"private claim phrase {i}".encode())}
                  for i, line in enumerate(lines)]
        for i, (line, claim) in enumerate(zip(lines, claims, strict=True)):
            line["claim_text_sha256"] = claim["claim_text_sha256"]
            target_doc["records"][i]["claim_text_sha256"] = claim["claim_text_sha256"]
            line["claim_id"] = "claim-" + sha(
                (line["video_sha256"] + "\0" + line["dataset_question_id"] + "\0"
                 + claim["claim_text_sha256"]).encode())
            claim["claim_id"] = line["claim_id"]
            target_doc["records"][i]["claim_id"] = line["claim_id"]
            selection_doc["records"][i]["claim_id"] = line["claim_id"]
        self.claims = json.dumps({"schema_version": 1, "records": claims}).encode()
        self.selection_raw = json.dumps(selection_doc).encode()
        self.lineage = json.dumps(line_doc).encode()
        source_selection = {"records": [{key: row[key] for key in
                             ("video_id", "claim_id", "dataset_video_id", "dataset_question_id")}
                            for row in selection_doc["records"]]}
        _, target_manifest = build_actual_qa_source(
            csv_raw=self.csv, selection=source_selection,
            target_manifest_raw=json.dumps(target_doc).encode(),
            lineage_manifest_raw=self.lineage)
        self.target = json.dumps(target_manifest["target_manifest"]).encode()
        self.protocol = b'{"protocol":"synthetic qta protocol v-test"}'
        self.config = b'{"config":"synthetic qta config v-test"}'

    def invoke(self, **changes):
        args = {
            "csv_raw": self.csv,
            "claim_manifest_raw": self.claims,
            "selection_raw": self.selection_raw,
            "target_manifest_raw": self.target,
            "lineage_manifest_raw": self.lineage,
            "qta_protocol_raw": self.protocol,
            "expected_qta_protocol_sha256": sha(self.protocol),
            "qta_config_raw": self.config,
            "expected_qta_config_sha256": sha(self.config),
        }
        args.update(changes)
        return bind_qta_actual_qa_source(**args)

    def assert_denied(self, result):
        summary = result["summary"]
        for field in ("model_load_permitted", "execution_authorized", "scoring_authorized"):
            self.assertIs(summary[field], False)
        for field in ("score_calls", "baseline_forwards", "branch_forwards"):
            self.assertEqual(summary[field], 0)

    def test_success_is_three_ordered_rows_private_and_aggregate_only(self):
        result = self.invoke()
        summary = result["summary"]
        self.assertEqual(summary["status"], "bound_unvalidated")
        self.assertEqual((summary["requested_rows"], summary["validated_rows"]), (3, 3))
        self.assertIs(summary["p_q_frozen"], False)
        self.assertEqual(summary["primary_estimand"],
                         "baseline_full_video_frozen_p_over_q_prediction_link")
        self.assertEqual(summary["gold_answer_index_use"],
                         "secondary_only_after_baseline_prediction_freeze")
        self.assertEqual(summary["rights_status"], "unverified_local_only")
        self.assertIs(summary["faithfulness_scored"], False)
        public = json.dumps(summary, ensure_ascii=False)
        for private_text in ("target-0", "claim-0", "private claim phrase", "question 0?",
                             "answer 0", "gold_index", "p/q", "opaque-token-0"):
            self.assertNotIn(private_text, public)
        bundle = result["private_bundle"]
        self.assertEqual(bundle["classification"], "private_local_only_unvalidated_qa_source")
        self.assertNotIn(sha(self.csv), public)
        self.assertNotIn("input_sha256", summary)
        self.assertNotIn("ordered_cohort_binding_sha256", summary)
        self.assertIn(b"question 0?", bundle["raw_json"])
        self.assertIn(b"answer 0", bundle["raw_json"])
        self.assertIn(b"gold_index", bundle["raw_json"])
        self.assertIn(b"opaque-token-0", bundle["raw_json"])
        self.assert_denied(result)

    def test_protocol_and_config_pins_fail_closed_with_fixed_denominator(self):
        for changes, code in (
            ({"expected_qta_protocol_sha256": "0" * 64}, "qta_protocol_digest_pin_mismatch"),
            ({"expected_qta_config_sha256": "0" * 64}, "qta_config_digest_pin_mismatch"),
        ):
            with self.subTest(code=code):
                result = self.invoke(**changes)
                self.assertEqual(result["summary"]["failure_code"], code)
                self.assertEqual((result["summary"]["requested_rows"],
                                  result["summary"]["validated_rows"]), (3, 0))
                self.assertIsNone(result["private_bundle"])
                self.assert_denied(result)

    def test_claim_selection_and_target_order_are_jointly_fixed(self):
        reordered_selection = json.loads(self.selection_raw)
        reordered_selection["records"].reverse()
        result = self.invoke(selection_raw=json.dumps(reordered_selection).encode())
        self.assertEqual(result["summary"]["failure_code"], "selection_lineage_mismatch")
        self.assertIsNone(result["private_bundle"])
        reordered_target = json.loads(self.target)
        reordered_target["records"].reverse()
        result = self.invoke(target_manifest_raw=json.dumps(reordered_target).encode())
        self.assertIn(result["summary"]["failure_code"],
                      {"claim_lineage_target_mismatch", "frozen_target_order_mismatch"})
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)

    def test_review_tokens_are_required_unique_and_bound_in_cohort_digest(self):
        missing = json.loads(self.selection_raw)
        missing["records"][0].pop("review_token")
        result = self.invoke(selection_raw=json.dumps(missing).encode())
        self.assertEqual(result["summary"]["failure_code"], "selection_row_schema_invalid")
        self.assertIsNone(result["private_bundle"])
        duplicate = json.loads(self.selection_raw)
        duplicate["records"][2]["review_token"] = duplicate["records"][1]["review_token"]
        result = self.invoke(selection_raw=json.dumps(duplicate).encode())
        self.assertEqual(result["summary"]["failure_code"], "review_token_identity_invalid")
        changed = json.loads(self.selection_raw)
        changed["records"][0]["review_token"] = "different-opaque-token"
        result = self.invoke(selection_raw=json.dumps(changed).encode())
        self.assertEqual(result["summary"]["status"], "bound_unvalidated")
        self.assertNotEqual(result["private_bundle"]["ordered_cohort_binding_sha256"],
                            self.invoke()["private_bundle"]["ordered_cohort_binding_sha256"])
        self.assert_denied(result)

    def test_target_claim_digest_and_lineage_video_hash_are_verified(self):
        target = json.loads(self.target)
        target["records"][0]["claim_text_sha256"] = "0" * 64
        result = self.invoke(target_manifest_raw=json.dumps(target).encode())
        self.assertEqual(result["summary"]["failure_code"], "claim_lineage_target_mismatch")
        self.assertIsNone(result["private_bundle"])
        lineage = json.loads(self.lineage)
        lineage["records"][0]["video_sha256"] = "not-a-sha"
        result = self.invoke(lineage_manifest_raw=json.dumps(lineage).encode())
        self.assertEqual(result["summary"]["failure_code"], "lineage_video_digest_invalid")
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)

    def test_frozen_target_source_digests_and_gold_must_match_csv(self):
        tamper_cases = (
            ("ordered_choice_sha256", ["0" * 64] * 5),
            ("dataset_source_row_sha256", "0" * 64),
            ("question_sha256", "0" * 64),
            ("dataset_qid_sha256", "0" * 64),
            ("gold_index", 1),
        )
        for field, value in tamper_cases:
            with self.subTest(field=field):
                target = json.loads(self.target)
                target["records"][0][field] = value
                result = self.invoke(target_manifest_raw=json.dumps(target).encode())
                self.assertEqual(result["summary"]["failure_code"],
                                 "target_source_binding_mismatch")
                self.assertIsNone(result["private_bundle"])
                self.assert_denied(result)

    def test_csv_choice_reordering_with_updated_csv_pin_fails_frozen_target(self):
        altered = self.csv.replace(b"foil 01,foil 02", b"foil 02,foil 01", 1)
        lineage = json.loads(self.lineage)
        lineage["dataset_inputs_sha256"]["test.csv"] = sha(altered)
        result = self.invoke(csv_raw=altered,
                             lineage_manifest_raw=json.dumps(lineage).encode())
        self.assertEqual(result["summary"]["failure_code"],
                         "target_source_binding_mismatch")
        self.assertEqual(result["summary"]["validated_rows"], 0)
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)

    def test_claim_text_digest_or_identity_tampering_fails(self):
        claim_doc = json.loads(self.claims)
        claim_doc["records"][0]["claim_text"] = "tampered claim"
        result = self.invoke(claim_manifest_raw=json.dumps(claim_doc).encode())
        self.assertEqual(result["summary"]["failure_code"], "claim_text_digest_mismatch")
        claim_doc = json.loads(self.claims)
        claim_doc["records"][0]["claim_id"] = "claim-forged"
        result = self.invoke(claim_manifest_raw=json.dumps(claim_doc).encode())
        self.assertEqual(result["summary"]["failure_code"], "claim_lineage_target_mismatch")
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)

    def test_duplicate_json_keys_and_bad_csv_binding_fail_closed(self):
        result = self.invoke(selection_raw=b'{"records":[],"records":[]}')
        self.assertEqual(result["summary"]["failure_code"], "duplicate_json_key")
        result = self.invoke(csv_raw=self.csv + b"\n")
        self.assertEqual(result["summary"]["failure_code"], "lineage_test_csv_hash_mismatch")
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)

    def test_oversized_input_is_rejected_before_hashing(self):
        with patch("qta_actual_qa_source_contract._sha",
                   side_effect=AssertionError("hash_should_not_run")):
            result = self.invoke(csv_raw=b"x" * (MAX_INPUT_BYTES + 1))
        self.assertEqual(result["summary"]["failure_code"],
                         "input_byte_limit_exceeded")
        self.assertEqual(result["summary"]["validated_rows"], 0)
        self.assertIsNone(result["private_bundle"])
        self.assert_denied(result)


if __name__ == "__main__":
    unittest.main()
