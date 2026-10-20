from __future__ import annotations

import unittest
import hashlib
import json

from nextgqa_prediction_link_review import (
    PredictionLinkReviewError,
    displayed_fields_sha256,
    validate_prediction_link_receipt,
)
from qta_selection_provenance import FROZEN_SELECTOR_SOURCE_SHA256


class PredictionLinkReviewTests(unittest.TestCase):
    def setUp(self):
        self.baseline_hash = "a" * 64
        self.protocol_hash = "b" * 64
        self.video_review_hash = "c" * 64
        self.interval_hash = "d" * 64
        self.rows = [
            {"review_token": "opaque000000000000000001", "question": "What happened?",
             "options": ["A", "B", "C", "D", "E"], "claim": "A person waves.",
             "option_log_probabilities": [-1.0, -1.2, -2.0, -2.5, -3.0],
             "p_index": 0, "q_index": 1, "video_support_decision": "supports_claim",
             "reviewed_intervals_sha256": self.interval_hash},
            {"review_token": "opaque000000000000000002", "question": "Where is it?",
             "options": ["left", "right", "up", "down", "center"],
             "claim": "The item is on the left.",
             "option_log_probabilities": [-1.0, -1.2, -2.0, -2.5, -3.0],
             "p_index": 0, "q_index": 1, "video_support_decision": "supports_claim",
             "reviewed_intervals_sha256": self.interval_hash},
            {"review_token": "opaque000000000000000003", "question": "What color is it?",
             "options": ["red", "blue", "green", "white", "black"],
             "claim": "The object is blue.",
             "option_log_probabilities": [-1.0, -1.2, -2.0, -2.5, -3.0],
             "p_index": 0, "q_index": 1, "video_support_decision": "supports_claim",
             "reviewed_intervals_sha256": self.interval_hash},
        ]
        for index, row in enumerate(self.rows):
            canonical = lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")
            positions = [2, 3, 4]
            row["score_binding"] = {
                "schema_version": 2,
                "review_token": row["review_token"],
                "answer_label_order": ["A", "B", "C", "D", "E"],
                "answer_token_ids": [100 + 5 * index + offset for offset in range(5)],
                "choice_token_binding_sha256": "1" * 64,
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "options_sha256": hashlib.sha256(canonical(row["options"])).hexdigest(),
                "claim_sha256": hashlib.sha256(row["claim"].encode()).hexdigest(),
                "source_video_sha256": f"{index + 2:064x}",
                "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
                "video_support_decision": "supports_claim",
                "prompt_sha256": "3" * 64,
                "input_ids_sha256": "4" * 64,
                "input_length": 11,
                "score_position": 10,
                "score_input_token_id": 100 + index,
                "video_token_id": 32000,
                "question_positions": positions,
                "question_positions_sha256": hashlib.sha256(canonical(positions)).hexdigest(),
                "question_position_mapper_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
                "processor_batch_sha256": "5" * 64,
                "option_log_probabilities": row["option_log_probabilities"],
                "p_index": row["p_index"],
                "q_index": row["q_index"],
                "run_digests": {
                    "protocol_sha256": "6" * 64,
                    "model_sha256": "7" * 64,
                    "tokenizer_sha256": "8" * 64,
                    "scorer_sha256": "9" * 64,
                    "processor_sha256": "a" * 64,
                    "runtime_sha256": "b" * 64,
                },
            }
            row["score_binding_sha256"] = hashlib.sha256(
                canonical(row["score_binding"])).hexdigest()
        self.manifest_raw = json.dumps(
            {"schema_version": 3,
             "purpose": "qta_baseline_prediction_link_manifest_v3",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": self.rows},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.baseline_hash = hashlib.sha256(self.manifest_raw).hexdigest()
        self.expected_source_hashes = {
            row["review_token"]: {
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "options_sha256": hashlib.sha256(json.dumps(
                    row["options"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "claim_sha256": hashlib.sha256(row["claim"].encode()).hexdigest(),
                "video_support_decision": row["video_support_decision"],
                "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
            }
            for row in self.rows
        }
        self.receipt = {
            "schema_version": 1,
            "purpose": "qta_prediction_link_adjudication",
            "status": "prediction_link_review_recorded_unvalidated",
            "baseline_manifest_sha256": self.baseline_hash,
            "protocol_sha256": self.protocol_hash,
            "prior_video_review_receipt_sha256": self.video_review_hash,
            "reviewer_role": "independent_prediction_link_reviewer",
            "reviewer_private_id": "reviewer-local-id",
            "independence_attestation": "independent_from_generation_scoring_custody_and_video_review",
            "gold_and_branch_outcomes_hidden": True,
            "reviewed_at_utc": "2026-10-18T10:00:00Z",
            "records": [
                {"review_token": row["review_token"],
                 "displayed_fields_sha256": displayed_fields_sha256(
                     row, video_claim_review_receipt_sha256=self.video_review_hash),
                 "judgment": "supports_p_over_q", "rationale": "Synthetic rationale."}
                for row in self.rows
            ],
            "execution_authorization": "not_granted",
        }

    def _validate(self, **overrides):
        values = {
            "receipt": self.receipt,
            "baseline_manifest_raw": self.manifest_raw,
            "expected_baseline_manifest_sha256": self.baseline_hash,
            "expected_protocol_sha256": self.protocol_hash,
            "expected_video_review_receipt_sha256": self.video_review_hash,
            "expected_cohort_tokens": [row["review_token"] for row in self.rows],
            "expected_source_field_sha256_by_token": self.expected_source_hashes,
        }
        values.update(overrides)
        return validate_prediction_link_receipt(**values)

    def test_valid_receipt_is_structural_and_never_authorizing(self):
        result = self._validate()
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["support_judgment_count"], 3)
        self.assertTrue(result["all_expected_rows_have_supporting_self_reported_judgment"])
        self.assertFalse(result["reviewer_identity_verified"])
        self.assertFalse(result["semantic_judgment_authenticated"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_baseline_manifest_v3_is_accepted_for_review(self):
        payload = {
            "schema_version": 3,
            "purpose": "qta_baseline_prediction_link_manifest_v3",
            "video_claim_review_receipt_sha256": self.video_review_hash,
            "records": self.rows,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        receipt = {**self.receipt, "baseline_manifest_sha256": digest}
        result = self._validate(receipt=receipt, baseline_manifest_raw=raw,
                                expected_baseline_manifest_sha256=digest)
        self.assertEqual(result["requested_records"], 3)

    def test_legacy_baseline_manifest_is_rejected(self):
        legacy = json.dumps(
            {"schema_version": 1,
             "purpose": "qta_baseline_prediction_link_manifest",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": self.rows}, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()
        digest = hashlib.sha256(legacy).hexdigest()
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_manifest_schema_invalid"):
            self._validate(
                baseline_manifest_raw=legacy,
                expected_baseline_manifest_sha256=digest,
                receipt={**self.receipt, "baseline_manifest_sha256": digest})

    def test_v3_missing_or_tampered_score_binding_is_rejected(self):
        for mutation in ("remove", "tamper_hash", "tamper_content"):
            rows = [dict(row) for row in self.rows]
            if mutation == "remove":
                rows[0].pop("score_binding")
            elif mutation == "tamper_hash":
                rows[0]["score_binding_sha256"] = "f" * 64
            else:
                binding = dict(rows[0]["score_binding"])
                binding["run_digests"] = {
                    **binding["run_digests"], "model_sha256": "not-a-digest"}
                rows[0]["score_binding"] = binding
                rows[0]["score_binding_sha256"] = hashlib.sha256(json.dumps(
                    binding, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            raw = json.dumps(
                {"schema_version": 3,
                 "purpose": "qta_baseline_prediction_link_manifest_v3",
                 "video_claim_review_receipt_sha256": self.video_review_hash,
                 "records": rows}, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    PredictionLinkReviewError,
                    "baseline_manifest_(record_schema_invalid|score_binding_invalid)"):
                self._validate(
                    baseline_manifest_raw=raw,
                    expected_baseline_manifest_sha256=digest,
                    receipt={**self.receipt, "baseline_manifest_sha256": digest})

    def test_v3_duplicate_source_video_hash_is_rejected_after_rehashing(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["score_binding"] = {
            **rows[1]["score_binding"],
            "source_video_sha256": rows[0]["score_binding"]["source_video_sha256"],
        }
        rows[1]["score_binding_sha256"] = hashlib.sha256(json.dumps(
            rows[1]["score_binding"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        raw = json.dumps(
            {"schema_version": 3,
             "purpose": "qta_baseline_prediction_link_manifest_v3",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": rows}, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(
                PredictionLinkReviewError,
                "baseline_manifest_score_binding_invalid"):
            self._validate(
                baseline_manifest_raw=raw,
                expected_baseline_manifest_sha256=digest,
                receipt={**self.receipt, "baseline_manifest_sha256": digest})

    def test_non_supporting_judgment_makes_cohort_ineligible(self):
        rows = list(self.receipt["records"])
        rows[1] = {**rows[1], "judgment": "unrelated"}
        receipt = {**self.receipt, "records": rows}
        result = self._validate(receipt=receipt)
        self.assertFalse(result["all_expected_rows_have_supporting_self_reported_judgment"])
        self.assertEqual(result["support_judgment_count"], 2)

    def test_prediction_or_claim_change_breaks_display_binding(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["p_index"] = 4
        changed_binding = dict(changed[0]["score_binding"])
        changed_binding["p_index"] = 4
        changed[0]["score_binding"] = changed_binding
        changed[0]["score_binding_sha256"] = hashlib.sha256(json.dumps(
            changed_binding, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        changed_raw = json.dumps(
            {"schema_version": 3,
             "purpose": "qta_baseline_prediction_link_manifest_v3",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": changed},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        changed_hash = hashlib.sha256(changed_raw).hexdigest()
        changed_receipt = {**self.receipt,
                           "baseline_manifest_sha256": changed_hash}
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_manifest_score_binding_invalid"):
            self._validate(
                receipt=changed_receipt,
                baseline_manifest_raw=changed_raw,
                expected_baseline_manifest_sha256=changed_hash)

    def test_manifest_bytes_and_source_claim_binding_are_checked(self):
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_manifest_bytes_hash_mismatch"):
            self._validate(baseline_manifest_raw=self.manifest_raw + b" ")
        changed_hashes = dict(self.expected_source_hashes)
        changed_hashes["opaque000000000000000001"] = {
            **changed_hashes["opaque000000000000000001"], "claim_sha256": "c" * 64}
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_source_field_binding_mismatch"):
            self._validate(expected_source_field_sha256_by_token=changed_hashes)
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "expected_source_digest_inventory_invalid"):
            self._validate(expected_source_field_sha256_by_token={"opaque000000000000000001":
                self.expected_source_hashes["opaque000000000000000001"]})

    def test_manifest_protocol_record_order_and_exact_schema_are_bound(self):
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_manifest_binding_mismatch"):
            self._validate(expected_protocol_sha256="c" * 64)
        reordered = {**self.receipt, "records": list(reversed(self.receipt["records"]))}
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_record_order_mismatch"):
            self._validate(receipt=reordered)
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_manifest_cohort_count_mismatch"):
            self._validate(expected_cohort_tokens=["opaque000000000000000001"])
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_receipt_schema_invalid"):
            self._validate(receipt={**self.receipt, "gold_answer": "A"})

    def test_missing_rationale_or_hidden_outcome_attestation_fails_closed(self):
        rows = list(self.receipt["records"])
        rows[0] = {**rows[0], "rationale": "  "}
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_rationale_invalid"):
            self._validate(receipt={**self.receipt, "records": rows})
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_reviewer_attestation_invalid"):
            self._validate(receipt={**self.receipt,
                                    "gold_and_branch_outcomes_hidden": False})

    def test_boolean_schema_versions_and_malformed_scores_or_unicode_are_rejected(self):
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "prediction_link_receipt_status_invalid"):
            self._validate(receipt={**self.receipt, "schema_version": True})

        bool_manifest = json.dumps(
            {"schema_version": True,
             "purpose": "qta_baseline_prediction_link_manifest_v3",
             "records": self.rows}, separators=(",", ":")).encode()
        bool_hash = hashlib.sha256(bool_manifest).hexdigest()
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_manifest_schema_invalid"):
            self._validate(
                baseline_manifest_raw=bool_manifest,
                expected_baseline_manifest_sha256=bool_hash,
                receipt={**self.receipt, "baseline_manifest_sha256": bool_hash})

        for changed_row in (
                {**self.rows[0], "option_log_probabilities": [10**4000, -2, -3, -4, -5]},
                {**self.rows[0], "question": "invalid-\ud800-surrogate"}):
            rows = [changed_row, self.rows[1], self.rows[2]]
            raw = json.dumps(
                {"schema_version": 3,
                 "purpose": "qta_baseline_prediction_link_manifest_v3",
                 "video_claim_review_receipt_sha256": self.video_review_hash,
                 "records": rows}, ensure_ascii=True,
                separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            with self.subTest(row=changed_row["review_token"]):
                with self.assertRaisesRegex(PredictionLinkReviewError,
                                            "baseline_manifest_score_binding_invalid"):
                    self._validate(
                        baseline_manifest_raw=raw,
                        expected_baseline_manifest_sha256=digest,
                        receipt={**self.receipt,
                                 "baseline_manifest_sha256": digest})

    def test_video_review_receipt_and_intervals_are_bound(self):
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_video_review_receipt_binding_mismatch"):
            self._validate(expected_video_review_receipt_sha256="e" * 64)
        changed = [dict(row) for row in self.rows]
        changed[0]["reviewed_intervals_sha256"] = "e" * 64
        changed_binding = dict(changed[0]["score_binding"])
        changed_binding["reviewed_intervals_sha256"] = "e" * 64
        changed[0]["score_binding"] = changed_binding
        changed[0]["score_binding_sha256"] = hashlib.sha256(json.dumps(
            changed_binding, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        raw = json.dumps({"schema_version": 3,
                          "purpose": "qta_baseline_prediction_link_manifest_v3",
                          "video_claim_review_receipt_sha256": self.video_review_hash,
                          "records": changed}, sort_keys=True,
                         separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(PredictionLinkReviewError,
                                    "baseline_video_review_binding_mismatch"):
            self._validate(baseline_manifest_raw=raw,
                           expected_baseline_manifest_sha256=digest,
                           receipt={**self.receipt, "baseline_manifest_sha256": digest})


if __name__ == "__main__":
    unittest.main()
