from __future__ import annotations

import hashlib
import json
import unittest

from qta_baseline_pq_candidate import (
    CandidateError,
    build_baseline_pq_candidate,
)
from render_nextgqa_prediction_link_review import render_prediction_link_review_page


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class BaselinePqCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tokens = ["opaque-1", "opaque-2", "opaque-3"]
        self.run_digests = {
            key: sha(key) for key in (
                "protocol_sha256", "model_sha256", "tokenizer_sha256",
                "scorer_sha256", "processor_sha256")
        }
        self.answer_ids = [101, 102, 103, 104, 105]
        self.video_review_sha = sha("prior-video-review")
        self.rows = []
        self.source_pins = {}
        for i, token in enumerate(self.tokens):
            question = f"Question {i}?"
            options = [f"choice {i}-{letter}" for letter in "ABCDE"]
            claim = f"Claim {i}."
            input_ids = [1, 10 + i, 20 + i]
            input_sha = hashlib.sha256(json.dumps(
                input_ids, separators=(",", ":")).encode()).hexdigest()
            video_sha, intervals_sha = sha(f"video-{i}"), sha(f"intervals-{i}")
            row = {
                "review_token": token,
                "question": question,
                "options": options,
                "claim": claim,
                "video_support_decision": "supports_claim",
                "reviewed_intervals_sha256": intervals_sha,
                "source_video_sha256": video_sha,
                "option_log_probabilities": [-1.5, -1.6, -1.7, -1.8, -1.9],
                "answer_token_ids": self.answer_ids,
                "input_ids": input_ids,
                "input_ids_sha256": input_sha,
                "prompt_sha256": sha(f"prompt-{i}"),
                "input_length": len(input_ids),
                "score_position": len(input_ids) - 1,
                "score_input_token_id": input_ids[-1],
                "processor_batch_sha256": sha(f"batch-{i}"),
                "scored_option_token_ids": self.answer_ids,
            }
            self.rows.append(row)
            self.source_pins[token] = {
                "question_sha256": sha(question),
                "options_sha256": hashlib.sha256(json.dumps(
                    options, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "claim_sha256": sha(claim),
                "source_video_sha256": video_sha,
                "reviewed_intervals_sha256": intervals_sha,
                "video_support_decision": "supports_claim",
            }
        self.input_obj = {
            "schema_version": 1,
            "purpose": "qta_baseline_full_forward_candidate_input",
            "video_claim_review_receipt_sha256": self.video_review_sha,
            "run_digests": self.run_digests,
            "records": self.rows,
        }

    def _build(self, **overrides):
        kwargs = {
            "input_raw": json.dumps(self.input_obj, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode(),
            "expected_cohort_tokens": self.tokens,
            "expected_run_digests": self.run_digests,
            "expected_video_claim_review_receipt_sha256": self.video_review_sha,
            "expected_source_bindings_by_token": self.source_pins,
            "expected_answer_token_ids": self.answer_ids,
        }
        kwargs.update(overrides)
        return build_baseline_pq_candidate(**kwargs)

    def test_candidate_recomputes_pq_ties_and_is_never_authorizing(self):
        self.input_obj["records"][0]["option_log_probabilities"] = [
            -1.6, -1.5, -1.5, -1.8, -1.9]
        result = self._build()
        self.assertEqual(result["p_indices"], [1, 0, 0])
        self.assertEqual(result["q_indices"], [2, 1, 1])
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertEqual(hashlib.sha256(result["baseline_manifest_raw"]).hexdigest(),
                         result["baseline_manifest_sha256"])
        manifest = json.loads(result["baseline_manifest_raw"])
        self.assertEqual(manifest["purpose"], "qta_baseline_prediction_link_manifest")
        self.assertEqual(len(manifest["records"]), 3)
        self.assertFalse(result["model_load_permitted"])
        self.assertEqual(result["branch_forwards"], 0)
        self.assertEqual(result["score_forwards"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertFalse(result["identity_proof"])
        self.assertFalse(result["semantic_proof"])
        worksheet = render_prediction_link_review_page(
            baseline_manifest_raw=result["baseline_manifest_raw"],
            expected_baseline_manifest_sha256=result["baseline_manifest_sha256"],
            protocol_sha256=self.run_digests["protocol_sha256"],
            video_claim_review_receipt_sha256=self.video_review_sha)
        self.assertEqual(worksheet.count('class="p-choice"'), 3)
        self.assertEqual(worksheet.count('class="q-choice"'), 3)

    def test_requires_fixed_three_rows_and_expected_three_tokens(self):
        for tokens in (["opaque-1", "opaque-2"], ["opaque-1"] * 3):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(
                    CandidateError, "expected_three_unique_cohort_tokens_required"):
                self._build(expected_cohort_tokens=tokens)
        self.input_obj["records"].pop()
        with self.assertRaisesRegex(CandidateError,
                                    "baseline_input_fixed_three_rows_required"):
            self._build()

    def test_rejects_wrong_run_source_prompt_position_and_token_bindings(self):
        cases = [
            ("run_digests", {**self.input_obj,
                              "run_digests": {**self.run_digests,
                                              "model_sha256": sha("other")}},
             "baseline_run_digest_binding_mismatch"),
            ("source", {**self.rows[0], "claim": "changed claim"},
             "baseline_source_field_binding_mismatch"),
            ("position", {**self.rows[0], "score_position": 1},
             "baseline_score_position_binding_invalid"),
            ("token", {**self.rows[0], "scored_option_token_ids": [1, 2, 3, 4, 5]},
             "baseline_option_token_binding_invalid"),
            ("input_hash", {**self.rows[0], "input_ids_sha256": sha("other")},
             "processor_input_ids_digest_mismatch"),
        ]
        for label, changed, error in cases:
            with self.subTest(label=label):
                original_obj, original_rows = self.input_obj, self.rows
                if label == "run_digests":
                    self.input_obj = changed
                else:
                    self.rows = [changed, *self.rows[1:]]
                    self.input_obj = {**self.input_obj, "records": self.rows}
                try:
                    with self.assertRaisesRegex(CandidateError, error):
                        self._build()
                finally:
                    self.input_obj, self.rows = original_obj, original_rows

    def test_rejects_duplicate_keys_and_nonfinite_scores(self):
        raw = (b'{"schema_version":1,"schema_version":1,"purpose":'
               b'"qta_baseline_full_forward_candidate_input"}')
        with self.assertRaisesRegex(CandidateError, "duplicate_json_key"):
            self._build(input_raw=raw)
        self.input_obj["records"][0]["option_log_probabilities"][0] = float("nan")
        with self.assertRaisesRegex(CandidateError, "nonfinite_json"):
            self._build()

    def test_rejects_wrong_receipt_and_impossible_log_probability_vectors(self):
        changed = {**self.input_obj,
                   "video_claim_review_receipt_sha256": "f" * 64}
        original = self.input_obj
        self.input_obj = changed
        try:
            with self.assertRaisesRegex(
                    CandidateError, "baseline_video_claim_review_receipt_binding_mismatch"):
                self._build()
        finally:
            self.input_obj = original

        for scores in ([0.1, -2.0, -2.1, -2.2, -2.3],
                       [-0.1, -0.2, -0.3, -0.4, -0.5],
                       [-1.0, -1.1, -1.2, -1.3, -1.4],
                       [-(2**53 + 1), -(2**53), -2.0, -3.0, -4.0]):
            with self.subTest(scores=scores):
                self.input_obj["records"][0]["option_log_probabilities"] = scores
                with self.assertRaisesRegex(
                        CandidateError, "baseline_full_ae_log_probabilities_invalid"):
                    self._build()

    def test_rejects_review_decision_disagreement_with_frozen_binding(self):
        row = {**self.rows[0], "video_support_decision": "supports_claim"}
        pins = {**self.source_pins,
                self.tokens[0]: {**self.source_pins[self.tokens[0]],
                                 "video_support_decision": "unsupported"}}
        self.input_obj["records"] = [row, *self.rows[1:]]
        with self.assertRaisesRegex(
                CandidateError, "baseline_video_support_decision_binding_mismatch"):
            self._build(expected_source_bindings_by_token=pins)


if __name__ == "__main__":
    unittest.main()
