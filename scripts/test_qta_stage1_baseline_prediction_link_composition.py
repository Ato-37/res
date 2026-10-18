from __future__ import annotations

import hashlib
import json
import unittest

from nextgqa_prediction_link_review import (
    displayed_fields_sha256,
    validate_prediction_link_receipt,
)
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_baseline_pq_candidate import CandidateError, build_baseline_pq_candidate


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class _SyntheticTokenizer:
    """Deterministic offset-aware tokenizer for this CPU structural fixture."""

    def _tokenize(self, text):
        ids, offsets = [], []
        cursor = 0
        marker = "<|video_pad|>"
        while cursor < len(text):
            if text.startswith(marker, cursor):
                ids.append(32000)
                offsets.append((cursor, cursor + len(marker)))
                cursor += len(marker)
            else:
                ids.append(ord(text[cursor]))
                offsets.append((cursor, cursor + 1))
                cursor += 1
        return ids, offsets

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._tokenize(text)[0]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class Stage1BaselinePredictionLinkCompositionTests(unittest.TestCase):
    """Join one Stage-1 adapter/QA fixture through baseline and link validation."""

    def setUp(self):
        from test_qta_stage1_source_pin_adapter import QtaStage1SourcePinAdapterTests

        self.adapter = QtaStage1SourcePinAdapterTests()
        self.adapter.setUp()
        self.addCleanup(self.adapter.doCleanups)

        # These are the actual outputs of the same synthetic Stage-1 and QA
        # inputs used by QtaStage1SourcePinAdapterTests, not parallel fixtures.
        self.pins = self.adapter.invoke()
        qa_result = bind_qta_actual_qa_source(**self.adapter.qa_inputs)
        self.assertEqual(qa_result["summary"]["status"], "bound_unvalidated")
        self.qa = json.loads(qa_result["private_bundle"]["raw_json"])["qa_source"]
        self.claims = json.loads(
            self.adapter.qa_inputs["claim_manifest_raw"])["records"]
        self.tokens = self.pins["expected_cohort_tokens"]
        self.review_sha = self.pins[
            "expected_video_claim_review_receipt_sha256"]
        self.protocol_sha = self.adapter.qa_inputs[
            "expected_qta_protocol_sha256"]
        self.run_digests = {
            "protocol_sha256": self.protocol_sha,
            "model_sha256": _sha(b"synthetic model identity; not loaded"),
            "tokenizer_sha256": _sha(b"synthetic tokenizer identity"),
            "scorer_sha256": _sha(b"synthetic score generator"),
            "processor_sha256": _sha(b"synthetic processor identity"),
            "runtime_sha256": _sha(b"synthetic runtime identity"),
        }
        self.tokenizer = _SyntheticTokenizer()
        self.prompts = {}
        self.rows = []
        bindings = self.pins["expected_baseline_candidate_source_bindings_by_token"]
        source_rows = self.pins["source_rows"]
        self.assertEqual(len(self.qa["records"]), 3)
        self.assertEqual(len(self.claims), 3)
        self.assertEqual([row["review_token"] for row in source_rows], self.tokens)

        # Scores are deliberately synthetic placeholders. This test exercises
        # joins and candidate validation, not model inference or faithfulness.
        synthetic_scores = [-1.5, -1.6, -1.7, -1.8, -1.9]
        for token, source, qa_row, claim in zip(
                self.tokens, source_rows, self.qa["records"], self.claims, strict=True):
            self.assertEqual(claim["video_id"], qa_row["video_id"])
            prompt = f"Video <|video_pad|> {qa_row['question']} Answer:"
            self.prompts[token] = prompt
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            # Three synthetic video placeholders model processor expansion;
            # no video or processor is loaded.
            input_ids = [item for value in raw_prompt_ids
                         for item in ([value] * 3 if value == 32000 else [value])]
            input_ids_sha = _sha(_canonical(input_ids))
            pin = bindings[token]
            self.rows.append({
                "review_token": token,
                "question": qa_row["question"],
                "options": qa_row["choices"],
                "claim": claim["claim_text"],
                "video_support_decision": source["video_support_decision"],
                "reviewed_intervals_sha256": source["reviewed_intervals_sha256"],
                "source_video_sha256": source["source_video_sha256"],
                "option_log_probabilities": list(synthetic_scores),
                "answer_token_ids": list(map(ord, "ABCDE")),
                "input_ids": input_ids,
                "input_ids_sha256": input_ids_sha,
                "prompt_sha256": _sha(prompt.encode("utf-8")),
                "input_length": len(input_ids),
                "score_position": len(input_ids) - 1,
                "score_input_token_id": input_ids[-1],
                "processor_batch_sha256": _sha(
                    f"synthetic processor batch:{token}".encode("ascii")),
                "scored_option_token_ids": list(map(ord, "ABCDE")),
                "video_token_id": 32000,
            })
            self.assertEqual(pin["source_video_sha256"], source["source_video_sha256"])

        self.input_raw = _canonical({
            "schema_version": 1,
            "purpose": "qta_baseline_full_forward_candidate_input",
            "video_claim_review_receipt_sha256": self.review_sha,
            "run_digests": self.run_digests,
            "records": self.rows,
        })

    def _build_baseline(self, *, source_bindings=None):
        return build_baseline_pq_candidate(
            input_raw=self.input_raw,
            expected_cohort_tokens=self.tokens,
            expected_run_digests=self.run_digests,
            expected_video_claim_review_receipt_sha256=self.review_sha,
            expected_source_bindings_by_token=(source_bindings or
                self.pins["expected_baseline_candidate_source_bindings_by_token"]),
            rendered_prompts_by_token=self.prompts,
            tokenizer=self.tokenizer,
        )

    def test_same_stage1_pins_and_qa_inputs_reach_prediction_link_validator(self):
        candidate = self._build_baseline()
        manifest_raw = candidate["baseline_manifest_raw"]
        manifest_sha = candidate["baseline_manifest_sha256"]
        manifest = json.loads(manifest_raw)
        self.assertEqual(candidate["effective_records"], 3)
        self.assertEqual(manifest["video_claim_review_receipt_sha256"], self.review_sha)
        self.assertEqual([row["review_token"] for row in manifest["records"]], self.tokens)

        expected_source = {}
        for token, source in zip(self.tokens, self.pins["source_rows"], strict=True):
            expected_source[token] = {
                key: self.pins["expected_baseline_candidate_source_bindings_by_token"][token][key]
                for key in ("question_sha256", "options_sha256", "claim_sha256",
                            "video_support_decision", "reviewed_intervals_sha256")
            }

        receipt = {
            "schema_version": 1,
            "purpose": "qta_prediction_link_adjudication",
            "status": "prediction_link_review_recorded_unvalidated",
            "baseline_manifest_sha256": manifest_sha,
            "protocol_sha256": self.protocol_sha,
            "prior_video_review_receipt_sha256": self.review_sha,
            "reviewer_role": "independent_prediction_link_reviewer",
            "reviewer_private_id": "synthetic-reviewer",
            "independence_attestation":
                "independent_from_generation_scoring_custody_and_video_review",
            "gold_and_branch_outcomes_hidden": True,
            "reviewed_at_utc": "2026-10-19T00:00:00Z",
            "records": [{
                "review_token": row["review_token"],
                "displayed_fields_sha256": displayed_fields_sha256(
                    row, video_claim_review_receipt_sha256=self.review_sha),
                "judgment": "supports_p_over_q",
                "rationale": "Synthetic structural composition only.",
            } for row in manifest["records"]],
            "execution_authorization": "not_granted",
        }
        report = validate_prediction_link_receipt(
            receipt=receipt,
            baseline_manifest_raw=manifest_raw,
            expected_baseline_manifest_sha256=manifest_sha,
            expected_protocol_sha256=self.protocol_sha,
            expected_video_review_receipt_sha256=self.review_sha,
            expected_cohort_tokens=self.tokens,
            expected_source_field_sha256_by_token=expected_source,
        )
        self.assertEqual(report["status"],
                         "prediction_link_review_structurally_valid_unvalidated")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["support_judgment_count"], 3)
        self.assertFalse(report["reviewer_identity_verified"])
        self.assertFalse(report["semantic_judgment_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(candidate["score_forwards"], 0)
        self.assertFalse(candidate["model_load_permitted"])

    def test_stale_stage1_interval_or_video_pin_is_rejected(self):
        original = self.pins["expected_baseline_candidate_source_bindings_by_token"]
        token = self.tokens[0]
        for field in ("reviewed_intervals_sha256", "source_video_sha256"):
            with self.subTest(field=field):
                stale = {key: dict(value) for key, value in original.items()}
                stale[token][field] = "0" * 64
                with self.assertRaisesRegex(CandidateError,
                                            "baseline_video_review_binding_mismatch"):
                    self._build_baseline(source_bindings=stale)


if __name__ == "__main__":
    unittest.main()
