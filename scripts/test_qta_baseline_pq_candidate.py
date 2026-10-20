from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from qta_baseline_pq_candidate import (
    CandidateError,
    build_baseline_pq_candidate,
)
from qta_choice_token_binding import (
    build_choice_token_binding,
    resolve_choice_continuation_ids,
    validate_processed_choice_suffix,
)
import qta_baseline_pq_candidate as baseline_candidate
from render_nextgqa_prediction_link_review import render_prediction_link_review_page


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class BaselinePqCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [f"opaque-{index:017d}" for index in range(1, 4)]
        self.run_digests = {
            key: sha(key) for key in (
                "protocol_sha256", "model_sha256", "tokenizer_sha256",
                "scorer_sha256", "processor_sha256", "runtime_sha256")
        }
        self.answer_ids = list(map(ord, "ABCDE"))
        class FakeTokenizer:
            def _tokenize(inner_self, text):
                ids, offsets = [], []
                cursor = 0
                marker = "<|video_pad|>"
                while cursor < len(text):
                    if text.startswith(marker, cursor):
                        ids.append(99)
                        offsets.append((cursor, cursor + len(marker)))
                        cursor += len(marker)
                    else:
                        ids.append(ord(text[cursor]))
                        offsets.append((cursor, cursor + 1))
                        cursor += 1
                return ids, offsets

            def encode(inner_self, text, add_special_tokens=False):
                del add_special_tokens
                return inner_self._tokenize(text)[0]

            def __call__(inner_self, text, add_special_tokens=False,
                         return_offsets_mapping=False):
                del add_special_tokens
                ids, offsets = inner_self._tokenize(text)
                result = {"input_ids": ids}
                if return_offsets_mapping:
                    result["offset_mapping"] = offsets
                return result
        self.tokenizer = FakeTokenizer()
        self.prompts = {}
        self.video_review_sha = sha("prior-video-review")
        self.rows = []
        self.source_pins = {}
        for i, token in enumerate(self.tokens):
            question = f"Question {i}?"
            prompt = f"Video <|video_pad|> {question} Answer:"
            self.prompts[token] = prompt
            options = [f"choice {i}-{letter}" for letter in "ABCDE"]
            claim = f"Claim {i}."
            input_ids = []
            for token_id in self.tokenizer.encode(prompt, add_special_tokens=False):
                input_ids.extend([token_id] * 3 if token_id == 99 else [token_id])
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
                "prompt_sha256": sha(prompt),
                "input_length": len(input_ids),
                "score_position": len(input_ids) - 1,
                "score_input_token_id": input_ids[-1],
                "processor_batch_sha256": sha(f"batch-{i}"),
                "scored_option_token_ids": self.answer_ids,
                "video_token_id": 99,
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
            "rendered_prompts_by_token": self.prompts,
            "tokenizer": self.tokenizer,
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
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["purpose"], "qta_baseline_prediction_link_manifest_v3")
        self.assertTrue(result["question_position_mapping_verified"])
        self.assertTrue(result["rendered_prompt_video_expansion_mapping_verified"])
        self.assertFalse(result["processor_prompt_association_verified"])
        binding = manifest["records"][0]["score_binding"]
        self.assertEqual(binding["schema_version"], 2)
        self.assertTrue(binding["question_positions"])
        self.assertEqual(binding["video_token_id"], 99)
        self.assertEqual(binding["question_position_mapper_sha256"],
                         baseline_candidate.FROZEN_SELECTOR_SOURCE_SHA256)
        self.assertEqual(len(manifest["records"]), 3)
        for row in manifest["records"]:
            binding = row["score_binding"]
            self.assertEqual(binding["answer_token_ids"], self.answer_ids)
            self.assertEqual(binding["score_position"], binding["input_length"] - 1)
            self.assertEqual(binding["p_index"], row["p_index"])
            self.assertEqual(binding["q_index"], row["q_index"])
            self.assertEqual(row["score_binding_sha256"], hashlib.sha256(
                json.dumps(binding, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode()).hexdigest())
        self.assertFalse(result["model_load_permitted"])
        self.assertTrue(result["text_token_binding_verified"])
        self.assertFalse(result["processor_prompt_association_verified"])
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
        self.assertNotIn('"score_binding"', worksheet)
        self.assertNotIn('"answer_token_ids"', worksheet)
        self.assertNotIn('"choice_token_binding_sha256"', worksheet)
        self.assertNotIn('"input_ids_sha256"', worksheet)
        for name in ("model_sha256", "tokenizer_sha256", "scorer_sha256",
                     "processor_sha256"):
            digest = self.run_digests[name]
            self.assertNotIn(digest, worksheet)

    def test_requires_fixed_three_rows_and_expected_three_tokens(self):
        for tokens in (["opaque-1", "opaque-2"], ["opaque-1"] * 3):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(
                    CandidateError, "expected_three_unique_cohort_tokens_required"):
                self._build(expected_cohort_tokens=tokens)
        self.input_obj["records"].pop()
        with self.assertRaisesRegex(CandidateError,
                                    "baseline_input_fixed_three_rows_required"):
            self._build()

    def test_rejects_duplicate_source_video_bytes_across_distinct_tokens(self):
        rows = [dict(row) for row in self.rows]
        duplicate_video_sha = rows[0]["source_video_sha256"]
        rows[1]["source_video_sha256"] = duplicate_video_sha
        pins = {token: dict(value) for token, value in self.source_pins.items()}
        pins[self.tokens[1]]["source_video_sha256"] = duplicate_video_sha
        input_obj = {**self.input_obj, "records": rows}
        input_raw = json.dumps(input_obj, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        with self.assertRaisesRegex(CandidateError,
                                    "baseline_duplicate_source_video_sha256"):
            self._build(input_raw=input_raw,
                        expected_source_bindings_by_token=pins)

    def test_candidate_manifest_output_size_is_capped(self):
        with patch.object(baseline_candidate, "MAX_CANDIDATE_MANIFEST_BYTES", 1):
            with self.assertRaisesRegex(CandidateError, "baseline_manifest_bytes_invalid"):
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
            ("video_token", {**self.rows[0], "video_token_id": 100},
             "baseline_question_position_mapping_invalid"),
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

    def test_question_position_binding_requires_exact_video_expansion(self):
        result = self._build()
        manifest = json.loads(result["baseline_manifest_raw"])
        binding = manifest["records"][0]["score_binding"]
        positions = binding["question_positions"]
        visual_positions = [i for i, value in enumerate(self.rows[0]["input_ids"])
                            if value == self.rows[0]["video_token_id"]]
        self.assertTrue(positions)
        self.assertGreater(positions[0], max(visual_positions))
        self.assertLess(positions[-1], binding["score_position"])
        self.assertEqual(binding["question_positions_sha256"], hashlib.sha256(
            json.dumps(positions, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest())
        original_obj, original_rows = self.input_obj, self.rows
        changed = dict(self.rows[0])
        changed_ids = list(changed["input_ids"])
        video_position = changed_ids.index(changed["video_token_id"])
        changed_ids[video_position] += 1
        changed["input_ids"] = changed_ids
        changed["input_ids_sha256"] = hashlib.sha256(json.dumps(
            changed_ids, separators=(",", ":")).encode()).hexdigest()
        self.rows = [changed, *self.rows[1:]]
        self.input_obj = {**self.input_obj, "records": self.rows}
        try:
            with self.assertRaisesRegex(
                    CandidateError, "baseline_question_position_mapping_invalid"):
                self._build()
        finally:
            self.input_obj, self.rows = original_obj, original_rows

    def test_question_offset_geometry_rejects_out_of_range_and_overlap(self):
        base = self.tokenizer

        class MalformedOffsetTokenizer:
            def __init__(inner_self, kind):
                inner_self.kind = kind

            def encode(inner_self, text, add_special_tokens=False):
                return base.encode(text, add_special_tokens=add_special_tokens)

            def __call__(inner_self, text, add_special_tokens=False,
                         return_offsets_mapping=False):
                result = base(text, add_special_tokens=add_special_tokens,
                              return_offsets_mapping=return_offsets_mapping)
                if return_offsets_mapping:
                    offsets = list(result["offset_mapping"])
                    if inner_self.kind == "out_of_range":
                        offsets[-1] = (len(text), len(text) + 1)
                    else:
                        offsets[2] = (offsets[1][0], offsets[1][1] + 1)
                    result["offset_mapping"] = offsets
                return result

        for kind in ("out_of_range", "overlap"):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                    CandidateError, "baseline_question_position_mapping_invalid"):
                self._build(tokenizer=MalformedOffsetTokenizer(kind))

    def test_rejects_self_consistent_but_forged_ids_and_prompt_hash(self):
        original = self.rows[0]
        forged_ids = [201, 202, 203, 204, 205]
        self.rows[0] = {**original, "answer_token_ids": forged_ids,
                        "scored_option_token_ids": forged_ids}
        self.input_obj["records"] = self.rows
        with self.assertRaisesRegex(CandidateError,
                                    "baseline_option_token_binding_invalid"):
            self._build()
        self.rows[0] = original
        self.input_obj["records"] = self.rows

        self.rows[0] = {**original, "prompt_sha256": sha("forged-prompt")}
        self.input_obj["records"] = self.rows
        with self.assertRaisesRegex(CandidateError, "baseline_prompt_binding_mismatch"):
            self._build()

    def test_tokenizer_prompt_merge_and_multitoken_suffix_fail_closed(self):
        class RetokenizingTokenizer:
            def encode(inner_self, text, add_special_tokens=False):
                del add_special_tokens
                return [1, 2] if text == "prompt-0" else [1, 99, 100]

        with self.assertRaisesRegex(CandidateError,
                                    "choice_prompt_tokenization_or_binding_failed"):
            self._build(tokenizer=RetokenizingTokenizer())

        class MultiTokenTokenizer:
            def encode(inner_self, text, add_special_tokens=False):
                del add_special_tokens
                if text == "prompt-0":
                    return [1, 2]
                if text == "prompt-0A":
                    return [1, 2, 9, 10]
                return [1, 2, ord(text[-1])]

        with self.assertRaisesRegex(CandidateError,
                                    "choice_prompt_tokenization_or_binding_failed"):
            self._build(tokenizer=MultiTokenTokenizer())

        class OneLabelChangedTokenizer:
            def encode(inner_self, text, add_special_tokens=False):
                del add_special_tokens
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if text.endswith("E") and text[:-1] in self.prompts.values():
                    ids[-1] += 1
                return ids

            def __call__(inner_self, *args, **kwargs):
                return self.tokenizer(*args, **kwargs)

        with self.assertRaisesRegex(CandidateError,
                                    "baseline_option_token_binding_invalid"):
            self._build(tokenizer=OneLabelChangedTokenizer())

    def test_tokenizer_binding_rejects_duplicate_ids_and_preserves_processor_suffix(self):
        class DuplicateTokenizer:
            def encode(inner_self, text, add_special_tokens=False):
                del add_special_tokens
                return [1] if text == "prefix" else [1, 7]

        with self.assertRaisesRegex(ValueError, "choice_continuation_ids_not_distinct"):
            resolve_choice_continuation_ids(DuplicateTokenizer(), "prefix")
        with self.assertRaisesRegex(ValueError, "choice_prompt_or_label_order_invalid"):
            resolve_choice_continuation_ids(self.tokenizer, "prompt-0", "EDCBA")
        binding = build_choice_token_binding(
            tokenizer=self.tokenizer, rendered_prompt="prompt-0",
            tokenizer_sha256=self.run_digests["tokenizer_sha256"])
        self.assertEqual(binding["answer_token_ids"], self.answer_ids)
        validate_processed_choice_suffix([1, 2, 3], [1, 2, 3, ord("A")], ord("A"))
        with self.assertRaisesRegex(ValueError, "processed_choice_suffix_prefix_mismatch"):
            validate_processed_choice_suffix([1, 2, 3], [1, 8, 3, ord("A")], ord("A"))
        with self.assertRaisesRegex(ValueError, "processed_choice_suffix_token_mismatch"):
            validate_processed_choice_suffix([1, 2, 3], [1, 2, 3, ord("B")], ord("A"))
        with self.assertRaisesRegex(ValueError, "processed_choice_suffix_input_invalid"):
            validate_processed_choice_suffix([1], [True, 1], 1)

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
