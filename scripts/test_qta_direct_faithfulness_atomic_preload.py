from __future__ import annotations

import hashlib
import json
import copy
import csv
import io
import unittest
from unittest import mock
from PIL import Image

import torch

import qta_actual_candidate_input_bridge as bridge
import qta_direct_faithfulness_atomic_preload as atomic
import qta_prebaseline_transaction_preflight as phase1_tx
from qta_signed_prebaseline_live_batches import verify_and_bind_signed_phase1_batches
from qta_processor_batch_digest_manifest_v1 import build_manifest
from qta_prompt_answer_token_bindings_v1 import build_qta_prompt_answer_token_bindings
from test_qta_prebaseline_transaction_preflight import QtaPrebaselineTransactionPreflightTests
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from test_qta_actual_candidate_input_bridge import QtaActualCandidateInputBridgeTests
from test_qta_prompt_answer_token_bindings_v1 import (
    PromptAnswerTokenBindingsTests, _prompt_inputs, _sampler_rows,
)


VIDEO = "<|video_pad|>"


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        ids, cursor = [], 0
        while cursor < len(text):
            if text.startswith(VIDEO, cursor):
                ids.append(99)
                cursor += len(VIDEO)
            else:
                ids.append(ord(text[cursor]) + 100)
                cursor += 1
        return ids


class _Processor:
    video_token_id = 99

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = next(part["text"] for part in messages[0]["content"]
                    if part["type"] == "text")
        return VIDEO + "\n" + text


def _prompt_material(tokens, qa_rows=None):
    rows = []
    for i, token in enumerate(tokens):
        qa = (qa_rows[i] if qa_rows is not None else {
            "question": f"Question {i}?",
            "choices": [f"Choice {i}-{letter}" for letter in "ABCDE"]})
        rows.append({"review_token": token, "question": qa["question"],
            "choices": list(qa["choices"]),
            "frames": [Image.new("RGB", (2, 2), (i, j, 10)) for j in range(32)],
            "sample_fps": [1, 1]})
    tok = _Tokenizer()
    return rows, _Processor(tok), tok


def _bound_prompt_material(tokens, sources, qa_rows=None):
    prompt_inputs, processor, tokenizer = _prompt_material(tokens, qa_rows)
    sampler_rows = _sampler_rows(prompt_inputs)
    expected, manifest_rows, batches, live_rows, binding_rows = [], [], [], [], []
    runtime_sha = hashlib.sha256(b"runtime").hexdigest()
    for i, (source_input, token, source_sha, sampler_row) in enumerate(zip(
            prompt_inputs, tokens, sources, sampler_rows, strict=True)):
        sampler_row["source_sha256"] = source_sha
        question_sha = hashlib.sha256(source_input["question"].encode()).hexdigest()
        options_sha = hashlib.sha256(json.dumps(source_input["choices"],
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        text = (f"{source_input['question']}\nOptions:\n" +
            "\n".join(f"{label}. {choice}" for label, choice in
                       zip("ABCDE", source_input["choices"], strict=True)) +
            "\nChoose the best answer. Respond with one letter only.\nAnswer:")
        rendered = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "video", "video": source_input["frames"], "sample_fps": 1.0},
            {"type": "text", "text": text}]}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        answer_ids = [tokenizer.encode(rendered + label, add_special_tokens=False)[len(prompt_ids)]
                      for label in "ABCDE"]
        token_sha = hashlib.sha256(b"test-tokenizer").hexdigest()
        prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
        prompt_ids_sha = hashlib.sha256(json.dumps(prompt_ids, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        choice_sha = hashlib.sha256(json.dumps({"labels": list("ABCDE"),
            "prompt_sha256": prompt_sha, "tokenizer_sha256": token_sha,
            "prompt_token_ids_sha256": prompt_ids_sha,
            "answer_token_ids": answer_ids}, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")).hexdigest()
        at = prompt_ids.index(processor.video_token_id)
        full_ids = prompt_ids[:at] + [processor.video_token_id] * 32 + prompt_ids[at + 1:]
        batch = FrozenBatch({"input_ids": torch.tensor([full_ids], dtype=torch.long),
            "video_grid_thw": torch.tensor([[32, 1, 1]], dtype=torch.long),
            "pixels": torch.arange(32, dtype=torch.float32) + i})
        batches.append(batch)
        batch_digest = batch_sha256(batch)
        input_sha = hashlib.sha256(json.dumps(full_ids, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        proc_sha = "a" * 64
        sampler_sha = hashlib.sha256(json.dumps(sampler_row, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        expected.append({"review_token": token, "source_video_sha256": source_sha,
            "question_sha256": question_sha, "options_sha256": options_sha,
            "processor_fingerprint_sha256": proc_sha, "tokenizer_sha256": token_sha,
            "sampler_row_sha256": sampler_sha})
        manifest_rows.append({"record_token": hashlib.sha256(token.encode("ascii")).hexdigest(),
            "base_processor_batch_sha256": batch_digest, "source_video_sha256": source_sha,
            "prompt_sha256": prompt_sha, "input_ids_sha256": input_sha,
            "processor_sha256": proc_sha, "tokenizer_sha256": token_sha,
            "runtime_sha256": runtime_sha, "sampler_metadata_sha256": sampler_sha,
            "video_grid_thw": [32, 1, 1]})
        live_rows.append({"review_token": token, "source_video_sha256": source_sha,
            "batch_sha256": batch_digest, "batch": batch})
        binding_rows.append({"review_token": token, "question_sha256": question_sha,
            "options_sha256": options_sha, "rendered_prompt_sha256": prompt_sha,
            "tokenizer_sha256": token_sha, "prompt_token_ids_sha256": prompt_ids_sha,
            "choice_token_binding_sha256": choice_sha, "input_ids_sha256": input_sha,
            "answer_token_ids": answer_ids})
    return (prompt_inputs, processor, tokenizer, sampler_rows, expected,
            manifest_rows, batches, live_rows,
            build_qta_prompt_answer_token_bindings(binding_rows))


def candidate_report():
    return {
        "status": "structural_inputs_composed_unvalidated",
        "structurally_composed_rows": 3,
        "signature_bound_frame_pair_decisions": 15,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
    }


def transaction_report():
    return {
        "status": "structural_bindings_consistent_unvalidated",
        "structurally_bound_rows": 3,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }


class QtaDirectFaithfulnessAtomicPreloadTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [f"token_{i:02d}" for i in range(3)]
        self.sources = [hashlib.sha256(f"video{i}".encode()).hexdigest()
                        for i in range(3)]
        self.preload = {
            "sampler_receipt_raw": b"sampler",
            "candidate_inventory_raw": b"inventory",
            "media_sha256_by_role": {"role": "a" * 64},
            "actual_source_inputs": {"source": b"shared-source"},
            "stage1_audit_inputs": {
                "worksheet_raw": b"worksheet", "lineage_raw": b"lineage",
                "review_raw": b"review", "review_index_raw": b"index",
                "expected_lineage_sha256": "b" * 64,
            },
            "actual_qa_source_inputs": {"qa": b"shared-qa"},
            "clock_inputs_by_token": {token: {"capture_raw": b"clock"}
                                       for token in self.tokens},
            "processor_pins_by_token": {token: {"processor_revision": "rev",
                "processor_fingerprint_sha256": "c" * 64} for token in self.tokens},
            "tokenizer_sha256": "d" * 64,
            "video_duration_seconds_by_token": {token: "2" for token in self.tokens},
        }
        self.batches = [FrozenBatch({
            "input_ids": torch.tensor([[i + 1]], dtype=torch.int64),
            "pixels": torch.tensor([float(i + 1)]),
        }) for i in range(3)]
        self.live_rows = [{"review_token": token,
            "source_video_sha256": source,
            "batch_sha256": batch_sha256(batch), "batch": batch}
            for token, source, batch in zip(self.tokens, self.sources,
                                            self.batches, strict=True)]
        self.transaction = {
            "actual_source_inputs": self.preload["actual_source_inputs"],
            "review_bundle_inputs": {
                "worksheet_raw": b"worksheet", "lineage_raw": b"lineage",
                "review_raw": b"review", "review_index_raw": b"index",
                "expected_lineage_sha256": "b" * 64,
            },
            "stage1_audit_inputs": self.preload["stage1_audit_inputs"],
            "actual_qa_source_inputs": self.preload["actual_qa_source_inputs"],
            "phase1_receipt_raw": b"phase1",
            "phase1_artifact_bytes": {"artifact": b"frozen"},
            "integrity_roster_raw": b"integrity",
            "release_roster_raw": b"release",
            "root_public_key_pem": b"root-key",
            "expected_root_public_der_sha256": "e" * 64,
            "expected_external_cohort_pin_sha256": "f" * 64,
            "live_processor_rows": self.live_rows,
        }
        self.expected = [{"review_token": token,
            "source_video_sha256": source,
            "claim_sha256": hashlib.sha256(f"claim{i}".encode()).hexdigest()}
            for i, (token, source) in enumerate(zip(self.tokens, self.sources,
                                                    strict=True))]
        self.source_rows = [{"review_token": token,
            "source_video_sha256": source}
            for token, source in zip(self.tokens, self.sources, strict=True)]
        self.prompt_reconstruction_inputs, self.processor, self.tokenizer = _prompt_material(self.tokens)
        self.sampler_rows = [{"synthetic": index} for index in range(3)]

    def invoke(self, *, expected=None, source_rows=None, preload_report=None,
               tx_report=None, transaction_side_effect=None):
        expected = self.expected if expected is None else expected
        source_rows = self.source_rows if source_rows is None else source_rows
        with (mock.patch.object(atomic, "build_qta_actual_candidate_input_material",
                return_value={"status": "three_row_candidate_inputs_derived_unvalidated",
                    "requested_records": 3, "effective_records": 3,
                    "expected_rows": expected, "sampler_rows": self.sampler_rows}) as builder,
              mock.patch.object(atomic, "derive_stage1_source_pins",
                return_value={"status": "three_row_source_pins_structurally_derived",
                    "effective_records": 3, "source_rows": source_rows}) as pins,
              mock.patch.object(atomic, "_validate_processor_manifest",
                                return_value=([], "a" * 64,
                                              {"status": "candidate_declarations_crossbound_unvalidated"})),
              mock.patch.object(atomic, "evaluate_qta_direct_faithfulness_preload",
                return_value=candidate_report() if preload_report is None
                    else preload_report) as preload_gate,
              mock.patch.object(atomic, "preflight_qta_phase1_batch_transaction",
                side_effect=transaction_side_effect or (lambda **_: transaction_report())
                    if tx_report is None else None,
                return_value=tx_report) as transaction_gate):
            report = atomic.preflight_qta_direct_faithfulness_atomic(
                candidate_preload_inputs=self.preload,
                phase1_transaction_inputs=self.transaction,
                prompt_reconstruction_inputs=self.prompt_reconstruction_inputs,
                processor=self.processor, tokenizer=self.tokenizer)
        return report, builder, pins, preload_gate, transaction_gate

    def test_same_ordered_three_row_cohort_composes_but_never_authorizes(self):
        report, builder, pins, preload_gate, transaction_gate = self.invoke()
        self.assertEqual(report["status"],
                         "same_cohort_structurally_consistent_unvalidated")
        self.assertEqual(report["matched_rows"], 3)
        self.assertEqual(report["signature_bound_frame_pair_decisions"], 0)
        self.assertEqual(report["validated_clock_frame_anchors"], 0)
        self.assertEqual(report["requested_clock_frame_anchors"], 15)
        for field in ("external_trust_root_origin_authenticated",
                      "reviewer_identity_authenticated", "semantic_review_authenticated",
                      "clock_frame_correspondence_authenticated",
                      "direct_faithfulness_ready", "model_load_permitted",
                      "scoring_authorized", "faithfulness_scored"):
            self.assertFalse(report[field])
        self.assertEqual(report["forward_calls"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertEqual(report["execution_authorization"], "not_granted")
        builder.assert_called_once()
        pins.assert_called_once()
        preload_gate.assert_called_once()
        transaction_gate.assert_called_once()
        encoded = json.dumps(report)
        for token, source in zip(self.tokens, self.sources, strict=True):
            self.assertNotIn(token, encoded)
            self.assertNotIn(source, encoded)

    def test_different_cohort_order_blocks_before_either_leaf_gate(self):
        wrong = list(reversed(self.expected))
        report, _, _, preload_gate, transaction_gate = self.invoke(expected=wrong)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)
        self.assertEqual(report["signature_bound_frame_pair_decisions"], 0)
        preload_gate.assert_not_called()
        transaction_gate.assert_not_called()

    def test_leaf_failure_is_atomic_with_no_partial_row_or_anchor_credit(self):
        bad = dict(candidate_report(), status="blocked", structurally_composed_rows=0)
        report, _, _, _, transaction_gate = self.invoke(preload_report=bad)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)
        self.assertEqual(report["validated_clock_frame_anchors"], 0)
        self.assertEqual(report["signature_bound_frame_pair_decisions"], 0)
        transaction_gate.assert_called_once()

        report, _, _, preload_gate, _ = self.invoke(
            tx_report=dict(transaction_report(), status="blocked", structurally_bound_rows=0))
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)
        preload_gate.assert_called_once()

    def test_cross_branch_raw_bytes_must_be_identical(self):
        self.transaction["actual_qa_source_inputs"] = {"qa": b"different"}
        report, _, _, preload_gate, transaction_gate = self.invoke()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["failure_code"], "atomic_preload_blocked")
        preload_gate.assert_not_called()
        transaction_gate.assert_not_called()

    def test_source_inputs_keep_their_branch_specific_schemas(self):
        self.preload["actual_source_inputs"] = {"claim_lineage": b"bridge schema"}
        self.transaction["actual_source_inputs"] = {"qa_binding": b"transaction schema"}
        report, _, _, preload_gate, transaction_gate = self.invoke()
        self.assertEqual(report["status"],
                         "same_cohort_structurally_consistent_unvalidated")
        preload_gate.assert_called_once()
        transaction_gate.assert_called_once()

    def test_live_tensor_mutation_after_leaf_transaction_is_detected(self):
        def mutate_then_pass(**_):
            self.batches[1]["pixels"][0] += 1.0
            return transaction_report()
        report, _, _, _, transaction_gate = self.invoke(
            transaction_side_effect=mutate_then_pass)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)
        transaction_gate.assert_called_once()

    def test_aliased_batches_are_rejected(self):
        self.transaction["live_processor_rows"][2]["batch"] = self.batches[0]
        self.transaction["live_processor_rows"][2]["batch_sha256"] = batch_sha256(self.batches[0])
        report, _, _, preload_gate, transaction_gate = self.invoke()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)
        preload_gate.assert_not_called()
        transaction_gate.assert_not_called()

    def test_unknown_summary_fields_are_not_accepted_as_inputs(self):
        preload = dict(self.preload, report=candidate_report())
        report = atomic.preflight_qta_direct_faithfulness_atomic(
            candidate_preload_inputs=preload,
            phase1_transaction_inputs=self.transaction,
            prompt_reconstruction_inputs=self.prompt_reconstruction_inputs,
            processor=self.processor, tokenizer=self.tokenizer)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["matched_rows"], 0)

    def test_real_bridge_output_joins_the_same_ordered_live_rows(self):
        fixture = QtaActualCandidateInputBridgeTests()
        fixture.setUp()
        batches = [FrozenBatch({"input_ids": torch.tensor([[i + 1]]),
                                "pixels": torch.tensor([float(i + 1)])})
                   for i in range(3)]
        live_rows = [{"review_token": token, "source_video_sha256": source,
                      "batch_sha256": batch_sha256(batch), "batch": batch}
                     for token, source, batch in zip(fixture.tokens, fixture.sources,
                                                     batches, strict=True)]
        preload = {
            "sampler_receipt_raw": fixture.sampler_raw,
            "candidate_inventory_raw": b"inventory",
            "media_sha256_by_role": {},
            "actual_source_inputs": fixture.source_inputs,
            "stage1_audit_inputs": fixture.stage1_inputs,
            "actual_qa_source_inputs": fixture.qa_inputs,
            "clock_inputs_by_token": fixture.clock_inputs,
            "processor_pins_by_token": fixture.processor,
            "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            "video_duration_seconds_by_token": fixture.durations,
        }
        review = {key: fixture.stage1_inputs[key] for key in (
            "worksheet_raw", "lineage_raw", "review_raw", "review_index_raw",
            "expected_lineage_sha256")}
        transaction = {
            "actual_source_inputs": fixture.qa_inputs,
            "review_bundle_inputs": review,
            "stage1_audit_inputs": fixture.stage1_inputs,
            "actual_qa_source_inputs": fixture.qa_inputs,
            "phase1_receipt_raw": b"receipt",
            "phase1_artifact_bytes": {"phase1": b"artifact"},
            "integrity_roster_raw": b"integrity", "release_roster_raw": b"release",
            "root_public_key_pem": b"root", "expected_root_public_der_sha256": "a" * 64,
            "expected_external_cohort_pin_sha256": "b" * 64,
            "live_processor_rows": live_rows,
        }
        with (mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                  "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                  "human_identity_verified": False, "semantic_review_authenticated": False}),
              mock.patch.object(bridge, "derive_stage1_source_pins",
                                return_value=fixture.stage1_pins),
              mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"),
              mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                  {"review_token": token, "support_decision": "supports_claim",
                   "evidence_intervals_seconds": [[1, 2]]} for token in fixture.tokens]),
              mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                  "guard_pts_seconds": "1/10", "clock_correspondence_authenticated": False}),
              mock.patch.object(atomic, "derive_stage1_source_pins",
                                return_value=fixture.stage1_pins),
              mock.patch.object(atomic, "_validate_processor_manifest",
                                return_value=([], "a" * 64,
                                              {"status": "candidate_declarations_crossbound_unvalidated"})),
              mock.patch.object(atomic, "evaluate_qta_direct_faithfulness_preload",
                                return_value=candidate_report()),
              mock.patch.object(atomic, "preflight_qta_phase1_batch_transaction",
                                return_value=transaction_report())):
            report = atomic.preflight_qta_direct_faithfulness_atomic(
                candidate_preload_inputs=preload,
                phase1_transaction_inputs=transaction,
                prompt_reconstruction_inputs=self.prompt_reconstruction_inputs,
                processor=self.processor, tokenizer=self.tokenizer)
        self.assertEqual(report["status"],
                         "same_cohort_structurally_consistent_unvalidated")
        self.assertEqual(report["matched_rows"], 3)
        self.assertFalse(report["model_load_permitted"])
        fixture.doCleanups()

    def test_processor_manifest_crossbinds_candidate_and_live_tensor_pins(self):
        (prompts, processor, tokenizer, sampler_rows, expected, rows, batches,
         live, prompt_raw) = _bound_prompt_material(self.tokens, self.sources)
        protocol = b"fixed-protocol"
        protocol_sha = hashlib.sha256(protocol).hexdigest()
        raw = build_manifest(rows=rows, live_batches=batches, expected_row_pins=rows,
                             protocol_sha256=protocol_sha, expected_protocol_sha256=protocol_sha)
        preload = {"actual_source_inputs": {"qta_protocol_raw": protocol}}
        transaction = {"phase1_artifact_bytes": {"processor_batch_digest_manifest": raw,
            "sampler_model_runtime": b"runtime",
            "prompt_answer_token_bindings": prompt_raw}}
        atomic._validate_processor_manifest(preload, transaction, expected, live,
            sampler_rows, prompts, processor, tokenizer)
        prompt_raw = transaction["phase1_artifact_bytes"]["prompt_answer_token_bindings"]
        altered_prompt = json.loads(prompt_raw.decode("utf-8"))
        altered_prompt["rows"][0]["prompt_sha256"] = "f" * 64
        transaction["phase1_artifact_bytes"]["prompt_answer_token_bindings"] = json.dumps(
            altered_prompt, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode()
        with self.assertRaises(ValueError):
            atomic._validate_processor_manifest(preload, transaction, expected, live,
                sampler_rows, prompts, processor, tokenizer)
        transaction["phase1_artifact_bytes"]["prompt_answer_token_bindings"] = prompt_raw
        altered_runtime = json.loads(raw.decode("utf-8"))
        altered_runtime["rows"][0]["runtime_sha256"] = "f" * 64
        transaction["phase1_artifact_bytes"]["processor_batch_digest_manifest"] = json.dumps(
            altered_runtime, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode()
        with self.assertRaises(ValueError):
            atomic._validate_processor_manifest(preload, transaction, expected, live,
                sampler_rows, prompts, processor, tokenizer)
        transaction["phase1_artifact_bytes"]["processor_batch_digest_manifest"] = raw
        altered = copy.deepcopy(expected)
        altered[1]["processor_fingerprint_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            atomic._validate_processor_manifest(preload, transaction, altered, live,
                sampler_rows, prompts, processor, tokenizer)
        batch = live[0]["batch"]
        batch["input_ids"] = torch.tensor([[1.0, 2.0]])
        live[0]["batch_sha256"] = batch_sha256(batch)
        forged = json.loads(raw.decode("utf-8"))
        forged["rows"][0]["base_processor_batch_sha256"] = live[0]["batch_sha256"]
        float_ids = [1.0, 2.0]
        forged["rows"][0]["input_ids_sha256"] = hashlib.sha256(json.dumps(
            float_ids, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        transaction["phase1_artifact_bytes"]["processor_batch_digest_manifest"] = json.dumps(
            forged, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()
        with self.assertRaises(ValueError):
            atomic._validate_processor_manifest(preload, transaction, expected, live,
                sampler_rows, prompts, processor, tokenizer)

    def test_real_signed_phase1_transaction_composes_with_candidate_preflight(self):
        source_fixture = QtaPrebaselineTransactionPreflightTests()
        source_fixture.setUp()
        self.addCleanup(source_fixture.doCleanups)
        source_rows = source_fixture.stage1_pins["source_rows"]
        protocol = source_fixture.source_fixture.qa_inputs["qta_protocol_raw"]
        protocol_sha = hashlib.sha256(protocol).hexdigest()
        tokens = [row["review_token"] for row in source_rows]
        sources = [row["source_video_sha256"] for row in source_rows]
        csv_rows = list(csv.DictReader(io.StringIO(
            source_fixture.source_fixture.qa_inputs["csv_raw"].decode("utf-8"))))
        qa_rows = [{"question": row["question"],
                    "choices": [row[f"a{index}"] for index in range(5)]}
                   for row in csv_rows]
        (prompt_inputs, processor, tokenizer, sampler_rows, expected,
         manifest_rows, batches, live, prompt_raw) = _bound_prompt_material(
             tokens, sources, qa_rows=qa_rows)
        source_hashes_match = all(
            actual["question_sha256"] == declared["question_sha256"]
            and actual["options_sha256"] == declared["options_sha256"]
            for actual, declared in zip(source_rows, expected, strict=True))
        self.assertTrue(source_hashes_match)
        records = [{"record_token": hashlib.sha256(row["review_token"].encode("ascii")).hexdigest(),
                    "processor_batch_sha256": row["batch_sha256"]} for row in live]
        raw_manifest = build_manifest(rows=manifest_rows, live_batches=batches,
            expected_row_pins=manifest_rows, protocol_sha256=protocol_sha,
            expected_protocol_sha256=protocol_sha)
        artifacts = dict(source_fixture.artifacts)
        artifacts.pop("full_video_processor_batches")
        artifacts["sampler_model_runtime"] = b"runtime"
        artifacts["processor_batch_digest_manifest"] = raw_manifest
        artifacts["prompt_answer_token_bindings"] = prompt_raw
        phase = source_fixture.phase
        receipt = phase._create_v2(records=records, artifacts=artifacts,
            rows=manifest_rows, protocol_sha=protocol_sha)
        tx = dict(source_fixture.kwargs, phase1_receipt_raw=receipt,
                  phase1_artifact_bytes=artifacts, live_processor_rows=live,
                  expected_processor_batch_row_pins=manifest_rows,
                  expected_processor_batch_protocol_sha256=protocol_sha)
        signed_report = verify_and_bind_signed_phase1_batches(
            receipt_raw=receipt, artifact_bytes=artifacts,
            integrity_roster_raw=source_fixture.phase.integrity,
            release_roster_raw=source_fixture.phase.release,
            root_public_key_pem=source_fixture.phase.keys["root"][1],
            expected_root_public_der_sha256=source_fixture.phase.root_pin,
            expected_external_cohort_pin_sha256=tx["expected_external_cohort_pin_sha256"],
            live_rows=live, expected_processor_batch_row_pins=manifest_rows,
            expected_processor_batch_protocol_sha256=protocol_sha)
        self.assertEqual(signed_report["status"], "passed", signed_report)
        phase_report = phase1_tx.preflight_qta_phase1_batch_transaction(**tx)
        self.assertEqual(phase_report["status"], "structural_bindings_consistent_unvalidated",
                         phase_report)
        preload = {"sampler_receipt_raw": b"sampler", "candidate_inventory_raw": b"inventory",
            "media_sha256_by_role": {}, "actual_source_inputs": source_fixture.source_fixture.qa_inputs,
            "stage1_audit_inputs": source_fixture.source_fixture.stage1_inputs,
            "actual_qa_source_inputs": source_fixture.source_fixture.qa_inputs,
            "clock_inputs_by_token": {row["review_token"]: {"capture_raw": b"capture"} for row in source_rows},
            "processor_pins_by_token": {row["review_token"]: {"processor_revision": "rev",
                "processor_fingerprint_sha256": "a" * 64} for row in source_rows},
            "tokenizer_sha256": expected[0]["tokenizer_sha256"],
            "video_duration_seconds_by_token": {row["review_token"]: "2" for row in source_rows}}
        with (mock.patch.object(atomic, "build_qta_actual_candidate_input_material", return_value={
                "status": "three_row_candidate_inputs_derived_unvalidated", "requested_records": 3,
                "effective_records": 3, "expected_rows": expected,
                "sampler_rows": sampler_rows}),
              mock.patch.object(atomic, "derive_stage1_source_pins", return_value={
                "status": "three_row_source_pins_structurally_derived", "effective_records": 3,
                "source_rows": source_rows}),
              mock.patch.object(atomic, "evaluate_qta_direct_faithfulness_preload",
                                return_value=candidate_report())):
            report = atomic.preflight_qta_direct_faithfulness_atomic(
                candidate_preload_inputs=preload, phase1_transaction_inputs=tx,
                prompt_reconstruction_inputs=prompt_inputs,
                processor=processor, tokenizer=tokenizer)
        self.assertEqual(report["status"], "same_cohort_structurally_consistent_unvalidated")
        self.assertEqual(report["matched_rows"], 3)
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["scoring_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
