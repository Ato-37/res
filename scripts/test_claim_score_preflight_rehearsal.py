import hashlib
import json
import unittest

from claim_score_preflight_rehearsal import rehearse_claim_score_preflight
from test_claim_answer_target_contract import fixture as target_fixture, h
from test_claim_review_assignment_plan import fixture as assignment_fixture, rebind_review


def integration_fixture():
    assignment_inputs = assignment_fixture()
    videos = assignment_inputs["expected_video_ids"]
    claims = json.loads(assignment_inputs["claim_input_manifest_raw"])["claims"]
    claim_ids = [next(row["claim_id"] for row in claims if row["video_id"] == video)
                 for video in videos]
    _, _, bindings, target, _ = target_fixture()
    bindings["claim_input_manifest_sha256"] = assignment_inputs["claim_input_manifest_sha256"]
    target["claim_input_manifest_sha256"] = assignment_inputs["claim_input_manifest_sha256"]
    worksheet_doc = json.loads(assignment_inputs["worksheet_raw"])
    source_inputs = worksheet_doc["dataset_inputs_sha256"]
    source_packet_sha = worksheet_doc["source_packet_sha256"]
    cohort_sha = worksheet_doc["cohort_sha256"]
    source_rows, qa_rows = [], []
    for index, (row, video, claim_id) in enumerate(zip(target["records"], videos, claim_ids,
                                                        strict=True)):
        sheet = assignment_inputs["worksheet_records"][index]
        claim = claims[index]
        source_row = {
            "video_id": video, "claim_id": claim_id,
            "dataset_question_id": f"private-qid-{index}",
            "dataset_video_id": f"dataset-video-{index}",
            "video_key": f"clips/video-{index}",
            "relative_video_path": f"clips/video-{index}.mp4",
            "video_sha256": sheet["video_sha256"],
            "question_sha256": h(sheet["question"]),
            "claim_text_sha256": claim["claim_text_sha256"],
        }
        source_rows.append(source_row)
        qa_row = {
            "video_id": video,
            "dataset_video_id": source_row["dataset_video_id"],
            "dataset_question_id": source_row["dataset_question_id"],
            "question": sheet["question"],
            "choices": [f"private choice {index}-{j}" for j in range(5)],
            "answer": f"private choice {index}-2",
        }
        qa_rows.append(qa_row)
        row["video_id"] = video
        row["claim_id"] = claim_id
        row["dataset_qid_sha256"] = h(source_row["dataset_question_id"])
        row["question_sha256"] = source_row["question_sha256"]
        row["ordered_choice_sha256"] = [h(choice) for choice in qa_row["choices"]]
        row["gold_index"] = 2
        row["dataset_source_row_sha256"] = hashlib.sha256(json.dumps(
            qa_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_manifest_sha = hashlib.sha256(json.dumps(
        source_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bindings["dataset_source_manifest_sha256"] = source_manifest_sha
    target["dataset_source_manifest_sha256"] = source_manifest_sha
    lineage = {
        "schema_version": 1,
        "purpose": "private_source_lineage_crosscheck_for_claim_review",
        "created_by": "synthetic_fixture",
        "source_packet_sha256": source_packet_sha, "cohort_sha256": cohort_sha,
        "dataset_inputs_sha256": source_inputs,
        "requested_records": 3, "effective_records": 3,
        "verification_scope": "synthetic metadata binding only",
        "status": "metadata_lineage_crosschecked", "records": source_rows,
    }
    lineage_raw = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
    assignment_inputs["review_audit_report"]["lineage_manifest_sha256"] = hashlib.sha256(
        lineage_raw).hexdigest()
    qa_source_raw = json.dumps({
        "schema_version": 1,
        "purpose": "synthetic_private_qa_source_rows",
        "dataset_inputs_sha256": source_inputs,
        "records": qa_rows,
    }, sort_keys=True, separators=(",", ":")).encode()
    target_raw = json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
    return assignment_inputs, bindings, target, target_raw, lineage_raw, qa_source_raw


class ClaimScorePreflightRehearsalTests(unittest.TestCase):
    def test_complete_synthetic_preflight_is_atomic_and_never_scores(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["status"], "combined_preflight_structurally_passed_unvalidated")
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(len(result["assignments"]), 3)
        self.assertEqual(result["model_forwards_planned"], 9)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        encoded = json.dumps(result)
        self.assertNotIn("private claim", encoded)
        self.assertNotIn("private-video", encoded)

    def test_late_review_failure_discards_all_assignments(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        assignment_inputs["review_records"][2]["review"]["support_decision"] = "partial"
        rebind_review(assignment_inputs)
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_codes"], ["assignment_plan_blocked"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["model_forwards_planned"], 0)
        self.assertEqual(result["score_calls"], 0)

    def test_late_target_failure_discards_all_frame_assignments(self):
        assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
        target["records"][2]["gold_index"] = True
        raw = json.dumps(target).encode()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_codes"], ["qa_target_choices_or_gold_mismatch"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["model_forwards_planned"], 0)
        self.assertEqual(result["score_calls"], 0)

    def test_target_qid_question_and_source_row_must_match_review_lineage(self):
        for field in ("dataset_qid_sha256", "question_sha256", "dataset_source_row_sha256"):
            with self.subTest(field=field):
                assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
                target["records"][1][field] = h("substituted " + field)
                raw = json.dumps(target).encode()
                result = rehearse_claim_score_preflight(
                    assignment_inputs=assignment_inputs, target_manifest_raw=raw,
                    expected_target_binding_hashes=bindings,
                    lineage_manifest_raw=lineage_raw, qa_source_rows_raw=qa_source_raw)
                self.assertEqual(result["failure_codes"], ["qa_target_source_row_mismatch"])
                self.assertEqual(result["assignments"], [])
                self.assertEqual(result["score_calls"], 0)

    def test_target_row_order_must_match_frozen_video_order(self):
        assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
        target["records"].reverse()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs,
            target_manifest_raw=json.dumps(target).encode(),
            expected_target_binding_hashes=bindings,
            lineage_manifest_raw=lineage_raw, qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["qa_target_contract_blocked"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)

    def test_choice_order_and_gold_index_are_crosschecked_to_source_rows(self):
        for mutate in ("choice_order", "gold_index"):
            with self.subTest(mutate=mutate):
                assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
                if mutate == "choice_order":
                    target["records"][0]["ordered_choice_sha256"].reverse()
                else:
                    target["records"][0]["gold_index"] = 1
                result = rehearse_claim_score_preflight(
                    assignment_inputs=assignment_inputs,
                    target_manifest_raw=json.dumps(target).encode(),
                    expected_target_binding_hashes=bindings,
                    lineage_manifest_raw=lineage_raw,
                    qa_source_rows_raw=qa_source_raw)
                self.assertEqual(result["failure_codes"], ["qa_target_choices_or_gold_mismatch"])
                self.assertEqual(result["assignments"], [])
                self.assertEqual(result["score_calls"], 0)

    def test_self_consistent_synthetic_qa_substitution_stays_unvalidated(self):
        assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
        qa_source = json.loads(qa_source_raw)
        qa_row = qa_source["records"][0]
        qa_row["choices"] = [f"replacement {i}" for i in range(5)]
        qa_row["answer"] = qa_row["choices"][4]
        target_row = target["records"][0]
        target_row["ordered_choice_sha256"] = [h(choice) for choice in qa_row["choices"]]
        target_row["gold_index"] = 4
        target_row["dataset_source_row_sha256"] = hashlib.sha256(json.dumps(
            qa_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs,
            target_manifest_raw=json.dumps(target).encode(),
            expected_target_binding_hashes=bindings,
            lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=json.dumps(qa_source, sort_keys=True,
                                          separators=(",", ":")).encode())
        # A caller can replace synthetic source rows and matching declarations.
        # This contract is structural only and must never imply source authenticity.
        self.assertEqual(result["status"], "combined_preflight_structurally_passed_unvalidated")
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["scope"], "synthetic_preflight_rehearsal_only")

    def test_caller_rebound_dataset_video_id_source_substitution_stays_unvalidated(self):
        assignment_inputs, bindings, target, _, lineage_raw, qa_source_raw = integration_fixture()
        lineage = json.loads(lineage_raw)
        qa_source = json.loads(qa_source_raw)
        video_id = assignment_inputs["expected_video_ids"][0]
        replacement = "caller-selected-dataset-video"
        lineage_row = next(row for row in lineage["records"]
                           if row["video_id"] == video_id)
        qa_row = next(row for row in qa_source["records"]
                      if row["video_id"] == video_id)
        lineage_row["dataset_video_id"] = replacement
        qa_row["dataset_video_id"] = replacement
        lineage_raw = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        assignment_inputs["review_audit_report"]["lineage_manifest_sha256"] = hashlib.sha256(
            lineage_raw).hexdigest()
        target_row = next(row for row in target["records"] if row["video_id"] == video_id)
        target_row["dataset_source_row_sha256"] = hashlib.sha256(json.dumps(
            qa_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs,
            target_manifest_raw=json.dumps(target, sort_keys=True,
                                           separators=(",", ":")).encode(),
            expected_target_binding_hashes=bindings,
            lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=json.dumps(qa_source, sort_keys=True,
                                          separators=(",", ":")).encode())
        # All related caller-controlled declarations can be rebound together.
        # Passing remains structural evidence only; it does not authenticate the
        # replacement identifier against the dataset named by dataset_inputs_sha256.
        self.assertEqual(result["status"], "combined_preflight_structurally_passed_unvalidated")
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["scope"], "synthetic_preflight_rehearsal_only")

    def test_lineage_dataset_keys_and_claim_qid_formula_fail_closed(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        lineage = json.loads(lineage_raw)
        lineage["dataset_inputs_sha256"] = {"wrong.csv": h("not a sha")}
        bad_lineage = json.dumps(lineage).encode()
        assignment_inputs["review_audit_report"]["lineage_manifest_sha256"] = hashlib.sha256(
            bad_lineage).hexdigest()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings,
            lineage_manifest_raw=bad_lineage, qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["lineage_dataset_inputs_invalid"])
        self.assertEqual(result["score_calls"], 0)

        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        lineage = json.loads(lineage_raw)
        lineage["records"][0]["dataset_question_id"] = "substituted-qid"
        bad_lineage = json.dumps(lineage).encode()
        assignment_inputs["review_audit_report"]["lineage_manifest_sha256"] = hashlib.sha256(
            bad_lineage).hexdigest()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings,
            lineage_manifest_raw=bad_lineage, qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["lineage_worksheet_binding_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_claim_input_manifest_binding_must_be_shared(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        bindings["claim_input_manifest_sha256"] = h("different claim manifest")
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["target_expected_bindings_invalid"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)

    def test_malformed_claim_manifest_is_blocked_without_exception(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        assignment_inputs["claim_input_manifest_raw"] = b'{"claims":[null]}'
        assignment_inputs["claim_input_manifest_sha256"] = hashlib.sha256(
            assignment_inputs["claim_input_manifest_raw"]).hexdigest()
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["claim_input_manifest_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_unexpected_assignment_input_is_blocked_without_exception(self):
        assignment_inputs, bindings, _, target_raw, lineage_raw, qa_source_raw = integration_fixture()
        assignment_inputs["unexpected"] = "ignored?"
        result = rehearse_claim_score_preflight(
            assignment_inputs=assignment_inputs, target_manifest_raw=target_raw,
            expected_target_binding_hashes=bindings, lineage_manifest_raw=lineage_raw,
            qa_source_rows_raw=qa_source_raw)
        self.assertEqual(result["failure_codes"], ["assignment_plan_invalid"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)


if __name__ == "__main__":
    unittest.main()
