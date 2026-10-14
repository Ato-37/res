import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from audit_claim_review_evidence_index import audit, main
from audit_claim_mapping_manifest import _intervals_sha256

H = "a" * 64
REVISION = "b" * 40
VIDEO_IDS = ["synthetic-video-a", "synthetic-video-b"]


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fixture():
    eligible = [{"video_id": video_id} for video_id in VIDEO_IDS]
    eligible_raw = raw(eligible)
    input_rows = []
    ledger_rows = []
    review_records = []
    for index, video_id in enumerate(VIDEO_IDS):
        claim_id = f"claim-{index}"
        claim_text = f"Frozen synthetic claim {index}."
        claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
        generation_hash = ("e" if index == 0 else "f") * 64
        review_hash = ("1" if index == 0 else "2") * 64
        model_revision = f"Qwen/model@{REVISION}"
        intervals = [[1.0, 2.0]]
        input_rows.append({
            "video_id": video_id, "claim_id": claim_id,
            "claim_text": claim_text, "claim_text_sha256": claim_hash, "model_revision": model_revision,
            "generation_run_sha256": generation_hash,
        })
        ledger_rows.append({
            "video_id": video_id, "claim_id": claim_id,
            "claim_text_sha256": claim_hash, "claim_origin": "model_generated_frozen",
            "model_revision": model_revision, "generation_run_sha256": generation_hash,
            "mapping_method": "independent_claim_review",
            "qa_seed": {"usage": "not_used", "intervals": []},
            "evidence_intervals": intervals,
            "review": {
                "status": "completed", "reviewer_role": "independent_human",
                "review_record_sha256": review_hash,
                "reviewed_at_utc": "2026-10-15T00:00:00Z",
                "claim_text_sha256_seen": claim_hash,
                "evidence_intervals_sha256_seen": _intervals_sha256(intervals),
                "video_reviewed": True, "answer_and_gold_hidden": True,
                "qa_seed_hidden_until_mapping_frozen": True,
                "outcome_scores_seen": False,
                "claim_support_decision": "supports_claim",
                "occurrence_policy_applied": "one_sufficient_occurrence",
            },
        })
        review_records.append({
            "video_id": video_id, "claim_id": claim_id,
            "review_record_sha256": review_hash,
            "private_locator": f"private://synthetic/review/{index}",
        })
    claim_input = {
        "schema_version": 1, "eligible_dataset_id": "synthetic-dataset",
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "claim_policy_sha256": H, "occurrence_policy": "one_sufficient_occurrence",
        "claims": input_rows,
    }
    claim_input_raw = raw(claim_input)
    ledger = {
        "schema_version": 2, "eligible_dataset_id": "synthetic-dataset",
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "claim_policy_sha256": H, "occurrence_policy": "one_sufficient_occurrence",
        "scoring_started_at_utc": None, "claims": ledger_rows,
    }
    ledger_raw = raw(ledger)
    review_index = {
        "schema_version": 1,
        "mapping_ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "records": review_records,
    }
    review_index_raw = raw(review_index)
    protocol = {
        "schema_version": 12,
        "model": {"revision": REVISION},
        "cohort": {"eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
                   "eligible_video_count": len(eligible)},
        "claim_review": {
            "claim_manifest_sha256": hashlib.sha256(claim_input_raw).hexdigest(),
            "mapping_sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "review_record_sha256": hashlib.sha256(review_index_raw).hexdigest(),
            "review_status": "completed", "review_timing": "before_scoring",
            "mapping_unit": "claim_to_evidence_frames",
        },
    }
    return protocol, eligible, eligible_raw, claim_input, claim_input_raw, ledger, ledger_raw, review_index, review_index_raw


def run_audit(parts):
    protocol, eligible, eligible_raw, claim_input, claim_input_raw, ledger, ledger_raw, review_index, review_index_raw = parts
    return audit(protocol, raw(protocol), eligible, eligible_raw,
                 claim_input, claim_input_raw, ledger, ledger_raw,
                 review_index, review_index_raw)


class ClaimReviewEvidenceIndexTests(unittest.TestCase):
    def test_complete_hash_chain_passes_without_authorizing_execution(self):
        result = run_audit(fixture())
        self.assertEqual(result["structural_status"], "passed")
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["requested_eligible_video_count"], 2)

    def test_each_protocol_hash_must_bind_its_exact_artifact(self):
        for field, failure in (
            ("claim_manifest_sha256", "protocol_claim_input_hash_mismatch"),
            ("mapping_sha256", "protocol_claim_mapping_hash_mismatch"),
            ("review_record_sha256", "protocol_review_index_hash_mismatch"),
        ):
            parts = list(fixture())
            parts[0] = copy.deepcopy(parts[0])
            parts[0]["claim_review"][field] = "0" * 64
            self.assertIn(failure, run_audit(parts)["failure_codes"])

    def test_claim_rows_must_match_frozen_input_identity_and_generation(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["claims"][0]["generation_run_sha256"] = "0" * 64
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["mapping_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        parts[7] = copy.deepcopy(parts[7])
        parts[7]["mapping_ledger_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        parts[8] = raw(parts[7])
        parts[0]["claim_review"]["review_record_sha256"] = hashlib.sha256(parts[8]).hexdigest()
        result = run_audit(parts)
        self.assertIn("claim_input_to_mapping_row_mismatch", result["failure_codes"])

    def test_frozen_claim_text_bytes_must_match_declared_sha256(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["claims"][0]["claim_text"] = "A different frozen claim."
        parts[4] = raw(parts[3])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["claim_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        result = run_audit(parts)
        self.assertIn("claim_input_text_hash_mismatch", result["failure_codes"])

    def test_unpaired_surrogate_claim_text_fails_closed(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["claims"][0]["claim_text"] = "bad\ud800text"
        parts[4] = raw(parts[3])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["claim_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        result = run_audit(parts)
        self.assertEqual(result["structural_status"], "failed")
        self.assertIn("claim_input_text_invalid_unicode", result["failure_codes"])

    def test_blank_claim_text_is_rejected(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["claims"][0]["claim_text"] = " \t\n"
        parts[4] = raw(parts[3])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["claim_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        result = run_audit(parts)
        self.assertIn("claim_input_row_invalid", result["failure_codes"])

    def test_claim_text_hash_does_not_normalize_unicode(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        nfc_text = "caf\u00e9"
        nfd_text = "cafe\u0301"
        self.assertNotEqual(nfc_text.encode("utf-8"), nfd_text.encode("utf-8"))
        parts[3]["claims"][0]["claim_text"] = nfd_text
        parts[3]["claims"][0]["claim_text_sha256"] = hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()
        parts[4] = raw(parts[3])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["claim_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        result = run_audit(parts)
        self.assertIn("claim_input_text_hash_mismatch", result["failure_codes"])

    def test_claim_model_revision_must_match_protocol_model_pin(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["claims"][0]["model_revision"] = "Qwen/model@" + "9" * 40
        parts[4] = raw(parts[3])
        result = run_audit(parts)
        self.assertIn("claim_input_model_revision_mismatch", result["failure_codes"])

    def test_invalid_occurrence_policy_type_fails_without_exception(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["occurrence_policy"] = []
        parts[4] = raw(parts[3])
        result = run_audit(parts)
        self.assertEqual(result["structural_status"], "failed")
        self.assertIn("claim_input_manifest_fields_invalid", result["failure_codes"])

    def test_review_index_must_reconcile_every_row_once(self):
        parts = list(fixture())
        parts[7] = copy.deepcopy(parts[7])
        parts[7]["records"].pop()
        parts[8] = raw(parts[7])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["review_record_sha256"] = hashlib.sha256(parts[8]).hexdigest()
        result = run_audit(parts)
        self.assertIn("review_evidence_ledger_link_mismatch", result["failure_codes"])
        parts = list(fixture())
        parts[7] = copy.deepcopy(parts[7])
        parts[7]["records"].append(copy.deepcopy(parts[7]["records"][0]))
        parts[8] = raw(parts[7])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["review_record_sha256"] = hashlib.sha256(parts[8]).hexdigest()
        self.assertIn("review_evidence_duplicate_video", run_audit(parts)["failure_codes"])

    def test_per_row_review_record_hashes_must_be_unique(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["claims"][1]["review"]["review_record_sha256"] = parts[5]["claims"][0]["review"]["review_record_sha256"]
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["claim_review"]["mapping_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        parts[7] = copy.deepcopy(parts[7])
        parts[7]["mapping_ledger_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        parts[7]["records"][1]["review_record_sha256"] = parts[7]["records"][0]["review_record_sha256"]
        parts[8] = raw(parts[7])
        parts[0]["claim_review"]["review_record_sha256"] = hashlib.sha256(parts[8]).hexdigest()
        self.assertIn("review_evidence_duplicate_record_hash", run_audit(parts)["failure_codes"])

    def test_byte_binding_rejects_parsed_object_detachment(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["claims"][0]["claim_id"] = "detached"
        result = run_audit(parts)
        self.assertIn("claim_input_manifest_byte_binding_invalid", result["failure_codes"])

    def test_cli_emits_no_private_values(self):
        parts = fixture()
        names = ["protocol", "eligible", "claim-input", "ledger", "review-index"]
        values = [raw(parts[0]), parts[2], parts[4], parts[6], parts[8]]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, value in zip(names, values):
                path = Path(tmp) / f"{name}.json"
                path.write_bytes(value)
                paths.append(str(path))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main([
                    "--protocol", paths[0], "--eligible-manifest", paths[1],
                    "--claim-input-manifest", paths[2], "--claim-mapping-ledger", paths[3],
                    "--review-evidence-index", paths[4],
                ])
        self.assertEqual(status, 0)
        self.assertIn('"execution_authorization": "not_granted"', output.getvalue())
        self.assertNotIn("synthetic-video", output.getvalue())
        self.assertNotIn("private://", output.getvalue())

    def test_cli_rejects_excessively_nested_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep.json"
            path.write_bytes(b"[" * 3000 + b"0" + b"]" * 3000)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main([
                    "--protocol", str(path), "--eligible-manifest", str(path),
                    "--claim-input-manifest", str(path), "--claim-mapping-ledger", str(path),
                    "--review-evidence-index", str(path),
                ])
        self.assertEqual(status, 1)
        self.assertIn("private_input_invalid", output.getvalue())

    def test_untrusted_nested_shapes_fail_without_exceptions(self):
        invalid_values = [None, 0, True, [], {}, {"nested": []}]
        for value in invalid_values:
            with self.subTest(value=value):
                parts = list(fixture())
                parts[0] = copy.deepcopy(parts[0])
                parts[0]["claim_review"] = copy.deepcopy(value)
                result = run_audit(parts)
                self.assertEqual(result["structural_status"], "failed")
                parts = list(fixture())
                parts[5] = copy.deepcopy(parts[5])
                parts[5]["claims"] = copy.deepcopy(value)
                result = run_audit(parts)
                self.assertEqual(result["structural_status"], "failed")
                parts = list(fixture())
                parts[7] = copy.deepcopy(parts[7])
                parts[7]["records"] = copy.deepcopy(value)
                result = run_audit(parts)
                self.assertEqual(result["structural_status"], "failed")


if __name__ == "__main__":
    unittest.main()
