import hashlib
import json
import unittest
import copy
from pathlib import Path

from audit_nextgqa_review_handoff import audit
from build_nextgqa_claim_mapping_packet import apply_review_policy, build
from test_build_nextgqa_claim_mapping_packet import encode, packet
from test_render_nextgqa_claim_review_html import RenderNextgqaClaimReviewHtmlTests


def frozen_worksheet():
    source, source_raw = packet(3)
    built, _ = build(source, source_raw)
    policy_raw = (Path(__file__).resolve().parents[1] / "docs" /
                  "nextgqa_claim_review_policy_v1.json").read_bytes()
    built = apply_review_policy(built, json.loads(policy_raw), policy_raw)
    return built, encode(built)


def handoff():
    worksheet, worksheet_raw = frozen_worksheet()
    _, _, lineage, lineage_raw = RenderNextgqaClaimReviewHtmlTests().payload()
    lineage["source_packet_sha256"] = worksheet["source_packet_sha256"]
    lineage["cohort_sha256"] = worksheet["cohort_sha256"]
    lineage["dataset_inputs_sha256"] = worksheet["dataset_inputs_sha256"]
    for expected, lr in zip(worksheet["records"], lineage["records"], strict=True):
        lr.update(video_id=expected["video_id"], claim_id=expected["claim_id"],
                  video_sha256=expected["video_sha256"],
                  claim_text_sha256=expected["claim_text_sha256"])
    lineage_raw = encode(lineage)
    lineage_sha = hashlib.sha256(lineage_raw).hexdigest()
    review = {key: value for key, value in worksheet.items() if key != "records"}
    review["source_lineage_manifest_sha256"] = lineage_sha
    for row in worksheet["records"]:
        lineage_row = lineage["records"][len(review.get("records", []))]
        review_row = {key: value for key, value in row.items() if key != "video_path"}
        review_row["review"] = {
            "source_identity_verified": True,
            "full_source_video_reviewed": True,
            "question_and_claim_only_context_shown": True,
            "gold_and_seed_hidden_until_mapping_frozen": True,
            "outcome_scores_not_accessed": True,
            "occurrence_policy_applied": "all_relevant_occurrences",
            "support_decision": "supports_claim",
            "evidence_intervals_seconds": [[1.001, 2.002]],
            "interval_rationales": ["synthetic rationale"],
            "reviewer_role": "independent_human",
            "review_record_reference": None,
            "reviewed_at_utc": "2026-10-15T21:00:00.000Z",
            "review_status": "completed",
            "claim_text_sha256_seen": row["claim_text_sha256"],
            "media_sha256_verified": True,
            "media_sha256_observed": row["video_sha256"],
            "media_duration_seconds": 3.0,
            "media_mime_type": "video/mp4",
            "cohort_lineage_reviewed": True,
            "display_player": "Synthetic browser UA",
            "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
            "source_lineage_manifest_sha256": lineage_sha,
        }
        review_row["review"]["source_lineage"] = {k: lineage_row[k] for k in (
            "dataset_question_id", "dataset_video_id", "video_key",
            "relative_video_path", "question_sha256")}
        review.setdefault("records", []).append(review_row)
    review.update({
        "status": "human_review_recorded_unvalidated",
        "review_completed": True,
        "execution_authorization": "not_granted",
        "human_reviewer_role": "independent_human",
        "human_reviewer_private_id": "synthetic-reviewer",
        "review_environment": {
            "browser_user_agent": "Synthetic browser UA",
            "offline_static_page": True,
            "network_fetch_available": False,
        },
    })
    review_raw = encode(review)
    index = {
        "schema_version": 1,
        "purpose": "unvalidated_private_human_review_handoff",
        "review_payload_sha256": hashlib.sha256(review_raw).hexdigest(),
        "source_worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
        "source_lineage_manifest_sha256": lineage_sha,
        "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
        "created_at_utc": "2026-10-15T21:00:00.000Z",
        "review_status": "unvalidated_requires_independent_audit",
        "execution_authorization": "not_granted",
    }
    return worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, encode(index)


class AuditNextgqaReviewHandoffTests(unittest.TestCase):
    def test_well_formed_private_handoff_remains_unvalidated(self):
        values = handoff()
        result = audit(*values, hashlib.sha256(values[3]).hexdigest())
        self.assertEqual(result["status"], "unvalidated_handoff_structurally_valid")
        self.assertEqual(result["reviewed_records"], 3)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertFalse(result["semantic_review_authenticated"])

    def test_payload_or_source_hash_changes_are_detected(self):
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = handoff()
        bad_index = dict(index, review_payload_sha256="0" * 64)
        result = audit(worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, bad_index, encode(bad_index), hashlib.sha256(lineage_raw).hexdigest())
        self.assertIn("review_payload_hash_mismatch", result["failure_codes"])
        bad_index = dict(index, source_worksheet_sha256="1" * 64)
        result = audit(worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, bad_index, encode(bad_index), hashlib.sha256(lineage_raw).hexdigest())
        self.assertIn("worksheet_hash_mismatch", result["failure_codes"])

    def test_extra_fields_path_leak_and_bad_intervals_fail_closed(self):
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = handoff()
        review["answer_sentinel"] = "secret"
        result = audit(worksheet, worksheet_raw, lineage, lineage_raw, review, encode(review), index, index_raw, hashlib.sha256(lineage_raw).hexdigest())
        self.assertIn("review_payload_schema_invalid", result["failure_codes"])
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = handoff()
        review["records"][0]["video_path"] = "/private/path"
        result = audit(worksheet, worksheet_raw, lineage, lineage_raw, review, encode(review), index, index_raw, hashlib.sha256(lineage_raw).hexdigest())
        self.assertIn("review_row_schema_invalid", result["failure_codes"])
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = handoff()
        review["records"][0]["review"]["evidence_intervals_seconds"] = [[2.001, 1.001]]
        review_raw = encode(review)
        index["review_payload_sha256"] = hashlib.sha256(review_raw).hexdigest()
        result = audit(worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, encode(index), hashlib.sha256(lineage_raw).hexdigest())
        self.assertIn("review_intervals_invalid", result["failure_codes"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_malformed_worksheet_shapes_fail_closed_without_traceback(self):
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, index_raw = handoff()
        for records in (None, "abc", [None, None, None]):
            with self.subTest(records_type=type(records).__name__):
                malformed = dict(worksheet, records=records)
                result = audit(malformed, encode(malformed), lineage, lineage_raw, review, review_raw, index, index_raw, hashlib.sha256(lineage_raw).hexdigest())
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["execution_authorization"], "not_granted")
                self.assertIn("frozen_worksheet_invalid", result["failure_codes"])


if __name__ == "__main__":
    unittest.main()
