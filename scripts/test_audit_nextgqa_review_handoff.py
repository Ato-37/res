import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit_nextgqa_review_handoff import audit, audit_custody
from build_nextgqa_claim_mapping_packet import apply_review_policy, build
from test_build_nextgqa_claim_mapping_packet import encode, packet
from test_render_nextgqa_claim_review_html import RenderNextgqaClaimReviewHtmlTests


def frozen_worksheet():
    source, raw = packet(3)
    built, _ = build(source, raw)
    policy_raw = (Path(__file__).resolve().parents[1] / "docs" / "nextgqa_claim_review_policy_v6.json").read_bytes()
    built = apply_review_policy(built, json.loads(policy_raw), policy_raw)
    return built, encode(built)


def fixture():
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
    timestamp = "2026-10-16T06:00:00.000Z"
    review = {
        "schema_version": 3, "purpose": "anonymous_local_claim_video_human_review",
        "status": "human_review_recorded_unvalidated", "review_completed": True,
        "execution_authorization": "not_granted", "claim_review_policy_id": "nextgqa_claim_review_v6",
        "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
        "human_reviewer_role": "independent_human", "human_reviewer_private_id": "reviewer-local",
        "review_environment": {"browser_user_agent": "Synthetic", "offline_static_page": True,
                               "network_fetch_available": False},
        "independence_attestation": "reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk",
        "export_disclosure_acknowledged": True,
        "records": []}
    for row in worksheet["records"]:
        review["records"].append({"review_token": row["review_token"], "review": {
            "full_source_video_reviewed": True, "question_and_claim_only_context_shown": True,
            "gold_and_seed_hidden_until_mapping_frozen": True, "outcome_scores_not_accessed": True,
            "media_bytes_verified": True,
            "occurrence_policy_applied": "all_relevant_occurrences", "support_decision": "supports_claim",
            "evidence_intervals_seconds": [[1.001, 2.002]], "interval_rationales": ["visual evidence"],
            "media_duration_seconds": 3.0,
            "reviewer_role": "independent_human", "first_played_at_utc": timestamp,
            "reviewed_at_utc": timestamp,
            "review_status": "completed", "display_player": "HTML5 muted",
            "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"]}})
    review_raw = encode(review)
    index = {"schema_version": 1, "purpose": "anonymous_unvalidated_human_review_handoff",
             "review_payload_sha256": hashlib.sha256(review_raw).hexdigest(),
             "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
             "created_at_utc": timestamp, "review_status": "unvalidated_requires_custodian_audit",
             "execution_authorization": "not_granted"}
    return worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw, index, encode(index)


class AuditNextgqaReviewHandoffTests(unittest.TestCase):
    def test_anonymous_handoff_is_structurally_valid_but_unvalidated(self):
        values = fixture()
        result = audit(values[0], values[1], values[2], values[3], values[4], values[5],
                       values[6], values[7], hashlib.sha256(values[3]).hexdigest())
        self.assertEqual(result["status"], "unvalidated_handoff_structurally_valid")
        self.assertEqual(result["reviewed_records"], 3)
        self.assertFalse(result["semantic_review_authenticated"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["worksheet_sha256"], hashlib.sha256(values[1]).hexdigest())
        self.assertEqual(result["review_payload_sha256"], hashlib.sha256(values[5]).hexdigest())
        self.assertEqual(result["review_index_sha256"], hashlib.sha256(values[7]).hexdigest())

    def test_old_review_schema_v2_fails_closed(self):
        v = fixture()
        review = json.loads(v[5])
        review["schema_version"] = 2
        review_raw = encode(review)
        index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        result = audit(v[0], v[1], v[2], v[3], review, review_raw, index, encode(index),
                       hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_payload_schema_invalid", result["failure_codes"])
        self.assertFalse(result["semantic_review_authenticated"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_export_disclosure_acknowledgment_is_required(self):
        for value in (False, None):
            with self.subTest(value=value):
                v = fixture()
                review = json.loads(v[5])
                if value is None:
                    review.pop("export_disclosure_acknowledged")
                else:
                    review["export_disclosure_acknowledged"] = value
                review_raw = encode(review)
                index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
                result = audit(v[0], v[1], v[2], v[3], review, review_raw, index, encode(index),
                               hashlib.sha256(v[3]).hexdigest())
                self.assertIn("review_payload_schema_invalid", result["failure_codes"])
                self.assertEqual(result["execution_authorization"], "not_granted")

    def test_per_item_review_times_may_precede_index_but_must_be_ordered(self):
        v = fixture()
        review = json.loads(v[5])
        review["records"][0]["review"]["first_played_at_utc"] = "2026-10-16T05:40:00Z"
        review["records"][0]["review"]["reviewed_at_utc"] = "2026-10-16T05:55:00Z"
        review_raw = encode(review)
        index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        result = audit(v[0], v[1], v[2], v[3], review, review_raw, index, encode(index),
                       hashlib.sha256(v[3]).hexdigest())
        self.assertEqual(result["status"], "unvalidated_handoff_structurally_valid")
        review["records"][0]["review"]["first_played_at_utc"] = "2026-10-16T05:56:00Z"
        review_raw = encode(review)
        index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        result = audit(v[0], v[1], v[2], v[3], review, review_raw, index, encode(index),
                       hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_declaration_incomplete", result["failure_codes"])

    def test_review_hash_token_and_interval_tampering_fail_closed(self):
        v = fixture()
        index = dict(v[6], review_payload_sha256="0" * 64)
        result = audit(*v[:6], index, encode(index), hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_payload_hash_mismatch", result["failure_codes"])
        review = json.loads(v[5]); review["records"][0]["review_token"] = "../../private"
        result = audit(v[0], v[1], v[2], v[3], review, encode(review), v[6], v[7], hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_token_binding_invalid", result["failure_codes"])
        review = json.loads(v[5]); review["records"][0]["review"]["evidence_intervals_seconds"] = [[2, 1]]
        result = audit(v[0], v[1], v[2], v[3], review, encode(review), v[6], v[7], hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_intervals_invalid", result["failure_codes"])

    def test_exact_duplicate_interval_is_retained_and_structurally_accepted(self):
        v = fixture()
        review = json.loads(v[5])
        item = review["records"][0]["review"]
        item["evidence_intervals_seconds"] = [[1.001, 2.002], [1.001, 2.002]]
        item["interval_rationales"] = ["first entry", "duplicate entry"]
        review_raw = encode(review)
        index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        result = audit(v[0], v[1], v[2], v[3], review, review_raw,
                       index, encode(index), hashlib.sha256(v[3]).hexdigest())
        self.assertEqual(result["status"], "unvalidated_handoff_structurally_valid")
        self.assertEqual(result["reviewed_records"], 3)
        self.assertEqual(review["records"][0]["review"]["evidence_intervals_seconds"],
                         [[1.001, 2.002], [1.001, 2.002]])

    def test_float_near_millisecond_is_rejected_without_tolerance_drift(self):
        v = fixture()
        review = json.loads(v[5])
        review["records"][0]["review"]["evidence_intervals_seconds"] = [[1.0010000000000001, 2.002]]
        review["records"][0]["review"]["interval_rationales"] = ["near-ms"]
        review_raw = encode(review)
        index = dict(v[6], review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        result = audit(v[0], v[1], v[2], v[3], review, review_raw,
                       index, encode(index), hashlib.sha256(v[3]).hexdigest())
        self.assertIn("review_intervals_invalid", result["failure_codes"])

    def test_custody_checks_manifest_bindings_file_modes_and_actual_bytes(self):
        v = fixture()
        worksheet, worksheet_raw, lineage, lineage_raw = v[:4]
        rows = []
        import tempfile, os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp) / "media"; media_dir.mkdir(mode=0o700)
            for ordinal, row in enumerate(worksheet["records"], 1):
                raw = f"synthetic video {ordinal}".encode()
                digest = hashlib.sha256(raw).hexdigest()
                # Test against a matching private worksheet hash without needing a source video.
                row["video_sha256"] = digest
                path = media_dir / (row["review_token"] + ".mp4"); path.write_bytes(raw); os.chmod(path, 0o600)
                rows.append({"review_token": row["review_token"], "row_ordinal": ordinal,
                             "media_file": path.name, "source_media_sha256": digest,
                             "staged_media_sha256": digest, "size_bytes": len(raw),
                             "media_check_salt": "01" * 32,
                             "media_check_sha256": hashlib.sha256(bytes.fromhex("01" * 32) + raw).hexdigest()})
            worksheet_raw = encode(worksheet)
            custody = {"schema_version": 1, "purpose": "private_custodian_token_to_verified_media_manifest",
                       "created_at_utc": "2026-10-16T06:00:00Z",
                       "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
                       "source_lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
                       "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
                       "record_count": 3, "records": rows}
            raw = encode(custody)
            self.assertEqual(audit_custody(worksheet, worksheet_raw, lineage, lineage_raw, custody, raw,
                                           media_dir, hashlib.sha256(raw).hexdigest()), [])
            (media_dir / rows[0]["media_file"]).write_bytes(b"tampered")
            self.assertIn("custody_media_hash_mismatch", audit_custody(
                worksheet, worksheet_raw, lineage, lineage_raw, custody, raw, media_dir,
                hashlib.sha256(raw).hexdigest()))

    def test_invalid_worksheet_token_is_rejected_before_path_access_and_extras_fail(self):
        v = fixture(); worksheet, worksheet_raw, lineage, lineage_raw = v[:4]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); media_dir = root / "media"; media_dir.mkdir(mode=0o700)
            sentinel = root / "sentinel"; sentinel.write_text("must remain untouched")
            worksheet["records"][0]["review_token"] = "../sentinel"
            invalid_raw = encode(worksheet)
            custody = {"schema_version":1,"purpose":"private_custodian_token_to_verified_media_manifest",
                       "created_at_utc":"2026-10-16T06:00:00Z","worksheet_sha256":hashlib.sha256(invalid_raw).hexdigest(),
                       "source_lineage_manifest_sha256":hashlib.sha256(lineage_raw).hexdigest(),
                       "claim_review_policy_sha256":worksheet["claim_review_policy_sha256"],"record_count":3,
                       "records":[{"review_token":r["review_token"]} for r in worksheet["records"]]}
            custody_raw=encode(custody)
            failures = audit_custody(worksheet, invalid_raw, lineage, lineage_raw, custody, custody_raw,
                                     media_dir, "0" * 64)
            self.assertIn("custody_worksheet_token_invalid", failures)
            self.assertEqual(sentinel.read_text(), "must remain untouched")

        v = fixture(); worksheet, worksheet_raw, lineage, lineage_raw = v[:4]
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp) / "media"; media_dir.mkdir(mode=0o700)
            rows=[]
            for ordinal, row in enumerate(worksheet["records"], 1):
                data=f"extra fixture {ordinal}".encode(); digest=hashlib.sha256(data).hexdigest()
                row["video_sha256"]=digest
                path=media_dir/(row["review_token"]+".mp4");path.write_bytes(data)
                import os
                os.chmod(path,0o600)
                salt=b"x"*32
                rows.append({"review_token":row["review_token"],"row_ordinal":ordinal,"media_file":path.name,
                             "source_media_sha256":digest,"staged_media_sha256":digest,"size_bytes":len(data),
                             "media_check_salt":salt.hex(),"media_check_sha256":hashlib.sha256(salt+data).hexdigest()})
            worksheet_raw=encode(worksheet)
            custody={"schema_version":1,"purpose":"private_custodian_token_to_verified_media_manifest",
                     "created_at_utc":"2026-10-16T06:00:00Z","worksheet_sha256":hashlib.sha256(worksheet_raw).hexdigest(),
                     "source_lineage_manifest_sha256":hashlib.sha256(lineage_raw).hexdigest(),
                     "claim_review_policy_sha256":worksheet["claim_review_policy_sha256"],"record_count":3,"records":rows}
            raw=encode(custody);(media_dir/"extra.txt").write_text("extra");os.chmod(media_dir/"extra.txt",0o600)
            self.assertIn("custody_media_directory_contents_invalid",audit_custody(
                worksheet,worksheet_raw,lineage,lineage_raw,custody,raw,media_dir,hashlib.sha256(raw).hexdigest()))


if __name__ == "__main__":
    unittest.main()
