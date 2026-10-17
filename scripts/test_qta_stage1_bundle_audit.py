from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nextgqa_completed_review_attestation import create_attestation
from qta_stage1_bundle_audit import audit_qta_stage1_bundle
import qta_stage1_bundle_audit as stage1_module
from test_audit_nextgqa_review_handoff import fixture as review_fixture
from test_build_nextgqa_claim_mapping_packet import encode, packet
from build_nextgqa_claim_mapping_packet import apply_review_policy, build
from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from nextgqa_pre_review_integrity_receipt import _canonical, _sha, build_integrity_receipt
from render_nextgqa_claim_review_html import render


class QtaStage1BundleAuditTests(unittest.TestCase):
    def setUp(self):
        from test_audit_nextgqa_review_release_order import ReviewReleaseOrderTests
        self.release_fixture = ReviewReleaseOrderTests()
        self.release_fixture.setUp()
        self.addCleanup(self.release_fixture.tearDown)
        _, _, _, _, template_review, _, _, _ = review_fixture()
        source, _ = packet(3)
        from build_nextgqa_v2_claim_input_bundle import V2_COHORT_SHA256
        source["cohort_sha256"] = V2_COHORT_SHA256
        source["model_revision"] = (
            "Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5")
        for i, row in enumerate(source["records"], 1):
            media = f"synthetic-media-{i}".encode()
            row["video_sha256"] = hashlib.sha256(media).hexdigest()
        source_raw = encode(source)
        self.worksheet, _ = build(source, source_raw)
        policy_path = Path(__file__).resolve().parents[1] / "docs" / "nextgqa_claim_review_policy_v6.json"
        self.worksheet = apply_review_policy(self.worksheet, json.loads(policy_path.read_bytes()),
                                             policy_path.read_bytes())
        self.worksheet["model_revision"] = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        lineage_records = []
        for i, row in enumerate(self.worksheet["records"]):
            lineage_records.append({
                "video_id": row["video_id"], "claim_id": row["claim_id"],
                "dataset_question_id": f"qid-{i}", "dataset_video_id": f"vid-{i}",
                "video_key": f"group/video-{i}", "relative_video_path": f"group/video-{i}.mp4",
                "video_sha256": row["video_sha256"],
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "claim_text_sha256": row["claim_text_sha256"],
            })
        self.lineage = {
            "schema_version": 1,
            "purpose": "private_source_lineage_crosscheck_for_claim_review",
            "created_by": "audit_nextgqa_claim_packet_lineage.py",
            "source_packet_sha256": self.worksheet["source_packet_sha256"],
            "cohort_sha256": self.worksheet["cohort_sha256"],
            "dataset_inputs_sha256": self.worksheet["dataset_inputs_sha256"],
            "requested_records": 3, "effective_records": 3,
            "verification_scope": "synthetic metadata crosscheck",
            "status": "metadata_lineage_crosschecked", "records": lineage_records,
        }
        self.review = dict(template_review)
        self.review["records"] = []
        for row, original in zip(self.worksheet["records"], template_review["records"], strict=True):
            self.review["records"].append({"review_token": row["review_token"],
                                           "review": dict(original["review"])})
        self.review_index = None
        self.media_root = tempfile.TemporaryDirectory()
        self.addCleanup(self.media_root.cleanup)
        self.media_dir = Path(self.media_root.name) / "media"
        self.media_dir.mkdir(mode=0o700)
        # Synthetic media bytes are privately staged and their hashes are
        # rebound consistently through worksheet, lineage, and custody.
        for ordinal, (row, line) in enumerate(zip(self.worksheet["records"],
                                                   self.lineage["records"], strict=True), 1):
            raw = f"synthetic-media-{ordinal}".encode()
            digest = hashlib.sha256(raw).hexdigest()
            row["video_sha256"] = digest
            line["video_sha256"] = digest
            path = self.media_dir / f"{row['review_token']}.mp4"
            path.write_bytes(raw)
            path.chmod(0o600)
        self.worksheet_raw = encode(self.worksheet)
        self.lineage_raw = encode(self.lineage)
        self.review_raw = encode(self.review)
        self.review_index = {
            "schema_version": 1,
            "purpose": "anonymous_unvalidated_human_review_handoff",
            "review_payload_sha256": hashlib.sha256(self.review_raw).hexdigest(),
            "claim_review_policy_sha256": FROZEN_POLICY_SHA256,
            "created_at_utc": "2026-10-16T06:00:00.000Z",
            "review_status": "unvalidated_requires_custodian_audit",
            "execution_authorization": "not_granted",
        }
        self.review_index_raw = encode(self.review_index)
        custody_rows = []
        for ordinal, row in enumerate(self.worksheet["records"], 1):
            media_raw = (self.media_dir / f"{row['review_token']}.mp4").read_bytes()
            salt = bytes([ordinal]) * 32
            custody_rows.append({
                "review_token": row["review_token"], "row_ordinal": ordinal,
                "media_file": f"{row['review_token']}.mp4",
                "source_media_sha256": hashlib.sha256(media_raw).hexdigest(),
                "staged_media_sha256": hashlib.sha256(media_raw).hexdigest(),
                "size_bytes": len(media_raw), "media_check_salt": salt.hex(),
                "media_check_sha256": hashlib.sha256(salt + media_raw).hexdigest(),
            })
        self.custody = {
            "schema_version": 1,
            "purpose": "private_custodian_token_to_verified_media_manifest",
            "created_at_utc": "2026-10-16T06:00:00Z",
            "worksheet_sha256": hashlib.sha256(self.worksheet_raw).hexdigest(),
            "source_lineage_manifest_sha256": hashlib.sha256(self.lineage_raw).hexdigest(),
            "claim_review_policy_sha256": self.worksheet["claim_review_policy_sha256"],
            "record_count": 3, "records": custody_rows,
        }
        self.custody_raw = encode(self.custody)
        self.expected_lineage_sha256 = hashlib.sha256(self.lineage_raw).hexdigest()
        self.expected_custody_sha256 = hashlib.sha256(self.custody_raw).hexdigest()
        self.f = self.release_fixture
        from test_render_nextgqa_claim_review_html import RenderNextgqaClaimReviewHtmlTests
        manifest, manifest_raw, sidecar, sidecar_raw = RenderNextgqaClaimReviewHtmlTests().bundle(
            self.worksheet, self.worksheet_raw, self.lineage, self.lineage_raw)
        html = render(self.worksheet, self.worksheet_raw, self.lineage, self.lineage_raw,
                      self.expected_lineage_sha256, self.custody, self.custody_raw,
                      self.expected_custody_sha256, manifest, manifest_raw, sidecar, sidecar_raw)
        self.f.artifacts.update({
            "claim_mapping_worksheet": self.worksheet_raw,
            "claim_lineage": self.lineage_raw,
            "review_custody": self.custody_raw,
            "claim_input_manifest": manifest_raw,
            "claim_input_sidecar": sidecar_raw,
            "review_package_html": html,
            "standalone_review_html": html,
        })
        for index, basename in enumerate(sorted(p.name for p in self.media_dir.glob("*.mp4")), 1):
            raw = (self.media_dir / basename).read_bytes()
            self.f.artifacts[f"staged_media_{index:02d}"] = raw
            self.f.artifacts[f"review_package_media_{index:02d}"] = raw
        rows = [{"role": role, "size_bytes": len(raw), "sha256": _sha(raw)}
                for role, raw in sorted(self.f.artifacts.items())]
        self.f.inventory_raw = _canonical({
            "schema_version": 1, "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only", "review_started": False,
            "review_completed": False, "joint_freeze": False,
            "score_ready": False, "execution_authorization": "not_granted",
            "rights_status": "unverified_local_only", "artifacts": rows,
            "missing_required_slots": [],
        }) + b"\n"
        self.f.receipt_raw = build_integrity_receipt(
            roster_raw=self.f.integrity_roster_raw,
            inventory_raw=self.f.inventory_raw, artifact_sources=self.f.artifacts,
            root_public_key_pem=self.f.keys["root"][1],
            expected_root_public_der_sha256=self.f.root_pin,
            owner_key_id="owner-key", owner_private_key_pem=self.f.keys["owner"][0],
            owner_public_key_pem=self.f.keys["owner"][1],
            created_at_utc="2026-10-17T12:00:00Z")
        self.f.attestation_raw = self.f._attestation(self.f.ceremony)
        self.receipt_raw = self.f.receipt_raw
        self.release_witness_raw = self.f.attestation_raw
        self.reviewer_attestation_raw = create_attestation(
            worksheet=self.worksheet, worksheet_raw=self.worksheet_raw,
            lineage=self.lineage, lineage_raw=self.lineage_raw,
            review=self.review, review_raw=self.review_raw,
            review_index=self.review_index, review_index_raw=self.review_index_raw,
            expected_lineage_sha256=self.expected_lineage_sha256,
            pre_review_receipt_sha256=hashlib.sha256(self.receipt_raw).hexdigest(),
            integrity_roster_raw=self.f.integrity_roster_raw,
            root_public_key_pem=self.f.keys["root"][1],
            expected_root_public_der_sha256=self.f.root_pin,
            reviewer_private_key_pem=self.f.keys["claim"][0],
            reviewer_public_key_pem=self.f.keys["claim"][1],
        )

    def _audit(self, **overrides):
        args = {
            "worksheet_raw": self.worksheet_raw,
            "lineage_raw": self.lineage_raw,
            "review_raw": self.review_raw,
            "review_index_raw": self.review_index_raw,
            "custody_raw": self.custody_raw,
            "pre_review_receipt_raw": self.receipt_raw,
            "integrity_roster_raw": self.f.integrity_roster_raw,
            "release_roster_raw": self.f.release_roster_raw,
            "inventory_raw": self.f.inventory_raw,
            "release_witness_raw": self.release_witness_raw,
            "reviewer_attestation_raw": self.reviewer_attestation_raw,
            "media_dir": self.media_dir,
            "expected_lineage_sha256": self.expected_lineage_sha256,
            "expected_custody_sha256": self.expected_custody_sha256,
            "root_public_key_pem": self.f.keys["root"][1],
            "expected_root_public_der_sha256": self.f.root_pin,
            "expected_ceremony_id": self.f.ceremony,
            "artifact_sources": self.f.artifacts,
        }
        args.update(overrides)
        return audit_qta_stage1_bundle(**args)

    def _resign_artifacts(self, changed_role):
        artifacts = dict(self.f.artifacts)
        artifacts[changed_role] = artifacts[changed_role] + b"different-but-resigned"
        rows = [{"role": role, "size_bytes": len(raw), "sha256": _sha(raw)}
                for role, raw in sorted(artifacts.items())]
        self.f.inventory_raw = _canonical({
            "schema_version": 1, "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only", "review_started": False,
            "review_completed": False, "joint_freeze": False,
            "score_ready": False, "execution_authorization": "not_granted",
            "rights_status": "unverified_local_only", "artifacts": rows,
            "missing_required_slots": [],
        }) + b"\n"
        self.f.artifacts = artifacts
        self.f.receipt_raw = build_integrity_receipt(
            roster_raw=self.f.integrity_roster_raw,
            inventory_raw=self.f.inventory_raw, artifact_sources=artifacts,
            root_public_key_pem=self.f.keys["root"][1],
            expected_root_public_der_sha256=self.f.root_pin,
            owner_key_id="owner-key", owner_private_key_pem=self.f.keys["owner"][0],
            owner_public_key_pem=self.f.keys["owner"][1],
            created_at_utc="2026-10-17T12:00:00Z")
        self.f.attestation_raw = self.f._attestation(self.f.ceremony)
        self.receipt_raw = self.f.receipt_raw
        self.release_witness_raw = self.f.attestation_raw
        self.reviewer_attestation_raw = create_attestation(
            worksheet=self.worksheet, worksheet_raw=self.worksheet_raw,
            lineage=self.lineage, lineage_raw=self.lineage_raw,
            review=self.review, review_raw=self.review_raw,
            review_index=self.review_index, review_index_raw=self.review_index_raw,
            expected_lineage_sha256=self.expected_lineage_sha256,
            pre_review_receipt_sha256=hashlib.sha256(self.receipt_raw).hexdigest(),
            integrity_roster_raw=self.f.integrity_roster_raw,
            root_public_key_pem=self.f.keys["root"][1],
            expected_root_public_der_sha256=self.f.root_pin,
            reviewer_private_key_pem=self.f.keys["claim"][0],
            reviewer_public_key_pem=self.f.keys["claim"][1],
        )
        return artifacts

    def test_valid_synthetic_bundle_is_structural_only_and_non_authorizing(self):
        result = self._audit()
        self.assertEqual(result["status"], "stage1_structurally_valid_non_authorizing")
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertEqual(result["structurally_valid_records"], 3)
        self.assertEqual(result["failure_count"], 0)
        self.assertTrue(result["components"]["release_witness"]["passed"])
        self.assertTrue(result["components"]["reviewer_attestation"]["passed"])
        self.assertEqual(result["clock_frame_status"], "not_audited_separate_gate_required")
        self.assertFalse(result["model_load_permitted"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_success_summaries_cannot_be_injected(self):
        self.assertNotIn("review_report", inspect.signature(audit_qta_stage1_bundle).parameters)
        with self.assertRaises(TypeError):
            self._audit(review_report={"status": "unvalidated_handoff_structurally_valid"})

    def test_receipt_tamper_breaks_release_and_review_receipt_binding(self):
        result = self._audit(pre_review_receipt_raw=self.receipt_raw + b" ")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertEqual(result["structurally_valid_records"], 0)
        self.assertFalse(result["components"]["release_witness"]["passed"])
        self.assertFalse(result["model_load_permitted"])

    def test_review_export_tamper_fails_handoff_and_attestation(self):
        result = self._audit(review_raw=self.review_raw + b" ")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["components"]["review_handoff"]["passed"])
        self.assertFalse(result["components"]["reviewer_attestation"]["passed"])

    def test_custody_tamper_fails_and_does_not_echo_private_bytes(self):
        result = self._audit(custody_raw=self.custody_raw + b" ")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["components"]["custody"]["passed"])
        serialized = json.dumps(result)
        self.assertNotIn("synthetic-media", serialized)
        self.assertNotIn(self.review["human_reviewer_private_id"], serialized)
        for row in self.worksheet["records"]:
            self.assertNotIn(row["review_token"], serialized)

    def test_individually_resigned_cross_artifact_mismatches_are_rejected(self):
        roles = ("claim_mapping_worksheet", "claim_lineage", "review_custody",
                 "review_package_media_01", "review_package_html",
                 "standalone_review_html")
        for role in roles:
            with self.subTest(role=role):
                artifacts = self._resign_artifacts(role)
                result = self._audit(artifact_sources=artifacts)
                # The receipt, release witness, custody audit, and reviewer
                # adoption remain individually valid; the exact role join fails.
                self.assertTrue(result["components"]["release_witness"]["passed"])
                self.assertTrue(result["components"]["reviewer_attestation"]["passed"])
                self.assertTrue(result["components"]["custody"]["passed"])
                self.assertEqual(result["status"], "blocked")
                self.assertFalse(result["artifact_role_bindings"][role]["passed"])

    def test_media_mutation_after_snapshot_cannot_pass_aggregate(self):
        original = stage1_module.verify_review_release_order
        first = sorted(self.media_dir.glob("*.mp4"))[0]

        def verify_then_mutate(**kwargs):
            report = original(**kwargs)
            first.write_bytes(b"post-snapshot-tamper")
            first.chmod(0o600)
            return report

        with patch.object(stage1_module, "verify_review_release_order", verify_then_mutate):
            result = self._audit()
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["components"]["release_witness"]["passed"])
        self.assertTrue(result["components"]["custody"]["passed"])
        self.assertFalse(result["components"]["media_snapshot_stability"]["passed"])
        self.assertTrue(any(code.endswith(":media_path_changed_after_snapshot")
                            for code in result["failure_codes"]))

    def test_invalid_worksheet_token_never_opens_sibling_private_file(self):
        worksheet = json.loads(self.worksheet_raw)
        worksheet["records"][0]["review_token"] = "../sentinel"
        worksheet_raw = encode(worksheet)
        sibling = self.media_dir.parent / "sentinel.mp4"
        sibling.write_bytes(b"private-sibling-sentinel")
        sibling.chmod(0o600)
        original = stage1_module._read_private_media_at
        opened = []

        def tracked_open(dir_fd, basename):
            opened.append(basename)
            return original(dir_fd, basename)

        with patch.object(stage1_module, "_read_private_media_at", tracked_open):
            result = self._audit(worksheet_raw=worksheet_raw)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(opened, [])
        self.assertNotIn("private-sibling-sentinel", json.dumps(result))

    def test_symlinked_media_directory_is_rejected_before_media_open(self):
        real_dir = self.media_dir.with_name("real-media")
        self.media_dir.rename(real_dir)
        self.media_dir.symlink_to(real_dir, target_is_directory=True)
        def restore_media_directory():
            self.media_dir.unlink(missing_ok=True)
            if real_dir.exists():
                real_dir.rename(self.media_dir)
        self.addCleanup(restore_media_directory)
        with patch.object(stage1_module, "_read_private_media_at") as opened:
            result = self._audit()
        self.assertEqual(result["status"], "blocked")
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
