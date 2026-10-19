from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_audit_qta_stage1_bundle_cli as stage1_cli_tests
from nextgqa_prediction_link_attestation import create_attestation
from nextgqa_prediction_link_review import displayed_fields_sha256
from nextgqa_pre_review_integrity_receipt import _canonical, _sha
from qta_test_baseline_v3_fixture import build_synthetic_baseline_v3

from nextgqa_formal_review_workspace import (
    FormalReviewSession, WorkspaceServer, local_host_allowed,
    local_post_allowed, workspace_page, workspace_url,
)
import nextgqa_formal_review_workspace as workspace_module


def _write(path: Path, raw: bytes):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


class FormalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.stage1 = stage1_cli_tests.Stage1BundleCliTests()
        self.stage1.setUp()
        self.addCleanup(self.stage1.doCleanups)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stage2_dir = Path(self.tmp.name) / "stage2"
        self.stage2_dir.mkdir(mode=0o700)
        self._write_stage2()
        f = self.stage1.fixture
        self.session = FormalReviewSession(
            stage1_bundle_dir=self.stage1.root,
            stage2_bundle_dir=self.stage2_dir,
            expected_lineage_sha256=f.expected_lineage_sha256,
            expected_custody_sha256=f.expected_custody_sha256,
            expected_root_public_der_sha256=f.f.root_pin,
            expected_ceremony_id=f.f.ceremony,
            expected_stage2_context_sha256=hashlib.sha256(
                (self.stage2_dir / "context.json").read_bytes()).hexdigest())

    def _write_stage2(self):
        f = self.stage1.fixture
        receipt_hash = _sha(f.receipt_raw)
        raw, rows, source_hashes = build_synthetic_baseline_v3(receipt_hash)
        baseline = json.loads(raw)
        tokens = [row["review_token"] for row in f.worksheet["records"]]
        for row, token in zip(baseline["records"], tokens, strict=True):
            row["review_token"] = token
            row["score_binding"]["review_token"] = token
            row["score_binding_sha256"] = _sha(_canonical(row["score_binding"]))
        baseline_raw = _canonical(baseline)
        rows = baseline["records"]
        source_hashes = {
            row["review_token"]: {
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "options_sha256": hashlib.sha256(_canonical(row["options"])).hexdigest(),
                "claim_sha256": hashlib.sha256(row["claim"].encode()).hexdigest(),
                "video_support_decision": row["video_support_decision"],
                "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
            } for row in rows
        }
        protocol_sha = "b" * 64
        receipt = {
            "schema_version": 1, "purpose": "qta_prediction_link_adjudication",
            "status": "prediction_link_review_recorded_unvalidated",
            "baseline_manifest_sha256": _sha(baseline_raw),
            "protocol_sha256": protocol_sha,
            "prior_video_review_receipt_sha256": receipt_hash,
            "reviewer_role": "independent_prediction_link_reviewer",
            "reviewer_private_id": "synthetic-reviewer",
            "independence_attestation":
                "independent_from_generation_scoring_custody_and_video_review",
            "gold_and_branch_outcomes_hidden": True,
            "reviewed_at_utc": "2026-10-18T10:00:00Z",
            "records": [{
                "review_token": row["review_token"],
                "displayed_fields_sha256": displayed_fields_sha256(
                    row, video_claim_review_receipt_sha256=receipt_hash),
                "judgment": "supports_p_over_q", "rationale": "Synthetic fixture only."}
                for row in rows],
            "execution_authorization": "not_granted",
        }
        receipt_raw = _canonical(receipt)
        expected = dict(
            receipt_raw=receipt_raw, baseline_manifest_raw=baseline_raw,
            expected_baseline_manifest_sha256=_sha(baseline_raw),
            expected_protocol_sha256=protocol_sha,
            expected_video_review_receipt_sha256=receipt_hash,
            expected_cohort_tokens=tokens,
            expected_source_field_sha256_by_token=source_hashes,
            integrity_roster_raw=f.f.integrity_roster_raw,
            release_roster_raw=f.f.release_roster_raw,
            root_public_key_pem=f.f.keys["root"][1],
            expected_root_public_der_sha256=f.f.root_pin)
        attestation_raw = create_attestation(
            **expected,
            reviewer_private_key_pem=f.f.keys["prediction"][0],
            reviewer_public_key_pem=f.f.keys["prediction"][1])
        context = {
            "schema_version": 1,
            "purpose": "qta_formal_workspace_expected_stage2_pins_v1",
            "protocol_sha256": protocol_sha,
            "baseline_manifest_sha256": _sha(baseline_raw),
            "cohort_tokens": tokens,
            "source_field_sha256_by_token": source_hashes,
        }
        manifest = {
            "schema_version": 1,
            "purpose": "qta_formal_workspace_stage2_bundle_v1",
            "baseline_manifest": "baseline.json",
            "prediction_link_receipt": "receipt.json",
            "stage2_attestation": "attestation.json",
            "context": "context.json",
        }
        for name, data in (
                ("baseline.json", baseline_raw), ("receipt.json", receipt_raw),
                ("attestation.json", attestation_raw),
                ("context.json", _canonical(context)),
                ("workspace.json", _canonical(manifest))):
            _write(self.stage2_dir / name, data)

    def test_stage1_then_stage2_exact_binding_and_denial(self):
        stage1 = self.session.verify_stage1()
        self.assertTrue(stage1["stage1_verified"])
        self.assertEqual(stage1["stage1"]["pre_review_receipt_sha256"],
                         _sha(self.stage1.fixture.f.receipt_raw))
        stage2 = self.session.verify_stage2()
        self.assertTrue(stage2["stage2_verified"])
        self.assertEqual(stage2["stage2"]["stage1_pre_review_receipt_sha256"],
                         stage1["stage1"]["pre_review_receipt_sha256"])
        self.assertEqual(stage2["stage2"]["stage2_context_sha256"],
                         self.session.expected_stage2_context_sha256)
        self.assertTrue(stage2["stage2"]["reviewer_key_signature_valid"])
        self.assertFalse(stage2["stage2"]["reviewer_identity_verified"])
        self.assertFalse(stage2["model_load_permitted"])
        self.assertEqual(stage2["authorized_forwards"], 0)
        reset = self.session.reset()
        self.assertIsNone(reset)
        reset_state = self.session.state()
        self.assertIsNone(reset_state["stage2"])
        self.assertFalse(reset_state["stage2_verified"])

    def test_source_change_clears_downstream_and_stage1_change_clears_all(self):
        self.session.verify_stage1()
        self.assertTrue(self.session.verify_stage2()["stage2_verified"])
        with (self.stage2_dir / "baseline.json").open("ab") as stream:
            stream.write(b" ")
        state = self.session.state()
        self.assertTrue(state["stage1_verified"])
        self.assertFalse(state["stage2_verified"])
        self.assertIsNone(state["stage2"])
        self.assertEqual(state["failure_code"], "stage2_source_changed_reload_required")

        self._write_stage2()
        self.session.verify_stage1()
        self.session.verify_stage2()
        source = self.stage1.root / "sources" / "pre_review_receipt.bin"
        with source.open("ab") as stream:
            stream.write(b" ")
        state = self.session.state()
        self.assertFalse(state["stage1_verified"])
        self.assertFalse(state["stage2_verified"])
        self.assertIsNone(state["stage1"])
        self.assertIsNone(state["stage2"])

    def test_stage1_source_mutation_during_audit_fails_closed(self):
        self.assertTrue(self.session.verify_stage1()["stage1_verified"])
        self.assertTrue(self.session.verify_stage2()["stage2_verified"])

        review_path = self.stage1.root / self.stage1.sources["review"]
        real_audit = workspace_module.audit_qta_stage1_bundle

        def mutate_after_audit(**kwargs):
            report = real_audit(**kwargs)
            with review_path.open("ab") as stream:
                stream.write(b" ")
            return report

        with patch.object(workspace_module, "audit_qta_stage1_bundle",
                          side_effect=mutate_after_audit) as audit:
            state = self.session.verify_stage1()

        audit.assert_called_once()
        self.assertFalse(state["stage1_verified"])
        self.assertFalse(state["stage2_verified"])
        self.assertIsNone(state["stage1"])
        self.assertIsNone(state["stage2"])
        self.assertEqual(state["failure_code"], "source_tree_changed_during_audit")
        self.assertFalse(state["model_load_permitted"])
        self.assertFalse(state["score_ready"])
        self.assertEqual(state["execution_authorization"], "not_granted")
        self.assertEqual(state["authorized_forwards"], 0)

    def test_wrong_stage1_upstream_digest_blocks_stage2(self):
        self.session.verify_stage1()
        manifest_path = self.stage2_dir / "baseline.json"
        baseline = json.loads(manifest_path.read_bytes())
        baseline["video_claim_review_receipt_sha256"] = "0" * 64
        baseline_raw = _canonical(baseline)
        _write(manifest_path, baseline_raw)
        context_path = self.stage2_dir / "context.json"
        context = json.loads(context_path.read_bytes())
        context["baseline_manifest_sha256"] = _sha(baseline_raw)
        context_raw = _canonical(context)
        _write(context_path, context_raw)
        self.session.expected_stage2_context_sha256 = _sha(context_raw)
        result = self.session.verify_stage2()
        self.assertFalse(result["stage2_verified"])
        self.assertIn("video_review_receipt_binding", result["failure_code"])
        self.assertEqual(result["authorized_forwards"], 0)

    def test_context_requires_exact_external_digest(self):
        self.session.verify_stage1()
        passed = self.session.verify_stage2()
        self.assertTrue(passed["stage2_verified"])
        self.session.expected_stage2_context_sha256 = "0" * 64
        result = self.session.verify_stage2()
        self.assertFalse(result["stage2_verified"])
        self.assertEqual(result["failure_code"], "stage2_context_external_pin_mismatch")
        self.assertIsNone(result["stage2"])

    def test_localhost_policy_page_and_non_authorization(self):
        self.assertEqual(WorkspaceServer.address_family, 2)  # AF_INET, never IPv6 wildcard.
        page = workspace_page("synthetic-session-token").decode()
        self.assertIn("connect-src 'self'", page)
        self.assertIn("execution_authorization=not_granted", page)
        self.assertNotIn("<input", page)
        port = 8765
        self.assertTrue(local_host_allowed(f"127.0.0.1:{port}", port))
        self.assertTrue(local_host_allowed(f"localhost:{port}", port))
        self.assertFalse(local_host_allowed(f"0.0.0.0:{port}", port))
        self.assertFalse(local_host_allowed(f"example.com:{port}", port))
        self.assertTrue(local_post_allowed(
            host=f"127.0.0.1:{port}", origin=f"http://127.0.0.1:{port}",
            supplied_token="secret", token="secret", port=port))
        self.assertFalse(local_post_allowed(
            host=f"127.0.0.1:{port}", origin="https://example.com",
            supplied_token="secret", token="secret", port=port))
        self.assertFalse(local_post_allowed(
            host=f"127.0.0.1:{port}", origin=f"http://127.0.0.1:{port}",
            supplied_token="wrong", token="secret", port=port))

    def test_startup_url_is_usable_and_loopback_only(self):
        token = "A" * 43
        url = workspace_url(token, 43210)
        self.assertEqual(url, f"http://127.0.0.1:43210/{token}/")
        self.assertIn("127.0.0.1", url)
        self.assertNotIn("0.0.0.0", url)
        self.assertNotIn("example.com", url)
        for invalid_token in ("short", "bad/token", "a" * 42, "?" + "a" * 42,
                              "a" * 42 + "#"):
            with self.subTest(invalid_token=invalid_token[:8]):
                with self.assertRaises(ValueError):
                    workspace_url(invalid_token, 43210)
        with self.assertRaises(ValueError):
            workspace_url("a" * 43, True)


if __name__ == "__main__":
    unittest.main()
