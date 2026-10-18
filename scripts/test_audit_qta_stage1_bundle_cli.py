from __future__ import annotations

import hashlib
import json
import io
from pathlib import Path
import shutil
import subprocess
import textwrap
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from audit_qta_stage1_bundle_cli import BundleError, PURPOSE, verify_bundle
import audit_qta_stage1_bundle_cli as cli_module
from nextgqa_completed_review_attestation import create_attestation


class Stage1BundleCliTests(unittest.TestCase):
    def setUp(self):
        from test_qta_stage1_bundle_audit import QtaStage1BundleAuditTests
        self.fixture = QtaStage1BundleAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir(mode=0o700)
        source_names = (
            "worksheet", "lineage", "review", "review_index", "custody",
            "pre_review_receipt", "integrity_roster", "release_roster", "inventory",
            "release_witness", "reviewer_attestation", "root_public_key",
        )
        source_bytes = {
            "worksheet": self.fixture.worksheet_raw,
            "lineage": self.fixture.lineage_raw,
            "review": self.fixture.review_raw,
            "review_index": self.fixture.review_index_raw,
            "custody": self.fixture.custody_raw,
            "pre_review_receipt": self.fixture.receipt_raw,
            "integrity_roster": self.fixture.f.integrity_roster_raw,
            "release_roster": self.fixture.f.release_roster_raw,
            "inventory": self.fixture.f.inventory_raw,
            "release_witness": self.fixture.release_witness_raw,
            "reviewer_attestation": self.fixture.reviewer_attestation_raw,
            "root_public_key": self.fixture.f.keys["root"][1],
        }
        self.sources = {}
        for name in source_names:
            rel = f"sources/{name}.bin"
            self._write(rel, source_bytes[name])
            self.sources[name] = rel
        self.artifact_sources = {}
        for index, (role, raw) in enumerate(sorted(self.fixture.f.artifacts.items())):
            rel = f"artifacts/{index:04d}.bin"
            self._write(rel, raw)
            self.artifact_sources[role] = rel
        media = self.root / "media"
        media.mkdir(mode=0o700)
        for path in self.fixture.media_dir.iterdir():
            shutil.copyfile(path, media / path.name)
            (media / path.name).chmod(0o600)
        self.manifest = {
            "schema_version": 1, "purpose": PURPOSE, "sources": self.sources,
            "media_dir": "media", "artifact_sources": self.artifact_sources,
        }
        self._save_manifest()

    def _write(self, relative: str, raw: bytes):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        current = path.parent
        while current != self.root:
            current.chmod(0o700)
            current = current.parent
        path.write_bytes(raw)
        path.chmod(0o600)

    def _save_manifest(self):
        self._write("bundle.json", json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")).encode())

    def _verify(self):
        return verify_bundle(
            bundle_dir=self.root,
            expected_lineage_sha256=self.fixture.expected_lineage_sha256,
            expected_custody_sha256=self.fixture.expected_custody_sha256,
            expected_root_public_der_sha256=self.fixture.f.root_pin,
            expected_ceremony_id=self.fixture.f.ceremony)

    def _render_synthetic_browser_export(self):
        """Run the rendered page's actual export handler with synthetic state."""
        if shutil.which("node") is None:
            self.skipTest("Node.js unavailable for rendered export handler integration")
        html = self.fixture.f.artifacts["review_package_html"].decode("utf-8")
        harness = textwrap.dedent(r'''
            const vm = require('node:vm');
            const crypto = require('node:crypto').webcrypto;
            let source = '';
            process.stdin.setEncoding('utf8');
            process.stdin.on('data', chunk => source += chunk);
            process.stdin.on('end', async () => {
              const dataMarker = '<script id="data" type="application/json">';
              const dataStart = source.indexOf(dataMarker) + dataMarker.length;
              const dataEnd = source.indexOf('</script>', dataStart);
              const scriptStart = source.lastIndexOf('<script>');
              const scriptEnd = source.indexOf('</script>', scriptStart);
              if (dataStart < dataMarker.length || dataEnd < 0 || scriptStart < 0 || scriptEnd < 0)
                throw new Error('rendered_script_missing');
              const base = JSON.parse(source.slice(dataStart, dataEnd));
              const script = source.slice(scriptStart + '<script>'.length, scriptEnd);
              const marker = "document.getElementById('export').onclick=async()=>{";
              const begin = script.indexOf(marker);
              const finish = script.indexOf("\n};\nfunction downloadReviewIndex()", begin);
              if (begin < 0 || finish < 0) throw new Error('export_handler_missing');
              const handler = script.slice(begin, finish + 3);
              const indexStart = script.indexOf('function downloadReviewIndex(){', finish);
              const clockStart = script.indexOf(
                "document.getElementById('export-clock').onclick", indexStart);
              if (indexStart < 0 || clockStart < 0) throw new Error('index_handler_missing');
              const indexHandler = script.slice(indexStart, clockStart);
              const downloads = [];
              const controls = new Map();
              const firstPlayedAt = new Date(Date.now() - 120000).toISOString();
              const reviewedAt = new Date(Date.now() - 60000).toISOString();
              for (const id of ['export','reviewer','role','attest','export-disclosure',
                'retry-review','retry-index','export-index','handoff-status'])
                controls.set(id, {disabled:false, value:'synthetic-reviewer', checked:true, textContent:''});
              controls.get('role').value = 'independent_human';
              const context = {
                base,
                state: Array.from({length:3}, () => ({tokenOK:true, duration:3.0,
                  firstPlayedAt, full:true, context:true,
                  seedHidden:true, outcomeHidden:true, decision:'supports_claim',
                  intervals:[{start:1.001,end:2.002,rationale:'synthetic rationale'}],
                  reviewedAt})),
                exportInProgress:false,
                rows:{querySelectorAll:() => []},
                document:{getElementById:id => controls.get(id)},
                window:{}, crypto, TextEncoder, Date, navigator:{userAgent:'Synthetic'},
                setTimeout:() => 0, alert:() => {}, checkReady:() => {},
                download:(content,name) => downloads.push({content,name}),
              };
              vm.runInNewContext(handler, context);
              await controls.get('export').onclick();
              vm.runInNewContext(indexHandler, context);
              controls.get('export-index').onclick();
              process.stdout.write(JSON.stringify({downloads, index:context.window.pendingReviewIndex,
                status:controls.get('handoff-status').textContent}));
            });
        ''')
        completed = subprocess.run(
            ["node", "-e", harness], input=html, text=True,
            capture_output=True, check=False, timeout=15,
        )
        if completed.returncode != 0:
            raise AssertionError("synthetic export harness failed: " + completed.stderr[:500])
        result = json.loads(completed.stdout)
        if len(result["downloads"]) != 2:
            raise AssertionError("browser export did not generate both downloads: " +
                                 result.get("status", "unknown"))
        review_download, index_download = result["downloads"]
        self.assertTrue(review_download["name"].endswith(".json"))
        self.assertEqual(index_download["name"], "nextgqa_human_review_index.json")
        self.assertEqual(index_download["content"], result["index"])
        return review_download["content"].encode("utf-8"), index_download["content"].encode("utf-8")

    def test_bundle_directory_matches_existing_audit_and_never_authorizes(self):
        report = self._verify()
        self.assertEqual(report, self.fixture._audit())
        self.assertEqual(report["status"], "stage1_structurally_valid_non_authorizing")
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["score_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_cli_denies_a_review_export_changed_after_signed_bundle_freeze(self):
        changed = self.root / "sources/review_changed.bin"
        self._write("sources/review_changed.bin",
                    (self.root / self.sources["review"]).read_bytes() + b" ")
        self.manifest["sources"]["review"] = "sources/review_changed.bin"
        self._save_manifest()

        output = io.StringIO()
        with redirect_stdout(output):
            status = cli_module.main([
                "--bundle-dir", str(self.root),
                "--expected-lineage-sha256", self.fixture.expected_lineage_sha256,
                "--expected-custody-sha256", self.fixture.expected_custody_sha256,
                "--expected-root-public-der-sha256", self.fixture.f.root_pin,
                "--expected-ceremony-id", self.fixture.f.ceremony,
            ])

        self.assertEqual(status, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["failure_codes"], ["stage1_audit_contract_invalid"])
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["score_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

        for row in self.fixture.worksheet["records"]:
            self.assertNotIn(row["review_token"], output.getvalue())
        self.assertTrue(changed.is_file())

    def test_actual_rendered_page_export_flows_through_bundle_cli_without_translation(self):
        review_raw, review_index_raw = self._render_synthetic_browser_export()
        review = json.loads(review_raw)
        review_index = json.loads(review_index_raw)
        self.assertEqual(review["purpose"], "anonymous_local_claim_video_human_review")
        self.assertEqual(len(review["records"]), 3)
        self.assertEqual(review_index["review_payload_sha256"],
                         hashlib.sha256(review_raw).hexdigest())

        reviewer_attestation_raw = create_attestation(
            worksheet=self.fixture.worksheet,
            worksheet_raw=self.fixture.worksheet_raw,
            lineage=self.fixture.lineage,
            lineage_raw=self.fixture.lineage_raw,
            review=review,
            review_raw=review_raw,
            review_index=review_index,
            review_index_raw=review_index_raw,
            expected_lineage_sha256=self.fixture.expected_lineage_sha256,
            pre_review_receipt_sha256=hashlib.sha256(self.fixture.receipt_raw).hexdigest(),
            integrity_roster_raw=self.fixture.f.integrity_roster_raw,
            root_public_key_pem=self.fixture.f.keys["root"][1],
            expected_root_public_der_sha256=self.fixture.f.root_pin,
            reviewer_private_key_pem=self.fixture.f.keys["claim"][0],
            reviewer_public_key_pem=self.fixture.f.keys["claim"][1],
        )
        for name, raw in (("review", review_raw), ("review_index", review_index_raw),
                          ("reviewer_attestation", reviewer_attestation_raw)):
            self._write(self.sources[name], raw)

        output = io.StringIO()
        with redirect_stdout(output):
            status = cli_module.main([
                "--bundle-dir", str(self.root),
                "--expected-lineage-sha256", self.fixture.expected_lineage_sha256,
                "--expected-custody-sha256", self.fixture.expected_custody_sha256,
                "--expected-root-public-der-sha256", self.fixture.f.root_pin,
                "--expected-ceremony-id", self.fixture.f.ceremony,
            ])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "stage1_structurally_valid_non_authorizing")
        self.assertEqual(report["structurally_valid_records"], 3)
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["score_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

        self._write("sources/review-export-mutated.bin", review_raw + b" ")
        self.manifest["sources"]["review"] = "sources/review-export-mutated.bin"
        self._save_manifest()
        blocked_output = io.StringIO()
        with redirect_stdout(blocked_output):
            blocked_status = cli_module.main([
                "--bundle-dir", str(self.root),
                "--expected-lineage-sha256", self.fixture.expected_lineage_sha256,
                "--expected-custody-sha256", self.fixture.expected_custody_sha256,
                "--expected-root-public-der-sha256", self.fixture.f.root_pin,
                "--expected-ceremony-id", self.fixture.f.ceremony,
            ])
        self.assertEqual(blocked_status, 2)
        blocked = json.loads(blocked_output.getvalue())
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["failure_codes"], ["stage1_audit_contract_invalid"])
        self.assertFalse(blocked["model_load_permitted"])
        self.assertFalse(blocked["score_ready"])
        self.assertEqual(blocked["execution_authorization"], "not_granted")
        for row in self.fixture.worksheet["records"]:
            self.assertNotIn(row["review_token"], blocked_output.getvalue())

    def test_inventory_role_omission_is_rejected_before_audit(self):
        self.manifest["artifact_sources"].pop("claim_lineage")
        self._save_manifest()
        with self.assertRaisesRegex(BundleError, "bundle_artifact_role_set_mismatch"):
            self._verify()

    def test_parent_traversal_is_rejected(self):
        self.manifest["sources"]["worksheet"] = "../worksheet.json"
        self._save_manifest()
        with self.assertRaisesRegex(BundleError, "bundle_relative_path_invalid"):
            self._verify()

    def test_symlink_source_is_rejected(self):
        path = self.root / self.sources["worksheet"]
        path.unlink()
        path.symlink_to(self.root / self.sources["lineage"])
        with self.assertRaisesRegex(BundleError, "bundle_symlink_rejected"):
            self._verify()

    def test_group_permissions_are_required(self):
        self.root.chmod(0o750)
        with self.assertRaisesRegex(BundleError, "bundle_root_permissions_invalid"):
            self._verify()

    def test_nested_directory_permissions_are_required(self):
        (self.root / "sources").chmod(0o755)
        with self.assertRaisesRegex(BundleError, "bundle_directory_permissions_invalid"):
            self._verify()

    def test_bundle_size_limit_is_checked_before_reading_all_artifacts(self):
        with patch.object(cli_module, "MAX_TOTAL_BYTES", 100), \
                patch.object(cli_module, "audit_qta_stage1_bundle",
                             side_effect=AssertionError("audit must not run")) as audit:
            with self.assertRaisesRegex(BundleError, "bundle_total_bytes_exceeded"):
                self._verify()
        audit.assert_not_called()

    def test_bundle_size_limit_includes_media_before_audit_reads_it(self):
        nonmedia_bytes = sum(
            path.stat().st_size for path in self.root.rglob("*")
            if path.is_file() and not path.is_relative_to(self.root / "media")
        )
        with patch.object(cli_module, "MAX_TOTAL_BYTES", nonmedia_bytes), \
                patch.object(cli_module, "audit_qta_stage1_bundle",
                             side_effect=AssertionError("audit must not run")) as audit:
            with self.assertRaisesRegex(BundleError, "bundle_total_bytes_exceeded"):
                self._verify()
        audit.assert_not_called()

    def test_audit_cannot_return_permissive_success(self):
        permissive = {
            "status": "stage1_structurally_valid_non_authorizing",
            "requested_records": 3.0, "effective_records": 3,
            "structurally_valid_records": 3, "failure_count": 0,
            "failure_codes": [], "model_load_permitted": True,
            "score_ready": False, "reviewer_identity_verified": False,
            "semantic_review_authenticated": False,
            "frame_correspondence_verified": False,
            "execution_authorization": "not_granted",
        }
        with patch.object(cli_module, "audit_qta_stage1_bundle", return_value=permissive):
            with self.assertRaisesRegex(BundleError, "stage1_audit_contract_invalid"):
                self._verify()

    def test_external_pin_format_is_required(self):
        with self.assertRaisesRegex(BundleError, "expected_lineage_sha256_invalid"):
            verify_bundle(bundle_dir=self.root, expected_lineage_sha256="bad",
                          expected_custody_sha256=self.fixture.expected_custody_sha256,
                          expected_root_public_der_sha256=self.fixture.f.root_pin,
                          expected_ceremony_id=self.fixture.f.ceremony)


if __name__ == "__main__":
    unittest.main()
