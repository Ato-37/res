import hashlib
import json
import os
import stat
import contextlib
import io
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import package_nextgqa_claim_review_handoff as package_cli
from package_nextgqa_claim_review_handoff import _package_review_after_audit
from render_nextgqa_claim_review_html import render
from build_nextgqa_v2_claim_input_bundle import V2_COHORT_SHA256
from test_audit_nextgqa_review_handoff import fixture as handoff_fixture, frozen_worksheet
from test_build_nextgqa_claim_mapping_packet import encode
from test_render_nextgqa_claim_review_html import RenderNextgqaClaimReviewHtmlTests


class PackageNextgqaReviewHandoffTests(unittest.TestCase):
    def cli_fixture(self, root, *, include_bundle=False):
        worksheet, _, lineage, _ = handoff_fixture()[:4]
        private = root / "data" / "private_nextgqa"
        private.mkdir(parents=True, mode=0o700)
        os.chmod(private, 0o700)
        media_dir = private / "media"
        media_dir.mkdir(mode=0o700)
        rows = []
        for ordinal, (row, lineage_row) in enumerate(
                zip(worksheet["records"], lineage["records"], strict=True), 1):
            media = f"synthetic CLI media {ordinal}".encode()
            digest = hashlib.sha256(media).hexdigest()
            row["video_sha256"] = digest
            row["video_id"] = "video-" + digest
            row["claim_id"] = "claim-" + hashlib.sha256(
                (digest + "\0" + lineage_row["dataset_question_id"] + "\0"
                 + row["claim_text_sha256"]).encode()).hexdigest()
            lineage_row["video_sha256"] = digest
            lineage_row["video_id"] = row["video_id"]
            lineage_row["claim_id"] = row["claim_id"]
            media_path = media_dir / f"{row['review_token']}.mp4"
            media_path.write_bytes(media)
            os.chmod(media_path, 0o600)
            salt = bytes([ordinal]) * 32
            rows.append({
                "review_token": row["review_token"], "row_ordinal": ordinal,
                "media_file": media_path.name, "source_media_sha256": digest,
                "staged_media_sha256": digest, "size_bytes": len(media),
                "media_check_salt": salt.hex(),
                "media_check_sha256": hashlib.sha256(salt + media).hexdigest(),
            })
        if include_bundle:
            worksheet["cohort_sha256"] = V2_COHORT_SHA256
            worksheet["model_revision"] = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
            lineage["cohort_sha256"] = V2_COHORT_SHA256
        worksheet_raw = encode(worksheet)
        lineage_raw = encode(lineage)
        custody = {
            "schema_version": 1,
            "purpose": "private_custodian_token_to_verified_media_manifest",
            "created_at_utc": "2026-10-16T06:00:00Z",
            "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
            "source_lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "claim_review_policy_sha256": worksheet["claim_review_policy_sha256"],
            "record_count": 3, "records": rows,
        }
        custody_raw = encode(custody)
        bundle_paths = None
        bundle_values = None
        if include_bundle:
            renderer_tests = RenderNextgqaClaimReviewHtmlTests()
            bundle_values = renderer_tests.bundle(worksheet, worksheet_raw, lineage, lineage_raw)
            bundle_paths = []
            for name, raw_value in zip(
                    ("claim-input.json", "claim-input-sidecar.json"),
                    (bundle_values[1], bundle_values[3]), strict=True):
                path = private / name
                path.write_bytes(raw_value)
                os.chmod(path, 0o600)
                bundle_paths.append(path)
        review_page = private / "review.html"
        page_raw = render(
            worksheet, worksheet_raw, lineage, lineage_raw,
            hashlib.sha256(lineage_raw).hexdigest(), custody, custody_raw,
            hashlib.sha256(custody_raw).hexdigest(),
            *(bundle_values if bundle_values is not None else (None, None, None, None)))
        review_page.write_bytes(page_raw)
        os.chmod(review_page, 0o600)
        inputs = []
        for name, raw in (("worksheet.json", worksheet_raw), ("lineage.json", lineage_raw),
                          ("custody.json", custody_raw)):
            path = private / name
            path.write_bytes(raw)
            os.chmod(path, 0o600)
            inputs.append(path)
        return private, media_dir, inputs, review_page, bundle_paths

    def test_cli_audits_and_binds_all_inputs_before_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            private, media_dir, inputs, review_page, bundle_paths = self.cli_fixture(
                root, include_bundle=True)
            output = private / "handoff"
            fake_file = root / "scripts" / "package_nextgqa_claim_review_handoff.py"
            fake_file.parent.mkdir(parents=True)
            argv = ["--worksheet", str(inputs[0]), "--lineage-manifest", str(inputs[1]),
                    "--expected-lineage-sha256", hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
                    "--custody-manifest", str(inputs[2]),
                    "--expected-custody-sha256", hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
                    "--media-dir", str(media_dir), "--review-page", str(review_page),
                    "--claim-input-manifest", str(bundle_paths[0]),
                    "--claim-input-sidecar", str(bundle_paths[1]),
                    "--output-dir", str(output), "--user-authorized-local-only"]
            stdout = io.StringIO()
            with mock.patch.object(package_cli, "__file__", str(fake_file)), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(package_cli.main(argv), 0)
            expected = {"index.html"} | {
                f"{row['review_token']}.mp4" for row in json.loads(inputs[0].read_text())["records"]}
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir()))
            self.assertIn('"execution_authorization": "not_granted"', stdout.getvalue())

    def test_cli_rejects_html_not_bound_to_frozen_inputs_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            private, media_dir, inputs, review_page, _ = self.cli_fixture(root)
            review_page.write_text("tampered page")
            os.chmod(review_page, 0o600)
            output = private / "handoff"
            fake_file = root / "scripts" / "package_nextgqa_claim_review_handoff.py"
            fake_file.parent.mkdir(parents=True)
            argv = ["--worksheet", str(inputs[0]), "--lineage-manifest", str(inputs[1]),
                    "--expected-lineage-sha256", hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
                    "--custody-manifest", str(inputs[2]),
                    "--expected-custody-sha256", hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
                    "--media-dir", str(media_dir), "--review-page", str(review_page),
                    "--output-dir", str(output), "--user-authorized-local-only"]
            with mock.patch.object(package_cli, "__file__", str(fake_file)), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                package_cli.main(argv)
            self.assertFalse(output.exists())

    def test_cli_requires_claim_manifest_and_sidecar_as_a_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            private, media_dir, inputs, review_page, _ = self.cli_fixture(root)
            output = private / "handoff"
            fake_file = root / "scripts" / "package_nextgqa_claim_review_handoff.py"
            fake_file.parent.mkdir(parents=True)
            argv = ["--worksheet", str(inputs[0]), "--lineage-manifest", str(inputs[1]),
                    "--expected-lineage-sha256", hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
                    "--custody-manifest", str(inputs[2]),
                    "--expected-custody-sha256", hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
                    "--media-dir", str(media_dir), "--review-page", str(review_page),
                    "--claim-input-manifest", str(inputs[0]),
                    "--output-dir", str(output), "--user-authorized-local-only"]
            with mock.patch.object(package_cli, "__file__", str(fake_file)), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                package_cli.main(argv)
            self.assertFalse(output.exists())

    def test_cli_rejects_stale_page_after_valid_bundle_digest_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            private, media_dir, inputs, review_page, bundle_paths = self.cli_fixture(
                root, include_bundle=True)
            manifest = json.loads(bundle_paths[0].read_bytes())
            sidecar = json.loads(bundle_paths[1].read_bytes())
            manifest["eligible_dataset_id"] += "-changed"
            manifest_raw = encode(manifest)
            sidecar["eligible_dataset_id"] = manifest["eligible_dataset_id"]
            sidecar["claim_input_manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            sidecar_raw = encode(sidecar)
            bundle_paths[0].write_bytes(manifest_raw)
            bundle_paths[1].write_bytes(sidecar_raw)
            os.chmod(bundle_paths[0], 0o600)
            os.chmod(bundle_paths[1], 0o600)
            output = private / "handoff"
            fake_file = root / "scripts" / "package_nextgqa_claim_review_handoff.py"
            fake_file.parent.mkdir(parents=True)
            argv = ["--worksheet", str(inputs[0]), "--lineage-manifest", str(inputs[1]),
                    "--expected-lineage-sha256", hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
                    "--custody-manifest", str(inputs[2]),
                    "--expected-custody-sha256", hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
                    "--media-dir", str(media_dir), "--review-page", str(review_page),
                    "--claim-input-manifest", str(bundle_paths[0]),
                    "--claim-input-sidecar", str(bundle_paths[1]),
                    "--output-dir", str(output), "--user-authorized-local-only"]
            stderr = io.StringIO()
            with mock.patch.object(package_cli, "__file__", str(fake_file)), \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                package_cli.main(argv)
            self.assertIn("review_page_not_bound_to_frozen_inputs", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_package_contains_only_blinded_page_and_verified_token_media(self):
        worksheet, _ = frozen_worksheet()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "private"; private.mkdir(mode=0o700); os.chmod(private, 0o700)
            media_dir = private / "media"; media_dir.mkdir(mode=0o700)
            custody_rows = []
            for ordinal, row in enumerate(worksheet["records"], 1):
                data = f"private media {ordinal}".encode(); digest = hashlib.sha256(data).hexdigest()
                row["video_sha256"] = digest
                salt = bytes([ordinal]) * 32
                path = media_dir / (row["review_token"] + ".mp4"); path.write_bytes(data); os.chmod(path, 0o600)
                custody_rows.append({"review_token": row["review_token"], "media_check_salt": salt.hex(),
                                     "media_check_sha256": hashlib.sha256(salt + data).hexdigest()})
            custody = {"records": custody_rows}
            html = private / "review.html"; html.write_text("<!doctype html><p>blinded review</p>"); os.chmod(html, 0o600)
            output = private / "handoff"
            result = _package_review_after_audit(html, worksheet, custody, media_dir, output, private)
            expected_names = {"index.html"} | {r["review_token"] + ".mp4" for r in worksheet["records"]}
            self.assertEqual({p.name for p in output.iterdir()}, expected_names)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir()))
            self.assertFalse(result["contains_custody_manifest"])

    def test_media_mismatch_fails_and_removes_partial_package(self):
        worksheet, _ = frozen_worksheet()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "private"; private.mkdir(mode=0o700); os.chmod(private, 0o700)
            media_dir = private / "media"; media_dir.mkdir(mode=0o700)
            rows=[]
            for i,row in enumerate(worksheet["records"]):
                raw=b"not matching worksheet hash"; p=media_dir/(row["review_token"]+".mp4");p.write_bytes(raw);os.chmod(p,0o600)
                rows.append({"review_token":row["review_token"],"media_check_salt":"00"*32,
                             "media_check_sha256":hashlib.sha256(bytes(32)+raw).hexdigest()})
            html=private/"review.html";html.write_text("offline");os.chmod(html,0o600)
            output=private/"handoff"
            with self.assertRaisesRegex(ValueError,"review_media_crosscheck_failed"):
                _package_review_after_audit(html,worksheet,{"records":rows},media_dir,output,private)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
