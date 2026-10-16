import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from PIL import Image

from build_blinded_frame_pair_review import ANCHOR_IDS, build_review, audit_export


class BlindedFramePairReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.inputs = self.root / "inputs"
        self.left = self.inputs / "crop"
        self.right = self.inputs / "decoder"
        self.left.mkdir(parents=True, mode=0o700)
        os.chmod(self.inputs, 0o700)
        self.right.mkdir(mode=0o700)
        self.manifest_path = self.inputs / "pairs.json"
        self.plan_path = self.inputs / "plan.json"
        self.records = []
        for i in range(15):
            token = f"tok_{i // 5:020d}"[:24]
            anchor = ANCHOR_IDS[i % 5]
            crop_name, frame_name = f"crop_{i}.png", f"frame_{i}.png"
            crop_raw = self.png((i * 10, 20, 30), metadata=True)
            frame_raw = self.png((i * 10, 20, 30))
            (self.left / crop_name).write_bytes(crop_raw)
            (self.right / frame_name).write_bytes(frame_raw)
            os.chmod(self.left / crop_name, 0o600)
            os.chmod(self.right / frame_name, 0o600)
            self.records.append({
                "review_token": token, "anchor_id": anchor,
                "screenshot_crop_file": crop_name,
                "screenshot_crop_sha256": hashlib.sha256(crop_raw).hexdigest(),
                "decoder_frame_file": frame_name,
                "decoder_frame_sha256": hashlib.sha256(frame_raw).hexdigest(),
            })
        self.write_plan()
        self.write_manifest(self.records)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def png(color, metadata=False):
        image = Image.new("RGB", (12, 9), color)
        stream = io.BytesIO()
        if metadata:
            from PIL.PngImagePlugin import PngInfo
            info = PngInfo()
            info.add_text("Private-Comment", "must be stripped")
            image.save(stream, format="PNG", pnginfo=info)
        else:
            image.save(stream, format="PNG")
        return stream.getvalue()

    def write_manifest(self, records):
        raw = json.dumps({"schema_version": 1,
                          "purpose": "private_blinded_frame_pair_review_input",
                          "plan_sha256": self.plan_sha256,
                          "records": records}).encode()
        self.manifest_path.write_bytes(raw)
        os.chmod(self.manifest_path, 0o600)

    def write_plan(self):
        records = []
        for i in range(3):
            token = f"tok_{i:020d}"[:24]
            records.append({"review_token": token, "source_media_sha256": "f" * 64,
                            "fit_anchors": [{"anchor_id": x, "decoder_pts_seconds": str(j)}
                                            for j, x in enumerate(("fit_begin", "fit_midpoint", "fit_end"))],
                            "heldout_anchors": [{"anchor_id": x, "decoder_pts_seconds": str(j + 3)}
                                                for j, x in enumerate(("heldout_quarter", "heldout_three_quarter"))]})
        raw = json.dumps({"schema_version": 1,
                          "purpose": "predeclared_html5_clock_pts_anchor_plan",
                          "status": "predeclared_frozen", "execution_authorization": "not_granted",
                          "records": records}).encode()
        self.plan_path.write_bytes(raw)
        os.chmod(self.plan_path, 0o600)
        self.plan_sha256 = hashlib.sha256(raw).hexdigest()

    def build(self, name="bundle"):
        return build_review(self.manifest_path, self.left, self.right,
                            self.root / (name + "_viewer"),
                            self.root / (name + "_identity"), self.plan_path, self.plan_sha256)

    def test_builds_separate_blinded_and_identity_artifacts(self):
        result = self.build()
        viewer = self.root / "bundle_viewer"
        identity_dir = self.root / "bundle_identity"
        html = (viewer / "index.html").read_text()
        package_raw = (viewer / "package_manifest.json").read_bytes()
        identity_raw = (identity_dir / "identity_map.json").read_bytes()
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(stat.S_IMODE(viewer.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(identity_dir.stat().st_mode), 0o700)
        self.assertEqual({p.name for p in viewer.iterdir()}, {"index.html", "package_manifest.json"})
        for row in self.records:
            self.assertNotIn(row["review_token"], html)
            self.assertNotIn(row["anchor_id"], html)
            self.assertNotIn(row["review_token"].encode(), package_raw)
        self.assertNotIn(b"screenshot_crop_file", identity_raw)
        identity = json.loads(identity_raw)
        self.assertEqual(len(identity["records"]), 15)
        self.assertEqual({x["pair_id"] for x in identity["records"]},
                         {x["pair_id"] for x in json.loads(package_raw)["pairs"]})
        self.assertTrue(all(x["A_source"] in {"screenshot_crop", "decoder_frame"}
                            and x["B_source"] in {"screenshot_crop", "decoder_frame"}
                            and x["A_source"] != x["B_source"] for x in identity["records"]))
        self.assertIn("not_authorizing", html)
        # Re-encoded PNGs contain only RGB pixels; source text metadata is absent.
        self.assertNotIn("must be stripped", html)
        self.assertEqual(stat.S_IMODE((viewer / "index.html").stat().st_mode), 0o600)

    def test_export_pass_and_blocking_outcomes(self):
        self.build()
        viewer = self.root / "bundle_viewer"
        package_raw = (viewer / "package_manifest.json").read_bytes()
        package = json.loads(package_raw)
        records = [{"pair_id": row["pair_id"], "crop_purity_attestation": True,
                    "decision": "match"} for row in package["pairs"]]
        export = {"schema_version": 1,
                  "purpose": "private_blinded_frame_pair_review_export",
                  "status": "not_authorizing", "execution_authorization": "not_granted",
                  "package_manifest_sha256": hashlib.sha256(package_raw).hexdigest(),
                  "package_nonce": package["package_nonce"],
                  "reviewer_pseudonym": "reviewer-1",
                  "reviewer_role": "independent_frame_pair_reviewer",
                  "independence_attestation": "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring",
                  "reviewer_provenance_status": "self_reported_unauthenticated",
                  "signature_status": "none", "reviewed_at_utc": "2026-10-17T00:00:00.000Z",
                  "records": records}
        export_path = self.root / "export.json"
        export_path.write_text(json.dumps(export))
        os.chmod(export_path, 0o600)
        result = audit_export(export_path, viewer / "package_manifest.json", viewer / "index.html")
        self.assertEqual(result["status"], "candidate_review_complete_not_authorizing")
        self.assertEqual(result["match_count"], 15)
        export["records"][0]["decision"] = "ambiguous"
        export_path.write_text(json.dumps(export))
        os.chmod(export_path, 0o600)
        self.assertEqual(audit_export(export_path, viewer / "package_manifest.json", viewer / "index.html")["status"],
                         "blocked_review_not_all_pure_and_matching")
        export["records"] = export["records"][:-1]
        export_path.write_text(json.dumps(export))
        os.chmod(export_path, 0o600)
        with self.assertRaisesRegex(ValueError, "review_export_schema_or_package_binding_invalid"):
            audit_export(export_path, viewer / "package_manifest.json", viewer / "index.html")

    def test_rejects_wrong_count_duplicate_and_hash_mismatch(self):
        self.write_manifest(self.records[:-1])
        with self.assertRaisesRegex(ValueError, "manifest_schema_or_count_invalid"):
            self.build("count")
        duplicate = list(self.records)
        duplicate[-1] = dict(duplicate[0])
        self.write_manifest(duplicate)
        with self.assertRaisesRegex(ValueError, "manifest_pair_identity_invalid_or_duplicate"):
            self.build("duplicate")
        self.write_manifest(self.records)
        self.records[0]["screenshot_crop_sha256"] = "0" * 64
        self.write_manifest(self.records)
        with self.assertRaisesRegex(ValueError, "screenshot_crop_hash_mismatch"):
            self.build("hash")

    def test_requires_exact_anchor_grid_and_pinned_plan_hash(self):
        altered = list(self.records)
        altered[0] = dict(altered[0], anchor_id="other_anchor")
        self.write_manifest(altered)
        with self.assertRaisesRegex(ValueError, "manifest_token_anchor_grid_invalid"):
            self.build("anchor-grid")
        self.write_manifest(self.records)
        with self.assertRaisesRegex(ValueError, "plan_sha256_mismatch"):
            build_review(self.manifest_path, self.left, self.right,
                         self.root / "wrongplan_viewer", self.root / "wrongplan_identity",
                         self.plan_path, "b" * 64)

    def test_rejects_swapped_token_anchor_assignments_from_plan(self):
        swapped = [dict(row) for row in self.records]
        # Swap token identities across different anchors. This changes the
        # token/anchor bindings while retaining exactly 15 records.
        first = next(i for i, row in enumerate(swapped)
                     if row["review_token"] == "tok_00000000000000000000"
                     and row["anchor_id"] == "fit_begin")
        second = next(i for i, row in enumerate(swapped)
                      if row["review_token"] == "tok_00000000000000000001"
                      and row["anchor_id"] == "heldout_quarter")
        swapped[first]["review_token"], swapped[second]["review_token"] = (
            swapped[second]["review_token"], swapped[first]["review_token"])
        self.write_manifest(swapped)
        with self.assertRaisesRegex(ValueError,
                                    "manifest_pair_identity_invalid_or_duplicate|manifest_token_anchor_grid_invalid|manifest_plan_assignment_mismatch"):
            self.build("swapped")

    def test_audit_rejects_tampered_html_and_embedded_image_payload(self):
        self.build()
        viewer = self.root / "bundle_viewer"
        package_raw = (viewer / "package_manifest.json").read_bytes()
        package = json.loads(package_raw)
        export = {"schema_version": 1,
                  "purpose": "private_blinded_frame_pair_review_export",
                  "status": "not_authorizing", "execution_authorization": "not_granted",
                  "package_manifest_sha256": hashlib.sha256(package_raw).hexdigest(),
                  "package_nonce": package["package_nonce"], "reviewer_pseudonym": "reviewer-1",
                  "reviewer_role": "independent_frame_pair_reviewer",
                  "independence_attestation": "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring",
                  "reviewer_provenance_status": "self_reported_unauthenticated", "signature_status": "none",
                  "reviewed_at_utc": "2026-10-17T00:00:00.000Z",
                  "records": [{"pair_id": row["pair_id"], "crop_purity_attestation": True,
                               "decision": "match"} for row in package["pairs"]]}
        export_path = self.root / "tamper_export.json"
        export_path.write_text(json.dumps(export))
        os.chmod(export_path, 0o600)
        html_path = viewer / "index.html"
        original = html_path.read_bytes()
        html_path.write_bytes(original.replace(b"Blinded frame-pair review", b"Changed review page", 1))
        os.chmod(html_path, 0o600)
        with self.assertRaisesRegex(ValueError, "reviewer_html_hash_mismatch"):
            audit_export(export_path, viewer / "package_manifest.json", html_path)
        html_path.write_bytes(original)
        os.chmod(html_path, 0o600)
        mutated = original.replace(b"data:image/png;base64,", b"data:image/png;base64,AA", 1)
        html_path.write_bytes(mutated)
        os.chmod(html_path, 0o600)
        with self.assertRaisesRegex(ValueError, "reviewer_html_hash_mismatch"):
            audit_export(export_path, viewer / "package_manifest.json", html_path)

    def test_export_rejects_boolean_schema_version(self):
        self.build()
        viewer = self.root / "bundle_viewer"
        package_raw = (viewer / "package_manifest.json").read_bytes()
        package = json.loads(package_raw)
        export_path = self.root / "bool_export.json"
        export = {"schema_version": True, "purpose": "private_blinded_frame_pair_review_export",
                  "status": "not_authorizing", "execution_authorization": "not_granted",
                  "package_manifest_sha256": hashlib.sha256(package_raw).hexdigest(),
                  "package_nonce": package["package_nonce"], "reviewer_pseudonym": "reviewer-1",
                  "reviewer_role": "independent_frame_pair_reviewer",
                  "independence_attestation": "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring",
                  "reviewer_provenance_status": "self_reported_unauthenticated", "signature_status": "none",
                  "reviewed_at_utc": "2026-10-17T00:00:00.000Z", "records": []}
        export_path.write_text(json.dumps(export))
        os.chmod(export_path, 0o600)
        with self.assertRaisesRegex(ValueError, "review_export_schema_or_package_binding_invalid"):
            audit_export(export_path, viewer / "package_manifest.json", viewer / "index.html")

    def test_rejects_symlink_and_unsafe_filename(self):
        symlink = self.left / "crop_0.png"
        target = self.left / "real.png"
        target.write_bytes(symlink.read_bytes())
        os.chmod(target, 0o600)
        symlink.unlink()
        symlink.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "screenshot_crop_file_invalid"):
            self.build("symlink")
        symlink.unlink()
        symlink.write_bytes(target.read_bytes())
        os.chmod(symlink, 0o600)
        self.records[0]["screenshot_crop_file"] = "../outside.png"
        self.write_manifest(self.records)
        with self.assertRaisesRegex(ValueError, "manifest_image_path_invalid"):
            self.build("path")

    def test_rejects_insecure_directories_and_invalid_image(self):
        os.chmod(self.left, 0o755)
        with self.assertRaisesRegex(ValueError, "screenshot_directory_not_private"):
            self.build("mode")
        os.chmod(self.left, 0o700)
        bad = self.left / "crop_0.png"
        bad.write_bytes(b"not png")
        os.chmod(bad, 0o600)
        self.records[0]["screenshot_crop_sha256"] = hashlib.sha256(b"not png").hexdigest()
        self.write_manifest(self.records)
        with self.assertRaisesRegex(ValueError, "screenshot_crop_image_invalid"):
            self.build("invalid")

    def test_rejects_oversized_dimensions(self):
        oversized = self.left / "oversized.png"
        image = Image.new("RGB", (10001, 9), (1, 2, 3))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        oversized.write_bytes(raw)
        os.chmod(oversized, 0o600)
        self.records[0]["screenshot_crop_file"] = "oversized.png"
        self.records[0]["screenshot_crop_sha256"] = hashlib.sha256(raw).hexdigest()
        self.write_manifest(self.records)
        with self.assertRaisesRegex(ValueError, "screenshot_crop_image_dimensions_exceeded"):
            self.build("oversized")


if __name__ == "__main__":
    unittest.main()
