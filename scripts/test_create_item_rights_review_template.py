import hashlib
import json
import os
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.audit_item_rights_manifest import audit
from scripts.create_item_rights_review_template import _read_manifest, build_template, write_private_bundle


def raw(value):
    return json.dumps(value).encode()


class ItemRightsTemplateTests(unittest.TestCase):
    def test_template_is_private_unresolved_and_fails_rights_gate(self):
        eligible = [{"video_id": "synthetic-a"}, {"video_id": "synthetic-b"}]
        eligible_raw = raw(eligible)
        ledger = build_template(["synthetic-a", "synthetic-b"], eligible_raw,
                                "synthetic eligible set", "synthetic local research", ["local_inference"])
        self.assertEqual(ledger["eligible_manifest_sha256"], hashlib.sha256(eligible_raw).hexdigest())
        self.assertTrue(all(row["decisions"]["local_inference"] == "unresolved" for row in ledger["items"]))
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertFalse(result["structurally_valid"])
        self.assertEqual(result["execution_authorization"], "not_granted")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private"
            eligible_bytes, ledger_bytes = write_private_bundle(path, eligible, ledger)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o700)
            for name in ("eligible_manifest.json", "rights_ledger_template.json"):
                self.assertEqual(os.stat(path / name).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(eligible_bytes), eligible)
            self.assertEqual(json.loads(ledger_bytes)["items"], ledger["items"])
            written_eligible = json.loads((path / "eligible_manifest.json").read_bytes())
            written_ledger = json.loads((path / "rights_ledger_template.json").read_bytes())
            self.assertEqual(written_ledger["eligible_manifest_sha256"],
                             hashlib.sha256((path / "eligible_manifest.json").read_bytes()).hexdigest())
            result = audit(written_eligible, (path / "eligible_manifest.json").read_bytes(),
                           written_ledger, (path / "rights_ledger_template.json").read_bytes())
            self.assertIn("item_source_lineage_missing", result["failure_codes"])
            self.assertFalse(result["structurally_valid"])
            with self.assertRaises(FileExistsError):
                write_private_bundle(path, eligible, ledger)

    def test_video_path_cohort_field_normalizes_only_to_filename_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cohort.json"
            path.write_bytes(b'[{"video":"/private/a/clip-a.mp4"}]')
            ids, _ = _read_manifest(path, "video")
            self.assertEqual(ids, ["clip-a"])
            path.write_bytes(b'[{"video":"folder/"}]')
            with self.assertRaises(ValueError):
                _read_manifest(path, "video")

    def test_cli_stdout_omits_synthetic_ids(self):
        from scripts.create_item_rights_review_template import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort_path = root / "cohort.json"
            cohort_path.write_text('[{"video":"private/secret-video.mp4"}]', encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([
                    "--cohort-manifest", str(cohort_path), "--video-field", "video",
                    "--eligible-dataset-id", "synthetic dataset",
                    "--intended-use-scope", "synthetic review scope",
                    "--intended-uses", "local_inference",
                    "--output-dir", str(root / "bundle"),
                ]), 0)
            self.assertNotIn("secret-video", output.getvalue())
            self.assertIn('"execution_authorization": "not_granted"', output.getvalue())

    def test_manifest_requires_unique_ids_and_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eligible.json"
            path.write_bytes(b'[{"video_id":"x"},{"video_id":"x"}]')
            with self.assertRaises(ValueError):
                _read_manifest(path)
            path.write_bytes(b'[{"video_id":"x","video_id":"y"}]')
            with self.assertRaises(ValueError):
                _read_manifest(path)

    def test_template_requires_scope_and_restricted_use_choices(self):
        with self.assertRaises(ValueError):
            build_template(["x"], b"[]", "", "scope", ["local_inference"])
        with self.assertRaises(ValueError):
            build_template(["x"], b"[]", "dataset", "scope", ["redistribution"])
        with self.assertRaises(ValueError):
            build_template(["x"], b"[]", "dataset", "scope", ["local_inference", "unknown"])


if __name__ == "__main__":
    unittest.main()
