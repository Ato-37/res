import hashlib
import json
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from scripts.audit_nextgqa_video_mapping import _decode, audit


def raw(value):
    return json.dumps(value).encode()


class NextGQAVideoMappingAuditTests(unittest.TestCase):
    def test_filename_stem_join_is_complete_and_summary_has_no_ids(self):
        rows = [{"video": "/private/path/clip-a.mp4"}, {"video": "clip-b.avi"}]
        mapping = {"clip-a": "vidor-1", "clip-b": "vidor-2"}
        result = audit(rows, raw(rows), mapping, raw(mapping))
        self.assertEqual(result["status"], "mapped")
        self.assertEqual((result["requested_rows"], result["mapped_rows"]), (2, 2))
        self.assertEqual(result["rights_or_inference_authorization"], "not_granted")
        serialized = json.dumps(result)
        for private_value in ("clip-a", "clip-b", "vidor-1", "/private/path"):
            self.assertNotIn(private_value, serialized)

    def test_missing_stem_and_duplicate_target_are_reported(self):
        rows = [{"video": "clip-a.mp4"}, {"video": "missing.mp4"}]
        mapping = {"clip-a": "same-target", "missing": "same-target"}
        result = audit(rows, raw(rows), mapping, raw(mapping))
        self.assertEqual(result["status"], "incomplete_or_invalid")
        self.assertIn("mapped_video_paths_duplicate", result["failure_codes"])

        result = audit(rows, raw(rows), {"clip-a": "only-a"}, raw({"clip-a": "only-a"}))
        self.assertEqual(result["unmapped_rows"], 1)

        rows = [{"video": "same.mp4"}, {"video": "same.mov"}]
        result = audit(rows, raw(rows), {"same": "one-target"}, raw({"same": "one-target"}))
        self.assertIn("sample_video_stems_duplicate", result["failure_codes"])

    def test_invalid_rows_or_map_fail_closed(self):
        result = audit([{"video": None}], b"[]", {"x": "y"}, b"{}")
        self.assertIn("sample_video_field_invalid", result["failure_codes"])
        result = audit([{"video": "x.mp4"}], b"[]", {"x": 42}, b"{}")
        self.assertIn("video_mapping_invalid", result["failure_codes"])

        result = audit([{"video": "folder/"}], b"[]", {}, b"{}")
        self.assertIn("sample_video_stem_invalid", result["failure_codes"])

        result = audit([{"video": "x.mp4"}], b"[]", {"x": "   "}, b"{}")
        self.assertIn("video_mapping_invalid", result["failure_codes"])

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            _decode(b'{"clip":"a","clip":"b"}')
        with self.assertRaises(ValueError):
            _decode(b'[{"video":"a.mp4","video":"b.mp4"}]')

    def test_path_dot_segments_and_separator_variants_are_normalized(self):
        rows = [{"video": "a.mp4"}, {"video": "b.mp4"}]
        mapping = {"a": "folder/../same.mp4", "b": "same.mp4"}
        result = audit(rows, raw(rows), mapping, raw(mapping))
        self.assertIn("mapped_video_paths_duplicate", result["failure_codes"])

    def test_manifest_hash_is_bound_to_exact_input_bytes(self):
        rows, mapping = [{"video": "a.mp4"}], {"a": "v"}
        result = audit(rows, b'{ "video": "a.mp4" }', mapping, raw(mapping))
        self.assertEqual(result["sample_manifest_sha256"], hashlib.sha256(b'{ "video": "a.mp4" }').hexdigest())

    def test_local_media_path_matches_mapped_relative_path_without_emitting_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NExTVideo"
            media = root / "1014" / "clip-a.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"synthetic video bytes")
            rows = [{"video": str(media)}]
            mapping = {"clip-a": "1014/clip-a"}
            result = audit(rows, raw(rows), mapping, raw(mapping), root)
            self.assertEqual(result["local_path_mapping_audit"]["status"], "matched")
            self.assertEqual(result["local_path_mapping_audit"]["mapping_relative_path_matches"], 1)
            serialized = json.dumps(result)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("clip-a", serialized)

    def test_local_media_path_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NExTVideo"
            media = root / "1014" / "clip-a.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"synthetic")
            rows = [{"video": str(media)}]
            mapping = {"clip-a": "other-folder/clip-a"}
            result = audit(rows, raw(rows), mapping, raw(mapping), root)
            self.assertEqual(result["status"], "incomplete_or_invalid")
            self.assertIn("local_video_path_mapping_mismatch", result["failure_codes"])

    def test_local_media_audit_rejects_windows_absolute_mapping_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NExTVideo"
            media = root / "C:" / "folder" / "clip-a.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"synthetic")
            rows = [{"video": str(media)}]
            mapping = {"clip-a": "C:\\folder\\clip-a"}
            result = audit(rows, raw(rows), mapping, raw(mapping), root)
            self.assertEqual(result["local_path_mapping_audit"]["status"], "incomplete_or_invalid")
            self.assertIn("local_mapping_path_invalid", result["failure_codes"])

    def test_local_media_audit_rejects_windows_unc_mapping_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NExTVideo"
            media = root / "clip-a.mp4"
            root.mkdir()
            media.write_bytes(b"synthetic")
            rows = [{"video": str(media)}]
            mapping = {"clip-a": r"\\server\share\clip-a"}
            result = audit(rows, raw(rows), mapping, raw(mapping), root)
            self.assertEqual(result["local_path_mapping_audit"]["status"], "incomplete_or_invalid")
            self.assertIn("local_mapping_path_invalid", result["failure_codes"])

    def test_local_path_substatus_fails_when_any_sample_row_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NExTVideo"
            media = root / "clip-a.mp4"
            root.mkdir()
            media.write_bytes(b"synthetic")
            rows = [{"video": str(media)}, {"video": "bad/"}]
            mapping = {"clip-a": "clip-a"}
            result = audit(rows, raw(rows), mapping, raw(mapping), root)
            self.assertEqual(result["local_path_mapping_audit"]["status"], "incomplete_or_invalid")
            self.assertEqual(result["local_path_mapping_audit"]["unresolved_rows"], 1)

    def test_cli_summary_artifact_is_private_and_does_not_emit_paths(self):
        from scripts.audit_nextgqa_video_mapping import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "samples.json"
            mapping_path = root / "mapping.json"
            output = root / "summary.json"
            sample.write_text('[{"video":"/private/root/clip.mp4"}]', encoding="utf-8")
            mapping_path.write_text('{"clip":"folder/clip"}', encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--sample-manifest", str(sample), "--video-map", str(mapping_path),
                                       "--output", str(output)]), 0)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertNotIn("clip", stdout.getvalue())
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(["--sample-manifest", str(sample), "--video-map", str(mapping_path),
                      "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
