import hashlib
import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
