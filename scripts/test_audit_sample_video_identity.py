"""CPU-only privacy and accounting tests for the sample video identity audit."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_sample_video_identity import audit_manifest_bytes, _write_exclusive  # noqa: E402


def raw_manifest(samples):
    return json.dumps(samples, separators=(",", ":")).encode("utf-8")


def expected_hash(raw):
    return hashlib.sha256(raw).hexdigest()


class AuditSampleVideoIdentityTests(unittest.TestCase):
    def test_distinguishes_duplicate_ids_paths_and_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "private-a.mp4"
            path_b = root / "private-b.mp4"
            path_a.write_bytes(b"clip")
            path_b.write_bytes(b"clip")
            samples = [
                {"video_id": " alpha ", "video": str(path_a), "question": "private"},
                {"video_id": "alpha", "video": str(path_a)},
                {"video_id": "beta", "video": str(path_b)},
            ]
            raw = raw_manifest(samples)
            report = audit_manifest_bytes(raw, expected_hash(raw), base_dir=root)

        self.assertEqual(report["requested_sample_records"], 3)
        self.assertEqual(report["effective_sample_records_with_hashed_video"], 3)
        self.assertEqual(report["unique_nonempty_normalized_video_ids"], 2)
        self.assertEqual(report["duplicate_video_id_excess_records"], 1)
        self.assertEqual(report["effective_unique_nonempty_normalized_video_ids"], 2)
        self.assertEqual(report["effective_duplicate_video_id_excess_records"], 1)
        self.assertEqual(report["unique_resolved_video_paths"], 2)
        self.assertEqual(report["duplicate_resolved_path_excess_records"], 1)
        self.assertEqual(report["files_present"], 2)
        self.assertEqual(report["files_hashed_successfully"], 2)
        self.assertEqual(report["duplicate_content_excess_files"], 1)
        self.assertEqual(report["total_bytes_hashed"], 8)
        serialized = json.dumps(report)
        for private_value in ("alpha", "beta", "private-a.mp4", "private-b.mp4", "private"):
            self.assertNotIn(private_value, serialized)

    def test_wrong_hash_fails_before_video_path_resolution_or_reads(self):
        raw = raw_manifest([{"video_id": "v", "video": "/never/read/this.mp4"}])
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("resolved path")), \
                mock.patch.object(Path, "is_file", side_effect=AssertionError("stat path")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                audit_manifest_bytes(raw, "0" * 64)

    def test_counts_missing_and_read_failed_files_without_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unreadable = root / "unreadable.mp4"
            unreadable.write_bytes(b"present but denied")
            missing = root / "missing.mp4"
            raw = raw_manifest([
                {"video_id": "missing", "video": str(missing)},
                {"video_id": "denied", "video": str(unreadable)},
            ])

            def opener(path, mode):
                if Path(path) == unreadable:
                    raise PermissionError("private filesystem detail")
                return open(path, mode)

            report = audit_manifest_bytes(
                raw, expected_hash(raw), base_dir=root, file_opener=opener)

        self.assertEqual(report["files_present"], 1)
        self.assertEqual(report["files_hashed_successfully"], 0)
        self.assertEqual(report["effective_unique_nonempty_normalized_video_ids"], 0)
        self.assertEqual(report["missing_files"], 1)
        self.assertEqual(report["read_failed_files"], 1)
        self.assertEqual(report["total_bytes_hashed"], 0)
        self.assertFalse(report['content_comparison_complete'])
        self.assertNotIn("missing.mp4", json.dumps(report))
        self.assertNotIn("unreadable.mp4", json.dumps(report))

    def test_rejects_empty_and_boolean_video_ids(self):
        for video_id in ("", "   ", True, False, None):
            with self.subTest(video_id=video_id):
                raw = raw_manifest([{"video_id": video_id, "video": "/ignored"}])
                with self.assertRaisesRegex(ValueError, "video ID"):
                    audit_manifest_bytes(raw, expected_hash(raw))

    def test_requested_prefix_is_bounded_and_reported(self):
        raw = raw_manifest([
            {"video_id": "first", "video": "missing-first"},
            {"video_id": "second", "video": "missing-second"},
        ])
        report = audit_manifest_bytes(raw, expected_hash(raw), n_requested=1)
        self.assertEqual(report["requested_sample_records"], 1)
        self.assertEqual(report["unique_nonempty_normalized_video_ids"], 1)
        with self.assertRaisesRegex(ValueError, "exceeds sample manifest"):
            audit_manifest_bytes(raw, expected_hash(raw), n_requested=3)

    def test_output_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "results" / "audit.json"
            _write_exclusive(output, {"count": 1}, repo_root=repo)
            with self.assertRaises(FileExistsError):
                _write_exclusive(output, {"count": 2}, repo_root=repo)
            self.assertEqual(json.loads(output.read_text()), {"count": 1})
            with self.assertRaises(ValueError):
                _write_exclusive(repo / 'other.json',{},repo_root=repo)

    def test_symlink_alias_is_counted_as_same_resolved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'original.mp4';video.write_bytes(b'fixture')
            alias = root / 'alias.mp4';alias.symlink_to(video)
            raw = raw_manifest([{'video_id':'first','video':str(video)},
                                {'video_id':'second','video':str(alias)}])
            report = audit_manifest_bytes(raw,expected_hash(raw))
            self.assertEqual(report['unique_resolved_video_paths'],1)
            self.assertEqual(report['duplicate_resolved_path_excess_records'],1)
            self.assertEqual(report['total_bytes_hashed'],7)


if __name__ == "__main__":
    unittest.main()
