from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import unittest

from audit_nextgqa_local_media_byte_diversity import audit_local_media
from audit_nextgqa_local_media_byte_diversity import _read_stable_fd


class NextgqaLocalMediaByteDiversityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        self.media = self.base / "media"
        self.media.mkdir()
        with (self.data / "test.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid", "question", "answer"])
            writer.writeheader()
            writer.writerows([
                {"video_id": "v1", "qid": "q1", "question": "PRIVATE QUESTION", "answer": "PRIVATE ANSWER"},
                {"video_id": "v2", "qid": "q2", "question": "PRIVATE QUESTION 2", "answer": "PRIVATE ANSWER 2"},
                {"video_id": "v3", "qid": "q3", "question": "PRIVATE QUESTION 3", "answer": "PRIVATE ANSWER 3"},
            ])
        (self.data / "map.json").write_text(json.dumps({
            "v1": "grp/a", "v2": "grp/b", "v3": "grp/c"}))
        group = self.media / "grp"
        group.mkdir()
        (group / "a.mp4").write_bytes(b"same-media")
        (group / "b.mp4").write_bytes(b"same-media")
        (group / "c.mp4").write_bytes(b"different-media")

    def _run(self, **kwargs):
        return audit_local_media(test_csv=self.data / "test.csv",
                                 video_map_json=self.data / "map.json",
                                 video_root=self.media, **kwargs)

    def test_exact_duplicates_counted_without_emitting_inputs(self):
        report = self._run()
        self.assertEqual(report["dataset_video_id_count"], 3)
        self.assertEqual(report["mapped_video_files_found"], 3)
        self.assertEqual(report["unique_media_byte_hashes"], 2)
        self.assertEqual(report["exact_duplicate_extra_files"], 1)
        self.assertEqual(report["exact_duplicate_hash_groups"], 1)
        self.assertFalse(report["source_cluster_independence_verified"])
        rendered = json.dumps(report)
        for secret in ("v1", "grp/a", "PRIVATE QUESTION", "PRIVATE ANSWER",
                       hashlib.sha256(b"same-media").hexdigest()):
            self.assertNotIn(secret, rendered)

    def test_private_partial_hmac_index_match_is_counted(self):
        key = b"k" * 32
        media_sha = hashlib.sha256(b"same-media").hexdigest()
        token = hmac.new(key, json.dumps(
            ["nextgqa-exposure-v1", "media-sha256", media_sha],
            ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            hashlib.sha256).hexdigest()
        index = self.data / "prior.json"
        index.write_text(json.dumps({
            "schema_version": True,
            "purpose": "outcome_blind_prior_exposure_index",
            "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
            "identities": {"media_sha256_hmac_sha256": [token]},
        }))
        index.chmod(0o600)
        secret = self.data / "key.bin"
        secret.write_bytes(key)
        secret.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "prior_index_schema_invalid"):
            self._run(prior_index=index, hmac_key=secret)
        index.write_text(json.dumps({
            "schema_version": 1,
            "purpose": "outcome_blind_prior_exposure_index",
            "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
            "identities": {"media_sha256_hmac_sha256": [token]},
        }))
        index.chmod(0o600)
        report = self._run(prior_index=index, hmac_key=secret)
        self.assertEqual(report["current_files_matching_prior_media_tokens"], 2)
        self.assertEqual(report["distinct_current_hashes_matching_prior_media_tokens"], 1)
        self.assertEqual(report["prior_exposure_match_status"], "partial_hmac_index_compared")
        self.assertFalse(report["prior_history_completeness_authenticated"])

    def test_stable_fd_reader_enforces_private_mode_and_byte_limit(self):
        private = self.data / "private.bin"
        private.write_bytes(b"safe")
        private.chmod(0o600)
        fd = os.open(private, os.O_RDONLY)
        try:
            self.assertEqual(_read_stable_fd(fd, code="bad", max_bytes=4, private=True), b"safe")
        finally:
            os.close(fd)
        with self.assertRaisesRegex(ValueError, "byte_limit"):
            fd = os.open(private, os.O_RDONLY)
            try:
                _read_stable_fd(fd, code="byte_limit", max_bytes=3, private=True)
            finally:
                os.close(fd)
        public = self.data / "public.bin"
        public.write_bytes(b"safe")
        public.chmod(0o644)
        fd = os.open(public, os.O_RDONLY)
        try:
            with self.assertRaisesRegex(ValueError, "private_mode"):
                _read_stable_fd(fd, code="private_mode", max_bytes=4, private=True)
        finally:
            os.close(fd)

    def test_incomplete_key_arguments_and_unsafe_paths_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "prior_index_and_hmac_key_must_be_paired"):
            self._run(prior_index=self.data / "prior.json")
        (self.data / "map.json").write_text(json.dumps({
            "v1": "../outside", "v2": "grp/b", "v3": "grp/c"}))
        with self.assertRaisesRegex(ValueError, "video_map_path_invalid"):
            self._run()

        (self.data / "map.json").write_text(json.dumps({
            "v1": "grp/./a", "v2": "grp/b", "v3": "grp/c"}))
        with self.assertRaisesRegex(ValueError, "video_map_path_invalid"):
            self._run()

    def test_intermediate_symlink_cannot_escape_media_root(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "a.mp4").write_bytes(b"outside")
        (self.media / "linked").symlink_to(outside, target_is_directory=True)
        (self.data / "map.json").write_text(json.dumps({
            "v1": "linked/a", "v2": "grp/b", "v3": "grp/c"}))
        with self.assertRaisesRegex(ValueError, "media_file_path_invalid"):
            self._run()

    def test_missing_files_are_reported_without_replacing_denominator(self):
        (self.media / "grp" / "a.mp4").unlink()
        report = self._run()
        self.assertEqual(report["dataset_video_id_count"], 3)
        self.assertEqual(report["mapped_video_files_found"], 2)
        self.assertEqual(report["mapped_video_files_missing"], 1)

    def test_even_count_median_uses_both_center_values(self):
        (self.media / "grp" / "b.mp4").unlink()
        report = self._run()
        self.assertEqual(report["mapped_video_files_found"], 2)
        self.assertEqual(report["median_file_bytes"], 12.5)


if __name__ == "__main__":
    unittest.main()
