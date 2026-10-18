import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_nextgqa_prior_exposure_manifest import build
from scripts.reconcile_nextgqa_prior_exposure_pool import main, reconcile


KEY = b"synthetic test key with enough entropy 0123456789abcdef"


class PriorExposurePoolReconciliationTests(unittest.TestCase):
    def test_exact_byte_match_is_aggregate_only_and_unmatched_is_not_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir(mode=0o700)
            matched_bytes = b"synthetic video A"
            duplicate_bytes = matched_bytes
            unmatched_bytes = b"synthetic video B"
            (root / "a.mp4").write_bytes(matched_bytes)
            (root / "b.mp4").write_bytes(duplicate_bytes)
            (root / "c.mp4").write_bytes(unmatched_bytes)
            root.chmod(0o700)
            prior, _ = build([("media-set-json", "nextgqa-media", "synthetic-prior",
                               json.dumps({"media_sha256": [hashlib.sha256(matched_bytes).hexdigest()]},
                                          sort_keys=True).encode())], KEY)
            result = reconcile(pool_root=root,
                               index_raws=[json.dumps(prior).encode()], secret=KEY,
                               expected_video_files=3)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["candidate_video_file_count"], 3)
            self.assertEqual(result["candidate_unique_media_count"], 2)
            self.assertEqual(result["exact_media_byte_matches"], 1)
            self.assertEqual(result["unique_media_unmatched_to_supplied_index"], 1)
            self.assertFalse(result["historical_completeness_authenticated"])
            self.assertFalse(result["source_event_independence_assessed"])
            self.assertFalse(result["faithfulness_scored"])
            serialized = json.dumps(result)
            for private in (str(root), matched_bytes.decode(), unmatched_bytes.decode()):
                self.assertNotIn(private, serialized)

    def test_candidate_count_mismatch_is_reported_without_authorizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir(mode=0o700)
            (root / "a.mp4").write_bytes(b"synthetic")
            root.chmod(0o700)
            empty, _ = build([("media-set-json", "nextgqa-media", "synthetic-empty",
                               b'{"media_sha256":[]}')], KEY)
            result = reconcile(pool_root=root, index_raws=[json.dumps(empty).encode()],
                               secret=KEY, expected_video_files=2)
            self.assertEqual(result["status"], "candidate_file_count_mismatch")
            self.assertEqual(result["candidate_video_file_count"], 1)
            self.assertEqual(result["model_forwards"], 0)

    def test_cli_count_mismatch_emits_summary_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "pool"
            root.mkdir(mode=0o700)
            (root / "a.mp4").write_bytes(b"synthetic")
            root.chmod(0o700)
            key = parent / "key"
            key.write_bytes(KEY)
            key.chmod(0o600)
            prior, encoded = build([("media-set-json", "nextgqa-media", "synthetic-cli",
                                     b'{"media_sha256":[]}')], KEY)
            index = parent / "index.json"
            index.write_bytes(encoded)
            index.chmod(0o600)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--pool-root", str(root), "--prior-index", str(index),
                             "--hmac-key", str(key), "--expected-video-files", "2"])
            self.assertEqual(code, 2)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["status"], "candidate_file_count_mismatch")
            self.assertNotIn(str(root), output.getvalue())

    def test_multiple_explicit_indexes_are_unioned_without_listing_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir(mode=0o700)
            media = b"synthetic only"
            (root / "a.mp4").write_bytes(media)
            root.chmod(0o700)
            indexes = []
            for i in range(2):
                prior, _ = build([("media-set-json", "nextgqa-media", f"synthetic-{i}",
                                   json.dumps({"media_sha256": [hashlib.sha256(media).hexdigest()]},
                                              sort_keys=True).encode())], KEY)
                indexes.append(json.dumps(prior).encode())
            result = reconcile(pool_root=root, index_raws=indexes, secret=KEY)
            self.assertEqual(result["prior_index_manifest_count"], 2)
            self.assertEqual(result["prior_index_media_identity_count"], 1)
            self.assertEqual(result["exact_media_byte_matches"], 1)

    def test_symlink_and_special_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir(mode=0o700)
            target = Path(tmp) / "outside.mp4"
            target.write_bytes(b"synthetic")
            (root / "alias.mp4").symlink_to(target)
            root.chmod(0o700)
            empty, _ = build([("media-set-json", "nextgqa-media", "synthetic-empty",
                               b'{"media_sha256":[]}')], KEY)
            with self.assertRaisesRegex(ValueError, "candidate_tree_symlink_rejected"):
                reconcile(pool_root=root, index_raws=[json.dumps(empty).encode()], secret=KEY)

    def test_symlinked_ancestor_of_candidate_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            actual = parent / "actual"
            actual.mkdir(mode=0o700)
            alias = parent / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            empty, _ = build([("media-set-json", "nextgqa-media", "synthetic-root",
                               b'{"media_sha256":[]}')], KEY)
            with self.assertRaisesRegex(ValueError, "candidate_root_unavailable"):
                reconcile(pool_root=alias, index_raws=[json.dumps(empty).encode()], secret=KEY)

    def test_swap_after_tree_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir(mode=0o700)
            original_bytes = b"original candidate bytes"
            (root / "a.mp4").write_bytes(original_bytes)
            root.chmod(0o700)
            prior, _ = build([("media-set-json", "nextgqa-media", "synthetic-swap",
                               json.dumps({"media_sha256": [hashlib.sha256(original_bytes).hexdigest()]},
                                          sort_keys=True).encode())], KEY)
            from scripts import reconcile_nextgqa_prior_exposure_pool as module
            original_hash = module._hash_stable_file

            def swap_then_hash(root_fd, relative, expected_tree, *, max_bytes):
                held = ".held-candidate"
                os.rename(relative, held, src_dir_fd=root_fd, dst_dir_fd=root_fd)
                fd = os.open(relative, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600, dir_fd=root_fd)
                try:
                    os.write(fd, b"substitute candidate bytes")
                finally:
                    os.close(fd)
                try:
                    return original_hash(root_fd, relative, expected_tree,
                                         max_bytes=max_bytes)
                finally:
                    os.unlink(relative, dir_fd=root_fd)
                    os.rename(held, relative, src_dir_fd=root_fd, dst_dir_fd=root_fd)

            with patch("scripts.reconcile_nextgqa_prior_exposure_pool._hash_stable_file",
                       side_effect=swap_then_hash):
                with self.assertRaisesRegex(ValueError, "candidate_tree_changed_during_audit"):
                    reconcile(pool_root=root, index_raws=[json.dumps(prior).encode()], secret=KEY)


if __name__ == "__main__":
    unittest.main()
