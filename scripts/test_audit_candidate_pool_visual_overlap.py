import unittest
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.audit_candidate_pool_visual_overlap import (
    _audit_status, _decode_or_failure, _frame_pair_support, _phash_bits,
    _validate_pool_manifest,
)
from scripts.reconcile_nextgqa_prior_exposure_pool import (
    _open_candidate_root, _tree_snapshot,
)


class CandidatePoolVisualOverlapTests(unittest.TestCase):
    def test_smoke_cannot_be_reported_as_full_completion(self):
        self.assertEqual(_audit_status(0, 4), ("bounded_smoke_completed", "bounded_smoke"))
        self.assertEqual(_audit_status(0, None), ("completed", "full_pool"))
        self.assertEqual(_audit_status(1, None), ("partial_decode_failures", "full_pool"))

    def test_manifest_path_rejects_symlinked_ancestor(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            real = base / "real"
            real.mkdir(mode=0o700)
            nested = real / "nested"
            nested.mkdir(mode=0o700)
            manifest = nested / "manifest.json"
            manifest.write_text("{}")
            os.chmod(manifest, 0o600)
            (base / "alias").symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                _validate_pool_manifest(-1, {}, [], base / "alias" / "nested" / "manifest.json",
                                        expected_videos=0,
                                        expected_manifest_sha256="0" * 64)

    def test_pool_manifest_binds_membership_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "pool"
            root.mkdir(mode=0o700)
            video = root / "one.mp4"
            video.write_bytes(b"synthetic video bytes")
            os.chmod(video, 0o600)
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "purpose": "private_candidate_pool_freeze",
                "expected_videos": 1,
                "entries": [{"path": "one.mp4", "sha256": digest}],
            }))
            os.chmod(manifest, 0o600)
            root_fd, _ = _open_candidate_root(root)
            try:
                snapshot, paths = _tree_snapshot(root_fd)
                self.assertEqual(_validate_pool_manifest(root_fd, snapshot, paths, manifest,
                                                         expected_videos=1,
                                                         expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
                                 hashlib.sha256(manifest.read_bytes()).hexdigest())
                video.write_bytes(b"changed synthetic video bytes")
                snapshot, paths = _tree_snapshot(root_fd)
                with self.assertRaisesRegex(ValueError, "pool_manifest_content_mismatch"):
                    _validate_pool_manifest(root_fd, snapshot, paths, manifest,
                                             expected_videos=1,
                                             expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            finally:
                os.close(root_fd)
    def test_cv2_decoder_error_is_counted_as_per_video_failure(self):
        class FakeCv2:
            class error(Exception):
                pass

        def decoder(*args, **kwargs):
            raise FakeCv2.error("synthetic decoder failure")

        images, frame_count = _decode_or_failure(
            "video.mp4", 0, (), frames_per_video=8, cv2=FakeCv2,
            decode_fn=decoder)
        self.assertIsNone(images)
        self.assertIsNone(frame_count)

    def test_identical_textured_frames_have_full_matching_support(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[4:28, 7:25] = 180
        image[11:18, :] = 240
        bits, informative = _phash_bits([image] * 16, torch=torch,
                                        device=torch.device("cpu"))
        a, valid_a = bits[:8], informative[:8]
        b, valid_b = bits[8:], informative[8:]
        support = _frame_pair_support(a, b, valid_a, valid_b,
                                      hamming_max=6, torch=torch)
        self.assertEqual(int(support), 8)
        self.assertTrue(bool(valid_a.all() and valid_b.all()))

    def test_low_texture_frames_never_form_false_match(self):
        flat_black = np.zeros((32, 32), dtype=np.uint8)
        flat_gray = np.full((32, 32), 160, dtype=np.uint8)
        bits, informative = _phash_bits([flat_black] * 8 + [flat_gray] * 8,
                                        torch=torch, device=torch.device("cpu"))
        support = _frame_pair_support(bits[:8], bits[8:], informative[:8],
                                      informative[8:], hamming_max=6, torch=torch)
        self.assertEqual(int(support), 0)
        self.assertFalse(bool(informative.any()))

    def test_distinct_pattern_does_not_match_under_fixed_threshold(self):
        a = np.zeros((32, 32), dtype=np.uint8)
        a[4:28, 7:25] = 180
        a[11:18, :] = 240
        b = np.zeros((32, 32), dtype=np.uint8)
        b[:, 3:9] = 210
        b[20:27, 10:30] = 250
        bits, informative = _phash_bits([a] * 8 + [b] * 8,
                                        torch=torch, device=torch.device("cpu"))
        support = _frame_pair_support(bits[:8], bits[8:], informative[:8],
                                      informative[8:], hamming_max=6, torch=torch)
        self.assertEqual(int(support), 0)


if __name__ == "__main__":
    unittest.main()
