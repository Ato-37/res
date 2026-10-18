#!/usr/bin/env python3
"""Aggregate-only CUDA perceptual-hash screen over a private MP4 pool.

This is an exploratory near-duplicate screen, not source/event verification or
faithfulness scoring. Video paths, frames, per-video hashes, and pair identities
never appear in the report; the private inventory fingerprint is aggregate-only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import stat
from typing import Any

try:
    from build_nextgqa_prior_exposure_manifest import _open_path_no_follow
    from reconcile_nextgqa_prior_exposure_pool import (
        _open_candidate_root, _stat_identity, _tree_snapshot,
    )
except ImportError:
    from scripts.build_nextgqa_prior_exposure_manifest import _open_path_no_follow
    from scripts.reconcile_nextgqa_prior_exposure_pool import (
        _open_candidate_root, _stat_identity, _tree_snapshot,
    )


MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_VIDEOS = 10000
FROZEN_POOL_SIZE = 490


def _open_relative_video(root_fd: int, relative: str) -> tuple[int, tuple[int, ...]]:
    parts = relative.split("/")
    current = root_fd
    owned: list[int] = []
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=current)
            owned.append(next_fd)
            current = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=current)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > MAX_VIDEO_BYTES:
            os.close(fd)
            raise ValueError("video_type_or_size_invalid")
        identity = _stat_identity(info)
        return fd, identity
    except OSError as exc:
        raise ValueError("video_open_failed") from exc
    finally:
        for fd in reversed(owned):
            os.close(fd)


def _decode_video(relative: str, root_fd: int, expected_identity: tuple[int, ...],
                  *, frames_per_video: int, cv2: Any) -> tuple[list[Any], int]:
    import numpy as np

    fd, identity = _open_relative_video(root_fd, relative)
    try:
        if identity != expected_identity:
            raise ValueError("video_tree_identity_changed")
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise ValueError("video_decode_open_failed")
        try:
            frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if frame_count < frames_per_video:
                raise ValueError("video_too_short_for_fixed_sampling")
            # Zero-based indices; Python round uses ties-to-even. This implements
            # the frozen interior fractions (i+1)/9 of the decoded frame count.
            positions = [round((i + 1) * frame_count / (frames_per_video + 1))
                         for i in range(frames_per_video)]
            if len(set(positions)) != frames_per_video:
                raise ValueError("sample_positions_not_unique")
            images = []
            for position in positions:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, position):
                    raise ValueError("video_seek_failed")
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    raise ValueError("video_frame_decode_failed")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                images.append(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA))
            return images, frame_count
        finally:
            cap.release()
    finally:
        after = os.fstat(fd)
        os.close(fd)
        if _stat_identity(after) != identity:
            raise ValueError("video_changed_during_decode")


def _sha256_fd(root_fd: int, relative: str, expected_identity: tuple[int, ...]) -> str:
    fd, identity = _open_relative_video(root_fd, relative)
    try:
        if identity != expected_identity:
            raise ValueError("video_tree_identity_changed")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if _stat_identity(os.fstat(fd)) != identity:
            raise ValueError("video_changed_during_hash")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _validate_pool_manifest(root_fd: int, snapshot: dict[str, tuple[int, ...]],
                            paths: list[str], manifest_path: Path,
                            *, expected_videos: int = FROZEN_POOL_SIZE,
                            expected_manifest_sha256: str) -> str:
    parent = manifest_path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode)
            or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700):
        raise ValueError("pool_manifest_parent_must_be_owned_mode_0700_directory")
    fd = _open_path_no_follow(manifest_path)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("pool_manifest_must_be_regular_mode_0600")
        before = _stat_identity(info)
        raw = b""
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            raw += block
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("pool_manifest_size_limit_exceeded")
        if _stat_identity(os.fstat(fd)) != before:
            raise ValueError("pool_manifest_changed_during_read")
    finally:
        os.close(fd)
    try:
        obj = json.loads(raw)
        rows = obj["entries"]
        expected = {row["path"]: row["sha256"] for row in rows}
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("pool_manifest_invalid") from exc
    if (obj.get("schema_version") != 1 or obj.get("purpose") != "private_candidate_pool_freeze"
            or obj.get("expected_videos") != expected_videos
            or len(expected) != len(rows) or set(expected) != set(paths)):
        raise ValueError("pool_manifest_membership_mismatch")
    for relative in paths:
        if (not isinstance(expected[relative], str) or len(expected[relative]) != 64
                or any(c not in "0123456789abcdef" for c in expected[relative])):
            raise ValueError("pool_manifest_invalid_digest")
        digest = _sha256_fd(root_fd, relative, snapshot[relative])
        if digest != expected[relative]:
            raise ValueError("pool_manifest_content_mismatch")
    actual_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("pool_manifest_fingerprint_mismatch")
    return actual_manifest_sha256


def _audit_status(failed: int, max_videos: int | None) -> tuple[str, str]:
    if failed:
        return "partial_decode_failures", "bounded_smoke" if max_videos is not None else "full_pool"
    if max_videos is not None:
        return "bounded_smoke_completed", "bounded_smoke"
    return "completed", "full_pool"


def _dct_basis(torch: Any, device: Any, size: int = 32):
    import math

    n = torch.arange(size, device=device, dtype=torch.float32)
    k = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((math.pi / (2 * size)) * k * (2 * n + 1))
    basis[0, :] *= (1.0 / size) ** 0.5
    basis[1:, :] *= (2.0 / size) ** 0.5
    return basis


def _phash_bits(images: list[Any], *, torch: Any, device: Any):
    import numpy as np

    batch = torch.as_tensor(np.stack(images), dtype=torch.float32, device=device) / 255.0
    informative = batch.std(dim=(1, 2), unbiased=False) >= (5.0 / 255.0)
    basis = _dct_basis(torch, device)
    coefficients = torch.matmul(torch.matmul(basis, batch), basis.T)
    low = coefficients[:, :8, :8].reshape(len(images), 64)
    median = low[:, 1:].median(dim=1).values
    return low > median[:, None], informative


def _frame_pair_support(a: Any, b: Any, valid_a: Any, valid_b: Any,
                        *, hamming_max: int, torch: Any):
    distances = (a[:, None, :] != b[None, :, :]).sum(dim=-1)
    hits = ((distances <= hamming_max)
            & valid_a[:, None] & valid_b[None, :])
    left_support = hits.any(dim=1).sum()
    right_support = hits.any(dim=0).sum()
    return torch.minimum(left_support, right_support)


def _decode_or_failure(relative: str, root_fd: int, identity: tuple[int, ...],
                       *, frames_per_video: int, cv2: Any,
                       decode_fn: Any = _decode_video) -> tuple[list[Any] | None, int | None]:
    try:
        return decode_fn(relative, root_fd, identity, frames_per_video=frames_per_video,
                         cv2=cv2)
    except (OSError, ValueError, RuntimeError, cv2.error):
        return None, None


def audit_pool(*, pool_root: Path, expected_videos: int,
               pool_manifest: Path, expected_manifest_sha256: str,
               max_videos: int | None = None, frames_per_video: int = 8,
               hamming_max: int = 6, min_matched_frames: int = 2,
               pair_batch_size: int = 256, device_name: str = "cuda:0") -> dict[str, Any]:
    import cv2
    import torch

    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise ValueError("cuda_required")
    if (frames_per_video != 8 or hamming_max != 6 or min_matched_frames != 2
            or pair_batch_size < 1 or pair_batch_size > 1024):
        raise ValueError("protocol_parameters_invalid")
    root_fd, root_identity = _open_candidate_root(pool_root)
    try:
        snapshot, all_video_paths = _tree_snapshot(root_fd)
        if expected_videos != FROZEN_POOL_SIZE:
            raise ValueError("frozen_candidate_video_count_must_be_490")
        if len(all_video_paths) != expected_videos:
            raise ValueError("candidate_video_count_mismatch")
        pool_manifest_sha256 = _validate_pool_manifest(root_fd, snapshot,
                                                       all_video_paths, pool_manifest,
                                                       expected_videos=expected_videos,
                                                       expected_manifest_sha256=expected_manifest_sha256)
        if max_videos is not None:
            if max_videos < 1 or max_videos > len(all_video_paths):
                raise ValueError("smoke_video_count_invalid")
            video_paths = all_video_paths[:max_videos]
        else:
            video_paths = all_video_paths
        if len(video_paths) > MAX_VIDEOS:
            raise ValueError("candidate_video_count_limit_exceeded")
        declared_total_bytes = sum(snapshot[relative][4] for relative in video_paths)
        if declared_total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("candidate_total_size_limit_exceeded")
        torch.cuda.set_device(device_name)
        torch.cuda.reset_peak_memory_stats()
        decoded_frame_groups = []
        informative_video_counts = []
        low_texture_frames = 0
        failed = 0
        frame_count_total = 0
        byte_count_total = 0
        for relative in video_paths:
            images, frame_count = _decode_or_failure(
                relative, root_fd, snapshot[relative], frames_per_video=frames_per_video,
                cv2=cv2)
            if images is not None and frame_count is not None:
                decoded_frame_groups.append(images)
                frame_count_total += frame_count
                byte_count_total += snapshot[relative][4]
                if byte_count_total > MAX_TOTAL_BYTES:
                    raise ValueError("candidate_total_size_limit_exceeded")
            else:
                failed += 1
        after_snapshot, _ = _tree_snapshot(root_fd)
        if after_snapshot != snapshot or _stat_identity(os.fstat(root_fd)) != root_identity:
            raise ValueError("candidate_tree_changed_during_audit")
        if decoded_frame_groups:
            flat_images = [image for group in decoded_frame_groups for image in group]
            flat_features, flat_valid = _phash_bits(flat_images, torch=torch,
                                                    device=torch.device(device_name))
            features = flat_features.reshape(len(decoded_frame_groups), frames_per_video, 64)
            informative = flat_valid.reshape(len(decoded_frame_groups), frames_per_video)
            informative_video_counts = informative.sum(dim=1).detach().cpu().tolist()
            low_texture_frames = int((~flat_valid).sum().item())
            n = len(decoded_frame_groups)
            pairs = list(itertools.combinations(range(n), 2))
            support_counts = [0] * (frames_per_video + 1)
            candidate_pairs = 0
            for start in range(0, len(pairs), pair_batch_size):
                chunk = pairs[start:start + pair_batch_size]
                left = torch.tensor([pair[0] for pair in chunk], device=device_name)
                right = torch.tensor([pair[1] for pair in chunk], device=device_name)
                a, b = features[left], features[right]
                distances = (a[:, :, None, :] != b[:, None, :, :]).sum(dim=-1)
                matches = ((distances <= hamming_max)
                           & informative[left, :, None]
                           & informative[right, None, :])
                left_support = matches.any(dim=2).sum(dim=1)
                right_support = matches.any(dim=1).sum(dim=1)
                support = torch.minimum(left_support, right_support).detach().cpu().tolist()
                for value in support:
                    support_counts[int(value)] += 1
                    if value >= min_matched_frames:
                        candidate_pairs += 1
            del features
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated()
        else:
            candidate_pairs = 0
            support_counts = [0] * (frames_per_video + 1)
            peak_vram = 0
        compared_pairs = len(decoded_frame_groups) * (len(decoded_frame_groups) - 1) // 2
        return {
            "schema_version": 1,
            "purpose": "candidate_pool_visual_overlap_screen_aggregate_only",
            "status": _audit_status(failed, max_videos)[0],
            "audit_scope": _audit_status(failed, max_videos)[1],
            "pool_manifest_sha256": pool_manifest_sha256,
            "requested_videos": len(video_paths),
            "decoded_videos": len(decoded_frame_groups),
            "failed_videos": failed,
            "frames_per_video": frames_per_video,
            "low_texture_frames_excluded": low_texture_frames,
            "videos_with_at_least_two_informative_frames": sum(
                count >= 2 for count in informative_video_counts),
            "source_frame_count_sum_for_decoded_videos": frame_count_total,
            "compared_video_pairs": compared_pairs,
            "candidate_pairs_at_frozen_threshold": candidate_pairs,
            "distinct_matching_frame_support_histogram": {
                str(i): support_counts[i] for i in range(len(support_counts))
            },
            "phash_hamming_max": hamming_max,
            "minimum_distinct_matching_frames_per_video": min_matched_frames,
            "minimum_grayscale_stddev_for_frame": 5.0,
            "decoded_video_file_bytes_total": byte_count_total,
            "device": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": int(peak_vram),
            "neural_model_loaded": False,
            "faithfulness_scored": False,
            "model_forwards": 0,
            "source_event_independence_verified": False,
            "rights_status": "unverified_local_only",
        }
    finally:
        os.close(root_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--expected-videos", type=int, required=True,
                        help="must equal the frozen pool size of 490")
    parser.add_argument("--pool-manifest", required=True, type=Path,
                        help="private mode-0600 frozen membership/content manifest")
    parser.add_argument("--expected-manifest-sha256", required=True,
                        help="out-of-band pre-run pin for the frozen manifest")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="bounded smoke prefix; omit for the complete pool")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        result = audit_pool(pool_root=args.pool_root, expected_videos=args.expected_videos,
                            pool_manifest=args.pool_manifest,
                            expected_manifest_sha256=args.expected_manifest_sha256,
                            max_videos=args.max_videos, device_name=args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "visual_overlap_audit_failed")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
