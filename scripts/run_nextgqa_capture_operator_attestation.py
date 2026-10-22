#!/usr/bin/env python3
"""Private filesystem wrapper for the non-authorizing capture-byte attestation.

This CLI is intentionally dormant for real artifacts pending independent audit.
It accepts only explicit paths beneath an owner-only root and emits only a
content-free status line. It never opens review HTML/video or authorizes work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

from nextgqa_capture_operator_attestation import create_attestation, verify_attestation
from nextgqa_pre_review_integrity_receipt import _public_der, _sha

MAX_TOTAL = 512 * 1024 * 1024
MAX_SMALL = 2_000_000
MAX_KEY = 64_000
MAX_LEAF = 40_000_000
DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _relative(value: str) -> tuple[str, ...]:
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError("path_must_be_relative_and_contained")
    return tuple(p.parts)


def _open_dir_chain(root_fd: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            nxt = os.open(part, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = nxt
            info = os.fstat(current)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError("private_directory_invalid")
        return current
    except Exception:
        os.close(current)
        raise


def _open_private_root(path: Path) -> int:
    if not path.is_absolute() or len(path.parts) < 2:
        raise ValueError("private_root_must_be_absolute")
    fd = os.open("/", DIR_FLAGS)
    try:
        for index, part in enumerate(path.parts[1:], 1):
            nxt = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd); fd = nxt
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("private_root_path_invalid")
            if index == len(path.parts) - 1 and (
                    info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError("private_root_mode_invalid")
        return fd
    except Exception:
        os.close(fd)
        raise


def _read_file(root_fd: int, relative: str, *, cap: int, aggregate: list[int]) -> bytes:
    parts = _relative(relative)
    parent = _open_dir_chain(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], FILE_FLAGS, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or not 0 < before.st_size <= cap):
                raise ValueError("private_input_file_invalid")
            chunks, remaining = [], before.st_size
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("private_input_changed_during_read")
                chunks.append(chunk); remaining -= len(chunk)
            if os.read(fd, 1):
                raise ValueError("private_input_changed_during_read")
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("private_input_changed_during_read")
            raw = b"".join(chunks)
            aggregate[0] += len(raw)
            if aggregate[0] > MAX_TOTAL:
                raise ValueError("private_input_aggregate_limit_exceeded")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def _read_dir(root_fd: int, relative: str, *, expected: set[str], aggregate: list[int]) -> dict[str, bytes]:
    parts = _relative(relative)
    dfd = _open_dir_chain(root_fd, parts)
    try:
        before_dir = os.fstat(dfd)
        if (not stat.S_ISDIR(before_dir.st_mode) or before_dir.st_uid != os.geteuid()
                or stat.S_IMODE(before_dir.st_mode) != 0o700):
            raise ValueError("private_artifact_directory_invalid")
        expected = set(expected)
        if (not expected or any(not isinstance(n, str) or Path(n).name != n or n in {".", ".."}
                                for n in expected)
                or set(os.listdir(dfd)) != expected):
            raise ValueError("artifact_directory_contents_invalid")
        files = {}
        identities = {}
        for name in sorted(expected):
            fd = os.open(name, FILE_FLAGS, dir_fd=dfd)
            try:
                before = os.fstat(fd)
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                        or stat.S_IMODE(before.st_mode) != 0o600
                        or not 0 < before.st_size <= MAX_LEAF):
                    raise ValueError("private_artifact_file_invalid")
                aggregate[0] += before.st_size
                if aggregate[0] > MAX_TOTAL:
                    raise ValueError("private_input_aggregate_limit_exceeded")
                content, remaining = [], before.st_size
                while remaining:
                    block = os.read(fd, min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("private_artifact_changed_during_read")
                    content.append(block); remaining -= len(block)
                if os.read(fd, 1):
                    raise ValueError("private_artifact_changed_during_read")
                after = os.fstat(fd)
                identity = (before.st_dev, before.st_ino, before.st_mode, before.st_uid,
                            before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                if identity != (after.st_dev, after.st_ino, after.st_mode, after.st_uid,
                                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("private_artifact_changed_during_read")
                files[name] = b"".join(content)
                identities[name] = identity
            finally:
                os.close(fd)
        if set(os.listdir(dfd)) != expected:
            raise ValueError("artifact_directory_changed_during_read")
        after_dir = os.fstat(dfd)
        dir_identity = (before_dir.st_dev, before_dir.st_ino, before_dir.st_mode, before_dir.st_uid)
        if dir_identity != (after_dir.st_dev, after_dir.st_ino, after_dir.st_mode, after_dir.st_uid):
            raise ValueError("artifact_directory_changed_during_read")
        # Ensure the originally named path still resolves to this opened directory.
        check_fd = _open_dir_chain(root_fd, parts)
        try:
            check = os.fstat(check_fd)
            if dir_identity != (check.st_dev, check.st_ino, check.st_mode, check.st_uid):
                raise ValueError("artifact_directory_path_changed_during_read")
            if set(os.listdir(check_fd)) != expected:
                raise ValueError("artifact_directory_changed_during_read")
            for name, identity in identities.items():
                check_file = os.open(name, FILE_FLAGS, dir_fd=check_fd)
                try:
                    now = os.fstat(check_file)
                    if identity != (now.st_dev, now.st_ino, now.st_mode, now.st_uid,
                                    now.st_size, now.st_mtime_ns, now.st_ctime_ns):
                        raise ValueError("private_artifact_path_changed_during_read")
                finally:
                    os.close(check_file)
        finally:
            os.close(check_fd)
        return files
    finally:
        os.close(dfd)


def _write_output(root_fd: int, relative: str, raw: bytes) -> None:
    parts = _relative(relative)
    parent = _open_dir_chain(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("short_output_write")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(parent)
    finally:
        os.close(parent)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-root", required=True)
    p.add_argument("--plan", required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--plan-provenance", required=True); p.add_argument("--plan-provenance-sha256", required=True)
    p.add_argument("--observations", required=True); p.add_argument("--observation-index", required=True)
    p.add_argument("--capture-receipt", required=True); p.add_argument("--crop-map", required=True)
    p.add_argument("--crop-crosswalk", required=True); p.add_argument("--screenshots-dir", required=True)
    p.add_argument("--crops-dir", required=True); p.add_argument("--integrity-roster", required=True)
    p.add_argument("--release-roster", required=True); p.add_argument("--root-public-key", required=True)
    p.add_argument("--root-public-der-sha256", required=True)
    p.add_argument("--operator-private-key", required=True); p.add_argument("--operator-public-key", required=True)
    p.add_argument("--output", required=True)
    return p


def run(args: argparse.Namespace) -> None:
    root_path = Path(args.private_root)
    info = root_path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_root_invalid")
    root_fd = _open_private_root(root_path)
    try:
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino, opened.st_uid, stat.S_IMODE(opened.st_mode)) != (
                info.st_dev, info.st_ino, os.geteuid(), 0o700):
            raise ValueError("private_root_changed")
        path_args = [args.plan, args.plan_provenance, args.observations,
            args.observation_index, args.capture_receipt, args.crop_map,
            args.crop_crosswalk, args.integrity_roster, args.release_roster,
            args.root_public_key, args.operator_private_key, args.operator_public_key,
            args.screenshots_dir, args.crops_dir, args.output]
        normalized = [_relative(path) for path in path_args]
        if len(set(normalized)) != len(normalized):
            raise ValueError("input_output_path_alias")
        for directory in (normalized[-3], normalized[-2]):
            if any(directory == other[:len(directory)] for other in normalized if other != directory):
                raise ValueError("artifact_directory_path_alias")
        aggregate = [0]
        def read(name, cap=MAX_SMALL): return _read_file(root_fd, getattr(args, name), cap=cap, aggregate=aggregate)
        plan = read("plan"); provenance = read("plan_provenance")
        observations = read("observations"); index = read("observation_index")
        receipt = read("capture_receipt"); cropmap = read("crop_map"); crosswalk = read("crop_crosswalk")
        integrity = read("integrity_roster"); release = read("release_roster")
        root_pub = read("root_public_key", MAX_KEY)
        operator_priv = read("operator_private_key", MAX_KEY); operator_pub = read("operator_public_key", MAX_KEY)
        try:
            receipt_obj = json.loads(receipt)
            crosswalk_obj = json.loads(crosswalk)
            if (not isinstance(receipt_obj, dict) or not isinstance(receipt_obj.get("records"), list)
                    or not isinstance(crosswalk_obj, dict) or not isinstance(crosswalk_obj.get("records"), list)):
                raise ValueError("capture_artifact_index_invalid")
            screenshot_names = {row["screenshot_file"] for row in receipt_obj["records"]}
            crop_names = {row["crop_file"] for row in crosswalk_obj["records"]}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("capture_artifact_index_invalid") from exc
        screenshots = _read_dir(root_fd, args.screenshots_dir, expected=screenshot_names, aggregate=aggregate)
        crops = _read_dir(root_fd, args.crops_dir, expected=crop_names, aggregate=aggregate)
        inputs = {"plan_raw": plan, "expected_plan_sha256": args.plan_sha256,
            "plan_provenance_raw": provenance, "expected_plan_provenance_sha256": args.plan_provenance_sha256,
            "observations_raw": observations, "observation_index_raw": index,
            "capture_receipt_raw": receipt, "crop_map_raw": cropmap,
            "crop_crosswalk_raw": crosswalk, "screenshot_files": screenshots,
            "crop_files": crops, "integrity_roster_raw": integrity,
            "release_roster_raw": release, "root_public_key_pem": root_pub,
            "expected_root_public_der_sha256": args.root_public_der_sha256}
        attestation = create_attestation(reviewer_private_key_pem=operator_priv,
            reviewer_public_key_pem=operator_pub, **inputs)
        report = verify_attestation(attestation_raw=attestation, **inputs)
        if report["execution_authorization"] != "not_granted" or report["model_load_permitted"]:
            raise ValueError("attestation_authorization_invariant_failed")
        _write_output(root_fd, args.output, attestation)
        print("status=candidate_capture_operator_attestation_created authorization=not_granted model_load_permitted=false")
    finally:
        os.close(root_fd)


def main() -> int:
    try:
        run(_parser().parse_args())
        return 0
    except Exception as exc:
        del exc
        print("status=failed code=private_attestation_error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
