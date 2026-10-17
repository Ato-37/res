#!/usr/bin/env python3
"""Build a local-only, non-authorizing inventory of pre-review input bytes.

This records hashes and sizes only. It never parses or renders review material,
reads video frames, verifies human review, freezes a protocol, or authorizes
scoring. Every output remains a pre-review candidate inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PRIVATE_ROOT = Path("data/private_nextgqa")
PACKAGE_DIR = PRIVATE_ROOT / "reviewer_package_v6_20261017_0635"
STAGED_MEDIA_DIR = PRIVATE_ROOT / "review_media_v6_20261017_0635"

PRIVATE_FILES = {
    "standalone_review_html": PRIVATE_ROOT / "claim_review_v6_20261017_0635.html",
    "claim_input_manifest": PRIVATE_ROOT / "claim_input_manifest_v6_20261017_0635.json",
    "claim_input_sidecar": PRIVATE_ROOT / "claim_input_sidecar_v6_20261017_0635.json",
    "claim_mapping_worksheet": PRIVATE_ROOT / "claim_mapping_sentence_v6_20261017_0635.json",
    "claim_lineage": PRIVATE_ROOT / "claim_lineage_sentence_exploratory_20261016_01.json",
    "claim_generation_packet": PRIVATE_ROOT / "claim_generation_sentence_exploratory_20261016_01.json",
    "review_custody": PRIVATE_ROOT / "review_custody_v6_20261017_0635.json",
    "clock_anchor_plan": PRIVATE_ROOT / "clock_anchor_plan_candidate_v6_20261017_0654.json",
    "clock_anchor_plan_provenance": PRIVATE_ROOT / "clock_anchor_plan_provenance_v6_20261017_0654.json",
    "diagnostic_target_candidate": PRIVATE_ROOT / "diagnostic_target_v6_current_20261017_01.json",
}

PUBLIC_OR_SOURCE_FILES = {
    "review_policy_v6": Path("docs/nextgqa_claim_review_policy_v6.json"),
    "protocol_v2_draft": Path("docs/nextgqa_finite_claim_descriptive_protocol_v2.md"),
    "review_runbook": Path("docs/nextgqa_v6_human_review_runbook.md"),
    "source_test_csv": Path("NExT-GQA/datasets/nextgqa/test.csv"),
}

CODE_FILES = {
    "review_html_renderer": "scripts/render_nextgqa_claim_review_html.py",
    "review_package_builder": "scripts/package_nextgqa_claim_review_handoff.py",
    "review_media_stager": "scripts/stage_nextgqa_review_media.py",
    "review_export_stager": "scripts/stage_nextgqa_review_exports.py",
    "review_handoff_audit": "scripts/audit_nextgqa_review_handoff.py",
    "pre_review_integrity_receipt_verifier": "scripts/nextgqa_pre_review_integrity_receipt.py",
    "review_release_order_auditor": "scripts/audit_nextgqa_review_release_order.py",
    "claim_lineage_audit": "scripts/audit_nextgqa_claim_packet_lineage.py",
    "claim_video_mapping_audit": "scripts/audit_nextgqa_video_mapping.py",
    "claim_mapping_packet_builder": "scripts/build_nextgqa_claim_mapping_packet.py",
    "claim_mapping_manifest_audit": "scripts/audit_claim_mapping_manifest.py",
    "actual_qa_source_audit": "scripts/audit_nextgqa_actual_qa_source_binding.py",
    "claim_bundle_builder": "scripts/build_nextgqa_v2_claim_input_bundle.py",
    "clock_anchor_plan_builder": "scripts/build_nextgqa_clock_anchor_plan.py",
    "clock_plan_provenance_builder": "scripts/bind_html5_clock_calibration_provenance.py",
    "clock_evidence_audit": "scripts/audit_nextgqa_clock_frame_evidence.py",
    "clock_evidence_manifest_builder": "scripts/build_nextgqa_clock_evidence_manifest.py",
    "clock_capture_input_preparer": "scripts/prepare_html5_clock_calibration_inputs.py",
    "clock_calibration": "scripts/html5_clock_capture_calibration.py",
    "clock_screenshot_capture": "scripts/capture_nextgqa_clock_screenshots.ps1",
    "clock_screenshot_cropper": "scripts/crop_nextgqa_clock_screenshots.py",
    "clock_decoder_exporter": "scripts/export_clock_anchor_decoder_pngs.py",
    "video_decoder": "scripts/claim_mask_video_decoder.py",
    "diagnostic_target_builder": "scripts/build_nextgqa_diagnostic_target_candidate.py",
    "mask_condition_builder": "scripts/nextgqa_claim_mask_conditions.py",
    "sampler_config_audit": "scripts/audit_claim_sampling_config.py",
    "condition_batch_builder": "scripts/nextgqa_condition_batch_builder.py",
    "prompt_tokenizer_audit": "scripts/audit_nextgqa_prompt_tokenizer.py",
    "atomic_batch_freezer": "scripts/nextgqa_atomic_batch_freeze.py",
    "batch_digest_freezer": "scripts/nextgqa_batch_digest_freeze.py",
    "scorer": "scripts/nextgqa_claim_mask_scoring.py",
    "production_prompt_builder": "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "pre_review_inventory_builder": "scripts/build_nextgqa_pre_review_byte_inventory.py",
}

PRIVATE_DIR_MODES = {
    "data/private_nextgqa": 0o700,
    PACKAGE_DIR.as_posix(): 0o700,
    STAGED_MEDIA_DIR.as_posix(): 0o700,
}
PRIVATE_FILE_MODE = 0o600
_MP4 = re.compile(r".+\.mp4\Z", re.IGNORECASE)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _canonical(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _open_dir(root_fd: int, relative: Path, *, private: bool = False) -> int:
    current = os.dup(root_fd)
    try:
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise OSError("unsafe_path")
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                os.close(next_fd)
                raise OSError("directory_invalid")
            os.close(current)
            current = next_fd
        if private and stat.S_IMODE(os.fstat(current).st_mode) != 0o700:
            raise OSError("private_directory_mode_invalid")
        return current
    except BaseException:
        os.close(current)
        raise


def _check_private_dir_chain(root_fd: int, relative: Path) -> None:
    current = os.dup(root_fd)
    walked = Path()
    try:
        for part in relative.parts:
            walked /= part
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            info = os.fstat(next_fd)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                    or (walked.as_posix() in PRIVATE_DIR_MODES
                        and stat.S_IMODE(info.st_mode) != PRIVATE_DIR_MODES[walked.as_posix()])):
                os.close(next_fd)
                raise OSError("private_directory_invalid")
            os.close(current)
            current = next_fd
    finally:
        os.close(current)


def _hash_file(root_fd: int, relative: Path, *, private: bool) -> dict[str, Any]:
    parent = _open_dir(root_fd, relative.parent)
    try:
        fd = os.open(relative.name, _FILE_FLAGS, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise OSError("input_file_invalid")
        if private and stat.S_IMODE(info.st_mode) != PRIVATE_FILE_MODE:
            raise OSError("private_file_mode_invalid")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (size != info.st_size or after.st_size != info.st_size
                or after.st_ino != info.st_ino or after.st_mtime_ns != info.st_mtime_ns
                or after.st_ctime_ns != info.st_ctime_ns):
            raise OSError("input_file_changed")
        return {"size_bytes": size, "sha256": digest.hexdigest()}
    finally:
        os.close(fd)


def _directory_files(root_fd: int, relative: Path) -> dict[str, dict[str, Any]]:
    _check_private_dir_chain(root_fd, relative)
    fd = _open_dir(root_fd, relative)
    try:
        names = os.listdir(fd)
        if any(not isinstance(name, str) or name in {"", ".", ".."} for name in names):
            raise OSError("directory_contents_invalid")
        output = {}
        for name in names:
            item_path = relative / name
            item = _hash_file(root_fd, item_path, private=True)
            output[name] = item
        return output
    finally:
        os.close(fd)


def _artifact(role: str, item: dict[str, Any]) -> dict[str, Any]:
    return {"role": role, "size_bytes": item["size_bytes"], "sha256": item["sha256"]}


def build_inventory(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Hash required local inputs and write a new private candidate receipt."""
    repo_root = repo_root.resolve(strict=True)
    root_fd = os.open(repo_root, _DIR_FLAGS)
    artifacts: list[dict[str, Any]] = []
    missing: set[str] = set()
    try:
        for relative in (Path("data/private_nextgqa"),):
            try:
                _check_private_dir_chain(root_fd, relative)
            except OSError:
                missing.add("private_data_root")

        for role, relative in PRIVATE_FILES.items():
            try:
                item = _hash_file(root_fd, relative, private=True)
            except OSError:
                missing.add(role)
            else:
                artifacts.append(_artifact(role, item))

        for role, relative in PUBLIC_OR_SOURCE_FILES.items():
            try:
                item = _hash_file(root_fd, relative, private=False)
            except OSError:
                missing.add(role)
            else:
                artifacts.append(_artifact(role, item))

        for role, relative_text in CODE_FILES.items():
            try:
                item = _hash_file(root_fd, Path(relative_text), private=False)
            except OSError:
                missing.add(role)
            else:
                artifacts.append(_artifact(role, item))

        package_files: dict[str, dict[str, Any]] = {}
        staged_files: dict[str, dict[str, Any]] = {}
        try:
            package_files = _directory_files(root_fd, PACKAGE_DIR)
        except OSError:
            missing.add("review_package_contents")
        try:
            staged_files = _directory_files(root_fd, STAGED_MEDIA_DIR)
        except OSError:
            missing.add("staged_media_contents")

        package_mp4 = sorted(name for name in package_files if _MP4.fullmatch(name))
        staged_mp4 = sorted(name for name in staged_files if _MP4.fullmatch(name))
        package_expected = {"index.html", *package_mp4}
        if (len(package_mp4) != 3 or set(package_files) != package_expected
                or "index.html" not in package_files):
            missing.add("review_package_exact_three_media_and_index")
        else:
            artifacts.append(_artifact("review_package_html", package_files["index.html"]))
        if len(staged_mp4) != 3 or set(staged_files) != set(staged_mp4):
            missing.add("staged_media_exactly_three_mp4")
        if len(package_mp4) == 3 and len(staged_mp4) == 3:
            if package_mp4 != staged_mp4:
                missing.add("package_staged_media_basename_set_match")
            for index, name in enumerate(package_mp4, start=1):
                artifacts.append(_artifact(f"review_package_media_{index:02d}", package_files[name]))
            for index, name in enumerate(staged_mp4, start=1):
                artifacts.append(_artifact(f"staged_media_{index:02d}", staged_files[name]))
            if package_mp4 == staged_mp4 and any(
                    package_files[name]["sha256"] != staged_files[name]["sha256"]
                    or package_files[name]["size_bytes"] != staged_files[name]["size_bytes"]
                    for name in package_mp4):
                missing.add("package_staged_media_byte_match")

        result = {
            "schema_version": 1,
            "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only",
            "review_started": False,
            "review_completed": False,
            "joint_freeze": False,
            "score_ready": False,
            "execution_authorization": "not_granted",
            "rights_status": "unverified_local_only",
            "artifacts": sorted(artifacts, key=lambda row: row["role"]),
            "missing_required_slots": sorted(missing),
        }
        _write_new_private_output(repo_root, output_dir, _canonical(result))
        return result
    finally:
        os.close(root_fd)


def _write_new_private_output(repo_root: Path, output_dir: Path, raw: bytes) -> None:
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    try:
        parent_relative = output_dir.parent.relative_to(repo_root)
    except (OSError, ValueError):
        raise ValueError("output_parent_invalid") from None
    root_fd = os.open(repo_root, _DIR_FLAGS)
    try:
        parent_fd = _open_dir(root_fd, parent_relative, private=True)
    finally:
        os.close(root_fd)
    parent_info = os.fstat(parent_fd)
    if (not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_uid != os.geteuid()
            or stat.S_IMODE(parent_info.st_mode) != 0o700):
        os.close(parent_fd)
        raise ValueError("output_parent_must_be_private_directory")
    try:
        os.mkdir(output_dir.name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        os.close(parent_fd)
        raise ValueError("output_directory_already_exists") from None
    except OSError:
        os.close(parent_fd)
        raise ValueError("output_directory_create_failed") from None
    output_fd = None
    directory_fd = None
    try:
        directory_fd = os.open(output_dir.name, _DIR_FLAGS, dir_fd=parent_fd)
        directory_info = os.fstat(directory_fd)
        if (not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != os.geteuid()
                or stat.S_IMODE(directory_info.st_mode) != 0o700):
            raise ValueError("output_directory_invalid")
        output_fd = os.open("pre_review_candidate_byte_inventory.json",
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                            0o600, dir_fd=directory_fd)
        view = memoryview(raw)
        while view:
            written = os.write(output_fd, view)
            if written <= 0:
                raise OSError("output_write_failed")
            view = view[written:]
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
        os.fsync(directory_fd)
    except BaseException:
        if output_fd is not None:
            os.close(output_fd)
        if directory_fd is not None:
            try:
                os.unlink("pre_review_candidate_byte_inventory.json", dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)
            directory_fd = None
        try:
            os.rmdir(output_dir.name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new unique directory under this checkout; parent must be mode 0700")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = build_inventory(repo_root, args.output_dir)
    except (OSError, ValueError):
        print(json.dumps({"status": "candidate_only", "execution_authorization": "not_granted",
                          "failure_code": "inventory_build_failed"}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"],
                      "inventory_slots": len(result["artifacts"]),
                      "missing_required_slot_count": len(result["missing_required_slots"]),
                      "execution_authorization": result["execution_authorization"]}, sort_keys=True))
    return 2 if result["missing_required_slots"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
