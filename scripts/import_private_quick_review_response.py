#!/usr/bin/env python3
"""Validate and copy an exploratory response into owner-only local storage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

from inspect_nextgqa_quick_review_response import (
    DEFAULT_ARTIFACT, MAX_JSON_BYTES, ReviewError, inspect_response,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data/private_nextgqa"
RESPONSE_DIRNAME = "quick_review_responses"


def _read_source(path: Path) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise ReviewError("response_source_open_failed") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size < 1
                or before.st_size > MAX_JSON_BYTES):
            raise ReviewError("response_source_type_or_size_invalid")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(64 * 1024, MAX_JSON_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_JSON_BYTES:
                raise ReviewError("response_source_too_large")
        after = os.fstat(fd)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns)
        if signature(before) != signature(after) or size != before.st_size:
            raise ReviewError("response_source_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _private_directory_fd(private_root: Path) -> int:
    """Create and open a private response directory under a mode-0700 root."""
    try:
        root_fd = os.open(private_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                          | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ReviewError("private_root_open_failed") from exc
    try:
        root_stat = os.fstat(root_fd)
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise ReviewError("private_root_permissions_invalid")
        try:
            os.mkdir(RESPONSE_DIRNAME, mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        try:
            response_fd = os.open(
                RESPONSE_DIRNAME, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        except OSError as exc:
            raise ReviewError("private_response_directory_invalid") from exc
        response_stat = os.fstat(response_fd)
        if (not stat.S_ISDIR(response_stat.st_mode)
                or response_stat.st_uid != os.geteuid()
                or stat.S_IMODE(response_stat.st_mode) != 0o700):
            os.close(response_fd)
            raise ReviewError("private_response_directory_permissions_invalid")
        os.close(root_fd)
        return response_fd
    except Exception:
        os.close(root_fd)
        raise


def _response_filename(raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"response_{stamp}_{digest[:12]}_{secrets.token_hex(3)}.json"


def _find_exact_duplicate(raw: bytes, response_fd: int) -> str | None:
    """Return an existing owner-only response with identical bytes, if present."""
    digest = hashlib.sha256(raw).hexdigest()
    prefix = f"response_*_{digest[:12]}_*.json"
    for name in os.listdir(response_fd):
        if not fnmatch.fnmatchcase(name, prefix):
            continue
        try:
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), dir_fd=response_fd)
        except OSError as exc:
            raise ReviewError("private_response_duplicate_open_failed") from exc
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_size < 1 or before.st_size > MAX_JSON_BYTES):
                raise ReviewError("private_response_duplicate_invalid")
            hasher = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(fd, min(64 * 1024, MAX_JSON_BYTES + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_JSON_BYTES:
                    raise ReviewError("private_response_duplicate_invalid")
                hasher.update(chunk)
            after = os.fstat(fd)
            signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                      value.st_mtime_ns, value.st_ctime_ns)
            if signature(before) != signature(after) or size != before.st_size:
                raise ReviewError("private_response_duplicate_changed_during_read")
            if hasher.hexdigest() == digest:
                return name
        finally:
            os.close(fd)
    return None


def import_response(source_path: Path, *, artifact_path: Path = DEFAULT_ARTIFACT,
                    private_root: Path | None = None,
                    allow_unbound_legacy: bool = False) -> tuple[Path, dict]:
    """Validate exact artifact mapping, then create a new mode-0600 private copy.

    The source is retained. No response text or nickname is returned or printed.
    """
    if private_root is None:
        private_root = PRIVATE_ROOT
    raw = _read_source(Path(source_path))
    # Validate the bytes before copying. The inspection result intentionally
    # contains only broad judgments and opaque item ordinals.
    from inspect_nextgqa_quick_review_response import (
        expected_artifact_fingerprint, expected_item_ids, summarize_response,
    )
    summary = summarize_response(
        raw, expected_item_ids(Path(artifact_path)),
        expected_artifact_fingerprint(Path(artifact_path)),
    )
    if summary["artifact_binding"] != "verified" and not allow_unbound_legacy:
        raise ReviewError("legacy_artifact_binding_requires_explicit_opt_in")

    response_fd = _private_directory_fd(Path(private_root))
    try:
        duplicate_name = _find_exact_duplicate(raw, response_fd)
    except Exception:
        os.close(response_fd)
        raise
    if duplicate_name is not None:
        os.close(response_fd)
        destination = Path(private_root) / RESPONSE_DIRNAME / duplicate_name
        summary = inspect_response(destination, Path(artifact_path),
                                   allow_unbound_legacy=allow_unbound_legacy)
        summary["_already_present"] = True
        return destination, summary
    name = _response_filename(raw)
    temp_name = f".{name}.partial"
    fd = None
    temp_created = False
    published = False
    try:
        try:
            fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600,
                         dir_fd=response_fd)
        except FileExistsError as exc:
            raise ReviewError("private_response_temporary_exists") from exc
        temp_created = True
        os.fchmod(fd, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError("short_write")
            written += count
        os.fsync(fd)
        file_stat = os.fstat(fd)
        if (not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.geteuid()
                or stat.S_IMODE(file_stat.st_mode) != 0o600
                or file_stat.st_size != len(raw)):
            raise ReviewError("private_response_file_verification_failed")
        os.close(fd)
        fd = None
        try:
            # Hard-link publication is atomic and fails if the final name exists.
            os.link(temp_name, name, src_dir_fd=response_fd,
                    dst_dir_fd=response_fd, follow_symlinks=False)
        except FileExistsError as exc:
            raise ReviewError("private_response_destination_exists") from exc
        published = True
        os.unlink(temp_name, dir_fd=response_fd)
        temp_created = False
        os.fsync(response_fd)
        destination = Path(private_root) / RESPONSE_DIRNAME / name
        # Read back through the normal inspector to verify the saved copy too.
        summary = inspect_response(destination, Path(artifact_path),
                                   allow_unbound_legacy=allow_unbound_legacy)
        summary["_already_present"] = False
        return destination, summary
    except Exception:
        if fd is not None:
            os.close(fd)
        if temp_created:
            try:
                os.unlink(temp_name, dir_fd=response_fd)
            except FileNotFoundError:
                pass
        if published:
            try:
                os.unlink(name, dir_fd=response_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(response_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_json", type=Path,
                        help="ブラウザーから保存した探索的回答JSON（元ファイルは残します）")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT,
                        help="item ID照合対象の簡易確認HTML")
    parser.add_argument("--show-items", action="store_true",
                        help="既定の集計に加え、参加者/項目別の回答も表示する")
    parser.add_argument("--allow-unbound-legacy", action="store_true",
                        help="旧schema v2〜v5をartifact版未検証の履歴としてprivate保管する")
    args = parser.parse_args(argv)
    try:
        destination, report = import_response(
            args.response_json, artifact_path=args.artifact,
            allow_unbound_legacy=args.allow_unbound_legacy)
    except (OSError, ReviewError) as exc:
        code = str(exc) if str(exc).isascii() else "private_import_failed"
        parser.error(code)
    counts = report["judgment_counts"]
    labels = {"roughly_matches": "だいたい合っている",
              "partly_differs": "一部ちがう",
              "cannot_tell": "動画だけではわからない"}
    print(f"形式確認: {report['respondent_count']}人・探索的回答 "
          f"{report['validated_exploratory_responses']}件")
    print("artifact HTML/JSON fingerprint一致: "
          + ("確認済み" if report["artifact_binding"] == "verified"
             else "未検証（明示許可された旧形式JSON）"))
    if counts is not None:
        for key, label in labels.items():
            print(f"{label}: {counts[key]}件")
    for item in report["item_judgments"] if args.show_items else []:
        if "judgment" in item:
            print(f"{item['respondent']}人目・第{item['item']}件目: "
                  f"{labels[item['judgment']]}")
        elif "question_clarity" in item:
            clarity = {"clear": "わかりやすい", "unclear": "わかりにくい",
                       "cannot_tell": "判断できない"}
            fit = {"fits": "合っている", "does_not_fit": "違っている",
                   "cannot_tell": "判断できない"}
            evidence = {"sufficient": "十分", "insufficient": "足りない点がある",
                        "cannot_tell": "判断できない"}
            print(f"{item['respondent']}人目・第{item['item']}件目: "
                  f"質問={clarity[item['question_clarity']]}、"
                  f"答え={fit[item['answer_question_fit']]}、"
                  f"動画根拠={evidence[item['video_evidence_sufficiency']]}")
        else:
            sufficiency_labels = {"sufficient": "十分", "insufficient": "足りない",
                                  "cannot_tell": "判断できない"}
            print(f"{item['respondent']}人目・第{item['item']}件目: "
                  f"主張={labels[item['claim_truth']]}、"
                  f"根拠={sufficiency_labels[item['explanation_sufficiency']]}")
    if report["sufficiency_counts"] is not None:
        sufficiency = report["sufficiency_counts"]
        print("根拠の十分さ: "
              f"十分={sufficiency['sufficient']}件、足りない={sufficiency['insufficient']}件、"
              f"判断できない={sufficiency['cannot_tell']}件")
    if report["question_clarity_counts"] is not None:
        clarity = report["question_clarity_counts"]
        fit = report["answer_question_fit_counts"]
        evidence = report["video_evidence_sufficiency_counts"]
        print("質問の明瞭さ: "
              f"わかりやすい={clarity['clear']}件、わかりにくい={clarity['unclear']}件、"
              f"判断できない={clarity['cannot_tell']}件")
        print("答えの適合: "
              f"合っている={fit['fits']}件、違っている={fit['does_not_fit']}件、"
              f"判断できない={fit['cannot_tell']}件")
        print("動画根拠の十分さ: "
              f"十分={evidence['sufficient']}件、足りない={evidence['insufficient']}件、"
              f"判断できない={evidence['cannot_tell']}件")
    print(f"private保存先: {destination}")
    if report.get("_already_present"):
        print("同一内容は既にprivate保存済みのため、既存ファイルを再利用しました。")
    print("保存JSONはowner-only (0600) です。元のダウンロードは削除していません。")
    print("これは印象確認でありfaithfulness推定・採点許可ではありません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
