"""Private, exact-byte source capsules for local reproducibility.

This module handles source files only. A capsule hash is an integrity digest,
not a signature or an externally trusted provenance anchor. No extraction is
performed: verified file bytes are returned to the caller in memory.

Capsules are not integrated into any runner. The owner-only directory and
no-follow checks protect against other users and ordinary path substitution,
but do not protect against a same-UID attacker who can alter the process or
private directory. Callers must preserve the returned archive digest outside
the capsule if they need a later integrity comparison.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import unicodedata
import zipfile
from typing import Iterable


FORMAT = "qta-source-capsule-v1"
MANIFEST_NAME = "manifest.json"
MAX_HEADER_BYTES = 1 << 20
MAX_SOURCE_FILE_BYTES = 32 << 20
MAX_CAPSULE_BYTES = 128 << 20
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _assert_private_directory(fd: int, error_code: str) -> None:
    st = os.fstat(fd)
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        raise ValueError(error_code)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _parts(path: str) -> tuple[str, ...]:
    if (not isinstance(path, str) or not path or "\\" in path or "\x00" in path
            or path.startswith("/") or ":" in path
            or unicodedata.normalize("NFC", path) != path):
        raise ValueError("capsule_relative_path_invalid")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("capsule_relative_path_invalid") from None
    parts = tuple(path.split("/"))
    if any(part in ("", ".", "..") for part in parts) or path == MANIFEST_NAME:
        raise ValueError("capsule_relative_path_invalid")
    return parts


def _validate_paths(paths: Iterable[str]) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)):
        raise ValueError("capsule_manifest_invalid")
    try:
        result = tuple(paths)
    except TypeError:
        raise ValueError("capsule_manifest_invalid") from None
    if not result:
        raise ValueError("capsule_manifest_empty")
    for path in result:
        _parts(path)
    if len(set(result)) != len(result):
        raise ValueError("capsule_manifest_duplicate_path")
    folded = [path.casefold() for path in result]
    if len(set(folded)) != len(folded):
        raise ValueError("capsule_manifest_casefold_collision")
    return result


def _open_absolute_directory(path: Path):
    absolute = Path(os.path.abspath(os.fspath(path)))
    fd = os.open(os.sep, _DIR_FLAGS)
    identities = []
    try:
        for component in absolute.parts[1:]:
            before = os.stat(component, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("capsule_directory_component_invalid")
            child = os.open(component, _DIR_FLAGS, dir_fd=fd)
            after = os.fstat(child)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(child)
                raise ValueError("capsule_directory_changed")
            identities.append((component, after.st_dev, after.st_ino))
            os.close(fd)
            fd = child
        return fd, tuple(identities)
    except Exception:
        os.close(fd)
        raise


def _recheck_directory(path: Path, expected) -> None:
    fd, observed = _open_absolute_directory(path)
    os.close(fd)
    if observed != expected:
        raise ValueError("capsule_directory_changed")


def _open_dir_chain(root_fd: int, components: tuple[str, ...]):
    current = os.dup(root_fd)
    identities = []
    try:
        for component in components:
            before = os.stat(component, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("capsule_source_path_not_directory")
            child = os.open(component, _DIR_FLAGS, dir_fd=current)
            after = os.fstat(child)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(child)
                raise ValueError("capsule_source_directory_changed")
            identities.append((component, after.st_dev, after.st_ino))
            os.close(current)
            current = child
        return current, tuple(identities)
    except Exception:
        os.close(current)
        raise


def _read_source(root_fd: int, root_path: Path, root_chain, root_identity,
                 relative: str) -> bytes:
    components = _parts(relative)
    parent_fd, parent_chain = _open_dir_chain(root_fd, components[:-1])
    try:
        before = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1):
            raise ValueError("capsule_source_file_invalid")
        if before.st_mode & 0o022:
            raise ValueError("capsule_source_permissions_invalid")
        if before.st_size > MAX_SOURCE_FILE_BYTES:
            raise ValueError("capsule_source_file_too_large")
        fd = os.open(components[-1], _FILE_FLAGS, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                    or not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid()
                    or opened.st_nlink != 1):
                raise ValueError("capsule_source_file_changed")
            chunks = []
            size = 0
            while True:
                block = os.read(fd, min(1024 * 1024, MAX_SOURCE_FILE_BYTES + 1 - size))
                if not block:
                    break
                chunks.append(block)
                size += len(block)
                if size > MAX_SOURCE_FILE_BYTES:
                    raise ValueError("capsule_source_file_too_large")
            after = os.fstat(fd)
            named = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_mode,
                                  s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(opened) != identity(after) or identity(after) != identity(named):
                raise ValueError("capsule_source_file_changed")
        finally:
            os.close(fd)
        root_now, root_chain_now = _open_absolute_directory(root_path)
        try:
            root_stat = os.fstat(root_now)
            if ((root_stat.st_dev, root_stat.st_ino) != root_identity
                    or root_chain_now != root_chain):
                raise ValueError("capsule_source_root_changed")
            parent_now, parent_chain_now = _open_dir_chain(root_now, components[:-1])
            os.close(parent_now)
            if parent_chain_now != parent_chain:
                raise ValueError("capsule_source_directory_changed")
        finally:
            os.close(root_now)
        return b"".join(chunks)
    finally:
        os.close(parent_fd)


def _repo_root_if_any(root: Path) -> Path | None:
    current = root
    while True:
        marker = current / ".git"
        if marker.is_dir():
            if ((marker / "HEAD").is_file() and (marker / "objects").is_dir()
                    and (marker / "refs").is_dir()):
                return current
        elif marker.is_file():
            try:
                line = marker.read_text(encoding="utf-8").strip()
                if line.startswith("gitdir:"):
                    gitdir = Path(line[7:].strip())
                    if not gitdir.is_absolute():
                        gitdir = current / gitdir
                    if (gitdir / "HEAD").is_file() and (gitdir / "objects").is_dir():
                        return current
            except (OSError, UnicodeError):
                pass
        if current.parent == current:
            return None
        current = current.parent


def _open_private_parent(output_path: Path, source_root: Path):
    parent_path = Path(os.path.abspath(os.fspath(output_path.parent)))
    fd, chain = _open_absolute_directory(parent_path)
    st = os.fstat(fd)
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        os.close(fd)
        raise ValueError("capsule_output_directory_permissions_invalid")
    repo = _repo_root_if_any(source_root)
    if repo is not None:
        try:
            parent_path.relative_to(repo)
        except ValueError:
            pass
        else:
            os.close(fd)
            raise ValueError("capsule_output_must_be_outside_repository")
    return fd, chain


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.extra = b""
    info.comment = b""
    return info


def _validate_zip_end_record(raw: bytes, max_entries: int) -> int:
    signature = b"PK\x05\x06"
    offset = raw.rfind(signature)
    if offset < 0 or offset + 22 != len(raw):
        raise ValueError("capsule_zip_end_record_invalid")
    disk, central_disk, disk_entries, total_entries, central_size, central_offset, comment_len = (
        int.from_bytes(raw[offset + start:offset + start + width], "little")
        for start, width in ((4, 2), (6, 2), (8, 2), (10, 2), (12, 4), (16, 4), (20, 2)))
    if (disk != 0 or central_disk != 0 or disk_entries != total_entries
            or total_entries == 0xFFFF or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF or comment_len != 0
            or central_offset + central_size != offset):
        raise ValueError("capsule_zip_end_record_invalid")
    # Scan the central directory before ZipFile can materialize its members.
    # The archive itself is capped, while the manifest-derived count gives a
    # much smaller bound and lets us reject as soon as an extra record appears.
    if total_entries > max_entries:
        raise ValueError("capsule_zip_member_count_mismatch")
    expected_count = total_entries
    cursor = central_offset
    end = central_offset + central_size
    actual_entries = 0
    while cursor < end:
        if end - cursor < 46 or raw[cursor:cursor + 4] != b"PK\x01\x02":
            raise ValueError("capsule_zip_central_directory_invalid")
        name_len = int.from_bytes(raw[cursor + 28:cursor + 30], "little")
        extra_len = int.from_bytes(raw[cursor + 30:cursor + 32], "little")
        comment_length = int.from_bytes(raw[cursor + 32:cursor + 34], "little")
        record_end = cursor + 46 + name_len + extra_len + comment_length
        if record_end > end:
            raise ValueError("capsule_zip_central_directory_invalid")
        # Reject per-entry ZIP64 sentinels and multi-disk entries. ZIP64
        # extra fields are rejected later too; the sentinel check avoids
        # accepting a directory whose sizes/offsets need ZIP64 interpretation.
        if (int.from_bytes(raw[cursor + 20:cursor + 24], "little") == 0xFFFFFFFF
                or int.from_bytes(raw[cursor + 24:cursor + 28], "little") == 0xFFFFFFFF
                or int.from_bytes(raw[cursor + 42:cursor + 46], "little") == 0xFFFFFFFF
                or int.from_bytes(raw[cursor + 34:cursor + 36], "little") != 0):
            raise ValueError("capsule_zip_central_directory_invalid")
        actual_entries += 1
        if actual_entries > max_entries:
            raise ValueError("capsule_zip_member_count_mismatch")
        cursor = record_end
    if cursor != end or actual_entries != expected_count:
        raise ValueError("capsule_zip_member_count_mismatch")
    return actual_entries


def capture_source_capsule(*, source_root: Path | str,
                           ordered_paths: Iterable[str],
                           output_path: Path | str) -> dict:
    """Write a new ZIP capsule; output parent must be owned mode 0700.

    The destination is created exclusively with mode 0600 and must be outside
    a detected Git worktree. Manifest order is preserved. Existing output is
    never overwritten.
    """
    paths = _validate_paths(ordered_paths)
    output = Path(os.path.abspath(os.fspath(output_path)))
    root = Path(os.path.abspath(os.fspath(source_root)))
    root_fd, root_chain = _open_absolute_directory(root)
    root_stat = os.fstat(root_fd)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    if output.name in ("", ".", ".."):
        os.close(root_fd)
        raise ValueError("capsule_output_path_invalid")
    parent_fd = -1
    temp_fd = -1
    temp_name = None
    temp_identity = None
    try:
        parent_fd, parent_chain = _open_private_parent(output, root)
        entries = []
        contents = []
        total_size = 0
        for relative in paths:
            data = _read_source(root_fd, root, root_chain, root_identity, relative)
            total_size += len(data)
            if total_size > MAX_CAPSULE_BYTES:
                raise ValueError("capsule_source_set_too_large")
            entries.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                            "size": len(data)})
            contents.append(data)
        manifest = _canonical({"format": FORMAT, "files": entries})
        if len(manifest) > MAX_HEADER_BYTES:
            raise ValueError("capsule_manifest_too_large")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED,
                             allowZip64=False) as archive:
            archive.writestr(_zip_info(MANIFEST_NAME), manifest)
            for relative, data in zip(paths, contents, strict=True):
                archive.writestr(_zip_info(relative), data)
        capsule_bytes = buffer.getvalue()
        if len(capsule_bytes) > MAX_CAPSULE_BYTES:
            raise ValueError("capsule_too_large")
        _assert_private_directory(parent_fd, "capsule_output_directory_permissions_invalid")
        _recheck_directory(output.parent, parent_chain)
        _recheck_directory(root, root_chain)
        for _attempt in range(32):
            candidate = f".{output.name}.tmp-{secrets.token_hex(12)}"
            try:
                temp_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                  | getattr(os, "O_NOFOLLOW", 0), 0o600,
                                  dir_fd=parent_fd)
                temp_name = candidate
                break
            except FileExistsError:
                continue
        if temp_fd < 0 or temp_name is None:
            raise FileExistsError("capsule_temp_name_exhausted")
        opened = os.fstat(temp_fd)
        temp_identity = (opened.st_dev, opened.st_ino)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid()
                or stat.S_IMODE(opened.st_mode) != 0o600 or opened.st_nlink != 1):
            raise ValueError("capsule_output_permissions_invalid")
        view = memoryview(capsule_bytes)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError("capsule_write_failed")
            view = view[written:]
        os.fsync(temp_fd)
        final = os.fstat(temp_fd)
        named_temp = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if ((final.st_dev, final.st_ino) != temp_identity
                or (named_temp.st_dev, named_temp.st_ino) != temp_identity
                or stat.S_IMODE(final.st_mode) != 0o600 or final.st_nlink != 1):
            raise ValueError("capsule_output_changed")
        _recheck_directory(output.parent, parent_chain)
        # link() publishes the already-fsynced inode atomically and refuses an
        # existing destination; both names point to this private inode briefly.
        os.link(temp_name, output.name, src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd, follow_symlinks=False)
        named_final = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if ((named_final.st_dev, named_final.st_ino) != temp_identity
                or named_final.st_uid != os.getuid()
                or stat.S_IMODE(named_final.st_mode) != 0o600
                or named_final.st_nlink != 2):
            raise ValueError("capsule_publication_identity_invalid")
        named_temp = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if (named_temp.st_dev, named_temp.st_ino) != temp_identity:
            raise ValueError("capsule_temp_identity_changed")
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_name = None
        os.fsync(parent_fd)
        _assert_private_directory(parent_fd, "capsule_output_directory_permissions_invalid")
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if ((published.st_dev, published.st_ino) != temp_identity
                or published.st_nlink != 1 or stat.S_IMODE(published.st_mode) != 0o600):
            raise ValueError("capsule_publication_changed")
        _recheck_directory(output.parent, parent_chain)
        return {"status": "captured", "format": FORMAT,
                "ordered_paths": list(paths),
                "file_sha256": [entry["sha256"] for entry in entries],
                "archive_sha256": hashlib.sha256(capsule_bytes).hexdigest(),
                "capsule_size": len(capsule_bytes), "mode": "0600",
                "provenance_authenticated": False}
    except Exception:
        if parent_fd >= 0 and temp_name is not None and temp_identity is not None:
            try:
                current = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == temp_identity:
                    os.unlink(temp_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)


def recover_source_capsule(*, capsule_path: Path | str,
                           expected_paths: Iterable[str],
                           expected_archive_sha256: str | None = None) -> dict:
    """Validate capsule and return exact bytes; never extract to paths.

    Supplying a previously retained ``expected_archive_sha256`` adds an
    out-of-band integrity comparison. The computed digest alone is not an
    authenticity anchor.
    """
    if (expected_archive_sha256 is not None
            and (not isinstance(expected_archive_sha256, str)
                 or len(expected_archive_sha256) != 64
                 or any(c not in "0123456789abcdef" for c in expected_archive_sha256))):
        raise ValueError("capsule_expected_archive_hash_invalid")
    expected = _validate_paths(expected_paths)
    capsule = Path(os.path.abspath(os.fspath(capsule_path)))
    parent_fd, parent_chain = _open_absolute_directory(capsule.parent)
    fd = -1
    try:
        parent = os.fstat(parent_fd)
        if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
            raise ValueError("capsule_directory_permissions_invalid")
        before = os.stat(capsule.name, dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                or before.st_size > MAX_CAPSULE_BYTES):
            raise ValueError("capsule_file_permissions_invalid")
        fd = os.open(capsule.name, _FILE_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_uid != os.getuid() or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_nlink != 1):
            raise ValueError("capsule_file_changed")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, MAX_CAPSULE_BYTES + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_CAPSULE_BYTES:
                raise ValueError("capsule_too_large")
        raw = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(capsule.name, dir_fd=parent_fd, follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_mode,
                              s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(opened) != identity(after) or identity(after) != identity(named):
            raise ValueError("capsule_file_changed")
        _recheck_directory(capsule.parent, parent_chain)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)

    declared_entries = _validate_zip_end_record(raw, len(expected) + 1)
    if declared_entries != len(expected) + 1:
        raise ValueError("capsule_zip_member_count_mismatch")
    archive_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise ValueError("capsule_archive_hash_mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            if archive.comment:
                raise ValueError("capsule_zip_comment_invalid")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or not infos or names[0] != MANIFEST_NAME:
                raise ValueError("capsule_zip_member_set_invalid")
            if len(infos) != len(expected) + 1:
                raise ValueError("capsule_zip_member_set_invalid")
            for info in infos:
                parts = _parts(info.filename) if info.filename != MANIFEST_NAME else (MANIFEST_NAME,)
                del parts
                if info.orig_filename != info.filename:
                    raise ValueError("capsule_zip_member_name_invalid")
                mode = info.external_attr >> 16
                member_limit = MAX_HEADER_BYTES if info.filename == MANIFEST_NAME else MAX_SOURCE_FILE_BYTES
                if (info.is_dir() or stat.S_IFMT(mode) != stat.S_IFREG
                        or stat.S_IMODE(mode) != 0o600 or info.flag_bits & 0x1
                        or info.flag_bits & ~0x800
                        or info.compress_type != zipfile.ZIP_STORED or info.extra
                        or info.comment or info.file_size > member_limit
                        or info.file_size != info.compress_size):
                    raise ValueError("capsule_zip_member_metadata_invalid")
            manifest_bytes = archive.read(infos[0])
            if len(manifest_bytes) > MAX_HEADER_BYTES:
                raise ValueError("capsule_manifest_too_large")
            manifest = json.loads(manifest_bytes.decode("utf-8"),
                                  object_pairs_hook=_unique_object)
            if (_canonical(manifest) != manifest_bytes or not isinstance(manifest, dict)
                    or set(manifest) != {"format", "files"}
                    or manifest["format"] != FORMAT or not isinstance(manifest["files"], list)):
                raise ValueError("capsule_manifest_invalid")
            entries = manifest["files"]
            listed = []
            total_manifest_size = 0
            for entry in entries:
                if (not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}
                        or type(entry["size"]) is not int or not 0 <= entry["size"] <= MAX_SOURCE_FILE_BYTES
                        or not isinstance(entry["sha256"], str)
                        or len(entry["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in entry["sha256"])):
                    raise ValueError("capsule_manifest_entry_invalid")
                listed.append(entry["path"])
                total_manifest_size += entry["size"]
            if total_manifest_size > MAX_CAPSULE_BYTES:
                raise ValueError("capsule_manifest_total_size_invalid")
            if _validate_paths(listed) != expected or names != [MANIFEST_NAME, *expected]:
                raise ValueError("capsule_manifest_mismatch")
            recovered = {}
            for entry, info in zip(entries, infos[1:], strict=True):
                data = archive.read(info)
                if (len(data) != entry["size"]
                        or hashlib.sha256(data).hexdigest() != entry["sha256"]):
                    raise ValueError("capsule_source_hash_mismatch")
                recovered[entry["path"]] = data
    except (zipfile.BadZipFile, RuntimeError, UnicodeError, json.JSONDecodeError,
            KeyError, TypeError, OverflowError):
        raise ValueError("capsule_archive_invalid") from None
    return {"status": "verified", "format": FORMAT,
            "ordered_paths": list(expected),
            "file_sha256": [hashlib.sha256(recovered[path]).hexdigest()
                            for path in expected],
            "archive_sha256": archive_sha256,
            "capsule_size": len(raw), "mode": "0600", "files": recovered,
            "provenance_authenticated": False}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("capsule_duplicate_json_key")
        result[key] = value
    return result
