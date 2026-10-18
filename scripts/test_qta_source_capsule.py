from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from qta_source_capsule import capture_source_capsule, recover_source_capsule


class QtaSourceCapsuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.private = self.base / "private"
        self.root.mkdir(mode=0o700)
        self.private.mkdir(mode=0o700)
        (self.root / "scripts").mkdir()
        (self.root / "docs").mkdir()
        self.sources = {
            "scripts/runner.py": b"print('synthetic')\n",
            "docs/protocol.md": b"synthetic protocol bytes\n",
        }
        for relative, content in self.sources.items():
            path = self.root / relative
            path.write_bytes(content)
            path.chmod(0o644)
        self.paths = tuple(self.sources)
        self.capsule = self.private / "source-001.qta.zip"

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, output=None, paths=None):
        return capture_source_capsule(source_root=self.root,
            ordered_paths=self.paths if paths is None else paths,
            output_path=self.capsule if output is None else output)

    def test_round_trip_binds_ordered_paths_hashes_and_private_modes(self):
        created = self.capture()
        self.assertEqual(created["ordered_paths"], list(self.paths))
        self.assertEqual(created["file_sha256"], [
            hashlib.sha256(self.sources[path]).hexdigest() for path in self.paths])
        self.assertEqual(self.capsule.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.private.stat().st_mode & 0o777, 0o700)
        self.assertEqual(created["archive_sha256"],
                         hashlib.sha256(self.capsule.read_bytes()).hexdigest())
        recovered = recover_source_capsule(capsule_path=self.capsule,
                                           expected_paths=self.paths,
                                           expected_archive_sha256=created["archive_sha256"])
        self.assertEqual(recovered["files"], self.sources)
        self.assertEqual(recovered["archive_sha256"], created["archive_sha256"])
        self.assertFalse(recovered["provenance_authenticated"])

    def test_recovery_returns_captured_bytes_after_source_files_change(self):
        self.capture()
        (self.root / self.paths[0]).write_bytes(b"changed after capture")
        recovered = recover_source_capsule(capsule_path=self.capsule,
                                           expected_paths=self.paths)
        self.assertEqual(recovered["files"][self.paths[0]], self.sources[self.paths[0]])

    def test_rejects_duplicate_missing_traversal_and_extra_expected_paths(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.capture(paths=(self.paths[0], self.paths[0]))
        with self.assertRaises(FileNotFoundError):
            self.capture(paths=("scripts/missing.py",))
        with self.assertRaisesRegex(ValueError, "relative_path"):
            self.capture(paths=("../escape.py",))
        with self.assertRaisesRegex(ValueError, "relative_path"):
            self.capture(paths=("C:/outside.py",))
        with self.assertRaisesRegex(ValueError, "relative_path"):
            self.capture(paths=("scripts/e\u0301.py",))
        with self.assertRaisesRegex(ValueError, "casefold"):
            self.capture(paths=("scripts/Readme.py", "scripts/README.py"))
        self.capture()
        with self.assertRaisesRegex(ValueError, "member_set|manifest_mismatch|member_count_mismatch"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=(*self.paths, "extra.py"))
        with self.assertRaisesRegex(ValueError, "manifest_mismatch"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=tuple(reversed(self.paths)))

    def test_rejects_source_symlink(self):
        path = self.root / self.paths[0]
        path.unlink()
        path.symlink_to(self.base / "outside.py")
        (self.base / "outside.py").write_bytes(b"outside")
        with self.assertRaisesRegex(ValueError, "source_file_invalid"):
            self.capture()

    def test_rejects_source_hardlink(self):
        alias = self.base / "source-alias.py"
        os.link(self.root / self.paths[0], alias)
        with self.assertRaisesRegex(ValueError, "source_file_invalid"):
            self.capture(paths=(self.paths[0],))

    def test_rejects_unsafe_source_permissions(self):
        (self.root / self.paths[0]).chmod(0o666)
        with self.assertRaisesRegex(ValueError, "source_permissions"):
            self.capture()

    def test_refuses_overwrite_and_requires_owner_only_parent(self):
        self.capture()
        with self.assertRaises(FileExistsError):
            self.capture()
        other = self.base / "public"
        other.mkdir(mode=0o700)
        other.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "directory_permissions"):
            self.capture(output=other / "capsule.zip")

    def test_refuses_output_symlink_and_does_not_replace_racing_final(self):
        target = self.base / "target"
        target.write_bytes(b"sentinel")
        self.capsule.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertTrue(self.capsule.is_symlink())
        self.assertEqual(target.read_bytes(), b"sentinel")
        self.capsule.unlink()

        real_link = os.link

        def racing_link(src, dst, *, src_dir_fd=None, dst_dir_fd=None,
                        follow_symlinks=True):
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                         dir_fd=dst_dir_fd)
            try:
                os.write(fd, b"winner")
            finally:
                os.close(fd)
            return real_link(src, dst, src_dir_fd=src_dir_fd,
                             dst_dir_fd=dst_dir_fd,
                             follow_symlinks=follow_symlinks)

        with patch("qta_source_capsule.os.link", side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                self.capture()
        self.assertEqual(self.capsule.read_bytes(), b"winner")
        self.assertEqual(sorted(p.name for p in self.private.iterdir()),
                         [self.capsule.name])

    def test_write_failure_leaves_no_final_or_temp_file(self):
        def fail_capsule_write(fd, data):
            raise OSError("injected_write_failure")

        with patch("qta_source_capsule.os.write", side_effect=fail_capsule_write):
            with self.assertRaisesRegex(OSError, "injected_write_failure"):
                self.capture()
        self.assertFalse(self.capsule.exists())
        self.assertEqual(list(self.private.iterdir()), [])

    def test_recovery_rejects_symlink_capsule_parent(self):
        self.capture()
        alias = self.base / "private-alias"
        alias.symlink_to(self.private, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory_component_invalid"):
            recover_source_capsule(capsule_path=alias / self.capsule.name,
                                   expected_paths=self.paths)

    def test_recovery_rejects_non_private_capsule_parent(self):
        self.capture()
        self.private.chmod(0o750)
        with self.assertRaisesRegex(ValueError, "directory_permissions"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

    def test_rejects_capsule_with_bad_file_permissions(self):
        self.capture()
        self.capsule.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "file_permissions"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

    def _replace_zip(self, replace):
        original = zipfile.ZipFile(self.capsule, "r")
        target = io.BytesIO()
        with original, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as out:
            for info in original.infolist():
                if info.filename in replace:
                    new_info, data = replace[info.filename]
                    out.writestr(new_info, data)
                else:
                    out.writestr(info, original.read(info))
            if "__extra__" in replace:
                new_info, data = replace["__extra__"]
                out.writestr(new_info, data)
            if "__duplicate__" in replace:
                new_info, data = replace["__duplicate__"]
                out.writestr(new_info, data)
        self.capsule.write_bytes(target.getvalue())
        self.capsule.chmod(0o600)

    def test_rejects_tampered_payload(self):
        self.capture()
        info = zipfile.ZipInfo(self.paths[0])
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        self._replace_zip({self.paths[0]: (info, b"tampered source bytes")})
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)
        tampered_hash = hashlib.sha256(self.capsule.read_bytes()).hexdigest()
        self.capture(output=self.private / "tamper-check.zip")
        with self.assertRaisesRegex(ValueError, "archive_hash_mismatch"):
            recover_source_capsule(capsule_path=self.private / "tamper-check.zip",
                expected_paths=self.paths, expected_archive_sha256=tampered_hash)

    def test_rejects_corrupt_zip_end_record(self):
        self.capture()
        raw = bytearray(self.capsule.read_bytes())
        raw[-1] ^= 0x01
        self.capsule.write_bytes(raw)
        with self.assertRaises(ValueError):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

    def test_rejects_extra_traversal_and_symlink_zip_members(self):
        self.capture()
        extra = zipfile.ZipInfo("extra.py")
        extra.create_system = 3
        extra.external_attr = (stat.S_IFREG | 0o600) << 16
        self._replace_zip({"__extra__": (extra, b"extra")})
        with self.assertRaisesRegex(ValueError, "member_set|member_count_mismatch"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

    def test_rejects_duplicate_zip_member_names(self):
        self.capture()
        duplicate = zipfile.ZipInfo(self.paths[0])
        duplicate.create_system = 3
        duplicate.external_attr = (stat.S_IFREG | 0o600) << 16
        with self.assertWarns(UserWarning):
            self._replace_zip({"__duplicate__": (duplicate, b"duplicate")})
        with self.assertRaisesRegex(ValueError, "member_set|member_count_mismatch"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

    def test_rejects_many_declared_zip_members_before_infolist(self):
        self.capture()
        rewritten = io.BytesIO()
        with zipfile.ZipFile(self.capsule, "r") as src, zipfile.ZipFile(
                rewritten, "w", compression=zipfile.ZIP_STORED) as dst:
            for info in src.infolist():
                dst.writestr(info, src.read(info))
            for index in range(128):
                info = zipfile.ZipInfo(f"extra/{index:03d}.py")
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                dst.writestr(info, b"x")
        self.capsule.write_bytes(rewritten.getvalue())
        self.capsule.chmod(0o600)
        with patch("qta_source_capsule.zipfile.ZipFile.infolist",
                   side_effect=AssertionError("infolist must not be called")) as listing:
            with self.assertRaisesRegex(ValueError, "member_count_mismatch"):
                recover_source_capsule(capsule_path=self.capsule,
                                       expected_paths=self.paths)
            listing.assert_not_called()

    def test_rejects_forged_low_eocd_count_before_infolist(self):
        self.capture()
        rewritten = io.BytesIO()
        with zipfile.ZipFile(self.capsule, "r") as src, zipfile.ZipFile(
                rewritten, "w", compression=zipfile.ZIP_STORED) as dst:
            for info in src.infolist():
                dst.writestr(info, src.read(info))
            extra = zipfile.ZipInfo("hidden-extra.py")
            extra.create_system = 3
            extra.external_attr = (stat.S_IFREG | 0o600) << 16
            dst.writestr(extra, b"x")
        raw = bytearray(rewritten.getvalue())
        eocd = raw.rfind(b"PK\x05\x06")
        declared = len(self.paths) + 1
        raw[eocd + 8:eocd + 10] = declared.to_bytes(2, "little")
        raw[eocd + 10:eocd + 12] = declared.to_bytes(2, "little")
        self.capsule.write_bytes(raw)
        self.capsule.chmod(0o600)
        with patch("qta_source_capsule.zipfile.ZipFile.infolist",
                   side_effect=AssertionError("infolist must not be called")) as listing:
            with self.assertRaisesRegex(ValueError, "member_count_mismatch"):
                recover_source_capsule(capsule_path=self.capsule,
                                       expected_paths=self.paths)
            listing.assert_not_called()

    def test_rejects_forged_huge_member_and_manifest_sizes(self):
        self.capture()
        raw = bytearray(self.capsule.read_bytes())
        offset = 0
        patched = False
        while True:
            offset = raw.find(b"PK\x01\x02", offset)
            if offset < 0:
                break
            name_len = int.from_bytes(raw[offset + 28:offset + 30], "little")
            name = bytes(raw[offset + 46:offset + 46 + name_len]).decode("utf-8")
            if name == self.paths[0]:
                raw[offset + 24:offset + 28] = ((32 << 20) + 1).to_bytes(4, "little")
                patched = True
                break
            offset += 46 + name_len
        self.assertTrue(patched)
        self.capsule.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "member_metadata"):
            recover_source_capsule(capsule_path=self.capsule,
                                   expected_paths=self.paths)

        self.capture(output=self.private / "huge-manifest.zip")
        case = self.private / "huge-manifest.zip"
        with zipfile.ZipFile(case, "r") as src:
            manifest = __import__("json").loads(src.read("manifest.json"))
            members = {info.filename: src.read(info) for info in src.infolist()
                       if info.filename != "manifest.json"}
        manifest["files"][0]["size"] = (32 << 20) + 1
        replacement = io.BytesIO()
        with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_STORED) as dst:
            info = zipfile.ZipInfo("manifest.json")
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            dst.writestr(info, __import__("json").dumps(
                manifest, sort_keys=True, separators=(",", ":")).encode())
            for path, data in members.items():
                info = zipfile.ZipInfo(path)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                dst.writestr(info, data)
        case.write_bytes(replacement.getvalue())
        case.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "manifest_entry_invalid"):
            recover_source_capsule(capsule_path=case, expected_paths=self.paths)

    def test_rejects_traversal_and_symlink_zip_member_metadata(self):
        for member_name, member_mode in (("../escape.py", stat.S_IFREG | 0o600),
                                         (self.paths[0], stat.S_IFLNK | 0o777),
                                         (self.paths[0], stat.S_IFIFO | 0o600)):
            self.capture(output=self.private / f"case-{member_mode}.zip")
            case = self.private / f"case-{member_mode}.zip"
            rewritten = io.BytesIO()
            with zipfile.ZipFile(case, "r") as src, zipfile.ZipFile(
                    rewritten, "w", compression=zipfile.ZIP_STORED) as dst:
                for info in src.infolist():
                    data = src.read(info)
                    if info.filename == self.paths[0]:
                        replacement = zipfile.ZipInfo(member_name)
                        replacement.create_system = 3
                        replacement.external_attr = member_mode << 16
                        dst.writestr(replacement, data)
                    else:
                        dst.writestr(info, data)
            case.write_bytes(rewritten.getvalue())
            case.chmod(0o600)
            with self.assertRaises(ValueError):
                recover_source_capsule(capsule_path=case, expected_paths=self.paths)


if __name__ == "__main__":
    unittest.main()
