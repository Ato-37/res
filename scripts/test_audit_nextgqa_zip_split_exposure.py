import csv
import hashlib
import json
import stat
import struct
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from audit_nextgqa_local_media_byte_diversity import _media_token
from audit_nextgqa_zip_split_exposure import audit_zip_split


class ZipSplitExposureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "split.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("video_id",))
            writer.writeheader()
            writer.writerows(({"video_id": "video-a"}, {"video_id": "video-b"}))
        self.map_path = self.root / "map.json"
        self.map_path.write_text(json.dumps({
            "video-a": "folder/a",
            "video-b": "folder/b",
        }), encoding="utf-8")
        self.zip_path = self.root / "videos.zip"
        self.bytes_a = b"small synthetic video a"
        self.bytes_b = b"small synthetic video b"
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            archive.writestr("NExTVideo/folder/a.mp4", self.bytes_a)
            archive.writestr("NExTVideo/folder/b.mp4", self.bytes_b)
        self.key = b"k" * 32
        self.key_path = self.root / "key.bin"
        self.key_path.write_bytes(self.key)
        self.key_path.chmod(0o600)
        self.index_path = self.root / "index.json"
        self.write_index([self.bytes_a])

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self, media_bytes):
        payload = {
            "schema_version": 1,
            "purpose": "outcome_blind_prior_exposure_index",
            "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
            "identities": {
                "media_sha256_hmac_sha256": [
                    _media_token(self.key, hashlib.sha256(item).hexdigest())
                    for item in media_bytes
                ]
            },
        }
        self.index_path.write_text(json.dumps(payload), encoding="utf-8")
        self.index_path.chmod(0o600)

    def audit(self, **overrides):
        args = dict(
            split_csv=self.csv_path,
            video_map_json=self.map_path,
            zip_path=self.zip_path,
            prior_indices=[self.index_path],
            hmac_key=self.key_path,
        )
        args.update(overrides)
        return audit_zip_split(**args)

    def test_streams_mapped_members_and_matches_only_aggregate_hash(self):
        result = self.audit()
        self.assertEqual(result["split_video_id_count"], 2)
        self.assertEqual(result["mapped_video_members_found"], 2)
        self.assertEqual(result["mapped_video_members_missing"], 0)
        self.assertEqual(result["unique_media_byte_hashes"], 2)
        self.assertEqual(result["distinct_hashes_matching_each_prior_index"], [1])
        self.assertFalse(result["media_extracted_or_emitted"])
        self.assertNotIn("video-a", json.dumps(result))
        self.assertNotIn("folder/a", json.dumps(result))

    def test_rejects_unsafe_mapped_path(self):
        self.map_path.write_text(json.dumps({
            "video-a": "../escape",
            "video-b": "folder/b",
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "video_map_path_invalid"):
            self.audit()

    def test_rejects_noncanonical_map_path_segments(self):
        self.map_path.write_text(json.dumps({
            "video-a": "./folder/a",
            "video-b": "folder/b",
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "video_map_path_invalid"):
            self.audit()

    def test_counts_duplicate_archive_names_as_ambiguous(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.zip_path, "a") as archive:
                archive.writestr("NExTVideo/folder/a.mp4", b"second copy")
        result = self.audit()
        self.assertEqual(result["mapped_video_members_found"], 1)
        self.assertEqual(result["mapped_video_members_ambiguous"], 1)

    def test_reports_distinct_matches_for_each_prior_index(self):
        second_index = self.root / "second-index.json"
        second_payload = {
            "schema_version": 1,
            "purpose": "outcome_blind_prior_exposure_index",
            "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
            "identities": {
                "media_sha256_hmac_sha256": [
                    _media_token(self.key, hashlib.sha256(self.bytes_b).hexdigest())
                ]
            },
        }
        second_index.write_text(json.dumps(second_payload), encoding="utf-8")
        second_index.chmod(0o600)
        result = self.audit(prior_indices=[self.index_path, second_index])
        self.assertEqual(result["distinct_hashes_matching_each_prior_index"], [1, 1])
        self.assertEqual(result["distinct_hashes_matching_prior_index_union"], 2)

    def test_rejects_two_ids_aliasing_one_zip_member(self):
        self.map_path.write_text(json.dumps({
            "video-a": "folder/a",
            "video-b": "folder/a",
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "video_map_member_alias"):
            self.audit()

    def test_crc_failure_is_a_generic_member_read_error(self):
        with zipfile.ZipFile(self.zip_path) as archive:
            info = archive.getinfo("NExTVideo/folder/a.mp4")
        with self.zip_path.open("r+b") as stream:
            stream.seek(info.header_offset + 26)
            filename_length, extra_length = struct.unpack("<HH", stream.read(4))
            data_offset = info.header_offset + 30 + filename_length + extra_length
            stream.seek(data_offset)
            first_byte = stream.read(1)
            stream.seek(data_offset)
            stream.write(bytes([first_byte[0] ^ 0x01]))
        with self.assertRaisesRegex(ValueError, "zip_member_read_failed"):
            self.audit()

    def test_enforces_archive_member_count_before_loading_directory(self):
        with patch("audit_nextgqa_zip_split_exposure.MAX_ARCHIVE_MEMBERS", 1):
            with self.assertRaisesRegex(ValueError, "zip_member_count_limit_exceeded"):
                self.audit()

    def test_enforces_archive_file_size_before_loading_directory(self):
        with patch("audit_nextgqa_zip_split_exposure.MAX_ARCHIVE_FILE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "zip_file_invalid"):
                self.audit()

    def test_enforces_central_directory_size_before_loading_directory(self):
        with patch("audit_nextgqa_zip_split_exposure.MAX_CENTRAL_DIRECTORY_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "zip_central_directory_limit_exceeded"):
                self.audit()

    def test_rejects_zip_symlink_members(self):
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            link = zipfile.ZipInfo("NExTVideo/folder/a.mp4")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, b"target")
            archive.writestr("NExTVideo/folder/b.mp4", self.bytes_b)
        with self.assertRaisesRegex(ValueError, "zip_member_invalid"):
            self.audit()

    def test_rejects_symlink_archive_path(self):
        link_path = self.root / "linked.zip"
        link_path.symlink_to(self.zip_path)
        with self.assertRaisesRegex(ValueError, "zip_input_unavailable"):
            self.audit(zip_path=link_path)


if __name__ == "__main__":
    unittest.main()
