"""Small synthetic tests for checkpoint-byte provenance checks; no model weights."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_qwen_checkpoint_bytes_cpu as audit


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


class CheckpointByteAuditTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        cache = root / "hf-cache" / "hub"
        model_cache = cache / "models--Qwen--Qwen2.5-VL-7B-Instruct"
        snapshot = model_cache / "snapshots" / audit.REVISION
        blobs = cache / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        (model_cache / "blobs").symlink_to(Path("../blobs"), target_is_directory=True)
        shard_bytes = {
            name: f"synthetic-shard-{index}".encode()
            for index, name in enumerate(audit.SHARD_NAMES)
        }
        siblings = []
        for name, data in shard_bytes.items():
            (blobs / sha256(data)).write_bytes(data)
            (snapshot / name).symlink_to(Path("../../blobs") / sha256(data))
            pointer = audit._lfs_pointer_bytes(sha256(data), len(data))
            siblings.append({
                "rfilename": name,
                "size": len(data),
                "blobId": git_blob(pointer),
                "lfs": {"sha256": sha256(data), "size": len(data), "pointerSize": len(pointer)},
            })
        index_data = json.dumps({
            "metadata": {"total_size": sum(map(len, shard_bytes.values()))},
            "weight_map": {f"tensor.{i}": name for i, name in enumerate(audit.SHARD_NAMES)},
        }, sort_keys=True, separators=(",", ":")).encode()
        (blobs / git_blob(index_data)).write_bytes(index_data)
        (snapshot / audit.INDEX_NAME).symlink_to(Path("../../blobs") / git_blob(index_data))
        siblings.append({
            "rfilename": audit.INDEX_NAME,
            "size": len(index_data),
            "blobId": git_blob(index_data),
        })
        metadata = {
            "id": audit.MODEL_ID,
            "modelId": audit.MODEL_ID,
            "sha": audit.REVISION,
            "siblings": siblings,
        }
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        return snapshot, metadata, metadata_bytes

    def run_audit(self, snapshot: Path, metadata_bytes: bytes, *, entries=5, tensor_bytes=85):
        return audit.audit_checkpoint(
            metadata_bytes=metadata_bytes,
            snapshot_dir=snapshot,
            expected_metadata_sha256=sha256(metadata_bytes),
            expected_index_entries=entries,
            expected_index_tensor_bytes=tensor_bytes,
        )

    def test_valid_synthetic_checkpoint_checks_sha256_and_git_blob_sha1(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["byte_authentication_status"], "verified_current_local_bytes")
        self.assertEqual(report["effective_shards"], 5)
        self.assertTrue(all(x["sha256_matches"] for x in report["shards"]))
        self.assertTrue(all(x["lfs_pointer_git_blob_sha1_matches"] for x in report["shards"]))
        self.assertEqual(report["historical_gpu_runs_byte_verified"], False)
        self.assertNotIn(str(directory), json.dumps(report))

    def test_pinned_metadata_hash_mismatch_fails_before_shard_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            report = audit.audit_checkpoint(
                metadata_bytes=metadata_bytes,
                snapshot_dir=snapshot,
                expected_metadata_sha256="0" * 64,
                expected_index_entries=5,
                expected_index_tensor_bytes=100,
            )
        self.assertEqual(report["failure_codes"], ["api_metadata_sha256_mismatch"])
        self.assertEqual(report["effective_shards"], 0)
        self.assertEqual(report["bytes_read"], 0)

    def test_wrong_revision_and_wrong_model_id_fail_closed(self):
        for key, bad_value, code in (
            ("sha", "0" * 40, "api_revision_mismatch"),
            ("id", "someone/else", "api_model_id_mismatch"),
        ):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as directory:
                snapshot, metadata, _ = self.make_fixture(Path(directory))
                metadata[key] = bad_value
                metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
                report = self.run_audit(snapshot, metadata_bytes)
            self.assertEqual(report["failure_codes"], [code])
            self.assertEqual(report["bytes_read"], 0)

    def test_shard_content_mismatch_is_reported_without_hash_disclosure(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            target = Path(os.path.realpath(snapshot / audit.SHARD_NAMES[0]))
            target.write_bytes(b"Xynthetic-shard-0")
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertIn("model-00001-of-00005.safetensors:shard_sha256_mismatch", report["failure_codes"])
        self.assertEqual(report["byte_authentication_status"], "not_verified")
        self.assertTrue(report["shards"][0]["lfs_pointer_git_blob_sha1_matches"])
        self.assertFalse(report["shards"][0]["sha256_matches"])
        self.assertNotIn("Xynthetic-shard-0", json.dumps(report))

    def test_metadata_blob_id_mismatch_is_distinct_from_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, _ = self.make_fixture(Path(directory))
            metadata["siblings"][0]["blobId"] = "0" * 40
            shard_link = snapshot / audit.SHARD_NAMES[0]
            shard_link.unlink()
            shard_link.symlink_to(Path("../../blobs") / sha256(b"synthetic-shard-0"))
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["api_lfs_pointer_git_blob_sha1_mismatch"])
        self.assertEqual(report["effective_shards"], 0)

    def test_index_hash_count_tensor_bytes_and_shard_names_are_checked(self):
        mutations = (
            ("blobId", "0" * 40, "index_git_blob_sha1_mismatch"),
        )
        for key, value, code in mutations:
            with self.subTest(index_field=key, failure=code), tempfile.TemporaryDirectory() as directory:
                snapshot, metadata, _ = self.make_fixture(Path(directory))
                index_sibling = next(x for x in metadata["siblings"] if x["rfilename"] == audit.INDEX_NAME)
                index_sibling[key] = value
                index_link = snapshot / audit.INDEX_NAME
                source_index_blob = Path(os.path.realpath(index_link))
                source_index_bytes = source_index_blob.read_bytes()
                index_link.unlink()
                (source_index_blob.parent / value).write_bytes(source_index_bytes)
                index_link.symlink_to(Path("../../blobs") / value)
                metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
                report = self.run_audit(snapshot, metadata_bytes)
            self.assertEqual(report["failure_codes"], [code])
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            report = self.run_audit(snapshot, metadata_bytes, entries=4)
        self.assertEqual(report["failure_codes"], ["index_weight_map_count_mismatch"])
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            report = self.run_audit(snapshot, metadata_bytes, tensor_bytes=99)
        self.assertEqual(report["failure_codes"], ["index_tensor_bytes_mismatch"])
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            index_path = Path(os.path.realpath(snapshot / audit.INDEX_NAME))
            index = json.loads(index_path.read_text())
            index["weight_map"]["tensor.0"] = "../outside.safetensors"
            new_data = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
            index_path.write_bytes(new_data)
            index_link = snapshot / audit.INDEX_NAME
            index_link.unlink()
            new_blob_id = git_blob(new_data)
            (index_path.parent / new_blob_id).write_bytes(new_data)
            index_link.symlink_to(Path("../../blobs") / new_blob_id)
            metadata = json.loads(metadata_bytes)
            index_metadata = next(x for x in metadata["siblings"] if x["rfilename"] == audit.INDEX_NAME)
            index_metadata["blobId"] = new_blob_id
            index_metadata["size"] = len(new_data)
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["index_shard_name_invalid"])

    def test_duplicate_or_missing_api_expected_shards_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, _ = self.make_fixture(Path(directory))
            metadata["siblings"].append(copy.deepcopy(metadata["siblings"][0]))
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["api_duplicate_or_invalid_expected_file"])
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, _ = self.make_fixture(Path(directory))
            metadata["siblings"] = [
                x for x in metadata["siblings"] if x["rfilename"] != audit.SHARD_NAMES[-1]
            ]
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["api_expected_file_set_mismatch"])

    def test_boolean_size_and_outside_cache_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, _ = self.make_fixture(Path(directory))
            metadata["siblings"][0]["size"] = True
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["api_file_size_invalid"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, _, metadata_bytes = self.make_fixture(root)
            name = audit.SHARD_NAMES[0]
            data = b"synthetic-shard-0"
            outside = root / "outside" / sha256(data)
            outside.parent.mkdir()
            outside.write_bytes(data)
            link = snapshot / name
            link.unlink()
            link.symlink_to(outside)
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertIn(f"{audit.SHARD_NAMES[0]}:cache_target_outside_blob_store", report["failure_codes"])
        self.assertEqual(report["effective_shards"], 4)
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, _ = self.make_fixture(Path(directory))
            metadata["siblings"][0]["lfs"]["pointerSize"] = True
            metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertEqual(report["failure_codes"], ["api_lfs_pointer_size_invalid"])

    def test_symlink_basename_must_match_expected_metadata_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            original = snapshot / audit.SHARD_NAMES[0]
            original.unlink()
            original.symlink_to(Path("../../blobs") / ("0" * 64))
            report = self.run_audit(snapshot, metadata_bytes)
        self.assertIn(
            "model-00001-of-00005.safetensors:cache_link_mismatch",
            report["failure_codes"],
        )

    def test_stat_change_during_stream_fails_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, metadata_bytes = self.make_fixture(Path(directory))
            path = snapshot / audit.SHARD_NAMES[0]
            real_stat = path.stat()
            with mock.patch.object(audit, "_same_stat", side_effect=[True, False]):
                result = audit._stream_verify_shard(snapshot, path, {
                    "size": real_stat.st_size,
                    "lfs": {"sha256": sha256(path.read_bytes())},
                    "blobId": git_blob(audit._lfs_pointer_bytes(
                        sha256(path.read_bytes()), real_stat.st_size
                    )),
                })
        self.assertFalse(result["stable_during_read"])
        self.assertEqual(result["failure_code"], "shard_changed_during_read")

    def test_post_read_nonregular_stat_failure_preserves_partial_byte_count(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, _ = self.make_fixture(Path(directory))
            path = snapshot / audit.SHARD_NAMES[0]
            data = path.read_bytes()
            real_stat = path.stat()
            expected = {
                "size": len(data),
                "lfs": {"sha256": sha256(data)},
                "blobId": git_blob(audit._lfs_pointer_bytes(sha256(data), len(data))),
            }
            with mock.patch.object(
                audit, "_stable_stat", side_effect=[real_stat, ValueError("not_regular_file")]
            ):
                result = audit._stream_verify_shard(snapshot, path, expected)
        self.assertEqual(result["bytes_read"], len(data))
        self.assertEqual(result["failure_code"], "shard_changed_during_read")
        self.assertFalse(result["stable_during_read"])
        self.assertTrue(result["sha256_matches"])

    def test_exclusive_output_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"schema_version": 1}
            output = audit.write_exclusive(Path("results/audit.json"), root, report)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                audit.write_exclusive(Path("results/audit.json"), root, report)
            with self.assertRaises(ValueError):
                audit.write_exclusive(Path("outside/audit.json"), root, report)
            with self.assertRaises(ValueError):
                audit.validate_output_path(Path("results/nested/audit.json"), root)


if __name__ == "__main__":
    unittest.main()
