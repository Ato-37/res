"""Synthetic integrity tests for exact 3 x 3 batch digest manifests."""
from __future__ import annotations

import copy
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from nextgqa_batch_digest_freeze import (
    CONDITIONS,
    build_batch_digest_manifest,
    verify_batch_digest_manifest,
)


def make_triplets():
    rows = []
    for index in range(3):
        rows.append({
            "gold_index": index,
            "batches": {
                condition: {
                    "input_ids": torch.tensor([[index, 2, 3]], dtype=torch.int64),
                    "attention_mask": torch.ones((1, 3), dtype=torch.int64),
                    "nested": {"pixel_values_videos": torch.full(
                        (2, 2), float(index), dtype=torch.bfloat16)},
                    "video_grid_thw": [torch.tensor([1, 2, 2], dtype=torch.int32)],
                    # Qwen2_5_VLProcessor's VideoMetadata-like object contains
                    # primitive/list fields that are intentionally not tensors.
                    "video_metadata": {
                        "fps": 30.0, "frames_indices": [0, 7, 15],
                        "duration": 1.0, "stream_index": 0,
                        "is_audio": False, "optional": None,
                        "label": "video",
                    },
                }
                for condition in CONDITIONS
            },
        })
    return rows


@unittest.skipIf(torch is None, "torch unavailable")
class NextgqaBatchDigestFreezeTests(unittest.TestCase):
    def test_manifest_rehashes_same_nested_cpu_and_cuda_independent_batches(self):
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        self.assertEqual(manifest["status"], "unvalidated_digest_manifest_only")
        self.assertEqual(manifest["execution_authorization"], "not_granted")
        self.assertEqual(len(manifest["records"]), 3)
        self.assertEqual(len(manifest["records"][0]["conditions"]), 3)
        self.assertTrue(verify_batch_digest_manifest(rows, manifest))

    def test_one_tensor_byte_change_fails_rehash(self):
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        rows[1]["batches"]["claim_evidence_black"]["input_ids"][0, 1] = 99
        with self.assertRaisesRegex(ValueError, "batch_digest_manifest_mismatch"):
            verify_batch_digest_manifest(rows, manifest)

    def test_dtype_shape_and_gold_are_bound(self):
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        rows[0]["batches"]["full"]["input_ids"] = rows[0]["batches"]["full"]["input_ids"].to(torch.int32)
        with self.assertRaisesRegex(ValueError, "batch_digest_manifest_mismatch"):
            verify_batch_digest_manifest(rows, manifest)

    def test_video_metadata_primitives_are_typed_hashed_and_rechecked(self):
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        self.assertTrue(verify_batch_digest_manifest(rows, manifest))
        metadata = (manifest["records"][0]["conditions"]["full"]["items"]
                    ["video_metadata"]["items"])
        self.assertEqual(metadata["fps"], {"type": "float64_hex", "value": float(30).hex()})
        self.assertEqual(metadata["stream_index"], {"type": "int", "value": "0"})
        self.assertEqual(metadata["is_audio"], {"type": "bool", "value": False})
        self.assertEqual(metadata["optional"], {"type": "none"})
        self.assertEqual(metadata["label"], {"type": "str", "value": "video"})
        rows[0]["batches"]["full"]["video_metadata"]["fps"] = 29.97
        with self.assertRaisesRegex(ValueError, "batch_digest_manifest_mismatch"):
            verify_batch_digest_manifest(rows, manifest)
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        rows[0]["gold_index"] = 4
        with self.assertRaisesRegex(ValueError, "batch_digest_manifest_mismatch"):
            verify_batch_digest_manifest(rows, manifest)

    def test_wrong_cardinality_condition_or_keys_rejected(self):
        rows = make_triplets()
        with self.assertRaisesRegex(ValueError, "requested_record_count_mismatch"):
            build_batch_digest_manifest(rows[:2])
        rows = make_triplets()
        del rows[0]["batches"]["temporal_control_black"]
        with self.assertRaisesRegex(ValueError, "condition_set_invalid"):
            build_batch_digest_manifest(rows)
        rows = make_triplets()
        rows[2]["batches"]["full"].pop("attention_mask")
        with self.assertRaisesRegex(ValueError, "batch_keys_mismatch"):
            build_batch_digest_manifest(rows)

    def test_nonfinite_and_unsupported_nested_values_rejected(self):
        rows = make_triplets()
        rows[0]["batches"]["full"]["nested"]["pixel_values_videos"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite_tensor"):
            build_batch_digest_manifest(rows)
        rows = make_triplets()
        for row in rows:
            for condition in CONDITIONS:
                row["batches"][condition]["unsafe"] = object()
        with self.assertRaisesRegex(ValueError, "unsupported_batch_value"):
            build_batch_digest_manifest(rows)
        rows = make_triplets()
        rows[0]["batches"]["full"]["video_metadata"]["fps"] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite_primitive"):
            build_batch_digest_manifest(rows)

    def test_tampered_manifest_metadata_rejected(self):
        rows = make_triplets()
        manifest = build_batch_digest_manifest(rows)
        altered = copy.deepcopy(manifest)
        altered["source_authentication"] = "passed"
        with self.assertRaisesRegex(ValueError, "batch_digest_manifest_mismatch"):
            verify_batch_digest_manifest(rows, altered)


if __name__ == "__main__":
    unittest.main()
