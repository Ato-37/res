from __future__ import annotations

import unittest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from run_saved_claim_frame_permutation_attention_smoke import (
    compute_bias_frame_order_interaction,
    deterministic_reverse_block_indices,
    reorder_frames,
    validate_video_metadata,
    _read_packet,
    validate_biased_mask_rows,
)
from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX, PROMPT_SUFFIX_BY_VERSION
import torch


class SavedClaimFramePermutationSmokeTests(unittest.TestCase):
    def test_claim_prompt_version_uses_recorded_suffix_not_legacy_default(self):
        row = {"generation_prompt_version": "v2_short"}
        version, suffix = resolve_claim_prompt_suffix(row, PROMPT_SUFFIX_BY_VERSION)
        self.assertEqual(version, "v2_short")
        self.assertEqual(suffix, PROMPT_SUFFIX_BY_VERSION["v2_short"])
        self.assertNotEqual(suffix, PROMPT_SUFFIX)
        with self.assertRaisesRegex(ValueError, "prompt_version_invalid"):
            resolve_claim_prompt_suffix({"generation_prompt_version": "future"},
                                        PROMPT_SUFFIX_BY_VERSION)

    def test_reverse_blocks_preserves_within_block_temporal_patch_order(self):
        self.assertEqual(deterministic_reverse_block_indices(8, 2),
                         (6, 7, 4, 5, 2, 3, 0, 1))
        self.assertEqual(deterministic_reverse_block_indices(6, 3),
                         (3, 4, 5, 0, 1, 2))

    def test_rejects_invalid_block_geometry_and_identity_permutation(self):
        for frame_count, patch in ((0, 1), (1, 1), (8, 0), (7, 2), (4, 4)):
            with self.subTest(frame_count=frame_count, patch=patch), self.assertRaises(ValueError):
                deterministic_reverse_block_indices(frame_count, patch)

    def test_reorder_copies_objects_and_does_not_mutate_source(self):
        class Frame:
            def __init__(self, value):
                self.value = value
            def copy(self):
                return Frame(self.value)

        source = [Frame(i) for i in range(4)]
        output = reorder_frames(source, (2, 3, 0, 1))
        self.assertEqual([x.value for x in output], [2, 3, 0, 1])
        self.assertTrue(all(a is not b for a, b in zip(source, output, strict=True)))
        self.assertEqual([x.value for x in source], [0, 1, 2, 3])

    def test_reorder_fails_closed_for_duplicate_or_missing_indices(self):
        source = [object(), object(), object()]
        for indices in ((0, 0, 2), (0, 1)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                reorder_frames(source, indices)

    def test_interaction_calculation(self):
        # Baseline chronology sensitivity is .4; biased sensitivity is .9.
        result = compute_bias_frame_order_interaction(
            original_baseline=-2.0, permuted_baseline=-2.4,
            original_biased=-1.2, permuted_biased=-2.1)
        self.assertAlmostEqual(
            result["original_minus_permuted_baseline_nats_per_token"], .4)
        self.assertAlmostEqual(
            result["original_minus_permuted_biased_nats_per_token"], .9)
        self.assertAlmostEqual(
            result["bias_by_frame_order_interaction_nats_per_token"], .5)

    def test_null_interaction_when_bias_changes_both_orders_equally(self):
        result = compute_bias_frame_order_interaction(
            original_baseline=-2.0, permuted_baseline=-2.5,
            original_biased=-1.0, permuted_biased=-1.5)
        self.assertAlmostEqual(
            result["bias_by_frame_order_interaction_nats_per_token"], 0.0)

    def test_interaction_rejects_nonfinite_and_bool_scores(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                compute_bias_frame_order_interaction(
                    original_baseline=value, permuted_baseline=0,
                    original_biased=0, permuted_biased=0)

    def test_biased_mask_audit_requires_bias_on_all_selected_cells(self):
        mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        mask[0, 0][torch.tensor([3, 4])[:, None], torch.tensor([0, 1])[None, :]] = 0.69314718
        self.assertTrue(validate_biased_mask_rows(mask,
            key_positions=(0, 1), query_positions=(3, 4),
            bias=0.69314718, torch=torch))
        mask[0, 0, 4, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "value_mismatch"):
            validate_biased_mask_rows(mask,
                key_positions=(0, 1), query_positions=(3, 4),
                bias=0.69314718, torch=torch)

    def test_packet_reader_selects_frozen_row_and_rejects_out_of_range(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            packet = private / "packet.json"
            payload = {"status": "passed", "dataset_split": "val",
                "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                "claim_review_completed": False,
                "records": [{"row": 0}, {"row": 1}]}
            fd = os.open(packet, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream)
            selected, digest = _read_packet(root, Path("data/private_nextgqa/packet.json"), 1)
            self.assertEqual(selected, {"row": 1})
            self.assertEqual(len(digest), 64)
            selected, _ = _read_packet(root, Path("data/private_nextgqa/packet.json"), 0,
                                       expected_split="val")
            self.assertEqual(selected, {"row": 0})
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, Path("data/private_nextgqa/packet.json"), 2)
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, Path("data/private_nextgqa/packet.json"), 0,
                             expected_split="test")
            payload["dataset_split"] = "test"
            packet.write_text(json.dumps(payload), encoding="utf-8")
            selected, _ = _read_packet(root, Path("data/private_nextgqa/packet.json"), 0,
                                       expected_split="test")
            self.assertEqual(selected, {"row": 0})
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, Path("data/private_nextgqa/packet.json"), 0)
            with self.assertRaisesRegex(ValueError, "path_or_split_invalid"):
                _read_packet(root, Path("../outside.json"), 0)

    def test_partial_packet_allows_only_count_bound_valid_target_rows(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            packet = private / "partial.json"
            payload = {"status": "partial", "dataset_split": "test",
                "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                "claim_review_completed": False,
                "fully_structured_claim_records": 1,
                "records": [
                    {"generated_parsed": {"valid": True, "claim": "A person waves."}},
                    {"generated_parsed": {"valid": False, "claim": None}}]}
            fd = os.open(packet, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream)
            rel = Path("data/private_nextgqa/partial.json")
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, rel, 0)
            selected, digest = _read_packet(root, rel, 0, allow_partial=True,
                                            expected_split="test")
            self.assertTrue(selected["generated_parsed"]["valid"])
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, rel, 1, allow_partial=True, expected_split="test")
            payload["fully_structured_claim_records"] = 2
            packet.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, rel, 0, allow_partial=True, expected_split="test")
            payload["fully_structured_claim_records"] = True
            packet.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, rel, 0, allow_partial=True, expected_split="test")
            payload["fully_structured_claim_records"] = 1
            payload["records"][0]["generated_parsed"]["claim"] = None
            payload["records"][1]["generated_parsed"] = {"valid": True,
                "claim": "A person waves."}
            packet.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract_invalid"):
                _read_packet(root, rel, 0, allow_partial=True, expected_split="test")

    def test_metadata_accepts_roundoff_but_not_wrong_frame_count(self):
        metadata = {"frames_indices": list(range(16)),
            "total_num_frames": 15.999999999999998, "fps": 0.5}
        self.assertTrue(validate_video_metadata(metadata,
            sample_fps=0.5, frame_count=16))
        metadata["total_num_frames"] = 15.9
        with self.assertRaisesRegex(ValueError, "metadata_mismatch"):
            validate_video_metadata(metadata, sample_fps=0.5, frame_count=16)


if __name__ == "__main__":
    unittest.main()
