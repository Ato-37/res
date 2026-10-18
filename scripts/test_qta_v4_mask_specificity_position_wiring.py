from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from qwen_internal_attention_ablation import ablate_selected_video_keys
from run_nextgqa_qta_v4_mask_specificity import prepare_mask_row


class PlaceboPositionWiringTests(unittest.TestCase):
    def test_synthetic_translation_reaches_exact_ablation_positions(self):
        groups = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        geometry_sha = "a" * 64
        binding = "b" * 64
        item = {"video_id": "synthetic-video", "qid": "synthetic-qid",
                "planned": {"geometry_sha256": geometry_sha,
                    "temporal_group_count": len(groups),
                    "annotation_union_covered_group_count": 1,
                    "selected_row_binding_sha256": binding}}
        annotation_positions = tuple(groups[1])
        mapped = {"evidence_group_indices": [1],
            "annotation_covered_group_indices": [1], "control_group_indices": [],
            "evidence_token_positions": annotation_positions}
        batch = {"input_ids": torch.zeros((1, 19), dtype=torch.long)}
        fake_geometry = (batch, mapped, groups, 18, (1, 2), geometry_sha)
        old = {"geometry_sha256": geometry_sha, "row_binding_sha256": binding}

        with patch("run_nextgqa_qta_v4_mask_specificity.kernel.prepare_geometry",
                   return_value=fake_geometry):
            prepared = prepare_mask_row(item=item, frames=[], decode={}, processor=object(),
                                        torch_module=torch, old=old)

        expected_annotation = tuple(pos for pos in groups[1])
        expected_placebo = tuple(sorted(
            pos for group_index in prepared["placebo_group_indices"]
            for pos in groups[group_index]))
        self.assertEqual(prepared["annotation_positions"], expected_annotation)
        self.assertEqual(prepared["placebo_positions"], expected_placebo)
        self.assertEqual(len(expected_annotation), len(expected_placebo))
        self.assertNotEqual(set(expected_annotation), set(expected_placebo))

        original = torch.zeros((1, 1, 19, 19), dtype=torch.float32)
        ablated = ablate_selected_video_keys(attention_mask=original,
            selected_positions=expected_placebo, query_position=18,
            torch_module=torch)
        changed = tuple(torch.isneginf(ablated[0, 0, 18]).nonzero().flatten().tolist())
        self.assertEqual(changed, expected_placebo)
        self.assertTrue(torch.equal(ablated[0, 0, 17], original[0, 0, 17]))


if __name__ == "__main__":
    unittest.main()
