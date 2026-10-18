from __future__ import annotations

import unittest

import torch

from qwen_internal_attention_ablation import ablate_selected_video_keys


class RetentionContrastGeometryTests(unittest.TestCase):
    def test_equal_count_evidence_and_control_retention_masks(self):
        # Four synthetic temporal groups, two visual positions in each.
        groups = ((2, 3), (4, 5), (6, 7), (8, 9))
        evidence_groups = (0, 2)
        control_groups = (1, 3)
        evidence = {p for g in evidence_groups for p in groups[g]}
        control = {p for g in control_groups for p in groups[g]}
        all_visual = {p for group in groups for p in group}
        self.assertEqual(len(evidence), len(control))
        self.assertTrue(evidence.isdisjoint(control))
        self.assertEqual(evidence | control, all_visual)

        # Retention is represented by ablating the complement of the kept set.
        evidence_only_ablation = tuple(sorted(all_visual - evidence))
        control_only_ablation = tuple(sorted(all_visual - control))
        qpos = 12
        length = 13
        base = torch.full((length, length), float("-inf"), dtype=torch.float32)
        base = torch.triu(base, diagonal=1)
        base = base.masked_fill(torch.tril(torch.ones_like(base), diagonal=0).bool(), 0.0)
        base = base.unsqueeze(0).unsqueeze(0)

        evidence_only = ablate_selected_video_keys(attention_mask=base,
            selected_positions=evidence_only_ablation, query_position=qpos,
            torch_module=torch)
        control_only = ablate_selected_video_keys(attention_mask=base,
            selected_positions=control_only_ablation, query_position=qpos,
            torch_module=torch)

        def retained_visual_positions(mask: torch.Tensor) -> set[int]:
            row = mask[0, 0, qpos]
            return {p for p in all_visual if float(row[p]) == 0.0}

        self.assertEqual(retained_visual_positions(evidence_only), evidence)
        self.assertEqual(retained_visual_positions(control_only), control)
        changed_evidence = set(torch.isneginf(evidence_only[0, 0, qpos, :qpos])
                               .nonzero().flatten().tolist())
        changed_control = set(torch.isneginf(control_only[0, 0, qpos, :qpos])
                              .nonzero().flatten().tolist())
        self.assertEqual(changed_evidence, all_visual - evidence)
        self.assertEqual(changed_control, all_visual - control)
        # Query and nonvisual positions remain available, and other query rows
        # are untouched by this fixed-query intervention helper.
        self.assertEqual(float(evidence_only[0, 0, qpos, qpos]), 0.0)
        self.assertTrue(torch.equal(evidence_only[0, 0, 11], base[0, 0, 11]))
        self.assertTrue(torch.equal(control_only[0, 0, 11], base[0, 0, 11]))


if __name__ == "__main__":
    unittest.main()
