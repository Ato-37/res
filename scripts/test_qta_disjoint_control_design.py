from __future__ import annotations

from itertools import combinations
import unittest

from qta_disjoint_control_design import (
    select_qta_disjoint_rank_matched_control_groups,
)


class QtaDisjointControlDesignTests(unittest.TestCase):
    def test_control_has_equal_count_and_excludes_evidence_and_treatment(self):
        result = select_qta_disjoint_rank_matched_control_groups(
            evidence_group_indices=(1, 5), qta_group_indices=(3,),
            temporal_group_count=8)
        self.assertEqual(result, [0, 4])
        self.assertEqual(len(result), 2)
        self.assertFalse(set(result) & {1, 5, 3})

    def test_equal_cost_ties_resolve_to_lexicographically_earlier(self):
        result = select_qta_disjoint_rank_matched_control_groups(
            evidence_group_indices=(2,), qta_group_indices=(0, 4),
            temporal_group_count=5)
        self.assertEqual(result, [1])

    def test_rejects_treatment_overlap_and_insufficient_controls(self):
        with self.assertRaisesRegex(ValueError, "qta_evidence_groups_overlap"):
            select_qta_disjoint_rank_matched_control_groups(
                evidence_group_indices=(1,), qta_group_indices=(1,),
                temporal_group_count=5)
        with self.assertRaisesRegex(ValueError, "equal_count_control_unavailable"):
            select_qta_disjoint_rank_matched_control_groups(
                evidence_group_indices=(0, 1), qta_group_indices=(2, 3),
                temporal_group_count=5)

    def test_rejects_malformed_or_out_of_range_indices(self):
        cases = [
            ((1, 1), (3,), 6, "evidence_group_indices_invalid"),
            ((1,), (3, 2), 6, "qta_group_indices_invalid"),
            ((True,), (3,), 6, "evidence_group_indices_invalid"),
            ((1,), (6,), 6, "qta_group_indices_invalid"),
        ]
        for evidence, qta, count, message in cases:
            with self.subTest(message=message, evidence=evidence, qta=qta):
                with self.assertRaisesRegex(ValueError, message):
                    select_qta_disjoint_rank_matched_control_groups(
                        evidence_group_indices=evidence, qta_group_indices=qta,
                        temporal_group_count=count)

    def test_dynamic_program_matches_exhaustive_selection_oracle(self):
        for total in range(3, 8):
            all_groups = range(total)
            for evidence_count in range(1, total):
                for evidence in combinations(all_groups, evidence_count):
                    remaining = tuple(x for x in all_groups if x not in evidence)
                    for qta_count in range(1, len(remaining)):
                        for qta in combinations(remaining, qta_count):
                            candidates = tuple(x for x in remaining if x not in qta)
                            if len(candidates) < evidence_count:
                                continue
                            expected = min(combinations(candidates, evidence_count),
                                key=lambda chosen: (
                                    sum(abs(a - b) for a, b in zip(evidence, chosen)),
                                    chosen))
                            actual = select_qta_disjoint_rank_matched_control_groups(
                                evidence_group_indices=evidence,
                                qta_group_indices=qta,
                                temporal_group_count=total)
                            self.assertEqual(tuple(actual), expected)


if __name__ == "__main__":
    unittest.main()
