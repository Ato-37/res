import unittest
from pathlib import Path

from run_noaa_annotation_matched_group_margin import (
    calculate_fixed_margin,
    load_annotation_points,
    matched_control_groups,
)
import run_noaa_qta_distribution_pilot as base
import analyze_noaa_qta_annotation_time_alignment as alignment


class NoaaAnnotationMatchedGroupMarginTests(unittest.TestCase):
    def test_controls_are_equal_size_disjoint_and_position_matched(self):
        token = "row-token"
        for evidence, expected_distance in [((0,), 1), ((1,), 1), ((2,), 1),
                                           ((3,), 1), ((1, 2), 2)]:
            control = matched_control_groups(evidence, 4, token)
            self.assertEqual(len(control), len(evidence))
            self.assertFalse(set(control) & set(evidence))
            distance = sum(abs(e - c) for e, c in zip(evidence, control, strict=True))
            self.assertEqual(distance, expected_distance)
            self.assertEqual(control, matched_control_groups(evidence, 4, token))
        self.assertEqual(matched_control_groups((1, 2), 4, token), (0, 3))

    def test_rejects_invalid_or_impossible_control(self):
        for evidence, count, token in [((1, 1), 4, "x"), ((-1,), 4, "x"),
                                       ((0, 1, 2), 4, "x"), ((0,), 1, "x"),
                                       ((0,), 4, "")]:
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    matched_control_groups(evidence, count, token)

    def test_fixed_margin_keeps_baseline_pair(self):
        self.assertEqual(calculate_fixed_margin([-1., -2., -4., -3., -5.], 0, 1), 1.)
        with self.assertRaises(ValueError):
            calculate_fixed_margin([0.0] * 5, 0, 0)

    def test_local_six_row_annotation_preflight_has_equal_budgets(self):
        repo = Path(__file__).resolve().parents[1]
        records, manifests = base.load_samples(repo)
        annotations, _ = load_annotation_points(repo)
        receipt, _ = alignment.read_private_json(repo, alignment.RUN_REL)
        self.assertEqual(len(records), 6)
        self.assertEqual(len(annotations), 6)
        self.assertEqual(set(manifests), {entry[0] for entry in alignment.INPUTS})
        receipt_by_key = {(row["source_video_sha256"], row["qid_sha256"]): row
                          for row in receipt["rows"]}
        cardinalities = []
        for record in records:
            points, _method = annotations[(record["video_sha256"], record["qid_sha256"])]
            run_row = receipt_by_key[(record["video_sha256"], record["qid_sha256"])]
            pts = run_row["sampled_pts_seconds"]
            group_count = run_row["grid_thw"][0]
            evidence = tuple(sorted({alignment.nearest_temporal_group(t, pts, group_count)
                                     for t in points}))
            control = matched_control_groups(evidence, group_count, record["row_token"])
            self.assertEqual(len(evidence), len(control))
            cardinalities.append(len(evidence))
        self.assertEqual(sorted(cardinalities), [1, 1, 1, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
