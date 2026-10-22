from __future__ import annotations

import os
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_qta_resolution_stability import (  # noqa: E402
    _write_private,
    main,
    _RUN_STATE,
    compare_selections,
    interval_jaccard,
    processor_group_support,
    pair_metric_rows,
    summarize_pairs,
    wasserstein1_empirical,
)


class QTAResolutionStabilityTests(unittest.TestCase):
    def test_same_support_has_unit_jaccard_and_zero_center_distance(self):
        intervals, centers = processor_group_support([i / 15 for i in range(16)], 8, 2)
        jaccard, distance = compare_selections(intervals, centers, [1, 7], intervals, centers, [1, 7])
        self.assertAlmostEqual(jaccard, 1.0, places=12)
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_metrics_reflect_disjoint_and_shifted_synthetic_support(self):
        intervals, centers = processor_group_support([i / 7 for i in range(8)], 4, 2)
        self.assertEqual(interval_jaccard([intervals[0]], [intervals[3]]), 0.0)
        self.assertAlmostEqual(wasserstein1_empirical([.1, .9], [.2, .8]), .1)
        smoke_metrics = summarize_pairs([(0.25, 0.4)])
        self.assertEqual(smoke_metrics["interval_jaccard_mean"], 0.25)
        self.assertEqual(smoke_metrics["selected_center_wasserstein1_mean"], 0.4)
        self.assertEqual(pair_metric_rows([(0.25, 0.4)]), [{
            "fixed_pair_ordinal": 1,
            "interval_jaccard": 0.25,
            "selected_center_wasserstein1": 0.4,
        }])

    def test_rejects_invalid_times_or_unequal_center_budget(self):
        self.assertAlmostEqual(wasserstein1_empirical([.2], [.2, .3]), .05)

    def test_frame_pts_use_adjacent_midpoint_support_boundaries(self):
        intervals, centers = processor_group_support([i / 15 for i in range(16)], 8, 2)
        self.assertEqual(len(intervals), 8)
        self.assertAlmostEqual(intervals[0][0], 0.0)
        self.assertAlmostEqual(intervals[0][1], 1.5 / 15)
        self.assertAlmostEqual(intervals[-1][1], 1.0)
        self.assertAlmostEqual(intervals[1][0], 1.5 / 15)
        self.assertAlmostEqual(intervals[1][1], 3.5 / 15)
        self.assertAlmostEqual(centers[0], .5 / 15)
        with self.assertRaisesRegex(ValueError, "mapping_shape"):
            processor_group_support([i / 15 for i in range(16)], 7, 2)

    def test_frame_pts_are_mapped_to_patch_centers(self):
        _, centers = processor_group_support([i / 15 for i in range(16)], 8, 2)
        self.assertEqual(len(centers), 8)
        self.assertAlmostEqual(centers[0], .5 / 15)
        self.assertAlmostEqual(centers[-1], 14.5 / 15)

    def test_interval_support_matches_reference_mapper_contract(self):
        from qwen_review_interval_token_groups import map_review_interval_to_qwen_groups
        pts = [Fraction(i, 15) for i in range(16)]
        ids = [42] * 8
        mapped = map_review_interval_to_qwen_groups(input_ids=ids, video_token_id=42,
            video_grid_thw=[8, 1, 1], spatial_merge_size=1, temporal_patch_size=2,
            sampled_pts_seconds=pts, video_duration_seconds=1,
            reviewed_interval_seconds=[Fraction(0), Fraction(1, 10)], _select_control=False)
        intervals, _ = processor_group_support([float(x) for x in pts], 8, 2)
        self.assertAlmostEqual(intervals[0][0], mapped["groups"][0]["coverage_start_seconds"])
        self.assertAlmostEqual(intervals[0][1], mapped["groups"][0]["coverage_end_seconds"])

    def test_private_writer_creates_mode_0600_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private"
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            path = directory / "summary.json"
            _write_private(path, {"synthetic": True})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                _write_private(path, {"synthetic": False})

    def test_private_writer_rejects_symlinks_and_unprivate_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            target = private / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = private / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                _write_private(link, {}, safe_roots=[root])
            os.chmod(private, 0o755)
            with self.assertRaisesRegex(PermissionError, "must_be_private"):
                _write_private(private / "new.json", {}, safe_roots=[root])

    def test_failure_writes_counts_without_partial_selection_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private"
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            output = directory / "failure.json"
            def fail_on_last_forward(_args):
                _RUN_STATE.update(completed_forwards=_RUN_STATE["requested_forwards"] - 1,
                                  effective_video_pairs=2 if _RUN_STATE["requested_forwards"] == 6 else 0,
                                  candidate_videos_inspected=_RUN_STATE["requested_video_pairs"])
                raise RuntimeError("synthetic_failure")
            with patch("smoke_qta_resolution_stability._execute", side_effect=fail_on_last_forward):
                code = main(["--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertEqual(report["requested_forwards"], 6)
            self.assertEqual(report["completed_forwards"], 5)
            self.assertEqual(report["failed_forwards"], 1)
            self.assertEqual(report["effective_video_pairs"], 2)
            self.assertEqual(report["failed_video_pairs"], 1)
            self.assertNotIn("selected_groups", report)
            smoke_output = directory / "smoke-failure.json"
            stdout = io.StringIO()
            with patch("smoke_qta_resolution_stability._execute", side_effect=fail_on_last_forward):
                with redirect_stdout(stdout):
                    code = main(["--output", str(smoke_output), "--smoke"])
            smoke_report = json.loads(smoke_output.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(smoke_report["requested_videos"], 1)
            self.assertEqual(smoke_report["requested_forwards"], 2)
            self.assertEqual(smoke_report["completed_forwards"], 1)
            self.assertEqual(smoke_report["failed_forwards"], 1)
            self.assertEqual(printed["requested_videos"], 1)
            self.assertEqual(printed["requested_forwards"], 2)


if __name__ == "__main__":
    unittest.main()
