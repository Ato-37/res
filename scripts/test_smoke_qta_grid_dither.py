from __future__ import annotations

import os
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_qta_grid_dither import (  # noqa: E402
    _write_private,
    _snapshot_sha256,
    STATE,
    dithered_indices,
    empirical_wasserstein1,
    exclusion_manifest_receipt,
    forward_count_receipt,
    group_set_jaccard,
    interval_union_jaccard,
    require_uniform_pts,
    scope_name,
    source_hash_receipt,
    video_pair_count_receipt,
    SOURCE_NAMES,
    main,
    support_intervals,
    validate_capture_receipt,
    validate_grid,
)


class GridDitherTests(unittest.TestCase):
    def test_frozen_offsets_preserve_endpoints_and_sorted_uniqueness(self):
        baseline, dither = dithered_indices(100)
        self.assertEqual((baseline[0], baseline[-1]), (0, 99))
        self.assertEqual((dither[0], dither[-1]), (0, 99))
        self.assertTrue(all(a < b for a, b in zip(dither, dither[1:])))
        for slot in range(1, 31):
            self.assertEqual(dither[slot] - baseline[slot], 1 if slot % 2 else 0)

    def test_rejects_duplicate_or_non_endpoint_grids_without_repair(self):
        with self.assertRaisesRegex(ValueError, "sorted_unique"):
            dithered_indices(40)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            validate_grid(list(range(1, 33)), 40)

    def test_pts_tolerance_checks_real_times(self):
        regular = [Fraction(i, 31) for i in range(32)]
        rate, deviation = require_uniform_pts(regular)
        self.assertGreater(rate, 0)
        self.assertLessEqual(deviation, 0.05)
        perturbed = [Fraction(i, 31) for i in range(15)] + [Fraction(1, 2)]
        perturbed += [Fraction(3, 5) + Fraction(i, 100) for i in range(16)]
        with self.assertRaises(ValueError):
            require_uniform_pts(perturbed)

    def test_metrics_distinguish_ordinal_and_actual_pts_support(self):
        self.assertEqual(group_set_jaccard([1, 2], [2, 3]), 1 / 3)
        a, ca = support_intervals([i / 31 for i in range(32)], 16, 2)
        shifted_pts = [i / 31 for i in range(32)]
        shifted_pts[1] += 0.01
        b, cb = support_intervals(shifted_pts, 16, 2)
        self.assertAlmostEqual(interval_union_jaccard([a[3]], [a[3]]), 1.0)
        self.assertGreaterEqual(interval_union_jaccard([a[3]], [b[3]]), 0.0)
        self.assertGreaterEqual(empirical_wasserstein1([ca[3]], [cb[3]]), 0.0)

    def test_capture_receipt_requires_exact_layers_calls_and_normalization(self):
        layers = list(range(5, 17))
        receipt = {
            "status": "one_forward_attention_rows_captured_non_authorizing",
            "forward_calls": 1,
            "layer_indices_1_based": layers,
            "layer_calls": {str(layer): 1 for layer in layers},
            "attention_rows_satisfy_source_dtype_normalization_tolerance": True,
            "attention_row_sum_max_abs_deviation": 1e-6,
        }
        self.assertEqual(validate_capture_receipt(receipt), 1e-6)
        receipt["layer_calls"]["5"] = 2
        with self.assertRaisesRegex(RuntimeError, "layer_receipt"):
            validate_capture_receipt(receipt)

    def test_forward_receipt_separates_failed_from_not_attempted(self):
        self.assertEqual(forward_count_receipt(6, 3, 2), {
            "requested_forwards": 6, "attempted_forwards": 3,
            "completed_forwards": 2, "failed_forwards": 1,
            "not_attempted_forwards": 3,
        })
        with self.assertRaisesRegex(ValueError, "count_contract"):
            forward_count_receipt(2, 1, 2)
        self.assertEqual(scope_name(True), "exploratory_qta_32_frame_grid_dither_smoke")
        self.assertEqual(scope_name(False), "exploratory_qta_32_frame_grid_dither")

    def test_video_receipt_separates_failed_attempts_from_unattempted(self):
        self.assertEqual(video_pair_count_receipt(3, 1, 0), {
            "requested_video_pairs": 3, "attempted_video_pairs": 1,
            "completed_video_pairs": 0, "failed_video_pairs": 1,
            "not_attempted_video_pairs": 2,
        })
        with self.assertRaisesRegex(ValueError, "video_pair_count"):
            video_pair_count_receipt(2, 1, 2)

    def test_source_receipt_binds_every_source_before_and_after(self):
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(temporary) / "scripts"
            scripts.mkdir()
            for name in SOURCE_NAMES:
                (scripts / name).write_text(name, encoding="utf-8")
            before = source_hash_receipt(temporary)
            self.assertEqual(before, source_hash_receipt(temporary))
            (scripts / SOURCE_NAMES[0]).write_text("changed", encoding="utf-8")
            after = source_hash_receipt(temporary)
            self.assertNotEqual(before, after)

    def test_exclusion_manifest_receipt_binds_names_hashes_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            (data / "nextgqa_samples_a.json").write_text("[]", encoding="utf-8")
            (data / "nextgqa_samples_b.json").write_text("{}", encoding="utf-8")
            receipt = exclusion_manifest_receipt(temporary)
            self.assertEqual(receipt["count"], 2)
            self.assertEqual([entry[0] for entry in receipt["names_and_sha256"]],
                             ["nextgqa_samples_a.json", "nextgqa_samples_b.json"])
            self.assertEqual(receipt["digest_sha256"], hashlib.sha256(json.dumps(
                receipt["names_and_sha256"], separators=(",", ":")).encode()).hexdigest())

    def test_snapshot_hash_binds_file_bytes_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            shard = snapshot / "model.bin"
            shard.write_bytes(b"first")
            first = _snapshot_sha256(snapshot)
            self.assertEqual(first, _snapshot_sha256(snapshot))
            shard.write_bytes(b"second")
            self.assertNotEqual(first, _snapshot_sha256(snapshot))

    def test_smoke_failure_receipt_separates_attempted_and_not_attempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary) / "private"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            output = private / "failure.json"

            def fail_after_one_attempt(_args):
                STATE["attempted_forwards"] = 1
                STATE["completed_forwards"] = 0
                STATE["attempted_videos"] = 1
                raise RuntimeError("synthetic_failure")

            stdout = io.StringIO()
            with patch("smoke_qta_grid_dither._execute", side_effect=fail_after_one_attempt):
                with redirect_stdout(stdout):
                    self.assertEqual(main(["--output", str(output), "--smoke"]), 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["scope"], "exploratory_qta_32_frame_grid_dither_smoke")
            self.assertIs(report["smoke_only"], True)
            self.assertEqual(report["requested_forwards"], 2)
            self.assertEqual(report["attempted_forwards"], 1)
            self.assertEqual(report["completed_forwards"], 0)
            self.assertEqual(report["failed_forwards"], 1)
            self.assertEqual(report["not_attempted_forwards"], 1)
            self.assertEqual(report["requested_video_pairs"], 1)
            self.assertEqual(report["attempted_video_pairs"], 1)
            self.assertEqual(report["completed_video_pairs"], 0)
            self.assertEqual(report["failed_video_pairs"], 1)
            self.assertEqual(report["not_attempted_video_pairs"], 0)

    def test_private_writer_mode_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private"
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            output = directory / "result.json"
            _write_private(output, {"synthetic": True})
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                _write_private(output, {})


if __name__ == "__main__":
    unittest.main()
