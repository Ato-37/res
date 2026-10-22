from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_qta_repeatability import (  # noqa: E402
    _STATE,
    _directory_content_sha256,
    _write_private,
    batch_identity_sha256,
    capture_same_batch_twice,
    initialize_rng_once,
    main,
    ordinal_rows,
    require_same_snapshot,
    repeatability_pair,
    summarize_pairs,
)
from smoke_qta_resolution_stability import processor_group_support  # noqa: E402


class QTARepeatabilityTests(unittest.TestCase):
    def test_mocked_repeats_reuse_object_advance_rng_and_check_both_receipts(self):
        class FakeCuda:
            def __init__(self, owner):
                self.owner = owner
            def manual_seed_all(self, seed):
                self.owner.seed_calls.append(("cuda", seed))

        class FakeTorch:
            def __init__(self):
                self.seed_calls = []
                self.cuda = FakeCuda(self)
            def manual_seed(self, seed):
                self.seed_calls.append(("cpu", seed))

        fake_torch = FakeTorch()
        initialize_rng_once(fake_torch, 123)
        model, batch = object(), object()
        rng_position = {"value": 0}
        attempts, completions, seen_batches, seen_rng = [], [], [], []
        receipt = {"status": "one_forward_attention_rows_captured_non_authorizing",
                   "forward_calls": 1,
                   "attention_rows_satisfy_source_dtype_normalization_tolerance": True,
                   "attention_row_sum_max_abs_deviation": 1e-7}

        def capture_once(received_model, received_batch, *, on_forward_complete):
            self.assertIs(received_model, model)
            self.assertIs(received_batch, batch)
            seen_batches.append(received_batch)
            rng_position["value"] += 1
            seen_rng.append(rng_position["value"])
            on_forward_complete()
            return [rng_position["value"]], receipt

        selections, digest, deviation, verified = capture_same_batch_twice(
            model=model, batch=batch, batch_identity=lambda value: "fixed-batch-hash",
            capture_once=capture_once,
            on_attempt=lambda: attempts.append(True),
            on_complete=lambda: completions.append(True))
        self.assertEqual(fake_torch.seed_calls, [("cpu", 123), ("cuda", 123)])
        self.assertEqual(seen_batches, [batch, batch])
        self.assertEqual(seen_rng, [1, 2])
        self.assertEqual(selections, [[1], [2]])
        self.assertEqual(digest, "fixed-batch-hash")
        self.assertEqual((len(attempts), len(completions), verified), (2, 2, 2))
        self.assertEqual(deviation, 1e-7)

    def test_completed_forward_is_counted_if_selection_postprocessing_fails(self):
        attempts, completions = [], []
        def captured_then_selection_fails(_model, _batch, *, on_forward_complete):
            on_forward_complete()
            raise RuntimeError("selection_failed_after_forward")
        with self.assertRaisesRegex(RuntimeError, "selection_failed"):
            capture_same_batch_twice(
                model=object(), batch=object(), batch_identity=lambda _batch: "same",
                capture_once=captured_then_selection_fails,
                on_attempt=lambda: attempts.append(True),
                on_complete=lambda: completions.append(True))
        self.assertEqual((len(attempts), len(completions)), (1, 1))

    def test_identical_synthetic_selection_has_perfect_repeatability(self):
        intervals, centers = processor_group_support([i / 15 for i in range(16)], 8, 2)
        metric = repeatability_pair(intervals, centers, [0, 7], [0, 7])
        self.assertAlmostEqual(metric[0], 1.0)
        self.assertAlmostEqual(metric[1], 0.0)

    def test_different_synthetic_selection_has_lower_jaccard_and_positive_w1(self):
        intervals, centers = processor_group_support([i / 15 for i in range(16)], 8, 2)
        metric = repeatability_pair(intervals, centers, [0], [7])
        self.assertEqual(metric[0], 0.0)
        self.assertGreater(metric[1], 0)

    def test_aggregate_and_ordinal_rows_keep_pair_hashes_only(self):
        metrics = [(1.0, 0.0), (0.5, 0.125)]
        hashes = ["a" * 64, "b" * 64]
        summary = summarize_pairs(metrics)
        rows = ordinal_rows(metrics, hashes)
        self.assertEqual(summary["interval_jaccard_mean"], .75)
        self.assertEqual(rows[0]["fixed_pair_ordinal"], 1)
        self.assertEqual(rows[1]["batch_sha256"], hashes[1])
        self.assertNotIn("selected_groups", rows[0])
        with self.assertRaisesRegex(ValueError, "count_mismatch"):
            ordinal_rows(metrics, hashes[:1])

    def test_batch_identity_binds_tensor_batch_digest_and_video_metadata(self):
        digest = "c" * 64
        first = batch_identity_sha256(digest, {"fps": 2.0, "frames_indices": list(range(32))})
        same = batch_identity_sha256(digest, {"frames_indices": list(range(32)), "fps": 2.0})
        changed = batch_identity_sha256(digest, {"fps": 2.1, "frames_indices": list(range(32))})
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_model_snapshot_hash_tracks_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "config.json"
            target.write_text("{}", encoding="utf-8")
            before = _directory_content_sha256(root)
            target.write_text("{\"x\":1}", encoding="utf-8")
            self.assertNotEqual(before, _directory_content_sha256(root))
            with self.assertRaisesRegex(RuntimeError, "snapshot_changed"):
                require_same_snapshot(before, _directory_content_sha256(root))
            require_same_snapshot(before, before)

    def test_private_writer_is_owner_only_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary) / "private"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            target = private / "summary.json"
            _write_private(target, {"synthetic": True})
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                _write_private(target, {})

    def test_failure_receipt_contains_counts_and_no_partial_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary) / "private"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            output = private / "failure.json"
            def fail_after_one_forward(_args):
                _STATE.update(attempted_forwards=2, completed_forwards=1, effective_video_pairs=0,
                              candidate_videos_inspected=1)
                raise RuntimeError("synthetic_failure")
            stream = io.StringIO()
            with patch("smoke_qta_repeatability._execute", side_effect=fail_after_one_forward):
                with contextlib.redirect_stdout(stream):
                    code = main(["--smoke", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertEqual(report["requested_forwards"], 2)
            self.assertEqual(report["completed_forwards"], 1)
            self.assertEqual(report["failed_forwards"], 1)
            self.assertEqual(report["failed_video_pairs"], 1)
            self.assertNotIn("metrics", report)
            self.assertNotIn("selected_groups", report)


if __name__ == "__main__":
    unittest.main()
