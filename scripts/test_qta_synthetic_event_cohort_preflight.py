from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from collections import UserDict

from qta_synthetic_event_construct_v1 import (
    CLIP_COUNT, EVENT_LENGTH, FRAME_COUNT, FROZEN_CHOICE_TEXTS, LABELS,
    ONSET_STRATA, PAIR_COUNT, SEED_DRAW_COUNT, SEED_SAMPLE_RANDOM_SEED,
    SEED_STRATIFY_RANDOM_SEED, COMBINATION_RANDOM_SEED_BASE,
    assignment_manifest, build_assignment, build_construct_clip,
    external_smoke_seed,
)
from run_qta_synthetic_event_cohort_preflight import (
    _assert_aggregate_only, _batch_geometry, _design_summary,
    event_and_control_groups, write_report_exclusive,
)


class FakeTensor:
    def __init__(self, data, *, shape=None, ndim=None, device="cpu"):
        self.data = data
        self.shape = shape if shape is not None else (len(data),)
        self.ndim = ndim if ndim is not None else len(self.shape)
        self.device = type("Device", (), {"type": device})()

    def __getitem__(self, index):
        value = self.data[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return value

    def tolist(self):
        return list(self.data)

    def any(self):
        flat = [item for row in self.data for item in row] if self.data and isinstance(self.data[0], list) else self.data
        return FakeTensor([any(flat)])

    def __le__(self, value):
        flat = [item for row in self.data for item in row] if self.data and isinstance(self.data[0], list) else self.data
        return FakeTensor([item <= value for item in flat])

    def item(self):
        return bool(self.data[0])


class FakeTorch:
    Tensor = FakeTensor


class SyntheticEventCohortPreflightTests(unittest.TestCase):
    def test_seed_sampling_smoke_reservation_and_stratification_exact(self):
        sampled = random.Random(SEED_SAMPLE_RANDOM_SEED).sample(range(2**32), 41)
        self.assertEqual(SEED_DRAW_COUNT, 41)
        self.assertEqual(external_smoke_seed(), sampled[0])
        self.assertEqual(len(set(sampled)), 41)
        cohort = sampled[1:]
        random.Random(SEED_STRATIFY_RANDOM_SEED).shuffle(cohort)
        designs = build_assignment()
        self.assertEqual([row.scene_seed for row in designs], cohort)
        self.assertNotIn(sampled[0], [row.scene_seed for row in designs])
        self.assertEqual(len(designs), PAIR_COUNT)
        self.assertEqual(len({row.scene_seed for row in designs}), PAIR_COUNT)
        self.assertEqual(len(designs) * 2, CLIP_COUNT)

    def test_each_stratum_has_exactly_each_orientation_rotation_once(self):
        designs = build_assignment()
        self.assertEqual([row.onset_frame for row in designs],
                         [onset for onset in ONSET_STRATA for _ in range(10)])
        expected = {(orientation, rotation)
                    for orientation in (True, False) for rotation in range(5)}
        for onset in ONSET_STRATA:
            stratum = [row for row in designs if row.onset_frame == onset]
            self.assertEqual(len(stratum), 10)
            self.assertEqual(Counter(row.circle_left for row in stratum),
                             Counter({True: 5, False: 5}))
            self.assertEqual(Counter((row.circle_left, row.choice_rotation)
                                     for row in stratum),
                             Counter({pair: 1 for pair in expected}))
        summary = _design_summary()
        self.assertEqual(summary["sampled_cohort_seeds_unique"], 40)
        self.assertEqual([row["pairs"] for row in summary["onset_pair_counts"]],
                         [10, 10, 10, 10])

    def test_correct_choice_labels_exactly_balanced_for_each_target(self):
        designs = build_assignment()
        for target in ("The circle", "The triangle"):
            counts = Counter(next(label for label, text in row.choices if text == target)
                             for row in designs)
            self.assertEqual(counts, Counter({label: 8 for label in LABELS}))

    def test_assignment_manifest_hash_is_reproducible_and_smoke_excluded(self):
        one = assignment_manifest(python_version="3.test")
        two = assignment_manifest(python_version="3.test")
        self.assertEqual(one, two)
        self.assertTrue(one["seed_sampling"]["external_smoke_seed_excluded_from_analysis"])
        self.assertEqual(len(one["pairs"]), 40)
        self.assertNotIn("prompt", json.dumps(one).lower())

    def test_pairs_share_everything_except_changed_shape_event(self):
        design = build_assignment()[0]
        circle = build_construct_clip(design, "circle")
        triangle = build_construct_clip(design, "triangle")
        self.assertEqual(len(circle.frames), FRAME_COUNT)
        self.assertEqual(circle.question, triangle.question)
        self.assertEqual(circle.choices, triangle.choices)
        self.assertEqual(circle.event_start_frame, triangle.event_start_frame)
        self.assertNotEqual(circle.correct_choice_label, triangle.correct_choice_label)
        start = design.onset_frame
        for index, (left, right) in enumerate(zip(circle.frames, triangle.frames, strict=True)):
            self.assertEqual(left.mode, "RGB")
            self.assertEqual(left.size, (256, 192))
            if not start <= index < start + EVENT_LENGTH:
                self.assertEqual(left.tobytes(), right.tobytes())

    def test_frozen_onsets_map_to_two_evidence_and_two_nearest_controls(self):
        for onset in ONSET_STRATA:
            evidence, controls = event_and_control_groups(onset, 16)
            self.assertEqual(evidence, [onset // 2, onset // 2 + 1])
            self.assertEqual(len(controls), 2)
            self.assertFalse(set(evidence) & set(controls))
            center = onset * 2 + EVENT_LENGTH
            expected = sorted(
                (g for g in range(16) if g not in evidence),
                key=lambda g: (abs((2 * g + 1) * 2 - center), g),
            )[:2]
            self.assertEqual(controls, sorted(expected))
        for bad in ((5, 16, 2), (4, 15, 2), (4, 16, 4), (4, 16, 0)):
            with self.assertRaises(ValueError):
                event_and_control_groups(*bad)

    def test_batchfeature_mapping_and_grid_validation(self):
        batch = UserDict({"input_ids": FakeTensor([[1, 2, 3]], shape=(1, 3), ndim=2),
                          "video_grid_thw": FakeTensor([[16, 28, 40]], shape=(1, 3), ndim=2)})
        geometry, digest = _batch_geometry(batch, FakeTorch, merge_size=2)
        self.assertEqual(geometry["video_grid_thw"], [16, 28, 40])
        self.assertEqual(geometry["merged_visual_tokens"], 16 * 14 * 20)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            _batch_geometry({"input_ids": FakeTensor([[1]], shape=(1, 1), ndim=2),
                             "video_grid_thw": FakeTensor([[16, 27, 42]], shape=(1, 3), ndim=2)},
                            FakeTorch, merge_size=2)

    def test_aggregate_only_sanitizer_accepts_qa_video_hash_key(self):
        _assert_aggregate_only({"source_sha256": {"qa_video_builder": "a" * 64}})
        with self.assertRaises(ValueError):
            _assert_aggregate_only({"question": "private"})
        with self.assertRaises(ValueError):
            _assert_aggregate_only({"raw_prompt": "private"})

    def test_private_receipt_writes_0600_exclusively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_report_exclusive(path, {"status": "passed", "clips": CLIP_COUNT})
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["clips"], CLIP_COUNT)
            with self.assertRaises(FileExistsError):
                write_report_exclusive(path, {"status": "again"})


if __name__ == "__main__":
    unittest.main()
