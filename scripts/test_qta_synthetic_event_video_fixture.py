from __future__ import annotations

import hashlib
import json
import unittest

from PIL import Image
from PIL import ImageDraw

from qta_synthetic_event_video_fixture import (
    BLUE,
    FRAME_COUNT,
    HEIGHT,
    RED,
    TRANSITION_FRAME,
    TRANSITION_STOP_FRAME,
    WIDTH,
    build_synthetic_event_pair,
)


def _frame_hash(frame: Image.Image) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


class SyntheticEventVideoFixtureTests(unittest.TestCase):
    def test_pair_shape_frames_and_fixed_choice_contract(self):
        pair = build_synthetic_event_pair()
        self.assertEqual(len(pair.clips), 2)
        self.assertEqual([clip.clip_id for clip in pair.clips],
                         ["circle_changes", "triangle_changes"])
        self.assertEqual([clip.correct_choice_label for clip in pair.clips], ["B", "D"])
        self.assertEqual(pair.clips[0].question, pair.clips[1].question)
        self.assertEqual(pair.clips[0].choices, pair.clips[1].choices)
        for clip in pair.clips:
            self.assertEqual(len(clip.frames), FRAME_COUNT)
            self.assertEqual(len(clip.choices), 5)
            self.assertEqual(tuple(label for label, _ in clip.choices),
                             ("A", "B", "C", "D", "E"))
            for frame in clip.frames:
                self.assertEqual(frame.mode, "RGB")
                self.assertEqual(frame.size, (WIDTH, HEIGHT))

    def test_pair_is_pixel_identical_before_transition_and_only_shapes_change(self):
        circle_clip, triangle_clip = build_synthetic_event_pair().clips
        for index in range(TRANSITION_FRAME):
            self.assertEqual(_frame_hash(circle_clip.frames[index]),
                             _frame_hash(triangle_clip.frames[index]))

        # The layout is frozen. Only the transient target differs and the
        # question's event is not recoverable from final-frame appearance.
        spec = build_synthetic_event_pair().spec
        circle_xy = tuple(spec["shape_geometry"]["circle"]["center_xy"])
        triangle_xy = tuple(spec["shape_geometry"]["triangle"]["center_xy"])
        mask = Image.new("1", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mask)
        radius = spec["shape_geometry"]["circle"]["radius"]
        cx, cy = circle_xy
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=1)
        draw.polygon(spec["shape_geometry"]["triangle"]["vertices_xy"], fill=1)
        for index in range(TRANSITION_FRAME, TRANSITION_STOP_FRAME):
            circle_frame = circle_clip.frames[index]
            triangle_frame = triangle_clip.frames[index]
            self.assertEqual(circle_frame.getpixel((0, 0)),
                             triangle_frame.getpixel((0, 0)))
            self.assertEqual(circle_frame.getpixel(circle_xy), BLUE)
            self.assertEqual(triangle_frame.getpixel(circle_xy), RED)
            self.assertEqual(circle_frame.getpixel(triangle_xy), RED)
            self.assertEqual(triangle_frame.getpixel(triangle_xy), BLUE)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if not mask.getpixel((x, y)):
                        self.assertEqual(circle_frame.getpixel((x, y)),
                                         triangle_frame.getpixel((x, y)))
        for index in range(TRANSITION_STOP_FRAME, FRAME_COUNT):
            for point in (circle_xy, triangle_xy):
                self.assertEqual(circle_clip.frames[index].getpixel(point), RED)
                self.assertEqual(triangle_clip.frames[index].getpixel(point), RED)

    def test_only_declared_palette_is_drawn(self):
        pair = build_synthetic_event_pair()
        allowed = {tuple(pair.spec["background_rgb"]), RED, BLUE}
        for clip in pair.clips:
            for frame in clip.frames:
                raw = frame.tobytes()
                colors = set(zip(raw[0::3], raw[1::3], raw[2::3]))
                self.assertLessEqual(colors, allowed)

    def test_frames_have_exact_ground_truth_and_no_embedded_text(self):
        pair = build_synthetic_event_pair()
        manifest = pair.manifest()
        # Manifest must be machine-readable without serializing PIL frame bytes.
        raw = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("data:image", raw)
        self.assertFalse(manifest["spec"]["text_rendered_into_frames"])
        self.assertEqual(manifest["spec"]["transition_frame_index"], 16)
        self.assertEqual(manifest["spec"]["event_interval_frames_half_open"], [16, 20])
        self.assertEqual(len(manifest["clips"]), 2)
        for clip, expected_target in zip(
                manifest["clips"], ("circle", "triangle"), strict=True):
            gt = clip["ground_truth"]
            self.assertEqual(gt["changed_object"], expected_target)
            self.assertEqual(gt["color_transition"], {"from": "red", "to": "blue"})
            self.assertEqual(gt["transition_interval_frames_half_open"], [16, 20])
            states = gt["state_intervals_frames_half_open"]
            self.assertEqual(states[0], {
                "start": 0, "stop": 16, "circle": "red", "triangle": "red"})
            self.assertEqual(states[1], {
                "start": 16, "stop": 20,
                "circle": "blue" if expected_target == "circle" else "red",
                "triangle": "blue" if expected_target == "triangle" else "red",
            })
            self.assertEqual(states[2], {
                "start": 20, "stop": 32, "circle": "red", "triangle": "red"})

    def test_fixed_seed_reproduces_every_frame_and_manifest(self):
        first = build_synthetic_event_pair(seed=11)
        second = build_synthetic_event_pair(seed=11)
        self.assertEqual(first.manifest(), second.manifest())
        self.assertEqual(
            [[_frame_hash(frame) for frame in clip.frames] for clip in first.clips],
            [[_frame_hash(frame) for frame in clip.frames] for clip in second.clips],
        )
        other = build_synthetic_event_pair(seed=12)
        self.assertNotEqual(_frame_hash(first.clips[0].frames[0]),
                            _frame_hash(other.clips[0].frames[0]))

    def test_invalid_seed_rejected(self):
        for seed in (-1, 2**32, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_synthetic_event_pair(seed=seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
