"""Deterministic paired shape-color event clips for construct-readability tests.

This fixture is deliberately separate from the NExT-GQA faithfulness study.
It creates PIL frames in memory and never writes video, prompts, or results.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from PIL import Image, ImageDraw


GENERATOR_VERSION = "qta_synthetic_event_pair_v2"
SEED = 20261019
FRAME_COUNT = 32
FPS = 1
WIDTH = 256
HEIGHT = 192
TRANSITION_FRAME = 16
TRANSITION_STOP_FRAME = 20
BACKGROUND = (250, 250, 250)
RED = (214, 39, 40)
BLUE = (31, 119, 180)
QUESTION = "Which shape briefly switches from red to blue and then returns to red?"

_CHOICES = (
    ("A", "Both shapes"),
    ("B", "The circle"),
    ("C", "Neither shape"),
    ("D", "The triangle"),
    ("E", "The background"),
)


@dataclass(frozen=True)
class EventClip:
    """One model-ready clip plus its private, JSON-safe ground truth."""

    clip_id: str
    frames: tuple[Image.Image, ...]
    question: str
    choices: tuple[tuple[str, str], ...]
    correct_choice_label: str
    ground_truth: dict[str, Any]


@dataclass(frozen=True)
class EventPair:
    """Two paired PIL-frame clips and the fixed generator specification."""

    spec: dict[str, Any]
    clips: tuple[EventClip, EventClip]

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable specification and answer/interval metadata."""
        return {
            "spec": dict(self.spec),
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "question": clip.question,
                    "choices": [
                        {"label": label, "text": text}
                        for label, text in clip.choices
                    ],
                    "correct_choice_label": clip.correct_choice_label,
                    "ground_truth": dict(clip.ground_truth),
                }
                for clip in self.clips
            ],
        }


def _shape_centers(seed: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Use a local seeded RNG so pair geometry is repeatable and process-safe."""
    rng = random.Random(seed)
    circle = (76 + rng.randint(-3, 3), 96 + rng.randint(-3, 3))
    triangle = (190 + rng.randint(-3, 3), 96 + rng.randint(-3, 3))
    return circle, triangle


def _draw_frame(circle_center: tuple[int, int], triangle_center: tuple[int, int],
                *, circle_color: tuple[int, int, int],
                triangle_color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    cx, cy = circle_center
    radius = 32
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 fill=circle_color)
    tx, ty = triangle_center
    draw.polygon(
        ((tx, ty - 38), (tx - 38, ty + 30), (tx + 38, ty + 30)),
        fill=triangle_color,
    )
    return image


def _make_clip(*, clip_id: str, changed_object: str,
               circle_center: tuple[int, int],
               triangle_center: tuple[int, int]) -> EventClip:
    frames = []
    for index in range(FRAME_COUNT):
        during_event = TRANSITION_FRAME <= index < TRANSITION_STOP_FRAME
        frames.append(_draw_frame(
            circle_center,
            triangle_center,
            circle_color=BLUE if during_event and changed_object == "circle" else RED,
            triangle_color=BLUE if during_event and changed_object == "triangle" else RED,
        ))
    choices = _CHOICES
    correct_choice_label = next(
        label for label, text in choices
        if text == ("The circle" if changed_object == "circle" else "The triangle")
    )
    return EventClip(
        clip_id=clip_id,
        frames=tuple(frames),
        question=QUESTION,
        choices=choices,
        correct_choice_label=correct_choice_label,
        ground_truth={
            "changed_object": changed_object,
            "color_transition": {"from": "red", "to": "blue"},
            "transition_frame_index": TRANSITION_FRAME,
            "transition_interval_frames_half_open": [
                TRANSITION_FRAME, TRANSITION_STOP_FRAME],
            "state_intervals_frames_half_open": [
                {
                    "start": 0,
                    "stop": TRANSITION_FRAME,
                    "circle": "red",
                    "triangle": "red",
                },
                {
                    "start": TRANSITION_FRAME,
                    "stop": TRANSITION_STOP_FRAME,
                    "circle": "blue" if changed_object == "circle" else "red",
                    "triangle": "blue" if changed_object == "triangle" else "red",
                },
                {
                    "start": TRANSITION_STOP_FRAME,
                    "stop": FRAME_COUNT,
                    "circle": "red",
                    "triangle": "red",
                },
            ],
        },
    )


def build_synthetic_event_pair(*, seed: int = SEED) -> EventPair:
    """Build the fixed circle-vs-triangle event pair in memory.

    Returns exactly two clips, each containing 32 256x192 RGB PIL frames at
    1 fps. The target briefly turns blue only in frames [16,20), then returns
    to red. Both clips have identical choices; correct labels are B and D.
    No text is rendered into any frame and no files are written.
    """
    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    circle_center, triangle_center = _shape_centers(seed)
    clips = (
        _make_clip(clip_id="circle_changes", changed_object="circle",
                   circle_center=circle_center, triangle_center=triangle_center),
        _make_clip(clip_id="triangle_changes", changed_object="triangle",
                   circle_center=circle_center, triangle_center=triangle_center),
    )
    spec = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "transition_frame_index": TRANSITION_FRAME,
        "transition_stop_frame_exclusive": TRANSITION_STOP_FRAME,
        "event_interval_frames_half_open": [TRANSITION_FRAME, TRANSITION_STOP_FRAME],
        "background_rgb": list(BACKGROUND),
        "red_rgb": list(RED),
        "blue_rgb": list(BLUE),
        "shape_geometry": {
            "circle": {"center_xy": list(circle_center), "radius": 32},
            "triangle": {
                "center_xy": list(triangle_center),
                "vertices_xy": [
                    [triangle_center[0], triangle_center[1] - 38],
                    [triangle_center[0] - 38, triangle_center[1] + 30],
                    [triangle_center[0] + 38, triangle_center[1] + 30],
                ],
            },
        },
        "question": QUESTION,
        "text_rendered_into_frames": False,
        "pair_relation": "same_geometry_background_question_and_choices_only_changed_object_identity_differs",
    }
    return EventPair(spec=spec, clips=clips)
