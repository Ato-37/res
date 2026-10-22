"""Frozen procedural stimuli and assignment for synthetic QTA study v1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
import random
from typing import Any

from PIL import Image, ImageDraw


GENERATOR_VERSION = "qta_synthetic_event_construct_v1"
PAIR_COUNT = 40
CLIP_COUNT = 80
SEED_DRAW_COUNT = PAIR_COUNT + 1
FRAME_COUNT = 32
FPS = 1
WIDTH = 256
HEIGHT = 192
EVENT_LENGTH = 4
TEMPORAL_PATCH_SIZE = 2
ONSET_STRATA = (4, 10, 18, 24)
SEED_SAMPLE_RANDOM_SEED = 20261019
SEED_STRATIFY_RANDOM_SEED = 20261020
COMBINATION_RANDOM_SEED_BASE = 20261021
QUESTION = "Which shape briefly switches from red to blue and then returns to red?"
FROZEN_CHOICE_TEXTS = (
    "Both shapes",
    "The circle",
    "Neither shape",
    "The triangle",
    "The background",
)
LABELS = "ABCDE"
BACKGROUND = (250, 250, 250)
RED = (214, 39, 40)
BLUE = (31, 119, 180)


@dataclass(frozen=True)
class PairDesign:
    ordinal: int
    stratum_index: int
    onset_frame: int
    scene_seed: int
    circle_left: bool
    choice_rotation: int
    choices: tuple[tuple[str, str], ...]

    def assignment_row(self) -> dict[str, Any]:
        return {
            "pair_ordinal": self.ordinal,
            "onset_stratum_index": self.stratum_index,
            "event_start_frame": self.onset_frame,
            "scene_seed": self.scene_seed,
            "circle_left": self.circle_left,
            "choice_rotation": self.choice_rotation,
        }


@dataclass(frozen=True)
class ConstructClip:
    changed_object: str
    frames: tuple[Image.Image, ...]
    question: str
    choices: tuple[tuple[str, str], ...]
    correct_choice_label: str
    event_start_frame: int


def _choice_list(rotation: int) -> tuple[tuple[str, str], ...]:
    if type(rotation) is not int or not 0 <= rotation < len(FROZEN_CHOICE_TEXTS):
        raise ValueError("choice_rotation_invalid")
    rotated = FROZEN_CHOICE_TEXTS[rotation:] + FROZEN_CHOICE_TEXTS[:rotation]
    return tuple(zip(LABELS, rotated, strict=True))


def _seed_draw() -> tuple[int, tuple[int, ...]]:
    sampled = random.Random(SEED_SAMPLE_RANDOM_SEED).sample(
        range(2**32), SEED_DRAW_COUNT)
    if len(set(sampled)) != SEED_DRAW_COUNT:
        raise ValueError("seed_draw_not_unique")
    smoke_seed, cohort_seeds = sampled[0], list(sampled[1:])
    random.Random(SEED_STRATIFY_RANDOM_SEED).shuffle(cohort_seeds)
    if smoke_seed in cohort_seeds:
        raise ValueError("external_smoke_seed_overlaps_cohort")
    return smoke_seed, tuple(cohort_seeds)


def external_smoke_seed() -> int:
    """Return the reserved first draw, excluded from all 40 cohort pairs."""
    return _seed_draw()[0]


def build_assignment() -> tuple[PairDesign, ...]:
    """Implement seed draw, seed stratification, and combo shuffles exactly."""
    _smoke_seed, seeds = _seed_draw()
    rows: list[PairDesign] = []
    # Frozen canonical pre-shuffle order: circle-left then circle-right,
    # and within each orientation choice rotations 0..4.
    canonical_combinations = [(True, rotation) for rotation in range(5)]
    canonical_combinations += [(False, rotation) for rotation in range(5)]
    for stratum_index, onset in enumerate(ONSET_STRATA):
        combinations = list(canonical_combinations)
        combination_seed = COMBINATION_RANDOM_SEED_BASE + stratum_index
        random.Random(combination_seed).shuffle(combinations)
        for position, (circle_left, rotation) in enumerate(combinations):
            ordinal = stratum_index * 10 + position
            seed = seeds[ordinal]
            rows.append(PairDesign(
                ordinal=ordinal,
                stratum_index=stratum_index,
                onset_frame=onset,
                scene_seed=seed,
                circle_left=circle_left,
                choice_rotation=rotation,
                choices=_choice_list(rotation),
            ))
    if len(rows) != PAIR_COUNT:
        raise ValueError("assignment_pair_count_invalid")
    return tuple(rows)


def assignment_manifest(*, python_version: str | None = None) -> dict[str, Any]:
    """Return the exact assignment payload; callers hash it, never persist it."""
    smoke_seed, cohort_seeds = _seed_draw()
    rows = build_assignment()
    return {
        "generator_version": GENERATOR_VERSION,
        "python_version": python_version or platform.python_version(),
        "seed_sampling": {
            "population": "uint32_range_exclusive_2_to_32",
            "sample_size": SEED_DRAW_COUNT,
            "random_seed": SEED_SAMPLE_RANDOM_SEED,
            "sampling_without_replacement": True,
            "assignment_shuffle_seed": SEED_STRATIFY_RANDOM_SEED,
            "reserved_external_smoke_seed": smoke_seed,
            "analysis_seed_count": len(cohort_seeds),
            "external_smoke_seed_excluded_from_analysis": (
                smoke_seed not in {row.scene_seed for row in rows}),
        },
        "onset_strata_frames": list(ONSET_STRATA),
        "combination_shuffle_seeds": {
            str(index): COMBINATION_RANDOM_SEED_BASE + index
            for index in range(len(ONSET_STRATA))
        },
        "canonical_combination_order": [
            {"circle_left": left, "choice_rotation": rotation}
            for left in (True, False) for rotation in range(5)
        ],
        "pairs": [row.assignment_row() for row in rows],
    }


def scene_geometry(scene_seed: int, circle_left: bool) -> dict[str, tuple[int, int]]:
    """Return deterministic jittered centers shared by a matched pair."""
    if type(scene_seed) is not int or not 0 <= scene_seed < 2**32:
        raise ValueError("scene_seed_invalid")
    if type(circle_left) is not bool:
        raise ValueError("orientation_invalid")
    rng = random.Random(scene_seed)
    left_center = (76 + rng.randint(-3, 3), 96 + rng.randint(-3, 3))
    right_center = (190 + rng.randint(-3, 3), 96 + rng.randint(-3, 3))
    if circle_left:
        return {"circle": left_center, "triangle": right_center}
    return {"circle": right_center, "triangle": left_center}


def _draw_frame(geometry: dict[str, tuple[int, int]], *, circle_blue: bool,
                triangle_blue: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    cx, cy = geometry["circle"]
    draw.ellipse((cx - 32, cy - 32, cx + 32, cy + 32),
                 fill=BLUE if circle_blue else RED)
    tx, ty = geometry["triangle"]
    draw.polygon(((tx, ty - 38), (tx - 38, ty + 30), (tx + 38, ty + 30)),
                 fill=BLUE if triangle_blue else RED)
    return image


def build_construct_clip(design: PairDesign, changed_object: str) -> ConstructClip:
    """Build exactly one clip so the processor path can stream by clip."""
    if changed_object not in {"circle", "triangle"}:
        raise ValueError("changed_object_invalid")
    geometry = scene_geometry(design.scene_seed, design.circle_left)
    frames = []
    for frame_index in range(FRAME_COUNT):
        active = design.onset_frame <= frame_index < design.onset_frame + EVENT_LENGTH
        frames.append(_draw_frame(
            geometry,
            circle_blue=active and changed_object == "circle",
            triangle_blue=active and changed_object == "triangle",
        ))
    answer_text = "The circle" if changed_object == "circle" else "The triangle"
    correct_label = next(label for label, text in design.choices if text == answer_text)
    return ConstructClip(
        changed_object=changed_object,
        frames=tuple(frames),
        question=QUESTION,
        choices=design.choices,
        correct_choice_label=correct_label,
        event_start_frame=design.onset_frame,
    )


def sha256_canonical(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
