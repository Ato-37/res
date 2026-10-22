"""Versioned in-memory stimuli for synthetic oracle-claim evidence tests.

This generator is a construct benchmark only. It writes no media and makes
no claim about natural-video or free-form explanation faithfulness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import platform
import random
from typing import Any

from PIL import Image, ImageChops, ImageDraw


GENERATOR_VERSION = "qta_synthetic_oracle_claim_v1"
MASTER_SEED = 20261021
FRAME_COUNT = 32
FPS = 1
WIDTH = 256
HEIGHT = 192
FAMILIES = ("gate", "rotation", "expansion")
ONSETS = (4, 10, 18, 24)
EVENT_LENGTH = 4
OBJECTS = ("star", "diamond")
LABEL_ORDERS = ("supported_A", "supported_B")
BACKGROUNDS = ((245, 245, 245), (235, 239, 243), (250, 244, 234))
OBJECT_COLORS = ((51, 102, 204), (204, 102, 51))


@dataclass(frozen=True)
class SceneDesign:
    family: str
    onset: int
    replicate: int
    scene_seed: int
    target_object: str
    label_order: str
    gate_x: int
    track_offset_y: int
    base_radius: int
    background_rgb: tuple[int, int, int]
    star_upper: bool
    star_uses_color0: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["background_rgb"] = list(self.background_rgb)
        return value


@dataclass(frozen=True)
class OracleClip:
    design: SceneDesign
    frames: tuple[Image.Image, ...]
    true_claim: str
    false_claim: str


def sha256_canonical(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def event_temporal_groups(onset: int, temporal_group_count: int = 16) -> tuple[int, ...]:
    """Map the four source frames to groups of two consecutive source frames."""
    if (type(onset) is not int or onset not in ONSETS
            or type(temporal_group_count) is not int or temporal_group_count != 16):
        raise ValueError("event_group_geometry_invalid")
    mapped = tuple(group for group in range(temporal_group_count)
                   if set(range(2 * group, 2 * group + 2))
                   & set(range(onset, onset + EVENT_LENGTH)))
    if mapped != (onset // 2, onset // 2 + 1):
        raise ValueError("event_group_mapping_invalid")
    return mapped


def control_temporal_groups(*, scene_seed: int, clip_token: str,
                            event_groups: tuple[int, ...],
                            temporal_group_count: int = 16) -> tuple[int, int]:
    """Choose a deterministic hash-ranked eligible pair, excluding event neighbors."""
    if (type(scene_seed) is not int or scene_seed < 0 or not isinstance(clip_token, str)
            or not clip_token or temporal_group_count < 5
            or len(event_groups) != 2):
        raise ValueError("control_group_input_invalid")
    excluded = {g for event in event_groups
                for g in (event - 1, event, event + 1)
                if 0 <= g < temporal_group_count}
    candidates = list(itertools.combinations(
        (g for g in range(temporal_group_count) if g not in excluded), 2))
    if not candidates:
        raise ValueError("control_group_candidates_empty")
    seed = f"control-v1|{scene_seed}|{clip_token}|"
    return min(candidates, key=lambda pair: (
        hashlib.sha256(f"{seed}{pair[0]}|{pair[1]}".encode("ascii")).hexdigest(),
        pair))


def sham_temporal_groups(*, scene_seed: int, clip_token: str, claim_kind: str,
                         temporal_group_count: int = 16) -> tuple[int, int]:
    """Choose a deterministic equal-budget pair independently of treatment."""
    if (type(scene_seed) is not int or scene_seed < 0 or not isinstance(clip_token, str)
            or not clip_token or claim_kind not in {"true", "false"}
            or type(temporal_group_count) is not int or temporal_group_count < 2):
        raise ValueError("sham_group_input_invalid")
    candidates = itertools.combinations(range(temporal_group_count), 2)
    seed = f"sham-v1|{scene_seed}|{clip_token}|{claim_kind}|"
    return min(candidates, key=lambda pair: (
        hashlib.sha256(f"{seed}{pair[0]}|{pair[1]}".encode("ascii")).hexdigest(),
        pair))


def select_attention_groups(masses: list[float], *, fraction: float = 0.10) -> tuple[int, ...]:
    """Select highest-mass groups, breaking ties toward earlier time."""
    if (not isinstance(masses, list) or not masses
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) or float(x) < 0 for x in masses)
            or isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1):
        raise ValueError("attention_group_selection_input_invalid")
    count = max(1, math.ceil(float(fraction) * len(masses)))
    return tuple(sorted(sorted(range(len(masses)), key=lambda i: (-float(masses[i]), i))[:count]))


def _legacy_reserved_seeds() -> set[int]:
    """Import the closed cohort's seed set so reuse fails closed."""
    from qta_synthetic_event_construct_v1 import (
        build_assignment, external_smoke_seed,
    )
    return {external_smoke_seed(), *(row.scene_seed for row in build_assignment())}


def _seed_population() -> tuple[int, ...]:
    reserved = _legacy_reserved_seeds()
    for attempt in range(1000):
        rng = random.Random(MASTER_SEED + attempt)
        values = tuple(rng.sample(range(2**32), len(FAMILIES) * len(ONSETS) * 4 + 1))
        smoke, seeds = values[0], values[1:]
        if len(set(values)) == len(values) and not (set(values) & reserved):
            return (smoke, *seeds)
    raise RuntimeError("seed_population_reservation_exhausted")


def build_assignment() -> tuple[SceneDesign, ...]:
    """Create 48 distinct scene seeds with balanced 2x2 cell assignments."""
    values = _seed_population()
    rows: list[SceneDesign] = []
    seed_cursor = 1
    for family_index, family in enumerate(FAMILIES):
        for onset_index, onset in enumerate(ONSETS):
            combinations = [(obj, order) for obj in OBJECTS for order in LABEL_ORDERS]
            order_seed = MASTER_SEED + 10_000 + family_index * len(ONSETS) + onset_index
            random.Random(order_seed).shuffle(combinations)
            for replicate, (seed_target, seed_order) in enumerate(combinations):
                scene_seed = values[seed_cursor]
                seed_cursor += 1
                rng = random.Random(scene_seed)
                rows.append(SceneDesign(
                    family=family,
                    onset=onset,
                    replicate=replicate,
                    scene_seed=scene_seed,
                    target_object=seed_target,
                    label_order=seed_order,
                    gate_x=rng.choice((112, 128, 144)),
                    track_offset_y=rng.choice((-8, 0, 8)),
                    base_radius=rng.choice((24, 28, 32)),
                    background_rgb=rng.choice(BACKGROUNDS),
                    star_upper=bool(rng.randrange(2)),
                    star_uses_color0=bool(rng.randrange(2)),
                ))
    if len(rows) != 48 or len({row.scene_seed for row in rows}) != 48:
        raise RuntimeError("assignment_count_or_seed_uniqueness_invalid")
    return tuple(rows)


def assignment_manifest() -> dict[str, Any]:
    seeds = _seed_population()
    rows = build_assignment()
    if rows and seeds[0] in {row.scene_seed for row in rows}:
        raise RuntimeError("smoke_seed_overlaps_cohort")
    return {
        "generator_version": GENERATOR_VERSION,
        "python_version": platform.python_version(),
        "master_seed": MASTER_SEED,
        "seed_sampling": "random.Random.sample uint32 without replacement; attempt seed increments on collision",
        "reserved_smoke_seed": seeds[0],
        "smoke_seed_excluded": True,
        "legacy_cohort_seed_overlap": False,
        "families": list(FAMILIES),
        "onsets": list(ONSETS),
        "event_length_frames": EVENT_LENGTH,
        "assignment": [row.as_dict() for row in rows],
    }


def _rotate(points: list[tuple[float, float]], degrees: float) -> list[tuple[float, float]]:
    import math
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def _shape_polygon(kind: str, x: float, y: float, radius: float,
                   angle: float) -> list[tuple[float, float]]:
    import math
    if kind == "diamond":
        # An asymmetric kite avoids a 90-degree rotation aliasing to itself.
        base = [(0, -radius), (0.72 * radius, 0),
                (0, 0.64 * radius), (-0.88 * radius, 0)]
    elif kind == "star":
        base = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            base.append((r * math.cos(a), r * math.sin(a)))
    else:
        raise ValueError("object_kind_invalid")
    return [(x + dx, y + dy) for dx, dy in _rotate(base, angle)]


def object_state(design: SceneDesign, object_name: str,
                 frame_index: int) -> tuple[float, float, float, float]:
    if (object_name not in OBJECTS or type(frame_index) is not int
            or not 0 <= frame_index < FRAME_COUNT):
        raise ValueError("object_state_input_invalid")
    y_by_object = {
        "star": 60 + design.track_offset_y if design.star_upper else 132 + design.track_offset_y,
        "diamond": 132 + design.track_offset_y if design.star_upper else 60 + design.track_offset_y,
    }
    active = design.onset <= frame_index < design.onset + EVENT_LENGTH
    event_offset = frame_index - design.onset if active else -1
    x, y = 80.0, float(y_by_object[object_name])
    radius, angle = float(design.base_radius), 0.0
    if active and object_name == design.target_object:
        if design.family == "gate":
            x = (80, design.gate_x + 12,
                 design.gate_x + 48, 80)[event_offset]
        elif design.family == "rotation":
            angle = (0.0, 90.0, 90.0, 0.0)[event_offset]
        elif design.family == "expansion":
            radius *= (1.0, 1.5, 1.5, 1.0)[event_offset]
        else:
            raise ValueError("event_family_invalid")
    return float(x), float(y), radius, angle


def _draw_frame(design: SceneDesign, frame_index: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), design.background_rgb)
    draw = ImageDraw.Draw(image)
    draw.line((design.gate_x, 20, design.gate_x, HEIGHT - 20),
              fill=(20, 20, 20), width=3)
    active = design.onset <= frame_index < design.onset + EVENT_LENGTH
    event_offset = frame_index - design.onset if active else -1
    states: dict[str, tuple[float, float, float, float]] = {}
    masks: dict[str, Image.Image] = {}
    for obj in OBJECTS:
        states[obj] = object_state(design, obj, frame_index)
        x, y, radius, angle = states[obj]
        color0_for_object = design.star_uses_color0 if obj == "star" else not design.star_uses_color0
        color = OBJECT_COLORS[0 if color0_for_object else 1]
        polygon = _shape_polygon(obj, x, y, radius, angle)
        if any(px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT
               for px, py in polygon):
            raise AssertionError("object_geometry_out_of_bounds")
        object_mask = Image.new("1", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(object_mask).polygon(polygon, fill=255)
        masks[obj] = object_mask
        draw.polygon(polygon, fill=color)
    if ImageChops.logical_and(masks["star"], masks["diamond"]).getbbox() is not None:
        raise AssertionError("object_raster_overlap")
    # Assertions are deliberately attached to the renderer, not just metadata.
    if active:
        target = states[design.target_object]
        foil = states["diamond" if design.target_object == "star" else "star"]
        if design.family == "gate" and event_offset in (1, 2):
            if not ((target[0] - design.gate_x) * (foil[0] - design.gate_x) < 0):
                raise AssertionError("gate_target_state_invalid")
            if foil[0] != 80 or target[0] != (design.gate_x + 12,
                                              design.gate_x + 48)[event_offset - 1]:
                raise AssertionError("gate_foil_or_trajectory_invalid")
        if design.family == "rotation" and event_offset in (1, 2):
            if target[3] != 90.0 or foil[3] != 0.0:
                raise AssertionError("rotation_target_state_invalid")
        if design.family == "expansion" and event_offset in (1, 2):
            if target[2] != design.base_radius * 1.5 or foil[2] != design.base_radius:
                raise AssertionError("expansion_target_state_invalid")
    return image


def claims_for(design: SceneDesign) -> tuple[str, str]:
    predicate = {
        "gate": "briefly crosses the gate and returns",
        "rotation": "briefly rotates by a quarter-turn and returns",
        "expansion": "briefly grows to 1.5 times its initial size and returns",
    }[design.family]
    foil = "diamond" if design.target_object == "star" else "star"
    return (f"The {design.target_object} {predicate}.",
            f"The {foil} {predicate}.")


def render_claim_prompt(claim: str, label_order: str) -> tuple[str, str]:
    """Return body text and the label mapped to Supported (A or B)."""
    if not isinstance(claim, str) or not claim.strip() or label_order not in LABEL_ORDERS:
        raise ValueError("claim_prompt_input_invalid")
    if label_order == "supported_A":
        choices = "A. Supported\nB. Not supported"
        supported_label = "A"
    else:
        choices = "A. Not supported\nB. Supported"
        supported_label = "B"
    return (f"Does this video support the following statement?\n"
            f"Statement: {claim}\n{choices}\nAnswer:", supported_label)


def map_exact_claim_positions(*, tokenizer: Any, rendered_text: str,
                              claim: str, processed_input_ids: Any,
                              video_token_id: int) -> tuple[int, ...]:
    """Map a unique exact claim span through one processor-expanded video token.

    Selects only tokenizer tokens fully contained in the claim's character
    interval. Boundary-crossing, ambiguous, or processor-rewritten prompts
    fail closed. This function assumes a single, unpadded sample.
    """
    if (not isinstance(rendered_text, str) or not rendered_text
            or not isinstance(claim, str) or not claim
            or type(video_token_id) is not int or video_token_id < 0):
        raise ValueError("claim_span_input_invalid")
    first = rendered_text.find(claim)
    if first < 0 or rendered_text.find(claim, first + 1) >= 0:
        raise ValueError("claim_span_not_unique")
    start, end = first, first + len(claim)
    try:
        encoded = tokenizer(rendered_text, add_special_tokens=False,
                            return_offsets_mapping=True)
        raw_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception:
        raise ValueError("claim_span_tokenization_failed") from None
    if (not isinstance(raw_ids, (list, tuple))
            or not isinstance(offsets, (list, tuple))
            or not raw_ids or len(raw_ids) != len(offsets)
            or any(type(token) is not int or token < 0 for token in raw_ids)):
        raise ValueError("claim_span_tokenization_invalid")
    placeholder_positions = [i for i, token in enumerate(raw_ids)
                             if token == video_token_id]
    if len(placeholder_positions) != 1:
        raise ValueError("claim_span_video_placeholder_invalid")
    placeholder = placeholder_positions[0]
    if hasattr(processed_input_ids, "detach"):
        processed_input_ids = processed_input_ids.detach().cpu().tolist()
    if not isinstance(processed_input_ids, (list, tuple)):
        raise ValueError("claim_span_processed_ids_invalid")
    if processed_input_ids and isinstance(processed_input_ids[0], (list, tuple)):
        if len(processed_input_ids) != 1:
            raise ValueError("claim_span_processed_batch_size_invalid")
        processed_input_ids = processed_input_ids[0]
    processed = list(processed_input_ids)
    if any(type(token) is not int or token < 0 for token in processed):
        raise ValueError("claim_span_processed_ids_invalid")
    visual = [i for i, token in enumerate(processed) if token == video_token_id]
    if (not visual or visual != list(range(visual[0], visual[-1] + 1))):
        raise ValueError("claim_span_visual_run_invalid")
    expected = (list(raw_ids[:placeholder]) + [video_token_id] * len(visual)
                + list(raw_ids[placeholder + 1:]))
    if processed != expected:
        raise ValueError("claim_span_processor_rewrite")
    if placeholder >= len(raw_ids) or visual[0] >= visual[-1] + 1:
        raise ValueError("claim_span_video_geometry_invalid")
    selected: list[int] = []
    prior_start = prior_end = 0
    for index, offset in enumerate(offsets):
        if (not isinstance(offset, (list, tuple)) or len(offset) != 2
                or type(offset[0]) is not int or type(offset[1]) is not int
                or offset[0] < 0 or offset[1] < offset[0]
                or offset[1] > len(rendered_text)
                or (index and (offset[0] < prior_start or offset[0] < prior_end))):
            raise ValueError("claim_span_offsets_invalid")
        lo, hi = offset
        prior_start, prior_end = lo, hi
        overlaps = hi > start and lo < end
        if overlaps and hi > end:
            if rendered_text[end:hi].isspace():
                # Fast tokenizers can attach the following line break to the
                # final punctuation token. It is not wholly inside the claim.
                continue
            raise ValueError("claim_span_token_crosses_right_boundary")
        if overlaps and lo < start:
            if not rendered_text[lo:start].isspace():
                raise ValueError("claim_span_token_crosses_left_boundary")
            # SentencePiece may include the separator whitespace in the first
            # word token. Ignore that boundary token; selection uses the last
            # wholly contained claim token below.
            continue
        if lo >= start and hi <= end and hi > lo:
            selected.append(index)
    if (not selected or selected != list(range(selected[0], selected[-1] + 1))
            or selected[0] <= placeholder):
        raise ValueError("claim_span_token_positions_invalid")
    shift = len(visual) - 1
    mapped = tuple(i if i < placeholder else i + shift for i in selected)
    if mapped != tuple(range(mapped[0], mapped[-1] + 1)):
        raise ValueError("claim_span_mapped_positions_invalid")
    return mapped


def build_clip(design: SceneDesign) -> OracleClip:
    if design.family not in FAMILIES or design.onset not in ONSETS:
        raise ValueError("scene_design_invalid")
    frames = tuple(_draw_frame(design, i) for i in range(FRAME_COUNT))
    target_states = [object_state(design, design.target_object, i)
                     for i in range(FRAME_COUNT)]
    foil_name = "diamond" if design.target_object == "star" else "star"
    foil_states = [object_state(design, foil_name, i) for i in range(FRAME_COUNT)]
    event_states = target_states[design.onset:design.onset + EVENT_LENGTH]
    if design.family == "gate":
        if (target_states[0][0] != 80 or event_states[0][0] != 80
                or event_states[-1][0] != 80
                or not all(state[0] > design.gate_x for state in event_states[1:3])
                or any(state[0] != 80 for state in foil_states)):
            raise AssertionError("gate_truth_or_foil_invalid")
    elif design.family == "rotation":
        if ([state[3] for state in event_states] != [0.0, 90.0, 90.0, 0.0]
                or any(state[3] != 0.0 for state in foil_states)):
            raise AssertionError("rotation_truth_or_foil_invalid")
    elif design.family == "expansion":
        if ([state[2] / design.base_radius for state in event_states]
                != [1.0, 1.5, 1.5, 1.0]
                or any(state[2] != design.base_radius for state in foil_states)):
            raise AssertionError("expansion_truth_or_foil_invalid")
    true_claim, false_claim = claims_for(design)
    if frames[0].tobytes() != frames[design.onset - 1].tobytes():
        raise AssertionError("pre_event_frames_changed")
    if frames[-1].tobytes() != frames[design.onset + EVENT_LENGTH].tobytes():
        raise AssertionError("post_event_frames_changed")
    if any(frames[i].tobytes() != frames[0].tobytes()
           for i in range(design.onset)):
        raise AssertionError("pre_event_static_state_invalid")
    if any(frames[i].tobytes() != frames[-1].tobytes()
           for i in range(design.onset + EVENT_LENGTH, FRAME_COUNT)):
        raise AssertionError("post_event_static_state_invalid")
    if (frames[design.onset].tobytes() != frames[0].tobytes()
            or frames[design.onset + 3].tobytes() != frames[0].tobytes()
            or any(frames[i].tobytes() == frames[0].tobytes()
                   for i in (design.onset + 1, design.onset + 2))):
        raise AssertionError("transient_event_pixel_change_invalid")
    return OracleClip(design, frames, true_claim, false_claim)
