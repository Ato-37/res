from __future__ import annotations

import unittest
import re

from qta_synthetic_event_construct_v1 import build_assignment as old_assignment
from qta_synthetic_event_construct_v1 import external_smoke_seed as old_smoke
from qta_synthetic_oracle_claim_v1 import (
    EVENT_LENGTH, FAMILIES, FRAME_COUNT, LABEL_ORDERS, OBJECTS, ONSETS,
    assignment_manifest, build_assignment, build_clip, claims_for,
    control_temporal_groups, event_temporal_groups, map_exact_claim_positions,
    render_claim_prompt, select_attention_groups, sham_temporal_groups,
)


class SyntheticOracleClaimGeneratorTests(unittest.TestCase):
    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            ids, offsets = [], []
            for match in re.finditer(r"<video>|[A-Za-z0-9]+|[^\w\s]", text):
                token = match.group(0)
                ids.append(999 if token == "<video>" else len(ids) + 1)
                offsets.append(match.span())
            return {"input_ids": ids, "offset_mapping": offsets}

    class LeadingSpaceTokenizer(FakeTokenizer):
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            encoded = super().__call__(text, add_special_tokens=add_special_tokens,
                                       return_offsets_mapping=return_offsets_mapping)
            start = text.index("The star briefly crosses the gate and returns.")
            for index, (lo, hi) in enumerate(encoded["offset_mapping"]):
                if lo == start:
                    encoded["offset_mapping"][index] = (lo - 1, hi)
                    break
            return encoded

    class TrailingSpaceTokenizer(FakeTokenizer):
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            encoded = super().__call__(text, add_special_tokens=add_special_tokens,
                                       return_offsets_mapping=return_offsets_mapping)
            claim = "The star briefly crosses the gate and returns."
            end = text.index(claim) + len(claim)
            for index, (lo, hi) in enumerate(encoded["offset_mapping"]):
                if lo == end - 1 and hi == end:
                    encoded["offset_mapping"][index] = (lo, hi + 1)
                    break
            return encoded

    def test_assignment_is_deterministic_balanced_and_disjoint(self):
        rows = build_assignment()
        self.assertEqual(rows, build_assignment())
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row.scene_seed for row in rows}), 48)
        old_seeds = {old_smoke(), *(row.scene_seed for row in old_assignment())}
        self.assertFalse(old_seeds & {row.scene_seed for row in rows})
        for family in FAMILIES:
            for onset in ONSETS:
                cell = [row for row in rows
                        if row.family == family and row.onset == onset]
                self.assertEqual(len(cell), 4)
                self.assertEqual({(r.target_object, r.label_order) for r in cell},
                                 {(obj, order) for obj in OBJECTS
                                  for order in LABEL_ORDERS})
        manifest = assignment_manifest()
        self.assertTrue(manifest["smoke_seed_excluded"])
        self.assertFalse(manifest["legacy_cohort_seed_overlap"])
        self.assertEqual(manifest["assignment"], [row.as_dict() for row in rows])

    def test_each_clip_has_static_context_and_four_frame_event_window(self):
        for design in build_assignment():
            clip = build_clip(design)
            self.assertEqual(len(clip.frames), FRAME_COUNT)
            reference = clip.frames[0].tobytes()
            for i in range(design.onset):
                self.assertEqual(clip.frames[i].tobytes(), reference)
            post = clip.frames[-1].tobytes()
            for i in range(design.onset + EVENT_LENGTH, FRAME_COUNT):
                self.assertEqual(clip.frames[i].tobytes(), post)
            self.assertEqual(clip.true_claim.split(" ", 2)[1], design.target_object)
            self.assertNotEqual(clip.true_claim, clip.false_claim)
            self.assertEqual(clip.true_claim.split(" ", 2)[2],
                             clip.false_claim.split(" ", 2)[2])
            self.assertTrue(any(clip.frames[i].tobytes() != reference
                                for i in range(design.onset,
                                               design.onset + EVENT_LENGTH)))

    def test_seed_draws_create_multiple_scene_realizations_per_cell(self):
        rows = build_assignment()
        for family in FAMILIES:
            for onset in ONSETS:
                cell = [row for row in rows
                        if row.family == family and row.onset == onset]
                fingerprints = {(row.gate_x, row.track_offset_y, row.base_radius,
                                 row.background_rgb, row.star_upper,
                                 row.star_uses_color0) for row in cell}
                self.assertGreater(len(fingerprints), 1)

    def test_claim_predicates_are_family_specific(self):
        for family in FAMILIES:
            row = next(row for row in build_assignment() if row.family == family)
            truth, foil = claims_for(row)
            self.assertIn(row.target_object, truth)
            self.assertIn("diamond" if row.target_object == "star" else "star", foil)
            self.assertIn({"gate": "crosses the gate", "rotation": "quarter-turn",
                           "expansion": "1.5 times"}[family], truth)

    def test_prompt_label_mapping_and_exact_claim_span_binding(self):
        claim = "The star briefly crosses the gate and returns."
        body_a, supported_a = render_claim_prompt(claim, "supported_A")
        body_b, supported_b = render_claim_prompt(claim, "supported_B")
        self.assertEqual((supported_a, supported_b), ("A", "B"))
        self.assertIn("A. Supported\nB. Not supported", body_a)
        self.assertIn("A. Not supported\nB. Supported", body_b)
        rendered = "<video>\n" + body_a
        encoded = self.FakeTokenizer()(rendered, add_special_tokens=False,
                                       return_offsets_mapping=True)
        placeholder = encoded["input_ids"].index(999)
        processor_ids = (encoded["input_ids"][:placeholder] + [999] * 6
                         + encoded["input_ids"][placeholder + 1:])
        positions = map_exact_claim_positions(
            tokenizer=self.FakeTokenizer(), rendered_text=rendered, claim=claim,
            processed_input_ids=processor_ids, video_token_id=999)
        self.assertTrue(positions)
        self.assertEqual(positions, tuple(range(positions[0], positions[-1] + 1)))
        whitespace_positions = map_exact_claim_positions(
            tokenizer=self.LeadingSpaceTokenizer(), rendered_text=rendered,
            claim=claim, processed_input_ids=processor_ids, video_token_id=999)
        self.assertEqual(whitespace_positions[-1], positions[-1])
        trailing_positions = map_exact_claim_positions(
            tokenizer=self.TrailingSpaceTokenizer(), rendered_text=rendered,
            claim=claim, processed_input_ids=processor_ids, video_token_id=999)
        self.assertLess(trailing_positions[-1], positions[-1])
        with self.assertRaisesRegex(ValueError, "not_unique"):
            map_exact_claim_positions(
                tokenizer=self.FakeTokenizer(),
                rendered_text=rendered + "\n" + claim, claim=claim,
                processed_input_ids=processor_ids, video_token_id=999)
        with self.assertRaisesRegex(ValueError, "processor_rewrite"):
            map_exact_claim_positions(
                tokenizer=self.FakeTokenizer(), rendered_text=rendered, claim=claim,
                processed_input_ids=processor_ids[:-1], video_token_id=999)

    def test_video_key_assignments_are_deterministic_and_exact(self):
        for onset in ONSETS:
            evidence = event_temporal_groups(onset)
            self.assertEqual(len(evidence), 2)
            controls = control_temporal_groups(scene_seed=123, clip_token="clip-1",
                                               event_groups=evidence)
            self.assertEqual(controls, control_temporal_groups(
                scene_seed=123, clip_token="clip-1", event_groups=evidence))
            self.assertEqual(len(set(controls)), 2)
            self.assertFalse(set(controls) & set(evidence))
            self.assertFalse(set(controls) & {evidence[0] - 1, evidence[-1] + 1})
        self.assertEqual(sham_temporal_groups(scene_seed=12, clip_token="x",
                                              claim_kind="true"),
                         sham_temporal_groups(scene_seed=12, clip_token="x",
                                              claim_kind="true"))
        self.assertNotEqual(sham_temporal_groups(scene_seed=12, clip_token="x",
                                                 claim_kind="true"),
                            sham_temporal_groups(scene_seed=12, clip_token="x",
                                                 claim_kind="false"))
        self.assertEqual(select_attention_groups([0.5] * 16), (0, 1))
        self.assertEqual(select_attention_groups([float(i) for i in range(16)]),
                         (14, 15))


if __name__ == "__main__":
    unittest.main()
