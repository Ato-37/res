"""Synthetic processor tests for local 3 x 3 condition input construction."""
from __future__ import annotations

import unittest
from fractions import Fraction

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    torch = None
    Image = None

from nextgqa_batch_digest_freeze import verify_batch_digest_manifest
from nextgqa_condition_batch_builder import build_exact_triplet_batches


def source_records():
    records = []
    for row in range(3):
        frames = []
        for i in range(32):
            yy, xx = np.mgrid[0:8, 0:8]
            pixels = np.stack((
                (row * 19 + i * 3 + xx * 7) % 255,
                (20 + yy * 23 + i) % 255,
                (40 + xx * 5 + yy * 11) % 255,
            ), axis=-1).astype(np.uint8)
            frames.append(Image.fromarray(pixels, mode="RGB"))
        records.append({
            "frames": frames,
            "presentation_times": [Fraction(i, 1) for i in range(32)],
            "sampled_indices": list(range(32)),
            "intervals": [[Fraction(5), Fraction(8)]],
            "qa_item": {"question": f"Q{row}?", "choices": ["a", "b", "c", "d", "e"],
                        "gold": row},
            "sample_fps": Fraction(1, 1),
        })
    return records


class FakeProcessor:
    def __init__(self, perturb_nonmask_call=None, bad_prefix=False,
                 identical_pixels=False):
        self.calls = 0
        self.perturb_nonmask_call = perturb_nonmask_call
        self.bad_prefix = bad_prefix
        self.identical_pixels = identical_pixels
        self.template_calls = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.template_calls += 1
        assert tokenize is False and add_generation_prompt is True
        text = messages[0]["content"][1]["text"]
        if self.bad_prefix and self.template_calls == 2:
            text += " changed"
        return "<video>\n" + text

    def __call__(self, *, text, images, videos, video_metadata, do_sample_frames,
                 cap_pixels_per_frame, padding, return_tensors):
        self.calls += 1
        assert images is None
        assert len(videos) == len(video_metadata) == 1
        assert do_sample_frames is False and cap_pixels_per_frame is True
        assert padding is True and return_tensors == "pt"
        metadata = video_metadata[0]
        ids = [ord(char) % 251 for char in text[0]]
        if self.perturb_nonmask_call == self.calls:
            ids[0] = (ids[0] + 1) % 251
        pixels = videos[0]
        if self.identical_pixels:
            pixels = torch.zeros_like(pixels)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.int64),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.int64),
            "pixel_values_videos": pixels.clone(),
            "video_grid_thw": torch.tensor([[8, 1, 1]], dtype=torch.int64),
            "second_per_grid_ts": torch.tensor([4.0], dtype=torch.float32),
            "video_metadata": [metadata],
            "mm_token_type_ids": torch.zeros((1, len(ids)), dtype=torch.int64),
        }


class FakeVisionInfo:
    def __init__(self, bad_fps=False, reorder_pixels=False, flip_pixels=False,
                 alternate_shape=None):
        self.calls = 0
        self.bad_fps = bad_fps
        self.reorder_pixels = reorder_pixels
        self.flip_pixels = flip_pixels
        self.alternate_shape = alternate_shape

    def __call__(self, messages, return_video_metadata):
        self.calls += 1
        assert return_video_metadata is True
        video = messages[0]["content"][0]
        frames = video["video"]
        fps = video["sample_fps"]
        if self.bad_fps:
            fps += 1
        from qwen_vl_utils.vision_process import fetch_video
        pixels, metadata = fetch_video(video, return_video_metadata=True)
        if self.alternate_shape is not None:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import resize
            pixels = resize(pixels, list(self.alternate_shape),
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True)
        if self.reorder_pixels:
            pixels = pixels.flip(0)
        if self.flip_pixels:
            pixels = pixels.flip(-1)
        if self.bad_fps:
            metadata["fps"] = fps
        return None, [(pixels, metadata)]


@unittest.skipIf(torch is None or Image is None or np is None, "torch/Pillow/NumPy unavailable")
class NextgqaConditionBatchBuilderTests(unittest.TestCase):
    def test_builds_all_nine_batches_and_rehashes_manifest(self):
        processor, vision = FakeProcessor(), FakeVisionInfo()
        result = build_exact_triplet_batches(records=source_records(), processor=processor,
                                             vision_info_fn=vision)
        self.assertEqual(processor.calls, 9)
        self.assertEqual(vision.calls, 9)
        self.assertEqual(result["requested_batches"], 9)
        self.assertEqual(result["model_forwards"], 0)
        self.assertFalse(result["model_loaded"])
        self.assertTrue(result["frame_pixel_identity_verified"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertTrue(verify_batch_digest_manifest(
            result["triplets"], result["digest_manifest"]))

    def test_masks_exact_sampled_ranks_and_keeps_other_inputs_fixed(self):
        result = build_exact_triplet_batches(records=source_records(),
                                             processor=FakeProcessor(),
                                             vision_info_fn=FakeVisionInfo())
        row = result["triplets"][0]
        meta = result["condition_metadata"][0]["assignment"]
        evidence = meta["evidence_ranks"]
        controls = meta["control_ranks"]
        self.assertEqual(len(evidence), 4)
        self.assertEqual(len(controls), 4)
        full = row["batches"]["full"]["pixel_values_videos"]
        evidence_pixels = row["batches"]["claim_evidence_black"]["pixel_values_videos"]
        control_pixels = row["batches"]["temporal_control_black"]["pixel_values_videos"]
        self.assertTrue(torch.equal(evidence_pixels[evidence], torch.zeros_like(evidence_pixels[evidence])))
        self.assertTrue(torch.equal(control_pixels[controls], torch.zeros_like(control_pixels[controls])))
        self.assertTrue(torch.equal(evidence_pixels[controls], full[controls]))
        self.assertTrue(torch.equal(control_pixels[evidence], full[evidence]))
        for condition in ("claim_evidence_black", "temporal_control_black"):
            for key in ("input_ids", "attention_mask", "video_grid_thw",
                        "second_per_grid_ts", "video_metadata", "mm_token_type_ids"):
                left, right = row["batches"]["full"][key], row["batches"][condition][key]
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right), key)
                else:
                    self.assertEqual(left, right)

    def test_bad_third_row_assignment_fails_before_any_processor_call(self):
        rows = source_records()
        rows[2]["intervals"] = [[Fraction(1000), Fraction(1001)]]
        processor, vision = FakeProcessor(), FakeVisionInfo()
        with self.assertRaises(ValueError):
            build_exact_triplet_batches(records=rows, processor=processor,
                                        vision_info_fn=vision)
        self.assertEqual(processor.calls, 0)
        self.assertEqual(vision.calls, 0)

    def test_processor_nonmask_change_blocks_result(self):
        processor = FakeProcessor(perturb_nonmask_call=5)
        with self.assertRaisesRegex(ValueError, "nonmask_processor_input_changed"):
            build_exact_triplet_batches(records=source_records(), processor=processor,
                                        vision_info_fn=FakeVisionInfo())

    def test_identical_processed_condition_pixels_block_batch(self):
        processor = FakeProcessor(identical_pixels=True)
        with self.assertRaisesRegex(ValueError,
                                   "condition_pixel_tensors_not_distinct"):
            build_exact_triplet_batches(records=source_records(), processor=processor,
                                        vision_info_fn=FakeVisionInfo())
        self.assertEqual(processor.calls, 9)

    def test_prefix_or_video_geometry_mismatch_blocks_result(self):
        with self.assertRaisesRegex(ValueError, "condition_prompt_mismatch"):
            build_exact_triplet_batches(records=source_records(),
                                        processor=FakeProcessor(bad_prefix=True),
                                        vision_info_fn=FakeVisionInfo())
        with self.assertRaisesRegex(ValueError, "video_geometry_invalid"):
            build_exact_triplet_batches(records=source_records(), processor=FakeProcessor(),
                                        vision_info_fn=FakeVisionInfo(bad_fps=True))

    def test_reordered_vision_pixels_fail_before_processor_call(self):
        for vision in (FakeVisionInfo(reorder_pixels=True), FakeVisionInfo(flip_pixels=True)):
            processor = FakeProcessor()
            with self.assertRaisesRegex(ValueError, "frame_pixel_identity_mismatch"):
                build_exact_triplet_batches(records=source_records(), processor=processor,
                                            vision_info_fn=vision)
            self.assertEqual(processor.calls, 0)

    def test_alternate_but_budget_valid_resize_fails_before_processor_call(self):
        processor = FakeProcessor()
        # The synthetic source is 8x8, which the pinned utility first raises
        # to its 128-token/frame minimum. 364x364 remains a valid 28-multiple
        # within that frame budget but is not the utility-derived target.
        vision = FakeVisionInfo(alternate_shape=(364, 364))
        with self.assertRaisesRegex(ValueError, "frame_pixel_identity_mismatch"):
            build_exact_triplet_batches(records=source_records(), processor=processor,
                                        vision_info_fn=vision)
        self.assertEqual(processor.calls, 0)


if __name__ == "__main__":
    unittest.main()
