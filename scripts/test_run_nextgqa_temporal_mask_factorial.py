"""CPU contract tests for the exploratory temporal-mask factorial runner."""
import unittest
from collections import Counter
from pathlib import Path
import tempfile

from PIL import Image
import torch

from run_nextgqa_temporal_mask_factorial import (
    CONDITIONS,
    derive_factorial_effects,
    condition_order,
    orient_completed_frames,
    processor_frame_hashes,
    require_complete_factorial,
    require_identical_prompts,
    read_private_preflight,
    source_mask_then_orient,
    validate_private_output_path,
    validate_reversal_hash_pairs,
    validate_signature_order,
    validate_source_binding_receipt,
)


class TemporalMaskFactorialTests(unittest.TestCase):
    def frames(self):
        return [Image.new("RGB", (3, 2), color=(i * 40, 0, 0)) for i in range(4)]

    def test_mask_is_applied_to_original_positions_before_reverse(self):
        frames = self.frames()
        result = source_mask_then_orient(frames, [1], "reversed")
        values = [frame.getpixel((0, 0))[0] for frame in result]
        self.assertEqual(values, [120, 80, 0, 0])
        self.assertEqual(frames[1].getpixel((0, 0))[0], 40)

    def test_signature_contract_accepts_exact_order_and_rejects_permutation(self):
        frames = self.frames()
        chronological = torch.tensor([
            [[[float(v)] * 2] * 3 for v in (i * 40, 0, 0)]
            for i in range(4)
        ])
        reversed_tensor = chronological.flip(0)
        validate_signature_order(frames, chronological, "chronological")
        validate_signature_order(frames, reversed_tensor, "reversed")
        with self.assertRaisesRegex(ValueError, "identity_or_order"):
            validate_signature_order(frames, chronological, "reversed")

    def test_processor_tensor_hashes_enforce_exact_reversal_for_each_mask(self):
        chronological = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
        reversed_tensor = chronological.flip(0)
        hashes = {
            f"{orientation}_{mask}": processor_frame_hashes(
                chronological if orientation == "chronological" else reversed_tensor)
            for orientation in ("chronological", "reversed")
            for mask in ("full", "black_gold_mask", "black_control_mask")
        }
        validate_reversal_hash_pairs(hashes)
        hashes["reversed_black_gold_mask"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reversed_processor_hash"):
            validate_reversal_hash_pairs(hashes)

    def test_all_six_prompts_must_match(self):
        prompts = {name: "same prompt" for name in CONDITIONS}
        require_identical_prompts(prompts)
        prompts[CONDITIONS[-1]] = "different"
        with self.assertRaisesRegex(ValueError, "prompt_text_mismatch"):
            require_identical_prompts(prompts)

    def test_counterbalanced_order_is_deterministic_and_position_balanced(self):
        first_six = [condition_order(i) for i in range(6)]
        self.assertEqual(len(set(first_six)), 6)
        for position in range(6):
            self.assertEqual({row[position] for row in first_six}, set(CONDITIONS))
        orders = [condition_order(i) for i in range(10)]
        counts = Counter(condition for row in orders for condition in row)
        self.assertEqual(counts, Counter({condition: 10 for condition in CONDITIONS}))
        self.assertEqual(condition_order(4), condition_order(4))

    def test_source_binding_receipt_matches_lock_and_media_set(self):
        assignments = [{"media_sha256": f"{i:064x}", "qid_sha256": f"{i+20:064x}"}
                       for i in range(2)]
        receipt = {"schema_version": 1,
                   "purpose": "private_source_binding_for_locked_32f_exploratory_run",
                   "lock_sha256": "a" * 64,
                   "archive_sha256": "b" * 64,
                   "source_commit": "c" * 40,
                   "locked_media": [{"archive_member": f"NExTVideo/{i}.mp4",
                                     "extracted_bytes_sha256": row["media_sha256"],
                                     "media_sha256": row["media_sha256"]}
                                    for i, row in enumerate(assignments)]}
        validate_source_binding_receipt(receipt, "a" * 64, assignments)
        receipt["locked_media"].pop()
        with self.assertRaisesRegex(ValueError, "media_set_mismatch"):
            validate_source_binding_receipt(receipt, "a" * 64, assignments)
        receipt["locked_media"].append({"archive_member": "NExTVideo/0.mp4",
            "extracted_bytes_sha256": "d" * 64,
            "media_sha256": assignments[1]["media_sha256"]})
        with self.assertRaisesRegex(ValueError, "media_row_invalid"):
            validate_source_binding_receipt(receipt, "a" * 64, assignments)
        receipt["locked_media"][1]["extracted_bytes_sha256"] = assignments[1]["media_sha256"]
        with self.assertRaisesRegex(ValueError, "media_set_mismatch"):
            validate_source_binding_receipt(receipt, "a" * 64, assignments)

    def test_output_must_be_inside_private_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            private = repo / "data/private_nextgqa"
            private.mkdir(parents=True)
            self.assertEqual(validate_private_output_path(
                "data/private_nextgqa/result.json", repo), private / "result.json")
            with self.assertRaisesRegex(ValueError, "private_directory"):
                validate_private_output_path("results/public.json", repo)

    def test_preflight_must_be_inside_private_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            private = repo / "data/private_nextgqa"
            private.mkdir(parents=True)
            outside = repo / "results"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "private_nextgqa_directory"):
                read_private_preflight(outside / "preflight.json", repo)

    def test_requires_all_requested_complete_six_condition_sets(self):
        rows = [{"margins": {name: 0.0 for name in CONDITIONS}} for _ in range(10)]
        self.assertEqual(len(require_complete_factorial(rows, [])), 10)
        self.assertEqual(len(require_complete_factorial(rows[:1], [], requested=1)), 1)
        with self.assertRaisesRegex(ValueError, "all_samples_complete"):
            require_complete_factorial(rows[:1], [], requested=10)
        with self.assertRaisesRegex(ValueError, "all_samples_complete"):
            require_complete_factorial(rows[:9], [])
        with self.assertRaisesRegex(ValueError, "all_samples_complete"):
            require_complete_factorial(rows, [{"failure": "synthetic"}])
        rows[0]["margins"].pop(CONDITIONS[-1])
        with self.assertRaisesRegex(ValueError, "six_conditions"):
            require_complete_factorial(rows, [])

    def test_effect_sign_and_interaction_definition(self):
        margins = {
            "chronological_black_control_mask": 3.0,
            "chronological_black_gold_mask": 1.0,
            "chronological_full": 9.0,
            "reversed_black_control_mask": 5.0,
            "reversed_black_gold_mask": 4.0,
            "reversed_full": 8.0,
        }
        result = derive_factorial_effects(margins)
        self.assertEqual(result["chronological_control_minus_gold"], 2.0)
        self.assertEqual(result["reversed_control_minus_gold"], 1.0)
        self.assertEqual(result["interaction_chronological_minus_reversed"], 1.0)


if __name__ == "__main__":
    unittest.main()
