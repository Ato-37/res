"""CPU contract tests for question-span mapping and temporal-group selection."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_question_temporal_attention_selector import (  # noqa: E402
    aggregate_temporal_group_attention,
    map_question_token_positions,
    select_hash_sham_temporal_groups,
    select_top_temporal_groups,
)


class _CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        ids = [99 if char == "V" else ord(char) for char in text]
        return {"input_ids": ids,
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class QuestionTemporalSelectorTests(unittest.TestCase):
    def test_maps_question_after_single_expanded_video_placeholder(self):
        text = "VQuestion?\nOptions"
        processed = [99, 99, 99, *[ord(char) for char in text[1:]]]
        result = map_question_token_positions(
            tokenizer=_CharacterTokenizer(), rendered_text=text,
            question_text="Question?", processed_input_ids=processed,
            video_token_id=99)
        self.assertEqual(result, list(range(3, 12)))

    def test_rejects_non_single_placeholder_rewrite_and_ambiguous_question(self):
        with self.assertRaisesRegex(ValueError, "single_video_expansion"):
            map_question_token_positions(
                tokenizer=_CharacterTokenizer(), rendered_text="VQ",
                question_text="Q", processed_input_ids=[99, 99, 81, 120],
                video_token_id=99)
        with self.assertRaisesRegex(ValueError, "not_unique"):
            map_question_token_positions(
                tokenizer=_CharacterTokenizer(), rendered_text="VQ Q",
                question_text="Q", processed_input_ids=[99, 81, 32, 81],
                video_token_id=99)
        with self.assertRaisesRegex(ValueError, "not_unique"):
            map_question_token_positions(
                tokenizer=_CharacterTokenizer(), rendered_text="Vaaa",
                question_text="aa", processed_input_ids=[99, 97, 97, 97],
                video_token_id=99)

    def test_aggregates_question_attention_by_complete_temporal_group(self):
        import torch

        weights = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
        # The two question queries place most of their mass on group 1.
        for head in range(2):
            weights[0, head, 6, 2:4] = 0.2
            weights[0, head, 6, 0:2] = 0.05
            weights[0, head, 6, 4:6] = 0.1
            weights[0, head, 7, 2:4] = 0.15
            weights[0, head, 7, 0:2] = 0.1
            weights[0, head, 7, 4:6] = 0.05
        scores = aggregate_temporal_group_attention(
            attention_by_layer={4: weights, 8: weights.clone()},
            layer_indices=(4, 8), question_positions=(6, 7),
            group_positions=((0, 1), (2, 3), (4, 5)), torch_module=torch)
        for actual, expected in zip(scores, [0.15, 0.35, 0.15], strict=True):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(select_top_temporal_groups(scores, top_k=2), [1, 0])

    def test_selector_ties_and_invalid_group_layout_fail_closed(self):
        import torch

        self.assertEqual(select_top_temporal_groups([1.0, 1.0, 0.5], top_k=2),
                         [0, 1])
        weights = torch.ones((1, 1, 5, 5), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "temporal_groups_overlap"):
            aggregate_temporal_group_attention(
                attention_by_layer={0: weights}, layer_indices=(0,),
                question_positions=(4,), group_positions=((0, 1), (1, 2)),
                torch_module=torch)
        with self.assertRaisesRegex(ValueError, "temporal_group_size_mismatch"):
            aggregate_temporal_group_attention(
                attention_by_layer={0: weights}, layer_indices=(0,),
                question_positions=(4,), group_positions=((0,), (1, 2)),
                torch_module=torch)
        with self.assertRaisesRegex(ValueError, "temporal_group_order_invalid"):
            aggregate_temporal_group_attention(
                attention_by_layer={0: weights}, layer_indices=(0,),
                question_positions=(4,), group_positions=((2, 3), (0, 1)),
                torch_module=torch)
        with self.assertRaisesRegex(ValueError, "selector_inputs_invalid"):
            aggregate_temporal_group_attention(
                attention_by_layer={0: weights}, layer_indices=(0,),
                question_positions=(4, 4), group_positions=((0, 1), (2, 3)),
                torch_module=torch)
        with self.assertRaisesRegex(ValueError, "not_causal"):
            aggregate_temporal_group_attention(
                attention_by_layer={0: weights}, layer_indices=(0,),
                question_positions=(3,), group_positions=((0, 1), (2, 3)),
                torch_module=torch)

    def test_hash_sham_is_stable_sorted_and_record_specific(self):
        selected = select_hash_sham_temporal_groups(
            temporal_group_count=16, top_k=2, protocol_seed="seed-v1",
            opaque_record_token="opaque-a")
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(selected, select_hash_sham_temporal_groups(
            temporal_group_count=16, top_k=2, protocol_seed="seed-v1",
            opaque_record_token="opaque-a"))
        self.assertNotEqual(selected, select_hash_sham_temporal_groups(
            temporal_group_count=16, top_k=2, protocol_seed="seed-v1",
            opaque_record_token="opaque-b"))
        with self.assertRaisesRegex(ValueError, "inputs_invalid"):
            select_hash_sham_temporal_groups(
                temporal_group_count=16, top_k=17, protocol_seed="seed-v1",
                opaque_record_token="opaque-a")


if __name__ == "__main__":
    unittest.main()
