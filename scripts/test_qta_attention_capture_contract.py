from __future__ import annotations

import copy
import hashlib
import struct
import unittest
from collections.abc import Sequence

from qta_attention_capture_contract import (
    LAYERS,
    build_attention_capture,
    validate_attention_capture,
)
from qwen_question_temporal_attention_selector import aggregate_temporal_group_attention


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class AttentionCaptureContractTests(unittest.TestCase):
    def setUp(self):
        self.binding = {
            "record_token": sha("record"),
            "row_binding_sha256": sha("row"),
            "typed_batch_sha256": sha("batch"),
            "baseline_pq_row_sha256": sha("baseline"),
            "prompt_sha256": sha("prompt"),
            "input_ids_sha256": sha("ids"),
            "model_sha256": sha("model"),
            "processor_sha256": sha("processor"),
            "tokenizer_sha256": sha("tokenizer"),
            "runtime_sha256": sha("runtime"),
            "question_positions": [4, 5],
            "temporal_groups": [[0, 1], [2, 3]],
        }
        # Each query row is a normalized eight-key distribution. Values are
        # deterministic fixtures, not outputs from a model forward.
        self.attention = [
            [
                [[0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15],
                 [0.2, 0.05, 0.1, 0.15, 0.1, 0.1, 0.15, 0.15]],
                [[0.05, 0.15, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15],
                 [0.15, 0.1, 0.05, 0.2, 0.1, 0.15, 0.1, 0.15]],
            ]
            for _ in LAYERS
        ]
        self.capture = build_attention_capture(binding=self.binding,
                                               attention=self.attention,
                                               input_length=8)

    def test_synthetic_bytes_and_group_masses_recompute(self):
        report = validate_attention_capture(self.capture,
                                            expected_binding=self.binding,
                                            input_length=8)
        self.assertEqual(report["layer_count"], 12)
        self.assertEqual(report["head_count"], 2)
        self.assertEqual(report["temporal_group_count"], 2)
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["scoring_authorized"])

    def test_rejects_bytes_changed_without_digest_update(self):
        changed = copy.deepcopy(self.capture)
        changed["attention_f32_be_hex"] = "0" + changed["attention_f32_be_hex"][1:]
        with self.assertRaisesRegex(ValueError, "capture_attention_digest_mismatch"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)

    def test_rejects_rehashed_attention_with_wrong_group_masses(self):
        changed = copy.deepcopy(self.capture)
        changed["attention_f32_be_hex"] = "".join(
            ["3f800000" if i == 0 else "00000000"
             for i in range(len(changed["attention_f32_be_hex"]) // 8)])
        changed["attention_bytes_sha256"] = hashlib.sha256(
            bytes.fromhex(changed["attention_f32_be_hex"])).hexdigest()
        with self.assertRaisesRegex(ValueError, "capture_attention_row_not_normalized"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)

    def test_rejects_group_mass_tampering(self):
        changed = copy.deepcopy(self.capture)
        changed["layer_group_masses_f32_be_hex"][0][0] = "00000000"
        with self.assertRaisesRegex(ValueError,
                                   "capture_group_mass_recomputation_mismatch"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)

    def test_rejects_batch_or_geometry_rebinding(self):
        for field, value in (("typed_batch_sha256", sha("other")),
                             ("question_positions", [5, 6]),
                             ("temporal_groups", [[0, 2], [3, 4]])):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "capture_binding_mismatch"):
                    validate_attention_capture(self.capture,
                                               expected_binding={**self.binding,
                                                                 field: value},
                                               input_length=8)

    def test_never_authorizes_scoring(self):
        changed = copy.deepcopy(self.capture)
        changed["scoring_authorized"] = True
        with self.assertRaisesRegex(ValueError, "capture_schema_or_authority_invalid"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)

    def test_builder_rejects_non_normalized_rows(self):
        changed = copy.deepcopy(self.attention)
        changed[0][0][0][0] += 0.1
        with self.assertRaisesRegex(ValueError, "capture_attention_row_not_normalized"):
            build_attention_capture(binding=self.binding, attention=changed,
                                    input_length=8)

    def test_selector_float32_reduction_matches_bitwise(self):
        import torch

        expected_hex = self.capture["layer_group_masses_f32_be_hex"]
        for layer_index, layer in enumerate(self.attention):
            full = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
            for head_index, head in enumerate(layer):
                full[0, head_index, 4, :] = torch.tensor(head[0], dtype=torch.float32)
                full[0, head_index, 5, :] = torch.tensor(head[1], dtype=torch.float32)
            actual = aggregate_temporal_group_attention(
                attention_by_layer={layer_index: full}, layer_indices=[layer_index],
                question_positions=[4, 5], group_positions=[[0, 1], [2, 3]],
                torch_module=torch)
            self.assertEqual([struct.pack(">f", value).hex() for value in actual],
                             expected_hex[layer_index])

    def test_rejects_bool_version_and_noncontiguous_question_span(self):
        changed = copy.deepcopy(self.capture)
        changed["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "capture_schema_or_authority_invalid"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)

        changed = copy.deepcopy(self.capture)
        changed["question_positions"] = [4, 6]
        binding = {**self.binding, "question_positions": [4, 6]}
        with self.assertRaisesRegex(ValueError, "capture_question_positions_noncontiguous"):
            validate_attention_capture(changed, expected_binding=binding,
                                       input_length=8)

    def test_rejects_reverse_ordered_temporal_groups(self):
        changed = copy.deepcopy(self.capture)
        groups = [[2, 3], [0, 1]]
        changed["temporal_groups"] = groups
        binding = {**self.binding, "temporal_groups": groups}
        with self.assertRaisesRegex(ValueError,
                                   "capture_temporal_group_order_or_size_invalid"):
            validate_attention_capture(changed, expected_binding=binding,
                                       input_length=8)

    def test_rejects_oversize_shape_before_iterating_attention(self):
        class HugeLayer(Sequence):
            def __len__(self):
                return 300_000

            def __getitem__(self, index):
                raise AssertionError("oversize tensor was indexed")

            def __iter__(self):
                raise AssertionError("oversize tensor was iterated")

        with self.assertRaisesRegex(ValueError, "capture_attention_payload_too_large"):
            build_attention_capture(binding=self.binding,
                                    attention=[HugeLayer() for _ in LAYERS],
                                    input_length=8)

    def test_rejects_extra_field_and_malformed_expected_digest(self):
        changed = {**self.capture, "unbound_extra": "value"}
        with self.assertRaisesRegex(ValueError, "capture_schema_or_authority_invalid"):
            validate_attention_capture(changed, expected_binding=self.binding,
                                       input_length=8)
        binding = {**self.binding, "model_sha256": "not-a-digest"}
        with self.assertRaisesRegex(ValueError,
                                   "capture_expected_binding_digest_invalid"):
            validate_attention_capture(self.capture, expected_binding=binding,
                                       input_length=8)


if __name__ == "__main__":
    unittest.main()
