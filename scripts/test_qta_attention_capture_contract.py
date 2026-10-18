from __future__ import annotations

import copy
import hashlib
import json
import struct
import unittest
from collections.abc import Sequence

from qta_attention_capture_contract import (
    LAYERS,
    build_attention_capture,
    validate_capture_selection_bridge,
    validate_attention_capture,
)
from qta_attention_capture_sidecars import (
    build_qta_capture_sidecar_index, pack_qta_attention_sidecar_bytes,
    validate_qta_capture_sidecar_index,
)
from qta_selection_provenance import (
    FROZEN_SELECTOR_SOURCE_SHA256, build_qta_selection_provenance,
)
from qta_three_row_plan_aggregate import FROZEN_QTA_SHAM_SEED
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

    def _selection_bindings(self):
        return {
            "record_token": self.binding["record_token"],
            "opaque_review_token": "synthetic_review_token01",
            "row_binding_sha256": self.binding["row_binding_sha256"],
            "typed_batch_sha256": self.binding["typed_batch_sha256"],
            "score_binding_sha256": sha("score-binding"),
            "baseline_pq_row_sha256": self.binding["baseline_pq_row_sha256"],
            "prompt_sha256": self.binding["prompt_sha256"],
            "input_ids_sha256": self.binding["input_ids_sha256"],
            "input_length": 8,
            "score_position": 7,
            "question_positions": self.binding["question_positions"],
            "temporal_groups": self.binding["temporal_groups"],
            "selector_source_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
            "selector_layers_1_based": LAYERS,
            "sham_seed": FROZEN_QTA_SHAM_SEED,
        }

    def _sidecar_bundle(self):
        rows = []
        entries = []
        masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                   for value in row]
                  for row in self.capture["layer_group_masses_f32_be_hex"]]
        for i in range(3):
            binding = {**self.binding,
                "record_token": sha(f"record-{i}"),
                "row_binding_sha256": sha(f"row-{i}"),
                "typed_batch_sha256": sha(f"batch-{i}"),
                "baseline_pq_row_sha256": sha(f"baseline-{i}"),
                "prompt_sha256": sha(f"prompt-{i}"),
                "input_ids_sha256": sha(f"ids-{i}"),
            }
            selection_bindings = {
                "record_token": binding["record_token"],
                "opaque_review_token": f"review_token_synthetic{i:02d}",
                "row_binding_sha256": binding["row_binding_sha256"],
                "typed_batch_sha256": binding["typed_batch_sha256"],
                "score_binding_sha256": sha(f"score-{i}"),
                "baseline_pq_row_sha256": binding["baseline_pq_row_sha256"],
                "prompt_sha256": binding["prompt_sha256"],
                "input_ids_sha256": binding["input_ids_sha256"],
                "input_length": 8, "score_position": 7,
                "question_positions": binding["question_positions"],
                "temporal_groups": binding["temporal_groups"],
                "selector_source_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
                "selector_layers_1_based": LAYERS,
                "sham_seed": FROZEN_QTA_SHAM_SEED,
            }
            selection = build_qta_selection_provenance(
                **selection_bindings, layer_group_masses=masses)
            entries.append({"binding": binding, "attention": self.attention,
                            "input_length": 8})
            rows.append({"expected_capture_binding": binding,
                "input_length": 8, "selection_provenance": selection,
                "expected_selection_bindings": selection_bindings,
                "plan": None})
        return build_qta_capture_sidecar_index(entries=entries), rows

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

    def test_rejects_rehashed_nonfinite_or_negative_values(self):
        for value_hex in ("7fc00000", "7f800000", "bf800000"):
            with self.subTest(value_hex=value_hex):
                changed = copy.deepcopy(self.capture)
                encoded = value_hex + changed["attention_f32_be_hex"][8:]
                changed["attention_f32_be_hex"] = encoded
                changed["attention_bytes_sha256"] = hashlib.sha256(
                    bytes.fromhex(encoded)).hexdigest()
                with self.assertRaisesRegex(ValueError,
                                           "capture_attention_value_invalid"):
                    validate_attention_capture(changed,
                                               expected_binding=self.binding,
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

    def test_capture_bytes_join_to_exact_selection_masses(self):
        masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                   for value in row]
                  for row in self.capture["layer_group_masses_f32_be_hex"]]
        selection = build_qta_selection_provenance(
            record_token=self.binding["record_token"],
            opaque_review_token="synthetic_review_token01",
            row_binding_sha256=self.binding["row_binding_sha256"],
            typed_batch_sha256=self.binding["typed_batch_sha256"],
            score_binding_sha256=sha("score-binding"),
            baseline_pq_row_sha256=self.binding["baseline_pq_row_sha256"],
            prompt_sha256=self.binding["prompt_sha256"],
            input_ids_sha256=self.binding["input_ids_sha256"],
            input_length=8, score_position=7,
            question_positions=self.binding["question_positions"],
            temporal_groups=self.binding["temporal_groups"],
            selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
            selector_layers_1_based=LAYERS,
            layer_group_masses=masses, sham_seed=FROZEN_QTA_SHAM_SEED)
        report = validate_capture_selection_bridge(
            capture=self.capture, expected_capture_binding=self.binding,
            input_length=8, selection_provenance=selection,
            expected_selection_bindings=self._selection_bindings())
        self.assertEqual(report["layer_count"], 12)
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["scoring_authorized"])

    def test_capture_selection_bridge_rejects_mass_or_binding_rebind(self):
        masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                   for value in row]
                  for row in self.capture["layer_group_masses_f32_be_hex"]]
        selection = build_qta_selection_provenance(
            record_token=self.binding["record_token"],
            opaque_review_token="synthetic_review_token01",
            row_binding_sha256=self.binding["row_binding_sha256"],
            typed_batch_sha256=self.binding["typed_batch_sha256"],
            score_binding_sha256=sha("score-binding"),
            baseline_pq_row_sha256=self.binding["baseline_pq_row_sha256"],
            prompt_sha256=self.binding["prompt_sha256"],
            input_ids_sha256=self.binding["input_ids_sha256"],
            input_length=8, score_position=7,
            question_positions=self.binding["question_positions"],
            temporal_groups=self.binding["temporal_groups"],
            selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
            selector_layers_1_based=LAYERS,
            layer_group_masses=masses, sham_seed=FROZEN_QTA_SHAM_SEED)
        changed_masses = copy.deepcopy(masses)
        changed_masses[0][0] += 0.01
        changed_mass = build_qta_selection_provenance(
            record_token=self.binding["record_token"],
            opaque_review_token="synthetic_review_token01",
            row_binding_sha256=self.binding["row_binding_sha256"],
            typed_batch_sha256=self.binding["typed_batch_sha256"],
            score_binding_sha256=sha("score-binding"),
            baseline_pq_row_sha256=self.binding["baseline_pq_row_sha256"],
            prompt_sha256=self.binding["prompt_sha256"],
            input_ids_sha256=self.binding["input_ids_sha256"],
            input_length=8, score_position=7,
            question_positions=self.binding["question_positions"],
            temporal_groups=self.binding["temporal_groups"],
            selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
            selector_layers_1_based=LAYERS,
            layer_group_masses=changed_masses, sham_seed=FROZEN_QTA_SHAM_SEED)
        with self.assertRaisesRegex(ValueError,
                                    "capture_selection_group_mass_mismatch"):
            validate_capture_selection_bridge(
                capture=self.capture, expected_capture_binding=self.binding,
                input_length=8, selection_provenance=changed_mass,
                expected_selection_bindings=self._selection_bindings())
        changed_binding = copy.deepcopy(selection)
        changed_binding["typed_batch_sha256"] = sha("other-batch")
        with self.assertRaisesRegex(ValueError,
                                    "qta_selection_binding_mismatch"):
            validate_capture_selection_bridge(
                capture=self.capture, expected_capture_binding=self.binding,
                input_length=8, selection_provenance=changed_binding,
                expected_selection_bindings=self._selection_bindings())

    def test_capture_selection_bridge_rejects_recomputed_token_or_seed_rebind(self):
        masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                   for value in row]
                  for row in self.capture["layer_group_masses_f32_be_hex"]]
        for token, seed in (("different_review_token01", FROZEN_QTA_SHAM_SEED),
                            ("synthetic_review_token01", "changed-seed")):
            with self.subTest(token=token, seed=seed):
                selection = build_qta_selection_provenance(
                    record_token=self.binding["record_token"],
                    opaque_review_token=token,
                    row_binding_sha256=self.binding["row_binding_sha256"],
                    typed_batch_sha256=self.binding["typed_batch_sha256"],
                    score_binding_sha256=sha("score-binding"),
                    baseline_pq_row_sha256=self.binding["baseline_pq_row_sha256"],
                    prompt_sha256=self.binding["prompt_sha256"],
                    input_ids_sha256=self.binding["input_ids_sha256"],
                    input_length=8, score_position=7,
                    question_positions=self.binding["question_positions"],
                    temporal_groups=self.binding["temporal_groups"],
                    selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
                    selector_layers_1_based=LAYERS,
                    layer_group_masses=masses, sham_seed=seed)
                with self.assertRaises(ValueError):
                    validate_capture_selection_bridge(
                        capture=self.capture,
                        expected_capture_binding=self.binding,
                        input_length=8, selection_provenance=selection,
                        expected_selection_bindings=self._selection_bindings())

    def test_binary_sidecar_index_joins_three_rows_and_deduplicates_payload(self):
        index_raw, sidecars = self._sidecar_bundle()[0]
        expected_rows = self._sidecar_bundle()[1]
        report = validate_qta_capture_sidecar_index(
            index_raw=index_raw, sidecars=sidecars, expected_rows=expected_rows)
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["sidecar_count"], 1)
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertFalse(report["scoring_authorized"])

    def test_tensor_packer_matches_capture_bytes_and_binary_index(self):
        import torch

        layers = []
        for layer in self.attention:
            full = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
            for head_index, head in enumerate(layer):
                full[0, head_index, 4, :] = torch.tensor(head[0])
                full[0, head_index, 5, :] = torch.tensor(head[1])
            layers.append(full)
        shape, raw, source_dtype = pack_qta_attention_sidecar_bytes(
            attention_layers=dict(zip(LAYERS, layers, strict=True)),
            question_positions=[4, 5], input_length=8)
        self.assertEqual(shape, [12, 2, 2, 8])
        self.assertEqual(source_dtype, "float32")
        self.assertEqual(raw.hex(), self.capture["attention_f32_be_hex"])
        binding = self.binding
        indexed_raw, indexed_sidecars = build_qta_capture_sidecar_index(entries=[{
            "binding": binding, "attention_shape": shape,
            "attention_bytes": raw, "attention_source_dtype": source_dtype,
            "input_length": 8,
        }])
        list_raw, list_sidecars = build_qta_capture_sidecar_index(entries=[{
            "binding": binding, "attention": self.attention, "input_length": 8,
        }])
        self.assertEqual(indexed_raw, list_raw)
        self.assertEqual(indexed_sidecars, list_sidecars)

    def test_bfloat16_attention_rounding_uses_dtype_bound_not_float32_bound(self):
        import torch

        torch.manual_seed(20261018)
        layers = [torch.softmax(torch.randn(2, 8, 8), dim=-1).to(torch.bfloat16)
                  for _ in LAYERS]
        shape, raw, source_dtype = pack_qta_attention_sidecar_bytes(
            attention_layers=dict(zip(LAYERS, layers, strict=True)),
            question_positions=[4, 5], input_length=8)
        self.assertEqual(source_dtype, "bfloat16")
        index_raw, sidecars = build_qta_capture_sidecar_index(entries=[{
            "binding": self.binding, "attention_shape": shape,
            "attention_bytes": raw, "attention_source_dtype": source_dtype,
            "input_length": 8,
        }])
        value = json.loads(index_raw)
        self.assertEqual(value["records"][0]["attention_source_dtype"], "bfloat16")
        self.assertEqual(len(sidecars), 1)

    def test_bfloat16_dtype_does_not_allow_excessive_row_sum_error(self):
        import numpy as np
        import torch

        torch.manual_seed(20261018)
        layers = [torch.softmax(torch.randn(2, 8, 8), dim=-1).to(torch.bfloat16)
                  for _ in LAYERS]
        shape, raw, source_dtype = pack_qta_attention_sidecar_bytes(
            attention_layers=dict(zip(LAYERS, layers, strict=True)),
            question_positions=[4, 5], input_length=8)
        values = np.frombuffer(raw, dtype=">f4").copy().reshape(shape)
        values[0, 0, 0, 0] += 0.01
        changed_raw = values.astype(">f4").tobytes()
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_attention_row_invalid"):
            build_qta_capture_sidecar_index(entries=[{
                "binding": self.binding, "attention_shape": shape,
                "attention_bytes": changed_raw,
                "attention_source_dtype": source_dtype, "input_length": 8,
            }])

    def test_bfloat16_sidecar_and_selection_mass_join_validate(self):
        import torch

        torch.manual_seed(20261019)
        layers = [torch.softmax(torch.randn(2, 8, 8), dim=-1).to(torch.bfloat16)
                  for _ in LAYERS]
        shape, raw, source_dtype = pack_qta_attention_sidecar_bytes(
            attention_layers=dict(zip(LAYERS, layers, strict=True)),
            question_positions=[4, 5], input_length=8)
        index_raw, sidecars = build_qta_capture_sidecar_index(entries=[{
            "binding": self.binding, "attention_shape": shape,
            "attention_bytes": raw, "attention_source_dtype": source_dtype,
            "input_length": 8,
        }])
        index = json.loads(index_raw)
        masses = [[struct.unpack(">f", bytes.fromhex(value))[0] for value in row]
                  for row in index["records"][0]["layer_group_masses_f32_be_hex"]]
        bindings = self._selection_bindings()
        selection = build_qta_selection_provenance(
            **bindings, layer_group_masses=masses)
        report = validate_qta_capture_sidecar_index(
            index_raw=index_raw, sidecars=sidecars, expected_rows=[{
                "expected_capture_binding": self.binding,
                "input_length": 8,
                "selection_provenance": selection,
                "expected_selection_bindings": bindings,
                "plan": None,
            }])
        self.assertEqual(report["record_count"], 1)
        self.assertFalse(report["attention_capture_authenticated"])

    def test_tensor_packer_rejects_oversize_before_tensor_slicing(self):
        class OversizeTensor:
            shape = (30_000, 4096, 4096)

            def __getitem__(self, _key):
                raise AssertionError("oversize tensor sliced before cap check")

        with self.assertRaisesRegex(ValueError, "capture_sidecar_payload_too_large"):
            pack_qta_attention_sidecar_bytes(
                attention_layers={layer: OversizeTensor() for layer in LAYERS},
                question_positions=[100, 101], input_length=4096)

    def test_binary_sidecar_index_rejects_payload_mutation_or_binding_rebind(self):
        (index_raw, sidecars), expected_rows = self._sidecar_bundle()
        digest, payload = next(iter(sidecars.items()))
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_digest_or_length_mismatch"):
            validate_qta_capture_sidecar_index(
                index_raw=index_raw, sidecars={digest: b"x" + payload[1:]},
                expected_rows=expected_rows)
        rebound_rows = copy.deepcopy(expected_rows)
        rebound_rows[0]["expected_capture_binding"]["typed_batch_sha256"] = sha("other")
        with self.assertRaisesRegex(ValueError, "capture_sidecar_binding_mismatch"):
            validate_qta_capture_sidecar_index(
                index_raw=index_raw, sidecars=sidecars, expected_rows=rebound_rows)

    def test_binary_sidecar_index_rejects_row_reordering(self):
        (index_raw, sidecars), expected_rows = self._sidecar_bundle()
        index = json.loads(index_raw)
        index["records"][0], index["records"][1] = (
            index["records"][1], index["records"][0])
        reordered = (json.dumps(index, sort_keys=True, separators=(",", ":"))
                     + "\n").encode("ascii")
        with self.assertRaisesRegex(ValueError, "capture_sidecar_binding_mismatch"):
            validate_qta_capture_sidecar_index(
                index_raw=reordered, sidecars=sidecars,
                expected_rows=expected_rows)

    def test_binary_sidecar_index_rejects_duplicate_record_token(self):
        (index_raw, sidecars), expected_rows = self._sidecar_bundle()
        index = json.loads(index_raw)
        index["records"][1] = copy.deepcopy(index["records"][0])
        duplicate = (json.dumps(index, sort_keys=True, separators=(",", ":"))
                     + "\n").encode("ascii")
        expected_rows[1] = copy.deepcopy(expected_rows[0])
        with self.assertRaisesRegex(ValueError, "capture_sidecar_record_token_duplicate"):
            validate_qta_capture_sidecar_index(
                index_raw=duplicate, sidecars=sidecars,
                expected_rows=expected_rows)

    def test_binary_sidecar_index_rejects_mass_mismatch(self):
        (index_raw, sidecars), expected_rows = self._sidecar_bundle()
        expected = expected_rows[0]
        changed = [[struct.unpack(">f", bytes.fromhex(value))[0]
                    for value in row]
                   for row in expected["selection_provenance"][
                       "layer_group_mass_f32_be_hex"]]
        changed[0][0] += 0.01
        bindings = expected["expected_selection_bindings"]
        expected["selection_provenance"] = build_qta_selection_provenance(
            **bindings, layer_group_masses=changed)
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_selection_mass_mismatch"):
            validate_qta_capture_sidecar_index(
                index_raw=index_raw, sidecars=sidecars, expected_rows=expected_rows)

    def test_binary_sidecar_index_rejects_authority_or_canonical_json_tampering(self):
        (index_raw, sidecars), expected_rows = self._sidecar_bundle()
        value = json.loads(index_raw)
        value["scoring_authorized"] = True
        tampered = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True) + "\n").encode("ascii")
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_index_schema_or_authority_invalid"):
            validate_qta_capture_sidecar_index(
                index_raw=tampered, sidecars=sidecars, expected_rows=expected_rows)
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_index_schema_or_authority_invalid"):
            validate_qta_capture_sidecar_index(
                index_raw=index_raw + b" ", sidecars=sidecars,
                expected_rows=expected_rows)

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
