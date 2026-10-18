from __future__ import annotations

import copy
import hashlib
import unittest

from qta_attention_call_receipts import (
    build_qta_attention_call_receipt,
    build_qta_attention_call_receipt_index,
    validate_qta_attention_call_receipt,
    validate_qta_attention_call_receipt_index,
)
from qta_attention_forward_capture import capture_qta_attention_forward_once


class _FakeAttention:
    def __init__(self, torch):
        self.torch = torch
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        owner = self

        class Handle:
            def remove(self):
                owner.hooks.remove(hook)

        return Handle()

    def __call__(self, hidden):
        attention = self.torch.full((1, 2, 4, 4), 0.25)
        output = (hidden, attention)
        for hook in tuple(self.hooks):
            hook(self, (hidden,), output)
        return output


class _FakeLayer:
    def __init__(self, torch):
        self.self_attn = _FakeAttention(torch)


class _FakeModel:
    def __init__(self, torch):
        self.torch = torch
        self.model = type("ModelRoot", (), {})()
        self.model.language_model = type("LanguageModel", (), {})()
        self.model.language_model.layers = [_FakeLayer(torch) for _ in range(16)]

    def __call__(self):
        hidden = self.torch.zeros((1, 4, 2))
        for layer in self.model.language_model.layers:
            hidden, _attention = layer.self_attn(hidden)
        return hidden


class QtaAttentionCallReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("synthetic capture requires torch")
        cls.torch = torch

    def setUp(self):
        self.model = _FakeModel(self.torch)
        self.positions = [1, 2]
        _result, rows, report = capture_qta_attention_forward_once(
            model=self.model, forward=self.model, question_positions=self.positions,
            input_length=4, torch_module=self.torch)
        # Stable synthetic payload digest over selected layer rows, layer ordered.
        import numpy as np
        payload = b"".join(np.asarray(rows[layer].detach().cpu(), dtype=">f4").tobytes()
                           for layer in range(5, 17))
        self.payload_sha = hashlib.sha256(payload).hexdigest()
        self.runtime_sha = hashlib.sha256(b"synthetic-runtime-descriptor-v1").hexdigest()
        self.record_token = hashlib.sha256(b"synthetic-record-token").hexdigest()
        self.classes = dict(report["layer_module_classes"])
        self.report = report
        self.receipt = build_qta_attention_call_receipt(
            report=report, record_token=self.record_token,
            runtime_sha256=self.runtime_sha, input_length=4,
            question_positions=self.positions,
            attention_bytes_sha256=self.payload_sha,
            expected_module_classes=self.classes)
        self.expected_row = {
            "record_token": self.record_token,
            "runtime_sha256": self.runtime_sha,
            "input_length": 4,
            "question_positions": self.positions,
            "attention_bytes_sha256": self.payload_sha,
        }
        self.index_raw = build_qta_attention_call_receipt_index(
            receipts=[self.receipt])

    def _validate(self, receipt=None, **overrides):
        args = {
            "expected_record_token": self.record_token,
            "expected_runtime_sha256": self.runtime_sha,
            "expected_input_length": 4,
            "expected_question_positions": self.positions,
            "expected_attention_bytes_sha256": self.payload_sha,
            "expected_module_classes": self.classes,
        }
        args.update(overrides)
        return validate_qta_attention_call_receipt(receipt or self.receipt, **args)

    def test_real_hook_report_joins_payload_and_stays_non_authorizing(self):
        result = self._validate()
        self.assertTrue(result["call_receipt_valid"])
        self.assertTrue(result["call_receipt_joined_to_record_runtime_and_attention_bytes"])
        self.assertFalse(result["forward_source_authenticated"])
        self.assertFalse(result["attention_capture_authenticated"])
        self.assertFalse(result["scoring_authorized"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_binding_tampering_is_rejected(self):
        for field, replacement in (
                ("record_token", "0" * 64),
                ("runtime_sha256", "1" * 64),
                ("attention_bytes_sha256", "2" * 64)):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.receipt)
                mutated[field] = replacement
                with self.assertRaises(ValueError):
                    self._validate(mutated)

    def test_call_path_class_and_shape_tampering_is_rejected(self):
        mutations = (
            ("forward_calls", 2),
            ("layer_calls", "5", 2),
            ("layer_module_paths", "5", "wrong.path"),
            ("layer_module_classes", "5", "wrong.Class"),
            ("attention_shape", [12, 2, 2, 5]),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                mutated = copy.deepcopy(self.receipt)
                if mutation[0] in {"layer_calls", "layer_module_paths", "layer_module_classes"}:
                    mutated[mutation[0]][mutation[1]] = mutation[2]
                else:
                    mutated[mutation[0]] = mutation[1]
                with self.assertRaises(ValueError):
                    self._validate(mutated)

    def test_non_authorizing_fields_cannot_be_promoted(self):
        for field, value in (("scoring_authorized", True),
                             ("forward_source_authenticated", True),
                             ("execution_authorization", "granted")):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.receipt)
                mutated[field] = value
                with self.assertRaises(ValueError):
                    self._validate(mutated)

    def test_embedded_capture_report_tampering_is_rejected(self):
        mutated = copy.deepcopy(self.receipt)
        mutated["capture_report"]["layer_module_paths"]["5"] = "wrong.path"
        with self.assertRaises(ValueError):
            self._validate(mutated)

    def test_incomplete_report_rejected(self):
        report = copy.deepcopy(self.report)
        del report["layer_calls"]["8"]
        with self.assertRaisesRegex(ValueError, "call_receipt_layer_calls_invalid"):
            build_qta_attention_call_receipt(
                report=report, record_token=self.record_token,
                runtime_sha256=self.runtime_sha, input_length=4,
                question_positions=self.positions,
                attention_bytes_sha256=self.payload_sha)

    def test_report_numeric_types_and_normalization_bound_are_strict(self):
        for field, value in (("forward_calls", True),
                             ("question_query_count", True),
                             ("attention_row_sum_max_abs_deviation", 0.01),
                             ("attention_row_sum_tolerance", True),
                             ("attention_rows_satisfy_source_dtype_normalization_tolerance", False)):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = value
                with self.assertRaises(ValueError):
                    build_qta_attention_call_receipt(
                        report=report, record_token=self.record_token,
                        runtime_sha256=self.runtime_sha, input_length=4,
                        question_positions=self.positions,
                        attention_bytes_sha256=self.payload_sha)

    def test_index_validates_ordered_join_and_is_non_authorizing(self):
        result = validate_qta_attention_call_receipt_index(
            index_raw=self.index_raw, expected_rows=[self.expected_row])
        self.assertEqual(result["call_receipt_records_validated"], 1)
        self.assertEqual(result["call_receipt_index_sha256"],
                         hashlib.sha256(self.index_raw).hexdigest())
        self.assertFalse(result["forward_source_authenticated"])
        self.assertFalse(result["attention_capture_authenticated"])
        self.assertFalse(result["scoring_authorized"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_index_order_binding_denials_and_duplicate_keys_are_strict(self):
        import json
        parsed = json.loads(self.index_raw)
        parsed["records"][0]["receipt"]["runtime_sha256"] = "f" * 64
        tampered = json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True).encode("ascii")
        with self.assertRaises(ValueError):
            validate_qta_attention_call_receipt_index(
                index_raw=tampered, expected_rows=[self.expected_row])

        parsed = json.loads(self.index_raw)
        parsed["scoring_authorized"] = True
        tampered_denial = json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")
        with self.assertRaises(ValueError):
            validate_qta_attention_call_receipt_index(
                index_raw=tampered_denial, expected_rows=[self.expected_row])

        duplicate_key = self.index_raw[:-1] + b',"purpose":"duplicate"}'
        with self.assertRaisesRegex(ValueError, "call_receipt_duplicate_json_key"):
            validate_qta_attention_call_receipt_index(
                index_raw=duplicate_key, expected_rows=[self.expected_row])

    def test_index_rejects_record_reordering_and_missing_rows(self):
        # A second synthetic receipt with a distinct row identity.
        second = build_qta_attention_call_receipt(
            report=self.report, record_token="a" * 64,
            runtime_sha256=self.runtime_sha, input_length=4,
            question_positions=self.positions,
            attention_bytes_sha256="b" * 64,
            expected_module_classes=self.classes)
        second_raw = build_qta_attention_call_receipt_index(
            receipts=[second, self.receipt])
        second_expected = dict(self.expected_row, record_token="a" * 64,
                               attention_bytes_sha256="b" * 64)
        with self.assertRaises(ValueError):
            validate_qta_attention_call_receipt_index(
                index_raw=second_raw,
                expected_rows=[self.expected_row, second_expected])
        with self.assertRaises(ValueError):
            validate_qta_attention_call_receipt_index(
                index_raw=second_raw, expected_rows=[second_expected])


if __name__ == "__main__":
    unittest.main()
