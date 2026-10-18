from __future__ import annotations

import json
import unittest

from qta_sampler_runtime_descriptor import (
    validate_qta_sampler_model_runtime_descriptor,
)


class SamplerRuntimeDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.value = {
            "schema_version": 1,
            "purpose": "qta_sampler_model_runtime_descriptor_non_authorizing_v1",
            "python_version": "3.12.0", "torch_version": "2.11.0+cu128",
            "cuda_runtime_version": "12.8", "transformers_version": "5.18.0",
            "bitsandbytes_version": "0.46.0", "device_type": "cuda",
            "precision": "bfloat16", "quantization": "4bit_nf4",
            "deterministic_algorithms": False, "allow_tf32": False,
        }

    def _raw(self):
        return json.dumps(self.value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode("ascii")

    def test_canonical_descriptor_is_hashed_and_non_authorizing(self):
        report = validate_qta_sampler_model_runtime_descriptor(self._raw())
        self.assertTrue(report["runtime_descriptor_valid"])
        self.assertFalse(report["runtime_descriptor_semantics_authenticated"])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_rejects_unknown_fields_and_noncanonical_bytes(self):
        self.value["hostname"] = "must-not-be-present"
        with self.assertRaisesRegex(ValueError, "sampler_runtime_descriptor_schema_invalid"):
            validate_qta_sampler_model_runtime_descriptor(self._raw())
        self.value.pop("hostname")
        with self.assertRaisesRegex(ValueError, "sampler_runtime_descriptor_not_canonical"):
            validate_qta_sampler_model_runtime_descriptor(self._raw() + b"\n")

    def test_rejects_invalid_runtime_and_inconsistent_quantization_device(self):
        self.value["device_type"] = "remote-machine-17"
        with self.assertRaisesRegex(ValueError, "sampler_runtime_descriptor_device_type_invalid"):
            validate_qta_sampler_model_runtime_descriptor(self._raw())
        self.value["device_type"] = "cpu"
        with self.assertRaisesRegex(ValueError,
                                    "sampler_runtime_descriptor_quantization_device_mismatch"):
            validate_qta_sampler_model_runtime_descriptor(self._raw())

    def test_rejects_quantization_without_runtime_dependency(self):
        self.value["bitsandbytes_version"] = None
        with self.assertRaisesRegex(ValueError,
                                    "sampler_runtime_descriptor_quantization_dependency_missing"):
            validate_qta_sampler_model_runtime_descriptor(self._raw())


if __name__ == "__main__":
    unittest.main()
