"""CPU-only tests for deriving six-condition bindings from batch tensors."""
from pathlib import Path
import sys
import unittest
from array import array

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_six_condition_binding import derive_six_condition_binding  # noqa: E402
from fixed_prediction_selective_dependence import INTERVENTION_CONDITIONS  # noqa: E402


class FakeArray:
    def __init__(self, values, shape, dtype="int64"):
        self.values = array("q", values)
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype

    def tobytes(self, order="C"):
        del order
        return self.values.tobytes()

    def tolist(self):
        values = self.values.tolist()
        if len(self.shape) == 2:
            return [values]
        return values


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class FakeTensor:
    def __init__(self, values, shape, device="cpu", dtype="int64"):
        self._array = FakeArray(values, shape, dtype)
        self.device = FakeDevice(device)
        self.dtype = dtype
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._array.tolist()

    def contiguous(self):
        return self

    def numpy(self):
        return self._array


class DeriveSixConditionBindingTests(unittest.TestCase):
    def setUp(self):
        self.batches = {
            condition: {
                "input_ids": FakeTensor([1, 2, 3, 4], (1, 4)),
                "attention_mask": FakeTensor([1, 1, 1, 1], (1, 4)),
                "pixel_values_videos": FakeTensor([5, 6, 7, 8], (1, 4)),
                "video_grid_thw": FakeTensor([1, 2, 2], (1, 3)),
                "second_per_grid_ts": FakeTensor([1], (1, 1)),
            }
            for condition in INTERVENTION_CONDITIONS
        }
        for pair, pixel_values in (
            (("baseline_evidence_ablation", "intervention_evidence_ablation"),
             [5, 6, 7, 9]),
            (("baseline_control_ablation", "intervention_control_ablation"),
             [5, 6, 7, 10]),
        ):
            for condition in pair:
                self.batches[condition]["pixel_values_videos"] = FakeTensor(
                    pixel_values, (1, 4))
        self.kwargs = {
            "answer_token_ids": [10, 11, 12, 13, 14],
            "model_revision": "pinned-model-revision",
            "tokenizer_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
        }

    def test_derives_identical_bindings_and_actual_tensor_hashes(self):
        result = derive_six_condition_binding(batches=self.batches, **self.kwargs)
        self.assertEqual(result["status"], "derived_synthetic_batch_binding_contract_only")
        self.assertEqual(result["bound_conditions"], 6)
        self.assertEqual(result["input_length"], 4)
        self.assertEqual(result["score_position"], 3)
        self.assertEqual(len(result["input_ids_sha256"]), 64)
        self.assertEqual(len(result["attention_mask_sha256"]), 64)
        self.assertEqual(set(result["batch_sha256_by_condition"]), INTERVENTION_CONDITIONS)
        self.assertEqual(result["batch_sha256_by_condition"]["baseline_full"],
                         result["batch_sha256_by_condition"]["intervention_full"])
        self.assertNotEqual(result["batch_sha256_by_condition"]["baseline_evidence_ablation"],
                            result["batch_sha256_by_condition"]["baseline_control_ablation"])
        self.assertEqual(len(result["full_binding_sha256"]), 64)
        self.assertNotEqual(result["text_binding_sha256"], result["full_binding_sha256"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertIn("caller-supplied", result["interpretation_limit"])

    def test_rejects_input_and_mask_mismatch(self):
        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["intervention_full"]["input_ids"] = FakeTensor([1, 2, 3, 5], (1, 4))
        with self.assertRaisesRegex(ValueError, "input_ids_mismatch"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["intervention_full"]["attention_mask"] = FakeTensor([1, 1, 0, 1], (1, 4))
        with self.assertRaisesRegex(ValueError, "attention_mask_mismatch"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

    def test_requires_video_payload_and_qwen_video_metadata_keys(self):
        for missing_key in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            changed = {name: dict(batch) for name, batch in self.batches.items()}
            del changed["baseline_full"][missing_key]
            with self.subTest(missing_key=missing_key), self.assertRaisesRegex(
                    ValueError, "required_tensors_missing"):
                derive_six_condition_binding(batches=changed, **self.kwargs)

    def test_requires_integer_input_ids_and_binary_integer_or_boolean_mask(self):
        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["baseline_full"]["input_ids"] = FakeTensor(
            [1, 2, 3, 4], (1, 4), dtype="float32")
        with self.assertRaisesRegex(ValueError, "input_ids_integer_dtype_required"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["baseline_full"]["attention_mask"] = FakeTensor(
            [1, 2, 1, 1], (1, 4))
        with self.assertRaisesRegex(ValueError, "attention_mask_not_binary"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["baseline_full"]["attention_mask"] = FakeTensor(
            [1, 1, 1, 1], (1, 4), dtype="float32")
        with self.assertRaisesRegex(
                ValueError, "attention_mask_integer_or_bool_dtype_required"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

        changed = {name: dict(batch) for name, batch in self.batches.items()}
        for batch in changed.values():
            batch["attention_mask"] = FakeTensor([1, 1, 1, 1], (1, 4), dtype="bool")
        self.assertEqual(derive_six_condition_binding(
            batches=changed, **self.kwargs)["bound_conditions"], 6)

    def test_visual_tensor_changes_branch_hash_and_full_binding(self):
        original = derive_six_condition_binding(batches=self.batches, **self.kwargs)
        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed_value = FakeTensor([5, 6, 7, 9], (1, 4))
        changed["baseline_full"]["pixel_values_videos"] = changed_value
        changed["intervention_full"]["pixel_values_videos"] = changed_value
        updated = derive_six_condition_binding(batches=changed, **self.kwargs)
        self.assertNotEqual(original["batch_sha256_by_condition"]["baseline_full"],
                            updated["batch_sha256_by_condition"]["baseline_full"])
        self.assertNotEqual(original["full_binding_sha256"], updated["full_binding_sha256"])

    def test_rejects_mismatched_paired_full_inputs(self):
        changed = {name: dict(batch) for name, batch in self.batches.items()}
        changed["intervention_full"]["pixel_values_videos"] = FakeTensor([5, 6, 7, 9], (1, 4))
        with self.assertRaisesRegex(ValueError, "paired_batch_hash_mismatch_full"):
            derive_six_condition_binding(batches=changed, **self.kwargs)

    def test_rejects_condition_hash_model_and_position_declaration_mismatch(self):
        result = derive_six_condition_binding(batches=self.batches, **self.kwargs)
        common = {
            "answer_label_order": list("ABCDE"),
            "answer_token_ids": self.kwargs["answer_token_ids"],
            "input_ids_sha256": result["input_ids_sha256"],
            "attention_mask_sha256": result["attention_mask_sha256"],
            "tokenizer_sha256": self.kwargs["tokenizer_sha256"],
            "model_revision": self.kwargs["model_revision"],
            "prompt_sha256": self.kwargs["prompt_sha256"],
            "input_length": 4,
            "score_position": 3,
        }
        for key, value in (
            ("input_ids_sha256", "c" * 64),
            ("tokenizer_sha256", "d" * 64),
            ("model_revision", "other-model"),
            ("score_position", 2),
        ):
            declarations = {condition: dict(common) for condition in INTERVENTION_CONDITIONS}
            declarations["intervention_control_ablation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "binding_declaration_mismatch"):
                derive_six_condition_binding(
                    batches=self.batches, condition_declarations=declarations, **self.kwargs)

    def test_rejects_invalid_metadata_and_non_cpu_batches(self):
        with self.assertRaisesRegex(ValueError, "tokenizer_sha256_invalid"):
            derive_six_condition_binding(
                batches=self.batches, **{**self.kwargs, "tokenizer_sha256": "bad"})
        gpu_batch = {name: dict(batch) for name, batch in self.batches.items()}
        # A declared non-CPU tensor proves the guard without allocating a GPU.
        gpu_batch["baseline_full"]["input_ids"] = FakeTensor([1, 2, 3, 4], (1, 4), "cuda")
        with self.assertRaisesRegex(ValueError, "must_be_cpu_resident"):
            derive_six_condition_binding(batches=gpu_batch, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
