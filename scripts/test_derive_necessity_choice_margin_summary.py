"""Privacy and accounting checks for the aggregate necessity publisher."""
import json
import tempfile
import unittest
from pathlib import Path

from derive_necessity_choice_margin_summary import derive, publish_manifest


def fixture():
    row = {
        "sample_id": "private-sample", "video_id": "private-video",
        "qid": "private-qid", "question": "private question",
        "choices": ["private"], "gold_choice_index": 0,
        "choice_logprobs": {"full": [-1, -2, -3, -4, -5]},
        "sampled_indices": [1, 2], "selective_necessity_margin_drop": 0.2,
        "baseline_choice_correct": True,
    }
    stats = {"n": 1, "mean": 0.2, "sd": None, "ci95_pointwise": None,
             "positive_fraction": 1.0, "signflip_p_two_sided": None}
    return {
        "schema_version": 3, "source_commit": "a" * 40,
        "samples_file": "/private/cohort.json", "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "b" * 40, "tokenizer_revision": "b" * 40,
        "quant": "4bit", "dtype": "bfloat16", "attn_implementation": "eager",
        "python_version": "3.12.0", "python_implementation": "CPython",
        "torch_version": "2.8.0+cu128", "transformers_version": "5.0.0",
        "device": "cuda", "gpu_name": "test GPU", "seed": 7,
        "bootstrap_draws": 1000,
        "summary_seed_schedule": {"all": 7, "baseline_correct_primary": 8,
                                  "baseline_incorrect_secondary": 9,
                                  "correct_minus_incorrect_interaction": 10},
        "num_frames": 16, "frame_size": 224, "max_pixels": 1000,
        "cap_pixels_per_frame": True,
        "mask_intervention": {"fill": "black", "blur_radius": None},
        "primary_estimand": "fixed margin contrast",
        "control_protocol": "matched temporal rank",
        "n_requested": 2, "n_effective": 1,
        "failures": [{"sample_index": 1, "failure_code": "media_io_error"}],
        "records": [row],
        "summary": {
            "all": stats, "baseline_correct_primary": stats,
            "baseline_incorrect_secondary": {
                "n": 0, "mean": None, "sd": None, "ci95_pointwise": None,
                "positive_fraction": None, "signflip_p_two_sided": None},
            "correct_minus_incorrect_interaction": {
                "mean_correct_minus_incorrect": None, "ci95_pointwise": None},
        },
        "multiplicity": "exploratory; no adjustment",
        "signflip_test": "two-sided; symmetric paired null",
        "limitations": ["Synthetic test only."],
    }


class DerivedSummaryTests(unittest.TestCase):
    def test_projection_omits_item_data_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            raw_path.write_text(json.dumps(fixture()))
            result = derive(raw_path)
            serialized = json.dumps(result)
            for private in ("private-sample", "private-video", "private-qid",
                            "private question", "choice_logprobs", "samples_file",
                            "records", "sample_index", "failure_code",
                            "sampled_indices"):
                self.assertNotIn(private, serialized)
            self.assertEqual(result["n_requested"], 2)
            self.assertEqual(result["n_effective"], 1)
            self.assertEqual(result["n_failures"], 1)
            self.assertEqual(result["failure_counts_by_code"], {"media_io_error": 1})
            self.assertEqual(result["n_baseline_correct"], 1)
            self.assertEqual(result["summary"]["correct_minus_incorrect_interaction"]
                             ["n_baseline_incorrect"], 0)
            self.assertEqual(len(result["code_sha256"]), 7)
            manifest = publish_manifest(result, serialized.encode())
            self.assertNotIn("private", json.dumps(manifest))
            self.assertNotIn("samples_file", manifest)
            self.assertNotIn("command", manifest)

    def test_rejects_duplicate_video_units_and_nonreconciled_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            value = fixture()
            value["records"].append(dict(value["records"][0], sample_id="other"))
            value["n_effective"] = 2
            value["n_requested"] = 3
            value["failures"] = [{"sample_index": 2, "failure_code": "runtime_error"}]
            raw_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "one independent"):
                derive(raw_path)

            value = fixture()
            value["n_requested"] = 3
            raw_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "does not reconcile"):
                derive(raw_path)

    def test_rejects_local_model_path(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            value = fixture()
            value["model"] = "/private/model"
            raw_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "model is outside"):
                derive(raw_path)

    def test_rejects_untrusted_text_and_metadata_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            for field, poisoned in (
                ("limitations", ["private question text: do not publish"]),
                ("primary_estimand", "private cohort label"),
                ("control_protocol", "host=/home/private"),
                ("gpu_name", "private machine identity"),
            ):
                value = fixture()
                value[field] = poisoned
                raw_path.write_text(json.dumps(value))
                result = derive(raw_path)
                self.assertNotIn("private", json.dumps(result).lower())
            for field, poisoned in (
                ("model_revision", "private revision"),
                ("torch_version", "private version"),
                ("python_version", "3.12.0-private"),
                ("device", "private device"),
            ):
                value = fixture()
                value[field] = poisoned
                raw_path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    derive(raw_path)

    def test_failure_only_run_retains_requested_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            value = fixture()
            value.update(n_effective=0, records=[],
                         failures=[{"sample_index": 0, "failure_code": "runtime_error"},
                                   {"sample_index": 1, "failure_code": "cuda_oom"}],
                         summary={"all": None, "baseline_correct_primary": None,
                                  "baseline_incorrect_secondary": None,
                                  "correct_minus_incorrect_interaction": None})
            raw_path.write_text(json.dumps(value))
            result = derive(raw_path)
            self.assertEqual(result["n_requested"], 2)
            self.assertEqual(result["n_effective"], 0)
            self.assertEqual(result["n_failures"], 2)
            self.assertEqual(result["failure_summary_status"], "no_successful_samples")
            self.assertEqual(result["summary"]["all"], None)


if __name__ == "__main__":
    unittest.main()
