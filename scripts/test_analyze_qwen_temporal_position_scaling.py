import json
from pathlib import Path
import tempfile
import unittest

from analyze_qwen_temporal_position_scaling import analyze


def summary(frames, scale):
    steps = frames // 2
    native = [15 + int(i * 0.75 * 2) for i in range(steps)]
    return {
        "schema_version": 1,
        "purpose": "synthetic_subsecond_temporal_position_regression_check",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "a" * 40,
        "model_config_sha256": "b" * 64,
        "cuda_version": "12.8",
        "versions": {"torch": "2.11.0+cu128"},
        "seed": 20261012,
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "source_commit": "c" * 40,
        "source_sha256": {
            "run_qwen_temporal_position_gpu_check.py": ("d" if frames == 32 else "e") * 64,
            "qwen_temporal_geometry.py": "f" * 64,
            "video_time_mapping.py": "0" * 64,
            "run_dataset_video_gpu_smoke.py": "1" * 64,
        },
        "sample_counts": {"requested": 1, "effective": 1, "failures": 0},
        "forward_counts": {"requested": 3, "effective": 3},
        "synthetic_video": {
            "frames": frames, "sampled_fps": 8 / 3,
            "grid_thw": [steps, 32, 32], "seconds_per_grid": 0.75,
            "native_temporal_anchors": native,
            "fractional_temporal_anchors": list(native),
            "legacy_v517_temporal_anchors": [15] * steps,
            "native_matches_fractional_reference": True,
            "changed_position_id_entries": 0,
        },
        "gpu_resources": {
            "device": "NVIDIA GeForce RTX 5070 Ti",
            "total_gib": 15.89, "free_before_gib": 14.0,
            "forwards": [
                {"condition": name, "forward_seconds": scale * (i + 1),
                 "peak_allocated_gib": scale / 10 + i, "finite_logits": True}
                for i, name in enumerate(("native_auto", "explicit_native_noop", "legacy_v517_reference"))
            ],
        },
    }


class QwenTemporalPositionScalingTests(unittest.TestCase):
    def write_inputs(self, root, sizes=(4, 16, 32)):
        paths = []
        for size in sizes:
            path = root / f"{size}.json"
            path.write_text(json.dumps(summary(size, size)), encoding="utf-8")
            paths.append(path)
        return paths

    def test_reproduces_adjacent_ratios_and_binds_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(self.write_inputs(Path(temp)))
        self.assertEqual([row["frames"] for row in result["conditions"]], [4, 16, 32])
        self.assertEqual([row["video_tokens"] for row in result["conditions"]], [512, 2048, 4096])
        self.assertNotEqual(result["conditions"][0]["runner_source_sha256"],
                            result["conditions"][2]["runner_source_sha256"])
        self.assertAlmostEqual(result["comparisons"][0]["forward_time_ratios"]["native_auto"], 4.0)
        self.assertAlmostEqual(result["comparisons"][1]["peak_allocated_memory_ratios"]["native_auto"], 2.0)
        self.assertIn("One run per frame count", result["limitations"][0])
        self.assertIn("Runner source hashes are reported", result["limitations"][3])

    def test_rejects_incomplete_or_duplicate_frame_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.write_inputs(root, (4, 16))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                analyze(paths)
            paths = self.write_inputs(root, (4, 16, 16))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                analyze(paths)

    def test_rejects_mismatched_run_identity_and_bool_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.write_inputs(root)
            d = summary(32, 32)
            d["gpu_resources"]["device"] = "NVIDIA Other"
            paths[-1].write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                analyze(paths)
            d = summary(32, 32)
            d["sample_counts"]["requested"] = True
            paths[-1].write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts"):
                analyze(paths)
            d = summary(32, 32)
            d["source_sha256"]["qwen_temporal_geometry.py"] = "2" * 64
            paths[-1].write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source differs"):
                analyze(paths)

    def test_rejects_geometry_or_measurement_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.write_inputs(root)
            d = summary(32, 32)
            d["synthetic_video"]["native_temporal_anchors"][4] += 1
            paths[-1].write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "anchors"):
                analyze(paths)
            d = summary(32, 32)
            d["gpu_resources"]["forwards"][0]["forward_seconds"] = float("nan")
            paths[-1].write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resource"):
                analyze(paths)


if __name__ == "__main__":
    unittest.main()
