import unittest

from claim_selective_dependence_utils import summarize


class Tests(unittest.TestCase):
    def test_video_level_contrast(self):
        digest = "a" * 64
        out = summarize([
            {"video_id": "v1", "target_sha256": digest, "full_score": 2.0, "selected_score": 1.0, "control_score": 1.5},
            {"video_id": "v2", "target_sha256": digest, "full_score": 3.0, "selected_score": 2.0, "control_score": 2.0},
        ])
        self.assertEqual(out["video_level_d"], [0.5, 0.0])
        self.assertEqual(out["mean_d"], 0.25)
        self.assertAlmostEqual(out["sample_sd_d"], 0.3535533905932738)
        self.assertAlmostEqual(out["standard_error_d"], 0.25)
        self.assertAlmostEqual(out["ci95_t_d"][0], -2.9265, places=3)
        self.assertAlmostEqual(out["ci95_t_d"][1], 3.4265, places=3)
        self.assertEqual(out["interval_method"].split(";")[0],
                         "two-sided Student-t interval over independent video-level contrasts")

    def test_single_video_has_no_inferential_interval(self):
        out = summarize([{"video_id": "v1", "target_sha256": "a" * 64,
                          "full_score": 2.0, "selected_score": 1.0,
                          "control_score": 1.5}])
        self.assertEqual(out["mean_d"], 0.5)
        self.assertIsNone(out["sample_sd_d"])
        self.assertIsNone(out["standard_error_d"])
        self.assertIsNone(out["ci95_t_d"])

    def test_constant_effect_has_zero_width_interval(self):
        rows = [{"video_id": f"v{i}", "target_sha256": "a" * 64,
                 "full_score": 2.0, "selected_score": 1.0,
                 "control_score": 1.5} for i in range(3)]
        out = summarize(rows)
        self.assertEqual(out["sample_sd_d"], 0.0)
        self.assertEqual(out["standard_error_d"], 0.0)
        self.assertEqual(out["ci95_t_d"], [0.5, 0.5])

    def test_cancels_shared_full_score_before_floating_point_subtraction(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64,
               "full_score": 1e16, "selected_score": 1.0,
               "control_score": 0.5}
        self.assertEqual(summarize([row])["video_level_d"], [-0.5])

    def test_rejects_duplicate_video(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64, "full_score": 1.0, "selected_score": 0.5, "control_score": 0.8}
        with self.assertRaises(ValueError): summarize([row, dict(row)])

    def test_rejects_nonfinite_score(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64, "full_score": float("nan"), "selected_score": 0.5, "control_score": 0.8}
        with self.assertRaises(ValueError): summarize([row])

    def test_rejects_nonfinite_derived_contrast(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64,
               "full_score": 1e308, "selected_score": -1e308,
               "control_score": 1e308}
        with self.assertRaises(ValueError): summarize([row])

    def test_handles_large_mean_and_rejects_dispersion_overflow(self):
        rows = [{"video_id": f"v{i}", "target_sha256": "a" * 64,
                 "full_score": 0.0, "selected_score": -1e308,
                 "control_score": 0.0} for i in range(2)]
        constant = summarize(rows)
        self.assertEqual(constant["mean_d"], 1e308)
        self.assertEqual(constant["ci95_t_d"], [1e308, 1e308])
        rows[1]["selected_score"] = 1e308  # effects +1e308 and -1e308
        with self.assertRaises(ValueError): summarize(rows)

    def test_allows_video_specific_target_digests_without_echoing_them(self):
        base = {"video_id": "v1", "target_sha256": "a" * 64, "full_score": 1.0, "selected_score": 0.5, "control_score": 0.8}
        other = dict(base, video_id="v2", target_sha256="b" * 64)
        out = summarize([base, other])
        self.assertEqual(out["n_videos"], 2)
        self.assertEqual(out["n_unique_target_hashes"], 2)
        self.assertNotIn("target_sha256", out)

    def test_rejects_bad_id_and_bool_score(self):
        row = {"video_id": 1, "target_sha256": "a" * 64, "full_score": 1.0, "selected_score": 0.5, "control_score": 0.8}
        with self.assertRaises(ValueError): summarize([row])
        row["video_id"] = "v1"; row["full_score"] = True
        with self.assertRaises(ValueError): summarize([row])


if __name__ == "__main__": unittest.main()
