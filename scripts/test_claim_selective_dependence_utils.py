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

    def test_rejects_duplicate_video(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64, "full_score": 1.0, "selected_score": 0.5, "control_score": 0.8}
        with self.assertRaises(ValueError): summarize([row, dict(row)])

    def test_rejects_nonfinite_score(self):
        row = {"video_id": "v1", "target_sha256": "a" * 64, "full_score": float("nan"), "selected_score": 0.5, "control_score": 0.8}
        with self.assertRaises(ValueError): summarize([row])

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
