import hashlib
import json
import unittest

from html5_clock_capture_calibration import build_html5_clock_calibration


class Html5ClockCaptureCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.media = b"synthetic local staged media"
        self.capture = {
            "schema_version": 2,
            "purpose": "local_html5_player_clock_anchor_capture",
            "player_clock": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
            "browser_user_agent": "Synthetic browser fixture",
            "current_src_matches_staged_media": True,
            "source_media_sha256": hashlib.sha256(self.media).hexdigest(),
            "playback_rate": "1", "muted": True, "text_tracks_disabled": True,
            "fit_anchors": [
                {"anchor_id": "f1", "decoder_pts_seconds": "0", "html5_media_time_seconds": "1/10"},
                {"anchor_id": "f2", "decoder_pts_seconds": "10", "html5_media_time_seconds": "101/10"},
                {"anchor_id": "f3", "decoder_pts_seconds": "20", "html5_media_time_seconds": "201/10"},
            ],
            "heldout_anchors": [
                {"anchor_id": "h1", "decoder_pts_seconds": "5", "html5_media_time_seconds": "51/10"},
                {"anchor_id": "h2", "decoder_pts_seconds": "15", "html5_media_time_seconds": "151/10"},
            ],
        }

    def make(self):
        raw = json.dumps(self.capture, sort_keys=True).encode()
        return build_html5_clock_calibration(raw, self.media, "1/100")

    def test_exact_affine_mapping_and_empirical_guard(self):
        result = self.make()
        self.assertEqual(result["slope"], "1")
        self.assertEqual(result["intercept"], "1/10")
        self.assertEqual(result["empirical_guard_band_pts_seconds"], "0")
        self.assertTrue(result["empirical_residual_within_tolerance"])
        self.assertNotIn("heldout_pass", result)
        self.assertEqual(result["uncertainty_status"], "not_evaluated")
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(hashlib.sha256(result["artifact_raw"]).hexdigest(),
                         result["artifact_sha256"])

    def test_heldout_empirical_guard_fails_frozen_tolerance(self):
        self.capture["heldout_anchors"][0]["html5_media_time_seconds"] = "53/10"
        result = self.make()
        self.assertEqual(result["empirical_guard_band_pts_seconds"], "1/5")
        self.assertFalse(result["empirical_residual_within_tolerance"])
        self.assertEqual(result["validation_scope"], "empirical_heldout_residual_only")

    def test_rejects_wrong_media_hash(self):
        self.capture["source_media_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_media_binding_mismatch"):
            self.make()

    def test_rejects_duplicate_json_keys(self):
        raw = json.dumps(self.capture, sort_keys=True).encode()
        raw = raw[:-1] + b',"schema_version":2}'
        with self.assertRaisesRegex(ValueError, "capture_duplicate_json_key"):
            build_html5_clock_calibration(raw, self.media, "1/100")

    def test_rejects_overlap_and_boolean_numeric_values(self):
        self.capture["heldout_anchors"][0]["anchor_id"] = "f1"
        with self.assertRaisesRegex(ValueError, "fit_heldout_identity_overlap"):
            self.make()
        self.capture["heldout_anchors"][0]["anchor_id"] = "h1"
        self.capture["fit_anchors"][0]["decoder_pts_seconds"] = True
        with self.assertRaisesRegex(ValueError, "decoder_pts_invalid"):
            self.make()

    def test_rejects_non_html5_review_player_conditions(self):
        self.capture["playback_rate"] = "2"
        with self.assertRaisesRegex(ValueError, "html5_player_state_invalid"):
            self.make()


if __name__ == "__main__":
    unittest.main()
