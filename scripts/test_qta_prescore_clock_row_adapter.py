from __future__ import annotations

import hashlib
import json
import unittest

from qta_prescore_clock_row_adapter import adapt_calibration_to_clock_row
from bind_html5_clock_calibration_provenance import build_bound_html5_clock_calibration


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seal(record: dict) -> dict:
    result = dict(record)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    result["artifact_sha256"] = sha(raw)
    result["artifact_raw"] = raw
    return result


class QtaPrescoreClockRowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.token = "opaque-review-token-0001"
        self.video_sha = sha(b"video")
        self.receipt_sha = sha(b"review-receipt")
        self.capture_raw = json.dumps({
            "schema_version": 3,
            "purpose": "local_html5_player_clock_anchor_capture",
            "player_clock": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
            "capture_policy_id": "rvfc_click_bound_v1",
            "capture_thresholds": {
                "max_callback_age_ms": 250.0,
                "max_request_delay_ms": 250.0,
                "max_click_player_time_delta_seconds": 0.1,
                "max_callback_player_time_delta_seconds": 0.1,
                "callback_wait_timeout_ms": 1000.0,
            },
            "browser_user_agent": "synthetic fixture",
            "current_src_matches_staged_media": True,
            "source_media_sha256": self.video_sha,
            "playback_rate": "1", "muted": True,
            "text_tracks_disabled": True,
            "fit_anchors": [
                {"anchor_id": "fit_begin", "decoder_pts_seconds": "1/10",
                 "html5_media_time_seconds": "1/10"},
                {"anchor_id": "fit_midpoint", "decoder_pts_seconds": "1",
                 "html5_media_time_seconds": "1"},
                {"anchor_id": "fit_end", "decoder_pts_seconds": "3",
                 "html5_media_time_seconds": "3"},
            ],
            "heldout_anchors": [
                {"anchor_id": "heldout_quarter", "decoder_pts_seconds": "2",
                 "html5_media_time_seconds": "2"},
                {"anchor_id": "heldout_three_quarter", "decoder_pts_seconds": "4",
                 "html5_media_time_seconds": "4"},
            ],
        }, separators=(",", ":")).encode()
        self.calibration = seal({
            "clock_scope": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
            "capture_policy_id": "rvfc_click_bound_v1",
            "capture_thresholds": {
                "max_callback_age_ms": 250.0,
                "max_request_delay_ms": 250.0,
                "max_click_player_time_delta_seconds": 0.1,
                "max_callback_player_time_delta_seconds": 0.1,
                "callback_wait_timeout_ms": 1000.0,
            },
            "validation_scope": "empirical_heldout_residual_only",
            "guard_band_source": "max_observed_heldout_residual_pts_seconds",
            "uncertainty_status": "not_evaluated",
            "purpose": "unvalidated_html5_clock_calibration_with_lineage",
            "provenance_status": "capture_and_media_byte_bound_upstream_digests_unverified",
            "upstream_hash_verification": "digest_strings_carried_from_prepared_bundle_not_reauthenticated",
            "source_media_sha256": self.video_sha,
            "review_token": self.token,
            "observations_sha256": sha(b"observations"),
            "observation_index_sha256": sha(b"index"),
            "anchor_plan_sha256": sha(b"plan"),
            "anchor_plan_provenance_sha256": sha(b"plan provenance"),
            "capture_sha256": sha(self.capture_raw),
            "limitations": ["synthetic fixture; no correspondence authenticated"],
            "slope": "1", "intercept": "0",
            "fit_anchor_count": 3, "heldout_anchor_count": 2,
            "empirical_guard_band_pts_seconds": "1/100",
            "frozen_tolerance_seconds": "1/20",
            "empirical_residual_within_tolerance": True,
            "execution_authorization": "not_granted",
            "limitations": [
                "prepared bundle digests are caller-supplied values and are not reauthenticated from original observation/index/plan bytes",
                "a fabricated bundle with matching capture and media bytes can produce this record",
                "browser telemetry, reviewer identity, and decoder-frame correspondence are not authenticated",
                "this record does not authorize scoring or inference",
            ],
        })

    def adapt(self, calibration=None, capture_raw=None, **kwargs):
        return adapt_calibration_to_clock_row(
            review_token=kwargs.get("review_token", self.token),
            source_video_sha256=kwargs.get("source_video_sha256", self.video_sha),
            video_claim_review_receipt_sha256=kwargs.get(
                "video_claim_review_receipt_sha256", self.receipt_sha),
            calibration=self.calibration if calibration is None else calibration,
            capture_raw=self.capture_raw if capture_raw is None else capture_raw,
        )

    def reseal_with(self, **changes):
        record = {key: value for key, value in self.calibration.items()
                  if key not in {"artifact_raw", "artifact_sha256"}}
        record.update(changes)
        return seal(record)

    def test_builds_join_compatible_row_and_uses_union_anchor_span(self):
        row = self.adapt()
        self.assertEqual(row["calibration_binding_payload"], {
            "source_video_sha256": self.video_sha,
            "slope": "1", "intercept": "0",
            "anchor_pts_min_seconds": "1/10",
            "anchor_pts_max_seconds": "4",
            "guard_pts_seconds": "1/100",
        })
        expected = hashlib.sha256(json.dumps(
            row["calibration_binding_payload"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        self.assertEqual(row["calibration_sha256"], expected)
        self.assertEqual(row["review_token"], self.token)
        self.assertEqual(row["video_claim_review_receipt_sha256"], self.receipt_sha)
        self.assertFalse(row["clock_correspondence_authenticated"])
        self.assertEqual(row["execution_authorization"], "not_granted")

    def test_rejects_token_source_or_receipt_digest_mismatch(self):
        with self.assertRaisesRegex(ValueError, "review_token_mismatch"):
            self.adapt(review_token="different-token-12345678")
        with self.assertRaisesRegex(ValueError, "source_video_mismatch"):
            self.adapt(source_video_sha256=sha(b"other"))
        with self.assertRaisesRegex(ValueError, "receipt_sha256_invalid"):
            self.adapt(video_claim_review_receipt_sha256="bad")

    def test_rejects_capture_hash_mismatch_and_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "capture_hash_mismatch"):
            self.adapt(capture_raw=b"{}")
        duplicate = (b'{"fit_anchors":[],"fit_anchors":[],"heldout_anchors":[]}')
        result = self.reseal_with(capture_sha256=sha(duplicate))
        with self.assertRaisesRegex(ValueError, "capture_json_invalid"):
            self.adapt(result, duplicate)

    def test_rejects_failed_frozen_residual_gate(self):
        failed = self.reseal_with(empirical_guard_band_pts_seconds="1/10",
                                  empirical_residual_within_tolerance=False)
        with self.assertRaisesRegex(ValueError, "residual_exceeds"):
            self.adapt(failed)

    def test_rejects_mutated_artifact_and_invalid_anchor_values(self):
        mutated = dict(self.calibration)
        mutated["slope"] = "2"
        with self.assertRaisesRegex(ValueError, "artifact_content_mismatch"):
            self.adapt(mutated)
        bad = json.loads(self.capture_raw)
        bad["fit_anchors"][0]["decoder_pts_seconds"] = "01/2"
        bad_capture = json.dumps(bad, separators=(",", ":")).encode()
        result = self.reseal_with(capture_sha256=sha(bad_capture))
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            self.adapt(result, bad_capture)

    def test_rejects_unknown_or_wrong_split_anchor_ids(self):
        capture = json.loads(self.capture_raw)
        capture["fit_anchors"][0]["anchor_id"] = "heldout_quarter"
        capture["heldout_anchors"][0]["anchor_id"] = "fit_begin"
        raw = json.dumps(capture, separators=(",", ":")).encode()
        calibration = self.reseal_with(capture_sha256=sha(raw))
        with self.assertRaisesRegex(ValueError, "anchor_id_sets_invalid"):
            self.adapt(calibration, raw)

    def test_rejects_missing_or_changed_builder_artifact_fields(self):
        with self.assertRaisesRegex(ValueError, "bound_artifact_keys_invalid"):
            self.adapt({key: value for key, value in self.calibration.items()
                        if key != "clock_scope"})
        bad = self.reseal_with(uncertainty_status="bounded_uncertainty")
        with self.assertRaisesRegex(ValueError, "builder_fields_invalid"):
            self.adapt(bad)

    def test_accepts_actual_bound_upstream_builder_output(self):
        media = b"synthetic clock adapter integration video"
        source_sha = sha(media)
        token = "opaque-integration-token"
        capture = {
            "schema_version": 3,
            "purpose": "local_html5_player_clock_anchor_capture",
            "player_clock": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
            "capture_policy_id": "rvfc_click_bound_v1",
            "capture_thresholds": {
                "max_callback_age_ms": 250.0,
                "max_request_delay_ms": 250.0,
                "max_click_player_time_delta_seconds": 0.1,
                "max_callback_player_time_delta_seconds": 0.1,
                "callback_wait_timeout_ms": 1000.0,
            },
            "browser_user_agent": "synthetic integration fixture",
            "current_src_matches_staged_media": True,
            "source_media_sha256": source_sha,
            "playback_rate": "1", "muted": True,
            "text_tracks_disabled": True,
            "fit_anchors": [
                {"anchor_id": "fit_begin", "decoder_pts_seconds": "0",
                 "html5_media_time_seconds": "0"},
                {"anchor_id": "fit_midpoint", "decoder_pts_seconds": "1",
                 "html5_media_time_seconds": "1"},
                {"anchor_id": "fit_end", "decoder_pts_seconds": "2",
                 "html5_media_time_seconds": "2"},
            ],
            "heldout_anchors": [
                {"anchor_id": "heldout_quarter", "decoder_pts_seconds": "1/2",
                 "html5_media_time_seconds": "1/2"},
                {"anchor_id": "heldout_three_quarter", "decoder_pts_seconds": "3/2",
                 "html5_media_time_seconds": "3/2"},
            ],
        }
        capture_raw = json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()
        bundle = {
            "schema_version": 1,
            "purpose": "unvalidated_html5_clock_calibration_input_bundle",
            "status": "raw_observations_paired_with_declared_plan_unvalidated",
            "observations_sha256": sha(b"obs"),
            "observation_index_sha256": sha(b"index"),
            "anchor_plan_sha256": sha(b"plan"),
            "anchor_plan_provenance_sha256": sha(b"plan provenance"),
            "captures": [
                {"review_token": f"token_{i:018d}", "capture": {}}
                for i in range(3)
            ],
            "execution_authorization": "not_granted",
            "limitations": [],
        }
        bundle["captures"][1] = {"review_token": token, "capture": capture}
        calibration = build_bound_html5_clock_calibration(
            prepared_bundle=bundle, review_token=token, capture_raw=capture_raw,
            media_bytes=media, frozen_tolerance_pts_seconds="1/20")
        row = self.adapt(calibration, capture_raw, review_token=token,
                         source_video_sha256=source_sha)
        self.assertEqual(row["slope"], "1")
        self.assertEqual(row["anchor_pts_min_seconds"], "0")
        self.assertEqual(row["anchor_pts_max_seconds"], "2")
        self.assertFalse(row["clock_correspondence_authenticated"])
        self.assertEqual(row["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
