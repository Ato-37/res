from __future__ import annotations

import hashlib
import json
import contextlib
import io
from pathlib import Path
import stat
import tempfile
import unittest

from prepare_html5_clock_calibration_inputs import (
    _write_private_exclusive,
    main,
    prepare_capture_bundle,
)
from html5_clock_capture_calibration import build_html5_clock_calibration


def enc(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    tokens = [f"token_{i:018d}" for i in range(3)]
    assert all(len(x) == 24 for x in tokens)
    plan_rows, observation_rows = [], []
    for i, token in enumerate(tokens):
        anchors = [
            {"anchor_id": "fit_a", "decoder_pts_seconds": "1/2"},
            {"anchor_id": "fit_b", "decoder_pts_seconds": "3/2"},
            {"anchor_id": "holdout_a", "decoder_pts_seconds": "5/2"},
        ]
        plan_rows.append({
            "review_token": token,
            "source_media_sha256": h(f"video-{i}"),
            "fit_anchors": anchors[:2],
            "heldout_anchors": anchors[2:],
        })
        events = []
        for sequence, anchor in enumerate(anchors, start=1):
            events.append({
                "anchor_id": anchor["anchor_id"], "sequence": sequence,
                "html5_current_time_seconds": float(sequence) / 2,
                "request_video_frame_callback_metadata": {
                    "media_time_seconds": float(sequence) / 2,
                    "presented_frames": sequence,
                    "presentation_time_ms": 1000.0 + sequence,
                    "expected_display_time_ms": 1001.0 + sequence,
                    "width": 640, "height": 360,
                    "callback_now_ms": 999.0 + sequence,
                },
                "frame_match_status": "last_presented_frame_metadata_unverified",
                "performance_now_ms": 1000.0 + sequence,
                "media_duration_seconds": 10.0, "playback_rate": 1.0,
                "muted": True, "volume": 0.0, "text_tracks_disabled": True,
                "paused": True, "seeking": False, "ready_state": 4,
                "current_src_is_staged_media": True,
                "visibility_state": "visible", "captured_at_utc": "2026-10-16T08:00:00.000Z",
            })
        observation_rows.append({"review_token": token, "events": events})
    observations = {
        "schema_version": 2,
        "purpose": "local_unvalidated_html5_player_clock_observations",
        "status": "raw_player_clock_observations_only",
        "player_clock": "html5_video_currentTime_seconds",
        "presented_frame_metadata": "requestVideoFrameCallback_when_available_not_decoder_matched",
        "decoder_frame_identity": "not_verified",
        "created_at_utc": "2026-10-16T08:01:00.000Z",
        "browser_user_agent": "synthetic-test-browser",
        "anchor_id_validation": "format_and_per_video_uniqueness_only_unverified",
        "predeclared_plan_binding": "not_checked", "fit_heldout_split": "not_assigned",
        "decoder_pts_pairing": "not_provided", "execution_authorization": "not_granted",
        "records": observation_rows,
    }
    observations_raw = enc(observations)
    index = {
        "schema_version": 1,
        "purpose": "unvalidated_html5_clock_observation_integrity_index",
        "payload_sha256": hashlib.sha256(observations_raw).hexdigest(),
        "created_at_utc": "2026-10-16T08:01:00.000Z",
        "status": "raw_observations_only", "execution_authorization": "not_granted",
    }
    plan = {
        "schema_version": 1, "purpose": "predeclared_html5_clock_pts_anchor_plan",
        "status": "predeclared_frozen", "execution_authorization": "not_granted",
        "records": plan_rows,
    }
    return observations_raw, enc(index), enc(plan)


class PrepareHtml5ClockInputsTests(unittest.TestCase):
    def test_pairs_exact_plan_events_and_remains_unvalidated(self):
        obs_raw, index_raw, plan_raw = fixture()
        result = prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        self.assertEqual(result["status"],
                         "raw_observations_paired_with_declared_plan_unvalidated")
        self.assertEqual(len(result["captures"]), 3)
        self.assertEqual(result["captures"][0]["capture"]["fit_anchors"][0], {
            "anchor_id": "fit_a", "decoder_pts_seconds": "1/2",
            "html5_media_time_seconds": "0.5",
        })
        self.assertEqual(result["captures"][0]["capture"]["player_clock"],
                         "html5_video_requestVideoFrameCallback_mediaTime_seconds")
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_bundle_capture_schema_feeds_empirical_validator_without_authorizing(self):
        obs_raw, index_raw, plan_raw = fixture()
        result = prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        for i, item in enumerate(result["captures"]):
            calibration = build_html5_clock_calibration(
                enc(item["capture"]), f"video-{i}".encode(), "1/20")
            self.assertTrue(calibration["empirical_residual_within_tolerance"])
            self.assertEqual(calibration["execution_authorization"], "not_granted")
            self.assertEqual(calibration["uncertainty_status"], "not_evaluated")

    def test_calibration_uses_presented_media_time_not_click_current_time(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        event = observations["records"][0]["events"][0]
        event["html5_current_time_seconds"] = 7.25
        changed_raw = enc(observations)
        changed_index = json.loads(index_raw)
        changed_index["payload_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        result = prepare_capture_bundle(changed_raw, enc(changed_index), plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        fit_anchor = result["captures"][0]["capture"]["fit_anchors"][0]
        self.assertEqual(fit_anchor["html5_media_time_seconds"], "0.5")
        self.assertNotIn("html5_current_time_seconds", fit_anchor)
        self.assertTrue(any("may be stale" in item for item in result["limitations"]))

    def test_rejects_wrong_plan_or_observation_index_hash(self):
        obs_raw, index_raw, plan_raw = fixture()
        with self.assertRaisesRegex(ValueError, "frozen_plan_hash_mismatch"):
            prepare_capture_bundle(obs_raw, index_raw, plan_raw, "0" * 64)
        bad_index = json.loads(index_raw)
        bad_index["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "observation_index_invalid"):
            prepare_capture_bundle(obs_raw, enc(bad_index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

    def test_rejects_posthoc_or_nonconforming_events(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        observations["fit_heldout_split"] = "not_checked"
        changed_raw = enc(observations)
        changed_index = json.loads(index_raw)
        changed_index["payload_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "raw_observation_schema_invalid"):
            prepare_capture_bundle(changed_raw, enc(changed_index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

        observations = json.loads(obs_raw)
        observations["records"][0]["events"][0]["muted"] = False
        bad_raw = enc(observations)
        index = json.loads(index_raw); index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "observed_event_state_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

        observations = json.loads(obs_raw)
        observations["records"][0]["events"][0]["request_video_frame_callback_metadata"] = None
        bad_raw = enc(observations)
        index = json.loads(index_raw); index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "presented_frame_metadata_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

        observations = json.loads(obs_raw)
        observations["records"][0]["events"][0]["paused"] = False
        bad_raw = enc(observations)
        index = json.loads(index_raw); index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "observed_event_state_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

    def test_rejects_event_set_or_fit_heldout_overlap(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        observations["records"][0]["events"].pop()
        bad_raw = enc(observations)
        index = json.loads(index_raw); index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "observed_anchor_count_mismatch"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

        plan = json.loads(plan_raw)
        plan["records"][0]["heldout_anchors"][0]["anchor_id"] = "fit_a"
        bad_plan = enc(plan)
        with self.assertRaisesRegex(ValueError, "planned_anchor_invalid"):
            prepare_capture_bundle(obs_raw, index_raw, bad_plan,
                                   hashlib.sha256(bad_plan).hexdigest())

    def test_rejects_duplicate_event_sequence_numbers(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        observations["records"][0]["events"][1]["sequence"] = 1
        bad_raw = enc(observations)
        index = json.loads(index_raw)
        index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "observed_event_sequence_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

    def test_private_output_is_exclusive_and_mode_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.json"
            _write_private_exclusive(path, {"execution_authorization": "not_granted"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                _write_private_exclusive(path, {})

    def test_cli_writes_aggregate_private_bundle_exclusively(self):
        obs_raw, index_raw, plan_raw = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = [root / name for name in ("observations.json", "index.json", "plan.json")]
            for path, raw in zip(inputs, (obs_raw, index_raw, plan_raw), strict=True):
                path.write_bytes(raw)
                path.chmod(0o600)
            output = root / "bundle.json"
            args = ["--observations", str(inputs[0]), "--index", str(inputs[1]),
                    "--plan", str(inputs[2]), "--expected-plan-sha256",
                    hashlib.sha256(plan_raw).hexdigest(), "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            saved = json.loads(output.read_bytes())
            self.assertEqual(saved["execution_authorization"], "not_granted")
            self.assertEqual(saved["status"],
                             "raw_observations_paired_with_declared_plan_unvalidated")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
