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
    prepare_capture_bundle as _prepare_capture_bundle,
)
from html5_clock_capture_calibration import build_html5_clock_calibration
from bind_html5_clock_calibration_provenance import build_bound_html5_clock_calibration


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
                "requested_player_time_seconds": float(sequence) / 2,
                "request_started_performance_ms": 900.0 + sequence,
                "frame_match_status": "fresh_click_bound_callback_unverified_decoder",
                "performance_now_ms": 1000.0 + sequence,
                "media_duration_seconds": 10.0, "playback_rate": 1.0,
                "muted": True, "volume": 0.0, "text_tracks_disabled": True,
                "paused": True, "seeking": False, "ready_state": 4,
                "current_src_is_staged_media": True,
                "visibility_state": "visible", "captured_at_utc": "2026-10-16T08:00:00.000Z",
            })
        observation_rows.append({"review_token": token, "events": events})
    observations = {
            "schema_version": 3,
        "purpose": "local_unvalidated_html5_player_clock_observations",
        "status": "raw_player_clock_observations_only",
            "capture_policy_id": "rvfc_click_bound_v1",
            "capture_thresholds": {
                "max_callback_age_ms": 250.0,
                "max_request_delay_ms": 250.0,
                "max_click_player_time_delta_seconds": 0.1,
                "max_callback_player_time_delta_seconds": 0.1,
                "callback_wait_timeout_ms": 1000.0,
            },
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


def provenance_fixture(plan_raw):
    plan = json.loads(plan_raw)
    return enc({
        "schema_version": 1,
        "purpose": "private_clock_anchor_plan_candidate_provenance",
        "status": "candidate_not_jointly_frozen",
        "execution_authorization": "not_granted",
        "input_sha256": {name: h(name) for name in
                          ("worksheet", "lineage_manifest", "custody_manifest")},
        "source_sha256": {name: h(name) for name in
                          ("builder", "decoder", "custody_auditor", "private_output_policy")},
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "created_at_utc": "2026-10-16T08:00:00Z",
        "python_version": "3.12.0", "pyav_version": "18.1.0",
        "ffmpeg_library_versions": {"libavcodec": [62, 28, 102]},
        "joint_freeze_receipt_sha256": None,
        "records": [{
            "review_token": row["review_token"],
            "staged_media_sha256": row["source_media_sha256"],
            "decoded_frame_count": 100,
            "decoded_pts_sequence_sha256": h(f"pts-{i}"),
            "time_base": [1, 30], "pixel_spec": [640, 360, "yuv420p"],
            "decoder_stream_info": {
                "stream_index": 0, "codec_name": "synthetic",
                "stream_time_base": "1/30", "decoder_thread_count": 1,
                "container_format": "mp4", "reported_stream_frame_count": 100,
            },
        } for i, row in enumerate(plan["records"])],
    })


def prepare_capture_bundle(*args):
    """Tests default to a synthetic sidecar; production API requires it explicitly."""
    if len(args) == 4:
        plan_raw = args[2]
        provenance_raw = provenance_fixture(plan_raw)
        args = (*args, provenance_raw, hashlib.sha256(provenance_raw).hexdigest())
    return _prepare_capture_bundle(*args)


class PrepareHtml5ClockInputsTests(unittest.TestCase):
    def test_python_api_rejects_omitted_provenance(self):
        obs_raw, index_raw, plan_raw = fixture()
        with self.assertRaises(TypeError):
            _prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                    hashlib.sha256(plan_raw).hexdigest())

    def test_binder_rejects_legacy_bundle_without_provenance_digest(self):
        obs_raw, index_raw, plan_raw = fixture()
        bundle = prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        legacy = dict(bundle)
        legacy.pop("anchor_plan_provenance_sha256")
        item = legacy["captures"][0]
        with self.assertRaisesRegex(ValueError, "prepared_bundle_invalid"):
            build_bound_html5_clock_calibration(
                prepared_bundle=legacy,
                review_token=item["review_token"],
                capture_raw=enc(item["capture"]),
                media_bytes=b"video-0",
                frozen_tolerance_pts_seconds="1/20",
            )

    def test_provenance_sidecar_binds_plan_and_remains_non_authorizing(self):
        obs_raw, index_raw, plan_raw = fixture()
        provenance_raw = provenance_fixture(plan_raw)
        result = prepare_capture_bundle(
            obs_raw, index_raw, plan_raw, hashlib.sha256(plan_raw).hexdigest(),
            provenance_raw, hashlib.sha256(provenance_raw).hexdigest())
        self.assertEqual(result["anchor_plan_provenance_sha256"],
                         hashlib.sha256(provenance_raw).hexdigest())
        self.assertEqual(result["execution_authorization"], "not_granted")
        item = result["captures"][0]
        bound = build_bound_html5_clock_calibration(
            prepared_bundle=result, review_token=item["review_token"],
            capture_raw=enc(item["capture"]), media_bytes=b"video-0",
            frozen_tolerance_pts_seconds="1/20")
        self.assertEqual(bound["anchor_plan_provenance_sha256"],
                         hashlib.sha256(provenance_raw).hexdigest())
        self.assertEqual(bound["execution_authorization"], "not_granted")

    def test_provenance_cannot_be_replayed_for_another_plan(self):
        obs_raw, index_raw, plan_raw = fixture()
        provenance_raw = provenance_fixture(plan_raw)
        changed_plan = plan_raw + b" "
        with self.assertRaisesRegex(ValueError, "anchor_provenance_schema_or_plan_mismatch"):
            prepare_capture_bundle(
                obs_raw, index_raw, changed_plan, hashlib.sha256(changed_plan).hexdigest(),
                provenance_raw, hashlib.sha256(provenance_raw).hexdigest())

    def test_provenance_digest_mismatch_fails_closed(self):
        obs_raw, index_raw, plan_raw = fixture()
        provenance_raw = provenance_fixture(plan_raw)
        with self.assertRaisesRegex(ValueError, "anchor_provenance_hash_invalid"):
            prepare_capture_bundle(
                obs_raw, index_raw, plan_raw, hashlib.sha256(plan_raw).hexdigest(),
                provenance_raw, "0" * 64)

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

    def test_synthetic_converter_to_calibrator_hash_bindings(self):
        """Check byte-hash continuity; synthetic browser claims remain unauthenticated."""
        obs_raw, index_raw, plan_raw = fixture()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        bundle = prepare_capture_bundle(obs_raw, index_raw, plan_raw, plan_sha)

        self.assertEqual(bundle["observations_sha256"], hashlib.sha256(obs_raw).hexdigest())
        self.assertEqual(bundle["observation_index_sha256"], hashlib.sha256(index_raw).hexdigest())
        self.assertEqual(bundle["anchor_plan_sha256"], plan_sha)

        # The fixture's media bytes deliberately match the synthetic source hash
        # in the plan. This verifies arithmetic and artifact binding only.
        for i, item in enumerate(bundle["captures"]):
            media = f"video-{i}".encode()
            capture_raw = enc(item["capture"])
            calibration = build_html5_clock_calibration(capture_raw, media, "1/20")

            self.assertEqual(calibration["source_media_sha256"], hashlib.sha256(media).hexdigest())
            self.assertEqual(calibration["capture_sha256"], hashlib.sha256(capture_raw).hexdigest())
            self.assertEqual(hashlib.sha256(calibration["artifact_raw"]).hexdigest(),
                             calibration["artifact_sha256"])
            self.assertEqual(json.loads(calibration["artifact_raw"]), {
                key: value for key, value in calibration.items()
                if key not in {"artifact_raw", "artifact_sha256"}
            })
            self.assertEqual(calibration["execution_authorization"], "not_granted")
            # Upstream observation/index/plan digests are bundle-level only;
            # the calibrator artifact currently carries capture and media
            # hashes, not those transitive references or the stable row token.
            self.assertNotIn("anchor_plan_sha256", calibration)
            self.assertNotIn("observation_index_sha256", calibration)
            self.assertNotIn("review_token", calibration)

    def test_bound_calibration_retains_lineage_hashes_and_token(self):
        obs_raw, index_raw, plan_raw = fixture()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        bundle = prepare_capture_bundle(obs_raw, index_raw, plan_raw, plan_sha)
        item = bundle["captures"][1]
        capture_raw = enc(item["capture"])
        record = build_bound_html5_clock_calibration(
            prepared_bundle=bundle,
            review_token=item["review_token"],
            capture_raw=capture_raw,
            media_bytes=b"video-1",
            frozen_tolerance_pts_seconds="1/20",
        )

        self.assertEqual(record["observations_sha256"], hashlib.sha256(obs_raw).hexdigest())
        self.assertEqual(record["observation_index_sha256"], hashlib.sha256(index_raw).hexdigest())
        self.assertEqual(record["anchor_plan_sha256"], plan_sha)
        self.assertEqual(record["review_token"], item["review_token"])
        self.assertEqual(record["capture_sha256"], hashlib.sha256(capture_raw).hexdigest())
        self.assertEqual(record["source_media_sha256"], hashlib.sha256(b"video-1").hexdigest())
        self.assertEqual(record["provenance_status"],
                         "capture_and_media_byte_bound_upstream_digests_unverified")
        self.assertEqual(record["upstream_hash_verification"],
                         "digest_strings_carried_from_prepared_bundle_not_reauthenticated")
        self.assertEqual(record["execution_authorization"], "not_granted")
        self.assertEqual(hashlib.sha256(record["artifact_raw"]).hexdigest(),
                         record["artifact_sha256"])
        self.assertEqual(json.loads(record["artifact_raw"]), {
            key: value for key, value in record.items()
            if key not in {"artifact_raw", "artifact_sha256"}
        })

    def test_fabricated_bundle_hashes_are_carried_but_explicitly_untrusted(self):
        obs_raw, index_raw, plan_raw = fixture()
        bundle = prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        item = bundle["captures"][0]
        fabricated = dict(bundle,
                          observations_sha256="f" * 64,
                          observation_index_sha256="e" * 64,
                          anchor_plan_sha256="d" * 64)
        record = build_bound_html5_clock_calibration(
            prepared_bundle=fabricated,
            review_token=item["review_token"],
            capture_raw=enc(item["capture"]),
            media_bytes=b"video-0",
            frozen_tolerance_pts_seconds="1/20",
        )
        self.assertEqual(record["observations_sha256"], "f" * 64)
        self.assertIn("fabricated bundle", " ".join(record["limitations"]))
        self.assertEqual(record["provenance_status"],
                         "capture_and_media_byte_bound_upstream_digests_unverified")
        self.assertEqual(record["execution_authorization"], "not_granted")

    def test_bound_calibration_rejects_token_or_media_mismatch(self):
        obs_raw, index_raw, plan_raw = fixture()
        bundle = prepare_capture_bundle(obs_raw, index_raw, plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        item = bundle["captures"][0]
        args = {
            "prepared_bundle": bundle,
            "review_token": item["review_token"],
            "capture_raw": enc(item["capture"]),
            "media_bytes": b"video-0",
            "frozen_tolerance_pts_seconds": "1/20",
        }
        wrong_token = dict(args, review_token=bundle["captures"][1]["review_token"])
        with self.assertRaisesRegex(ValueError, "capture_bundle_content_mismatch"):
            build_bound_html5_clock_calibration(**wrong_token)
        with self.assertRaisesRegex(ValueError, "source_media_binding_mismatch"):
            build_bound_html5_clock_calibration(**dict(args, media_bytes=b"wrong media"))

    def test_rejects_presented_media_time_far_from_player_time(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        event = observations["records"][0]["events"][0]
        event["html5_current_time_seconds"] = 7.25
        changed_raw = enc(observations)
        changed_index = json.loads(index_raw)
        changed_index["payload_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "presented_frame_stale_or_time_mismatch"):
            prepare_capture_bundle(changed_raw, enc(changed_index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

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
        observations["records"][0]["events"][0]["frame_match_status"] = "last_presented_frame_metadata_unverified"
        bad_raw = enc(observations)
        index = json.loads(index_raw); index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "observed_event_state_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

    def test_rejects_stale_or_player_time_mismatched_presented_frame(self):
        obs_raw, index_raw, plan_raw = fixture()
        for field, value in (("performance_now_ms", 2000.0),
                             ("html5_current_time_seconds", 7.25),
                             ("requested_player_time_seconds", 7.25),
                             ("request_started_performance_ms", 700.0)):
            observations = json.loads(obs_raw)
            observations["records"][0]["events"][0][field] = value
            bad_raw = enc(observations)
            index = json.loads(index_raw)
            index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "presented_frame_stale_or_time_mismatch"):
                prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                       hashlib.sha256(plan_raw).hexdigest())

    def test_capture_threshold_policy_is_explicit_and_fixed(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        observations["capture_thresholds"]["max_request_delay_ms"] = 500.0
        bad_raw = enc(observations)
        index = json.loads(index_raw)
        index["payload_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "raw_observation_schema_invalid"):
            prepare_capture_bundle(bad_raw, enc(index), plan_raw,
                                   hashlib.sha256(plan_raw).hexdigest())

    def test_fresh_presented_frame_may_be_captured_while_playing(self):
        obs_raw, index_raw, plan_raw = fixture()
        observations = json.loads(obs_raw)
        observations["records"][0]["events"][0]["paused"] = False
        fresh_raw = enc(observations)
        index = json.loads(index_raw)
        index["payload_sha256"] = hashlib.sha256(fresh_raw).hexdigest()
        result = prepare_capture_bundle(fresh_raw, enc(index), plan_raw,
                                        hashlib.sha256(plan_raw).hexdigest())
        self.assertEqual(len(result["captures"]), 3)

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
            provenance_raw = provenance_fixture(plan_raw)
            inputs = [root / name for name in
                      ("observations.json", "index.json", "plan.json", "provenance.json")]
            for path, raw in zip(inputs, (obs_raw, index_raw, plan_raw, provenance_raw), strict=True):
                path.write_bytes(raw)
                path.chmod(0o600)
            output = root / "bundle.json"
            args = ["--observations", str(inputs[0]), "--index", str(inputs[1]),
                    "--plan", str(inputs[2]), "--expected-plan-sha256",
                    hashlib.sha256(plan_raw).hexdigest(), "--plan-provenance", str(inputs[3]),
                    "--expected-plan-provenance-sha256", hashlib.sha256(provenance_raw).hexdigest(),
                    "--output", str(output)]
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
