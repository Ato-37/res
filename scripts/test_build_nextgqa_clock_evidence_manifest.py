#!/usr/bin/env python3
"""Synthetic-only tests for the offline clock evidence manifest builder."""
from __future__ import annotations

import hashlib
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from build_nextgqa_clock_evidence_manifest import INPUTS, build_manifest, main
from build_blinded_frame_pair_review import _html
from test_audit_nextgqa_clock_frame_evidence import ClockFrameEvidenceAuditTests, enc, sha
from test_nextgqa_capture_operator_attestation import CaptureAttestationTests
from nextgqa_capture_operator_attestation import create_attestation
from build_nextgqa_clock_evidence_manifest import (
    EVIDENCE_DOMAIN_V3, _artifact_directory, _canonical as evidence_canonical,
    _load_public_der, _sign, verify_manifest_signature,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _rgb_hash_for_test(image):
    rgb = image.convert("RGB")
    return sha(f"{rgb.width}x{rgb.height}:".encode("ascii") + rgb.tobytes())


class BuilderSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.crop_dir = self.root / "crops"
        self.crop_dir.mkdir(mode=0o700)
        self.screenshot_dir = self.root / "screenshots"
        self.screenshot_dir.mkdir(mode=0o700)
        self.operator_fixture = CaptureAttestationTests()
        self.operator_fixture.setUp()
        self.key = self.root / "frame-reviewer.pem"
        self.pub = self.root / "frame-reviewer.pub.pem"
        self.key.write_bytes(self.operator_fixture.keys["frame"][0])
        self.pub.write_bytes(self.operator_fixture.keys["frame"][1])
        os.chmod(self.key, 0o600)
        os.chmod(self.pub, 0o600)
        self.files = self._synthetic_inputs()

    def tearDown(self):
        self.operator_fixture.tearDown()
        self.temp.cleanup()

    def _synthetic_inputs(self):
        for directory in (self.crop_dir, self.screenshot_dir):
            for item in directory.iterdir():
                item.unlink()
        # Tokens must have exactly 24 characters.
        tokens = [f"tok_{i:020d}" for i in range(3)]
        anchors = ["fit_begin", "heldout_quarter", "fit_midpoint",
                   "heldout_three_quarter", "fit_end"]
        rows_plan, rows_obs, rows_capture = [], [], []
        rows_crop, rows_decoder, rows_identity, rows_review, pairs = [], [], [], [], []
        for ti, token in enumerate(tokens):
            fit, held = [], []
            for ai, anchor in enumerate(anchors):
                target = {"anchor_id": anchor, "decoder_pts_seconds": str(ai)}
                (held if anchor.startswith("heldout") else fit).append(target)
            rows_plan.append({"review_token": token, "source_media_sha256": sha(token.encode()),
                              "fit_anchors": fit, "heldout_anchors": held})
            events = []
            for ai, anchor in enumerate(anchors):
                events.append({"anchor_id": anchor, "sequence": ai + 1,
                               "frame_match_status": "fresh_click_bound_callback_unverified_decoder",
                               "captured_at_utc": "2026-01-01T00:00:00.000Z",
                               "requested_player_time_seconds": float(ai),
                               "request_started_performance_ms": 1000.0 + ai,
                               "html5_current_time_seconds": float(ai),
                               "request_video_frame_callback_metadata": {
                                   "media_time_seconds": float(ai), "presented_frames": ai + 1,
                                   "presentation_time_ms": 998.0 + ai,
                                   "expected_display_time_ms": 999.0 + ai,
                                   "width": 2, "height": 2, "callback_now_ms": 1002.0 + ai,
                               },
                               "performance_now_ms": 1005.0 + ai,
                               "media_duration_seconds": 5.0, "playback_rate": 1.0,
                               "muted": True, "volume": 0.0, "text_tracks_disabled": True,
                               "paused": True, "seeking": False, "ready_state": 4,
                               "current_src_is_staged_media": True, "visibility_state": "visible"})
                screenshot_file = token + "_" + anchor + ".png"
                screenshot = Image.new("RGB", (100, 100), (ti * 20, ai * 20, 7))
                import io
                screenshot_buffer = io.BytesIO()
                screenshot.save(screenshot_buffer, format="PNG")
                screenshot_raw = screenshot_buffer.getvalue()
                (self.screenshot_dir / screenshot_file).write_bytes(screenshot_raw)
                os.chmod(self.screenshot_dir / screenshot_file, 0o600)
                shot_sha = sha(screenshot_raw)
                rows_capture.append({
                    "review_token": token, "anchor_id": anchor,
                    "screenshot_file": screenshot_file, "screenshot_sha256": shot_sha,
                    "screenshot_captured_at_utc": "2026-01-01T00:00:00.000Z",
                    "capture_method": "windows_gdi_copyfromscreen_foreground_window_v1",
                    "foreground_process": "msedge", "foreground_pid": 123,
                    "foreground_window_title": "Private offline video review",
                    "foreground_window_rect_xywh": [0, 0, 100, 100],
                    "virtual_screen_rect_xywh": [0, 0, 100, 100],
                    "hotkey_message_tick_ms": ai, "capture_complete_tick_ms": ai + 10,
                    "hotkey_to_capture_ms": 10,
                })
                image = screenshot.crop((0, 0, 2, 2)).convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                crop_raw = buf.getvalue()
                crop_file = token + "_" + anchor + "_crop.png"
                (self.crop_dir / crop_file).write_bytes(crop_raw)
                os.chmod(self.crop_dir / crop_file, 0o600)
                crop_sha = sha(crop_raw)
                decoder_image = Image.new("RGB", (2, 2), (ti * 20 + 1, ai * 20 + 2, 8))
                decoder_buffer = io.BytesIO()
                decoder_image.save(decoder_buffer, format="PNG")
                decoder_png = decoder_buffer.getvalue()
                decoder_png_sha = sha(decoder_png)
                self.decoder_pngs = getattr(self, "decoder_pngs", {})
                self.decoder_pngs[(token, anchor)] = decoder_png
                decoder_rgb_sha = _rgb_hash_for_test(decoder_image)
                # 24-character opaque-shaped synthetic identifier.
                pair_id = f"pair{ti:02d}{ai:02d}".ljust(24, "x")
                crop_on_a = (ti + ai) % 2 == 0
                pairs.append({"pair_id": pair_id,
                              "A_sha256": crop_sha if crop_on_a else decoder_png_sha,
                              "B_sha256": decoder_png_sha if crop_on_a else crop_sha})
                rows_identity.append({"pair_id": pair_id, "review_token": token,
                                      "anchor_id": anchor,
                                      "A_source": "screenshot_crop" if crop_on_a else "decoder_frame",
                                      "B_source": "decoder_frame" if crop_on_a else "screenshot_crop"})
                rows_review.append({"pair_id": pair_id, "crop_purity_attestation": True,
                                    "decision": "match"})
                rows_crop.append({"review_token": token, "anchor_id": anchor,
                                  "screenshot_file": screenshot_file,
                                  "screenshot_sha256": shot_sha,
                                  "bbox_xywh": [0, 0, 2, 2], "crop_file": crop_file,
                                  "crop_sha256": crop_sha,
                                  "crop_rgb_dimensions": [2, 2],
                                  "crop_purity": "human_review_required"})
                rows_decoder.append({"review_token": token, "anchor_id": anchor,
                                     "source_media_sha256": sha(token.encode()),
                                     "decoder_frame_index": ai,
                                     "decoder_relative_pts_seconds": str(ai),
                                     "rgb_sha256": decoder_rgb_sha,
                                     "png_sha256": decoder_png_sha,
                                     "png_file": token + "_" + anchor + ".png"})
            rows_obs.append({"review_token": token, "events": events})

        plan = {"schema_version": 1, "purpose": "predeclared_html5_clock_pts_anchor_plan",
                "status": "predeclared_frozen", "execution_authorization": "not_granted",
                "records": rows_plan}
        plan_raw = canonical(plan)
        prov = {"schema_version": 1, "purpose": "private_clock_anchor_plan_candidate_provenance",
                "status": "candidate_not_jointly_frozen", "execution_authorization": "not_granted",
                "input_sha256": {key: sha(key.encode()) for key in
                                 ("worksheet", "lineage_manifest", "custody_manifest")},
                "source_sha256": {key: sha(key.encode()) for key in
                                   ("builder", "decoder", "custody_auditor", "private_output_policy")},
                "plan_sha256": sha(plan_raw), "joint_freeze_receipt_sha256": None,
                "created_at_utc": "2026-01-01T00:00:00Z", "python_version": "3.12.0",
                "pyav_version": "synthetic", "ffmpeg_library_versions": {"codec": [1, 2, 3]},
                "records": [{"review_token": token, "staged_media_sha256": sha(token.encode()),
                             "decoded_frame_count": 5, "decoded_pts_sequence_sha256": sha(b"pts"),
                             "time_base": [1, 1], "pixel_spec": [224, 224, "rgb24"],
                             "decoder_stream_info": {"stream_index": 0, "codec_name": "synthetic",
                                 "stream_time_base": "1/1", "decoder_thread_count": 1,
                                 "container_format": "synthetic", "reported_stream_frame_count": 5}}
                            for token in tokens]}
        obs = {"schema_version": 3, "purpose": "local_unvalidated_html5_player_clock_observations",
               "status": "raw_player_clock_observations_only",
               "capture_policy_id": "rvfc_click_bound_v1",
               "capture_thresholds": {"max_callback_age_ms": 250.0, "max_request_delay_ms": 250.0,
                   "max_click_player_time_delta_seconds": 0.1,
                   "max_callback_player_time_delta_seconds": 0.1, "callback_wait_timeout_ms": 1000.0},
               "player_clock": "html5_video_currentTime_seconds",
               "presented_frame_metadata": "requestVideoFrameCallback_when_available_not_decoder_matched",
               "decoder_frame_identity": "not_verified", "created_at_utc": "2026-01-01T00:00:00Z",
               "browser_user_agent": "synthetic-browser",
               "anchor_id_validation": "format_and_per_video_uniqueness_only_unverified",
               "predeclared_plan_binding": "not_checked", "fit_heldout_split": "not_assigned",
               "decoder_pts_pairing": "not_provided", "execution_authorization": "not_granted",
               "records": rows_obs}
        obs_raw = canonical(obs)
        receipt = {"schema_version": 1,
                   "purpose": "private_windows_compositor_screenshot_capture_receipt_candidate",
                   "status": "self_reported_capture_receipt_not_independently_authenticated",
                   "execution_authorization": "not_granted", "plan_sha256": sha(plan_raw),
                   "plan_provenance_sha256": sha(canonical(prov)),
                   "capture_method": "windows_gdi_copyfromscreen_foreground_window_v1",
                   "capture_helper_file_sha256": sha(b"synthetic helper"),
                   "created_at_utc": "2026-01-01T00:00:00Z",
                   "records": rows_capture}
        package = {"schema_version": 1, "purpose": "private_blinded_frame_pair_review_package",
                   "status": "not_authorizing", "execution_authorization": "not_granted",
                   "pair_count": 15, "package_nonce": "a" * 64, "plan_sha256": sha(plan_raw),
                   "pairs": pairs}
        package_raw = canonical(package)
        files_for_html = {}
        for item in rows_identity:
            key = (item["review_token"], item["anchor_id"])
            crop_raw = (self.crop_dir / (key[0] + "_" + key[1] + "_crop.png")).read_bytes()
            sides = {"screenshot_crop": crop_raw, "decoder_frame": self.decoder_pngs[key]}
            files_for_html[item["pair_id"]] = {
                "id": item["pair_id"],
                "A": "data:image/png;base64," + base64.b64encode(sides[item["A_source"]]).decode(),
                "B": "data:image/png;base64," + base64.b64encode(sides[item["B_source"]]).decode(),
            }
        viewer_pairs = [files_for_html[row["pair_id"]] for row in rows_identity]
        template = _html("__PACKAGE_SHA__", "__PACKAGE_NONCE__", viewer_pairs)
        package["reviewer_html_template_sha256"] = sha(template)
        package_raw = canonical(package)
        reviewer_html_raw = _html(sha(package_raw), package["package_nonce"], viewer_pairs)
        crop_map_raw = b"synthetic crop map bytes\n"
        files = {
            "plan": plan_raw,
            "provenance": canonical(prov),
            "observations": obs_raw,
            "observation_index": canonical({"schema_version": 1,
                "purpose": "unvalidated_html5_clock_observation_integrity_index",
            "payload_sha256": sha(obs_raw), "created_at_utc": "2026-01-01T00:00:00Z",
            "status": "raw_observations_only",
                "execution_authorization": "not_granted"}),
            "capture_receipt": canonical(receipt),
            "crop_crosswalk": canonical({"schema_version": 1,
                "purpose": "private_candidate_screenshot_crop_crosswalk", "status": "candidate_only",
                "execution_authorization": "not_granted", "plan_sha256": sha(plan_raw),
                "plan_provenance_sha256": sha(canonical(prov)),
                "capture_receipt_sha256": sha(canonical(receipt)),
                "crop_map_sha256": sha(crop_map_raw),
                "bbox_video_only_purity": "not_established_human_review_required",
                "records": rows_crop}),
            "decoder_crosswalk": canonical({"schema_version": 1,
                "purpose": "private_candidate_decoder_anchor_png_crosswalk",
                "status": "candidate_artifact_export_only", "execution_authorization": "not_granted",
                "plan_sha256": sha(plan_raw), "provenance_sha256": sha(canonical(prov)),
                "effective_anchors": 15, "records": rows_decoder}),
            "pair_package": package_raw,
            "pair_reviewer_html": reviewer_html_raw,
            "identity_map": canonical({"schema_version": 1,
                "purpose": "private_blinded_frame_pair_identity_map", "status": "custodian_only",
                "package_manifest_sha256": sha(package_raw), "records": rows_identity}),
            "review_export": canonical({"schema_version": 1,
                "purpose": "private_blinded_frame_pair_review_export", "status": "not_authorizing",
                "execution_authorization": "not_granted", "package_manifest_sha256": sha(package_raw),
                "package_nonce": package["package_nonce"], "reviewer_pseudonym": "synthetic-reader",
                "reviewer_role": "independent_frame_pair_reviewer",
                "independence_attestation": "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring",
                "reviewer_provenance_status": "self_reported_unauthenticated", "signature_status": "none",
                "reviewed_at_utc": "2026-01-01T00:01:00Z",
                "records": rows_review}),
            "crop_map": crop_map_raw,
            "integrity_roster": self.operator_fixture.integrity,
            "release_roster": self.operator_fixture.release,
        }
        operator_inputs = {
            "plan_raw": files["plan"], "expected_plan_sha256": sha(files["plan"]),
            "plan_provenance_raw": files["provenance"],
            "expected_plan_provenance_sha256": sha(files["provenance"]),
            "observations_raw": files["observations"],
            "observation_index_raw": files["observation_index"],
            "capture_receipt_raw": files["capture_receipt"],
            "crop_map_raw": crop_map_raw,
            "crop_crosswalk_raw": files["crop_crosswalk"],
            "screenshot_files": {p.name: p.read_bytes() for p in self.screenshot_dir.iterdir()},
            "crop_files": {p.name: p.read_bytes() for p in self.crop_dir.iterdir()},
            "integrity_roster_raw": files["integrity_roster"],
            "release_roster_raw": files["release_roster"],
            "root_public_key_pem": self.operator_fixture.keys["root"][1],
            "expected_root_public_der_sha256": self.operator_fixture.root_pin,
        }
        files["capture_operator_attestation"] = create_attestation(
            reviewer_private_key_pem=self.operator_fixture.keys["capture"][0],
            reviewer_public_key_pem=self.operator_fixture.keys["capture"][1],
            **operator_inputs)
        self.assertEqual(set(files), set(INPUTS))
        self.root_public_key_raw = self.operator_fixture.keys["root"][1]
        self.root_pin = self.operator_fixture.root_pin
        return files

    def _build(self):
        public_raw = self.pub.read_bytes()
        return build_manifest(input_files=self.files, crop_dir=self.crop_dir,
                              screenshot_dir=self.screenshot_dir,
                              private_key_path=self.key, public_key_path=self.pub,
                              expected_public_key_sha256=sha(public_raw),
                              expected_capture_helper_sha256=sha(b"synthetic helper"),
                              root_public_key_raw=self.root_public_key_raw,
                              expected_root_public_der_sha256=self.root_pin,
                              reviewer_pseudonym="synthetic-reader",
                              created_at_utc="2026-01-01T00:01:00Z",
                              attest_independence=True, attest_visual_comparison=True)

    def test_builds_v3_signed_candidate_with_authority_denied(self):
        result = self._build()
        self.assertEqual(len(result["records"]), 15)
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["purpose"], "private_clock_frame_correspondence_reader_attestation_v3")
        self.assertEqual(len(result["signature_ed25519_base64"]), 88)
        self.assertEqual(result["capture_operator_signature_verified"], True)
        self.assertEqual(result["capture_operator_human_identity_verified"], False)
        self.assertEqual(result["capture_operator_process_authenticated"], False)
        self.assertEqual(result["capture_operator_frame_correspondence_reviewed"], False)
        self.assertEqual(result["capture_operator_score_ready"], False)
        self.assertEqual(result["reviewer_identity_verified"], False)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertTrue(all(row["correspondence_decision"] == "match" for row in result["records"]))
        self.assertEqual(result["reviewer_role"], "frame_reviewer")
        self.assertEqual(result["frame_reviewer_key_id"], "frame-key")
        self.assertTrue(result["frame_reviewer_role_signature_verified"])
        raw = canonical(result)
        self.assertEqual(verify_manifest_signature(
            manifest_raw=raw, reviewer_public_key_raw=self.pub.read_bytes(),
            expected_reviewer_public_key_sha256=sha(self.pub.read_bytes()))["status"],
            "evidence_manifest_v3_signature_valid_non_authorizing")

    def test_identity_map_exact_schema_is_required_before_signing(self):
        original = self.files["identity_map"]
        mutations = (
            ("unknown top-level field", "identity_map_schema_invalid",
             lambda value: value.update(unexpected=True)),
            ("boolean schema version", "identity_map_schema_invalid",
             lambda value: value.update(schema_version=True)),
            ("unknown record field", "identity_row_schema_invalid",
             lambda value: value["records"][0].update(unexpected=True)),
        )
        for label, error, mutate in mutations:
            with self.subTest(case=label):
                value = json.loads(original)
                mutate(value)
                self.files["identity_map"] = canonical(value)
                with self.assertRaisesRegex(ValueError, error):
                    self._build()
        self.files["identity_map"] = original

    def test_unrostered_reviewer_key_cannot_sign_candidate(self):
        capture_private, capture_public = self.operator_fixture.keys["capture"]
        private_path = self.root / "capture-operator.pem"
        public_path = self.root / "capture-operator.pub.pem"
        private_path.write_bytes(capture_private)
        public_path.write_bytes(capture_public)
        private_path.chmod(0o600)
        public_path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "reviewer_key_not_rostered_frame_reviewer"):
            build_manifest(
                input_files=self.files, crop_dir=self.crop_dir,
                screenshot_dir=self.screenshot_dir,
                private_key_path=private_path, public_key_path=public_path,
                expected_public_key_sha256=sha(capture_public),
                expected_capture_helper_sha256=sha(b"synthetic helper"),
                root_public_key_raw=self.root_public_key_raw,
                expected_root_public_der_sha256=self.root_pin,
                reviewer_pseudonym="synthetic-reader",
                created_at_utc="2026-01-01T00:01:00Z",
                attest_independence=True, attest_visual_comparison=True)

    def test_capture_operator_signature_rechecks_screenshots_and_crops(self):
        first = next(iter(json.loads(self.files["capture_receipt"])["records"]))
        path = self.screenshot_dir / first["screenshot_file"]
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ValueError, "capture_operator_.*screenshot_artifact_hash_mismatch"):
            self._build()
        path.write_bytes(original)
        os.chmod(path, 0o600)
        crop = json.loads(self.files["crop_crosswalk"])["records"][0]
        crop_path = self.crop_dir / crop["crop_file"]
        crop_original = crop_path.read_bytes()
        crop_path.write_bytes(crop_original + b"tamper")
        os.chmod(crop_path, 0o600)
        with self.assertRaisesRegex(ValueError, "capture_operator_.*crop_artifact_hash_mismatch"):
            self._build()

    def test_capture_operator_bad_signature_and_manifest_downgrade_rejected(self):
        envelope = json.loads(self.files["capture_operator_attestation"])
        sig = bytearray(base64.b64decode(envelope["signature_ed25519_base64"]))
        sig[0] ^= 1
        envelope["signature_ed25519_base64"] = base64.b64encode(sig).decode()
        self.files["capture_operator_attestation"] = canonical(envelope)
        with self.assertRaisesRegex(ValueError, "capture_operator_capture_operator_signature_invalid"):
            self._build()
        self.files = self._synthetic_inputs()
        result = self._build()
        result["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "evidence_manifest_schema_invalid"):
            verify_manifest_signature(
                manifest_raw=canonical(result), reviewer_public_key_raw=self.pub.read_bytes(),
                expected_reviewer_public_key_sha256=sha(self.pub.read_bytes()))

    def test_missing_duplicate_or_ambiguous_review_rows_fail_closed(self):
        review = json.loads(self.files["review_export"])
        review["records"].pop()
        self.files["review_export"] = canonical(review)
        with self.assertRaisesRegex(ValueError, "review_record_count_invalid"):
            self._build()
        self.files = self._synthetic_inputs()
        review = json.loads(self.files["review_export"])
        review["records"][1] = dict(review["records"][0])
        self.files["review_export"] = canonical(review)
        with self.assertRaisesRegex(ValueError, "review_pair_duplicate_or_invalid"):
            self._build()
        self.files = self._synthetic_inputs()
        review = json.loads(self.files["review_export"])
        review["records"][0]["decision"] = "ambiguous"
        self.files["review_export"] = canonical(review)
        with self.assertRaisesRegex(ValueError, "review_not_all_pure_and_matching"):
            self._build()

    def test_duplicate_identity_join_is_rejected(self):
        identity = json.loads(self.files["identity_map"])
        identity["records"][1] = dict(identity["records"][0])
        self.files["identity_map"] = canonical(identity)
        with self.assertRaisesRegex(ValueError, "identity_pair_ambiguous_or_invalid"):
            self._build()

    def test_decoder_pts_must_match_the_plan_before_signing(self):
        decoder = json.loads(self.files["decoder_crosswalk"])
        decoder["records"][0]["decoder_relative_pts_seconds"] = "99"
        self.files["decoder_crosswalk"] = canonical(decoder)
        with self.assertRaisesRegex(ValueError, "decoder_crosswalk_plan_pts_mismatch"):
            self._build()

    def test_equivalent_but_noncanonical_decoder_pts_is_rejected(self):
        decoder = json.loads(self.files["decoder_crosswalk"])
        decoder["records"][0]["decoder_relative_pts_seconds"] = "0/1"
        self.files["decoder_crosswalk"] = canonical(decoder)
        with self.assertRaisesRegex(ValueError, "decoder_crosswalk_plan_pts_mismatch"):
            self._build()

    def test_missing_expected_anchor_fails_with_value_error(self):
        receipt = json.loads(self.files["capture_receipt"])
        receipt["records"][0]["anchor_id"] = "fit_midpoint"
        self.files["capture_receipt"] = canonical(receipt)
        with self.assertRaisesRegex(ValueError, "capture_receipt_identity_duplicate_or_invalid"):
            self._build()

    def test_malformed_capture_declaration_is_rejected_before_signing(self):
        receipt = json.loads(self.files["capture_receipt"])
        receipt["records"][0]["hotkey_to_capture_ms"] = 251
        self.files["capture_receipt"] = canonical(receipt)
        with self.assertRaisesRegex(ValueError, "capture_receipt_timing_invalid"):
            self._build()

    def test_capture_helper_hash_must_match_caller_pin(self):
        public_raw = self.pub.read_bytes()
        with self.assertRaisesRegex(ValueError, "capture_helper_hash_pin_mismatch"):
            build_manifest(input_files=self.files, crop_dir=self.crop_dir,
                           screenshot_dir=self.screenshot_dir,
                           private_key_path=self.key, public_key_path=self.pub,
                           expected_public_key_sha256=sha(public_raw),
                           expected_capture_helper_sha256=sha(b"different helper"),
                           root_public_key_raw=self.root_public_key_raw,
                           expected_root_public_der_sha256=self.root_pin,
                           reviewer_pseudonym="synthetic-reader",
                           created_at_utc="2026-01-01T00:01:00Z",
                           attest_independence=True, attest_visual_comparison=True)

    def test_malformed_unhashable_plan_anchor_fails_closed(self):
        plan = json.loads(self.files["plan"])
        plan["records"][0]["fit_anchors"][0]["anchor_id"] = []
        self.files["plan"] = canonical(plan)
        provenance = json.loads(self.files["provenance"])
        provenance["plan_sha256"] = sha(self.files["plan"])
        self.files["provenance"] = canonical(provenance)
        with self.assertRaisesRegex(ValueError, "plan_anchor_invalid"):
            self._build()

    def test_non_string_digest_input_fails_with_value_error(self):
        receipt = json.loads(self.files["capture_receipt"])
        receipt["records"][0]["screenshot_sha256"] = []
        self.files["capture_receipt"] = canonical(receipt)
        with self.assertRaisesRegex(ValueError, "capture_receipt_screenshot_hash_invalid"):
            self._build()

    def test_changed_crop_bytes_fail_closed(self):
        row = json.loads(self.files["crop_crosswalk"])["records"][0]
        path = self.crop_dir / row["crop_file"]
        path.write_bytes(b"changed synthetic bytes")
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ValueError, "capture_operator_crop_artifact_hash_mismatch"):
            self._build()

    def test_cli_writes_private_candidate_and_reports_caller_pin_limit(self):
        input_paths = {}
        for name, raw in self.files.items():
            path = self.root / (name + ".json")
            path.write_bytes(raw)
            os.chmod(path, 0o600)
            input_paths[name] = path
        output = self.root / "signed_candidate"
        argv = []
        for name in INPUTS:
            argv += ["--" + name.replace("_", "-"), str(input_paths[name])]
        root_key = self.root / "trust-root.pub.pem"
        root_key.write_bytes(self.root_public_key_raw)
        os.chmod(root_key, 0o600)
        argv += ["--crop-dir", str(self.crop_dir),
                 "--screenshot-dir", str(self.screenshot_dir),
                 "--trust-root-public-key", str(root_key),
                 "--expected-trust-root-der-sha256", self.root_pin,
                 "--reviewer-private-key", str(self.key),
                 "--reviewer-public-key", str(self.pub),
                 "--expected-reviewer-public-key-sha256", sha(self.pub.read_bytes()),
                 "--expected-capture-helper-sha256", sha(b"synthetic helper"),
                 "--reviewer-pseudonym", "synthetic-reader",
                 "--created-at-utc", "2026-01-01T00:01:00Z",
                 "--attest-independent-of-claim-generation-scoring-custody",
                 "--attest-personally-compared-every-pair",
                 "--output-dir", str(output)]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(argv), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["reviewer_key_trust_status"],
                         "root_signed_frame_reviewer_role_key_match_root_pin_externality_unverified")
        self.assertEqual(report["capture_helper_pin_status"],
                         "caller_pinned_hash_match_not_joint_freeze_authenticated")
        self.assertEqual(report["capture_operator_signature_status"],
                         "rostered_key_signature_valid_exact_artifact_hashes_recomputed")
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(output / "clock_frame_evidence_manifest.json").st_mode & 0o777, 0o600)


class ArtifactDirectoryRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.parent = self.root / "private-parent"
        self.parent.mkdir(mode=0o700)
        self.directory = self.parent / "artifacts"
        self.directory.mkdir(mode=0o700)
        self.leaf = self.directory / "shot.png"
        self.leaf.write_bytes(b"synthetic screenshot bytes")
        self.leaf.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_parent_path_swap_to_symlink_is_rejected(self):
        original_listdir = os.listdir
        moved = self.root / "moved-parent"
        calls = 0

        def swap_after_snapshot(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.parent.rename(moved)
                self.parent.symlink_to(moved, target_is_directory=True)
            return original_listdir(fd)

        with patch("build_nextgqa_clock_evidence_manifest.os.listdir", side_effect=swap_after_snapshot):
            with self.assertRaisesRegex(ValueError, "screenshots_directory_path_changed_during_read"):
                _artifact_directory(self.directory, {"shot.png"}, "screenshots")

    def test_leaf_swap_to_symlink_is_rejected(self):
        original_listdir = os.listdir
        target = self.root / "replacement.png"
        target.write_bytes(b"different synthetic bytes")
        target.chmod(0o600)
        calls = 0

        def swap_after_snapshot(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.leaf.unlink()
                self.leaf.symlink_to(target)
            return original_listdir(fd)

        with patch("build_nextgqa_clock_evidence_manifest.os.listdir", side_effect=swap_after_snapshot):
            with self.assertRaisesRegex(ValueError, "screenshots_artifact_path_changed_during_read"):
                _artifact_directory(self.directory, {"shot.png"}, "screenshots")

    def test_static_leaf_symlink_is_rejected(self):
        target = self.root / "replacement.png"
        target.write_bytes(b"synthetic bytes")
        target.chmod(0o600)
        self.leaf.unlink()
        self.leaf.symlink_to(target)
        with self.assertRaises(OSError):
            _artifact_directory(self.directory, {"shot.png"}, "screenshots")

    def test_unexpected_directory_member_is_rejected(self):
        extra = self.directory / "extra.png"
        extra.write_bytes(b"extra synthetic bytes")
        extra.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "screenshots_artifact_set_invalid"):
            _artifact_directory(self.directory, {"shot.png"}, "screenshots")


class BuilderAuditIntegrationTests(unittest.TestCase):
    """The downstream frame auditor verifies v2 operator provenance end-to-end."""

    def test_builder_output_passes_real_audit_without_authorizing(self):
        fixture = ClockFrameEvidenceAuditTests()
        fixture.setUp()
        operator = CaptureAttestationTests()
        operator.setUp()
        try:
            root = fixture.root
            root.chmod(0o700)
            crop_dir = root / "builder-crops"
            crop_dir.mkdir(mode=0o700)
            crop_rows, decoder_rows = [], []
            pair_image_bytes = {}
            identity_rows, review_rows, pairs = [], [], []
            rows_by_key = {(r["review_token"], r["anchor_id"]): r for r in fixture.evidence}
            plan_media = {r["review_token"]: r["source_media_sha256"] for r in fixture.plan_rows}
            plan_sha = sha(fixture.plan_raw)
            provenance_sha = sha(fixture.provenance_raw)
            receipt_sha = sha(fixture.capture_receipt_raw)

            for index, ((token, anchor), source) in enumerate(sorted(rows_by_key.items())):
                screenshot_path = fixture.screenshot_dir / source["screenshot_file"]
                with Image.open(screenshot_path) as screenshot:
                    crop_image = screenshot.crop((0, 0, 32, 24)).convert("RGB")
                crop_buffer = io.BytesIO()
                crop_image.save(crop_buffer, format="PNG", optimize=False)
                crop_raw = crop_buffer.getvalue()
                crop_file = f"{token}_{anchor}_crop.png"
                (crop_dir / crop_file).write_bytes(crop_raw)
                (crop_dir / crop_file).chmod(0o600)
                crop_sha = sha(crop_raw)

                decoder_buffer = io.BytesIO()
                fixture.frames[token][source["decoder_frame_index"]].save(
                    decoder_buffer, format="PNG", optimize=False)
                decoder_png = decoder_buffer.getvalue()
                decoder_png_sha = sha(decoder_png)
                pair_image_bytes[(token, anchor)] = {
                    "screenshot_crop": crop_raw, "decoder_frame": decoder_png}
                pair_id = f"pair{index:020d}"
                crop_on_a = index % 2 == 0
                pairs.append({"pair_id": pair_id,
                              "A_sha256": crop_sha if crop_on_a else decoder_png_sha,
                              "B_sha256": decoder_png_sha if crop_on_a else crop_sha})
                identity_rows.append({"pair_id": pair_id, "review_token": token,
                                      "anchor_id": anchor,
                                      "A_source": "screenshot_crop" if crop_on_a else "decoder_frame",
                                      "B_source": "decoder_frame" if crop_on_a else "screenshot_crop"})
                review_rows.append({"pair_id": pair_id, "crop_purity_attestation": True,
                                    "decision": "match"})
                crop_rows.append({"review_token": token, "anchor_id": anchor,
                                  "screenshot_file": source["screenshot_file"],
                                  "screenshot_sha256": source["screenshot_sha256"],
                                  "bbox_xywh": [0, 0, 32, 24], "crop_file": crop_file,
                                  "crop_sha256": crop_sha,
                                  "crop_rgb_dimensions": [32, 24],
                                  "crop_purity": "human_review_required"})
                decoder_rows.append({"review_token": token, "anchor_id": anchor,
                                     "source_media_sha256": plan_media[token],
                                     "decoder_frame_index": source["decoder_frame_index"],
                                     "decoder_pts": source["decoder_pts_seconds"],
                                     "decoder_relative_pts_seconds": source["decoder_pts_seconds"],
                                     "rgb_sha256": source["decoder_rgb_sha256"],
                                     "png_sha256": decoder_png_sha,
                                     "png_file": f"{token}_{anchor}.png"})

            package = {"schema_version": 1, "purpose": "private_blinded_frame_pair_review_package",
                       "status": "not_authorizing", "execution_authorization": "not_granted",
                       "pair_count": 15, "package_nonce": "b" * 64,
                       "plan_sha256": plan_sha, "pairs": pairs}
            package_raw = enc(package)
            viewer_pairs = []
            for item in identity_rows:
                images = pair_image_bytes[(item["review_token"], item["anchor_id"])]
                viewer_pairs.append({
                    "id": item["pair_id"],
                    "A": "data:image/png;base64," + base64.b64encode(
                        images[item["A_source"]]).decode(),
                    "B": "data:image/png;base64," + base64.b64encode(
                        images[item["B_source"]]).decode(),
                })
            package["reviewer_html_template_sha256"] = sha(
                _html("__PACKAGE_SHA__", "__PACKAGE_NONCE__", viewer_pairs))
            package_raw = enc(package)
            reviewer_html_raw = _html(sha(package_raw), package["package_nonce"], viewer_pairs)
            review = {"schema_version": 1,
                      "purpose": "private_blinded_frame_pair_review_export",
                      "status": "not_authorizing", "execution_authorization": "not_granted",
                      "package_manifest_sha256": sha(package_raw),
                      "package_nonce": package["package_nonce"],
                      "reviewer_pseudonym": "synthetic-independent-reviewer",
                      "reviewer_role": "independent_frame_pair_reviewer",
                      "independence_attestation": "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring",
                      "reviewer_provenance_status": "self_reported_unauthenticated",
                      "signature_status": "none", "reviewed_at_utc": "2026-10-17T06:01:00Z",
                      "records": review_rows}
            crop_map_raw = enc({"schema_version": 1,
                                "purpose": "synthetic_crop_map_for_v2_audit",
                                "records": crop_rows})
            crop_crosswalk_raw = enc({"schema_version": 1,
                    "purpose": "private_candidate_screenshot_crop_crosswalk",
                    "status": "candidate_only", "execution_authorization": "not_granted",
                    "plan_sha256": plan_sha, "plan_provenance_sha256": provenance_sha,
                    "capture_receipt_sha256": receipt_sha,
                    "crop_map_sha256": sha(crop_map_raw),
                    "bbox_video_only_purity": "not_established_human_review_required",
                    "records": crop_rows})
            file_map = {
                "plan": fixture.plan_raw,
                "provenance": fixture.provenance_raw,
                "observations": fixture.observations_raw,
                "observation_index": fixture.index_raw,
                "capture_receipt": fixture.capture_receipt_raw,
                "crop_crosswalk": crop_crosswalk_raw,
                "decoder_crosswalk": enc({"schema_version": 1,
                    "purpose": "private_candidate_decoder_anchor_png_crosswalk",
                    "status": "candidate_artifact_export_only", "execution_authorization": "not_granted",
                    "plan_sha256": plan_sha, "provenance_sha256": provenance_sha,
                    "requested_anchors": 15, "effective_anchors": 15, "records": decoder_rows}),
                "pair_package": package_raw,
                "pair_reviewer_html": reviewer_html_raw,
                "identity_map": enc({"schema_version": 1,
                    "purpose": "private_blinded_frame_pair_identity_map", "status": "custodian_only",
                    "package_manifest_sha256": sha(package_raw), "records": identity_rows}),
                "review_export": enc(review),
                "crop_map": crop_map_raw,
            }
            operator_inputs = {
                "plan_raw": fixture.plan_raw,
                "expected_plan_sha256": plan_sha,
                "plan_provenance_raw": fixture.provenance_raw,
                "expected_plan_provenance_sha256": provenance_sha,
                "observations_raw": fixture.observations_raw,
                "observation_index_raw": fixture.index_raw,
                "capture_receipt_raw": fixture.capture_receipt_raw,
                "crop_map_raw": crop_map_raw,
                "crop_crosswalk_raw": crop_crosswalk_raw,
                "screenshot_files": {p.name: p.read_bytes()
                                      for p in fixture.screenshot_dir.iterdir()},
                "crop_files": {p.name: p.read_bytes() for p in crop_dir.iterdir()},
                "integrity_roster_raw": operator.integrity,
                "release_roster_raw": operator.release,
                "root_public_key_pem": operator.keys["root"][1],
                "expected_root_public_der_sha256": operator.root_pin,
            }
            file_map.update({
                "capture_operator_attestation": create_attestation(
                    reviewer_private_key_pem=operator.keys["capture"][0],
                    reviewer_public_key_pem=operator.keys["capture"][1],
                    **operator_inputs),
                "integrity_roster": operator.integrity,
                "release_roster": operator.release,
            })
            fixture.key_path.chmod(0o600)
            fixture.public_key_path.chmod(0o600)
            fixture.key_path.write_bytes(operator.keys["frame"][0])
            fixture.public_key_path.write_bytes(operator.keys["frame"][1])
            fixture.key_path.chmod(0o600)
            fixture.public_key_path.chmod(0o600)
            fixture.public_key_raw = fixture.public_key_path.read_bytes()
            evidence = build_manifest(
                input_files=file_map, crop_dir=crop_dir,
                screenshot_dir=fixture.screenshot_dir,
                private_key_path=fixture.key_path, public_key_path=fixture.public_key_path,
                expected_public_key_sha256=sha(fixture.public_key_raw),
                expected_capture_helper_sha256=fixture.capture_helper_sha256,
                root_public_key_raw=operator.keys["root"][1],
                expected_root_public_der_sha256=operator.root_pin,
                reviewer_pseudonym="synthetic-independent-reviewer",
                created_at_utc="2026-10-17T06:01:00Z",
                attest_independence=True, attest_visual_comparison=True)
            evidence_raw = enc(evidence)
            audit_kwargs = dict(crop_dir=crop_dir,
                capture_operator_attestation_raw=file_map["capture_operator_attestation"],
                crop_map_raw=crop_map_raw, crop_crosswalk_raw=crop_crosswalk_raw,
                integrity_roster_raw=operator.integrity,
                release_roster_raw=operator.release,
                root_public_key_raw=operator.keys["root"][1],
                expected_root_public_der_sha256=operator.root_pin,
                pair_package_raw=file_map["pair_package"],
                pair_identity_map_raw=file_map["identity_map"],
                pair_review_export_raw=file_map["review_export"],
                pair_reviewer_html_raw=file_map["pair_reviewer_html"])
            result = fixture._audit(evidence_raw, **audit_kwargs)
            self.assertEqual(result["status"], "candidate_signed_evidence_bytes_consistent")
            self.assertEqual(result["verified_anchors"], 15)
            self.assertEqual(result["execution_authorization"], "not_granted")
            self.assertEqual(result["joint_freeze_status"], "not_jointly_frozen")
            self.assertTrue(result["capture_operator_signature_verified"])
            self.assertTrue(result["frame_reviewer_role_key_verified"])
            self.assertEqual(
                result["reviewer_key_trust_status"],
                "root_signed_frame_reviewer_role_key_match_root_pin_externality_unverified")
            self.assertFalse(result["capture_operator_human_identity_verified"])
            for artifact_name, audit_name in (
                    ("pair_package", "pair_package_raw"),
                    ("identity_map", "pair_identity_map_raw"),
                    ("review_export", "pair_review_export_raw"),
                    ("pair_reviewer_html", "pair_reviewer_html_raw")):
                with self.subTest(artifact=artifact_name):
                    tampered = dict(audit_kwargs)
                    tampered[audit_name] += b"\n"
                    with self.assertRaisesRegex(ValueError, "pair_review_artifact_hash_mismatch"):
                        fixture._audit(evidence_raw, **tampered)
            bad_identity = json.loads(file_map["identity_map"])
            bad_identity["unexpected"] = True
            bad_identity_raw = enc(bad_identity)
            resigned = json.loads(evidence_raw)
            resigned["pair_identity_map_sha256"] = sha(bad_identity_raw)
            payload = evidence_canonical({key: value for key, value in resigned.items()
                                         if key != "signature_ed25519_base64"})
            signature = _sign(EVIDENCE_DOMAIN_V3 + payload, fixture.key_path,
                              _load_public_der(fixture.public_key_path))
            resigned["signature_ed25519_base64"] = base64.b64encode(signature).decode()
            with self.assertRaisesRegex(ValueError, "pair_identity_map_schema_invalid"):
                fixture._audit(enc(resigned), **{**audit_kwargs,
                    "pair_identity_map_raw": bad_identity_raw})
            broken_attestation = json.loads(file_map["capture_operator_attestation"])
            broken_attestation["signature_ed25519_base64"] = "A" * 88
            with self.assertRaisesRegex(ValueError, "capture_operator_signature_invalid"):
                fixture._audit(evidence_raw,
                    **{**audit_kwargs,
                       "capture_operator_attestation_raw": enc(broken_attestation)})
            with self.assertRaisesRegex(ValueError, "capture_operator_audit_inputs_missing"):
                fixture._audit(evidence_raw, crop_dir=crop_dir)
            downgraded = json.loads(evidence_raw)
            downgraded["schema_version"] = 1
            with self.assertRaisesRegex(ValueError, "evidence_manifest_schema_or_binding_invalid"):
                fixture._audit(enc(downgraded), **audit_kwargs)
        finally:
            operator.tearDown()
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
