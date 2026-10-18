from __future__ import annotations

import base64
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import av
from PIL import Image, ImageDraw

from audit_nextgqa_clock_frame_evidence import (
    _canonical, _rgb_digest, _verify_signature, audit_evidence,
)
from claim_mask_video_decoder import _stream_and_decode


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def enc(value) -> bytes:
    return _canonical(value)


class ClockFrameEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.media_dir = self.root / "media"
        self.screenshot_dir = self.root / "screenshots"
        self.media_dir.mkdir()
        self.screenshot_dir.mkdir()
        self.media_dir.chmod(0o700)
        self.screenshot_dir.chmod(0o700)
        self.tokens = [f"token_{i:018d}" for i in range(3)]
        self.media = {}
        self.plan_rows = []
        self.observation_rows = []
        self.frames = {}
        for video_index, token in enumerate(self.tokens):
            media_raw = self._make_video(video_index)
            media_path = self.media_dir / (token + ".mp4")
            media_path.write_bytes(media_raw)
            media_path.chmod(0o600)
            self.media[token] = media_raw
            pts, _time_base, relative_times, _pixel_spec, pixel_hashes, _info = \
                _stream_and_decode(io.BytesIO(media_raw))
            frames = self._decoded_rgb(media_raw)
            self.frames[token] = frames
            anchor_ids = ["fit_begin", "heldout_quarter", "fit_midpoint",
                          "heldout_three_quarter", "fit_end"]
            anchors = [
                {"anchor_id": anchor_id, "decoder_pts_seconds": str(t)}
                for anchor_id, t in zip(anchor_ids, relative_times, strict=True)
            ]
            self.plan_rows.append({
                "review_token": token,
                "source_media_sha256": sha(media_raw),
                "fit_anchors": [anchors[0], anchors[2], anchors[4]],
                "heldout_anchors": [anchors[1], anchors[3]],
            })
            events = []
            for i, anchor in enumerate(anchors):
                media_time = float(relative_times[i])
                callback_now = 1000.0 + i
                events.append({
                    "anchor_id": anchor["anchor_id"], "sequence": i + 1,
                    "frame_match_status": "fresh_click_bound_callback_unverified_decoder",
                    "captured_at_utc": "2026-10-17T06:00:00.000Z",
                    "requested_player_time_seconds": media_time,
                    "request_started_performance_ms": callback_now - 10,
                    "html5_current_time_seconds": media_time,
                    "request_video_frame_callback_metadata": {
                        "media_time_seconds": media_time, "presented_frames": i + 1,
                        "presentation_time_ms": callback_now - 2,
                        "expected_display_time_ms": callback_now - 1,
                        "width": 32, "height": 24, "callback_now_ms": callback_now,
                    },
                    "performance_now_ms": callback_now + 5,
                    "media_duration_seconds": 4.0, "playback_rate": 1.0,
                    "muted": True, "volume": 0.0, "text_tracks_disabled": True,
                    "paused": True, "seeking": False, "ready_state": 4,
                    "current_src_is_staged_media": True, "visibility_state": "visible",
                })
            self.observation_rows.append({"review_token": token, "events": events})

        self.plan_raw = enc({
            "schema_version": 1, "purpose": "predeclared_html5_clock_pts_anchor_plan",
            "status": "predeclared_frozen", "execution_authorization": "not_granted",
            "records": self.plan_rows,
        })
        self.provenance_raw = enc({
            "schema_version": 1,
            "purpose": "private_clock_anchor_plan_candidate_provenance",
            "status": "candidate_not_jointly_frozen", "execution_authorization": "not_granted",
            "input_sha256": {key: sha(key.encode()) for key in
                              ("worksheet", "lineage_manifest", "custody_manifest")},
            "source_sha256": {key: sha(key.encode()) for key in
                               ("builder", "decoder", "custody_auditor", "private_output_policy")},
            "plan_sha256": sha(self.plan_raw), "joint_freeze_receipt_sha256": None,
            "created_at_utc": "2026-10-17T06:00:00Z",
            "python_version": "3.12.0", "pyav_version": av.__version__,
            "ffmpeg_library_versions": {"libavcodec": [62, 28, 102]},
            "records": [{
                "review_token": token, "staged_media_sha256": sha(self.media[token]),
                "decoded_frame_count": 5, "decoded_pts_sequence_sha256": sha(b"pts-sequence"),
                "time_base": [1, 1], "pixel_spec": [32, 24, "yuv420p"],
                "decoder_stream_info": {
                    "stream_index": 0, "codec_name": "mpeg4", "stream_time_base": "1/16384",
                    "decoder_thread_count": 1, "container_format": "mp4",
                    "reported_stream_frame_count": 5,
                },
            } for token in self.tokens],
        })
        self.observations_raw = enc({
            "schema_version": 3, "status": "raw_player_clock_observations_only",
            "execution_authorization": "not_granted", "decoder_frame_identity": "not_verified",
            "purpose": "local_unvalidated_html5_player_clock_observations",
            "capture_policy_id": "rvfc_click_bound_v1",
            "capture_thresholds": {
                "max_callback_age_ms": 250.0, "max_request_delay_ms": 250.0,
                "max_click_player_time_delta_seconds": 0.1,
                "max_callback_player_time_delta_seconds": 0.1,
                "callback_wait_timeout_ms": 1000.0,
            },
            "player_clock": "html5_video_currentTime_seconds",
            "presented_frame_metadata": "requestVideoFrameCallback_when_available_not_decoder_matched",
            "created_at_utc": "2026-10-17T06:00:00Z",
            "browser_user_agent": "synthetic-test-browser",
            "anchor_id_validation": "format_and_per_video_uniqueness_only_unverified",
            "predeclared_plan_binding": "not_checked", "fit_heldout_split": "not_assigned",
            "decoder_pts_pairing": "not_provided",
            "records": self.observation_rows,
        })
        self.index_raw = enc({
            "schema_version": 1,
            "purpose": "unvalidated_html5_clock_observation_integrity_index",
            "payload_sha256": sha(self.observations_raw), "status": "raw_observations_only",
            "execution_authorization": "not_granted",
            "created_at_utc": "2026-10-17T06:00:00Z",
        })
        self.key_path = self.root / "reviewer-private.pem"
        self.public_key_path = self.root / "reviewer-public.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out",
                        str(self.key_path)], check=True, capture_output=True)
        subprocess.run(["openssl", "pkey", "-in", str(self.key_path), "-pubout", "-out",
                        str(self.public_key_path)], check=True, capture_output=True)
        self.public_key_raw = self.public_key_path.read_bytes()
        self.capture_helper_sha256 = sha(b"synthetic capture helper source")
        self.evidence = self._make_evidence()
        self.capture_receipt_raw = self._make_capture_receipt(self.evidence)

    def tearDown(self):
        self.temp.cleanup()

    def _make_video(self, video_index):
        output = self.root / f"video-{video_index}.mp4"
        container = av.open(str(output), mode="w", format="mp4")
        stream = container.add_stream("mpeg4", rate=1)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for frame_index in range(5):
            image = Image.new("RGB", (32, 24),
                              (20 + video_index * 30, 30 + frame_index * 25, 120))
            draw = ImageDraw.Draw(image)
            draw.rectangle((frame_index * 3, 2, frame_index * 3 + 5, 10),
                           fill=(220, 40 + video_index, 10 + frame_index))
            frame = av.VideoFrame.from_image(image)
            frame.pts = frame_index
            frame.time_base = Fraction(1, 1)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return output.read_bytes()

    def _decoded_rgb(self, media_raw):
        with av.open(io.BytesIO(media_raw), mode="r") as container:
            stream = next(iter(container.streams.video))
            stream.codec_context.thread_count = 1
            return [frame.to_image().convert("RGB") for frame in container.decode(stream)]

    def _make_evidence(self):
        rows = []
        for token, plan_row, observation_row in zip(
                self.tokens, self.plan_rows, self.observation_rows, strict=True):
            anchors = plan_row["fit_anchors"] + plan_row["heldout_anchors"]
            for anchor in anchors:
                frame_index = ["fit_begin", "heldout_quarter", "fit_midpoint",
                               "heldout_three_quarter", "fit_end"].index(anchor["anchor_id"])
                event = next(e for e in observation_row["events"]
                             if e["anchor_id"] == anchor["anchor_id"])
                filename = f"{token}_{anchor['anchor_id']}.png"
                screenshot_path = self.screenshot_dir / filename
                full_window = Image.new("RGB", (320, 240), (5, 7, 9))
                full_window.paste(self.frames[token][frame_index], (0, 0))
                full_window.save(screenshot_path, format="PNG")
                screenshot_path.chmod(0o600)
                screenshot_raw = screenshot_path.read_bytes()
                with Image.open(io.BytesIO(screenshot_raw)) as image:
                    crop_digest = _rgb_digest(image.crop((0, 0, 32, 24)))
                media_info = _stream_and_decode(io.BytesIO(self.media[token]))
                rows.append({
                    "review_token": token, "anchor_id": anchor["anchor_id"],
                    "capture_event_sha256": sha(enc(event)),
                    "screenshot_file": filename, "screenshot_sha256": sha(screenshot_raw),
                    "screenshot_captured_at_utc": "2026-10-17T06:00:00.050Z",
                    "viewport_bbox_xywh": [0, 0, 32, 24],
                    "screenshot_crop_rgb_sha256": crop_digest,
                    "decoder_frame_index": frame_index,
                    "decoder_pts_seconds": anchor["decoder_pts_seconds"],
                    "decoder_rgb_sha256": media_info[4][frame_index],
                    "correspondence_decision": "match",
                })
        return rows

    def _make_capture_receipt(self, rows):
        capture_records = [{
            "review_token": row["review_token"], "anchor_id": row["anchor_id"],
            "screenshot_file": row["screenshot_file"],
            "screenshot_sha256": row["screenshot_sha256"],
            "screenshot_captured_at_utc": row["screenshot_captured_at_utc"],
            "capture_method": "windows_gdi_copyfromscreen_foreground_window_v1",
            "foreground_process": "msedge", "foreground_pid": 9999,
            "foreground_window_title": "Private offline video review — Microsoft Edge",
            "foreground_window_rect_xywh": [0, 0, 320, 240],
            "virtual_screen_rect_xywh": [0, 0, 1920, 1080],
            "hotkey_message_tick_ms": 1000,
            "capture_complete_tick_ms": 1012,
            "hotkey_to_capture_ms": 12,
        } for row in rows]
        return enc({
            "schema_version": 1,
            "purpose": "private_windows_compositor_screenshot_capture_receipt_candidate",
            "status": "self_reported_capture_receipt_not_independently_authenticated",
            "execution_authorization": "not_granted",
            "plan_sha256": sha(self.plan_raw),
            "plan_provenance_sha256": sha(self.provenance_raw),
            "capture_method": "windows_gdi_copyfromscreen_foreground_window_v1",
            "capture_helper_file_sha256": self.capture_helper_sha256,
            "created_at_utc": "2026-10-17T06:02:00Z",
            "records": capture_records,
        })

    def _signed_evidence_raw(self, rows=None):
        signed = {
            "schema_version": 1,
            "purpose": "private_clock_frame_correspondence_reader_attestation",
            "plan_sha256": sha(self.plan_raw),
            "plan_provenance_sha256": sha(self.provenance_raw),
            "observations_sha256": sha(self.observations_raw),
            "observation_index_sha256": sha(self.index_raw),
            "capture_receipt_sha256": sha(self.capture_receipt_raw),
            "reviewer_public_key_sha256": sha(self.public_key_raw),
            "reviewer_pseudonym": "synthetic-independent-reviewer",
            "reviewer_role": "independent_evidence_reader",
            "independence_attestation": "independent_of_claim_generation_scoring_and_custody",
            "visual_inspection_attestation":
                "personally_compared_each_screenshot_crop_to_decoder_frame",
            "created_at_utc": "2026-10-17T06:01:00Z",
            "records": self.evidence if rows is None else rows,
        }
        payload_path, signature_path = self.root / "signed-payload", self.root / "signature"
        payload_path.write_bytes(enc(signed))
        subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(self.key_path),
                        "-rawin", "-in", str(payload_path), "-out", str(signature_path)],
                       check=True, capture_output=True)
        signed["signature_ed25519_base64"] = base64.b64encode(signature_path.read_bytes()).decode("ascii")
        return enc(signed)

    def _audit(self, evidence_raw=None, **kwargs):
        write_capture_receipt = kwargs.pop("write_capture_receipt", True)
        if write_capture_receipt:
            capture_receipt_path = self.screenshot_dir / "capture_receipt.json"
            capture_receipt_path.write_bytes(self.capture_receipt_raw)
            capture_receipt_path.chmod(0o600)
        parameters = {
            "expected_plan_sha256": sha(self.plan_raw),
            "expected_plan_provenance_sha256": sha(self.provenance_raw),
            "expected_capture_helper_sha256": self.capture_helper_sha256,
            "expected_reviewer_public_key_sha256": sha(self.public_key_raw),
        }
        parameters.update(kwargs)
        return audit_evidence(
            plan_raw=self.plan_raw, provenance_raw=self.provenance_raw,
            observations_raw=self.observations_raw,
            observation_index_raw=self.index_raw,
            capture_receipt_raw=self.capture_receipt_raw,
            evidence_raw=self._signed_evidence_raw() if evidence_raw is None else evidence_raw,
            public_key_raw=self.public_key_raw,
            media_dir=self.media_dir, screenshot_dir=self.screenshot_dir,
            **parameters,
        )

    def test_exact_signed_three_by_five_byte_joins_pass_without_authorizing(self):
        result = self._audit()
        self.assertEqual(result["status"], "legacy_v1_unrostered_key_candidate")
        self.assertEqual((result["requested_records"], result["requested_anchors"]), (3, 15))
        self.assertEqual(result["verified_anchors"], 15)
        self.assertTrue(result["reviewer_signature_verified"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["joint_freeze_status"], "not_jointly_frozen")

    def test_ambiguous_or_mismatch_decision_blocks(self):
        rows = [dict(row) for row in self.evidence]
        rows[0]["correspondence_decision"] = "ambiguous"
        raw = self._signed_evidence_raw(rows)
        result = self._audit(raw)
        self.assertEqual(result["status"], "blocked_correspondence_not_all_match")
        self.assertEqual(result["ambiguous_decisions"], 1)

    def test_reader_assertion_is_not_an_automated_visual_match(self):
        # The utility verifies a signed human assertion and byte joins. It
        # intentionally does not claim that its own code compared the images.
        rows = [dict(row) for row in self.evidence]
        row = rows[0]
        other_frame = self.frames[row["review_token"]][1]
        screenshot_path = self.screenshot_dir / row["screenshot_file"]
        full_window = Image.new("RGB", (320, 240), (5, 7, 9))
        full_window.paste(other_frame, (0, 0))
        full_window.save(screenshot_path, format="PNG")
        screenshot_path.chmod(0o600)
        raw = screenshot_path.read_bytes()
        row["screenshot_sha256"] = sha(raw)
        with Image.open(io.BytesIO(raw)) as image:
            row["screenshot_crop_rgb_sha256"] = _rgb_digest(
                image.crop((0, 0, 32, 24)))
        # A trusted reviewer must not sign this unless they visually verified
        # correspondence. The machine can only verify the signature and joins.
        self.capture_receipt_raw = self._make_capture_receipt(rows)
        result = self._audit(self._signed_evidence_raw(rows))
        self.assertEqual(result["status"], "legacy_v1_unrostered_key_candidate")
        self.assertTrue(any("does not itself inspect" in x for x in result["limitations"]))

    def test_screenshot_outside_candidate_event_skew_fails(self):
        rows = [dict(row) for row in self.evidence]
        rows[0]["screenshot_captured_at_utc"] = "2026-10-17T06:00:01.000Z"
        self.capture_receipt_raw = self._make_capture_receipt(rows)
        with self.assertRaisesRegex(ValueError, "screenshot_capture_event_time_skew"):
            self._audit(self._signed_evidence_raw(rows))

    def test_capture_receipt_helper_pin_and_screenshot_binding_are_required(self):
        with self.assertRaisesRegex(ValueError, "os_capture_receipt_invalid_or_unpinned"):
            self._audit(expected_capture_helper_sha256="0" * 64)
        receipt = json.loads(self.capture_receipt_raw)
        receipt["records"][0]["screenshot_sha256"] = "0" * 64
        self.capture_receipt_raw = enc(receipt)
        with self.assertRaisesRegex(ValueError, "os_capture_screenshot_binding_mismatch"):
            self._audit(self._signed_evidence_raw())

    def test_screenshot_file_must_be_unique_and_bound_to_each_anchor(self):
        rows = [dict(row) for row in self.evidence]
        shared_filename = rows[0]["screenshot_file"]
        for row in rows:
            row["screenshot_file"] = shared_filename
        self.capture_receipt_raw = self._make_capture_receipt(rows)
        with self.assertRaisesRegex(ValueError,
                                    "screenshot_anchor_filename_binding_invalid"):
            self._audit(self._signed_evidence_raw(rows))

    def test_file_replacement_after_snapshot_fails_closed(self):
        import audit_nextgqa_clock_frame_evidence as audit_module
        original = audit_module._snapshot_private_directory

        def snapshot_then_replace(path, expected_names, *, label, additional_names=None):
            values, identities, directory_identity = original(
                path, expected_names, label=label,
                additional_names=additional_names)
            if label == "screenshot":
                filename = sorted(expected_names)[0]
                (path / filename).write_bytes(values[filename] + b"changed-after-snapshot")
                (path / filename).chmod(0o600)
            return values, identities, directory_identity

        with patch.object(audit_module, "_snapshot_private_directory",
                          snapshot_then_replace):
            with self.assertRaisesRegex(ValueError,
                                        "screenshot_path_changed_after_snapshot"):
                self._audit()

    def test_new_directory_entry_after_snapshot_fails_closed(self):
        import audit_nextgqa_clock_frame_evidence as audit_module
        original = audit_module._snapshot_private_directory

        def snapshot_then_add(path, expected_names, *, label, additional_names=None):
            values, identities, directory_identity = original(
                path, expected_names, label=label,
                additional_names=additional_names)
            if label == "screenshot":
                extra = path / "late-extra.png"
                extra.write_bytes(b"late entry")
                extra.chmod(0o600)
            return values, identities, directory_identity

        with patch.object(audit_module, "_snapshot_private_directory", snapshot_then_add):
            with self.assertRaisesRegex(
                    ValueError, "screenshot_directory_contents_changed_after_snapshot"):
                self._audit()

    def test_snapshot_and_decode_resource_limits_fail_before_unbounded_work(self):
        import audit_nextgqa_clock_frame_evidence as audit_module
        with patch.object(audit_module, "MAX_SCREENSHOT_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "screenshot_file_size_exceeds_limit"):
                self._audit()
        with patch.object(audit_module, "MAX_STAGED_VIDEO_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "staged_media_file_size_exceeds_limit"):
                self._audit()
        with patch.object(audit_module, "MAX_SNAPSHOT_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "screenshot_snapshot_size_exceeds_limit"):
                self._audit()
        with patch.object(audit_module, "MAX_SCREENSHOT_PIXELS", 1):
            with self.assertRaisesRegex(ValueError, "screenshot_dimensions_exceed_limit"):
                self._audit()
        with patch.object(audit_module, "MAX_DECODED_VIDEO_FRAMES", 1):
            with self.assertRaisesRegex(ValueError, "decoded_video_frame_limit_exceeded"):
                self._audit()

    def test_capture_receipt_bytes_must_match_copy_in_screenshot_directory(self):
        path = self.screenshot_dir / "capture_receipt.json"
        path.write_bytes(b"different receipt bytes")
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "capture_receipt_file_byte_mismatch"):
            self._audit(write_capture_receipt=False)

    def test_bounded_reader_rejects_early_eof_and_growth_during_read(self):
        import audit_nextgqa_clock_frame_evidence as audit_module
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "private"
            directory.mkdir(mode=0o700)
            path = directory / "item.bin"
            path.write_bytes(b"four")
            path.chmod(0o600)
            real_read = os.read

            def early_eof(fd, size):
                if size == 4:
                    return b""
                return real_read(fd, size)

            with patch.object(audit_module.os, "read", side_effect=early_eof):
                with self.assertRaisesRegex(ValueError, "screenshot_file_changed_during_read"):
                    audit_module._snapshot_private_directory(
                        directory, {"item.bin"}, label="screenshot")

            def injected_growth(fd, size):
                if size == 1:
                    return b"x"
                return real_read(fd, size)

            with patch.object(audit_module.os, "read", side_effect=injected_growth):
                with self.assertRaisesRegex(ValueError, "screenshot_file_changed_during_read"):
                    audit_module._snapshot_private_directory(
                        directory, {"item.bin"}, label="screenshot")

    def test_capture_receipt_timing_is_independently_validated(self):
        receipt = json.loads(self.capture_receipt_raw)
        receipt["records"][0]["hotkey_to_capture_ms"] = 251
        receipt["records"][0]["capture_complete_tick_ms"] = 1251
        self.capture_receipt_raw = enc(receipt)
        with self.assertRaisesRegex(ValueError, "os_capture_timing_declaration_invalid"):
            self._audit()

    def test_capture_receipt_timing_handles_uint32_tick_wrap(self):
        receipt = json.loads(self.capture_receipt_raw)
        receipt["records"][0]["hotkey_message_tick_ms"] = 0xFFFFFFF8
        receipt["records"][0]["capture_complete_tick_ms"] = 4
        receipt["records"][0]["hotkey_to_capture_ms"] = 12
        self.capture_receipt_raw = enc(receipt)
        self.assertEqual(self._audit()["status"],
                         "legacy_v1_unrostered_key_candidate")

    def test_missing_anchor_and_duplicate_anchor_block(self):
        for rows in (self.evidence[:-1], self.evidence + [self.evidence[0]]):
            with self.subTest(count=len(rows)), self.assertRaisesRegex(
                    ValueError, "evidence_anchor_count_invalid"):
                self._audit(self._signed_evidence_raw(rows))

    def test_tampered_screenshot_and_changed_media_fail(self):
        row = self.evidence[0]
        screenshot_path = self.screenshot_dir / row["screenshot_file"]
        original = screenshot_path.read_bytes()
        screenshot_path.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "screenshot_hash_mismatch"):
            self._audit()
        screenshot_path.write_bytes(original)
        token = row["review_token"]
        (self.media_dir / (token + ".mp4")).write_bytes(self.media[token] + b"tamper")
        with self.assertRaisesRegex(ValueError, "staged_media_hash_mismatch"):
            self._audit()

    def test_signature_and_trust_key_mismatch_fail(self):
        raw = bytearray(self._signed_evidence_raw())
        raw[-10] = ord("A") if raw[-10] != ord("A") else ord("B")
        with self.assertRaisesRegex(ValueError, "reviewer_signature_invalid|evidence_manifest_schema_or_binding_invalid"):
            self._audit(bytes(raw))
        with self.assertRaisesRegex(ValueError, "reviewer_public_key_hash_mismatch"):
            self._audit(expected_reviewer_public_key_sha256="0" * 64)

    def test_non_ed25519_public_key_is_rejected(self):
        rsa_private = self.root / "rsa-private.pem"
        rsa_public = self.root / "rsa-public.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt",
                        "rsa_keygen_bits:1024", "-out", str(rsa_private)],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "pkey", "-in", str(rsa_private), "-pubout", "-out",
                        str(rsa_public)], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "reviewer_public_key_not_ed25519"):
            _verify_signature(rsa_public.read_bytes(), b"test", base64.b64encode(b"x" * 64).decode())

    def test_plan_pts_or_crop_binding_mismatch_fail(self):
        rows = [dict(row) for row in self.evidence]
        rows[0]["decoder_pts_seconds"] = "99"
        with self.assertRaisesRegex(ValueError, "decoder_frame_pts_plan_mismatch"):
            self._audit(self._signed_evidence_raw(rows))
        rows = [dict(row) for row in self.evidence]
        rows[0]["screenshot_crop_rgb_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "screenshot_crop_hash_mismatch"):
            self._audit(self._signed_evidence_raw(rows))


if __name__ == "__main__":
    unittest.main()
