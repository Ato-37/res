import json
import hashlib
import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path

from build_nextgqa_claim_mapping_packet import apply_review_policy, build
from render_nextgqa_claim_review_html import main, render
from test_build_nextgqa_claim_mapping_packet import encode, packet


class RenderNextgqaClaimReviewHtmlTests(unittest.TestCase):
    def payload(self):
        source, source_raw = packet(3)
        for i, row in enumerate(source["records"]):
            row["generated_parsed"]["claim"] = f"Safe claim example {i}."
        source_raw = encode(source)
        built, _ = build(source, source_raw)
        policy_raw = (Path(__file__).resolve().parents[1] / "docs" /
                      "nextgqa_claim_review_policy_v5.json").read_bytes()
        built = apply_review_policy(built, json.loads(policy_raw), policy_raw)
        records = []
        for i, row in enumerate(built["records"]):
            relative = f"group/video-{i}.mp4"
            records.append({
                "video_id": row["video_id"], "claim_id": row["claim_id"],
                "dataset_question_id": f"qid-{i}", "dataset_video_id": f"vid-{i}",
                "video_key": f"group/video-{i}", "relative_video_path": relative,
                "video_sha256": row["video_sha256"],
                "question_sha256": __import__("hashlib").sha256(row["question"].encode()).hexdigest(),
                "claim_text_sha256": row["claim_text_sha256"],
            })
        lineage = {
            "schema_version": 1,
            "purpose": "private_source_lineage_crosscheck_for_claim_review",
            "created_by": "audit_nextgqa_claim_packet_lineage.py",
            "source_packet_sha256": built["source_packet_sha256"],
            "cohort_sha256": built["cohort_sha256"],
            "dataset_inputs_sha256": built["dataset_inputs_sha256"],
            "requested_records": 3,
            "effective_records": 3,
            "verification_scope": "synthetic metadata crosscheck",
            "status": "metadata_lineage_crosschecked",
            "records": records,
        }
        lineage_raw = encode(lineage)
        return built, encode(built), lineage, lineage_raw

    @staticmethod
    def custody(payload, worksheet_raw, lineage_raw):
        salt = "ab" * 32
        return {
            "schema_version": 1, "purpose": "private_custodian_token_to_verified_media_manifest",
            "created_at_utc": "2026-10-16T06:00:00Z",
            "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
            "source_lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "claim_review_policy_sha256": payload["claim_review_policy_sha256"], "record_count": 3,
            "records": [{"review_token": row["review_token"], "row_ordinal": i,
                         "media_file": row["review_token"] + ".mp4", "source_media_sha256": row["video_sha256"],
                         "staged_media_sha256": row["video_sha256"], "size_bytes": 1,
                         "media_check_salt": salt, "media_check_sha256": "cd" * 32}
                        for i, row in enumerate(payload["records"], 1)]}

    def render_checked(self, payload, raw, lineage, lineage_raw, expected_lineage, *custody_args):
        if not custody_args:
            custody = self.custody(payload, raw, lineage_raw)
            custody_raw = encode(custody)
            custody_args = (custody, custody_raw, hashlib.sha256(custody_raw).hexdigest())
        return render(payload, raw, lineage, lineage_raw, expected_lineage, *custody_args)

    def test_offline_page_never_authorizes_scoring(self):
        payload, raw, lineage, lineage_raw = self.payload()
        custody = self.custody(payload, raw, lineage_raw)
        custody_raw = encode(custody)
        page = render(payload, raw, lineage, lineage_raw,
                      hashlib.sha256(lineage_raw).hexdigest(), custody, custody_raw,
                      hashlib.sha256(custody_raw).hexdigest()).decode("utf-8")
        self.assertIn("connect-src 'none'", page)
        self.assertIn("media-src blob:", page)
        self.assertIn("all_relevant_occurrences", page)
        self.assertIn("execution_authorization", page)
        self.assertIn("not_granted", page)
        self.assertIn("r.review_token+'.mp4'", page)
        self.assertIn("${r.review_token}.mp4", page)
        self.assertIn("media_check_digest", page)
        self.assertIn("crypto.subtle.digest('SHA-256',joined)", page)
        self.assertIn("const current=++selection", page)
        self.assertIn("metadataSelection===selection", page)
        self.assertIn("video.currentSrc===objectUrl", page)
        self.assertIn("metadataSelection=null", page)
        self.assertIn("video.load()", page)
        self.assertIn("s.intervals=[]", page)
        self.assertIn("custodian照合値", page)
        self.assertIn("window.pendingReviewIndex", page)
        self.assertIn("焼き付いた字幕", page)
        self.assertIn("1 ms精度の測定を意味しません", page)
        self.assertIn("video.muted=true", page)
        self.assertIn("HTML5時計アンカー観測を追加", page)
        self.assertIn("performance_now_ms:performance.now()", page)
        self.assertIn("video.requestVideoFrameCallback", page)
        self.assertIn("review_started_at_utc:s.reviewStartedAt", page)
        self.assertIn("reviewed_at_utc:s.reviewedAt", page)
        self.assertIn("s.reviewStartedAt===null", page)
        self.assertIn("自己申告ログであり", page)
        self.assertIn("presented_frames:meta.presentedFrames", page)
        self.assertIn("media_time_seconds:meta.mediaTime", page)
        self.assertIn("frame_match_status:frame?'last_presented_frame_metadata_unverified':'presented_frame_metadata_missing'", page)
        self.assertIn("!video.paused||video.seeking||video.readyState<2", page)
        self.assertIn("schema_version:2,purpose:'local_unvalidated_html5_player_clock_observations'", page)
        self.assertIn("decoder_frame_identity:'not_verified'", page)
        self.assertIn("muted:video.muted", page)
        self.assertIn("volume:video.volume", page)
        self.assertIn("text_tracks_disabled:Array.from(video.textTracks||[]).every", page)
        self.assertIn("input:not(.clock-control),select,textarea,button:not(.clock-control)", page)
        self.assertIn("current_src_is_staged_media:video.currentSrc===objectUrl", page)
        self.assertIn("fit_heldout_split:'not_assigned'", page)
        self.assertIn("decoder_pts_pairing:'not_provided'", page)
        self.assertIn("predeclared_plan_binding:'not_checked'", page)
        self.assertIn("anchor_id_validation:'format_and_per_video_uniqueness_only_unverified'", page)
        self.assertIn("unvalidated_html5_clock_observation_integrity_index", page)
        self.assertIn("s.clockEvents=[]", page)
        self.assertIn("claim根拠区間と別に保存します", page)
        self.assertIn("currentTimeだけの観測", page)
        self.assertNotIn("/private/video-", page)
        self.assertNotIn("qid-0", page)
        self.assertNotIn("vid-0", page)
        self.assertNotIn("group/video-0", page)
        self.assertNotIn("dataset_question_id", page)
        self.assertNotIn("relative_video_path", page)
        self.assertNotIn("opaque ID", page)
        self.assertNotIn("期待する動画SHA-256", page)
        self.assertNotIn("source_lineage:r.source_lineage", page)
        self.assertNotIn('"video_sha256"', page)
        self.assertNotIn('"video_id"', page)
        self.assertNotIn('"claim_id"', page)
        self.assertNotIn('"claim_text_sha256"', page)
        self.assertNotIn("worksheetSha", page)
        self.assertNotIn("lineageSha", page)

    def test_renderer_fails_closed_without_custody(self):
        payload, raw, lineage, lineage_raw = self.payload()
        with self.assertRaisesRegex(ValueError, "custody_manifest_required"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   None, None, None)

    def test_renderer_rejects_malformed_claim_text(self):
        payload, raw, lineage, lineage_raw = self.payload()
        bad = json.loads(encode(payload))
        bad["records"][0]["claim_text"] = "A readable claim\ncontinued"
        bad["records"][0]["claim_text_sha256"] = hashlib.sha256(
            bad["records"][0]["claim_text"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                                hashlib.sha256(lineage_raw).hexdigest())
        bad["records"][0]["claim_text"] = "A safe claim <script>alert(1)</script>."
        bad["records"][0]["claim_text_sha256"] = hashlib.sha256(
            bad["records"][0]["claim_text"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                                hashlib.sha256(lineage_raw).hexdigest())

    def test_non_frozen_or_incomplete_worksheet_rejected(self):
        payload, raw, lineage, lineage_raw = self.payload()
        bad = dict(payload, execution_authorization="granted")
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = dict(payload, effective_records=2)
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["review"]["review_status"] = "completed"
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

    def test_unknown_sensitive_fields_and_mismatched_claim_hash_are_rejected(self):
        payload, raw, lineage, lineage_raw = self.payload()
        bad = json.loads(encode(payload))
        bad["answer_secret_sentinel"] = "MUST_NOT_APPEAR"
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["new_private_location_sentinel"] = "MUST_NOT_APPEAR"
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

    def test_independently_pinned_lineage_digest_and_safe_relative_path(self):
        payload, raw, lineage, lineage_raw = self.payload()
        with self.assertRaisesRegex(ValueError, "lineage_manifest_sha256_mismatch"):
            self.render_checked(payload, raw, lineage, lineage_raw, "0" * 64)
        bad = json.loads(lineage_raw)
        bad["records"][0]["relative_video_path"] = "../../outside/video.mp4"
        bad["records"][0]["video_key"] = "outside/video"
        bad_raw = __import__("json").dumps(bad, ensure_ascii=False, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "lineage_row_binding_invalid"):
            self.render_checked(payload, raw, bad, bad_raw, hashlib.sha256(bad_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["claim_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            self.render_checked(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

        bad_lineage = json.loads(lineage_raw)
        bad_lineage["records"][0]["relative_video_path"] = r"..\outside\video.mp4"
        bad_lineage["records"][0]["video_key"] = "outside/video"
        bad_lineage_raw = json.dumps(bad_lineage, ensure_ascii=False, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "lineage_row_binding_invalid"):
            self.render_checked(payload, raw, bad_lineage, bad_lineage_raw,
                   hashlib.sha256(bad_lineage_raw).hexdigest())

    def test_cli_requires_private_inputs_and_writes_exclusive_private_page(self):
        payload, worksheet_raw, lineage, lineage_raw = self.payload()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worksheet = root / "worksheet.json"
            manifest = root / "lineage.json"
            custody_path = root / "custody.json"
            output = root / "review.html"
            worksheet.write_bytes(worksheet_raw)
            manifest.write_bytes(lineage_raw)
            custody = self.custody(payload, worksheet_raw, lineage_raw)
            custody_raw = encode(custody)
            custody_path.write_bytes(custody_raw)
            os.chmod(worksheet, 0o600); os.chmod(manifest, 0o600); os.chmod(custody_path, 0o600)
            args = ["--input", str(worksheet), "--lineage-manifest", str(manifest),
                    "--expected-lineage-sha256", hashlib.sha256(lineage_raw).hexdigest(),
                    "--custody-manifest", str(custody_path),
                    "--expected-custody-sha256", hashlib.sha256(custody_raw).hexdigest(),
                    "--output", str(output), "--user-authorized-local-only"]
            self.assertEqual(main(args), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertIn("Safe claim example 0.", output.read_text())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            output.unlink()
            os.chmod(manifest, 0o644)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
