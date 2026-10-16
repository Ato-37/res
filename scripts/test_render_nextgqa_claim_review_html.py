import json
import hashlib
import contextlib
import io
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path

from build_nextgqa_claim_mapping_packet import apply_review_policy, build
from render_nextgqa_claim_review_html import main, render
from build_nextgqa_v2_claim_input_bundle import V2_COHORT_SHA256
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

    def bundle(self, payload, worksheet_raw, lineage, lineage_raw):
        manifest = {
            "schema_version": 1,
            "eligible_dataset_id": "synthetic-nextgqa",
            "eligible_manifest_sha256": "11" * 32,
            "claim_policy_sha256": payload["claim_review_policy_sha256"],
            "occurrence_policy": payload["occurrence_policy"],
            "claims": [],
        }
        sidecar_records = []
        lineage_by_video = {row["video_id"]: row for row in lineage["records"]}
        for row in payload["records"]:
            manifest["claims"].append({
                "video_id": row["video_id"], "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "claim_text_sha256": row["claim_text_sha256"],
                "model_revision": "Qwen/Qwen2.5-VL-7B-Instruct@" + payload["model_revision"],
                "generation_run_sha256": "22" * 32,
            })
            source = lineage_by_video[row["video_id"]]
            sidecar_records.append({
                "video_id": row["video_id"], "dataset_video_id": source["dataset_video_id"],
                "dataset_question_id": source["dataset_question_id"],
                "video_sha256": source["video_sha256"],
                "question_sha256": source["question_sha256"],
                "claim_id": row["claim_id"], "claim_text_sha256": row["claim_text_sha256"],
            })
        manifest_raw = encode(manifest)
        sidecar = {
            "schema_version": 1,
            "purpose": "private_nextgqa_v2_claim_input_bundle_provenance",
            "status": "candidate_unreviewed", "scope": "structural_only",
            "review_completed": False, "scoring_ready": False,
            "execution_authorization": "not_granted",
            "eligible_dataset_id": manifest["eligible_dataset_id"],
            "cohort_sha256": payload["cohort_sha256"],
            "cohort_digest_check": "pinned_value_and_input_declarations_match_not_recomputed_from_full_planner_inputs",
            "occurrence_policy": payload["occurrence_policy"],
            "policy_id": payload["claim_review_policy_id"],
            "policy_sha256": payload["claim_review_policy_sha256"],
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": payload["model_revision"],
            "v1_generation_run_sha256_semantics": "generation_packet_container_byte_digest_not_unique_run_id",
            "worksheet_binding_check": "claim_and_identity_hashes_matched_only_not_review_or_custody_validation",
            "input_sha256": {
                "eligible_manifest": manifest["eligible_manifest_sha256"],
                "generation_packet": "22" * 32,
                "lineage_manifest": hashlib.sha256(lineage_raw).hexdigest(),
                "policy": payload["claim_review_policy_sha256"],
                "test_csv": "33" * 32, "video_map": "44" * 32,
                "worksheet": hashlib.sha256(worksheet_raw).hexdigest(),
            },
            "claim_input_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "records": sidecar_records,
        }
        return manifest, manifest_raw, sidecar, encode(sidecar)

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
        self.assertIn("s.intervals.some(x=>a===x.start&&b===x.end)", page)
        self.assertNotIn("a<=x.end&&b>=x.start", page)
        self.assertIn("別の根拠出現は重なる場合も個別に記録してください", page)
        self.assertIn("s.intervals.sort((x,y)=>x.start-y.start)", page)
        self.assertIn("custodian照合値", page)
        self.assertIn("window.pendingReviewIndex", page)
        self.assertIn("1/2 レビューJSONを保存", page)
        self.assertIn("2/2 完全性indexを保存（必須）", page)
        self.assertIn("レビューJSONを再ダウンロード", page)
        self.assertIn("完全性indexを再ダウンロード", page)
        self.assertIn("ダウンロード開始は保存完了を保証しません", page)
        self.assertIn("非公開のcustodianへ一緒に渡してください", page)
        self.assertIn("画面を閉じるとindexを再生成できない", page)
        self.assertIn("window.pendingReviewPayload=raw", page)
        self.assertIn("downloadReviewIndex()", page)
        self.assertIn("if(exportButton.disabled||exportInProgress)return;exportInProgress=true;exportButton.disabled=true", page)
        self.assertIn("||exportInProgress", page)
        self.assertIn("catch(_error){exportInProgress=false;checkReady()", page)
        self.assertIn("document.getElementById('retry-review').onclick", page)
        self.assertNotIn("localStorage", page)
        self.assertIn("焼き付いた字幕", page)
        self.assertIn("1 ms精度の測定を意味しません", page)
        self.assertIn("video.muted=true", page)
        self.assertIn("HTML5時計アンカー観測を追加", page)
        self.assertIn("performance_now_ms:capturedAt", page)
        self.assertIn("video.requestVideoFrameCallback", page)
        self.assertIn("first_played_at_utc:s.firstPlayedAt", page)
        self.assertIn("const out={schema_version:3,purpose:'anonymous_local_claim_video_human_review'", page)
        self.assertIn("reviewed_at_utc:s.reviewedAt", page)
        self.assertIn("s.firstPlayedAt===null", page)
        self.assertIn("自己申告ログであり", page)
        self.assertIn("presented_frames:meta.presentedFrames", page)
        self.assertIn("media_time_seconds:mediaTime", page)
        self.assertIn("frame_match_status:'fresh_click_bound_callback_unverified_decoder'", page)
        self.assertIn("const capturedAt=performance.now(),age=capturedAt-callbackNow", page)
        self.assertIn("requestGeneration=++clockCaptureGeneration", page)
        self.assertIn("requestGeneration!==clockCaptureGeneration||!clockCapturePending", page)
        self.assertIn("MAX_RVFC_WAIT_MS=1000", page)
        self.assertIn("Math.abs(mediaTime-currentTime)>MAX_PLAYER_TIME_DELTA_SECONDS", page)
        self.assertIn("requestStarted=performance.now()", page)
        self.assertIn("requestDelay=callbackNow-requestStarted", page)
        self.assertIn("requestDelay>MAX_RVFC_REQUEST_DELAY_MS", page)
        self.assertIn("Math.abs(mediaTime-requestedPlayerTime)>MAX_PLAYER_TIME_DELTA_SECONDS", page)
        self.assertIn("video.requestVideoFrameCallback((callbackNow,meta)", page)
        self.assertNotIn("lastPresentedFrame", page)
        self.assertIn("schema_version:3,purpose:'local_unvalidated_html5_player_clock_observations'", page)
        self.assertIn("capture_policy_id:'rvfc_click_bound_v1'", page)
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
        # Keep the reviewer-facing export disclosure synchronized with the fields
        # the v5 renderer actually writes into its JSON payload.
        for disclosure in (
                "schema_version", "human_reviewer_private_id", "browser_user_agent",
                "independence_attestation", "review_token", "interval_rationales",
                "media_duration_seconds", "first_played_at_utc", "display_player",
                "claim_review_policy_sha256", "review_payload_sha256",
                "request_video_frame_callback_metadata", "export_disclosure_acknowledged"):
            self.assertIn(disclosure, page)
        self.assertIn("凍結v5方針文書には", page)
        self.assertIn("レビューJSONのexport_disclosure_acknowledged", page)
        self.assertIn("id=\"export-disclosure\"", page)
        self.assertIn("&&document.getElementById('export-disclosure').checked", page)
        self.assertIn("export_disclosure_acknowledged:document.getElementById('export-disclosure').checked", page)
        disclosure = page.split('<div id="rows">', 1)[0]
        self.assertLess(page.index('レビューJSONのexport_disclosure_acknowledged'),
                        page.index('<div id="rows">'))
        expected_pre_review_fields = {
            # Review JSON top-level, environment, record, and review fields.
            "schema_version", "purpose", "status", "review_completed",
            "execution_authorization", "claim_review_policy_id",
            "claim_review_policy_sha256", "human_reviewer_role",
            "human_reviewer_private_id", "review_environment",
            "browser_user_agent", "offline_static_page",
            "network_fetch_available", "independence_attestation",
            "export_disclosure_acknowledged", "records", "review_token",
            "review", "full_source_video_reviewed",
            "question_and_claim_only_context_shown",
            "gold_and_seed_hidden_until_mapping_frozen",
            "outcome_scores_not_accessed", "media_bytes_verified",
            "occurrence_policy_applied", "support_decision",
            "evidence_intervals_seconds", "interval_rationales",
            "media_duration_seconds", "reviewer_role", "first_played_at_utc",
            "reviewed_at_utc", "review_status", "display_player",
            # Review integrity index.
            "review_payload_sha256", "created_at_utc",
            # Clock payload and event fields.
            "player_clock", "presented_frame_metadata", "decoder_frame_identity",
            "anchor_id_validation", "predeclared_plan_binding",
            "fit_heldout_split", "decoder_pts_pairing", "events",
            "anchor_id", "sequence", "html5_current_time_seconds",
            "request_video_frame_callback_metadata", "frame_match_status",
            "performance_now_ms", "playback_rate", "muted", "volume",
            "text_tracks_disabled", "paused", "seeking", "ready_state",
            "current_src_is_staged_media", "visibility_state",
            # Clock integrity index.
            "payload_sha256",
        }
        self.assertTrue(expected_pre_review_fields.issubset(set(re.findall(
            r"[A-Za-z][A-Za-z0-9_]*", disclosure))),
            "pre-review disclosure omits one or more exported fields")
        self.assertIn("s.clockEvents=[]", page)
        self.assertIn("claim根拠区間と別に保存します", page)
        self.assertIn("decoder PTSの同一性を証明しません", page)
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

    def test_optional_v2_bundle_binds_displayed_claims_and_embeds_hashes(self):
        payload, raw, lineage, lineage_raw = self.payload()
        payload["cohort_sha256"] = V2_COHORT_SHA256
        payload["model_revision"] = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        lineage["cohort_sha256"] = V2_COHORT_SHA256
        raw, lineage_raw = encode(payload), encode(lineage)
        custody = self.custody(payload, raw, lineage_raw)
        custody_raw = encode(custody)
        manifest, manifest_raw, sidecar, sidecar_raw = self.bundle(
            payload, raw, lineage, lineage_raw)
        page = render(
            payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
            custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
            manifest, manifest_raw, sidecar, sidecar_raw,
        ).decode("utf-8")
        self.assertIn(f'name="claim-input-manifest-sha256" content="{hashlib.sha256(manifest_raw).hexdigest()}"', page)
        self.assertIn(f'name="claim-input-bundle-sidecar-sha256" content="{hashlib.sha256(sidecar_raw).hexdigest()}"', page)
        self.assertIn("execution_authorization", page)
        self.assertIn("not_granted", page)

        bad_sidecar = json.loads(sidecar_raw)
        bad_sidecar["input_sha256"]["worksheet"] = "00" * 32
        with self.assertRaisesRegex(ValueError, "claim_bundle_artifact_hash_mismatch"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                   manifest, manifest_raw, bad_sidecar, encode(bad_sidecar))

        for missing_key in ("test_csv", "video_map"):
            bad_sidecar = json.loads(sidecar_raw)
            del bad_sidecar["input_sha256"][missing_key]
            with self.subTest(missing_hash=missing_key):
                with self.assertRaisesRegex(ValueError, "claim_bundle_artifact_hash_mismatch"):
                    render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                           custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                           manifest, manifest_raw, bad_sidecar, encode(bad_sidecar))
        bad_sidecar = json.loads(sidecar_raw)
        bad_sidecar["input_sha256"]["test_csv"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "claim_bundle_artifact_hash_mismatch"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                   manifest, manifest_raw, bad_sidecar, encode(bad_sidecar))

        bad_manifest = json.loads(manifest_raw)
        bad_manifest["claims"][0]["claim_text"] = "different displayed claim"
        bad_manifest_raw = encode(bad_manifest)
        bad_sidecar = json.loads(sidecar_raw)
        bad_sidecar["claim_input_manifest_sha256"] = hashlib.sha256(bad_manifest_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "claim_manifest_worksheet_row_mismatch"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                   bad_manifest, bad_manifest_raw, bad_sidecar, encode(bad_sidecar))

        bad_manifest = json.loads(manifest_raw)
        bad_manifest["eligible_dataset_id"] = None
        bad_manifest_raw = encode(bad_manifest)
        bad_sidecar = json.loads(sidecar_raw)
        bad_sidecar["eligible_dataset_id"] = None
        bad_sidecar["claim_input_manifest_sha256"] = hashlib.sha256(bad_manifest_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "claim_manifest_schema_or_policy_invalid"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                   bad_manifest, bad_manifest_raw, bad_sidecar, encode(bad_sidecar))

    def test_claim_bundle_arguments_must_be_paired(self):
        payload, raw, lineage, lineage_raw = self.payload()
        custody = self.custody(payload, raw, lineage_raw)
        custody_raw = encode(custody)
        with self.assertRaisesRegex(ValueError, "claim_bundle_pair_required"):
            render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                   custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(), {}, b"{}")

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
