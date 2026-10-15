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
            row["generated_parsed"]["claim"] = f"Safe claim {i} <script>alert(1)</script>"
        source_raw = encode(source)
        built, _ = build(source, source_raw)
        policy_raw = (Path(__file__).resolve().parents[1] / "docs" /
                      "nextgqa_claim_review_policy_v1.json").read_bytes()
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

    def test_offline_page_escapes_claims_and_never_authorizes_scoring(self):
        payload, raw, lineage, lineage_raw = self.payload()
        page = render(payload, raw, lineage, lineage_raw,
                      hashlib.sha256(lineage_raw).hexdigest()).decode("utf-8")
        self.assertIn("connect-src 'none'", page)
        self.assertIn("media-src blob:", page)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("\\u003cscript\\u003e", page)
        self.assertIn("all_relevant_occurrences", page)
        self.assertIn("execution_authorization", page)
        self.assertIn("not_granted", page)
        self.assertIn("crypto.subtle.digest", page)
        self.assertIn("1 ms精度の測定を意味しません", page)
        self.assertIn("video.muted=true", page)
        self.assertNotIn("/private/video-", page)

    def test_non_frozen_or_incomplete_worksheet_rejected(self):
        payload, raw, lineage, lineage_raw = self.payload()
        bad = dict(payload, execution_authorization="granted")
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = dict(payload, effective_records=2)
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["review"]["review_status"] = "completed"
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

    def test_unknown_sensitive_fields_and_mismatched_claim_hash_are_rejected(self):
        payload, raw, lineage, lineage_raw = self.payload()
        bad = json.loads(encode(payload))
        bad["answer_secret_sentinel"] = "MUST_NOT_APPEAR"
        with self.assertRaisesRegex(ValueError, "worksheet_not_review_ready"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["new_private_location_sentinel"] = "MUST_NOT_APPEAR"
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

    def test_independently_pinned_lineage_digest_and_safe_relative_path(self):
        payload, raw, lineage, lineage_raw = self.payload()
        with self.assertRaisesRegex(ValueError, "lineage_manifest_sha256_mismatch"):
            render(payload, raw, lineage, lineage_raw, "0" * 64)
        bad = json.loads(lineage_raw)
        bad["records"][0]["relative_video_path"] = "../../outside/video.mp4"
        bad["records"][0]["video_key"] = "outside/video"
        bad_raw = __import__("json").dumps(bad, ensure_ascii=False, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "lineage_row_binding_invalid"):
            render(payload, raw, bad, bad_raw, hashlib.sha256(bad_raw).hexdigest())
        bad = json.loads(encode(payload))
        bad["records"][0]["claim_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "worksheet_record_invalid"):
            render(bad, encode(bad), lineage, lineage_raw,
                   hashlib.sha256(lineage_raw).hexdigest())

        bad_lineage = json.loads(lineage_raw)
        bad_lineage["records"][0]["relative_video_path"] = r"..\outside\video.mp4"
        bad_lineage["records"][0]["video_key"] = "outside/video"
        bad_lineage_raw = json.dumps(bad_lineage, ensure_ascii=False, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "lineage_row_binding_invalid"):
            render(payload, raw, bad_lineage, bad_lineage_raw,
                   hashlib.sha256(bad_lineage_raw).hexdigest())

    def test_cli_requires_private_inputs_and_writes_exclusive_private_page(self):
        payload, worksheet_raw, lineage, lineage_raw = self.payload()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worksheet = root / "worksheet.json"
            manifest = root / "lineage.json"
            output = root / "review.html"
            worksheet.write_bytes(worksheet_raw)
            manifest.write_bytes(lineage_raw)
            os.chmod(worksheet, 0o600); os.chmod(manifest, 0o600)
            args = ["--input", str(worksheet), "--lineage-manifest", str(manifest),
                    "--expected-lineage-sha256", hashlib.sha256(lineage_raw).hexdigest(),
                    "--output", str(output), "--user-authorized-local-only"]
            self.assertEqual(main(args), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertIn("Safe claim 0", output.read_text())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            output.unlink()
            os.chmod(manifest, 0o644)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
