from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import render_nextgqa_prediction_link_review as renderer
from render_nextgqa_prediction_link_review import (
    render_from_private_files,
    render_prediction_link_review_page,
)


class PredictionLinkRendererTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "review_token": "opaque-1", "question": "Question __PROTOCOL_HASH__ <script>alert(1)</script>?",
            "options": ["A", "B", "C", "D", "E"],
            "claim": "A person waves. </script><script>bad()</script>",
            "option_log_probabilities": [-0.1, -0.2, -1, -2, -3],
            "p_index": 0, "q_index": 1,
        }
        self.protocol_hash = "b" * 64
        self.video_review_hash = "c" * 64
        self.row.update(video_support_decision="supports_claim",
                        reviewed_intervals_sha256="d" * 64)
        self.manifest_raw = json.dumps(
            {"schema_version": 1, "purpose": "qta_baseline_prediction_link_manifest",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": [self.row]}, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()
        self.manifest_hash = hashlib.sha256(self.manifest_raw).hexdigest()

    def _render(self):
        return render_prediction_link_review_page(
            baseline_manifest_raw=self.manifest_raw,
            expected_baseline_manifest_sha256=self.manifest_hash,
            protocol_sha256=self.protocol_hash,
            video_claim_review_receipt_sha256=self.video_review_hash)

    def test_html_escapes_text_and_binds_fixed_prediction(self):
        page = self._render()
        self.assertIn("Question __PROTOCOL_HASH__ &lt;script&gt;alert(1)&lt;/script&gt;?", page)
        self.assertIn("A person waves. &lt;/script&gt;&lt;script&gt;bad()", page)
        self.assertNotIn("<script>bad()</script>", page)
        self.assertEqual(page.count("class=\"p-choice\""), 1)
        self.assertEqual(page.count("class=\"q-choice\""), 1)
        self.assertIn(self.manifest_hash, page)
        self.assertIn(self.protocol_hash, page)
        self.assertIn("execution_authorization:'not_granted'", page)
        self.assertIn("先行動画レビューを前提に", page)
        self.assertIn(self.video_review_hash, page)
        self.assertIn("prior_video_review_receipt_sha256:videoReviewHash", page)
        self.assertIn("第1段階の動画claim-reviewから独立", page)
        self.assertIn("independent_from_generation_scoring_custody_and_video_review", page)
        self.assertIn("connect-src 'none'", page)
        self.assertIn("JSON.stringify(receipt,null,2)", page)

    def test_baseline_manifest_v3_is_renderable_for_review_only(self):
        raw = json.dumps({
            "schema_version": 3,
            "purpose": "qta_baseline_prediction_link_manifest_v3",
            "video_claim_review_receipt_sha256": self.video_review_hash,
            "records": [self.row],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        page = render_prediction_link_review_page(
            baseline_manifest_raw=raw,
            expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            protocol_sha256=self.protocol_hash,
            video_claim_review_receipt_sha256=self.video_review_hash)
        self.assertIn("execution_authorization:'not_granted'", page)
        self.assertNotIn("question_positions", page)

    def test_hash_protocol_schema_and_prediction_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "baseline_manifest_bytes_hash_mismatch"):
            render_prediction_link_review_page(
                baseline_manifest_raw=self.manifest_raw + b" ",
                expected_baseline_manifest_sha256=self.manifest_hash,
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)
        with self.assertRaisesRegex(ValueError, "protocol_hash_invalid"):
            render_prediction_link_review_page(
                baseline_manifest_raw=self.manifest_raw,
                expected_baseline_manifest_sha256=self.manifest_hash,
                protocol_sha256="BAD",
                video_claim_review_receipt_sha256=self.video_review_hash)
        broken = dict(self.row, p_index=4)
        raw = json.dumps({"schema_version": 1,
                          "purpose": "qta_baseline_prediction_link_manifest",
                          "video_claim_review_receipt_sha256": self.video_review_hash,
                          "records": [broken]}).encode()
        with self.assertRaisesRegex(ValueError, "baseline_prediction_rule_mismatch"):
            render_prediction_link_review_page(
                baseline_manifest_raw=raw,
                expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)

    def test_private_cli_output_is_exclusive_and_mode_0600(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            source = root / "baseline.json"
            source.write_bytes(self.manifest_raw)
            os.chmod(source, 0o600)
            digest = render_from_private_files(
                private_root=root, manifest_name=source.name,
                output_name="review.html", expected_manifest_sha256=self.manifest_hash,
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)
            output = root / "review.html"
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                render_from_private_files(
                    private_root=root, manifest_name=source.name,
                    output_name="review.html", expected_manifest_sha256=self.manifest_hash,
                    protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)

    def test_private_writer_removes_partial_file_after_write_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            source = root / "baseline.json"
            source.write_bytes(self.manifest_raw)
            os.chmod(source, 0o600)
            real_write = os.write
            calls = 0

            def partial_then_fail(fd, data):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(fd, data[:8])
                raise OSError("injected write failure")

            with patch.object(renderer.os, "write", side_effect=partial_then_fail):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    renderer.render_from_private_files(
                        private_root=root, manifest_name=source.name,
                        output_name="review.html", expected_manifest_sha256=self.manifest_hash,
                        protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)
            self.assertFalse((root / "review.html").exists())

    def test_private_cli_rejects_symlink_manifest_and_unsafe_root(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "private"
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            outside = parent / "outside.json"
            outside.write_bytes(self.manifest_raw)
            (root / "baseline.json").symlink_to(outside)
            with self.assertRaises(OSError):
                render_from_private_files(
                    private_root=root, manifest_name="baseline.json",
                    output_name="review.html", expected_manifest_sha256=self.manifest_hash,
                    protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)
            os.chmod(root, 0o755)
            with self.assertRaisesRegex(ValueError, "private_root_permissions_invalid"):
                render_from_private_files(
                    private_root=root, manifest_name="baseline.json",
                    output_name="review.html", expected_manifest_sha256=self.manifest_hash,
                    protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)


if __name__ == "__main__":
    unittest.main()
