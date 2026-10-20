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
from qta_selection_provenance import FROZEN_SELECTOR_SOURCE_SHA256


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _bound_row(index, question, claim, options, video_review_hash):
    token = f"opaque00000000000000000{index + 1}"
    scores = [-1.0, -1.2, -2.0, -2.5, -3.0]
    intervals_hash = "d" * 64
    row = {
        "review_token": token, "question": question, "options": options,
        "claim": claim, "option_log_probabilities": scores,
        "p_index": 0, "q_index": 1,
        "video_support_decision": "supports_claim",
        "reviewed_intervals_sha256": intervals_hash,
    }
    positions = [2, 3, 4]
    binding = {
        "schema_version": 2, "review_token": token,
        "answer_label_order": ["A", "B", "C", "D", "E"],
        "answer_token_ids": [100 + 5 * index + offset for offset in range(5)],
        "choice_token_binding_sha256": "1" * 64,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "options_sha256": hashlib.sha256(_canonical(options)).hexdigest(),
        "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
        "source_video_sha256": f"{index + 2:064x}",
        "reviewed_intervals_sha256": intervals_hash,
        "video_support_decision": "supports_claim",
        "prompt_sha256": "3" * 64, "input_ids_sha256": "4" * 64,
        "input_length": 11, "score_position": 10,
        "score_input_token_id": 100 + index, "video_token_id": 32000,
        "question_positions": positions,
        "question_positions_sha256": hashlib.sha256(_canonical(positions)).hexdigest(),
        "question_position_mapper_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
        "processor_batch_sha256": "5" * 64,
        "option_log_probabilities": scores, "p_index": 0, "q_index": 1,
        "run_digests": {
            name: str(value) * 64 for name, value in zip(
                ("protocol_sha256", "model_sha256", "tokenizer_sha256",
                 "scorer_sha256", "processor_sha256", "runtime_sha256"),
                "6789ab")
        },
    }
    row["score_binding"] = binding
    row["score_binding_sha256"] = hashlib.sha256(_canonical(binding)).hexdigest()
    return row


class PredictionLinkRendererTests(unittest.TestCase):
    def setUp(self):
        self.protocol_hash = "b" * 64
        self.video_review_hash = "c" * 64
        self.rows = [
            _bound_row(0, "Question __PROTOCOL_HASH__ <script>alert(1)</script>?",
                       "A person waves. </script><script>bad()</script>",
                       ["A", "B", "C", "D", "E"], self.video_review_hash),
            _bound_row(1, "Question two?", "A person sits.",
                       ["A", "B", "C", "D", "E"], self.video_review_hash),
            _bound_row(2, "Question three?", "A car turns.",
                       ["A", "B", "C", "D", "E"], self.video_review_hash),
        ]
        self.row = self.rows[0]
        self.manifest_raw = json.dumps(
            {"schema_version": 3,
             "purpose": "qta_baseline_prediction_link_manifest_v3",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": self.rows}, ensure_ascii=False, sort_keys=True,
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
        self.assertEqual(page.count("class=\"p-choice\""), len(self.rows))
        self.assertEqual(page.count("class=\"q-choice\""), len(self.rows))
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
            "records": self.rows,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        page = render_prediction_link_review_page(
            baseline_manifest_raw=raw,
            expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            protocol_sha256=self.protocol_hash,
            video_claim_review_receipt_sha256=self.video_review_hash)
        self.assertIn("execution_authorization:'not_granted'", page)
        self.assertNotIn("question_positions", page)

    def test_legacy_baseline_manifests_v1_and_v2_are_rejected(self):
        for version, purpose in (
                (1, "qta_baseline_prediction_link_manifest"),
                (2, "qta_baseline_prediction_link_manifest_v2")):
            with self.subTest(version=version):
                raw = json.dumps({
                    "schema_version": version,
                    "purpose": purpose,
                    "video_claim_review_receipt_sha256": self.video_review_hash,
                    "records": self.rows,
                }, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode()
                with self.assertRaisesRegex(ValueError, "baseline_manifest_schema_invalid"):
                    render_prediction_link_review_page(
                        baseline_manifest_raw=raw,
                        expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                        protocol_sha256=self.protocol_hash,
                        video_claim_review_receipt_sha256=self.video_review_hash)

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
        raw = json.dumps({"schema_version": 3,
                          "purpose": "qta_baseline_prediction_link_manifest_v3",
                          "video_claim_review_receipt_sha256": self.video_review_hash,
                          "records": [broken, *self.rows[1:]]},
                         ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError,
                                    "baseline_manifest_score_binding_invalid"):
            render_prediction_link_review_page(
                baseline_manifest_raw=raw,
                expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)

    def test_v3_manifest_without_score_binding_is_rejected(self):
        broken = dict(self.row)
        broken.pop("score_binding")
        broken.pop("score_binding_sha256")
        raw = json.dumps({"schema_version": 3,
                          "purpose": "qta_baseline_prediction_link_manifest_v3",
                          "video_claim_review_receipt_sha256": self.video_review_hash,
                          "records": [broken, *self.rows[1:]]},
                         ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError,
                                    "baseline_manifest_score_binding_invalid"):
            render_prediction_link_review_page(
                baseline_manifest_raw=raw,
                expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)

    def test_v3_manifest_with_stale_score_binding_digest_is_rejected(self):
        rows = [dict(row) for row in self.rows]
        rows[0] = dict(rows[0], score_binding_sha256="f" * 64)
        raw = json.dumps({"schema_version": 3,
                          "purpose": "qta_baseline_prediction_link_manifest_v3",
                          "video_claim_review_receipt_sha256": self.video_review_hash,
                          "records": rows}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError,
                                    "baseline_manifest_score_binding_invalid"):
            render_prediction_link_review_page(
                baseline_manifest_raw=raw,
                expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                protocol_sha256=self.protocol_hash,
                video_claim_review_receipt_sha256=self.video_review_hash)

    def test_v3_manifest_rejects_two_or_four_records(self):
        for rows in (self.rows[:2], self.rows + [dict(self.rows[0])]):
            with self.subTest(record_count=len(rows)):
                raw = _canonical({
                    "schema_version": 3,
                    "purpose": "qta_baseline_prediction_link_manifest_v3",
                    "video_claim_review_receipt_sha256": self.video_review_hash,
                    "records": rows,
                })
                with self.assertRaisesRegex(ValueError,
                                            "baseline_manifest_score_binding_invalid"):
                    render_prediction_link_review_page(
                        baseline_manifest_raw=raw,
                        expected_baseline_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                        protocol_sha256=self.protocol_hash,
                        video_claim_review_receipt_sha256=self.video_review_hash)

    def test_v3_manifest_rejects_rehashed_tampered_source_digest(self):
        rows = json.loads(json.dumps(self.rows))
        rows[0]["score_binding"]["question_sha256"] = "e" * 64
        rows[0]["score_binding_sha256"] = hashlib.sha256(
            _canonical(rows[0]["score_binding"])).hexdigest()
        raw = _canonical({
            "schema_version": 3,
            "purpose": "qta_baseline_prediction_link_manifest_v3",
            "video_claim_review_receipt_sha256": self.video_review_hash,
            "records": rows,
        })
        with self.assertRaisesRegex(ValueError,
                                    "baseline_manifest_score_binding_invalid"):
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
