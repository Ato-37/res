from __future__ import annotations

import hashlib
import json
import unittest

from qta_prescore_review_row_adapter import (
    ReviewRowAdapterError,
    adapt_review_audit_to_qta_rows,
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class QtaPrescoreReviewRowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [f"review-token-opaque-{index:04d}" for index in range(1, 4)]
        self.worksheet = {
            "records": [
                {"review_token": token,
                 "video_sha256": sha(f"synthetic-video-{index}".encode())}
                for index, token in enumerate(self.tokens)
            ],
        }
        self.review = {
            "records": [
                {"review_token": token, "review": {
                    "support_decision": "supports_claim",
                    "evidence_intervals_seconds": [[1.001 + index, 2.002 + index]],
                }}
                for index, token in enumerate(self.tokens)
            ],
        }
        self.worksheet_raw = encode(self.worksheet)
        self.review_raw = encode(self.review)
        self.audit_report = {
            "status": "unvalidated_handoff_structurally_valid",
            "failure_codes": [],
            "reviewed_records": 3,
            "worksheet_sha256": sha(self.worksheet_raw),
            "review_payload_sha256": sha(self.review_raw),
            "execution_authorization": "not_granted",
            "semantic_review_authenticated": False,
        }

    def adapt(self, **overrides):
        inputs = {
            "audit_report": self.audit_report,
            "worksheet": self.worksheet,
            "worksheet_raw": self.worksheet_raw,
            "review": self.review,
            "review_raw": self.review_raw,
        }
        inputs.update(overrides)
        return adapt_review_audit_to_qta_rows(**inputs)

    def test_adapts_three_rows_in_worksheet_order_with_bundle_digest(self):
        rows = self.adapt()
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["review_token"] for row in rows], self.tokens)
        self.assertEqual(
            [row["source_video_sha256"] for row in rows],
            [record["video_sha256"] for record in self.worksheet["records"]],
        )
        self.assertEqual(
            [row["review_payload_sha256"] for row in rows],
            [sha(self.review_raw)] * 3,
        )
        self.assertEqual(
            [row["support_decision"] for row in rows],
            [record["review"]["support_decision"] for record in self.review["records"]],
        )
        self.assertEqual(
            [row["evidence_intervals_seconds"] for row in rows],
            [record["review"]["evidence_intervals_seconds"] for record in self.review["records"]],
        )

    def test_non_support_decision_is_preserved_for_downstream_binder(self):
        review = json.loads(self.review_raw)
        review["records"][1]["review"]["support_decision"] = "ambiguous"
        raw = encode(review)
        report = {**self.audit_report, "review_payload_sha256": sha(raw)}
        rows = self.adapt(review=review, review_raw=raw, audit_report=report)
        self.assertEqual(rows[1]["support_decision"], "ambiguous")

    def test_rejects_unsuccessful_audit_reports(self):
        cases = (
            {**self.audit_report, "status": "invalid"},
            {**self.audit_report, "failure_codes": ["review_invalid"]},
            {**self.audit_report, "reviewed_records": 2},
            {**self.audit_report, "execution_authorization": "granted"},
            {**self.audit_report, "semantic_review_authenticated": True},
        )
        for report in cases:
            with self.subTest(report=report):
                with self.assertRaisesRegex(ReviewRowAdapterError, "review_audit_not_successful"):
                    self.adapt(audit_report=report)

    def test_rejects_raw_digest_mismatch_and_objects_not_matching_bytes(self):
        with self.assertRaisesRegex(ReviewRowAdapterError, "worksheet_raw_digest_mismatch"):
            self.adapt(worksheet_raw=self.worksheet_raw + b" ")
        with self.assertRaisesRegex(ReviewRowAdapterError, "review_payload_raw_digest_mismatch"):
            self.adapt(review_raw=self.review_raw + b" ")
        altered = {**self.worksheet, "records": list(self.worksheet["records"])}
        altered["records"][0] = {**altered["records"][0], "video_sha256": "0" * 64}
        with self.assertRaisesRegex(ReviewRowAdapterError, "review_objects_do_not_match_raw_bytes"):
            self.adapt(worksheet=altered)

    def test_rejects_wrong_row_count_order_or_token_binding(self):
        short_worksheet = {"records": self.worksheet["records"][:2]}
        short_raw = encode(short_worksheet)
        short_report = {**self.audit_report, "worksheet_sha256": sha(short_raw)}
        with self.assertRaisesRegex(ReviewRowAdapterError, "review_bundle_row_count_invalid"):
            self.adapt(worksheet=short_worksheet, worksheet_raw=short_raw,
                       audit_report=short_report)

        reordered = json.loads(self.review_raw)
        reordered["records"][0], reordered["records"][1] = (
            reordered["records"][1], reordered["records"][0])
        raw = encode(reordered)
        report = {**self.audit_report, "review_payload_sha256": sha(raw)}
        with self.assertRaisesRegex(ReviewRowAdapterError, "row_1_review_token_order_mismatch"):
            self.adapt(review=reordered, review_raw=raw, audit_report=report)

    def test_rejects_invalid_source_hash_and_review_fields(self):
        worksheet = json.loads(self.worksheet_raw)
        worksheet["records"][1]["video_sha256"] = "bad"
        raw = encode(worksheet)
        report = {**self.audit_report, "worksheet_sha256": sha(raw)}
        with self.assertRaisesRegex(ReviewRowAdapterError, "row_2_source_video_digest_invalid"):
            self.adapt(worksheet=worksheet, worksheet_raw=raw, audit_report=report)

        review = json.loads(self.review_raw)
        review["records"][2]["review"]["evidence_intervals_seconds"] = None
        raw = encode(review)
        report = {**self.audit_report, "review_payload_sha256": sha(raw)}
        with self.assertRaisesRegex(ReviewRowAdapterError, "row_3_evidence_intervals_invalid"):
            self.adapt(review=review, review_raw=raw, audit_report=report)

    def test_duplicate_json_keys_fail_closed(self):
        duplicate_raw = b'{"records":[],"records":[]}'
        report = {**self.audit_report, "worksheet_sha256": sha(duplicate_raw)}
        with self.assertRaisesRegex(ReviewRowAdapterError, "duplicate_json_key"):
            self.adapt(worksheet_raw=duplicate_raw, audit_report=report)


if __name__ == "__main__":
    unittest.main()
