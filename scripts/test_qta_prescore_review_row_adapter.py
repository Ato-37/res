from __future__ import annotations

import hashlib
import json
import unittest

from qta_prescore_review_row_adapter import (
    ReviewRowAdapterError,
    adapt_review_bundle_to_qta_rows,
)
from test_audit_nextgqa_review_handoff import fixture as handoff_fixture


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class QtaPrescoreReviewRowAdapterTests(unittest.TestCase):
    def setUp(self):
        (self.worksheet, self.worksheet_raw, self.lineage, self.lineage_raw,
         self.review, self.review_raw, self.review_index,
         self.review_index_raw) = handoff_fixture()
        self.tokens = [row["review_token"] for row in self.worksheet["records"]]

    def adapt(self, **overrides):
        inputs = {
            "worksheet_raw": self.worksheet_raw,
            "lineage_raw": self.lineage_raw,
            "review_raw": self.review_raw,
            "review_index_raw": self.review_index_raw,
            "expected_lineage_sha256": sha(self.lineage_raw),
        }
        inputs.update(overrides)
        return adapt_review_bundle_to_qta_rows(**inputs)

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
        self.assertEqual([row["support_decision"] for row in rows],
                         [record["review"]["support_decision"]
                          for record in self.review["records"]])
        self.assertEqual(
            [row["evidence_intervals_seconds"] for row in rows],
            [record["review"]["evidence_intervals_seconds"] for record in self.review["records"]],
        )

    def test_non_support_decision_is_preserved_for_downstream_binder(self):
        review = json.loads(self.review_raw)
        review["records"][1]["review"]["support_decision"] = "ambiguous"
        review_raw = encode(review)
        review_index = dict(self.review_index,
                            review_payload_sha256=sha(review_raw))
        rows = self.adapt(review_raw=review_raw,
                          review_index_raw=encode(review_index))
        self.assertEqual(rows[1]["support_decision"], "ambiguous")

    def test_rejects_invalid_handoff_and_forged_success_summary_is_not_an_input(self):
        with self.assertRaisesRegex(ReviewRowAdapterError,
                                    "review_audit_not_successful"):
            self.adapt(review_raw=b"{}")
        with self.assertRaises(TypeError):
            self.adapt(audit_report={"status": "unvalidated_handoff_structurally_valid"})

    def test_rejects_raw_digest_mismatch_and_objects_not_matching_bytes(self):
        with self.assertRaisesRegex(ReviewRowAdapterError, "review_audit_not_successful"):
            self.adapt(review_raw=self.review_raw + b" ")
        changed_review = json.loads(self.review_raw)
        changed_review["records"][0]["review"]["support_decision"] = "ambiguous"
        with self.assertRaisesRegex(ReviewRowAdapterError,
                                    "review_audit_not_successful"):
            self.adapt(review_raw=encode(changed_review))

    def test_rejects_wrong_row_count_or_order_from_source_audit(self):
        short = json.loads(self.worksheet_raw)
        short["records"] = short["records"][:2]
        with self.assertRaisesRegex(ReviewRowAdapterError,
                                    "review_audit_not_successful"):
            self.adapt(worksheet_raw=encode(short))

        reordered = json.loads(self.review_raw)
        reordered["records"][0], reordered["records"][1] = (
            reordered["records"][1], reordered["records"][0])
        with self.assertRaisesRegex(ReviewRowAdapterError,
                                    "review_audit_not_successful"):
            self.adapt(review_raw=encode(reordered))

    def test_duplicate_json_keys_fail_closed(self):
        duplicate_raw = b'{"records":[],"records":[]}'
        with self.assertRaisesRegex(ReviewRowAdapterError, "duplicate_json_key"):
            self.adapt(worksheet_raw=duplicate_raw)


if __name__ == "__main__":
    unittest.main()
