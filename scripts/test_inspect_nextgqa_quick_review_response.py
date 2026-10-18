import json
import tempfile
import unittest
from pathlib import Path

from inspect_nextgqa_quick_review_response import (
    EXPECTED_LIMITATIONS, ReviewError, expected_item_ids, inspect_response,
    summarize_response,
)


ITEM_IDS = ["0123456789abcdef0123", "123456789abcdef01234",
            "23456789abcdef012345"]


def valid_response():
    judgments = ["roughly_matches", "partly_differs", "cannot_tell"]
    return {
        "schema_version": 2,
        "purpose": "exploratory_human_impression_check",
        "status": "informal_observations_only",
        "created_at_utc": "2026-10-19T03:00:00.000Z",
        "nickname": "PRIVATE NICKNAME SHOULD NOT LEAK",
        "answers": [
            {"item_id": item_id, "item_number": index, "judgment": judgments[index - 1],
             "approximate_time": "private time", "note": "private free text"}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ],
        "limitations": sorted(EXPECTED_LIMITATIONS),
    }


class QuickReviewResponseTests(unittest.TestCase):
    def test_extracts_only_the_three_opaque_ids_from_html_script(self):
        items = [{"item_id": item_id, "question": "private", "video_base64": "x"}
                 for item_id in ITEM_IDS]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.html"
            path.write_text(
                "<script>const items=" + json.dumps(items)
                + ";\nlet nickname='';</script>", encoding="utf-8",
            )
            self.assertEqual(expected_item_ids(path), ITEM_IDS)

    def test_inspect_response_uses_artifact_ids_and_emits_only_summary(self):
        items = [{"item_id": item_id} for item_id in ITEM_IDS]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.html"
            response_path = Path(tmp) / "response.json"
            artifact.write_text(
                "<script>const items=" + json.dumps(items)
                + ";\nlet nickname='';</script>", encoding="utf-8",
            )
            response_path.write_text(json.dumps(valid_response()), encoding="utf-8")
            report = inspect_response(response_path, artifact)
            self.assertEqual(report["validated_exploratory_responses"], 3)
            self.assertNotIn("nickname", report)
            self.assertNotIn("answers", report)

    def test_valid_response_summary_excludes_nickname_and_free_text(self):
        report = summarize_response(json.dumps(valid_response(), ensure_ascii=False).encode(), ITEM_IDS)
        self.assertEqual(report["validated_exploratory_responses"], 3)
        self.assertEqual(sum(report["judgment_counts"].values()), 3)
        self.assertFalse(report["nickname_included_in_output"])
        self.assertFalse(report["free_text_included_in_output"])
        self.assertNotIn("PRIVATE NICKNAME", repr(report))
        self.assertNotIn("private free text", repr(report))
        self.assertNotIn("private time", repr(report))
        self.assertEqual(report["faithfulness_inference"], "not_performed")
        self.assertEqual(report["scoring_authorization"], "not_granted")

    def test_rejects_incomplete_duplicate_or_reordered_items(self):
        for mutate in (
            lambda obj: obj["answers"].pop(),
            lambda obj: obj["answers"].__setitem__(1, obj["answers"][0]),
            lambda obj: obj["answers"].reverse(),
        ):
            candidate = valid_response()
            mutate(candidate)
            with self.assertRaises(ReviewError):
                summarize_response(json.dumps(candidate).encode(), ITEM_IDS)

    def test_rejects_wrong_artifact_id_and_unapproved_judgment(self):
        candidate = valid_response()
        candidate["answers"][0]["item_id"] = "f" * 20
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)

    def test_rejects_unhashable_malformed_field_types_cleanly(self):
        candidate = valid_response()
        candidate["answers"][0]["judgment"] = []
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)
        candidate = valid_response()
        candidate["limitations"][0] = []
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)
        candidate = valid_response()
        candidate["answers"][0]["judgment"] = "hallucinated"
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)

    def test_rejects_duplicate_json_keys_and_forged_purpose(self):
        with self.assertRaises(ReviewError):
            summarize_response(b'{"schema_version":2,"schema_version":2}', ITEM_IDS)
        candidate = valid_response()
        candidate["purpose"] = "formal_claim_review"
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)


if __name__ == "__main__":
    unittest.main()
