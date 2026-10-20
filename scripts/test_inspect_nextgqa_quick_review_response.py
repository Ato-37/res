import json
import io
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inspect_nextgqa_quick_review_response import (
    EXPECTED_LIMITATIONS, ReviewError, expected_artifact_fingerprint, expected_item_ids, inspect_response, main,
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
    def test_v5_keeps_clarity_answer_fit_and_video_evidence_separate(self):
        response = valid_response()
        response["schema_version"] = 5
        response.pop("nickname")
        response.pop("answers")
        response["respondents"] = [{"nickname": "private", "answers": [
            {"item_id": item_id, "item_number": index,
             "question_clarity": ["unclear", "clear", "clear"][index - 1],
             "answer_question_fit": ["fits", "does_not_fit", "fits"][index - 1],
             "video_evidence_sufficiency": ["sufficient", "insufficient", "insufficient"][index - 1]}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]}]
        report = summarize_response(json.dumps(response).encode(), ITEM_IDS)
        self.assertIsNone(report["judgment_counts"])
        self.assertEqual(report["question_clarity_counts"],
                         {"clear": 2, "unclear": 1, "cannot_tell": 0})
        self.assertEqual(report["answer_question_fit_counts"],
                         {"fits": 2, "does_not_fit": 1, "cannot_tell": 0})
        self.assertEqual(report["video_evidence_sufficiency_counts"],
                         {"sufficient": 1, "insufficient": 2, "cannot_tell": 0})
        self.assertEqual(report["item_judgments"][1]["answer_question_fit"], "does_not_fit")
        self.assertEqual(report["item_judgments"][2]["video_evidence_sufficiency"], "insufficient")
        self.assertEqual(report["item_summary"][1]["answer_question_fit_counts"],
                         {"fits": 0, "does_not_fit": 1, "cannot_tell": 0})
        self.assertNotIn("private", repr(report))
        self.assertEqual(report["faithfulness_inference"], "not_performed")
        self.assertEqual(report["scoring_authorization"], "not_granted")

    def test_v3_separates_claim_truth_from_explanation_sufficiency(self):
        response = valid_response()
        response["schema_version"] = 3
        response["answers"] = [
            {"item_id": item_id, "item_number": index,
             "claim_truth": ["cannot_tell", "partly_differs", "roughly_matches"][index - 1],
             "explanation_sufficiency": ["cannot_tell", "insufficient", "insufficient"][index - 1]}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]
        report = summarize_response(json.dumps(response).encode(), ITEM_IDS)
        self.assertEqual(sum(report["judgment_counts"].values()), 3)
        self.assertEqual(report["sufficiency_counts"],
                         {"sufficient": 0, "insufficient": 2, "cannot_tell": 1})
        self.assertEqual(report["item_judgments"][1]["claim_truth"], "partly_differs")
        self.assertEqual(report["item_judgments"][1]["explanation_sufficiency"], "insufficient")

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

    def test_v6_is_bound_to_exact_artifact_and_modified_bytes_fail(self):
        items = [{"item_id": item_id} for item_id in ITEM_IDS]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.html"
            response_path = Path(tmp) / "response.json"
            html = ('<meta name="review-artifact-sha256" content="__ARTIFACT_FINGERPRINT__">'
                    + "<script>const items=" + json.dumps(items)
                    + ";\nlet nickname='';</script>")
            digest = hashlib.sha256(html.encode()).hexdigest()
            html = html.replace("__ARTIFACT_FINGERPRINT__", digest)
            artifact.write_text(html, encoding="utf-8")
            self.assertEqual(expected_artifact_fingerprint(artifact), digest)
            response = valid_response()
            response.pop("nickname")
            response.pop("answers")
            response["schema_version"] = 6
            response["artifact_fingerprint"] = digest
            response["respondents"] = [{"nickname": "private", "answers": [
                {"item_id": item_id, "item_number": index,
                 "question_clarity": "clear", "answer_question_fit": "fits",
                 "video_evidence_sufficiency": "sufficient"}
                for index, item_id in enumerate(ITEM_IDS, start=1)
            ]}]
            response_path.write_text(json.dumps(response), encoding="utf-8")
            report = inspect_response(response_path, artifact)
            self.assertEqual(report["artifact_binding"], "verified")
            artifact.write_text(html.replace("<script>", "<script> "), encoding="utf-8")
            with self.assertRaisesRegex(ReviewError, "artifact_fingerprint_mismatch"):
                expected_artifact_fingerprint(artifact)

    def test_legacy_responses_are_explicitly_unbound(self):
        report = summarize_response(json.dumps(valid_response()).encode(), ITEM_IDS)
        self.assertEqual(report["artifact_binding"], "legacy_unbound")

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
            report = inspect_response(response_path, artifact, allow_unbound_legacy=True)
            self.assertEqual(report["validated_exploratory_responses"], 3)
            self.assertNotIn("nickname", report)
            self.assertNotIn("answers", report)

    def test_inspector_requires_opt_in_before_summarizing_legacy_response(self):
        items = [{"item_id": item_id} for item_id in ITEM_IDS]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.html"
            response_path = Path(tmp) / "response.json"
            artifact.write_text("<script>const items=" + json.dumps(items)
                                + ";\nlet nickname='';</script>", encoding="utf-8")
            response_path.write_text(json.dumps(valid_response()), encoding="utf-8")
            with self.assertRaisesRegex(ReviewError, "legacy_artifact_binding_requires_explicit_opt_in"):
                inspect_response(response_path, artifact)

    def test_cli_rejects_legacy_by_default(self):
        items = [{"item_id": item_id} for item_id in ITEM_IDS]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.html"
            response_path = Path(tmp) / "response.json"
            artifact.write_text("<script>const items=" + json.dumps(items)
                                + ";\nlet nickname='';</script>", encoding="utf-8")
            response_path.write_text(json.dumps(valid_response()), encoding="utf-8")
            with patch.object(sys, "argv", ["inspect", str(response_path),
                                              "--artifact", str(artifact)]), \
                    patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 2)

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

    def test_cli_defaults_to_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            response_path = Path(tmp) / "response.json"
            artifact_path = Path(tmp) / "artifact.html"
            response_path.write_text(json.dumps(valid_response()), encoding="utf-8")
            items = [{"item_id": item_id} for item_id in ITEM_IDS]
            artifact_path.write_text(
                "<script>const items=" + json.dumps(items)
                + ";\nlet nickname='';</script>", encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(sys, "argv", ["inspect", str(response_path),
                                             "--artifact", str(artifact_path),
                                             "--allow-unbound-legacy"]), \
                    patch.object(sys, "stdout", output):
                self.assertEqual(main(), 0)
            rendered = output.getvalue()
            self.assertIn("だいたい合っている: 1件", rendered)
            self.assertIn("未検証（旧形式JSON）", rendered)
            self.assertNotIn("1人目・第", rendered)
            self.assertNotIn("PRIVATE NICKNAME", rendered)
            self.assertNotIn("private free text", rendered)

    def test_cli_can_explicitly_show_item_choices_without_names_or_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            response_path = Path(tmp) / "response.json"
            artifact_path = Path(tmp) / "artifact.html"
            response_path.write_text(json.dumps(valid_response()), encoding="utf-8")
            items = [{"item_id": item_id} for item_id in ITEM_IDS]
            artifact_path.write_text(
                "<script>const items=" + json.dumps(items)
                + ";\nlet nickname='';</script>", encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(sys, "argv", ["inspect", str(response_path),
                                             "--artifact", str(artifact_path), "--show-items",
                                             "--allow-unbound-legacy"]), \
                    patch.object(sys, "stdout", output):
                self.assertEqual(main(), 0)
            rendered = output.getvalue()
            self.assertIn("1人目・第1件目: だいたい合っている", rendered)
            self.assertNotIn("PRIVATE NICKNAME", rendered)
            self.assertNotIn("private free text", rendered)

    def test_item_summary_aggregates_without_exposing_respondent_rows(self):
        response = valid_response()
        response["schema_version"] = 5
        response.pop("nickname")
        response.pop("answers")
        response["respondents"] = []
        for judgments in (("unclear", "does_not_fit", "insufficient"),
                          ("clear", "fits", "sufficient"),
                          ("unclear", "does_not_fit", "insufficient")):
            response["respondents"].append({"nickname": "private", "answers": [
                {"item_id": item_id, "item_number": index,
                 "question_clarity": judgments[0],
                 "answer_question_fit": judgments[1],
                 "video_evidence_sufficiency": judgments[2]}
                for index, item_id in enumerate(ITEM_IDS, start=1)
            ]})
        report = summarize_response(json.dumps(response).encode(), ITEM_IDS)
        self.assertEqual(report["item_summary"][0]["question_clarity_counts"],
                         {"clear": 1, "unclear": 2, "cannot_tell": 0})
        self.assertEqual(report["item_summary"][2]["video_evidence_sufficiency_counts"],
                         {"sufficient": 1, "insufficient": 2, "cannot_tell": 0})
        self.assertNotIn("private", repr(report))

    def test_item_summary_cli_prints_counts_but_no_reviewer_rows(self):
        response = valid_response()
        response["schema_version"] = 5
        response.pop("nickname")
        response.pop("answers")
        response["respondents"] = [{"nickname": "PRIVATE NAME", "answers": [
            {"item_id": item_id, "item_number": index,
             "question_clarity": "unclear", "answer_question_fit": "does_not_fit",
             "video_evidence_sufficiency": "insufficient"}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]}]
        with tempfile.TemporaryDirectory() as tmp:
            response_path = Path(tmp) / "response.json"
            artifact_path = Path(tmp) / "artifact.html"
            response_path.write_text(json.dumps(response), encoding="utf-8")
            items = [{"item_id": item_id} for item_id in ITEM_IDS]
            artifact_path.write_text(
                "<script>const items=" + json.dumps(items)
                + ";\nlet nickname='';</script>", encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(sys, "argv", ["inspect", str(response_path),
                                             "--artifact", str(artifact_path),
                                             "--show-item-summary", "--allow-unbound-legacy"]), \
                    patch.object(sys, "stdout", output):
                self.assertEqual(main(), 0)
            rendered = output.getvalue()
            self.assertIn("第2問（n=1）", rendered)
            self.assertIn("回答適合 適合=0, 不適合=1, 判断不能=0", rendered)
            self.assertNotIn("PRIVATE NAME", rendered)
            self.assertNotIn("人目・第", rendered)

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

        for field in ("question_clarity", "answer_question_fit",
                      "video_evidence_sufficiency"):
            candidate = valid_response()
            candidate["schema_version"] = 5
            candidate.pop("nickname")
            candidate.pop("answers")
            candidate["respondents"] = [{"nickname": "private", "answers": [
                {"item_id": item_id, "item_number": index,
                 "question_clarity": "clear", "answer_question_fit": "fits",
                 "video_evidence_sufficiency": "sufficient"}
                for index, item_id in enumerate(ITEM_IDS, start=1)
            ]}]
            for malformed in ([], {}):
                candidate["respondents"][0]["answers"][1][field] = malformed
                with self.subTest(field=field, malformed_type=type(malformed).__name__), \
                        self.assertRaises(ReviewError):
                    summarize_response(json.dumps(candidate).encode(), ITEM_IDS)

    def test_rejects_duplicate_json_keys_and_forged_purpose(self):
        with self.assertRaises(ReviewError):
            summarize_response(b'{"schema_version":2,"schema_version":2}', ITEM_IDS)
        candidate = valid_response()
        candidate["purpose"] = "formal_claim_review"
        with self.assertRaises(ReviewError):
            summarize_response(json.dumps(candidate).encode(), ITEM_IDS)


    def test_v4_validates_multiple_reviewers_in_one_json(self):
        one = valid_response()
        answers = [
            {"item_id": item_id, "item_number": index,
             "claim_truth": ["cannot_tell", "partly_differs", "roughly_matches"][index - 1],
             "explanation_sufficiency": ["cannot_tell", "insufficient", "insufficient"][index - 1]}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]
        response = {"schema_version": 4, "purpose": one["purpose"], "status": one["status"],
                    "created_at_utc": one["created_at_utc"], "respondents": [
                        {"nickname": "PERSON A", "answers": answers},
                        {"nickname": "PERSON B", "answers": answers}],
                    "limitations": one["limitations"]}
        report = summarize_response(json.dumps(response).encode(), ITEM_IDS)
        self.assertEqual(report["respondent_count"], 2)
        self.assertEqual(report["validated_exploratory_responses"], 6)
        self.assertEqual(sum(report["judgment_counts"].values()), 6)
        self.assertNotIn("PERSON A", repr(report))


if __name__ == "__main__":
    unittest.main()
