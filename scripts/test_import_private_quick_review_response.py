from __future__ import annotations

import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from inspect_nextgqa_quick_review_response import EXPECTED_LIMITATIONS, ReviewError
from import_private_quick_review_response import import_response, main
import import_private_quick_review_response as importer


ITEM_IDS = ["0123456789abcdef0123", "123456789abcdef01234",
            "23456789abcdef012345"]


def _artifact(path: Path, item_ids=ITEM_IDS):
    items = [{"item_id": item_id} for item_id in item_ids]
    path.write_text("<script>const items=" + json.dumps(items)
                    + ";\nlet nickname='';</script>", encoding="utf-8")


def _response(item_ids=ITEM_IDS):
    judgments = ["roughly_matches", "partly_differs", "cannot_tell"]
    return {
        "schema_version": 2,
        "purpose": "exploratory_human_impression_check",
        "status": "informal_observations_only",
        "created_at_utc": "2026-10-19T03:00:00.000Z",
        "nickname": "PRIVATE NICKNAME SHOULD NOT LEAK",
        "answers": [
            {"item_id": item_id, "item_number": index,
             "judgment": judgments[index - 1],
             "approximate_time": None, "note": None}
            for index, item_id in enumerate(item_ids, start=1)
        ],
        "limitations": sorted(EXPECTED_LIMITATIONS),
    }


class PrivateQuickReviewImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.private_root = self.root / "private"
        self.private_root.mkdir(mode=0o700)
        self.private_root.chmod(0o700)
        self.artifact = self.root / "artifact.html"
        _artifact(self.artifact)
        self.source = self.root / "download.json"
        self.raw = json.dumps(_response(), ensure_ascii=False).encode("utf-8")
        self.source.write_bytes(self.raw)

    def test_valid_response_is_revalidated_and_copied_owner_only(self):
        destination, report = import_response(
            self.source, artifact_path=self.artifact,
            private_root=self.private_root)
        self.assertEqual(destination.read_bytes(), self.raw)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(destination.parent.stat().st_mode), 0o700)
        self.assertEqual(report["validated_exploratory_responses"], 3)
        self.assertEqual(sum(report["judgment_counts"].values()), 3)
        self.assertTrue(self.source.exists())
        self.assertNotIn("PRIVATE NICKNAME", repr(report))

    def test_invalid_response_is_not_copied(self):
        self.source.write_text('{"schema_version":2}', encoding="utf-8")
        with self.assertRaises(ReviewError):
            import_response(self.source, artifact_path=self.artifact,
                            private_root=self.private_root)
        self.assertFalse((self.private_root / importer.RESPONSE_DIRNAME).exists())

    def test_symlink_source_is_rejected(self):
        link = self.root / "response-link.json"
        link.symlink_to(self.source)
        with self.assertRaises(ReviewError):
            import_response(link, artifact_path=self.artifact,
                            private_root=self.private_root)

    def test_response_directory_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        (self.private_root / importer.RESPONSE_DIRNAME).symlink_to(outside)
        with self.assertRaises(ReviewError):
            import_response(self.source, artifact_path=self.artifact,
                            private_root=self.private_root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_write_failure_cleans_partial_private_file(self):
        with patch.object(importer.os, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                import_response(self.source, artifact_path=self.artifact,
                                private_root=self.private_root)
        response_dir = self.private_root / importer.RESPONSE_DIRNAME
        self.assertEqual(list(response_dir.iterdir()), [])

    def test_group_readable_private_root_is_rejected(self):
        self.private_root.chmod(0o750)
        with self.assertRaisesRegex(ReviewError, "private_root_permissions_invalid"):
            import_response(self.source, artifact_path=self.artifact,
                            private_root=self.private_root)

    def test_filename_collision_preserves_existing_response(self):
        with patch.object(importer, "_response_filename",
                          return_value="response_fixed.json"):
            original, _ = import_response(
                self.source, artifact_path=self.artifact,
                private_root=self.private_root)
            with self.assertRaisesRegex(ReviewError,
                                        "private_response_destination_exists"):
                import_response(self.source, artifact_path=self.artifact,
                                private_root=self.private_root)
        self.assertEqual(original.read_bytes(), self.raw)
        self.assertEqual(
            sorted(path.name for path in original.parent.iterdir()),
            ["response_fixed.json"],
        )

    def test_cli_reports_redacted_summary_and_retains_download(self):
        output = io.StringIO()
        with patch.object(importer, "PRIVATE_ROOT", self.private_root), \
                patch.object(sys, "argv", ["import", str(self.source),
                                            "--artifact", str(self.artifact), "--show-items"]), \
                patch.object(sys, "stdout", output):
            self.assertEqual(main(), 0)
        rendered = output.getvalue()
        self.assertIn("形式確認: 1人・探索的回答 3件", rendered)
        self.assertIn("owner-only (0600)", rendered)
        self.assertNotIn("PRIVATE NICKNAME", rendered)
        self.assertNotIn("download.json", rendered)
        self.assertTrue(self.source.exists())

    def test_cli_reports_multireviewer_v4_without_names(self):
        answers = [
            {"item_id": item_id, "item_number": index,
             "claim_truth": ["cannot_tell", "partly_differs", "roughly_matches"][index - 1],
             "explanation_sufficiency": ["cannot_tell", "insufficient", "insufficient"][index - 1]}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]
        payload = {"schema_version": 4, "purpose": "exploratory_human_impression_check",
                   "status": "informal_observations_only",
                   "created_at_utc": "2026-10-19T03:00:00.000Z",
                   "respondents": [{"nickname": "PRIVATE PERSON A", "answers": answers},
                                   {"nickname": "PRIVATE PERSON B", "answers": answers}],
                   "limitations": sorted(EXPECTED_LIMITATIONS)}
        self.source.write_text(json.dumps(payload), encoding="utf-8")
        output = io.StringIO()
        with patch.object(importer, "PRIVATE_ROOT", self.private_root), \
                patch.object(sys, "argv", ["import", str(self.source),
                                            "--artifact", str(self.artifact)]), \
                patch.object(sys, "stdout", output):
            self.assertEqual(main(), 0)
        rendered = output.getvalue()
        self.assertIn("形式確認: 2人・探索的回答 6件", rendered)
        self.assertNotIn("1人目・第", rendered)
        self.assertIn("一部ちがう: 2件", rendered)
        self.assertIn("根拠の十分さ: 十分=0件、足りない=4件、判断できない=2件", rendered)
        self.assertNotIn("PRIVATE PERSON", rendered)

    def test_cli_reports_multireviewer_v5_dimensions_without_names(self):
        answers = [
            {"item_id": item_id, "item_number": index,
             "question_clarity": ["unclear", "clear", "clear"][index - 1],
             "answer_question_fit": ["fits", "does_not_fit", "fits"][index - 1],
             "video_evidence_sufficiency": ["sufficient", "insufficient", "insufficient"][index - 1]}
            for index, item_id in enumerate(ITEM_IDS, start=1)
        ]
        payload = {"schema_version": 5, "purpose": "exploratory_human_impression_check",
                   "status": "informal_observations_only",
                   "created_at_utc": "2026-10-19T03:00:00.000Z",
                   "respondents": [{"nickname": "PRIVATE PERSON", "answers": answers}],
                   "limitations": sorted(EXPECTED_LIMITATIONS)}
        self.source.write_text(json.dumps(payload), encoding="utf-8")
        output = io.StringIO()
        with patch.object(importer, "PRIVATE_ROOT", self.private_root), \
                patch.object(sys, "argv", ["import", str(self.source),
                                            "--artifact", str(self.artifact)]), \
                patch.object(sys, "stdout", output):
            self.assertEqual(main(), 0)
        rendered = output.getvalue()
        self.assertIn("形式確認: 1人・探索的回答 3件", rendered)
        self.assertIn("答えの適合: 合っている=2件、違っている=1件、判断できない=0件", rendered)
        self.assertIn("動画根拠の十分さ: 十分=1件、足りない=2件、判断できない=0件", rendered)
        self.assertNotIn("1人目・第", rendered)
        self.assertNotIn("PRIVATE PERSON", rendered)


if __name__ == "__main__":
    unittest.main()
