from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_clevrer_annotation_audit_and_publish import safe_results_path, tracked_status_lines


class AuditPublisherPathTests(unittest.TestCase):
    def test_untracked_local_dataset_does_not_count_as_tracked_change(self) -> None:
        status = " M scripts/example.py\n?? clevrer_download.zip\n?? NExT-GQA/\n"
        self.assertEqual(tracked_status_lines(status), [" M scripts/example.py"])

    def test_accepts_json_directly_inside_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "results").mkdir()
            result = safe_results_path(repo, "results/audit.json")
            self.assertEqual(result, (repo / "results/audit.json").resolve())

    def test_rejects_paths_outside_results_or_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "results").mkdir()
            for value in ("data/audit.json", "results/nested/audit.json", "../audit.json"):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        safe_results_path(repo, value)

    def test_refuses_to_overwrite_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "results").mkdir()
            report = repo / "results/audit.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                safe_results_path(repo, "results/audit.json")


if __name__ == "__main__":
    unittest.main()
