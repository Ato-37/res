from __future__ import annotations

import unittest

from run_qta_full_video_accuracy_diagnostic import _argmax, _gold_index, summarize_records


class FullVideoAccuracyDiagnosticTests(unittest.TestCase):
    def test_argmax_uses_stable_low_index_tie_break(self):
        self.assertEqual(_argmax([0.1, 0.8, 0.8, -2, 0]), 1)

    def test_gold_choice_joins_exactly_one_normalized_answer(self):
        row = {"answer": "Lay on floor", "a0": "clap", "a1": "Lay   on floor",
               "a2": "sit", "a3": "stand", "a4": "crawl"}
        self.assertEqual(_gold_index(row), 1)

    def test_gold_choice_rejects_ambiguous_or_absent_answer(self):
        row = {"answer": "yes", "a0": "yes", "a1": "yes", "a2": "no",
               "a3": "maybe", "a4": "unknown"}
        with self.assertRaisesRegex(ValueError, "gold_choice_join_invalid"):
            _gold_index(row)
        row["answer"] = "other"
        with self.assertRaisesRegex(ValueError, "gold_choice_join_invalid"):
            _gold_index(row)

    def test_summary_reports_finite_accuracy_and_paired_flips(self):
        def row(base, qta, sham):
            return {"conditions": {
                name: {"gold_choice_correct": value}
                for name, value in (("baseline", base), ("qta", qta), ("sham", sham))}}
        summary = summarize_records([row(True, True, False), row(False, True, True),
                                     row(True, False, True)])
        self.assertEqual(summary["baseline"], {"correct": 2, "n": 3,
                                               "accuracy": 2 / 3})
        self.assertEqual(summary["qta"], {"correct": 2, "n": 3,
                                          "accuracy": 2 / 3})
        self.assertEqual(summary["qta_minus_baseline_correct"], 0)
        self.assertEqual(summary["baseline_to_qta_flips"],
                         {"wrong_to_correct": 1, "correct_to_wrong": 1})
        self.assertEqual(summary["qta_to_sham_flips"],
                         {"sham_wrong_qta_correct": 1, "sham_correct_qta_wrong": 1})
        self.assertIn("direct-faithfulness inference",
                      summary["interpretation"])

    def test_summary_requires_rows(self):
        with self.assertRaisesRegex(ValueError, "records_required"):
            summarize_records([])


if __name__ == "__main__":
    unittest.main()
