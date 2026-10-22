import unittest
import json
from pathlib import Path
import tempfile

from run_qta_generated_text_annotation_dependence import (
    execute_candidate_loop, scoring_cohort_complete, tracked_call,
    validate_scoring_partition, load_dataset_indexes, selector_forward,
)


class ScoringPartitionTests(unittest.TestCase):
    def setUp(self):
        self.source_hashes = {"runner": "abc", "capture": "def"}
        self.preflight_hash = "123"
        self.smoke = {
            "status": "completed_exploratory_finite_cohort",
            "requested_n": 1,
            "effective_n": 1,
            "start_index": 0,
            "failures": [],
            "preflight_sha256": self.preflight_hash,
            "source_sha256": self.source_hashes,
            "direct_faithfulness": False,
            "faithfulness_result": False,
        }

    def test_only_smoke_then_disjoint_three_row_partition_is_allowed(self):
        validate_scoring_partition(start_index=0, n=1)
        validate_scoring_partition(start_index=1, n=3,
            smoke_receipt=self.smoke, preflight_sha256=self.preflight_hash,
            source_sha256=self.source_hashes)
        for start, n in ((0, 3), (1, 1), (0, 2), (2, 3)):
            with self.subTest(start=start, n=n), self.assertRaisesRegex(ValueError,
                    "scoring_partition_invalid"):
                validate_scoring_partition(start_index=start, n=n)

    def test_descriptive_partition_requires_successful_same_source_smoke(self):
        for receipt in (None, {**self.smoke, "effective_n": 0},
                        {**self.smoke, "failures": [{"stage": "generation"}]},
                        {**self.smoke, "source_sha256": {"runner": "changed"}},
                        {**self.smoke, "preflight_sha256": "other"}):
            with self.subTest(receipt=receipt), self.assertRaisesRegex(ValueError,
                    "successful_same_preflight_smoke_receipt_required"):
                validate_scoring_partition(start_index=1, n=3,
                    smoke_receipt=receipt, preflight_sha256=self.preflight_hash,
                    source_sha256=self.source_hashes)


class DatasetIndexTests(unittest.TestCase):
    def test_question_csv_and_gsub_json_load_as_bound_lookup_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.csv").write_text(
                "video_id,qid,question\nv1,q1,What happens?\n", encoding="utf-8")
            (root / "gsub_test.json").write_text(
                json.dumps({"v1": {"location": {"q1": [[0.0, 1.0]]}}}),
                encoding="utf-8")
            rows, gsub = load_dataset_indexes(root)
        self.assertEqual(rows[("v1", "q1")]["question"], "What happens?")
        self.assertEqual(gsub["v1"]["location"]["q1"], [[0.0, 1.0]])

    def test_selector_forward_matches_existing_qta_noncollecting_call(self):
        class Model:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return "output"
        model = Model()
        batch = {"input_ids": "input"}
        self.assertEqual(selector_forward(model, batch), "output")
        self.assertEqual(model.kwargs, {"input_ids": "input",
            "use_cache": False, "logits_to_keep": 1})

class FailureReceiptTests(unittest.TestCase):
    def test_partial_attempt_counts_or_any_failure_make_receipt_incomplete(self):
        branches = {"baseline_full": {"attempted": 1, "completed": 1},
                    "qta_control_ablation": {"attempted": 1, "completed": 0}}
        auxiliary = {"selector_forward": {"attempted": 1, "completed": 1},
                     "generation_call": {"attempted": 1, "completed": 1}}
        self.assertFalse(scoring_cohort_complete(requested_n=1,
            records=[{"id": "c1"}], failures=[], branch_counts=branches,
            auxiliary_counts=auxiliary))
        branches["qta_control_ablation"]["completed"] = 1
        self.assertTrue(scoring_cohort_complete(requested_n=1,
            records=[{"id": "c1"}], failures=[], branch_counts=branches,
            auxiliary_counts=auxiliary))
        self.assertFalse(scoring_cohort_complete(requested_n=1,
            records=[{"id": "c1"}], failures=[{"stage": "cuda_synchronize"}],
            branch_counts=branches, auxiliary_counts=auxiliary))

    def test_generation_failure_keeps_attempted_auxiliary_count_and_stage(self):
        counts = {"selector_forward": {"attempted": 0, "completed": 0},
                  "generation_call": {"attempted": 0, "completed": 0}}

        def score(_candidate, _expected, stage):
            tracked_call(counts["selector_forward"], lambda: "captured")
            stage["value"] = "generation_call"
            def fail_generation():
                raise RuntimeError("synthetic generation failure")
            return tracked_call(counts["generation_call"], fail_generation)

        records, failures = execute_candidate_loop(candidates=["c1", "c2"],
            expected_rows=["e1", "e2"], score_one=score, synchronize=lambda: None)
        self.assertEqual(records, [])
        self.assertEqual(failures, [{"candidate_ordinal": 1,
            "failure_type": "RuntimeError", "failure_code":
            "synthetic generation failure", "stage": "generation_call"}])
        self.assertEqual(counts["generation_call"], {"attempted": 1, "completed": 0})
        self.assertEqual(counts["selector_forward"], {"attempted": 1, "completed": 1})

    def test_partial_condition_failure_and_cuda_sync_failure_are_retained(self):
        completed = {"baseline_full": {"attempted": 0, "completed": 0},
                     "qta_control_ablation": {"attempted": 0, "completed": 0}}

        def score(_candidate, _expected, stage):
            tracked_call(completed["baseline_full"], lambda: "score")
            stage["value"] = "score_qta_control_ablation"
            def fail_condition():
                raise ValueError("synthetic late condition failure")
            return tracked_call(completed["qta_control_ablation"], fail_condition)

        def synchronize():
            raise RuntimeError("synthetic CUDA sync failure")

        records, failures = execute_candidate_loop(candidates=["c1"],
            expected_rows=["e1"], score_one=score, synchronize=synchronize)
        self.assertEqual(records, [])
        self.assertEqual(failures, [
            {"candidate_ordinal": 1, "failure_type": "ValueError",
             "failure_code": "synthetic late condition failure",
             "stage": "score_qta_control_ablation"},
            {"candidate_ordinal": None, "failure_type": "RuntimeError",
             "failure_code": "synthetic CUDA sync failure",
             "stage": "cuda_synchronize"},
        ])
        self.assertEqual(completed["qta_control_ablation"],
                         {"attempted": 1, "completed": 0})
        self.assertEqual(completed["baseline_full"],
                         {"attempted": 1, "completed": 1})

    def test_successful_rows_and_failed_sync_cannot_be_called_complete(self):
        records, failures = execute_candidate_loop(candidates=["c1"],
            expected_rows=["e1"], score_one=lambda c, e, stage: {"id": c},
            synchronize=lambda: None)
        self.assertEqual(records, [{"id": "c1"}])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
