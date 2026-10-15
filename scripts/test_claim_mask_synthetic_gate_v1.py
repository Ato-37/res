from fractions import Fraction
import unittest

from claim_mask_synthetic_gate_v1 import (
    FIXTURE_TYPE,
    SyntheticScorerSpy,
    run_synthetic_gate_fixture_v1,
)
from preflight_claim_mask_timing import sha256_frame_buffer
from validate_claim_mask_protocol import validate as validate_protocol


def record(video_id):
    hashes = [sha256_frame_buffer(bytes([i + len(video_id)]) * (32 * 32 * 3))
              for i in range(4)]
    timing = sha256_frame_buffer(f"{video_id}:timing".encode())
    return {
        "video_id": video_id,
        "pts": list(range(7)),
        "time_base": [1, 2],
        "num_frames": 4,
        "evidence_intervals": [[0, 0]],
        "sample_indices": [0, 2, 4, 6],
        "sampled_pixel_sha256": hashes,
        "processor_pixel_sha256": hashes.copy(),
        "processor_fps": 1.0,
        "processor_frame_indices": [0, 1, 2, 3],
        "condition_timing_sha256": [timing, timing, timing],
    }


def fixture(rows=None):
    return {"fixture_type": FIXTURE_TYPE,
            "records": rows if rows is not None else [record("video-a"), record("video-b")]}


def valid_structure(_records):
    return {"status": "synthetic_structure_passed", "failure_codes": [],
            "execution_authorization": "not_granted"}


class SyntheticGateOrderingTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "expected_video_ids": ["video-a", "video-b"],
            "expected_num_frames": 4,
            "tolerance_seconds": Fraction(0),
            "structural_validator": valid_structure,
        }

    def run_gate(self, value, scorer=None, **overrides):
        options = dict(self.kwargs)
        options.update(overrides)
        if scorer is None:
            scorer = SyntheticScorerSpy()
        return run_synthetic_gate_fixture_v1(value, scorer=scorer, **options)

    def test_valid_fixture_scores_once_after_both_cohort_gates(self):
        events = []
        scorer = SyntheticScorerSpy()

        def validator(rows):
            events.append(("structure", [row["video_id"] for row in rows]))
            return valid_structure(rows)

        result = self.run_gate(fixture([record("video-b"), record("video-a")]), scorer,
                               structural_validator=validator)
        self.assertEqual(result["status"], "synthetic_scorer_callback_completed")
        self.assertEqual(result["score_calls"], 1)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(events, [("structure", ["video-b", "video-a"])])
        self.assertEqual(scorer.calls, 1)
        self.assertEqual(scorer.video_id_batches, [["video-a", "video-b"]])

    def test_late_member_timing_failure_means_zero_validator_and_score_calls(self):
        rows = [record("video-a"), record("video-b")]
        rows[1]["processor_fps"] = 2.0
        calls = []
        scorer = SyntheticScorerSpy()
        result = self.run_gate(fixture(rows), scorer,
                               structural_validator=lambda cohort: calls.append("validate"))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["timing_preflight"]["denominator"], 2)
        self.assertEqual(result["timing_preflight"]["validated_videos"], 1)
        self.assertIn("video_timing_invalid", result["timing_preflight"]["failure_codes"])
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(scorer.calls, 0)
        self.assertEqual(calls, [])

    def test_invalid_tolerance_blocks_before_structure_and_scorer(self):
        validator_calls = []
        scorer = SyntheticScorerSpy()
        result = self.run_gate(
            fixture(), scorer,
            tolerance_seconds=0.0,
            structural_validator=lambda rows: validator_calls.append("validate"),
        )
        self.assertEqual(result["status"], "blocked")
        self.assertIn("video_timing_invalid", result["timing_preflight"]["failure_codes"])
        self.assertEqual(result["timing_preflight"]["validated_videos"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(scorer.calls, 0)
        self.assertEqual(validator_calls, [])

    def test_structural_failure_after_valid_timing_means_zero_score_calls(self):
        calls = []
        scorer = SyntheticScorerSpy()

        def invalid_structure(_rows):
            calls.append("validate")
            return {"status": "blocked", "failure_codes": ["protocol_invalid"],
                    "execution_authorization": "not_granted"}

        result = self.run_gate(fixture(), scorer,
                               structural_validator=invalid_structure)
        self.assertEqual(result["failure_codes"], ["structural_preflight_failed"])
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(scorer.calls, 0)
        self.assertEqual(calls, ["validate"])

    def test_actual_protocol_validator_rejection_means_zero_score_calls(self):
        validator_reports = []

        def protocol_adapter(_records):
            report = validate_protocol({})
            validator_reports.append(report)
            valid = report.get("structurally_valid") is True
            return {"status": "synthetic_structure_passed" if valid else "blocked",
                    "failure_codes": [] if valid else report.get("failure_codes", ["protocol_invalid"]),
                    "execution_authorization": "not_granted"}

        scorer = SyntheticScorerSpy()
        result = self.run_gate(fixture(), scorer,
                               structural_validator=protocol_adapter)
        self.assertEqual(result["failure_codes"], ["structural_preflight_failed"])
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(len(validator_reports), 1)
        self.assertFalse(validator_reports[0]["structurally_valid"])
        self.assertEqual(scorer.calls, 0)

    def test_inconsistent_structural_reports_fail_closed(self):
        reports = [
            {"status": "synthetic_structure_passed", "failure_codes": ["hidden_failure"],
             "execution_authorization": "not_granted"},
            {"status": "synthetic_structure_passed", "failure_codes": [],
             "execution_authorization": "granted"},
            {"status": "synthetic_structure_passed", "failure_codes": [],
             "execution_authorization": "not_granted", "extra": "unknown"},
        ]
        for report in reports:
            with self.subTest(report=report):
                scorer = SyntheticScorerSpy()
                result = self.run_gate(fixture(), scorer,
                                       structural_validator=lambda rows: report)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["score_calls"], 0)
                self.assertEqual(scorer.calls, 0)

    def test_missing_duplicate_members_and_bad_marker_block_before_callbacks(self):
        cases = [
            fixture([record("video-a")]),
            fixture([record("video-a"), record("video-a")]),
            {"fixture_type": "project_video_fixture", "records": [record("video-a"), record("video-b")]},
        ]
        for value in cases:
            with self.subTest(value=value.get("fixture_type")):
                calls = []
                scorer = SyntheticScorerSpy()
                result = self.run_gate(value, scorer,
                                       structural_validator=lambda rows: calls.append("validate"))
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["score_calls"], 0)
                self.assertEqual(scorer.calls, 0)
                self.assertEqual(calls, [])

    def test_callback_exceptions_are_reported_without_claiming_completion(self):
        structure = self.run_gate(fixture(), SyntheticScorerSpy(),
                                  structural_validator=lambda rows: (_ for _ in ()).throw(RuntimeError()))
        self.assertEqual(structure["failure_codes"], ["structural_validator_failed"])
        self.assertEqual(structure["score_calls"], 0)
        failing_spy = SyntheticScorerSpy(fail_on_call=True)
        scored = self.run_gate(fixture(), failing_spy)
        self.assertEqual(scored["status"], "synthetic_scorer_failed")
        self.assertEqual(scored["score_calls"], 1)
        self.assertEqual(failing_spy.calls, 1)

    def test_callbacks_cannot_mutate_snapshot_or_score_inputs(self):
        source = fixture()
        scorer = SyntheticScorerSpy()

        def mutate(rows):
            rows.clear()
            return valid_structure(rows)

        result = self.run_gate(source, scorer, structural_validator=mutate)
        self.assertEqual(result["status"], "synthetic_scorer_callback_completed")
        self.assertEqual(scorer.video_id_batches, [["video-a", "video-b"]])
        self.assertEqual([row["video_id"] for row in source["records"]], ["video-a", "video-b"])

    def test_arbitrary_callable_is_not_accepted_as_a_model_scorer(self):
        calls = []
        result = self.run_gate(fixture(), lambda rows: calls.append(rows))
        self.assertEqual(result["failure_codes"], ["synthetic_scorer_spy_required"])
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(calls, [])
        spy = SyntheticScorerSpy()
        with self.assertRaises(AttributeError):
            spy.record_batch_call = lambda rows: calls.append(rows)


if __name__ == "__main__":
    unittest.main()
