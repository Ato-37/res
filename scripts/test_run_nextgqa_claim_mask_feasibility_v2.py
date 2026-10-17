"""Synthetic regressions for the non-authorizing v2 candidate skeleton."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import unittest

from run_nextgqa_claim_mask_feasibility_v2 import (
    BINDING_KEYS,
    CONDITIONS,
    FRAME_COUNT,
    REQUESTED_RECORDS,
    run_candidate,
    validate_synthetic_candidate,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_row(index: int) -> dict:
    shared_input_ids_hash = digest(f"ids-{index}")
    shared_video_metadata_hash = digest(f"metadata-{index}")
    return {
        "record_key_sha256": digest(f"synthetic-record-{index}"),
        "source_media_sha256": digest(f"synthetic-media-{index}"),
        "requested_frames": FRAME_COUNT,
        "declared_gate_status": "synthetic_pass_unverified",
        "evidence_ranks": [4, 5],
        "control_ranks": [3, 6],
        "condition_input_hashes": {
            condition: {
                "input_ids_sha256": shared_input_ids_hash,
                "pixel_values_sha256": digest(f"pixels-{index}-{condition}"),
                "video_metadata_sha256": shared_video_metadata_hash,
            }
            for condition in CONDITIONS
        },
    }


def make_candidate(protocol_raw: bytes) -> dict:
    return {
        "schema_version": 1,
        "purpose": "synthetic_v2_atomic_preflight_fixture",
        "fixture_kind": "synthetic_only",
        "status": "candidate_unvalidated",
        "execution_authorization": "not_granted",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "cohort_sha256": digest("synthetic-cohort"),
        "requested_records": REQUESTED_RECORDS,
        "frame_count": FRAME_COUNT,
        "conditions": list(CONDITIONS),
        "binding_hashes": {key: digest(f"synthetic-binding-{key}")
                           for key in BINDING_KEYS},
        "records": [make_row(index) for index in range(REQUESTED_RECORDS)],
    }


def raw_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class AtomicPreflightTests(unittest.TestCase):
    def setUp(self):
        self.protocol_raw = b"synthetic protocol bytes; not the real v2 protocol"
        self.candidate = make_candidate(self.protocol_raw)
        self.expected_protocol_sha256 = hashlib.sha256(self.protocol_raw).hexdigest()

    def run_fixture(self, candidate=None, protocol_raw=None, expected_hash=None):
        return run_candidate(
            raw_json(candidate if candidate is not None else self.candidate),
            self.protocol_raw if protocol_raw is None else protocol_raw,
            self.expected_protocol_sha256 if expected_hash is None else expected_hash,
        )

    def test_positive_synthetic_structure_stays_non_authorizing(self):
        report = self.run_fixture()
        self.assertEqual(report["structural_status"],
                         "synthetic_candidate_passed_unvalidated")
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["validated_records"], 3)
        self.assertEqual(report["requested_model_forwards"], 9)
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])
        self.assertFalse(report["production_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

        # The prompt tokens and temporal/grid metadata are invariant across
        # interventions; only the pixels are expected to change.
        for row in self.candidate["records"]:
            inputs = row["condition_input_hashes"]
            self.assertEqual(len({inputs[c]["input_ids_sha256"] for c in CONDITIONS}), 1)
            self.assertEqual(len({inputs[c]["video_metadata_sha256"] for c in CONDITIONS}), 1)
            self.assertEqual(len({inputs[c]["pixel_values_sha256"] for c in CONDITIONS}), 3)

    def test_condition_input_ids_must_match_within_each_row(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["records"][1]["condition_input_hashes"]["claim_evidence_black"][
            "input_ids_sha256"] = digest("mutated claim-mask input ids")
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], [
            "record_2_nonmask_condition_binding_mismatch"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_video_metadata_must_match_within_each_row(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["records"][0]["condition_input_hashes"]["temporal_control_black"][
            "video_metadata_sha256"] = digest("mutated control metadata")
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], [
            "record_1_nonmask_condition_binding_mismatch"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_condition_pixel_hashes_must_differ_within_each_row(self):
        candidate = make_candidate(self.protocol_raw)
        shared_hash = digest("identical condition pixels")
        for condition in CONDITIONS:
            candidate["records"][1]["condition_input_hashes"][condition][
                "pixel_values_sha256"] = shared_hash
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], [
            "record_2_condition_pixel_hashes_not_distinct"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])
        self.assertFalse(report["production_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

        pair_duplicate = make_candidate(self.protocol_raw)
        pair_hash = digest("two identical condition pixels")
        pair_duplicate["records"][0]["condition_input_hashes"]["full"][
            "pixel_values_sha256"] = pair_hash
        pair_duplicate["records"][0]["condition_input_hashes"][
            "claim_evidence_black"]["pixel_values_sha256"] = pair_hash
        pair_report = self.run_fixture(pair_duplicate)
        self.assertEqual(pair_report["failure_codes"], [
            "record_1_condition_pixel_hashes_not_distinct"])
        self.assertEqual(pair_report["model_loads"], 0)
        self.assertEqual(pair_report["model_forwards"], 0)
        self.assertEqual(pair_report["score_calls"], 0)
        self.assertIsNone(pair_report["D3"])

    def test_late_third_row_failure_has_zero_calls_and_null_d3(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["records"][2]["condition_input_hashes"].pop("temporal_control_black")
        report = self.run_fixture(candidate)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("record_3_condition_set_invalid", report["failure_codes"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_protocol_hash_mismatch_blocks_before_any_calls(self):
        report = self.run_fixture(expected_hash=digest("different protocol"))
        self.assertEqual(report["failure_codes"], ["protocol_hash_mismatch"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_protocol_binding_mismatch_blocks(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["protocol_sha256"] = digest("different protocol")
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], ["candidate_protocol_binding_mismatch"])
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_wrong_record_cardinality_blocks(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["records"] = candidate["records"][:2]
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], ["record_cardinality_invalid"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertIsNone(report["D3"])

    def test_wrong_top_level_frame_count_blocks(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["frame_count"] = 16
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"],
                         ["cohort_frame_or_condition_contract_mismatch"])
        self.assertEqual(report["score_calls"], 0)
        self.assertIsNone(report["D3"])

    def test_wrong_late_record_frame_count_blocks(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["records"][2]["requested_frames"] = 16
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], ["record_3_frame_count_invalid"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertIsNone(report["D3"])

    def test_condition_set_must_be_exactly_three_protocol_conditions(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["conditions"] = ["full", "black_gold_interval_mask"]
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"],
                         ["cohort_frame_or_condition_contract_mismatch"])
        self.assertEqual(report["score_calls"], 0)

    def test_real_or_authorizing_fixture_is_rejected(self):
        candidate = make_candidate(self.protocol_raw)
        candidate["fixture_kind"] = "real_inputs"
        candidate["execution_authorization"] = "granted"
        report = self.run_fixture(candidate)
        self.assertEqual(report["failure_codes"], [
            "candidate_not_explicitly_synthetic_and_non_authorizing"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["score_calls"], 0)

    def test_preflight_and_entrypoint_have_no_torch_or_callbacks(self):
        source_path = Path(__file__).with_name("run_nextgqa_claim_mask_feasibility_v2.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertNotIn("torch", imported)
        self.assertEqual(list(inspect.signature(run_candidate).parameters), [
            "candidate_raw", "protocol_raw", "expected_protocol_sha256"])


if __name__ == "__main__":
    unittest.main()
