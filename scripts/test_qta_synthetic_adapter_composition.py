from __future__ import annotations

import copy
import hashlib
import json
import unittest

from qta_prescore_clock_row_adapter import adapt_calibration_to_clock_row
from qta_prescore_processor_row_adapter import processor_binding_row
from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort
import test_qta_prescore_clock_row_adapter as clock_adapter_test
import test_qta_prescore_processor_row_adapter as processor_adapter_test
from test_qta_prescore_clock_row_adapter import seal
from test_qta_synthetic_cohort_preflight import synthetic_mapping_inputs


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _replace_tokens(inputs):
    old_tokens = list(inputs["expected_cohort_tokens"])
    new_tokens = [f"{index + 1:024x}" for index in range(3)]
    mapping = dict(zip(old_tokens, new_tokens, strict=True))
    inputs["expected_cohort_tokens"] = new_tokens
    for name in ("review_rows", "clock_rows", "processor_rows", "baseline_rows"):
        for row in inputs[name]:
            row["review_token"] = mapping[row["review_token"]]
    for name in ("expected_video_sha256_by_token",
                 "expected_processor_batch_sha256_by_token",
                 "expected_sampling_metadata_sha256_by_token",
                 "expected_processor_groups_by_token",
                 "expected_calibration_sha256_by_token",
                 "expected_baseline_source_sha256_by_token"):
        inputs[name] = {mapping[token]: value for token, value in inputs[name].items()}
    return inputs


def _actual_adapter_rows(inputs):
    clock_fixture = clock_adapter_test.QtaPrescoreClockRowAdapterTests()
    clock_fixture.setUp()
    processor_fixture = processor_adapter_test.QtaPrescoreProcessorRowAdapterTests()
    processor_fixture.setUp()
    clock_rows, processor_rows = [], []
    try:
        for index, token in enumerate(inputs["expected_cohort_tokens"]):
            video_sha = inputs["expected_video_sha256_by_token"][token]
            capture_obj = json.loads(clock_fixture.capture_raw)
            capture_obj["source_media_sha256"] = video_sha
            capture_raw = json.dumps(capture_obj, sort_keys=True,
                                     separators=(",", ":")).encode()
            calibration_body = {key: value for key, value in clock_fixture.calibration.items()
                                if key not in {"artifact_raw", "artifact_sha256"}}
            calibration_body.update({
                "review_token": token,
                "source_media_sha256": video_sha,
                "capture_sha256": sha(capture_raw),
            })
            calibration = seal(calibration_body)
            clock_row = adapt_calibration_to_clock_row(
                review_token=token,
                source_video_sha256=video_sha,
                video_claim_review_receipt_sha256=(
                    inputs["expected_video_review_receipt_sha256"]),
                calibration=calibration,
                capture_raw=capture_raw,
            )
            clock_rows.append(clock_row)
            inputs["expected_calibration_sha256_by_token"][token] = (
                clock_row["calibration_sha256"])

            mapping = copy.deepcopy(processor_fixture.mapping)
            mapping["typed_batch_sha256"] = (
                inputs["expected_processor_batch_sha256_by_token"][token])
            mapping["sampling_metadata_sha256"] = (
                inputs["expected_sampling_metadata_sha256_by_token"][token])
            mapping["video_grid_thw"] = inputs[
                "expected_processor_groups_by_token"][token]["video_grid_thw"]
            groups = inputs["expected_processor_groups_by_token"][token]
            mapping["interval_boundary_guard_rational_seconds"] = (
                clock_row["calibration_binding_payload"]["guard_pts_seconds"])
            mapping["evidence_group_indices"] = groups["evidence_group_indices"]
            mapping["control_group_indices"] = groups["control_group_indices"]
            mapping["groups"] = [{"token_positions": positions}
                                  for positions in groups["temporal_group_token_positions"]]
            processor_rows.append(processor_binding_row(
                review_token=token, source_video_sha256=video_sha,
                mapping=mapping))
    finally:
        clock_fixture.tearDown()
    return clock_rows, processor_rows


def composed_inputs():
    inputs = _replace_tokens(synthetic_mapping_inputs())
    clocks, processors = _actual_adapter_rows(inputs)
    inputs["clock_rows"] = clocks
    inputs["processor_rows"] = processors
    return inputs


class QtaSyntheticAdapterCompositionTests(unittest.TestCase):
    def denied(self, report):
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["branch_forwards"], 0)
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_real_clock_and_processor_row_adapters_compose_for_three_synthetic_rows(self):
        inputs = composed_inputs()
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "structural_row_binding_passed_unvalidated")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 3)
        self.assertEqual(len(report["row_binding_sha256"]), 3)
        self.assertEqual(len(report["bundle_binding_sha256"]), 64)
        self.assertFalse(report["clock_joint_freeze_authenticated"])
        self.assertEqual(report["prediction_link_authentication"],
                         "self_asserted_unauthenticated")
        self.denied(report)
        public = json.dumps(report, sort_keys=True)
        for token in inputs["expected_cohort_tokens"]:
            self.assertNotIn(token, public)
        for video_sha in inputs["expected_video_sha256_by_token"].values():
            self.assertNotIn(video_sha, public)

    def test_one_processor_batch_mismatch_blocks_whole_three_row_join(self):
        inputs = composed_inputs()
        inputs["processor_rows"][1]["typed_batch_sha256"] = "f" * 64
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self.denied(report)

    def test_clock_video_mismatch_blocks_whole_three_row_join(self):
        inputs = composed_inputs()
        inputs["clock_rows"][2]["source_video_sha256"] = "e" * 64
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self.denied(report)


if __name__ == "__main__":
    unittest.main()
