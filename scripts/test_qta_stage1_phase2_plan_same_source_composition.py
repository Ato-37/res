from __future__ import annotations

import hashlib
import json
import unittest
from fractions import Fraction

from annotation_clock_calibration import map_annotation_intervals_to_pts
from qta_phase_freeze_receipts import build_external_cohort_pin_candidate
from qta_phase2_plan_manifest import create_qta_phase2_plan_manifest_candidate
from qta_nine_branch_plan import build_qta_nine_branch_plan
from qta_selection_provenance import (
    FROZEN_SELECTOR_SOURCE_SHA256,
    build_qta_selection_provenance,
)
from qta_three_row_plan_aggregate import (
    FROZEN_QTA_LAYERS_1_BASED,
    FROZEN_QTA_LOGIT_BIAS,
    FROZEN_QTA_SHAM_SEED,
)
from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class Stage1Phase2PlanSameSourceCompositionTests(unittest.TestCase):
    """Use one Stage-1/QA-derived baseline throughout the phase-2 plan join."""

    def setUp(self):
        from test_qta_phase_freeze_receipts import QtaPhaseFreezeReceiptTests

        # Reuse the preceding same-source composition. It invokes the actual
        # Stage-1 adapter and binds that adapter's exact QA inputs into baseline
        # v3; no second source or baseline fixture is introduced here.
        from test_qta_stage1_baseline_prediction_link_composition import (
            Stage1BaselinePredictionLinkCompositionTests,
        )

        self.source = Stage1BaselinePredictionLinkCompositionTests()
        self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        self.pins = self.source.pins
        self.tokens = self.pins["expected_cohort_tokens"]
        self.pins_raw = _canonical(self.pins)
        self.baseline = self.source._build_baseline()
        self.baseline_raw = self.baseline["baseline_manifest_raw"]
        self.baseline_manifest = json.loads(self.baseline_raw)

        self.phase = QtaPhaseFreezeReceiptTests()
        self.phase.setUp()
        self.addCleanup(self.phase.tearDown)

        self.preflight, self.preflight_raw, self.preflight_inputs = (
            self._build_same_source_synthetic_preflight())
        self.records = [
            {"record_token": _sha(token.encode("ascii")),
             "processor_batch_sha256": baseline_row["score_binding"][
                 "processor_batch_sha256"]}
            for token, baseline_row in zip(
                self.tokens, self.baseline_manifest["records"], strict=True)
        ]
        self.phase1_artifacts = dict(self.phase.p1_artifacts)
        self.phase1_artifacts["stage1_review_and_lineage"] = self.pins_raw
        self.phase1_artifacts["atomic_cohort_preflight"] = self.preflight_raw
        external_pin = build_external_cohort_pin_candidate(self.pins_raw)
        self.phase1_artifacts["external_cohort_pin"] = external_pin
        self.phase.expected_external_cohort_pin_sha256 = _sha(external_pin)
        self.phase1_receipt_raw = self.phase._create(
            "pre_baseline", self.records, self.phase1_artifacts)
        receipt_report = self.phase._verify(
            self.phase1_receipt_raw, "pre_baseline", self.phase1_artifacts)
        self.assertEqual(receipt_report["receipt_signature_valid"], True)
        self.assertEqual(receipt_report["execution_authorization"], "not_granted")

        self.plans = self._build_plans()

    def _build_same_source_synthetic_preflight(self):
        """Make structural rows from Stage-1 evidence and candidate batch pins.

        The annotation intervals and source digests come from the Stage-1
        fixture. Clock calibration and video-grid layout are explicitly
        synthetic scaffolding, not measurements from a player or processor.
        """
        stage1_review = json.loads(self.source.adapter.stage1_inputs["review_raw"])
        review_rows_by_token = {
            row["review_token"]: row for row in stage1_review["records"]}
        baseline_by_token = {
            row["review_token"]: row for row in self.baseline_manifest["records"]}
        stage1_by_token = {row["review_token"]: row
                           for row in self.pins["source_rows"]}
        batch_by_token = {
            token: baseline_by_token[token]["score_binding"][
                "processor_batch_sha256"] for token in self.tokens}

        receipt_sha = self.pins["expected_video_claim_review_receipt_sha256"]
        review_payload_sha = self.pins["expected_review_payload_sha256"]
        expected_video = {}
        expected_batch = {}
        expected_sampling = {}
        expected_groups = {}
        expected_calibration = {}
        expected_source = {}
        reviews, clocks, processors, baselines = [], [], [], []
        for token in self.tokens:
            stage1 = stage1_by_token[token]
            exported = review_rows_by_token[token]["review"]
            baseline_row = baseline_by_token[token]
            score_binding = baseline_row["score_binding"]
            intervals = exported["evidence_intervals_seconds"]
            video_sha = stage1["source_video_sha256"]
            self.assertEqual(exported["support_decision"],
                             stage1["video_support_decision"])
            self.assertEqual(score_binding["source_video_sha256"], video_sha)
            self.assertEqual(score_binding["reviewed_intervals_sha256"],
                             stage1["reviewed_intervals_sha256"])

            # Identity clock mapping is a declared synthetic scaffold. The
            # source interval values remain the exact Stage-1 review values.
            calibration_payload = {
                "source_video_sha256": video_sha,
                "slope": "1", "intercept": "0",
                "anchor_pts_min_seconds": "0",
                "anchor_pts_max_seconds": str(max(
                    Fraction(str(pair[1])) for pair in intervals) + 1),
                "guard_pts_seconds": "1/20",
            }
            calibration_sha = _sha(json.dumps(
                calibration_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("ascii"))
            mapped = map_annotation_intervals_to_pts(intervals, "1", "0")
            sampling_sha = _sha(f"synthetic-sampling-layout:{token}".encode("ascii"))
            temporal_groups = [[index] for index in range(5)]
            evidence_groups, control_groups = [1], [3]
            grid = [5, 1, 1]
            expected_video[token] = video_sha
            expected_batch[token] = batch_by_token[token]
            expected_sampling[token] = sampling_sha
            expected_groups[token] = {
                "evidence_group_indices": evidence_groups,
                "control_group_indices": control_groups,
                "video_grid_thw": grid,
                "temporal_group_token_positions": temporal_groups,
            }
            expected_calibration[token] = calibration_sha
            expected_source[token] = {
                field: self.pins["expected_baseline_source_sha256_by_token"][token][field]
                for field in ("question_sha256", "options_sha256", "claim_sha256")
            }
            reviews.append({
                "review_token": token,
                "source_video_sha256": video_sha,
                "review_payload_sha256": review_payload_sha,
                "support_decision": stage1["video_support_decision"],
                "evidence_intervals_seconds": intervals,
            })
            clocks.append({
                "review_token": token,
                "source_video_sha256": video_sha,
                "video_claim_review_receipt_sha256": receipt_sha,
                "calibration_sha256": calibration_sha,
                "calibration_binding_payload": calibration_payload,
                "slope": "1", "intercept": "0",
                "anchor_pts_min_seconds": "0",
                "anchor_pts_max_seconds": calibration_payload[
                    "anchor_pts_max_seconds"],
                "guard_pts_seconds": "1/20",
            })
            processors.append({
                "review_token": token,
                "source_video_sha256": video_sha,
                "typed_batch_sha256": batch_by_token[token],
                "sampling_metadata_sha256": sampling_sha,
                "video_grid_thw": grid,
                "reviewed_intervals_rational_seconds": mapped,
                "interval_boundary_guard_rational_seconds": "1/20",
                "processor_uniform_grid_assignment_check":
                    "passed_actual_pts_and_scalar_fps_group_sets_equal",
                "evidence_group_indices": evidence_groups,
                "control_group_indices": control_groups,
                "temporal_group_token_positions": temporal_groups,
            })
            baselines.append({
                "review_token": token,
                "source_video_sha256": video_sha,
                "video_claim_review_receipt_sha256": receipt_sha,
                "video_support_decision": stage1["video_support_decision"],
                "reviewed_intervals_sha256": stage1[
                    "reviewed_intervals_sha256"],
                "question_sha256": score_binding["question_sha256"],
                "options_sha256": score_binding["options_sha256"],
                "claim_sha256": score_binding["claim_sha256"],
            })

        inputs = {
            "expected_cohort_tokens": list(self.tokens),
            "expected_video_review_receipt_sha256": receipt_sha,
            "expected_review_payload_sha256": review_payload_sha,
            "expected_video_sha256_by_token": expected_video,
            "expected_processor_batch_sha256_by_token": expected_batch,
            "expected_sampling_metadata_sha256_by_token": expected_sampling,
            "expected_processor_groups_by_token": expected_groups,
            "expected_calibration_sha256_by_token": expected_calibration,
            "expected_baseline_source_sha256_by_token": expected_source,
            "review_rows": reviews,
            "clock_rows": clocks,
            "processor_rows": processors,
            "baseline_rows": baselines,
        }
        report = preflight_synthetic_qta_cohort(**inputs)
        raw = _canonical(report)
        return report, raw, inputs

    def _build_plans(self):
        plans = []
        rows_by_token = {row["review_token"]: row
                         for row in self.baseline_manifest["records"]}
        for index, token in enumerate(self.tokens):
            baseline_row = rows_by_token[token]
            score_binding = baseline_row["score_binding"]
            groups = self.preflight_inputs[
                "expected_processor_groups_by_token"][token]
            temporal_groups = groups["temporal_group_token_positions"]
            layer_group_masses = [
                [0.01 * (group + 1) for group in range(len(temporal_groups))]
                for _ in FROZEN_QTA_LAYERS_1_BASED]
            for mass_row in layer_group_masses:
                mass_row[1] = 1.0 - sum(
                    value for group, value in enumerate(mass_row) if group != 1)
            baseline_row_sha = _sha(_canonical(baseline_row))
            row_binding_sha = self.preflight["row_binding_sha256"][index]
            batch_sha = score_binding["processor_batch_sha256"]
            selection = build_qta_selection_provenance(
                record_token=_sha(token.encode("ascii")),
                opaque_review_token=token,
                row_binding_sha256=row_binding_sha,
                typed_batch_sha256=batch_sha,
                score_binding_sha256=baseline_row["score_binding_sha256"],
                baseline_pq_row_sha256=baseline_row_sha,
                prompt_sha256=score_binding["prompt_sha256"],
                input_ids_sha256=score_binding["input_ids_sha256"],
                input_length=score_binding["input_length"],
                score_position=score_binding["score_position"],
                question_positions=score_binding["question_positions"],
                temporal_groups=temporal_groups,
                selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
                selector_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
                layer_group_masses=layer_group_masses,
                sham_seed=FROZEN_QTA_SHAM_SEED,
            )
            plans.append(build_qta_nine_branch_plan(
                row_binding_sha256=row_binding_sha,
                temporal_groups=temporal_groups,
                target_group_indices=selection["target_group_indices"],
                sham_group_indices=selection["sham_group_indices"],
                evidence_group_indices=groups["evidence_group_indices"],
                control_group_indices=groups["control_group_indices"],
                typed_batch_sha256=batch_sha,
                score_binding_sha256=baseline_row["score_binding_sha256"],
                intervention_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
                logit_bias=FROZEN_QTA_LOGIT_BIAS,
                sham_seed=FROZEN_QTA_SHAM_SEED,
                selection_provenance=selection,
            ))
        return plans

    def _create_candidate(self, *, phase1_records=None, phase1_source_raw=None,
                          baseline_raw=None):
        return create_qta_phase2_plan_manifest_candidate(
            phase1_receipt_raw=self.phase1_receipt_raw,
            phase1_record_bindings=(self.records if phase1_records is None
                                    else phase1_records),
            phase1_preflight_raw=self.preflight_raw,
            phase1_stage1_source_pins_raw=(self.pins_raw if phase1_source_raw is None
                                           else phase1_source_raw),
            baseline_pq_manifest_raw=(self.baseline_raw if baseline_raw is None
                                      else baseline_raw),
            plans=self.plans,
        )

    def test_same_source_baseline_binds_into_phase2_plan_candidate(self):
        self.assertEqual(self.preflight["status"],
                         "structural_row_binding_passed_unvalidated")
        self.assertEqual(self.preflight["verified_structural_rows"], 3)
        manifest_raw = self._create_candidate()
        manifest = json.loads(manifest_raw)
        self.assertEqual(manifest["purpose"],
                         "qta_phase2_nine_branch_plan_manifest_v2")
        self.assertEqual(manifest["phase1_receipt_sha256"],
                         _sha(self.phase1_receipt_raw))
        self.assertEqual(manifest["baseline_pq_manifest_sha256"],
                         _sha(self.baseline_raw))
        self.assertEqual(len(manifest["records"]), 3)
        baseline_rows = self.baseline_manifest["records"]
        for index, (token, plan_row, baseline_row, record) in enumerate(zip(
                self.tokens, manifest["records"], baseline_rows, self.records,
                strict=True)):
            self.assertEqual(record["record_token"], _sha(token.encode("ascii")))
            self.assertEqual(plan_row["record_token"], record["record_token"])
            self.assertEqual(plan_row["processor_batch_sha256"],
                             baseline_row["score_binding"]["processor_batch_sha256"])
            self.assertEqual(plan_row["processor_batch_sha256"],
                             record["processor_batch_sha256"])
            self.assertEqual(plan_row["baseline_pq_row_sha256"],
                             _sha(_canonical(baseline_row)))
            self.assertEqual(plan_row["baseline_p_index"], baseline_row["p_index"])
            self.assertEqual(plan_row["baseline_q_index"], baseline_row["q_index"])
            self.assertEqual(plan_row["score_binding_sha256"],
                             baseline_row["score_binding_sha256"])
        self.assertEqual(manifest["execution_authorization"], "not_granted")
        self.assertIs(manifest["model_load_permitted"], False)
        self.assertIs(manifest["scoring_authorized"], False)
        self.assertEqual(manifest["authorized_forwards"], 0)

    def test_stage1_source_mutations_and_baseline_or_order_tamper_are_rejected(self):
        for field in ("reviewed_intervals_sha256", "source_video_sha256", "claim_sha256"):
            with self.subTest(stage1_field=field):
                changed = json.loads(self.pins_raw)
                changed["source_rows"][0][field] = "0" * 64
                with self.assertRaisesRegex(ValueError,
                                            "phase2_plan_stage1_source_pin_mismatch"):
                    self._create_candidate(phase1_source_raw=_canonical(changed))

        # Reordering baseline rows changes which opaque token/source is joined
        # to each ordered Phase-1 receipt row; the candidate must reject it.
        reordered = dict(self.baseline_manifest)
        reordered["records"] = list(reversed(reordered["records"]))
        with self.assertRaisesRegex(ValueError,
                                    "phase2_plan_record_or_batch_mismatch"):
            self._create_candidate(baseline_raw=_canonical(reordered))

        changed_baseline = json.loads(self.baseline_raw)
        changed_baseline["records"][0]["claim"] += " tampered"
        with self.assertRaisesRegex(ValueError,
                                    "baseline_pq_score_binding_source_mismatch"):
            self._create_candidate(baseline_raw=_canonical(changed_baseline))

        changed_records = [dict(row) for row in self.records]
        changed_records[0] = {**changed_records[0],
                              "processor_batch_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError,
                                    "phase2_plan_binding_mismatch"):
            self._create_candidate(phase1_records=changed_records)


if __name__ == "__main__":
    unittest.main()
