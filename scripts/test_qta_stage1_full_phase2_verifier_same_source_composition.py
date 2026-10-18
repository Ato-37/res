from __future__ import annotations

import hashlib
import json
import struct
import unittest

from qta_attention_call_receipts import (
    build_qta_attention_call_receipt,
    build_qta_attention_call_receipt_index,
)
from qta_attention_capture_sidecars import build_qta_capture_sidecar_index
from qta_phase2_plan_manifest import verify_qta_phase2_plan_candidate
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
from test_qta_stage1_phase2_plan_same_source_composition import (
    Stage1Phase2PlanSameSourceCompositionTests,
    _canonical,
    _sha,
)


class Stage1FullPhase2VerifierSameSourceCompositionTests(
        Stage1Phase2PlanSameSourceCompositionTests):
    """Run the full signed phase-2 verifier on the same Stage-1/QA fixture.

    Attention arrays and call reports below are fabricated CPU structures. In
    particular, the call report's ``forward_calls=1`` is not a model execution.
    """

    def setUp(self):
        from test_qta_phase_freeze_receipts import QtaPhaseFreezeReceiptTests
        from test_qta_stage1_baseline_prediction_link_composition import (
            Stage1BaselinePredictionLinkCompositionTests,
        )

        self.source = Stage1BaselinePredictionLinkCompositionTests()
        self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        self.pins = self.source.pins
        self.tokens = self.pins["expected_cohort_tokens"]
        self.pins_raw = _canonical(self.pins)

        # Freeze the exact signed runtime descriptor before baseline creation;
        # baseline score bindings must hash these exact bytes.
        self.phase = QtaPhaseFreezeReceiptTests()
        self.phase.setUp()
        self.addCleanup(self.phase.tearDown)
        self.source.run_digests["runtime_sha256"] = _sha(
            self.phase.p1_artifacts["sampler_model_runtime"])
        source_input = json.loads(self.source.input_raw)
        source_input["run_digests"] = self.source.run_digests
        self.source.input_raw = _canonical(source_input)

        self.baseline = self.source._build_baseline()
        self.baseline_raw = self.baseline["baseline_manifest_raw"]
        self.baseline_manifest = json.loads(self.baseline_raw)
        self.preflight, self.preflight_raw, self.preflight_inputs = (
            self._build_same_source_synthetic_preflight())
        self.records = [
            {"record_token": _sha(token.encode("ascii")),
             "processor_batch_sha256": row["score_binding"][
                 "processor_batch_sha256"]}
            for token, row in zip(
                self.tokens, self.baseline_manifest["records"], strict=True)
        ]
        self.phase1_artifacts = dict(self.phase.p1_artifacts)
        self.phase1_artifacts["stage1_review_and_lineage"] = self.pins_raw
        self.phase1_artifacts["atomic_cohort_preflight"] = self.preflight_raw
        # Recompute the pin from this exact Stage-1 bytestring, rather than
        # retaining the unrelated three-row fixture's external-pin artifact.
        from qta_phase_freeze_receipts import build_external_cohort_pin_candidate
        external_pin = build_external_cohort_pin_candidate(self.pins_raw)
        self.phase1_artifacts["external_cohort_pin"] = external_pin
        self.phase.expected_external_cohort_pin_sha256 = _sha(external_pin)
        self.phase1_receipt_raw = self.phase._create(
            "pre_baseline", self.records, self.phase1_artifacts)
        self.phase._verify(self.phase1_receipt_raw, "pre_baseline",
                           self.phase1_artifacts)
        self.plans = self._build_plans()
        self.plan_raw = self._create_candidate()
        self.p2_artifacts, self.p2_receipt_raw, self.sidecars = (
            self._build_signed_phase2_artifacts())

    def _build_plans(self):
        """Use exactly representable synthetic masses for capture arithmetic."""
        plans = []
        rows = {row["review_token"]: row
                for row in self.baseline_manifest["records"]}
        for index, token in enumerate(self.tokens):
            baseline_row = rows[token]
            binding = baseline_row["score_binding"]
            groups = self.preflight_inputs[
                "expected_processor_groups_by_token"][token]
            temporal_groups = groups["temporal_group_token_positions"]
            # Binary fractions make the fabricated float32 attention exactly
            # reproduce its declared group-mass bytes through torch mean/sum.
            distribution = [0.125, 0.5, 0.125, 0.125, 0.125]
            if len(temporal_groups) != len(distribution):
                raise AssertionError("synthetic_temporal_group_count_changed")
            layer_masses = [list(distribution)
                            for _ in FROZEN_QTA_LAYERS_1_BASED]
            baseline_row_sha = _sha(_canonical(baseline_row))
            row_binding_sha = self.preflight["row_binding_sha256"][index]
            batch_sha = binding["processor_batch_sha256"]
            selection = build_qta_selection_provenance(
                record_token=_sha(token.encode("ascii")),
                opaque_review_token=token,
                row_binding_sha256=row_binding_sha,
                typed_batch_sha256=batch_sha,
                score_binding_sha256=baseline_row["score_binding_sha256"],
                baseline_pq_row_sha256=baseline_row_sha,
                prompt_sha256=binding["prompt_sha256"],
                input_ids_sha256=binding["input_ids_sha256"],
                input_length=binding["input_length"],
                score_position=binding["score_position"],
                question_positions=binding["question_positions"],
                temporal_groups=temporal_groups,
                selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
                selector_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
                layer_group_masses=layer_masses,
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

    def _build_signed_phase2_artifacts(self):
        manifest = json.loads(self.plan_raw)
        capture_entries = []
        for index, (token, plan_row, baseline_row) in enumerate(zip(
                self.tokens, manifest["records"], self.baseline_manifest["records"],
                strict=True)):
            plan = plan_row["plan"]
            selection = plan["selection_provenance"]
            binding = baseline_row["score_binding"]
            groups = plan["temporal_groups"]
            input_length = binding["input_length"]
            qpos = binding["question_positions"]
            masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                       for value in row]
                      for row in selection["layer_group_mass_f32_be_hex"]]
            group_positions = {p for group in groups for p in group}
            residual_positions = [p for p in range(input_length)
                                  if p not in group_positions and p < qpos[0]]
            attention = []
            for layer_masses in masses:
                heads = []
                for _head in range(2):
                    queries = []
                    for _query in qpos:
                        values = [0.0] * input_length
                        for group_index, group in enumerate(groups):
                            values[group[0]] = layer_masses[group_index]
                        # Float32 mass encoding can make a mathematically zero
                        # residual a few ulps negative; the sidecar contract
                        # rejects even tiny negative attention values.
                        residual = max(0.0, 1.0 - sum(layer_masses))
                        if residual_positions:
                            values[residual_positions[0]] = residual
                        elif abs(residual) > 1e-6:
                            raise AssertionError("synthetic_attention_residual_unplaceable")
                        queries.append(values)
                    heads.append(queries)
                attention.append(heads)
            capture_entries.append({
                "binding": {
                    "record_token": _sha(token.encode("ascii")),
                    "row_binding_sha256": plan_row["row_binding_sha256"],
                    "typed_batch_sha256": plan_row["processor_batch_sha256"],
                    "baseline_pq_row_sha256": plan_row["baseline_pq_row_sha256"],
                    "prompt_sha256": binding["prompt_sha256"],
                    "input_ids_sha256": binding["input_ids_sha256"],
                    "model_sha256": binding["run_digests"]["model_sha256"],
                    "processor_sha256": binding["run_digests"]["processor_sha256"],
                    "tokenizer_sha256": binding["run_digests"]["tokenizer_sha256"],
                    "runtime_sha256": self.source.run_digests["runtime_sha256"],
                    "question_positions": qpos,
                    "temporal_groups": groups,
                },
                "attention": attention,
                "input_length": input_length,
            })
        capture_index_raw, sidecars = build_qta_capture_sidecar_index(
            entries=capture_entries)
        call_receipts = []
        for row in json.loads(capture_index_raw)["records"]:
            fields = {key: row[key] for key in (
                "record_token", "runtime_sha256", "input_length",
                "question_positions", "attention_bytes_sha256")}
            classes = {str(layer): "synthetic.FakeSelfAttention"
                       for layer in FROZEN_QTA_LAYERS_1_BASED}
            fabricated_report = {
                "status": "one_forward_attention_rows_captured_non_authorizing",
                "forward_calls": 1,
                "layer_calls": {str(layer): 1
                                for layer in FROZEN_QTA_LAYERS_1_BASED},
                "layer_indices_1_based": list(FROZEN_QTA_LAYERS_1_BASED),
                "layer_module_classes": classes,
                "layer_module_paths": {
                    str(layer): ("model.model.language_model.layers["
                                 f"{layer - 1}].self_attn")
                    for layer in FROZEN_QTA_LAYERS_1_BASED},
                "question_query_count": len(fields["question_positions"]),
                "input_length": fields["input_length"],
                "attention_row_sum_max_abs_deviation": 0.0,
                "attention_source_dtype": row["attention_source_dtype"],
                "attention_row_sum_tolerance": 2e-4,
                "attention_rows_satisfy_source_dtype_normalization_tolerance": True,
                "attention_shape": row["attention_shape"],
                "attention_source_dtypes": {
                    str(layer): row["attention_source_dtype"]
                    for layer in FROZEN_QTA_LAYERS_1_BASED},
                "forward_source_authenticated": False,
                "attention_capture_authenticated": False,
                "scoring_authorized": False,
            }
            call_receipts.append(build_qta_attention_call_receipt(
                report=fabricated_report, record_token=fields["record_token"],
                runtime_sha256=fields["runtime_sha256"],
                input_length=fields["input_length"],
                question_positions=fields["question_positions"],
                attention_bytes_sha256=fields["attention_bytes_sha256"],
                expected_module_classes=classes))
        call_index_raw = build_qta_attention_call_receipt_index(
            receipts=call_receipts)
        p2_artifacts = dict(self.phase.p2_artifacts)
        p2_artifacts.update({
            "phase1_receipt": self.phase1_receipt_raw,
            "baseline_pq_manifest": self.baseline_raw,
            "stage2_prediction_link_review": _canonical({
                "status": "synthetic_same_baseline_review_unvalidated",
                "baseline_pq_manifest_sha256": _sha(self.baseline_raw),
                "non_authorizing": True,
            }),
            "nine_branch_plan_manifest": self.plan_raw,
            "attention_capture_index": capture_index_raw,
            "attention_call_receipts": call_index_raw,
        })
        p2 = self.phase._create(
            "post_baseline", self.records, p2_artifacts,
            prior_phase1_receipt_raw=self.phase1_receipt_raw)
        return p2_artifacts, p2, sidecars

    def _verify(self, *, sidecars=None):
        return verify_qta_phase2_plan_candidate(
            phase1_receipt_raw=self.phase1_receipt_raw,
            phase1_artifact_bytes=self.phase1_artifacts,
            phase2_receipt_raw=self.p2_receipt_raw,
            phase2_artifact_bytes=self.p2_artifacts,
            attention_capture_sidecars=self.sidecars if sidecars is None else sidecars,
            integrity_roster_raw=self.phase.integrity,
            release_roster_raw=self.phase.release,
            root_public_key_pem=self.phase.keys["root"][1],
            expected_root_public_der_sha256=self.phase.root_pin,
            expected_external_cohort_pin_sha256=(
                self.phase.expected_external_cohort_pin_sha256),
        )

    def test_same_source_signed_candidate_passes_full_verifier_non_authorizing(self):
        report = self._verify()
        self.assertEqual(report["status"], "phase2_plan_manifest_bound_unvalidated")
        self.assertEqual(report["validated_plan_rows"], 3)
        self.assertEqual(report["capture_sidecar_records_validated"], 3)
        self.assertEqual(report["capture_call_receipts_validated"], 3)
        self.assertTrue(report["capture_call_receipts_joined_to_sidecar_payloads"])
        self.assertTrue(report["phase1_receipt_signature_valid"])
        self.assertTrue(report["phase2_receipt_signature_valid"])
        self.assertTrue(report["phase1_receipt_recursively_verified"])
        self.assertTrue(report["capture_sidecars_joined_to_selection_masses"])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertFalse(report["plan_semantics_authenticated"])
        self.assertIs(report["model_load_permitted"], False)
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["authorized_forwards"], 0)
        phase2_receipt = json.loads(self.p2_receipt_raw)
        plan = json.loads(self.plan_raw)
        for candidate in (phase2_receipt, plan):
            self.assertIs(candidate["model_load_permitted"], False)
            self.assertIs(candidate["scoring_authorized"], False)
            self.assertEqual(candidate["execution_authorization"], "not_granted")

    def test_full_verifier_rejects_tampered_sidecar_bytes(self):
        changed = dict(self.sidecars)
        key = next(iter(changed))
        changed[key] = changed[key] + b"tampered"
        with self.assertRaisesRegex(ValueError, "capture_sidecar"):
            self._verify(sidecars=changed)


if __name__ == "__main__":
    unittest.main()
