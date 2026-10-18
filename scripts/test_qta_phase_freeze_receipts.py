from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
import struct
from pathlib import Path

from nextgqa_pre_review_integrity_receipt import (
    ROLES, _ROSTER_DOMAIN as _INTEGRITY_DOMAIN, _canonical, _public_der,
    _sha, _sign, ReceiptError,
)
from audit_nextgqa_review_release_order import RELEASE_ROLES, _ROSTER_DOMAIN as _RELEASE_DOMAIN
from qta_phase_freeze_receipts import (
    _PHASE_DOMAINS,
    PHASE_ARTIFACT_ROLES, PHASE_BRANCHES, create_phase_freeze_receipt,
    verify_phase_freeze_receipt,
)
from qta_nine_branch_plan import BRANCH_ORDER, _plan_digest
from qta_phase2_plan_manifest import (
    create_qta_phase2_plan_manifest_candidate,
    verify_qta_phase2_plan_candidate,
)
from qta_attention_capture_sidecars import build_qta_capture_sidecar_index
from qta_nine_branch_plan import build_qta_nine_branch_plan
from qta_three_row_plan_aggregate import (
    FROZEN_QTA_LAYERS_1_BASED, FROZEN_QTA_LOGIT_BIAS, FROZEN_QTA_SHAM_SEED,
)
from qta_selection_provenance import (
    FROZEN_SELECTOR_SOURCE_SHA256, build_qta_selection_provenance,
)
from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort
import test_qta_synthetic_adapter_composition as synthetic_composition


class QtaPhaseFreezeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.keys = {}
        for name in ("root", "owner", "capture", "claim", "frame", "prediction", "witness"):
            private = self.root / f"{name}.private.pem"
            public = self.root / f"{name}.public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                           check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                           check=True, capture_output=True)
            self.keys[name] = (private.read_bytes(), public.read_bytes())
        self.root_pin = _sha(_public_der(self.keys["root"][1]))
        integrity_entries = [self._entry(role, name, name) for role, name in (
            ("freeze_owner", "owner"), ("claim_reviewer", "claim"),
            ("frame_reviewer", "frame"))]
        self.integrity = self._signed_roster(
            "nextgqa_pre_review_trust_roster_v1", "integrity-01",
            integrity_entries, _INTEGRITY_DOMAIN)
        release_entries = [self._entry(role, name, name) for role, name in (
            ("freeze_owner", "owner"), ("custodian", "owner"),
            ("capture_operator", "capture"), ("claim_reviewer", "claim"),
            ("frame_reviewer", "frame"), ("prediction_link_reviewer", "prediction"),
            ("release_witness", "witness"))]
        self.release = self._signed_roster(
            "nextgqa_review_release_roster_v1", "release-01",
            release_entries, _RELEASE_DOMAIN)
        self.records = [{"record_token": f"{i + 1:064x}",
                         "processor_batch_sha256": f"{i + 1:064x}"}
                        for i in range(3)]
        self.p1_artifacts = {role: ("fixture:" + role).encode()
                             for role in PHASE_ARTIFACT_ROLES["pre_baseline"]}
        self.p1 = self._create("pre_baseline", self.records, self.p1_artifacts)
        self.p2_artifacts = {
            "phase1_receipt": self.p1,
            "baseline_pq_manifest": b"fixture:baseline-pq",
            "stage2_prediction_link_review": b"fixture:stage2-review",
            "nine_branch_plan_manifest": b"fixture:branch-plans",
            "condition_scorer_source_closure": b"fixture:scorer-closure",
            "attention_capture_index": b"fixture:attention-capture-index",
        }
        self.p2 = self._create("post_baseline", self.records, self.p2_artifacts,
                               prior_phase1_receipt_raw=self.p1)

    def tearDown(self):
        self.temp.cleanup()

    def _entry(self, role, name, key_id):
        public = self.keys[name][1]
        return {"role": role, "key_id": key_id,
                "public_key_pem_base64": base64.b64encode(public).decode(),
                "public_key_der_sha256": _sha(_public_der(public))}

    def _signed_roster(self, purpose, roster_id, entries, domain):
        body = {"schema_version": 1, "purpose": purpose,
                "roster_id": roster_id, "keys": entries}
        signature = _sign(self.keys["root"][0], self.keys["root"][1],
                          domain + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64":
                           base64.b64encode(signature).decode()}) + b"\n"

    def _create(self, phase, records, artifacts, *, prior_phase1_receipt_raw=None,
                owner_key="owner"):
        return create_phase_freeze_receipt(
            phase=phase, freeze_id="synthetic-freeze-01", record_rows=records,
            artifact_bytes=artifacts, integrity_roster_raw=self.integrity,
            release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
            freeze_owner_private_key_pem=self.keys[owner_key][0],
            prior_phase1_receipt_raw=prior_phase1_receipt_raw)

    def _verify(self, receipt, phase, artifacts, **kwargs):
        return verify_phase_freeze_receipt(
            receipt_raw=receipt, phase=phase, artifact_bytes=artifacts,
            integrity_roster_raw=self.integrity, release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin, **kwargs)

    def test_phase1_signs_exact_synthetic_artifacts_and_denies_execution(self):
        self.assertEqual(PHASE_BRANCHES["pre_baseline"], BRANCH_ORDER[:1])
        self.assertEqual(PHASE_BRANCHES["post_baseline"], BRANCH_ORDER[1:])
        report = self._verify(self.p1, "pre_baseline", self.p1_artifacts)
        self.assertEqual(report["status"], "verified_signed_byte_bindings_non_authorizing")
        self.assertEqual(report["requested_rows"], 3)
        self.assertEqual(report["scheduled_branches"], 3)
        self.assertTrue(report["receipt_signature_valid"])
        self.assertTrue(report["artifact_bytes_match"])
        self.assertFalse(report["human_identity_verified"])
        self.assertFalse(report["semantic_review_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)
        signed_body = json.loads(self.p1)
        self.assertEqual(signed_body["status"], "signed_non_authorizing_candidate")
        self.assertEqual(signed_body["execution_authorization"], "not_granted")
        self.assertFalse(signed_body["model_load_permitted"])
        self.assertFalse(signed_body["scoring_authorized"])
        self.assertFalse(signed_body["phase_authorized"])
        self.assertEqual(signed_body["authorized_forwards"], 0)
        self.assertEqual(signed_body["planned_forward_count"], 3)

    def test_phase2_recursively_verifies_and_binds_phase1(self):
        report = self._verify(
            self.p2, "post_baseline", self.p2_artifacts,
            prior_phase1_receipt_raw=self.p1,
            prior_phase1_artifact_bytes=self.p1_artifacts)
        self.assertEqual(report["scheduled_branches"], 24)
        self.assertTrue(report["prior_phase1_receipt_valid"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)
        signed_body = json.loads(self.p2)
        self.assertEqual(signed_body["status"], "signed_non_authorizing_candidate")
        self.assertEqual(signed_body["planned_forward_count"], 24)
        self.assertEqual(signed_body["execution_authorization"], "not_granted")
        self.assertFalse(signed_body["model_load_permitted"])
        self.assertFalse(signed_body["scoring_authorized"])
        self.assertFalse(signed_body["phase_authorized"])
        self.assertEqual(signed_body["authorized_forwards"], 0)

    def test_artifact_byte_mutation_fails_verification(self):
        changed = dict(self.p1_artifacts)
        changed["clock_frame_evidence"] += b"tampered"
        with self.assertRaisesRegex(ReceiptError, "phase_receipt_binding_mismatch"):
            self._verify(self.p1, "pre_baseline", changed)

    def test_phase2_cannot_replace_prior_receipt_or_its_review_artifacts(self):
        changed = dict(self.p2_artifacts)
        changed["phase1_receipt"] = b"other receipt"
        with self.assertRaisesRegex(ReceiptError, "phase1_receipt_required_or_mismatched"):
            self._verify(self.p2, "post_baseline", changed,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)
        prior_inputs = dict(self.p1_artifacts)
        prior_inputs["stage1_review_and_lineage"] += b"changed"
        with self.assertRaises(ReceiptError):
            self._verify(self.p2, "post_baseline", self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=prior_inputs)

    def test_phase2_v2_requires_signed_capture_index_role(self):
        incomplete = dict(self.p2_artifacts)
        incomplete.pop("attention_capture_index")
        with self.assertRaisesRegex(ReceiptError, "phase_artifact_role_set_invalid"):
            self._create("post_baseline", self.records, incomplete,
                          prior_phase1_receipt_raw=self.p1)

    def test_phase2_v2_rejects_v1_domain_downgrade(self):
        changed = json.loads(self.p2)
        changed["schema_version"] = 1
        changed["purpose"] = "qta_two_phase_score_freeze_candidate_v1"
        with self.assertRaisesRegex(ReceiptError,
                                   "phase_receipt_schema_version_invalid"):
            self._verify(_canonical(changed) + b"\n", "post_baseline",
                         self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_phase2_v2_signature_rejects_v1_domain_even_with_matching_key(self):
        changed = json.loads(self.p2)
        body = {key: value for key, value in changed.items()
                if key != "signature_ed25519_base64"}
        changed["signature_ed25519_base64"] = base64.b64encode(_sign(
            self.keys["owner"][0], self.keys["owner"][1],
            _PHASE_DOMAINS["pre_baseline"] + _canonical(body))).decode("ascii")
        with self.assertRaisesRegex(ReceiptError, "phase_receipt_signature_invalid"):
            self._verify(_canonical(changed) + b"\n", "post_baseline",
                         self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_phase2_must_preserve_phase1_record_and_batch_bindings(self):
        changed_records = [dict(row) for row in self.records]
        changed_records[1]["processor_batch_sha256"] = "f" * 64
        changed = self._create("post_baseline", changed_records, self.p2_artifacts,
                               prior_phase1_receipt_raw=self.p1)
        with self.assertRaisesRegex(ReceiptError, "phase2_record_binding_differs_from_phase1"):
            self._verify(changed, "post_baseline", self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_freeze_owner_signature_key_must_match_root_roster(self):
        with self.assertRaisesRegex(ReceiptError, "signing_key_does_not_match_roster"):
            self._create("pre_baseline", self.records, self.p1_artifacts,
                         owner_key="capture")

    def test_receipt_signature_tampering_is_rejected(self):
        obj = json.loads(self.p1)
        obj["signature_ed25519_base64"] = base64.b64encode(b"x" * 64).decode()
        tampered = _canonical(obj) + b"\n"
        with self.assertRaisesRegex(ReceiptError, "phase_receipt_signature_invalid"):
            self._verify(tampered, "pre_baseline", self.p1_artifacts)

    def test_trust_root_pin_mismatch_fails_before_accepting_receipt(self):
        with self.assertRaisesRegex(ReceiptError, "external_trust_root_pin_mismatch"):
            verify_phase_freeze_receipt(
                receipt_raw=self.p1, phase="pre_baseline", artifact_bytes=self.p1_artifacts,
                integrity_roster_raw=self.integrity, release_roster_raw=self.release,
                root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256="0" * 64)

    def test_missing_artifact_role_is_rejected(self):
        incomplete = dict(self.p1_artifacts)
        incomplete.pop("atomic_cohort_preflight")
        with self.assertRaisesRegex(ReceiptError, "phase_artifact_role_set_invalid"):
            self._create("pre_baseline", self.records, incomplete)

    def test_record_tokens_must_be_opaque_fixed_length_hex(self):
        readable = [dict(row) for row in self.records]
        readable[0]["record_token"] = "video_123_question_5"
        with self.assertRaisesRegex(ReceiptError, "phase_record_token_invalid_or_duplicate"):
            self._create("pre_baseline", readable, self.p1_artifacts)

    def test_phase1_verifier_rejects_phase2_only_arguments(self):
        with self.assertRaisesRegex(ReceiptError, "phase1_receipt_not_allowed"):
            self._verify(self.p1, "pre_baseline", self.p1_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_signed_receipts_do_not_echo_protected_artifact_contents(self):
        self.assertNotIn(b"fixture:clock_frame_evidence", self.p1)
        self.assertNotIn(b"fixture:stage2-review", self.p2)
        self.assertNotIn(b"fixture:baseline-pq", self.p2)

    def _phase2_candidate_bundle(self):
        inputs = synthetic_composition._replace_tokens(
            synthetic_composition.synthetic_mapping_inputs())
        clocks, processors = synthetic_composition._actual_adapter_rows(inputs)
        inputs["clock_rows"], inputs["processor_rows"] = clocks, processors
        preflight = preflight_synthetic_qta_cohort(**inputs)
        preflight_raw = json.dumps(preflight, sort_keys=True,
                                   separators=(",", ":")).encode()
        tokens = inputs["expected_cohort_tokens"]
        plans = []
        baseline_rows = []
        for index, token in enumerate(tokens):
            group_data = inputs["expected_processor_groups_by_token"][token]
            batch_sha = inputs["expected_processor_batch_sha256_by_token"][token]
            temporal_groups = group_data["temporal_group_token_positions"]
            input_length = temporal_groups[-1][-1] + 6
            question_positions = list(range(temporal_groups[-1][-1] + 1,
                                            temporal_groups[-1][-1] + 5))
            scores = [-0.1, -3.0, -4.0, -5.0, -6.0]
            score_binding = {
                "schema_version": 2, "review_token": token,
                "answer_label_order": ["A", "B", "C", "D", "E"],
                "answer_token_ids": [10, 11, 12, 13, 14],
                "choice_token_binding_sha256": f"{index + 301:064x}",
                "question_sha256": hashlib.sha256(
                    b"synthetic question").hexdigest(),
                "options_sha256": hashlib.sha256(json.dumps(
                    ["A", "B", "C", "D", "E"], ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "claim_sha256": hashlib.sha256(b"synthetic claim").hexdigest(),
                "source_video_sha256": f"{index + 701:064x}",
                "reviewed_intervals_sha256": f"{index + 201:064x}",
                "video_support_decision": "supports_claim",
                "prompt_sha256": f"{index + 401:064x}",
                "input_ids_sha256": f"{index + 501:064x}",
                "input_length": input_length, "score_position": input_length - 1,
                "score_input_token_id": 3,
                "video_token_id": 99,
                "question_positions": question_positions,
                "question_positions_sha256": hashlib.sha256(json.dumps(
                    question_positions, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "question_position_mapper_sha256": FROZEN_SELECTOR_SOURCE_SHA256,
                "processor_batch_sha256": batch_sha,
                "option_log_probabilities": scores,
                "p_index": 0, "q_index": 1,
                "run_digests": {name: f"{index + 601:064x}" for name in (
                    "protocol_sha256", "model_sha256", "tokenizer_sha256",
                    "scorer_sha256", "processor_sha256")},
            }
            row = {
                "review_token": token, "question": "synthetic question",
                "options": ["A", "B", "C", "D", "E"],
                "claim": "synthetic claim",
                "option_log_probabilities": scores,
                "p_index": 0, "q_index": 1,
                "video_support_decision": "supports_claim",
                "reviewed_intervals_sha256": f"{index + 201:064x}",
                "score_binding": score_binding,
                "score_binding_sha256": hashlib.sha256(json.dumps(
                    score_binding, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
            }
            baseline_rows.append(row)
            baseline_pq_row_sha256 = hashlib.sha256(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            layer_masses = [[0.01 * (group + 1) for group in range(len(temporal_groups))]
                            for _ in FROZEN_QTA_LAYERS_1_BASED]
            for mass_row in layer_masses:
                mass_row[1] = 1.0 - sum(value for group, value in enumerate(mass_row)
                                        if group != 1)
            selection_provenance = build_qta_selection_provenance(
                record_token=hashlib.sha256(token.encode("ascii")).hexdigest(),
                opaque_review_token=token,
                row_binding_sha256=preflight["row_binding_sha256"][index],
                typed_batch_sha256=batch_sha,
                score_binding_sha256=row["score_binding_sha256"],
                baseline_pq_row_sha256=baseline_pq_row_sha256,
                prompt_sha256=score_binding["prompt_sha256"],
                input_ids_sha256=score_binding["input_ids_sha256"],
                input_length=score_binding["input_length"],
                score_position=score_binding["score_position"],
                question_positions=question_positions,
                temporal_groups=temporal_groups,
                selector_source_sha256=FROZEN_SELECTOR_SOURCE_SHA256,
                selector_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
                layer_group_masses=layer_masses,
                sham_seed=FROZEN_QTA_SHAM_SEED)
            plan = build_qta_nine_branch_plan(
                row_binding_sha256=preflight["row_binding_sha256"][index],
                temporal_groups=temporal_groups,
                target_group_indices=selection_provenance["target_group_indices"],
                sham_group_indices=selection_provenance["sham_group_indices"],
                evidence_group_indices=group_data["evidence_group_indices"],
                control_group_indices=group_data["control_group_indices"],
                typed_batch_sha256=batch_sha,
                score_binding_sha256=row["score_binding_sha256"],
                intervention_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
                logit_bias=FROZEN_QTA_LOGIT_BIAS,
                sham_seed=FROZEN_QTA_SHAM_SEED,
                selection_provenance=selection_provenance)
            plans.append(plan)
        baseline_raw = json.dumps({
            "schema_version": 3,
            "purpose": "qta_baseline_prediction_link_manifest_v3",
            "video_claim_review_receipt_sha256": "a" * 64,
            "records": baseline_rows,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        p1_artifacts = dict(self.p1_artifacts)
        p1_artifacts["atomic_cohort_preflight"] = preflight_raw
        p1_artifacts["stage1_review_and_lineage"] = json.dumps({
            "schema_version": 1,
            "purpose": "qta_stage1_source_pins_structurally_derived_non_authorizing",
            "classification": "private_local_only_structural_source_pins",
            "status": "three_row_source_pins_structurally_derived",
            "requested_records": 3, "effective_records": 3,
            "expected_video_claim_review_receipt_sha256": "a" * 64,
            "expected_cohort_tokens": tokens,
            "source_rows": [{
                "review_token": row["review_token"],
                "question_sha256": row["score_binding"]["question_sha256"],
                "options_sha256": row["score_binding"]["options_sha256"],
                "claim_sha256": row["score_binding"]["claim_sha256"],
                "source_video_sha256": row["score_binding"]["source_video_sha256"],
                "video_support_decision": "supports_claim",
                "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
            } for row in baseline_rows],
            "expected_baseline_source_sha256_by_token": {},
            "cohort_review_source_pins_by_token": {},
            "expected_baseline_candidate_source_bindings_by_token": {},
            "expected_review_payload_sha256": "b" * 64,
            "stage1_audit_status": "synthetic",
            "actual_qa_source_status": "synthetic",
            "stage1_source_bytes_sha256": {"worksheet": "c" * 64},
            "private_qa_bundle_sha256": "d" * 64,
            "reviewer_identity_verified": False,
            "semantic_review_authenticated": False,
            "clock_frame_correspondence_verified": False,
            "limitations": ["synthetic fixture"],
            "scoring_authorized": False, "model_load_permitted": False,
            "execution_authorization": "not_granted",
        }, sort_keys=True, separators=(",", ":")).encode()
        records = [{"record_token": hashlib.sha256(token.encode("ascii")).hexdigest(),
                    "processor_batch_sha256": inputs[
                        "expected_processor_batch_sha256_by_token"][token]}
                   for token in tokens]
        p1 = self._create("pre_baseline", records, p1_artifacts)
        manifest = create_qta_phase2_plan_manifest_candidate(
            phase1_receipt_raw=p1, phase1_record_bindings=records,
            phase1_preflight_raw=preflight_raw,
            phase1_stage1_source_pins_raw=p1_artifacts["stage1_review_and_lineage"],
            baseline_pq_manifest_raw=baseline_raw, plans=plans)
        capture_entries = []
        for index, (token, plan, baseline_row) in enumerate(
                zip(tokens, plans, baseline_rows, strict=True)):
            selection = plan["selection_provenance"]
            score_binding = baseline_row["score_binding"]
            groups = plan["temporal_groups"]
            input_length = score_binding["input_length"]
            qpos = score_binding["question_positions"]
            masses = [[struct.unpack(">f", bytes.fromhex(value))[0]
                       for value in row]
                      for row in selection["layer_group_mass_f32_be_hex"]]
            group_positions = {position for group in groups for position in group}
            residual_positions = [position for position in range(input_length)
                                  if position not in group_positions
                                  and position < qpos[0]]
            attention = []
            for layer_masses in masses:
                heads = []
                for _head in range(2):
                    queries = []
                    for _query in qpos:
                        values = [0.0] * input_length
                        for group_index, group in enumerate(groups):
                            values[group[0]] = layer_masses[group_index]
                        residual = 1.0 - sum(layer_masses)
                        if residual_positions:
                            values[residual_positions[0]] = residual
                        elif abs(residual) > 1e-6:
                            raise AssertionError("synthetic_attention_residual_unplaceable")
                        queries.append(values)
                    heads.append(queries)
                attention.append(heads)
            capture_entries.append({
                "binding": {
                    "record_token": hashlib.sha256(token.encode("ascii")).hexdigest(),
                    "row_binding_sha256": preflight["row_binding_sha256"][index],
                    "typed_batch_sha256": score_binding["processor_batch_sha256"],
                    "baseline_pq_row_sha256": hashlib.sha256(json.dumps(
                        baseline_row, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest(),
                    "prompt_sha256": score_binding["prompt_sha256"],
                    "input_ids_sha256": score_binding["input_ids_sha256"],
                    "model_sha256": score_binding["run_digests"]["model_sha256"],
                    "processor_sha256": score_binding["run_digests"]["processor_sha256"],
                    "tokenizer_sha256": score_binding["run_digests"]["tokenizer_sha256"],
                    "runtime_sha256": hashlib.sha256(b"synthetic runtime").hexdigest(),
                    "question_positions": qpos,
                    "temporal_groups": groups,
                }, "attention": attention, "input_length": input_length,
            })
        capture_index_raw, self.capture_sidecars = build_qta_capture_sidecar_index(
            entries=capture_entries)
        p2_artifacts = dict(self.p2_artifacts)
        p2_artifacts.update({"phase1_receipt": p1,
                             "baseline_pq_manifest": baseline_raw,
                             "nine_branch_plan_manifest": manifest,
                             "attention_capture_index": capture_index_raw})
        p2 = self._create("post_baseline", records, p2_artifacts,
                          prior_phase1_receipt_raw=p1)
        return p1_artifacts, p1, p2_artifacts, p2, records

    def _verify_phase2_candidate(self, bundle):
        p1_artifacts, p1, p2_artifacts, p2, _records = bundle
        return verify_qta_phase2_plan_candidate(
            phase1_receipt_raw=p1, phase1_artifact_bytes=p1_artifacts,
            phase2_receipt_raw=p2, phase2_artifact_bytes=p2_artifacts,
            attention_capture_sidecars=self.capture_sidecars,
            integrity_roster_raw=self.integrity, release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin)

    def test_phase2_plan_manifest_binds_exact_receipts_preflight_and_baseline_rows(self):
        report = self._verify_phase2_candidate(self._phase2_candidate_bundle())
        self.assertEqual(report["status"], "phase2_plan_manifest_bound_unvalidated")
        self.assertEqual(report["validated_plan_rows"], 3)
        self.assertEqual(report["scheduled_phase2_forwards"], 24)
        self.assertEqual(report["total_phase_forwards_including_phase1"], 27)
        self.assertEqual(report["selection_arithmetic_reproduced_records"], 3)
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertTrue(report["capture_sidecars_joined_to_selection_masses"])
        self.assertEqual(report["capture_sidecar_records_validated"], 3)
        self.assertTrue(report["phase1_receipt_signature_valid"])
        self.assertTrue(report["phase2_receipt_signature_valid"])
        self.assertTrue(report["phase1_receipt_recursively_verified"])
        self.assertTrue(report["same_freeze_id"])
        self.assertFalse(report["plan_semantics_authenticated"])
        self.assertEqual(report["authorized_forwards"], 0)

    def test_signed_phase2_capture_index_rejects_rehashed_mass_divergence(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        index = json.loads(p2_artifacts["attention_capture_index"])
        index["records"][0]["layer_group_masses_f32_be_hex"][0][0] = "00000000"
        p2_artifacts["attention_capture_index"] = (
            json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n").encode()
        changed_p2 = self._create("post_baseline", records, p2_artifacts,
                                  prior_phase1_receipt_raw=p1)
        changed_bundle = (p1_artifacts, p1, p2_artifacts, changed_p2, records)
        with self.assertRaisesRegex(ValueError,
                                   "capture_sidecar_group_mass_recomputation_mismatch"):
            self._verify_phase2_candidate(changed_bundle)

    def test_signed_phase2_capture_index_rejects_missing_or_changed_sidecar(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, p2, records = bundle
        digest, payload = next(iter(self.capture_sidecars.items()))
        original = self.capture_sidecars
        try:
            self.capture_sidecars = {digest: b"x" + payload[1:]}
            changed_bundle = (p1_artifacts, p1, p2_artifacts, p2, records)
            with self.assertRaisesRegex(ValueError,
                                       "capture_sidecar_digest_or_length_mismatch"):
                self._verify_phase2_candidate(changed_bundle)
            self.capture_sidecars = {}
            with self.assertRaisesRegex(ValueError, "capture_sidecar_payload_set_invalid"):
                self._verify_phase2_candidate(changed_bundle)
        finally:
            self.capture_sidecars = original

    def test_phase2_plan_manifest_rejects_changed_signed_baseline_bytes(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        p2_artifacts["baseline_pq_manifest"] += b" "
        changed_p2 = self._create("post_baseline", records, p2_artifacts,
                                  prior_phase1_receipt_raw=p1)
        changed_bundle = (p1_artifacts, p1, p2_artifacts, changed_p2, records)
        with self.assertRaisesRegex(ValueError, "phase2_manifest_baseline_pq_manifest_mismatch"):
            self._verify_phase2_candidate(changed_bundle)

    def test_phase2_plan_manifest_rejects_nonfrozen_treatment(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        valid = candidate["records"][0]["plan"]
        wrong = build_qta_nine_branch_plan(
            row_binding_sha256=valid["row_binding_sha256"],
            temporal_groups=valid["temporal_groups"],
            target_group_indices=valid["selected_groups"]["target"],
            sham_group_indices=valid["selected_groups"]["sham"],
            evidence_group_indices=valid["selected_groups"]["evidence"],
            control_group_indices=valid["selected_groups"]["control"],
            typed_batch_sha256=valid["typed_batch_sha256"],
            score_binding_sha256=valid["score_binding_sha256"],
            intervention_layers_1_based=[5, 6], logit_bias=0.5,
            sham_seed="wrong-synthetic-seed")
        with self.assertRaisesRegex(ValueError,
                                    "phase2_plan_differs_from_frozen_qta_protocol"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=p1_artifacts["stage1_review_and_lineage"],
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=[wrong, *(row["plan"] for row in candidate["records"][1:])])

    def test_phase2_plan_manifest_rejects_unbound_score_hash(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        valid = candidate["records"][0]["plan"]
        wrong = build_qta_nine_branch_plan(
            row_binding_sha256=valid["row_binding_sha256"],
            temporal_groups=valid["temporal_groups"],
            target_group_indices=valid["selected_groups"]["target"],
            sham_group_indices=valid["selected_groups"]["sham"],
            evidence_group_indices=valid["selected_groups"]["evidence"],
            control_group_indices=valid["selected_groups"]["control"],
            typed_batch_sha256=valid["typed_batch_sha256"],
            score_binding_sha256="f" * 64,
            intervention_layers_1_based=FROZEN_QTA_LAYERS_1_BASED,
            logit_bias=FROZEN_QTA_LOGIT_BIAS, sham_seed=FROZEN_QTA_SHAM_SEED)
        with self.assertRaisesRegex(ValueError, "phase2_plan_score_binding_mismatch"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=p1_artifacts["stage1_review_and_lineage"],
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=[wrong, *(row["plan"] for row in candidate["records"][1:])])

    def test_phase2_candidate_rejects_stage1_receipt_mismatch(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        stage1 = _strict_json(p1_artifacts["stage1_review_and_lineage"])
        stage1["expected_video_claim_review_receipt_sha256"] = "b" * 64
        altered_stage1_raw = json.dumps(
            stage1, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(
                ValueError, "phase1_baseline_video_review_receipt_mismatch"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=altered_stage1_raw,
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=[row["plan"] for row in candidate["records"]])

    def test_phase2_candidate_rejects_noncanonical_stage1_source_pin_schema(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        for mutate in (
                lambda obj: obj.update(schema_version=True),
                lambda obj: obj.update(unexpected_extension="ignored")):
            with self.subTest(mutate=mutate):
                stage1 = _strict_json(p1_artifacts["stage1_review_and_lineage"])
                mutate(stage1)
                altered_stage1_raw = json.dumps(
                    stage1, sort_keys=True, separators=(",", ":")).encode()
                with self.assertRaisesRegex(
                        ValueError, "phase1_stage1_source_pin_artifact_invalid"):
                    create_qta_phase2_plan_manifest_candidate(
                        phase1_receipt_raw=p1,
                        phase1_record_bindings=records,
                        phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                        phase1_stage1_source_pins_raw=altered_stage1_raw,
                        baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                        plans=[row["plan"] for row in candidate["records"]])

    def test_phase2_candidate_rejects_stage1_source_row_mismatch(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        stage1 = _strict_json(p1_artifacts["stage1_review_and_lineage"])
        stage1["source_rows"][1]["claim_sha256"] = "b" * 64
        altered_stage1_raw = json.dumps(
            stage1, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "phase2_plan_stage1_source_pin_mismatch"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=altered_stage1_raw,
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=[row["plan"] for row in candidate["records"]])

    def test_phase2_candidate_rejects_selection_provenance_binding_mismatch(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        plans = [row["plan"] for row in candidate["records"]]
        plans[0]["selection_provenance"]["prompt_sha256"] = "f" * 64
        plans[0]["plan_sha256"] = _plan_digest(plans[0])
        with self.assertRaisesRegex(ValueError, "qta_selection_binding_mismatch"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=p1_artifacts["stage1_review_and_lineage"],
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=plans)

    def test_phase2_candidate_anchors_question_positions_to_baseline_binding(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        plans = [row["plan"] for row in candidate["records"]]
        selection = plans[0]["selection_provenance"]
        selection["question_positions"] = [p + 1 for p in selection["question_positions"]]
        selection["question_positions_sha256"] = hashlib.sha256(json.dumps(
            selection["question_positions"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        plans[0]["plan_sha256"] = _plan_digest(plans[0])
        with self.assertRaisesRegex(ValueError,
                                    "qta_selection_geometry_or_protocol_mismatch"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=p1_artifacts[
                    "stage1_review_and_lineage"],
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=plans)

    def test_phase2_candidate_requires_selection_provenance(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase2_plan_manifest import _strict_json
        candidate = _strict_json(p2_artifacts["nine_branch_plan_manifest"])
        plans = [row["plan"] for row in candidate["records"]]
        plans[0]["selection_provenance"] = None
        plans[0]["plan_sha256"] = _plan_digest(plans[0])
        with self.assertRaisesRegex(ValueError, "phase2_selection_provenance_missing"):
            create_qta_phase2_plan_manifest_candidate(
                phase1_receipt_raw=p1, phase1_record_bindings=records,
                phase1_preflight_raw=p1_artifacts["atomic_cohort_preflight"],
                phase1_stage1_source_pins_raw=p1_artifacts["stage1_review_and_lineage"],
                baseline_pq_manifest_raw=p2_artifacts["baseline_pq_manifest"],
                plans=plans)

    def test_phase2_candidate_rejects_different_freeze_id(self):
        bundle = self._phase2_candidate_bundle()
        p1_artifacts, p1, p2_artifacts, _p2, records = bundle
        from qta_phase_freeze_receipts import create_phase_freeze_receipt
        changed_p2 = create_phase_freeze_receipt(
            phase="post_baseline", freeze_id="different-freeze-02",
            record_rows=records, artifact_bytes=p2_artifacts,
            integrity_roster_raw=self.integrity, release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
            freeze_owner_private_key_pem=self.keys["owner"][0],
            prior_phase1_receipt_raw=p1)
        with self.assertRaisesRegex(ValueError, "phase2_freeze_id_differs_from_phase1"):
            self._verify_phase2_candidate((p1_artifacts, p1, p2_artifacts,
                                           changed_p2, records))


if __name__ == "__main__":
    unittest.main()
