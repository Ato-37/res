from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qta_phase_freeze_receipts import PRE_BASELINE_V2_ARTIFACT_ROLES
from qta_phase1_processor_artifact_adapter import (
    audit_qta_phase1_processor_artifacts,
    build_qta_phase1_processor_artifacts,
)
from test_qta_actual_processor_row_candidate import _build


class Phase1ProcessorArtifactAdapterTests(unittest.TestCase):
    def test_composes_real_processor_candidate_into_exact_non_authorizing_roles(self):
        candidate, *_ = _build()
        generated = {"prompt_answer_token_bindings", "processor_batch_digest_manifest"}
        base = {role: f"synthetic:{role}".encode()
                for role in set(PRE_BASELINE_V2_ARTIFACT_ROLES) - generated}
        protocol_raw = b"fixed synthetic protocol"
        artifacts = build_qta_phase1_processor_artifacts(
            candidate=candidate, base_artifact_bytes=base,
            protocol_raw=protocol_raw)
        self.assertEqual(set(artifacts), set(PRE_BASELINE_V2_ARTIFACT_ROLES))
        self.assertEqual(artifacts["prompt_answer_token_bindings"],
                         candidate["prompt_answer_token_bindings_raw"])
        manifest = json.loads(artifacts["processor_batch_digest_manifest"])
        self.assertEqual(manifest["protocol_sha256"], hashlib.sha256(protocol_raw).hexdigest())
        self.assertEqual(len(manifest["rows"]), 3)
        self.assertTrue(all(row["runtime_sha256"] == hashlib.sha256(
            base["sampler_model_runtime"]).hexdigest() for row in manifest["rows"]))
        self.assertEqual([row["prompt_sha256"] for row in manifest["rows"]],
                         [row["rendered_prompt_sha256"] for row in candidate["candidate_rows"]])
        report = audit_qta_phase1_processor_artifacts(artifacts)
        self.assertEqual(report["status"],
                         "complete_phase1_v2_artifact_mapping_candidate_only")
        self.assertEqual(report["processor_rows"], 3)
        for key in ("signature_created", "human_review_authenticated",
                    "model_load_permitted", "scoring_authorized"):
            self.assertFalse(report[key])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_refuses_missing_or_overlapping_role_inputs(self):
        candidate, *_ = _build()
        generated = {"prompt_answer_token_bindings", "processor_batch_digest_manifest"}
        base = {role: b"x" for role in set(PRE_BASELINE_V2_ARTIFACT_ROLES) - generated}
        missing = dict(base)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "inputs_invalid"):
            build_qta_phase1_processor_artifacts(
                candidate=candidate, base_artifact_bytes=missing,
                protocol_raw=b"protocol")
        with self.assertRaisesRegex(ValueError, "inputs_invalid"):
            build_qta_phase1_processor_artifacts(
                candidate=candidate,
                base_artifact_bytes={**base, "prompt_answer_token_bindings": b"splice"},
                protocol_raw=b"protocol")

    def test_rejects_candidate_batch_tampering(self):
        candidate, *_ = _build()
        candidate["candidate_rows"][0]["batch"]["input_ids"][0, 0] += 1
        generated = {"prompt_answer_token_bindings", "processor_batch_digest_manifest"}
        base = {role: b"x" for role in set(PRE_BASELINE_V2_ARTIFACT_ROLES) - generated}
        with self.assertRaisesRegex(ValueError, "candidate_batch_digest_mismatch"):
            build_qta_phase1_processor_artifacts(
                candidate=candidate, base_artifact_bytes=base,
                protocol_raw=b"protocol")


if __name__ == "__main__":
    unittest.main(verbosity=2)
