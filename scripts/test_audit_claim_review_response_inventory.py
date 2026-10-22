from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from audit_claim_review_response_inventory import inventory
from test_build_test_split_claim_support_frame_review import ClaimSupportFrameReviewTests


class ClaimReviewResponseInventoryTests(unittest.TestCase):
    def test_counts_only_exact_fingerprint_bound_responses_without_authenticating_them(self):
        fixture = ClaimSupportFrameReviewTests()
        fixture.setUp()
        artifact_raw, response = fixture.response()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "review.html"
            artifact.write_bytes(artifact_raw)
            artifact.chmod(0o600)
            artifact_duplicate = root / "review-copy.html"
            artifact_duplicate.write_bytes(artifact_raw)
            artifact_duplicate.chmod(0o600)
            response_path = root / "response.json"
            response_path.write_text(json.dumps(response), encoding="utf-8")
            (root / "duplicate.json").write_bytes(response_path.read_bytes())
            unrelated = dict(response, artifact_fingerprint="0" * 64)
            (root / "unrelated.json").write_text(json.dumps(unrelated), encoding="utf-8")
            result = inventory([artifact, artifact_duplicate], [root])

        self.assertEqual(result["candidate_json_files_scanned"], 3)
        self.assertEqual(result["bound_response_files_total"], 1)
        row = result["artifacts"][0]
        self.assertEqual(row["structurally_valid_response_files"], 1)
        self.assertEqual(row["self_reported_reviewer_entries"], 1)
        self.assertEqual(row["self_reported_item_decisions"], 3)
        self.assertFalse(result["reviewer_identity_or_independence_verified"])
        self.assertFalse(result["semantic_support_validated"])
        self.assertFalse(result["signed_stage1_or_direct_faithfulness_gate_satisfied"])
        self.assertEqual(result["direct_intervention_score_inventory"], "not_scanned")

    def test_rejects_artifact_without_private_permissions(self):
        fixture = ClaimSupportFrameReviewTests()
        fixture.setUp()
        artifact_raw, _ = fixture.response()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "review.html"
            artifact.write_bytes(artifact_raw)
            artifact.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "review_artifact_permissions_invalid"):
                inventory([artifact], [Path(tmp)])


if __name__ == "__main__":
    unittest.main()
