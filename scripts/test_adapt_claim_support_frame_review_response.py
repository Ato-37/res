from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapt_claim_support_frame_review_response import adapt
from build_test_split_claim_support_frame_review import render
from inspect_claim_support_frame_review_response import FINGERPRINT_RE


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture(item_count: int = 2):
    item_ids = [f"opaque-{i}" for i in range(item_count)]
    items = [{"item_id": item_id, "question": f"q{i}", "claim": f"c{i}",
              "frames": [{"data": "data:image/jpeg;base64,AA==", "time": float(j)}
                         for j in range(16)]}
             for i, item_id in enumerate(item_ids)]
    provenance = {
        "schema_version": 1,
        "purpose": "generated_claim_support_frame_review_source_manifest",
        "packet_sha256": sha(b"packet"), "lock_sha256": sha(b"lock"),
        "pair_filter_sha256": sha(b"filter"),
        "exclusion_manifest_sha256": sha(b"exclusion"), "runner_sha256": sha(b"runner"),
        "model_revision": "fixture", "generation_prompt_version": "fixture_v1",
        "requested_rows": item_count, "structured_rows": item_count,
        "unreviewable_attrition_rows": 0,
        "items": [{"record_index": i, "item_id": item_id,
                   "video_sha256": sha(f"video{i}".encode()),
                   "qid_sha256": sha(f"qid{i}".encode()),
                   "candidate_pair_sha256": sha(f"pair{i}".encode())}
                  for i, item_id in enumerate(item_ids)],
    }
    artifact = render(items, provenance)
    fingerprint = FINGERPRINT_RE.search(artifact).group(1).decode()
    reviewers = [{"nickname": "reviewer1", "answers": [
        {"item_id": item_id, "decision": "supports", "frame_indices": [1, 3]}
        for item_id in item_ids]}]
    response = {"schema_version": 1, "purpose": "local_claim_support_frame_review",
                "status": "human_responses_self_reported_unvalidated",
                "artifact_fingerprint": fingerprint, "created_at_utc": "2026-10-22T00:00:00Z",
                "reviewers": reviewers,
                "limitations": ["a", "b", "c"]}
    return items, artifact, response


class ClaimSupportFrameAdapterTests(unittest.TestCase):
    def test_unanimous_support_maps_exact_indices_to_times_and_keeps_denominator(self):
        _, artifact, response = fixture()
        result = adapt(response, artifact)
        self.assertEqual(result["requested_rows"], 2)
        self.assertEqual(result["effective_eligible_rows"], 2)
        self.assertEqual(result["rows"][0]["selected_frame_indices_zero_based"], [1, 3])
        self.assertEqual(result["rows"][0]["selected_frame_times_seconds"], [1.0, 3.0])
        self.assertFalse(result["semantic_support_validated"])
        self.assertFalse(result["model_load_permitted"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_reviewer_disagreement_is_retained_but_not_scored_as_support(self):
        _, artifact, response = fixture(1)
        second = copy.deepcopy(response["reviewers"][0])
        second["nickname"] = "reviewer2"
        second["answers"][0]["decision"] = "cannot_judge"
        second["answers"][0]["frame_indices"] = []
        response["reviewers"].append(second)
        result = adapt(response, artifact)
        self.assertEqual(result["requested_rows"], 1)
        self.assertEqual(result["effective_eligible_rows"], 0)
        self.assertEqual(result["rows"][0]["decision_counts"]["supports"], 1)
        self.assertFalse(result["rows"][0]["eligible_for_exploratory_score"])

    def test_frame_mapping_uses_cross_reviewer_intersection(self):
        _, artifact, response = fixture(1)
        second = copy.deepcopy(response["reviewers"][0])
        second["nickname"] = "reviewer2"
        second["answers"][0]["frame_indices"] = [3, 4]
        response["reviewers"].append(second)
        result = adapt(response, artifact)
        row = result["rows"][0]
        self.assertEqual(row["selected_frame_indices_zero_based"], [3])
        self.assertTrue(row["eligible_for_exploratory_score"])

    def test_no_consensus_frame_keeps_unanimous_claim_ineligible(self):
        _, artifact, response = fixture(1)
        second = copy.deepcopy(response["reviewers"][0])
        second["nickname"] = "reviewer2"
        second["answers"][0]["frame_indices"] = [4]
        response["reviewers"].append(second)
        result = adapt(response, artifact)
        row = result["rows"][0]
        self.assertTrue(row["unanimous_support"])
        self.assertEqual(row["selected_frame_indices_zero_based"], [])
        self.assertFalse(row["eligible_for_exploratory_score"])

    def test_non_support_decision_is_retained_and_not_mapped(self):
        _, artifact, response = fixture(1)
        response["reviewers"][0]["answers"][0] = {
            "item_id": "opaque-0", "decision": "contradicts", "frame_indices": [4]}
        result = adapt(response, artifact)
        self.assertEqual(result["effective_eligible_rows"], 0)
        self.assertEqual(result["rows"][0]["selected_frame_times_seconds"], [])

    def test_wrong_fingerprint_is_rejected(self):
        _, artifact, response = fixture(1)
        response["artifact_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "review_binding_or_schema_invalid"):
            adapt(response, artifact)

    def test_nonmonotone_artifact_frame_times_are_rejected(self):
        items, _, response = fixture(1)
        items[0]["frames"][3]["time"] = 2.0
        provenance = {
            "schema_version": 1, "purpose": "generated_claim_support_frame_review_source_manifest",
            "packet_sha256": sha(b"packet"), "lock_sha256": sha(b"lock"),
            "pair_filter_sha256": sha(b"filter"), "exclusion_manifest_sha256": sha(b"exclusion"),
            "runner_sha256": sha(b"runner"), "model_revision": "fixture",
            "generation_prompt_version": "fixture_v1", "requested_rows": 1,
            "structured_rows": 1, "unreviewable_attrition_rows": 0,
            "items": [{"record_index": 0, "item_id": "opaque-0", "video_sha256": sha(b"v"),
                       "qid_sha256": sha(b"q"), "candidate_pair_sha256": sha(b"p")}],
        }
        bad_artifact = render(items, provenance)
        bad_fingerprint = FINGERPRINT_RE.search(bad_artifact).group(1).decode()
        response["artifact_fingerprint"] = bad_fingerprint
        with self.assertRaisesRegex(ValueError, "artifact_frame_times_invalid"):
            adapt(response, bad_artifact)


if __name__ == "__main__":
    unittest.main()
