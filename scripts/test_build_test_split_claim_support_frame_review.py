from __future__ import annotations

import unittest

from build_test_split_claim_support_frame_review import render, validate_decision
from inspect_claim_support_frame_review_response import inspect


class ClaimSupportFrameReviewTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"item_id": f"{i:024x}", "question": f"Q{i}", "claim": f"C{i}",
                       "frames": [{"data": "data:image/jpeg;base64,AA==", "time": j / 15}
                                  for j in range(16)]} for i in range(1, 4)]

    def response(self):
        digests = {key: f"{i:064x}" for i, key in enumerate(
            ["generator_sha256", "locked_assignment_sha256", "pair_filter_sha256", "translation_sha256",
             "decoder_sha256", "packet_sha256",
             "video_sha256", "baseline_sha256", "interaction_sha256", "row_token_sha256"], 1)}
        provenance = {"schema_version": 1, **{k: digests[k] for k in (
            "generator_sha256", "locked_assignment_sha256", "pair_filter_sha256", "translation_sha256",
            "decoder_sha256")},
            "items": [{"item_id": x["item_id"], **{k: digests[k] for k in (
                "packet_sha256", "video_sha256", "baseline_sha256", "interaction_sha256", "row_token_sha256")}}
                for x in self.items]}
        raw = render(self.items, provenance=provenance)
        fingerprint = raw.split(b'<meta name="review-artifact-sha256" content="', 1)[1].split(b'">', 1)[0].decode()
        return raw, {"schema_version": 1, "purpose": "local_claim_support_frame_review",
            "status": "human_responses_self_reported_unvalidated", "artifact_fingerprint": fingerprint,
            "created_at_utc": "2026-10-21T12:00:00.000Z", "limitations": ["a", "b", "c"],
            "reviewers": [{"nickname": "A", "answers": [
                {"item_id": x["item_id"], "decision": "supports", "frame_indices": [0, 4]}
                for x in self.items]}]}

    def test_decision_requires_evidence_frame_for_support_and_contradiction(self):
        self.assertTrue(validate_decision("supports", [0]))
        self.assertTrue(validate_decision("contradicts", [2, 3]))
        self.assertTrue(validate_decision("not_visible_in_model_frames", []))
        self.assertTrue(validate_decision("cannot_judge", []))
        self.assertFalse(validate_decision("supports", []))
        self.assertFalse(validate_decision("cannot_judge", [1]))
        self.assertFalse(validate_decision("supports", [1, 1]))
        self.assertFalse(validate_decision("supports", [16]))
        self.assertFalse(validate_decision([], [0]))
        self.assertFalse(validate_decision({}, [0]))

    def test_renderer_binds_private_items_and_reviewer_flow(self):
        raw = render(self.items)
        self.assertRegex(raw, rb'<meta name="review-artifact-sha256" content="[0-9a-f]{64}">')
        self.assertIn(b"const items=", raw)
        self.assertIn("16枚".encode(), raw)
        self.assertIn("スライド表示（動画の連続再生ではありません）".encode(), raw)
        self.assertIn(b"id=\"player-seek\"", raw)
        self.assertIn(b"id=\"frame-prev\"", raw)
        self.assertIn(b"id=\"frame-next\"", raw)
        self.assertIn(b"setInterval", raw)
        self.assertIn(b"next-reviewer", raw)
        self.assertIn(b"URL.revokeObjectURL(exportUrl)", raw)
        self.assertIn(b"saved-file').classList.add('hidden')", raw)
        self.assertIn(b"frame_indices:frames", raw)
        self.assertIn(b"connect-src 'none'", raw)
        self.assertNotIn(b"gold_answer", raw)

    def test_inspector_accepts_complete_artifact_bound_response(self):
        artifact, payload = self.response()
        result = inspect(payload, artifact)
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["reviewer_count"], 1)
        self.assertFalse(result["semantic_support_validated"])
        self.assertFalse(result["source_manifest_is_signed"])

    def test_inspector_accepts_two_item_generated_packet_with_explicit_attrition(self):
        items = self.items[:2]
        digest = lambda i: f"{i:064x}"
        provenance = {
            "schema_version": 1,
            "purpose": "generated_claim_support_frame_review_source_manifest",
            "packet_sha256": digest(1), "lock_sha256": digest(2),
            "pair_filter_sha256": digest(3), "exclusion_manifest_sha256": digest(4),
            "runner_sha256": digest(5), "model_revision": "pinned-test-revision",
            "generation_prompt_version": "v2_short", "requested_rows": 3,
            "structured_rows": 2, "unreviewable_attrition_rows": 1,
            "items": [{"record_index": i, "item_id": item["item_id"],
                       "video_sha256": digest(10+i), "qid_sha256": digest(20+i),
                       "candidate_pair_sha256": digest(30+i)}
                      for i, item in enumerate(items)],
        }
        artifact = render(items, provenance=provenance)
        fingerprint = __import__("re").search(
            rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">',
            artifact).group(1).decode()
        payload = {"schema_version": 1, "purpose": "local_claim_support_frame_review",
            "status": "human_responses_self_reported_unvalidated", "artifact_fingerprint": fingerprint,
            "created_at_utc": "2026-10-21T12:00:00.000Z", "limitations": ["a", "b", "c"],
            "reviewers": [{"nickname": "pilot", "answers": [
                {"item_id": item["item_id"], "decision": "supports", "frame_indices": [0]}
                for item in items]}]}
        result = inspect(payload, artifact)
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["items_per_reviewer"], 2)
        self.assertEqual(result["unreviewable_attrition_rows"], 1)
        self.assertFalse(result["semantic_support_validated"])

    def test_inspector_rejects_inconsistent_generated_packet_attrition(self):
        items = self.items[:2]
        digest = lambda i: f"{i:064x}"
        provenance = {
            "schema_version": 1,
            "purpose": "generated_claim_support_frame_review_source_manifest",
            "packet_sha256": digest(1), "lock_sha256": digest(2),
            "pair_filter_sha256": digest(3), "exclusion_manifest_sha256": digest(4),
            "runner_sha256": digest(5), "model_revision": "pinned-test-revision",
            "generation_prompt_version": "v2_short", "requested_rows": 3,
            "structured_rows": 2, "unreviewable_attrition_rows": 0,
            "items": [{"record_index": i, "item_id": item["item_id"],
                       "video_sha256": digest(10+i), "qid_sha256": digest(20+i),
                       "candidate_pair_sha256": digest(30+i)}
                      for i, item in enumerate(items)],
        }
        artifact = render(items, provenance=provenance)
        fingerprint = __import__("re").search(
            rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">',
            artifact).group(1).decode()
        payload = {"schema_version": 1, "purpose": "local_claim_support_frame_review",
            "status": "human_responses_self_reported_unvalidated", "artifact_fingerprint": fingerprint,
            "created_at_utc": "2026-10-21T12:00:00.000Z", "limitations": ["a", "b", "c"],
            "reviewers": [{"nickname": "pilot", "answers": [
                {"item_id": item["item_id"], "decision": "supports", "frame_indices": [0]}
                for item in items]}]}
        result = inspect(payload, artifact)
        self.assertFalse(result["structurally_valid"])
        self.assertEqual(result["failure"], "artifact_source_provenance_invalid")

    def test_inspector_rejects_wrong_artifact_or_bad_mapping(self):
        artifact, payload = self.response()
        payload["artifact_fingerprint"] = "0" * 64
        self.assertEqual(inspect(payload, artifact)["failure"], "response_header_or_artifact_binding_invalid")
        artifact, payload = self.response()
        payload["reviewers"][0]["answers"][0]["frame_indices"] = []
        self.assertEqual(inspect(payload, artifact)["failure"], "answer_decision_or_frames_invalid")

    def test_inspector_accepts_reused_nickname_without_claiming_identity(self):
        artifact, payload = self.response()
        second = {"nickname": "A", "answers": [dict(answer) for answer in payload["reviewers"][0]["answers"]]}
        payload["reviewers"].append(second)
        result = inspect(payload, artifact)
        self.assertTrue(result["structurally_valid"])
        self.assertFalse(result["reviewer_identity_or_independence_verified"])

    def test_inspector_rejects_empty_disclosure_strings(self):
        artifact, payload = self.response()
        payload["limitations"] = ["ok", " ", "ok"]
        self.assertEqual(inspect(payload, artifact)["failure"], "limitations_disclosure_invalid")

    def test_inspector_rejects_missing_source_manifest(self):
        artifact, payload = self.response()
        artifact = artifact.replace(b"const sourceProvenance=", b"const sourceProvenance={};const ignored=")
        fingerprint = __import__("re").search(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">', artifact).group(1)
        artifact = artifact.replace(fingerprint, b"__FINGERPRINT__", 1)
        fingerprint = __import__("hashlib").sha256(artifact).hexdigest().encode()
        artifact = artifact.replace(b"__FINGERPRINT__", fingerprint, 1)
        payload["artifact_fingerprint"] = fingerprint.decode()
        self.assertEqual(inspect(payload, artifact)["failure"], "artifact_source_provenance_invalid")

    def test_renderer_counts_partial_batch_without_inventing_replacements(self):
        raw = render(self.items[:2])
        self.assertIn("2件の動画説明".encode(), raw)
        self.assertIn("各人が2件すべて".encode(), raw)

    def test_renderer_rejects_empty_or_oversized_batch(self):
        with self.assertRaisesRegex(ValueError, "review_item_count_must_be_one_to_three"):
            render([])
        with self.assertRaisesRegex(ValueError, "review_item_count_must_be_one_to_three"):
            render(self.items + self.items[:1])


if __name__ == "__main__":
    unittest.main()
