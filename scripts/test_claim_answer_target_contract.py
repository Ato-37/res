import hashlib
import json
import unittest

from claim_answer_target_contract import BINDING_KEYS, validate_target_manifest


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    videos = ["video-a", "video-b", "video-c"]
    claims = ["claim-a", "claim-b", "claim-c"]
    bindings = {key: h("binding-" + key) for key in BINDING_KEYS}
    rows = []
    for i, (video, claim) in enumerate(zip(videos, claims, strict=True)):
        rows.append({
            "video_id": video, "claim_id": claim,
            "dataset_qid_sha256": h("qid-" + str(i)),
            "dataset_source_row_sha256": h("source-row-" + str(i)),
            "question_sha256": h("question-" + str(i)),
            "ordered_choice_sha256": [h(f"choice-{i}-{j}") for j in range(5)],
            "gold_index": i, "answer_label_order": ["A", "B", "C", "D", "E"],
            "answer_label_token_ids": [10, 11, 12, 13, 14],
            "answer_label_token_piece_counts": [1, 1, 1, 1, 1],
            "rendered_scoring_prefix_sha256": h("prefix-" + str(i)),
        })
    manifest = {
        "schema_version": 1,
        "purpose": "private_claim_linked_answer_target_contract",
        "status": "frozen_unvalidated_target_manifest",
        "execution_authorization": "not_granted",
        **bindings,
        "records": rows,
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return videos, claims, bindings, manifest, raw


class ClaimAnswerTargetContractTests(unittest.TestCase):
    def validate(self, fixture_parts):
        videos, claims, bindings, _, raw = fixture_parts
        return validate_target_manifest(raw, videos, claims, bindings)

    def test_complete_contract_is_aggregate_only_and_never_authorizes(self):
        parts = fixture()
        result = self.validate(parts)
        self.assertEqual(result["status"], "target_contract_structurally_valid_unvalidated")
        self.assertEqual(result["validated_records"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        encoded = json.dumps(result)
        for secret in ("video-a", "claim-a", "qid-0", "question-0"):
            self.assertNotIn(secret, encoded)

    def test_single_token_assertion_requires_five_unique_ids(self):
        videos, claims, bindings, manifest, _ = fixture()
        manifest["records"][0]["answer_label_token_piece_counts"][0] = 2
        raw = json.dumps(manifest).encode()
        result = validate_target_manifest(raw, videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["single_token_label_assertion_invalid"])
        manifest["records"][0]["answer_label_token_piece_counts"] = [1] * 5
        manifest["records"][0]["answer_label_token_ids"][1] = 10
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["single_token_label_assertion_invalid"])
        for invalid in ([True] * 5, [1.0] * 5):
            manifest["records"][0]["answer_label_token_piece_counts"] = invalid
            result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
            self.assertEqual(result["failure_codes"], ["single_token_label_assertion_invalid"])

    def test_choice_order_gold_index_and_label_order_are_closed(self):
        for field, value in (("gold_index", True), ("answer_label_order", list("EDCBA")),
                             ("ordered_choice_sha256", [h("same")] * 4 + ["invalid"])):
            with self.subTest(field=field):
                videos, claims, bindings, manifest, _ = fixture()
                manifest["records"][0][field] = value
                result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
                self.assertIn(result["status"], {"blocked"})
                self.assertEqual(result["score_calls"], 0)
        videos, claims, bindings, manifest, _ = fixture()
        manifest["records"][0]["ordered_choice_sha256"][1] = manifest["records"][0]["ordered_choice_sha256"][0]
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["choice_order_binding_invalid"])

    def test_source_manifest_and_source_row_hashes_are_required(self):
        videos, claims, bindings, manifest, _ = fixture()
        bindings["dataset_source_manifest_sha256"] = h("changed-source")
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["manifest_binding_mismatch"])
        del manifest["records"][0]["dataset_source_row_sha256"]
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, fixture()[2])
        self.assertEqual(result["failure_codes"], ["target_row_schema_invalid"])

    def test_manifest_and_claim_bindings_cannot_be_reordered(self):
        videos, claims, bindings, manifest, _ = fixture()
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims[::-1], bindings)
        self.assertEqual(result["failure_codes"], ["target_claim_binding_mismatch"])
        manifest["records"].reverse()
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["target_video_order_mismatch"])
        manifest["records"].reverse()
        manifest["model_config_sha256"] = h("changed")
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["manifest_binding_mismatch"])

    def test_duplicate_json_keys_are_rejected(self):
        videos, claims, bindings, _, raw = fixture()
        duplicate = raw[:-1] + b',"schema_version":1}'
        result = validate_target_manifest(duplicate, videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["manifest_json_invalid"])

    def test_any_invalid_cohort_or_row_fails_closed(self):
        videos, claims, bindings, manifest, _ = fixture()
        result = validate_target_manifest(json.dumps(manifest).encode(), videos[:2], claims[:2], bindings)
        self.assertEqual(result["failure_codes"], ["expected_cohort_invalid"])
        manifest["records"].pop()
        result = validate_target_manifest(json.dumps(manifest).encode(), videos, claims, bindings)
        self.assertEqual(result["failure_codes"], ["manifest_record_count_invalid"])


if __name__ == "__main__":
    unittest.main()
