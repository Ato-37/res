import hashlib
import json
import unittest

from claim_actual_qa_target_integration import bind_actual_csv_to_complete_target
from claim_answer_target_contract import BINDING_KEYS
from test_nextgqa_actual_qa_source_adapter import h, make_inputs


def complete_inputs():
    csv_raw, selection, _, lineage_raw, _ = make_inputs()
    lineage = json.loads(lineage_raw)
    source_manifest_hash = hashlib.sha256(json.dumps(
        lineage["dataset_inputs_sha256"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    bindings = {key: h("binding-" + key) for key in BINDING_KEYS}
    bindings["dataset_source_manifest_sha256"] = source_manifest_hash
    rows = []
    for i, item in enumerate(selection["records"]):
        rows.append({
            "video_id": item["video_id"], "claim_id": item["claim_id"],
            "dataset_qid_sha256": h("qid-" + str(i)),
            "dataset_source_row_sha256": h("row-" + str(i)),
            "question_sha256": h("question-" + str(i)),
            "ordered_choice_sha256": [h(f"choice-{i}-{j}") for j in range(5)],
            "gold_index": 0, "answer_label_order": list("ABCDE"),
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
    target_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return csv_raw, selection, target_raw, lineage_raw, bindings


class ActualTargetIntegrationTests(unittest.TestCase):
    def run_bind(self, parts, bindings=None):
        csv_raw, selection, target_raw, lineage_raw, expected = parts
        return bind_actual_csv_to_complete_target(
            csv_raw=csv_raw, selection=selection,
            target_manifest_raw=target_raw, lineage_manifest_raw=lineage_raw,
            expected_binding_hashes=expected if bindings is None else bindings)

    def test_actual_csv_bindings_complete_target_without_authorizing(self):
        result = self.run_bind(complete_inputs())
        self.assertEqual(result["status"], "actual_csv_target_structurally_bound_unvalidated")
        self.assertEqual(result["bound_records"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        target = json.loads(result["target_manifest_raw"])
        source = json.loads(result["qa_source_raw"])
        self.assertEqual(len(target["records"]), 3)
        self.assertEqual(len(source["records"]), 3)
        self.assertEqual(source["purpose"], "private_actual_nextgqa_qa_source_rows_unvalidated")
        self.assertEqual(target["records"][0]["gold_index"], 0)
        self.assertEqual(target["records"][0]["ordered_choice_sha256"], [
            hashlib.sha256(choice.encode()).hexdigest()
            for choice in source["records"][0]["choices"]])

    def test_rejects_source_manifest_mismatch_and_incomplete_target(self):
        parts = complete_inputs()
        bindings = dict(parts[4])
        bindings["dataset_source_manifest_sha256"] = h("wrong")
        with self.assertRaisesRegex(ValueError, "dataset_source_manifest_binding_mismatch"):
            self.run_bind(parts, bindings)

        csv_raw, selection, target_raw, lineage_raw, expected = parts
        incomplete = json.loads(target_raw)
        del incomplete["records"][0]["rendered_scoring_prefix_sha256"]
        bad_parts = (csv_raw, selection, json.dumps(incomplete).encode(), lineage_raw, expected)
        with self.assertRaisesRegex(ValueError, "target_contract_target_row_schema_invalid"):
            self.run_bind(bad_parts)

    def test_rejects_invalid_binding_hashes(self):
        parts = complete_inputs()
        expected = dict(parts[4])
        expected["tokenizer_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "expected_bindings_invalid"):
            self.run_bind(parts, expected)


if __name__ == "__main__":
    unittest.main()
