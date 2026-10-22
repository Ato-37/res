import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from build_minerva_generation_bias_review import _finalize_page, _render
from inspect_minerva_generation_bias_review import (inspect_bytes, _read_response_json,
    _verify_mapping_receipts)
from test_build_minerva_generation_bias_review import sample_item


def fixture():
    builder = b"synthetic builder bytes"
    builder_sha = hashlib.sha256(builder).hexdigest()
    items = [sample_item(i) for i in range(5)]
    rows = [0, 0, 1, 2, 2]
    branches = [["baseline"], ["bbox_key_bias", "spatial_sham_key_bias"],
        ["baseline", "bbox_key_bias", "spatial_sham_key_bias"],
        ["baseline", "bbox_key_bias"], ["spatial_sham_key_bias"]]
    maps = []
    for item, row, treatment in zip(items, rows, branches, strict=True):
        source = item["source"]
        source.update({"row_key_sha256": f"{row + 1:064x}",
            "cpu_receipt_sha256": f"{row + 10:064x}",
            "gpu_receipt_sha256": f"{row + 20:064x}",
            "cohort_preflight_sha256": f"{row + 30:064x}",
            "generated_token_ids_sha256": f"{row + 40:064x}"})
        maps.append({"item_id": item["item_id"], "row_index": row,
            "branches": treatment, "claim_sha256": source["claim_sha256"],
            "row_key_sha256": source["row_key_sha256"],
            "cpu_receipt_sha256": source["cpu_receipt_sha256"],
            "gpu_receipt_sha256": source["gpu_receipt_sha256"],
            "cohort_preflight_sha256": source["cohort_preflight_sha256"],
            "generated_token_ids_sha256": source["generated_token_ids_sha256"]})
    provenance = {"schema_version": 1, "purpose": "private_blinded_minerva_generation_bias_review_source_manifest",
        "items": [{k: v for k, v in item["source"].items() if k != "item_id"}
            | {"item_id": item["item_id"]} for item in items],
        "requested_rows": 3, "distinct_review_items": 5, "frames_per_item": 64,
        "rights_status": "unverified_local_only", "builder_sha256": builder_sha}
    source_hash = hashlib.sha256(json.dumps(provenance, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    provenance["source_manifest_sha256"] = source_hash
    page, fingerprint = _finalize_page(_render(items, provenance).encode())
    mapping = {"schema_version": 1,
        "purpose": "private_blinded_review_item_to_generation_branch_map",
        "status": "private_mapping_do_not_share_with_reviewers",
        "artifact_fingerprint": fingerprint, "builder_sha256": builder_sha,
        "source_manifest_sha256": source_hash, "items": maps}
    respondents = []
    for nickname in ("reviewer-one", "reviewer-two"):
        answers = []
        for item in items:
            source = item["source"]
            answers.append({"item_id": item["item_id"], "verdict": "supports",
                "claim_sha256": source["claim_sha256"],
                "question_sha256": source["question_sha256"],
                "frame_image_inventory_sha256": source["frame_image_inventory_sha256"],
                "frame_image_sha256": source["frame_image_sha256"],
                "source_frame_indices": source["sample_frame_indices"],
                "frame_pts_seconds": source["frame_pts_seconds"]})
        respondents.append({"nickname": nickname, "answers": answers})
    response = {"schema_version": 1,
        "purpose": "private_blinded_minerva_generation_support_review",
        "status": "informal_frame_grounded_verdicts_only",
        "artifact_fingerprint": fingerprint, "builder_sha256": builder_sha,
        "source_manifest_sha256": source_hash, "respondents": respondents}
    return page, mapping, response, builder


class ReviewInspectorTests(unittest.TestCase):
    def test_valid_export_produces_only_aggregate_exploratory_counts(self):
        page, mapping, response, builder = fixture()
        result = inspect_bytes(page=page, mapping=mapping, response=response,
            builder_bytes=builder)
        self.assertEqual(result["status"], "valid_private_review_export")
        self.assertEqual(result["respondents"], 2)
        self.assertEqual(result["distinct_claim_items"], 5)
        self.assertEqual(result["items_with_multiple_generating_branches"], 3)
        self.assertEqual(result["branch_review_totals"], {
            "baseline": 6, "bbox_key_bias": 6, "spatial_sham_key_bias": 6})
        rendered = json.dumps(result)
        self.assertNotIn("reviewer-one", rendered)
        self.assertNotIn("A person moves", rendered)
        self.assertIn("not direct model faithfulness evidence", result["interpretation"])

    def test_rejects_tampered_page_and_response_page_mismatch(self):
        page, mapping, response, builder = fixture()
        changed_page = page.replace(b"<title>", b"<title>altered", 1)
        with self.assertRaisesRegex(ValueError, "page_fingerprint_mismatch"):
            inspect_bytes(page=changed_page, mapping=mapping, response=response,
                builder_bytes=builder)
        changed_response = copy.deepcopy(response)
        changed_response["artifact_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "response_page_binding_invalid"):
            inspect_bytes(page=page, mapping=mapping, response=changed_response,
                builder_bytes=builder)

    def test_rejects_missing_duplicate_or_invalid_answers(self):
        page, mapping, response, builder = fixture()
        bad = copy.deepcopy(response)
        bad["respondents"][0]["answers"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete_five_item"):
            inspect_bytes(page=page, mapping=mapping, response=bad, builder_bytes=builder)
        bad = copy.deepcopy(response)
        bad["respondents"][0]["answers"][0]["verdict"] = "true"
        with self.assertRaisesRegex(ValueError, "answer_item_or_verdict"):
            inspect_bytes(page=page, mapping=mapping, response=bad, builder_bytes=builder)

    def test_rejects_mapping_branch_or_source_binding_tampering(self):
        page, mapping, response, builder = fixture()
        bad_mapping = copy.deepcopy(mapping)
        bad_mapping["items"][0]["branches"] = ["baseline", "baseline"]
        with self.assertRaisesRegex(ValueError, "mapping_entry_invalid"):
            inspect_bytes(page=page, mapping=bad_mapping, response=response,
                builder_bytes=builder)
        bad_response = copy.deepcopy(response)
        bad_response["respondents"][0]["answers"][0]["source_frame_indices"][0] += 1
        with self.assertRaisesRegex(ValueError, "answer_source_binding_mismatch"):
            inspect_bytes(page=page, mapping=mapping, response=bad_response,
                builder_bytes=builder)

    def test_rejects_changed_builder_bytes(self):
        page, mapping, response, builder = fixture()
        with self.assertRaisesRegex(ValueError, "builder_hash_mismatch"):
            inspect_bytes(page=page, mapping=mapping, response=response,
                builder_bytes=builder + b" changed")

    def test_repeated_nickname_keeps_both_sessions_without_identity_claim(self):
        page, mapping, response, builder = fixture()
        response["respondents"][1]["nickname"] = "REVIEWER-ONE"
        result = inspect_bytes(page=page, mapping=mapping, response=response,
            builder_bytes=builder)
        self.assertEqual(result["respondents"], 2)
        self.assertEqual(result["unique_nickname_count"], 1)

    def test_reads_local_browser_download_without_private_copy(self):
        _, _, response, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(response), encoding="utf-8")
            os.chmod(path, 0o644)
            self.assertEqual(_read_response_json(path), response)
            link = Path(directory) / "review_link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "response_file_contract_invalid"):
                _read_response_json(link)

    def test_receipt_binding_rejects_coverage_preserving_branch_swap(self):
        page, mapping, _, _ = fixture()
        from inspect_minerva_generation_bias_review import _page_payload, _validate_items
        payload, _, _, _ = _page_payload(page)
        items = _validate_items(payload["items"])
        by_id = {entry["item_id"]: entry for entry in mapping["items"]}
        receipts = []
        for row in range(3):
            outcomes = {}
            for entry in mapping["items"]:
                if entry["row_index"] != row:
                    continue
                source = items[entry["item_id"]]["source"]
                for branch in entry["branches"]:
                    outcomes[branch] = {
                        "generated_text_sha256": source["claim_sha256"],
                        "generated_token_ids_sha256": source["generated_token_ids_sha256"]}
            receipt = {"outcomes": outcomes}
            raw = json.dumps(receipt).encode()
            digest = hashlib.sha256(raw).hexdigest()
            for entry in mapping["items"]:
                if entry["row_index"] == row:
                    entry["gpu_receipt_sha256"] = digest
            receipts.append((raw, receipt))
        _verify_mapping_receipts(by_id, items, receipts)
        row_zero = [entry for entry in mapping["items"] if entry["row_index"] == 0]
        row_zero[0]["branches"], row_zero[1]["branches"] = (
            row_zero[1]["branches"], row_zero[0]["branches"])
        with self.assertRaisesRegex(ValueError, "mapping_gpu_receipt_branch_mismatch"):
            _verify_mapping_receipts(by_id, items, receipts)


if __name__ == "__main__":
    unittest.main()
