from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_minerva_generated_object_pixel_pilot import MODEL_REVISION
from build_minerva_generated_object_pixel_review import (
    FRAME_COUNT,
    _validate_attrition,
    _row_preflight,
    finalize_page,
    render,
    validate_response_binding,
    validate_receipt_pair,
    validate_review_items,
    verify_page_fingerprint,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _item(item_id: str) -> dict:
    question = "何が起きていますか？"
    claim = "人がカップを持ち上げています。"
    frames = [{"rank": rank, "source_frame_index": rank + 10,
                    "time_seconds": rank / 2,
                    "data_url": "data:image/jpeg;base64,AA=="}
                   for rank in range(FRAME_COUNT)]
    image_hashes = [_sha(b"\x00")] * FRAME_COUNT
    return {"item_id": item_id, "question": question, "claim": claim,
        "frames": frames, "source": {"question_sha256": _sha(question.encode()),
            "claim_sha256": _sha(claim.encode()), "frame_image_sha256": image_hashes,
            "frame_image_inventory_sha256": _sha(json.dumps(
                image_hashes, separators=(",", ":")).encode("ascii"))}}


class ReviewItemTests(unittest.TestCase):
    def test_requires_exactly_three_unique_items_and_64_ordered_frames(self):
        items = [_item(f"i{index}") for index in range(3)]
        validate_review_items(items)
        validate_review_items(items[:2])
        with self.assertRaisesRegex(ValueError, "one_to_three"):
            validate_review_items([])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_review_items([items[0], items[0], items[2]])
        short = [_item(f"i{index}") for index in range(3)]
        short[0]["frames"].pop()
        with self.assertRaisesRegex(ValueError, "item_contract"):
            validate_review_items(short)
        reordered = [_item(f"i{index}") for index in range(3)]
        reordered[0]["frames"][10]["rank"] = 11
        with self.assertRaisesRegex(ValueError, "frame_contract"):
            validate_review_items(reordered)
        altered = [_item(f"i{index}") for index in range(3)]
        altered[0]["frames"][0]["data_url"] = "data:image/jpeg;base64,AQ=="
        with self.assertRaisesRegex(ValueError, "image_inventory_binding"):
            validate_review_items(altered)

    def test_html_embeds_local_payload_safely_and_has_only_simple_verdicts(self):
        items = [_item(f"item-{index}") for index in range(3)]
        items[0]["claim"] = "</script><script>alert(1)</script>"
        items[0]["source"]["claim_sha256"] = _sha(items[0]["claim"].encode())
        html = render(items, {"manifest_sha256": "a" * 64,
            "builder_sha256": "b" * 64, "source_manifest_sha256": "c" * 64})
        self.assertIn("supports", html)
        self.assertIn("contradicts", html)
        self.assertIn("not_visible", html)
        self.assertIn("cannot_judge", html)
        self.assertNotIn("</script><script>alert(1)", html)
        self.assertIn("\\u003c/script>", html)
        self.assertIn("connect-src 'none'", html)
        self.assertIn("source_frame_indices", html)
        self.assertIn("frame_image_inventory_sha256", html)
        self.assertIn("claim_sha256", html)
        self.assertIn("question_sha256", html)
        self.assertIn("cohort_preflight_sha256", html)

    def test_canonical_page_fingerprint_and_export_bindings_detect_tampering(self):
        items = [_item(f"item-{index}") for index in range(3)]
        doc = render(items, {"manifest_sha256": "a" * 64,
            "builder_sha256": "b" * 64, "source_manifest_sha256": "c" * 64})
        page, fingerprint = finalize_page(doc.encode("utf-8"))
        self.assertEqual(verify_page_fingerprint(page),
            (fingerprint, "b" * 64, "c" * 64))
        response = {"artifact_fingerprint": fingerprint,
            "builder_sha256": "b" * 64, "source_manifest_sha256": "c" * 64}
        validate_response_binding(page, response)
        with self.assertRaisesRegex(ValueError, "page_fingerprint_mismatch"):
            verify_page_fingerprint(page.replace("確認を始める".encode(), "改変された".encode()))
        with self.assertRaisesRegex(ValueError, "response_page_binding_mismatch"):
            validate_response_binding(page, dict(response,
                artifact_fingerprint="0" * 64))
        with self.assertRaisesRegex(ValueError, "response_page_binding_mismatch"):
            validate_response_binding(page, dict(response, builder_sha256="0" * 64))


class ReceiptBindingTests(unittest.TestCase):
    def _pair(self):
        cpu_raw = b"exact cpu receipt"
        video_sha, video_md5, time_sha = "vsha", "vmd5", "tsha"
        ids = [10, 11]
        text = "The person lifts a cup."
        cpu = {"status": "completed_minerva_generated_object_pixel_cpu_preflight",
            "cohort_sha256": "cohort", "row_index": 1,
            "video_sha256": video_sha, "video_md5": video_md5,
            "time_map_sha256": time_sha, "row_key_sha256": "row",
            "annotation_source_sha256": {"minerva_json": "ann"},
            "runner_sha256": "runner",
            "cohort_cpu_preflight_sha256": "preflight"}
        gpu = {"status": "completed_single_video_generated_object_pixel_exploratory_pilot",
            "direct_generated_explanation_faithfulness": False,
            "cohort_sha256": "cohort", "row_index": 1,
            "video_sha256": video_sha, "video_md5": video_md5,
            "time_map_sha256": time_sha,
            "cpu_pixel_receipt_sha256": _sha(cpu_raw),
            "cohort_preflight_sha256": "preflight", "row_key_sha256": "row",
            "annotation_source_sha256": {"minerva_json": "ann"},
            "runner_sha256": "runner",
            "model_revision": MODEL_REVISION,
            "source_sha256": {"run_minerva_generated_object_pixel_pilot.py": "runner"},
            "generation": {"generated_text": text,
                "generated_text_sha256": _sha(text.encode()),
                "generated_token_ids": ids,
                "generated_token_ids_sha256": _sha(json.dumps(
                    ids, separators=(",", ":")).encode("ascii"))}}
        return gpu, cpu, b"gpu receipt", cpu_raw, video_sha, video_md5, time_sha

    def test_accepts_bound_receipt_pair(self):
        pair = self._pair()
        validate_receipt_pair(pair[0], pair[1], pair[2], pair[3],
            cohort_sha256="cohort", row_index=1, video_sha256=pair[4],
            video_md5=pair[5], time_map_sha256=pair[6])

    def test_rejects_changed_cpu_receipt_and_changed_generated_claim_hash(self):
        pair = self._pair()
        with self.assertRaisesRegex(ValueError, "gpu_cpu_source_binding"):
            validate_receipt_pair(pair[0], pair[1], pair[2], b"substituted receipt",
                cohort_sha256="cohort", row_index=1, video_sha256=pair[4],
                video_md5=pair[5], time_map_sha256=pair[6])
        altered = dict(pair[0])
        altered["generation"] = dict(pair[0]["generation"], generated_text="Other claim.")
        with self.assertRaisesRegex(ValueError, "gpu_cpu_source_binding"):
            validate_receipt_pair(altered, pair[1], pair[2], pair[3],
                cohort_sha256="cohort", row_index=1, video_sha256=pair[4],
                video_md5=pair[5], time_map_sha256=pair[6])


class RowPreflightTests(unittest.TestCase):
    def _preflight(self, record, **overrides):
        payload = {"schema_version": 1,
            "status": "completed_frozen_cohort_cpu_preflight",
            "requested_rows": 3, "effective_rows": 3,
            "cohort_sha256": "cohort", "manifest_sha256": "manifest",
            "preflight_source_sha256": "preflight-code",
            "decoder_source_sha256": "decoder-code",
            "generated_runner_source_sha256": "runner-code",
            "bbox_mapper_source_sha256": "mapper-code",
            "qwen_video_processor": "processor-config",
            "human_review_performed": False, "rights_status": "local_only",
            "records": [record]}
        payload.update(overrides)
        return payload

    def test_accepts_alternate_private_receipt_with_identical_selected_record(self):
        record = {"video_id": "P01-a", "row_key_sha256": "row",
            "status": "pass", "video_sha256": "video"}
        final = self._preflight(record, records=[record, {"video_id": "P02-b"}])
        alternate = self._preflight(record, requested_rows=1, effective_rows=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "row.json"
            raw = json.dumps(alternate).encode()
            path.write_bytes(raw)
            os.chmod(path, 0o600)
            with patch("build_minerva_generated_object_pixel_review._resolve_private",
                       return_value=path):
                got, got_raw, digest, got_record = _row_preflight(
                    {"cohort_preflight": "private/row.json"},
                    final_preflight=final, final_preflight_raw=b"final",
                    cohort_sha256="cohort", row_index=0,
                    selected={"video_id": "P01-a", "row_key_sha256": "row"})
        self.assertEqual(got, alternate)
        self.assertEqual(got_raw, raw)
        self.assertEqual(digest, _sha(raw))
        self.assertEqual(got_record, record)

    def test_rejects_alternate_record_or_source_manifest_mismatch(self):
        final_record = {"video_id": "P01-a", "row_key_sha256": "row", "status": "pass"}
        final = self._preflight(final_record)
        for alternate in (
            self._preflight({**final_record, "status": "fail"}),
            self._preflight(final_record, manifest_sha256="other-manifest"),
            self._preflight(final_record, decoder_source_sha256="other-decoder"),
        ):
            with self.subTest(alternate=alternate):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "row.json"
                    path.write_text(json.dumps(alternate))
                    os.chmod(path, 0o600)
                    with patch("build_minerva_generated_object_pixel_review._resolve_private",
                               return_value=path):
                        with self.assertRaisesRegex(ValueError, "row_preflight"):
                            _row_preflight({"cohort_preflight": "private/row.json"},
                                final_preflight=final, final_preflight_raw=b"final",
                                cohort_sha256="cohort", row_index=0,
                                selected={"video_id": "P01-a", "row_key_sha256": "row"})

    def test_rejects_non_private_alternate_receipt(self):
        record = {"video_id": "P01-a", "row_key_sha256": "row", "status": "pass"}
        final = self._preflight(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "row.json"
            path.write_text(json.dumps(self._preflight(record)))
            os.chmod(path, 0o644)
            with patch("build_minerva_generated_object_pixel_review._resolve_private",
                       return_value=path):
                with self.assertRaisesRegex(ValueError, "permissions_or_size"):
                    _row_preflight({"cohort_preflight": "private/row.json"},
                        final_preflight=final, final_preflight_raw=b"final",
                        cohort_sha256="cohort", row_index=0,
                        selected={"video_id": "P01-a", "row_key_sha256": "row"})
class AttritionTests(unittest.TestCase):
    def test_preflight_failure_and_missing_output_are_explicit_row_attrition(self):
        selected = {"video_id": "P07-20240623-120359", "row_key_sha256": "row"}
        base = {"row_index": 2, "reason_code": "rationale_mask_frame_missing_in_timestamp_bin"}
        failure = _validate_attrition(dict(base, status="preflight_failed"),
            row_index=2, selected=selected,
            record={"status": "fail", "failure_code": base["reason_code"]},
            cohort_sha256="c", preflight={"records": []}, preflight_raw=b"p",
            preflight_sha256="p", manifest={},
            manifest_sha256="m", minerva_json_sha256="q")
        self.assertEqual(failure["row_index"], 2)
        self.assertEqual(failure["status"], "preflight_failed")
        missing = _validate_attrition(dict(base, status="missing_output"),
            row_index=2, selected=selected, record={"status": "pass"},
            cohort_sha256="c", preflight={"records": []}, preflight_raw=b"p",
            preflight_sha256="p", manifest={},
            manifest_sha256="m", minerva_json_sha256="q")
        self.assertEqual(missing["reason_code"], base["reason_code"])
        with self.assertRaisesRegex(ValueError, "preflight_failure_attrition"):
            _validate_attrition(dict(base, status="preflight_failed"),
                row_index=2, selected=selected, record={"status": "pass"},
                cohort_sha256="c", preflight={"records": []}, preflight_raw=b"p",
                preflight_sha256="p", manifest={},
                manifest_sha256="m", minerva_json_sha256="q")
if __name__ == "__main__":
    unittest.main()
