from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.build_nextgqa_prior_exposure_manifest import build
from scripts.audit_nextgqa_v2_cohort_prior_exposure import (
    DATA_FILES, EXPECTED_N, audit_media_entries, build_receipt,
    _validate_frozen_inputs, write_private_exclusive,
)


KEY = b"synthetic HMAC audit key with enough entropy 0123456789"


class NextGQAV2PriorExposureAuditTests(unittest.TestCase):
    def _frozen_shapes(self):
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in DATA_FILES}
        plan_hashes = {
            "test_csv_sha256": hashes["test_csv"],
            "gsub_test_sha256": hashes["gsub_test_json"],
            "frame2time_test_sha256": hashes["frame2time_test_json"],
            "map_vid_vidorID_sha256": hashes["map_vid_vidorID_json"],
        }
        media = [hashlib.sha256(f"synthetic-media-{i}".encode()).hexdigest()
                 for i in range(EXPECTED_N)]
        plan = {"status": "test_associated_exploratory_qid_valid_plan_not_evaluation",
            "calibration": [{} for _ in range(343)],
            "evaluation": [{"video_id": f"synthetic-video-{i}", "qid": f"qid-{i}",
                            "media_sha256": item,
                            "selected_row_binding_sha256": hashlib.sha256(
                                f"row-{i}".encode()).hexdigest()}
                           for i, item in enumerate(media)],
            "provenance": plan_hashes}
        preflight = {"status": "preflight_receipt_only_no_model_scoring",
            "counts": {"evaluation_rows_bound": EXPECTED_N,
                       "invalid_annotation_intervals": 0, "join_failures": 0},
            "input_sha256": {**dict(hashes), "split_plan": hashlib.sha256(
                json.dumps(plan).encode()).hexdigest()},
            "evaluation_row_bindings": [{
                "row_binding_sha256": hashlib.sha256(f"row-{i}".encode()).hexdigest()}
                for i, item in enumerate(media)]}
        return plan, preflight, hashes

    def _frozen_bytes(self, plan, preflight, hashes):
        plan_raw = json.dumps(plan).encode()
        preflight_raw = json.dumps(preflight).encode()
        gate = {"schema_version": 1,
            "status": "local_v2_inputs_frozen_integrity_only_not_authorization",
            "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "preflight_receipt_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "data_sha256": {DATA_FILES[key]: value for key, value in hashes.items()},
            "counts": {"evaluation_rows_bound": EXPECTED_N,
                       "evaluation_videos": EXPECTED_N}}
        return plan_raw, preflight_raw, json.dumps(gate).encode()

    def test_v2_plan_preflight_and_current_dataset_hashes_must_agree(self):
        plan, preflight, hashes = self._frozen_shapes()
        plan_raw, preflight_raw, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        loaded_plan, loaded_receipt = _validate_frozen_inputs(
            plan_raw=plan_raw, preflight_raw=preflight_raw, input_freeze_raw=freeze_raw,
            dataset_hashes=hashes, plan_sha256=hashlib.sha256(plan_raw).hexdigest())
        self.assertEqual(len(loaded_plan["evaluation"]), EXPECTED_N)
        self.assertEqual(len(loaded_receipt["evaluation_row_bindings"]), EXPECTED_N)
        hashes["test_csv"] = "f" * 64
        _, _, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        with self.assertRaisesRegex(ValueError, "current_dataset_hash_mismatch"):
            _validate_frozen_inputs(plan_raw=plan_raw, preflight_raw=preflight_raw,
                input_freeze_raw=freeze_raw, dataset_hashes=hashes,
                plan_sha256=hashlib.sha256(plan_raw).hexdigest())

    def test_self_consistent_plan_and_preflight_substitution_rejected_by_freeze(self):
        plan, preflight, hashes = self._frozen_shapes()
        _, _, original_freeze = self._frozen_bytes(plan, preflight, hashes)
        plan["evaluation"][0]["video_id"] = "substituted-video"
        plan_raw = json.dumps(plan).encode()
        preflight["input_sha256"]["split_plan"] = hashlib.sha256(plan_raw).hexdigest()
        preflight_raw = json.dumps(preflight).encode()
        with self.assertRaisesRegex(ValueError, "v2_local_input_freeze_binding_mismatch"):
            _validate_frozen_inputs(plan_raw=plan_raw, preflight_raw=preflight_raw,
                input_freeze_raw=original_freeze, dataset_hashes=hashes,
                plan_sha256=hashlib.sha256(plan_raw).hexdigest())

    def test_preflight_must_bind_exact_split_plan_bytes(self):
        plan, preflight, hashes = self._frozen_shapes()
        plan_raw, _, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        actual = hashlib.sha256(plan_raw).hexdigest()
        preflight["input_sha256"]["split_plan"] = "0" * 64
        _, preflight_raw, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        with self.assertRaisesRegex(ValueError, "v2_preflight_split_plan_hash_mismatch"):
            _validate_frozen_inputs(plan_raw=plan_raw, preflight_raw=preflight_raw,
                input_freeze_raw=freeze_raw, dataset_hashes=hashes, plan_sha256=actual)

    def test_preflight_row_binding_order_must_match_plan(self):
        plan, preflight, hashes = self._frozen_shapes()
        plan_raw, _, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        preflight["evaluation_row_bindings"][0]["row_binding_sha256"] = "0" * 64
        _, preflight_raw, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
        with self.assertRaisesRegex(ValueError, "v2_plan_preflight_row_binding_mismatch"):
            _validate_frozen_inputs(plan_raw=plan_raw, preflight_raw=preflight_raw,
                input_freeze_raw=freeze_raw, dataset_hashes=hashes,
                plan_sha256=hashlib.sha256(plan_raw).hexdigest())

    def test_preflight_counts_reject_boolean_substitutions(self):
        for key in ("evaluation_rows_bound", "invalid_annotation_intervals", "join_failures"):
            with self.subTest(key=key):
                plan, preflight, hashes = self._frozen_shapes()
                plan_raw, _, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
                preflight["counts"][key] = False if key != "evaluation_rows_bound" else True
                _, preflight_raw, freeze_raw = self._frozen_bytes(plan, preflight, hashes)
                with self.assertRaisesRegex(ValueError, "v2_frozen_denominator_invalid"):
                    _validate_frozen_inputs(plan_raw=plan_raw, preflight_raw=preflight_raw,
                        input_freeze_raw=freeze_raw, dataset_hashes=hashes,
                        plan_sha256=hashlib.sha256(plan_raw).hexdigest())

    def test_aggregate_match_and_unmatched_counts_redact_identity_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            match_bytes, novel_bytes = b"synthetic media already exposed", b"synthetic novel media"
            matched = private / "media-a.mp4"
            unmatched = private / "media-b.mp4"
            matched.write_bytes(match_bytes)
            unmatched.write_bytes(novel_bytes)
            matched_hash, unmatched_hash = map(lambda b: hashlib.sha256(b).hexdigest(),
                                               (match_bytes, novel_bytes))
            index, index_raw = build([("media-set-json", "nextgqa", "synthetic-source",
                json.dumps({"media_sha256": [matched_hash]}).encode())], KEY)
            index_tokens = set(index["identities"]["media_sha256_hmac_sha256"])
            entries = [
                {"video_id": "SECRET_VIDEO_A", "qid": "SECRET_QID_A",
                 "media_path": str(matched), "media_sha256": matched_hash},
                {"video_id": "SECRET_VIDEO_B", "qid": "SECRET_QID_B",
                 "media_path": str(unmatched), "media_sha256": unmatched_hash},
            ]
            counts = audit_media_entries(entries=entries, index_tokens=index_tokens,
                                         key=KEY, private_root=private, expected_count=2)
            self.assertEqual(counts["evaluation_rows_requested"], 2)
            self.assertEqual(counts["exact_byte_matches_to_supplied_partial_index"], 1)
            self.assertEqual(counts["unique_media_unmatched_to_supplied_index"], 1)
            plan = {"evaluation": entries}
            receipt = build_receipt(plan=plan, plan_sha256="a" * 64,
                preflight_sha256="b" * 64, dataset_hashes={"test_csv": "c" * 64},
                input_freeze_sha256="e" * 64,
                auditor_script_sha256="f" * 64,
                index_raw=index_raw, key=KEY, private_root=private, expected_count=2)
            rendered = json.dumps(receipt)
            self.assertEqual(receipt["provenance"]["input_freeze_sha256"], "e" * 64)
            self.assertEqual(receipt["provenance"]["auditor_script_sha256"], "f" * 64)
            for secret in ("SECRET_VIDEO_A", "SECRET_QID_A", str(private),
                           matched_hash, unmatched_hash, *index["identities"]["media_sha256_hmac_sha256"]):
                self.assertNotIn(secret, rendered)
            self.assertEqual(receipt["provenance"]["prior_index_completeness"],
                             "known_partial_unauthenticated")
            self.assertFalse(receipt["interpretation"]["unmatched_means_fresh"])
            self.assertFalse(receipt["interpretation"]["matches_excluded_or_resampled"])
            self.assertFalse(receipt["interpretation"]["faithfulness_scored"])

    def test_plan_media_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            path = private / "media.mp4"
            path.write_bytes(b"actual")
            with self.assertRaisesRegex(ValueError, "evaluation_media_hash_mismatch"):
                audit_media_entries(entries=[{"media_path": str(path),
                    "media_sha256": "0" * 64}], index_tokens=set(), key=KEY,
                    private_root=private, expected_count=1)

    def test_index_source_count_rejects_boolean_alias_for_integer(self):
        index, raw = build([("media-set-json", "nextgqa", "synthetic-source",
            json.dumps({"media_sha256": []}).encode())], KEY)
        index["coverage"]["source_count"] = True
        malformed = json.dumps(index, separators=(",", ":")).encode()
        from scripts.audit_nextgqa_v2_cohort_prior_exposure import _load_index
        with self.assertRaisesRegex(ValueError, "prior_index_schema_invalid"):
            _load_index(malformed)

    def test_symlinked_media_and_outside_private_media_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            private = base / "private"
            private.mkdir(mode=0o700)
            external = base / "external.mp4"
            external.write_bytes(b"synthetic")
            alias = private / "alias.mp4"
            alias.symlink_to(external)
            entry = {"media_path": str(alias), "media_sha256": hashlib.sha256(b"synthetic").hexdigest()}
            with self.assertRaisesRegex(ValueError, "input_unavailable_or_symlink"):
                audit_media_entries(entries=[entry], index_tokens=set(), key=KEY,
                    private_root=private, expected_count=1)
            with self.assertRaisesRegex(ValueError, "evaluation_media_outside_private_root"):
                audit_media_entries(entries=[{"media_path": str(external),
                    "media_sha256": hashlib.sha256(b"synthetic").hexdigest()}],
                    index_tokens=set(), key=KEY, private_root=private, expected_count=1)

    def test_output_is_private_exclusive_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            output = private / "receipt.json"
            write_private_exclusive(output, {"counts": {"rows": 2}}, private_root=private)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_exclusive(output, {"counts": {"rows": 3}}, private_root=private)
            self.assertEqual(json.loads(output.read_text()), {"counts": {"rows": 2}})


if __name__ == "__main__":
    unittest.main()
