from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from build_nextgqa_pre_review_byte_inventory import CODE_FILES
from qta_actual_media_source_preflight import (
    DENIAL, REPO_SOURCE_PATHS, _code_closure, _read_private_file,
    _read_repo_source, _sha,
    preflight_actual_media_source_join,
)
from test_qta_actual_prebaseline_diagnostic import combined_inputs


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _fixture():
    source_inputs, _review_inputs, _private_values = combined_inputs()
    claim_records = json.loads(source_inputs["claim_manifest_raw"])["records"]
    selection = json.loads(source_inputs["selection_raw"])
    target_records = json.loads(source_inputs["target_manifest_raw"])["records"]
    lineage = json.loads(source_inputs["lineage_manifest_raw"])
    input_claim_doc = {
        "schema_version": 1, "eligible_dataset_id": "synthetic-nextgqa",
        "eligible_manifest_sha256": "a" * 64, "claim_policy_sha256": "b" * 64,
        "occurrence_policy": "all_relevant_occurrences",
        "claims": [{**row, "model_revision": "synthetic-model@" + "c" * 40,
                    "generation_run_sha256": "d" * 64} for row in claim_records],
    }
    worksheet_raw = _canonical({"records": [
        {"video_id": row["video_id"], "claim_id": row["claim_id"],
         "review_token": row["review_token"]}
        for row in selection["records"]]})
    target_candidate = {
        "purpose": "private_nextgqa_diagnostic_target_candidate_unvalidated",
        "status": "diagnostic_candidate_unvalidated",
        "requested_records": 3, "effective_records": 3,
        "records": target_records,
    }
    protocol = source_inputs["qta_protocol_raw"]
    config = source_inputs["qta_config_raw"]
    source_inputs = {
        "csv_raw": source_inputs["csv_raw"],
        "claim_input_manifest_raw": _canonical(input_claim_doc),
        "worksheet_raw": worksheet_raw,
        "target_candidate_raw": _canonical(target_candidate),
        "lineage_manifest_raw": source_inputs["lineage_manifest_raw"],
        "qta_protocol_raw": protocol,
        "qta_config_raw": config,
    }
    video_hashes = [row["video_sha256"] for row in lineage["records"]]
    inventory = {
        "schema_version": 1,
        "purpose": "pre_review_candidate_byte_inventory",
        "status": "candidate_only",
        "execution_authorization": "not_granted",
        "joint_freeze": False,
        "score_ready": False,
        "missing_required_slots": [],
        "artifacts": [],
    }
    for role, relpath in CODE_FILES.items():
        raw = (Path(__file__).resolve().parents[1] / relpath).read_bytes()
        inventory["artifacts"].append({
            "role": role, "sha256": _sha(raw), "size_bytes": len(raw)})
    for input_key, role in {
        "csv_raw": "source_test_csv",
        "claim_input_manifest_raw": "claim_input_manifest",
        "worksheet_raw": "claim_mapping_worksheet",
        "target_candidate_raw": "diagnostic_target_candidate",
        "lineage_manifest_raw": "claim_lineage",
        "qta_protocol_raw": "qta_protocol_draft",
        "qta_config_raw": "qta_treatment_selection",
    }.items():
        raw = source_inputs[input_key]
        inventory["artifacts"].append({"role": role, "sha256": _sha(raw),
                                       "size_bytes": len(raw)})
    media_by_role = {f"staged_media_{index:02d}": digest
                     for index, digest in enumerate(video_hashes, 1)}
    for role, digest in media_by_role.items():
        inventory["artifacts"].append({"role": role, "sha256": digest,
                                       "size_bytes": 1})
    inventory_raw = _canonical(inventory)
    rows = [{"record_index": index, "source_sha256": digest,
             "sampled_frame_count": 32,
             "sampled_pts": [[n, 1] for n in range(32)],
             "sampled_rgb_sha256": [_sha(f"{digest}:{n}".encode()) for n in range(32)]}
            for index, digest in enumerate(video_hashes, 1)]
    sampler = {
        "schema_version": 1,
        "purpose": "local_qta_real_video_sampler_eligibility_preflight",
        "status": "passed", "requested_records": 3, "effective_records": 3,
        "sampler": "endpoint_inclusive_uniform_decoded_frame_index_v1",
        "frames_per_video": 32,
        "candidate_inventory_sha256": _sha(inventory_raw), "rows": rows,
        "gpu_used": False, "model_loaded": False,
        "faithfulness_scored": False, "execution_authorization": "not_granted",
    }
    return source_inputs, inventory_raw, media_by_role, _canonical(sampler), rows


class QtaActualMediaSourcePreflightTests(unittest.TestCase):
    def run_join(self, source, inventory_raw, media_by_role, sampler_raw,
                 *, code_files=CODE_FILES):
        return preflight_actual_media_source_join(
            sampler_receipt_raw=sampler_raw,
            candidate_inventory_raw=inventory_raw,
            media_sha256_by_role=media_by_role,
            actual_source_inputs=source,
            repo_root=Path(__file__).resolve().parents[1],
            code_files=code_files)

    def assert_denied(self, report):
        for field, value in DENIAL.items():
            self.assertEqual(report[field], value)
        self.assertFalse(report["sampler_provenance_authenticated"])
        self.assertFalse(report["actual_source_provenance_authenticated"])

    def test_hash_join_succeeds_independent_of_sampler_order(self):
        source, inventory_raw, media, sampler_raw, rows = _fixture()
        parsed = json.loads(sampler_raw)
        parsed["rows"].reverse()
        report = self.run_join(source, inventory_raw, media, _canonical(parsed))
        self.assertEqual(report["status"], "source_video_hash_joined_unvalidated")
        self.assertEqual((report["requested_rows"], report["sampler_rows"],
                          report["actual_qa_rows"], report["joined_rows"]), (3, 3, 3, 3))
        self.assertIn("stage2_prediction_link_review", report["later_gates"])
        self.assert_denied(report)
        public = json.dumps(report)
        for value in ("opaque-token-0", "private claim phrase", "question 0?",
                      "answer 0", "target-0", rows[0]["source_sha256"]):
            self.assertNotIn(value, public)

    def test_duplicate_sampler_hash_fails_atomic_three_row_denominator(self):
        source, inventory_raw, media, sampler_raw, _ = _fixture()
        parsed = json.loads(sampler_raw)
        parsed["rows"][2]["source_sha256"] = parsed["rows"][1]["source_sha256"]
        report = self.run_join(source, inventory_raw, media, _canonical(parsed))
        self.assertEqual(report["failure_code"], "sampler_receipt_row_invalid")
        self.assertEqual((report["requested_rows"], report["joined_rows"],
                          report["sampler_rows"], report["actual_qa_rows"]), (3, 0, 0, 0))
        self.assert_denied(report)

    def test_sampler_must_bind_exact_candidate_inventory_and_media_set(self):
        source, inventory_raw, media, sampler_raw, _ = _fixture()
        parsed = json.loads(sampler_raw)
        parsed["candidate_inventory_sha256"] = "0" * 64
        report = self.run_join(source, inventory_raw, media, _canonical(parsed))
        self.assertEqual(report["failure_code"],
                         "sampler_receipt_schema_or_inventory_binding_invalid")
        parsed = json.loads(sampler_raw)
        parsed["rows"][0]["source_sha256"] = "f" * 64
        report = self.run_join(source, inventory_raw, media, _canonical(parsed))
        self.assertEqual(report["failure_code"],
                         "sampler_receipt_media_inventory_mismatch")
        self.assert_denied(report)

    def test_stale_current_code_closure_is_reported_and_blocks_join(self):
        source, inventory_raw, media, sampler_raw, _ = _fixture()
        inventory = json.loads(inventory_raw)
        inventory["artifacts"][0]["sha256"] = "0" * 64
        inventory_raw = _canonical(inventory)
        report = self.run_join(source, inventory_raw, media, sampler_raw)
        self.assertEqual(report["failure_code"], "candidate_inventory_code_closure_stale")
        self.assertEqual(report["joined_rows"], 0)
        self.assert_denied(report)

    def test_actual_source_binder_failure_blocks_all_rows(self):
        role_by_input = {
            "csv_raw": "source_test_csv",
            "claim_input_manifest_raw": "claim_input_manifest",
            "worksheet_raw": "claim_mapping_worksheet",
            "target_candidate_raw": "diagnostic_target_candidate",
            "lineage_manifest_raw": "claim_lineage",
            "qta_protocol_raw": "qta_protocol_draft",
            "qta_config_raw": "qta_treatment_selection",
        }
        for input_key, role in role_by_input.items():
            with self.subTest(role=role):
                source, inventory_raw, media, sampler_raw, _ = _fixture()
                source[input_key] += b"\nchanged"
                report = self.run_join(source, inventory_raw, media, sampler_raw)
                self.assertEqual(
                    report["failure_code"],
                    f"candidate_inventory_source_role_mismatch_{role}")
                self.assertEqual((report["requested_rows"], report["joined_rows"],
                                  report["sampler_rows"], report["actual_qa_rows"]),
                                 (3, 0, 0, 0))
                self.assert_denied(report)

    def test_missing_inventory_source_role_fails_closed(self):
        source, inventory_raw, media, sampler_raw, _ = _fixture()
        inventory = json.loads(inventory_raw)
        inventory["artifacts"] = [row for row in inventory["artifacts"]
                                  if row["role"] != "claim_lineage"]
        inventory_raw = _canonical(inventory)
        sampler = json.loads(sampler_raw)
        sampler["candidate_inventory_sha256"] = _sha(inventory_raw)
        report = self.run_join(source, inventory_raw, media, _canonical(sampler))
        self.assertEqual(report["failure_code"],
                         "candidate_inventory_source_role_mismatch_claim_lineage")
        self.assertEqual(report["joined_rows"], 0)
        self.assert_denied(report)

    def test_lineage_video_id_must_be_unique(self):
        source, inventory_raw, media, sampler_raw, _ = _fixture()
        lineage = json.loads(source["lineage_manifest_raw"])
        lineage["records"][2]["video_id"] = lineage["records"][1]["video_id"]
        source["lineage_manifest_raw"] = _canonical(lineage)
        inventory = json.loads(inventory_raw)
        for item in inventory["artifacts"]:
            if item["role"] == "claim_lineage":
                raw = source["lineage_manifest_raw"]
                item.update(sha256=_sha(raw), size_bytes=len(raw))
        inventory_raw = _canonical(inventory)
        parsed_sampler = json.loads(sampler_raw)
        parsed_sampler["candidate_inventory_sha256"] = _sha(inventory_raw)
        report = self.run_join(source, inventory_raw, media, _canonical(parsed_sampler))
        self.assertEqual(report["failure_code"],
                         "actual_qa_lineage_video_identity_invalid")
        self.assertEqual(report["joined_rows"], 0)
        self.assert_denied(report)

    def test_fixed_repository_source_paths_read_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role, relative in REPO_SOURCE_PATHS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic:{role}".encode())
                path.chmod(0o644)
                self.assertEqual(_read_repo_source(
                    root, relative, role=role), f"synthetic:{role}".encode())
                with self.assertRaisesRegex(ValueError, "path_not_allowlisted"):
                    _read_repo_source(root, "elsewhere/synthetic", role=role)

    def test_fixed_repository_source_rejects_symlinks_and_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role, relative = next(iter(REPO_SOURCE_PATHS.items()))
            path = root / relative
            path.parent.mkdir(parents=True)
            target = path.with_name("target")
            target.write_bytes(b"synthetic source")
            target.chmod(0o644)
            path.symlink_to(target.name)
            with self.assertRaisesRegex(ValueError, "path_unavailable_or_symlink"):
                _read_repo_source(root, relative, role=role)
            path.unlink()
            path.write_bytes(b"synthetic source")
            path.chmod(0o644)
            real_read = os.read
            write_fd = os.open(path, os.O_WRONLY)
            changed = False

            def mutate_after_read(fd, count):
                nonlocal changed
                data = real_read(fd, count)
                if not changed:
                    changed = True
                    os.pwrite(write_fd, b"X", 0)
                return data

            try:
                with mock.patch("qta_actual_media_source_preflight.os.read",
                                side_effect=mutate_after_read):
                    with self.assertRaisesRegex(ValueError, "changed_during_read"):
                        _read_repo_source(root, relative, role=role)
            finally:
                os.close(write_fd)

    def test_nested_candidate_inventory_path_is_safe_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.chmod(0o700)
            directory = root / "inventory-qta-join-synthetic"
            directory.mkdir(mode=0o700)
            inventory = directory / "pre_review_candidate_byte_inventory.json"
            inventory.write_bytes(b"synthetic inventory")
            inventory.chmod(0o600)
            relative = f"{directory.name}/{inventory.name}"
            self.assertEqual(_read_private_file(
                root, relative, 1024, allow_inventory_path=True),
                b"synthetic inventory")
            with self.assertRaisesRegex(ValueError, "input_filename_invalid"):
                _read_private_file(root, "../outside", 1024,
                                   allow_inventory_path=True)
            inventory.unlink()
            inventory.symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "private_input_path_unavailable_or_symlink"):
                _read_private_file(root, relative, 1024,
                                   allow_inventory_path=True)

    def test_code_closure_rejects_symlink_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            target = root / "real.py"
            target.write_text("synthetic code\n")
            (root / "scripts" / "linked.py").symlink_to(target)
            inventory = {
                "purpose": "pre_review_candidate_byte_inventory",
                "status": "candidate_only", "execution_authorization": "not_granted",
                "joint_freeze": False, "score_ready": False,
                "missing_required_slots": [],
                "artifacts": [{"role": "test_code", "sha256": _sha(target.read_bytes()),
                               "size_bytes": target.stat().st_size}],
            }
            with self.assertRaisesRegex(
                    ValueError, "candidate_inventory_code_path_unavailable_or_symlink"):
                _code_closure(_canonical(inventory), repo_root=root,
                              code_files={"test_code": "scripts/linked.py"})


if __name__ == "__main__":
    unittest.main()
