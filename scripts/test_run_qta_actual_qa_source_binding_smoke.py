"""Synthetic-only tests for the aggregate QTA source-binding CLI."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_qta_actual_qa_source_binding_smoke import (
    BindingInputError, INPUT_PATHS, PINNED_CODE_PATHS, main, run_binding,
)
from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from test_nextgqa_actual_qa_source_adapter import make_inputs


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class QtaBindingCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Synthetic Test"], check=True)

        csv_raw, _selection, target_raw, lineage_raw, _ = make_inputs()
        target_doc = json.loads(target_raw)
        lineage_doc = json.loads(lineage_raw)
        claim_rows, sheet_rows = [], []
        for index, line in enumerate(lineage_doc["records"]):
            claim_text = f"synthetic private claim {index}"
            claim_sha = _sha(claim_text.encode())
            claim_id = "claim-" + _sha((line["video_sha256"] + "\0" +
                    line["dataset_question_id"] + "\0" + claim_sha).encode())
            line["claim_id"] = claim_id
            line["claim_text_sha256"] = claim_sha
            target_doc["records"][index]["claim_id"] = claim_id
            target_doc["records"][index]["claim_text_sha256"] = claim_sha
            claim_rows.append({
                "video_id": line["video_id"], "claim_id": claim_id,
                "claim_text": claim_text, "claim_text_sha256": claim_sha,
                "model_revision": "synthetic-model@" + "a" * 40,
                "generation_run_sha256": "b" * 64,
            })
            sheet_rows.append({"video_id": line["video_id"], "claim_id": claim_id,
                               "review_token": f"private-token-{index}"})
        source_selection = {"records": [{
            "video_id": row["video_id"], "claim_id": row["claim_id"],
            "dataset_video_id": row["dataset_video_id"],
            "dataset_question_id": row["dataset_question_id"],
        } for row in lineage_doc["records"]]}
        _, bound = build_actual_qa_source(
            csv_raw=csv_raw, selection=source_selection,
            target_manifest_raw=json.dumps({"records": target_doc["records"]}).encode(),
            lineage_manifest_raw=json.dumps(lineage_doc).encode())
        target_doc["records"] = bound["target_manifest"]["records"]
        claim_manifest = {
            "schema_version": 1, "eligible_dataset_id": "synthetic-nextgqa",
            "eligible_manifest_sha256": "c" * 64, "claim_policy_sha256": "d" * 64,
            "occurrence_policy": "all_relevant_occurrences", "claims": claim_rows,
        }
        target_candidate = {
            "purpose": "private_nextgqa_diagnostic_target_candidate_unvalidated",
            "status": "diagnostic_candidate_unvalidated", "requested_records": 3,
            "effective_records": 3, "records": target_doc["records"],
        }
        inputs = {
            "csv": csv_raw,
            "claim_manifest": json.dumps(claim_manifest).encode(),
            "worksheet": json.dumps({"records": sheet_rows}).encode(),
            "lineage": json.dumps(lineage_doc).encode(),
            "target_candidate": json.dumps(target_candidate).encode(),
            "protocol": b"synthetic protocol bytes v1\n",
            "config": b"synthetic config bytes v1\n",
        }
        self.input_bytes = dict(inputs)
        self.paths = dict(INPUT_PATHS)
        for role, raw in inputs.items():
            rel = self.paths[role]
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            if role in {"claim_manifest", "worksheet", "lineage", "target_candidate"}:
                path.chmod(0o600)
            elif role == "csv":
                path.chmod(0o755)
            elif role == "protocol":
                path.chmod(0o600)
            else:
                path.chmod(0o644)
        (self.root / "data").chmod(0o755)
        (self.root / "data" / "private_nextgqa").chmod(0o700)
        source_root = Path(__file__).resolve().parent
        for rel in PINNED_CODE_PATHS:
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source_root / Path(rel).name).read_bytes())
            path.chmod(0o644)
        self.inventory_path = "data/private_nextgqa/inventory-synthetic/pre_review_candidate_byte_inventory.json"
        inventory_parent = (self.root / self.inventory_path).parent
        inventory_parent.mkdir(mode=0o700)
        inventory_parent.chmod(0o700)
        inventory_roles = {
            "source_test_csv": self.paths["csv"],
            "claim_input_manifest": self.paths["claim_manifest"],
            "claim_mapping_worksheet": self.paths["worksheet"],
            "claim_lineage": self.paths["lineage"],
            "diagnostic_target_candidate": self.paths["target_candidate"],
            "qta_protocol_draft": self.paths["protocol"],
            "qta_treatment_selection": self.paths["config"],
            "qta_actual_qa_source_binding_cli": PINNED_CODE_PATHS[0],
            "qta_actual_qa_source_contract": PINNED_CODE_PATHS[1],
            "actual_qa_source_adapter": PINNED_CODE_PATHS[2],
        }
        artifacts = []
        for role, rel in inventory_roles.items():
            raw = (self.root / rel).read_bytes()
            artifacts.append({"role": role, "size_bytes": len(raw), "sha256": _sha(raw)})
        inventory = {
            "schema_version": 1, "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only", "review_started": False,
            "review_completed": False, "joint_freeze": False, "score_ready": False,
            "execution_authorization": "not_granted", "rights_status": "unverified_local_only",
            "artifacts": artifacts, "missing_required_slots": [],
        }
        inventory_raw = json.dumps(inventory, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode() + b"\n"
        inventory_file = self.root / self.inventory_path
        inventory_file.write_bytes(inventory_raw)
        inventory_file.chmod(0o600)
        self.expected_inventory_sha256 = _sha(inventory_raw)
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "pin synthetic protocol/config"], check=True)
        self.commit = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                                     check=True, stdout=subprocess.PIPE, text=True).stdout.strip()

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **changes):
        values = {role: rel for role, rel in self.paths.items()}
        values["pin_commit"] = self.commit
        values["inventory"] = self.inventory_path
        values["expected_inventory_sha256"] = self.expected_inventory_sha256
        values.update(changes)
        return type("Args", (), values)()

    def assert_denials(self, report):
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["execution_authorized"], False)
        self.assertIs(report["scoring_authorized"], False)
        self.assertEqual(report["forwards"], 0)
        self.assertEqual(report["score_calls"], 0)

    def run_cli_report(self, *, expected_inventory_sha256=None):
        argv = []
        for role, flag in (("csv", "--csv"), ("claim_manifest", "--claim-manifest"),
                           ("worksheet", "--worksheet"), ("lineage", "--lineage"),
                           ("target_candidate", "--target-candidate"),
                           ("protocol", "--protocol"), ("config", "--config")):
            argv.extend((flag, self.paths[role]))
        argv.extend(("--pin-commit", self.commit, "--inventory", self.inventory_path,
                     "--expected-inventory-sha256",
                     expected_inventory_sha256 or self.expected_inventory_sha256))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(argv, repo_root=self.root)
        return code, json.loads(stdout.getvalue()), stdout.getvalue()

    def test_synthetic_success_is_fixed_denominator_allowlist_only(self):
        stdout = io.StringIO()
        argv = []
        for role, flag in (("csv", "--csv"), ("claim_manifest", "--claim-manifest"),
                           ("worksheet", "--worksheet"), ("lineage", "--lineage"),
                           ("target_candidate", "--target-candidate"),
                           ("protocol", "--protocol"), ("config", "--config")):
            argv.extend((flag, self.paths[role]))
        argv.extend(("--pin-commit", self.commit))
        argv.extend(("--inventory", self.inventory_path,
                     "--expected-inventory-sha256", self.expected_inventory_sha256))
        with contextlib.redirect_stdout(stdout):
            code = main(argv, repo_root=self.root)
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "source_binding_passed_unvalidated")
        self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 3))
        self.assertTrue(report["protocol_blob_match"] and report["config_blob_match"])
        self.assertEqual(report["pin_provenance"], "caller_supplied_local_git_unverified")
        self.assertEqual(report["inventory_pin_provenance"], "caller_supplied_local_unverified")
        self.assertFalse(report["inventory_is_trust_root"])
        self.assertTrue(report["inventory_content_match"])
        self.assertFalse(report["private_bundle_retained"])
        self.assert_denials(report)
        public = stdout.getvalue()
        for secret in ("synthetic private claim", "private-token", "answer 0", "question 0?"):
            self.assertNotIn(secret, public)
        allowed = {
            "schema_version", "purpose", "status", "requested_rows", "validated_rows",
            "pin_commit", "pin_provenance", "protocol_blob_match", "config_blob_match",
            "inventory_pin_provenance", "inventory_content_match",
            "inventory_is_trust_root",
            "failure_code", "rights_status",
            "faithfulness_scored", "model_load_permitted", "execution_authorized",
            "scoring_authorized", "forwards", "score_calls", "private_bundle_retained",
        }
        self.assertEqual(set(report), allowed)
        self.assertNotIn(_sha((self.root / self.paths["protocol"]).read_bytes()), public)
        self.assertNotIn(_sha((self.root / self.paths["config"]).read_bytes()), public)

    def test_official_pre_review_candidate_inventory_path_is_accepted(self):
        old_directory = (self.root / self.inventory_path).parent
        new_relative = (
            "data/private_nextgqa/"
            "pre_review_inventory_candidate_20261018_synthetic"
            "/pre_review_candidate_byte_inventory.json")
        new_directory = self.root / new_relative
        old_directory.rename(new_directory.parent)
        self.inventory_path = new_relative
        code, report, _ = self.run_cli_report()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "source_binding_passed_unvalidated")
        self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 3))
        self.assertEqual(report["forwards"], 0)
        self.assertEqual(report["score_calls"], 0)
        self.assertEqual(report["inventory_pin_provenance"],
                         "caller_supplied_local_unverified")
        self.assertFalse(report["inventory_is_trust_root"])
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["scoring_authorized"])
        self.assert_denials(report)

    def test_protocol_worktree_must_equal_pinned_git_blob(self):
        path = self.root / self.paths["protocol"]
        path.write_bytes(b"changed synthetic protocol\n")
        path.chmod(0o644)
        # A public source mismatch must be caught by exact Git pin validation.
        with self.assertRaisesRegex(BindingInputError, "protocol_worktree_blob_mismatch"):
            run_binding(args=self.args(), repo_root=self.root)

    def test_cli_error_output_is_content_free_and_denied(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main([
                "--csv", self.paths["csv"], "--claim-manifest", self.paths["claim_manifest"],
                "--worksheet", self.paths["worksheet"], "--lineage", self.paths["lineage"],
                "--target-candidate", self.paths["target_candidate"],
                "--protocol", self.paths["protocol"], "--config", self.paths["config"],
                "--pin-commit", "not-a-full-commit-id",
                "--inventory", self.inventory_path,
                "--expected-inventory-sha256", self.expected_inventory_sha256,
            ], repo_root=self.root)
        self.assertEqual(code, 2)
        report = json.loads(stdout.getvalue())
        self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 0))
        self.assertEqual(report["failure_code"], "pin_commit_must_be_full_object_id")
        self.assertEqual(report["pin_provenance"], "caller_supplied_local_git_unverified")
        self.assertFalse(report["inventory_content_match"])
        self.assertNotIn("synthetic private claim", stdout.getvalue())
        self.assert_denials(report)

    def test_unsafe_permissions_symlink_and_outside_repo_fail_closed(self):
        claim_path = self.root / self.paths["claim_manifest"]
        claim_path.chmod(0o644)
        with self.assertRaisesRegex(BindingInputError, "claim_manifest_owner_mode_type_or_size_invalid"):
            run_binding(args=self.args(), repo_root=self.root)
        claim_path.chmod(0o600)
        with self.assertRaisesRegex(BindingInputError, "csv_path_not_allowlisted"):
            run_binding(args=self.args(csv="../outside.csv"), repo_root=self.root)
        private_dir = self.root / "data" / "private_nextgqa"
        private_dir.chmod(0o755)
        with self.assertRaisesRegex(BindingInputError, "private_data_directory_mode_invalid"):
            run_binding(args=self.args(), repo_root=self.root)
        private_dir.chmod(0o700)
        saved = private_dir / "claim_input_manifest_v6_20261017_0635.json"
        moved = private_dir / "claim-manifest.saved"
        saved.rename(moved)
        saved.symlink_to(moved.name)
        with self.assertRaisesRegex(BindingInputError, "claim_manifest_open_failed"):
            run_binding(args=self.args(), repo_root=self.root)
        saved.unlink()
        moved.rename(saved)

    def test_wrong_commit_or_untracked_protocol_fails_closed(self):
        with self.assertRaisesRegex(BindingInputError, "pin_commit_must_be_full_object_id"):
            run_binding(args=self.args(pin_commit=""), repo_root=self.root)
        with self.assertRaisesRegex(BindingInputError, "pin_commit_must_equal_current_head"):
            run_binding(args=self.args(pin_commit="0" * 40), repo_root=self.root)

    def test_each_execution_source_must_match_pinned_head_blob(self):
        for rel in PINNED_CODE_PATHS:
            with self.subTest(rel=rel):
                path = self.root / rel
                original = path.read_bytes()
                path.write_bytes(original + b"\n# uncommitted synthetic change\n")
                with self.assertRaisesRegex(BindingInputError,
                                            "pinned_code_worktree_blob_mismatch"):
                    run_binding(args=self.args(), repo_root=self.root)
                path.write_bytes(original)

    def test_self_consistent_cohort_replacement_is_blocked_by_unchanged_inventory(self):
        claim_path = self.root / self.paths["claim_manifest"]
        worksheet_path = self.root / self.paths["worksheet"]
        lineage_path = self.root / self.paths["lineage"]
        target_path = self.root / self.paths["target_candidate"]
        claim_doc = json.loads(claim_path.read_bytes())
        worksheet_doc = json.loads(worksheet_path.read_bytes())
        lineage_doc = json.loads(lineage_path.read_bytes())
        target_doc = json.loads(target_path.read_bytes())
        new_text = "synthetic substituted claim phrase"
        new_sha = _sha(new_text.encode())
        line = lineage_doc["records"][0]
        new_claim_id = "claim-" + _sha((line["video_sha256"] + "\0" +
                line["dataset_question_id"] + "\0" + new_sha).encode())
        claim_doc["claims"][0]["claim_text"] = new_text
        claim_doc["claims"][0]["claim_text_sha256"] = new_sha
        claim_doc["claims"][0]["claim_id"] = new_claim_id
        worksheet_doc["records"][0]["claim_id"] = new_claim_id
        line["claim_id"] = new_claim_id
        line["claim_text_sha256"] = new_sha
        target_doc["records"][0]["claim_id"] = new_claim_id
        target_doc["records"][0]["claim_text_sha256"] = new_sha
        for path, doc in ((claim_path, claim_doc), (worksheet_path, worksheet_doc),
                          (lineage_path, lineage_doc), (target_path, target_doc)):
            path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")),
                            encoding="utf-8")
            path.chmod(0o600)

        stdout = io.StringIO()
        argv = []
        for role, flag in (("csv", "--csv"), ("claim_manifest", "--claim-manifest"),
                           ("worksheet", "--worksheet"), ("lineage", "--lineage"),
                           ("target_candidate", "--target-candidate"),
                           ("protocol", "--protocol"), ("config", "--config")):
            argv.extend((flag, self.paths[role]))
        argv.extend(("--pin-commit", self.commit, "--inventory", self.inventory_path,
                     "--expected-inventory-sha256", self.expected_inventory_sha256))
        with contextlib.redirect_stdout(stdout):
            code = main(argv, repo_root=self.root)
        self.assertEqual(code, 2)
        report = json.loads(stdout.getvalue())
        self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 0))
        self.assertEqual(report["failure_code"], "inventory_artifact_bytes_mismatch")
        self.assertNotIn(new_text, stdout.getvalue())
        self.assert_denials(report)

    def test_inventory_duplicate_or_missing_role_blocks_fixed_cohort(self):
        inventory_file = self.root / self.inventory_path
        original = inventory_file.read_bytes()
        base = json.loads(original)
        for mutation, expected_code in (
            (lambda items: items.append(dict(items[0])), "inventory_artifact_role_duplicate"),
            (lambda items: items.__delitem__(next(i for i, row in enumerate(items)
                                                    if row["role"] == "source_test_csv")),
             "inventory_required_source_roles_missing"),
        ):
            with self.subTest(expected_code=expected_code):
                changed = json.loads(json.dumps(base))
                mutation(changed["artifacts"])
                raw = json.dumps(changed, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode() + b"\n"
                inventory_file.write_bytes(raw)
                inventory_file.chmod(0o600)
                code, report, _ = self.run_cli_report(expected_inventory_sha256=_sha(raw))
                self.assertEqual(code, 2)
                self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 0))
                self.assertEqual(report["failure_code"], expected_code)
                self.assert_denials(report)
        inventory_file.write_bytes(original)
        inventory_file.chmod(0o600)

    def test_every_inventory_role_hash_and_size_is_checked(self):
        inventory_file = self.root / self.inventory_path
        original = json.loads(inventory_file.read_bytes())
        try:
            for role in sorted({row["role"] for row in original["artifacts"]}):
                for field in ("sha256", "size_bytes"):
                    with self.subTest(role=role, field=field):
                        changed = json.loads(json.dumps(original))
                        artifact = next(row for row in changed["artifacts"]
                                        if row["role"] == role)
                        if field == "sha256":
                            artifact[field] = "0" * 64
                        else:
                            artifact[field] += 1
                        raw = json.dumps(changed, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode() + b"\n"
                        inventory_file.write_bytes(raw)
                        inventory_file.chmod(0o600)
                        code, report, _ = self.run_cli_report(
                            expected_inventory_sha256=_sha(raw))
                        self.assertEqual(code, 2)
                        self.assertEqual((report["requested_rows"],
                                          report["validated_rows"]), (3, 0))
                        self.assertEqual(report["failure_code"],
                                         "inventory_artifact_bytes_mismatch")
                        self.assert_denials(report)
        finally:
            inventory_file.write_bytes(json.dumps(
                original, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode() + b"\n")
            inventory_file.chmod(0o600)

    def test_inventory_pin_path_and_private_mode_fail_closed(self):
        code, report, _ = self.run_cli_report(expected_inventory_sha256="0" * 64)
        self.assertEqual(code, 2)
        self.assertEqual((report["requested_rows"], report["validated_rows"]), (3, 0))
        self.assertEqual(report["failure_code"], "inventory_expected_sha256_mismatch")
        self.assertFalse(report["inventory_content_match"])
        inventory_file = self.root / self.inventory_path
        inventory_file.chmod(0o644)
        with self.assertRaisesRegex(BindingInputError,
                                    "inventory_owner_mode_type_or_size_invalid"):
            run_binding(args=self.args(), repo_root=self.root)
        inventory_file.chmod(0o600)
        with self.assertRaisesRegex(BindingInputError, "inventory_path_not_allowlisted_pattern"):
            run_binding(args=self.args(inventory="data/private_nextgqa/other.json"),
                        repo_root=self.root)
        for unsafe in (
                "data/private_nextgqa/pre_review_inventory_candidate_/"
                "pre_review_candidate_byte_inventory.json",
                "data/private_nextgqa/pre_review_inventory_candidate_ok/../"
                "inventory-x/pre_review_candidate_byte_inventory.json",
                "/data/private_nextgqa/pre_review_inventory_candidate_x/"
                "pre_review_candidate_byte_inventory.json"):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    BindingInputError, "inventory_path_not_allowlisted_pattern"):
                run_binding(args=self.args(inventory=unsafe), repo_root=self.root)


if __name__ == "__main__":
    unittest.main()
