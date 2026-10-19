from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from build_nextgqa_pre_review_byte_inventory import (
    CODE_FILES,
    PACKAGE_DIR,
    PRIVATE_FILES,
    PRIVATE_ROOT,
    PUBLIC_OR_SOURCE_FILES,
    STAGED_MEDIA_DIR,
    build_inventory,
)


class SyntheticPreReviewInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir(mode=0o700)
        private_root = self.root / PRIVATE_ROOT
        private_root.mkdir(mode=0o700)
        for relative in PRIVATE_FILES.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(("synthetic private bytes " + relative.name).encode())
            path.chmod(0o600)
        for role, relative in PUBLIC_OR_SOURCE_FILES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic public bytes " + role).encode())
        for relative in CODE_FILES.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            source = (("# synthetic source " + relative + "\n")
                      if relative.endswith(".py") else "# synthetic script\n")
            path.write_bytes(source.encode())

        self.package = self.root / PACKAGE_DIR
        self.staged = self.root / STAGED_MEDIA_DIR
        self.package.mkdir(mode=0o700)
        self.staged.mkdir(mode=0o700)
        (self.package / "index.html").write_bytes(b"synthetic review page")
        (self.package / "index.html").chmod(0o600)
        self.media_names = ["synthetic-token-a.mp4", "synthetic-token-b.mp4",
                            "synthetic-token-c.mp4"]
        for index, name in enumerate(self.media_names):
            data = f"synthetic video bytes {index}".encode()
            for directory in (self.package, self.staged):
                path = directory / name
                path.write_bytes(data)
                path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_synthetic_inventory_is_candidate_only_and_redacted(self):
        output = self.root / PRIVATE_ROOT / "inventory-unique"
        result = build_inventory(self.root, output)
        raw = (output / "pre_review_candidate_byte_inventory.json").read_bytes()
        parsed = json.loads(raw)

        self.assertEqual(parsed, result)
        self.assertEqual(result["purpose"], "pre_review_candidate_byte_inventory")
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["review_started"])
        self.assertFalse(result["review_completed"])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["rights_status"], "unverified_local_only")
        self.assertEqual(result["missing_required_slots"], [])
        self.assertTrue(all(set(row) == {"role", "size_bytes", "sha256"}
                            for row in result["artifacts"]))
        encoded = raw.decode("ascii")
        for sensitive in (*self.media_names, str(self.root), "synthetic video bytes",
                          "synthetic private bytes", "synthetic-token"):
            self.assertNotIn(sensitive, encoded)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual((output / "pre_review_candidate_byte_inventory.json").stat().st_mode & 0o777,
                         0o600)

    def test_inventory_binds_current_opening_gate_and_its_own_procedure(self):
        required_code_roles = {
            "pre_review_integrity_receipt_verifier",
            "review_release_order_auditor",
            "review_ceremony_signing_cli",
            "completed_review_attestation",
            "qta_prediction_link_attestation",
            "qta_prediction_link_validator",
            "qta_prediction_link_renderer",
            "actual_qa_source_adapter",
            "qta_actual_qa_source_contract",
            "qta_actual_qa_source_binding_cli",
            "qta_actual_prebaseline_diagnostic",
            "qta_stage1_bundle_audit",
            "qta_stage1_source_pin_adapter",
            "qta_prescore_mapping_binding",
            "qta_prescore_review_row_adapter",
            "qta_live_batch_binding",
            "annotation_clock_calibration",
            "claim_input_manifest_builder",
            "claim_text_quality",
            "necessity_choice_margin_utils",
            "presentation_time_utils",
            "private_output_paths",
            "pre_review_inventory_builder",
            "blinded_pair_review_validator",
            "combined_review_envelope",
        }
        self.assertTrue(required_code_roles.issubset(CODE_FILES))
        self.assertEqual(CODE_FILES["review_html_renderer"],
                         "scripts/render_nextgqa_claim_review_html.py")
        self.assertEqual(CODE_FILES["combined_review_envelope"],
                         "scripts/nextgqa_combined_review_envelope.py")
        self.assertIn("review_runbook", PUBLIC_OR_SOURCE_FILES)
        self.assertIn("review_runbook_ja", PUBLIC_OR_SOURCE_FILES)
        self.assertIn("qta_protocol_draft", PUBLIC_OR_SOURCE_FILES)
        self.assertIn("qta_treatment_selection", PUBLIC_OR_SOURCE_FILES)
        output = self.root / PRIVATE_ROOT / "inventory-opening-gate-bound"
        result = build_inventory(self.root, output)
        role_rows = {row["role"]: row for row in result["artifacts"]}
        roles = set(role_rows)
        self.assertTrue(required_code_roles.issubset(roles))
        for role in ("review_html_renderer", "combined_review_envelope"):
            relative = CODE_FILES[role]
            expected_source = ("# synthetic source " + relative + "\n").encode()
            self.assertEqual(role_rows[role]["sha256"],
                             hashlib.sha256(expected_source).hexdigest())
        expected_japanese_runbook = b"synthetic public bytes review_runbook_ja"
        self.assertEqual(role_rows["review_runbook_ja"]["size_bytes"],
                         len(expected_japanese_runbook))
        self.assertEqual(role_rows["review_runbook_ja"]["sha256"],
                         hashlib.sha256(expected_japanese_runbook).hexdigest())
        self.assertIn("review_runbook", roles)
        self.assertIn("review_runbook_ja", roles)
        self.assertIn("qta_protocol_draft", roles)
        self.assertIn("qta_treatment_selection", roles)
        self.assertNotIn("python_source_import_closure",
                         result["missing_required_slots"])
        self.assertEqual(result["missing_required_slots"], [])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])

    def test_inventory_can_bind_an_explicit_new_review_package(self):
        package_relative = PRIVATE_ROOT / "reviewer_package_v6_20261018_0243"
        package = self.root / package_relative
        package.mkdir(mode=0o700)
        html = b"synthetic repaired disclosure page"
        (package / "index.html").write_bytes(html)
        (package / "index.html").chmod(0o600)
        for name in self.media_names:
            source = self.staged / name
            target = package / name
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
        html_relative = PRIVATE_ROOT / "claim_review_v6_20261018_0243.html"
        standalone = self.root / html_relative
        standalone.write_bytes(html)
        standalone.chmod(0o600)

        output = self.root / PRIVATE_ROOT / "inventory-explicit-package"
        result = build_inventory(
            self.root, output, package_dir=package_relative,
            staged_media_dir=STAGED_MEDIA_DIR,
            standalone_review_html=html_relative)
        roles = {row["role"]: row["sha256"] for row in result["artifacts"]}
        self.assertEqual(result["missing_required_slots"], [])
        self.assertEqual(roles["review_package_html"], roles["standalone_review_html"])
        self.assertNotEqual(
            roles["review_package_html"],
            hashlib.sha256(b"synthetic review page").hexdigest())
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_candidate_path_overrides_are_all_or_none_and_private(self):
        output = self.root / PRIVATE_ROOT / "inventory-invalid-overrides"
        with self.assertRaisesRegex(ValueError, "supplied_together"):
            build_inventory(self.root, output, package_dir=PACKAGE_DIR)
        with self.assertRaisesRegex(ValueError, "repository_private_path"):
            build_inventory(
                self.root, output,
                package_dir=Path("data/private_nextgqa/../outside"),
                staged_media_dir=STAGED_MEDIA_DIR,
                standalone_review_html=PRIVATE_FILES["standalone_review_html"])

    def test_unmapped_local_python_import_blocks_inventory(self):
        scripts = self.root / "scripts"
        (scripts / "audit_nextgqa_actual_qa_source_binding.py").write_text(
            "import inventory_closure_probe\n", encoding="utf-8")
        (scripts / "inventory_closure_probe.py").write_text(
            "# unregistered local dependency\n", encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-unmapped-import"
        result = build_inventory(self.root, output)
        self.assertEqual(result["status"], "candidate_only")
        self.assertIn("python_source_import_closure", result["missing_required_slots"])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_from_scripts_import_form_fails_closed(self):
        scripts = self.root / "scripts"
        (scripts / "audit_nextgqa_actual_qa_source_binding.py").write_text(
            "from scripts import inventory_closure_probe\n", encoding="utf-8")
        (scripts / "inventory_closure_probe.py").write_text(
            "# unregistered local dependency\n", encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-from-scripts-import"
        result = build_inventory(self.root, output)
        self.assertIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_scripts_namespace_import_fails_closed(self):
        scripts = self.root / "scripts"
        (scripts / "audit_nextgqa_actual_qa_source_binding.py").write_text(
            "import scripts as local_scripts\n"
            "helper = local_scripts.inventory_closure_probe\n", encoding="utf-8")
        (scripts / "inventory_closure_probe.py").write_text(
            "# unregistered local dependency\n", encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-scripts-namespace-import"
        result = build_inventory(self.root, output)
        self.assertIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_repo_root_module_import_fails_closed(self):
        importer = self.root / "scripts" / "audit_nextgqa_actual_qa_source_binding.py"
        importer.write_text("import local_repo_helper\n", encoding="utf-8")
        (self.root / "local_repo_helper.py").write_text(
            "# unregistered repository-local dependency\n", encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-root-module-import"
        result = build_inventory(self.root, output)
        self.assertIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_dynamic_local_imports_are_resolved_or_fail_closed(self):
        importer = self.root / "scripts" / "audit_nextgqa_actual_qa_source_binding.py"
        local_module = self.root / "scripts" / "inventory_closure_probe.py"
        local_module.write_text("# unregistered local dependency\n", encoding="utf-8")
        importer.write_text(
            "import importlib as il\n"
            "il.import_module('inventory_closure_probe')\n", encoding="utf-8")
        literal_output = self.root / PRIVATE_ROOT / "inventory-dynamic-local-import"
        literal_result = build_inventory(self.root, literal_output)
        self.assertIn("python_source_import_closure",
                      literal_result["missing_required_slots"])

        importer.write_text(
            "from importlib import import_module as load_module\n"
            "load_module(module_name)\n", encoding="utf-8")
        variable_output = self.root / PRIVATE_ROOT / "inventory-dynamic-variable-import"
        variable_result = build_inventory(self.root, variable_output)
        self.assertIn("python_source_import_closure",
                      variable_result["missing_required_slots"])

        importer.write_text(
            "import importlib\n"
            "loader = importlib.import_module\n"
            "loader('inventory_closure_probe')\n", encoding="utf-8")
        alias_output = self.root / PRIVATE_ROOT / "inventory-aliased-dynamic-import"
        alias_result = build_inventory(self.root, alias_output)
        self.assertIn("python_source_import_closure",
                      alias_result["missing_required_slots"])

        importer.write_text(
            "import builtins\n"
            "loader = builtins.__import__\n"
            "loader('inventory_closure_probe')\n", encoding="utf-8")
        builtin_alias_output = self.root / PRIVATE_ROOT / "inventory-builtin-import-alias"
        builtin_alias_result = build_inventory(self.root, builtin_alias_output)
        self.assertIn("python_source_import_closure",
                      builtin_alias_result["missing_required_slots"])

        importer.write_text(
            "import builtins\n"
            "builtins.__import__(module_name)\n", encoding="utf-8")
        builtin_dynamic_output = self.root / PRIVATE_ROOT / "inventory-builtin-dynamic-import"
        builtin_dynamic_result = build_inventory(self.root, builtin_dynamic_output)
        self.assertIn("python_source_import_closure",
                      builtin_dynamic_result["missing_required_slots"])

        importer.write_text(
            "import importlib\n"
            "importlib.import_module('.inventory_closure_probe', package='scripts')\n",
            encoding="utf-8")
        relative_output = self.root / PRIVATE_ROOT / "inventory-relative-dynamic-import"
        relative_result = build_inventory(self.root, relative_output)
        self.assertIn("python_source_import_closure",
                      relative_result["missing_required_slots"])

        importer.write_text(
            "import importlib\n"
            "getattr(importlib, 'import_module')(module_name)\n", encoding="utf-8")
        reflected_output = self.root / PRIVATE_ROOT / "inventory-reflected-importer"
        reflected_result = build_inventory(self.root, reflected_output)
        self.assertIn("python_source_import_closure",
                      reflected_result["missing_required_slots"])

    def test_registered_script_with_repo_root_name_collision_fails_closed(self):
        importer = self.root / "scripts" / "audit_nextgqa_actual_qa_source_binding.py"
        importer.write_text(
            "from nextgqa_actual_qa_source_adapter import build_actual_qa_source\n",
            encoding="utf-8")
        (self.root / "nextgqa_actual_qa_source_adapter.py").write_text(
            "# ambiguous repository-root module\n", encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-root-collision"
        result = build_inventory(self.root, output)
        self.assertIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_imports_already_in_inventory_keep_closure_complete(self):
        scripts = self.root / "scripts"
        (scripts / "audit_nextgqa_actual_qa_source_binding.py").write_text(
            "from nextgqa_actual_qa_source_adapter import build_actual_qa_source\n",
            encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-complete-import"
        result = build_inventory(self.root, output)
        self.assertNotIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["missing_required_slots"], [])

    def test_qta_binding_cli_direct_contract_dependency_is_registered(self):
        cli = self.root / "scripts" / "run_qta_actual_qa_source_binding_smoke.py"
        cli.write_text(
            "from qta_actual_qa_source_contract import bind_qta_actual_qa_source\n",
            encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-qta-binding-cli-dependency"
        result = build_inventory(self.root, output)
        roles = {row["role"] for row in result["artifacts"]}
        self.assertIn("qta_actual_qa_source_binding_cli", roles)
        self.assertIn("qta_actual_qa_source_contract", roles)
        self.assertNotIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["missing_required_slots"], [])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_qta_phase_plan_transitive_dependencies_are_registered(self):
        scripts = self.root / "scripts"
        (scripts / "nextgqa_prediction_link_review.py").write_text(
            "from qta_phase2_plan_manifest import validate_baseline_pq_manifest\n",
            encoding="utf-8")
        (scripts / "qta_phase2_plan_manifest.py").write_text(
            "from qta_nine_branch_plan import BRANCH_ORDER\n"
            "from qta_attention_call_receipts import validate_receipt\n",
            encoding="utf-8")
        output = self.root / PRIVATE_ROOT / "inventory-qta-phase-dependencies"
        result = build_inventory(self.root, output)
        roles = {row["role"] for row in result["artifacts"]}
        self.assertTrue({"qta_phase2_plan_manifest", "qta_nine_branch_plan",
                         "qta_attention_call_receipts"}.issubset(roles))
        self.assertNotIn("python_source_import_closure", result["missing_required_slots"])
        self.assertEqual(result["missing_required_slots"], [])
        self.assertFalse(result["joint_freeze"])
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_unsupported_relative_or_package_imports_block_inventory(self):
        scripts = self.root / "scripts"
        importer = scripts / "audit_nextgqa_actual_qa_source_binding.py"
        importer.write_text("from .relative_helper import build\n", encoding="utf-8")
        relative_output = self.root / PRIVATE_ROOT / "inventory-relative-import"
        relative_result = build_inventory(self.root, relative_output)
        self.assertIn("python_source_import_closure",
                      relative_result["missing_required_slots"])

        importer.write_text("from inventory_helper import build\n", encoding="utf-8")
        package = scripts / "inventory_helper"
        package.mkdir()
        (package / "__init__.py").write_text("def build(): pass\n", encoding="utf-8")
        package_output = self.root / PRIVATE_ROOT / "inventory-package-import"
        package_result = build_inventory(self.root, package_output)
        self.assertIn("python_source_import_closure",
                      package_result["missing_required_slots"])

    def test_media_hash_or_basename_difference_is_missing_slot_not_pass(self):
        (self.staged / self.media_names[0]).write_bytes(b"changed synthetic video bytes")
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-hash-mismatch")
        self.assertIn("package_staged_media_byte_match", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_media_basename_difference_is_missing_slot_not_pass(self):
        old_path = self.staged / self.media_names[0]
        new_path = self.staged / "synthetic-other-token.mp4"
        old_path.rename(new_path)
        new_path.chmod(0o600)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-name-mismatch")
        self.assertIn("package_staged_media_basename_set_match", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])

    def test_private_input_symlink_or_wrong_mode_is_missing(self):
        symlink_path = self.root / PRIVATE_FILES["claim_input_manifest"]
        original = symlink_path.read_bytes()
        symlink_path.unlink()
        target = self.root / "synthetic-target"
        target.write_bytes(original)
        symlink_path.symlink_to(target)
        wrong_mode_path = self.root / PRIVATE_FILES["review_custody"]
        wrong_mode_path.chmod(0o644)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-invalid-inputs")
        self.assertIn("claim_input_manifest", result["missing_required_slots"])
        self.assertIn("review_custody", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["score_ready"])

    def test_extra_package_file_or_wrong_private_directory_mode_is_missing(self):
        extra = self.package / "extra.txt"
        extra.write_bytes(b"unplanned synthetic file")
        extra.chmod(0o600)
        self.package.chmod(0o755)
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-extra-file")
        self.assertIn("review_package_exact_three_media_and_index",
                      result["missing_required_slots"])
        self.assertIn("review_package_contents", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["joint_freeze"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_output_symlink_is_rejected_without_touching_target(self):
        target = self.root / PRIVATE_ROOT / "inventory-symlink-target"
        target.mkdir(mode=0o700)
        marker = target / "preserve.txt"
        marker.write_text("synthetic")
        link = self.root / PRIVATE_ROOT / "inventory-output-link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "output_directory_already_exists|output_directory_create_failed"):
            build_inventory(self.root, link)
        self.assertEqual(marker.read_text(), "synthetic")

    def test_missing_input_is_reported_without_authorizing(self):
        (self.root / PRIVATE_FILES["clock_anchor_plan"]).unlink()
        result = build_inventory(self.root, self.root / PRIVATE_ROOT / "inventory-missing")
        self.assertIn("clock_anchor_plan", result["missing_required_slots"])
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["score_ready"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / PRIVATE_ROOT / "inventory-existing"
        output.mkdir(mode=0o700)
        marker = output / "preserve.txt"
        marker.write_text("synthetic")
        with self.assertRaisesRegex(ValueError, "output_directory_already_exists"):
            build_inventory(self.root, output)
        self.assertEqual(marker.read_text(), "synthetic")


if __name__ == "__main__":
    unittest.main()
