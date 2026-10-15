"""Static guard-order checks for closed real-media inference entrypoints."""
import ast
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
GUARDED_ENTRYPOINTS = (
    "necessity_mask_test.py",
    "evidence_control_probe.py",
    "evidence_control_choice_probe.py",
    "evidence_control_choice_confirmation.py",
    "head_faithfulness_test.py",
    "head_grounding_scan.py",
    "head_group_activation_patching.py",
    "head_faithfulness_readout_randomization.py",
    "head_faithfulness_layer_randomization.py",
    "run_evidence_control_probe_and_publish.py",
    "run_evidence_control_choice_probe_and_publish.py",
    "run_evidence_control_choice_confirmation_and_publish.py",
    "run_head_faithfulness_and_publish.py",
    "run_head_faithfulness_randomization_and_publish.py",
    "run_head_faithfulness_layer_randomization_and_publish.py",
    "run_head_faithfulness_confirmation.py",
    "run_all_layer_and_publish.py",
    "audit_evidence_probe_task_validity.py",
    "activation_patching_sweep.py",
    "activation_steering_spike.py",
    "soft_ablation_sweep.py",
    "head_ablation_spike.py",
    "head_selection_unsupervised.py",
    "steering_layer_survey.py",
    "grounding_check.py",
    "run_faithfulness_validation_pilot.py",
)
EXPECTED_BRANCH_CODES = {
    "necessity_mask_test.py": "legacy_necessity_mask",
    "evidence_control_probe.py": "legacy_evidence_control",
    "evidence_control_choice_probe.py": "legacy_evidence_control",
    "evidence_control_choice_confirmation.py": "legacy_evidence_control",
    "head_faithfulness_test.py": "closed_head_faithfulness",
    "head_grounding_scan.py": "closed_head_search",
    "head_group_activation_patching.py": "closed_head_search",
    "head_faithfulness_readout_randomization.py": "closed_head_faithfulness",
    "head_faithfulness_layer_randomization.py": "closed_head_faithfulness",
    "run_evidence_control_probe_and_publish.py": "legacy_evidence_control",
    "run_evidence_control_choice_probe_and_publish.py": "legacy_evidence_control",
    "run_evidence_control_choice_confirmation_and_publish.py": "legacy_evidence_control",
    "run_head_faithfulness_and_publish.py": "closed_head_faithfulness",
    "run_head_faithfulness_randomization_and_publish.py": "closed_head_faithfulness",
    "run_head_faithfulness_layer_randomization_and_publish.py": "closed_head_faithfulness",
    "run_head_faithfulness_confirmation.py": "closed_head_faithfulness",
    "run_all_layer_and_publish.py": "closed_head_search",
    "audit_evidence_probe_task_validity.py": "legacy_evidence_probe_task_validity",
    "activation_patching_sweep.py": "legacy_activation_patching",
    "activation_steering_spike.py": "legacy_activation_steering",
    "soft_ablation_sweep.py": "legacy_soft_ablation",
    "head_ablation_spike.py": "legacy_head_ablation",
    "head_selection_unsupervised.py": "legacy_unsupervised_head_selection",
    "steering_layer_survey.py": "legacy_steering_layer_survey",
    "grounding_check.py": "legacy_grounding_check",
    "run_faithfulness_validation_pilot.py": "legacy_head_faithfulness_pilot",
}


class ClosedResearchBranchGuardTests(unittest.TestCase):
    def test_every_retired_cli_stops_immediately_after_argument_parsing(self):
        for filename in GUARDED_ENTRYPOINTS:
            with self.subTest(script=filename):
                tree = ast.parse((REPO / "scripts" / filename).read_text(encoding="utf-8"))
                main = next(node for node in tree.body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == "main")
                parse_index = next(i for i, node in enumerate(main.body)
                                   if isinstance(node, ast.Assign)
                                   and isinstance(node.value, ast.Call)
                                   and isinstance(node.value.func, ast.Attribute)
                                   and node.value.func.attr == "parse_args")
                guard_index = next(i for i, node in enumerate(main.body)
                                   if any(isinstance(call, ast.Call)
                                          and isinstance(call.func, ast.Name)
                                          and call.func.id == "reject_inference"
                                          for call in ast.walk(node)))
                self.assertGreater(guard_index, parse_index)
                guard_call = main.body[guard_index].value
                self.assertGreaterEqual(len(guard_call.args), 2)
                self.assertIn(filename, EXPECTED_BRANCH_CODES)
                self.assertEqual(ast.literal_eval(guard_call.args[1]),
                                 EXPECTED_BRANCH_CODES[filename])
                between = main.body[parse_index + 1:guard_index]
                self.assertLessEqual(len(between), 1)
                forbidden_calls = {"from_pretrained", "read_text", "read_bytes",
                                   "load", "loads", "open", "VideoCapture", "run",
                                   "Popen", "subprocess"}
                for statement in between:
                    for call in (node for node in ast.walk(statement)
                                 if isinstance(node, ast.Call)):
                        name = (call.func.id if isinstance(call.func, ast.Name)
                                else call.func.attr if isinstance(call.func, ast.Attribute)
                                else "")
                        self.assertNotIn(name, forbidden_calls)
                guard = main.body[guard_index]
                self.assertIsInstance(guard, ast.Expr)
                self.assertIsInstance(guard.value, ast.Call)
                self.assertIsInstance(guard.value.func, ast.Name)
                self.assertEqual(guard.value.func.id, "reject_inference")


if __name__ == "__main__":
    unittest.main()
