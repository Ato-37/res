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
    "run_task_validity_audit_and_publish.py",
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
    "run_task_validity_audit_and_publish.py": "legacy_evidence_probe_task_validity",
    "activation_patching_sweep.py": "legacy_activation_patching",
    "activation_steering_spike.py": "legacy_activation_steering",
    "soft_ablation_sweep.py": "legacy_soft_ablation",
    "head_ablation_spike.py": "legacy_head_ablation",
    "head_selection_unsupervised.py": "legacy_unsupervised_head_selection",
    "steering_layer_survey.py": "legacy_steering_layer_survey",
    "grounding_check.py": "legacy_grounding_check",
    "run_faithfulness_validation_pilot.py": "legacy_head_faithfulness_pilot",
}
GATE_CONTROLLED_LOADERS = {
    "necessity_choice_margin_test.py",
    "run_nextgqa_gold_interval_margin_pilot.py",
}
USER_AUTHORIZED_LOCAL_DATA_PILOTS = {
    "generate_nextgqa_claim_review_pilot.py",
}
ALLOWED_ENGINEERING_LOADERS = {
    "audit_legacy_video_time_metadata.py",
    "audit_qwen_claim_tokenization_cpu.py",
    "audit_qwen_pts_to_processor_cpu.py",
    "run_dataset_video_gpu_smoke.py",
    "run_claim_mask_real_video_gpu_smoke.py",
    "run_claim_mask_real_video_capacity_sweep.py",
    "run_nextgqa_real_video_gpu_smoke.py",
    "run_qwen_frozen_teacher_forcing_gpu_smoke.py",
    "run_qwen_sequence_length_numerics_gpu_diagnostic.py",
    "run_qwen_temporal_position_gpu_check.py",
    "run_qwen_video_batch_throughput_gpu_smoke.py",
    "run_qwen_video_cache_gpu_smoke.py",
    "smoke_qwen_attention.py",
    "smoke_qwen_video_time_alignment.py",
    "spike_test_attention.py",
    "spike_test_video_attention.py",
}
SYNTHETIC_CPU_MODEL_CONSTRUCTORS = {
    "audit_qwen_cache_positions_cpu.py": "Qwen2_5_VLForConditionalGeneration",
}
WEIGHT_LOADER_APIS = {
    "torch.load",
    "torch.jit.load",
    "torch.load_state_dict",
    "safetensors.torch.load_file",
    "safetensors.torch.safe_open",
    "safetensors.safe_open",
}


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _weight_loader_calls(tree):
    """Find common weight-loading APIs, including direct import aliases."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name in {"torch", "torch.jit", "safetensors", "safetensors.torch"}:
                    # Without `as`, Python binds only the root package name:
                    # `import torch.jit` makes `torch` refer to `torch`.
                    bound_name = item.asname or item.name.split(".")[0]
                    aliases[bound_name] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"torch", "torch.jit", "safetensors", "safetensors.torch"}:
                for item in node.names:
                    aliases[item.asname or item.name] = f"{module}.{item.name}"

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if not name:
            continue
        root, dot, suffix = name.partition(".")
        canonical = f"{aliases.get(root, root)}{dot}{suffix}"
        if (canonical in WEIGHT_LOADER_APIS or name.endswith(".from_pretrained")
                or name.endswith(".load_state_dict")):
            found.add(canonical)
    return found


class ClosedResearchBranchGuardTests(unittest.TestCase):
    def test_every_literal_pretrained_loader_has_an_explicit_class(self):
        discovered = set()
        for path in (REPO / "scripts").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "from_pretrained"
                   for node in ast.walk(tree)):
                discovered.add(path.name)
        expected = (set(GUARDED_ENTRYPOINTS) & discovered
                    | GATE_CONTROLLED_LOADERS | ALLOWED_ENGINEERING_LOADERS
                    | USER_AUTHORIZED_LOCAL_DATA_PILOTS)
        self.assertEqual(discovered, expected)

    def test_other_common_weight_loaders_are_explicitly_classified(self):
        discovered = set()
        for path in (REPO / "scripts").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if _weight_loader_calls(tree):
                discovered.add(path.name)
        expected = (set(GUARDED_ENTRYPOINTS) | GATE_CONTROLLED_LOADERS
                    | ALLOWED_ENGINEERING_LOADERS | USER_AUTHORIZED_LOCAL_DATA_PILOTS)
        self.assertLessEqual(discovered, expected)

    def test_common_loader_aliases_are_detected(self):
        examples = (
            "import torch as t\nt.load('weights.pt')",
            "import torch.jit\ntorch.jit.load('weights.pt')",
            "from torch import load as read_weights\nread_weights('weights.pt')",
            "import safetensors.torch as st\nst.load_file('weights.safetensors')",
            "import safetensors.torch\nsafetensors.torch.load_file('weights.safetensors')",
            "import safetensors.torch\nsafetensors.torch.safe_open('weights.safetensors')",
            "from safetensors.torch import load_file as read_weights\nread_weights('weights.safetensors')",
            "from safetensors import safe_open as open_weights\nopen_weights('weights.safetensors')",
            "model.load_state_dict(state)",
        )
        for source in examples:
            with self.subTest(source=source):
                self.assertTrue(_weight_loader_calls(ast.parse(source)))
        self.assertFalse(_weight_loader_calls(ast.parse("json.load(stream)")))

    def test_every_list_referencing_a_retired_cli_is_guarded(self):
        retired = set(GUARDED_ENTRYPOINTS)
        for path in (REPO / "scripts").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.List):
                    continue
                targets = {
                    Path(value).name
                    for value in (child.value for child in ast.walk(node)
                                  if isinstance(child, ast.Constant)
                                  and isinstance(child.value, str))
                    if value.startswith("scripts/") and value.endswith(".py")
                }
                if targets & retired:
                    self.assertIn(
                        path.name, GUARDED_ENTRYPOINTS,
                        f"{path.name} launches retired CLI(s) {sorted(targets & retired)} "
                        "without its own early-stop guard",
                    )

    def test_known_random_config_model_constructor_is_classified_synthetic_cpu(self):
        discovered = set()
        for filename, constructor in SYNTHETIC_CPU_MODEL_CONSTRUCTORS.items():
            tree = ast.parse((REPO / "scripts" / filename).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == constructor
                     and node.args and isinstance(node.args[0], ast.Name)
                     and node.args[0].id == "config"]
            self.assertTrue(calls, f"{filename} no longer constructs the declared random config model")
            discovered.add(filename)
        self.assertEqual(discovered, set(SYNTHETIC_CPU_MODEL_CONSTRUCTORS))

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
