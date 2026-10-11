"""Compare pinned 4.49 and installed Qwen RoPE geometry helpers on fixed CPU fixtures.

No model weights, media, processor, inference, downloads, or generation are used.
The 4.49 method is AST-extracted only after the cached official-source bytes match
the expected SHA-256. The three synthetic geometries are fixed prospectively.

Usage::

    .venv/bin/python scripts/audit_qwen_rope_version_geometry_cpu.py
    .venv/bin/python scripts/audit_qwen_rope_version_geometry_cpu.py \
      --output results/qwen_rope_version_geometry_cpu_20261013_01.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
from typing import Any


LEGACY_SOURCE = Path(__file__).resolve().parents[1] / (
    "datasets/runtime_source_audit/private/transformers_v4_49_0_qwen.py"
)
LEGACY_SOURCE_SHA256 = "076a9b98955abe003ad90f9240372996bbc72d9f24e303346f23b77735106cd7"
LEGACY_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v4.49.0/"
    "src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
)
INSTALLED_MODEL_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.17.0/"
    "src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
)
INSTALLED_VISION_UTILS_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.17.0/"
    "src/transformers/vision_utils.py"
)
EXPECTED_TRANSFORMERS_VERSION = "5.17.0"
SPATIAL_MERGE_SIZE = 2
TOKENS_PER_SECOND = 2
VISION_START_ID = 900_001
VIDEO_TOKEN_ID = 900_002
IMAGE_TOKEN_ID = 900_003

FIXED_CASES = (
    {"name": "tiny_subsecond", "grid_thw": (8, 4, 4), "prefix_tokens_including_vision_start": 2,
     "suffix_tokens": 2, "seconds_per_grid": 0.75},
    {"name": "integer_interval_control", "grid_thw": (8, 4, 4), "prefix_tokens_including_vision_start": 2,
     "suffix_tokens": 2, "seconds_per_grid": 5.0},
    {"name": "noaa_derived_geometry", "grid_thw": (2, 24, 46), "prefix_tokens_including_vision_start": 15,
     "suffix_tokens": 17, "seconds_per_grid": 199.9998016357422},
)
CONFIG_SPEC = {
    "cases": [dict(case) for case in FIXED_CASES],
    "spatial_merge_size": SPATIAL_MERGE_SIZE,
    "tokens_per_second": TOKENS_PER_SECOND,
    "synthetic_vision_start_precedes_video_tokens": True,
    "prefix_token_count_includes_vision_start_marker": True,
    "candidate_policy": "replace_only_video_temporal_ids_using_fractional_interval; retain_installed_spatial_and_text_ids",
    "expected_transformers_version": EXPECTED_TRANSFORMERS_VERSION,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))


def _strip_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.FunctionDef:
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("unexpected legacy method type")
    function.decorator_list = []
    function.returns = None
    function.type_comment = None
    args = function.args
    for argument in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        argument.annotation = None
        argument.type_comment = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None
    return function


def extract_legacy_get_rope_index(source_bytes: bytes, *, expected_sha256: str) -> Any:
    """Compile only the hash-verified 4.49 class method, with annotations removed."""
    observed = sha256_bytes(source_bytes)
    if observed != expected_sha256:
        raise ValueError("legacy source hash mismatch")
    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("legacy source could not be parsed") from exc
    class_node = next((node for node in tree.body
                       if isinstance(node, ast.ClassDef)
                       and node.name == "Qwen2_5_VLForConditionalGeneration"), None)
    if class_node is None:
        raise ValueError("legacy source class missing")
    method = next((node for node in class_node.body
                   if isinstance(node, ast.FunctionDef) and node.name == "get_rope_index"), None)
    if method is None:
        raise ValueError("legacy rope method missing")
    isolated_method = _strip_annotations(copy.deepcopy(method))
    module = ast.fix_missing_locations(ast.Module(body=[isolated_method], type_ignores=[]))
    namespace = {"torch": __import__("torch")}
    try:
        exec(compile(module, "<hash-verified-transformers-v4.49-qwen-method>", "exec"), namespace)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError("legacy rope method could not be compiled") from exc
    return namespace["get_rope_index"]


def _fixture_inputs(case: dict[str, Any], torch: Any) -> tuple[Any, Any, Any, Any]:
    t, h, w = case["grid_thw"]
    grid_tokens = t * (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)
    prefix_count = case["prefix_tokens_including_vision_start"]
    suffix_count = case["suffix_tokens"]
    input_ids = (
        [101 + index for index in range(prefix_count - 1)]
        + [VISION_START_ID]
        + [VIDEO_TOKEN_ID] * grid_tokens
        + [201 + index for index in range(suffix_count)]
    )
    mm_types = [0] * prefix_count + [2] * grid_tokens + [0] * suffix_count
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    types_tensor = torch.tensor([mm_types], dtype=torch.int32)
    grid_tensor = torch.tensor([case["grid_thw"]], dtype=torch.long)
    seconds_tensor = torch.tensor([case["seconds_per_grid"]], dtype=torch.float32)
    return ids_tensor, types_tensor, grid_tensor, seconds_tensor


def _simple_model(torch: Any, installed_class: Any) -> Any:
    config = SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        tokens_per_second=TOKENS_PER_SECOND,
    ))
    instance = SimpleNamespace(config=config)
    instance.get_vision_position_ids = MethodType(installed_class.get_vision_position_ids, instance)
    instance.get_rope_index = MethodType(installed_class.get_rope_index, instance)
    return instance


def _per_axis_changed(left: Any, right: Any, torch: Any) -> list[int]:
    difference = left != right
    return [int(difference[axis].sum().item()) for axis in range(3)]


def _delta_values(value: Any) -> list[int]:
    return [int(item) for item in value.reshape(-1).tolist()]


def _position_vector(position_ids: Any, index: int) -> list[int]:
    return [int(position_ids[axis, 0, index].item()) for axis in range(3)]


def _case_geometry(case: dict[str, Any], *, legacy_method: Any, installed_model: Any,
                   torch: Any) -> dict[str, Any]:
    ids, mm_types, grid, seconds = _fixture_inputs(case, torch)
    text_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            tokens_per_second=TOKENS_PER_SECOND,
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
    )
    legacy_self = SimpleNamespace(config=text_config)
    legacy_positions, legacy_delta = legacy_method(
        legacy_self, input_ids=ids, video_grid_thw=grid, second_per_grid_ts=seconds
    )
    installed_positions, installed_delta = installed_model.get_rope_index(
        ids, mm_types, video_grid_thw=grid, second_per_grid_ts=seconds
    )

    video_start = case["prefix_tokens_including_vision_start"]
    video_count = int(grid[0, 0].item()) * (
        int(grid[0, 1].item()) // SPATIAL_MERGE_SIZE
    ) * (int(grid[0, 2].item()) // SPATIAL_MERGE_SIZE)
    video_end = video_start + video_count
    prefix_end = video_start
    sequence_length = ids.shape[-1]
    suffix_start = video_end

    float32_seconds = float(seconds[0].item())
    fractional_interval = float32_seconds * TOKENS_PER_SECOND
    current_integer_interval = int(float32_seconds) * TOKENS_PER_SECOND
    candidate_positions = installed_positions.clone()
    # The 5.17 helper's time_interval argument is integral in its caller's
    # current policy; a fractional float passed to that helper is not a valid
    # candidate API (its temporal tensor then has a different dtype from H/W).
    # Compute only the proposed video temporal row directly with the legacy
    # float32 multiply-then-long formula, retaining installed spatial/text IDs.
    candidate_temporal = (
        torch.arange(int(grid[0, 0].item()), device=ids.device)
        * seconds[0]
        * TOKENS_PER_SECOND
    ).long() + case["prefix_tokens_including_vision_start"]
    patches_per_tick = (int(grid[0, 1].item()) // SPATIAL_MERGE_SIZE) * (
        int(grid[0, 2].item()) // SPATIAL_MERGE_SIZE
    )
    candidate_temporal = candidate_temporal.repeat_interleave(patches_per_tick)
    candidate_positions[0, 0, video_start:video_end] = candidate_temporal.to(
        candidate_positions.dtype
    )
    candidate_delta = candidate_positions[:, 0, :].max() + 1 - sequence_length

    axis_names = ("temporal", "height", "width")
    changed = _per_axis_changed(legacy_positions, installed_positions, torch)
    candidate_changed = _per_axis_changed(legacy_positions, candidate_positions, torch)
    video_changed = _per_axis_changed(
        legacy_positions[:, :, video_start:video_end],
        installed_positions[:, :, video_start:video_end], torch,
    )
    prefix_changed = _per_axis_changed(
        legacy_positions[:, :, :prefix_end], installed_positions[:, :, :prefix_end], torch
    )
    suffix_changed = _per_axis_changed(
        legacy_positions[:, :, suffix_start:], installed_positions[:, :, suffix_start:], torch
    )
    anchors = {}
    anchor_indices = {
        "video_first": video_start,
        "video_last": video_end - 1,
        "suffix_first": suffix_start,
        "suffix_last": sequence_length - 1,
    }
    for name, index in anchor_indices.items():
        anchors[name] = {
            "sequence_index": index,
            "legacy_v4_49": _position_vector(legacy_positions, index),
            "installed_v5_17": _position_vector(installed_positions, index),
            "candidate_temporal_only": _position_vector(candidate_positions, index),
        }

    return {
        "case": case["name"],
        "grid_thw": list(case["grid_thw"]),
        "prefix_token_count_including_vision_start_marker": case["prefix_tokens_including_vision_start"],
        "suffix_token_count_in_fixture": case["suffix_tokens"],
        "video_position_count": video_count,
        "sequence_length": sequence_length,
        "seconds_per_grid_input": case["seconds_per_grid"],
        "seconds_per_grid_float32": float32_seconds,
        "legacy_temporal_formula": "long(arange(T) * seconds_per_grid_float32 * tokens_per_second) + vision_start_context_offset",
        "installed_temporal_policy": "tokens_per_second * int(seconds_per_grid_float32)",
        "installed_effective_integer_interval": current_integer_interval,
        "candidate_temporal_policy": "fractional_seconds_per_grid_float32 * tokens_per_second; replace video temporal axis only",
        "candidate_effective_float_interval": fractional_interval,
        "changed_count_by_axis_legacy_vs_installed": dict(zip(axis_names, changed)),
        "video_changed_count_by_axis_legacy_vs_installed": dict(zip(axis_names, video_changed)),
        "prefix_text_changed_count_by_axis_legacy_vs_installed": dict(zip(axis_names, prefix_changed)),
        "suffix_text_changed_count_by_axis_legacy_vs_installed": dict(zip(axis_names, suffix_changed)),
        "candidate_changed_count_by_axis_legacy_vs_candidate": dict(zip(axis_names, candidate_changed)),
        "legacy_mrope_delta": _delta_values(legacy_delta),
        "installed_mrope_delta": _delta_values(installed_delta),
        "candidate_mrope_delta": _delta_values(candidate_delta.reshape(1, 1)),
        "candidate_temporal_only_matches_full_legacy_positions": bool(
            torch.equal(candidate_positions, legacy_positions)
        ),
        "anchor_positions": anchors,
    }


def _source_sha256_for_object(obj: Any) -> str | None:
    try:
        source = Path(inspect.getfile(obj))
        return sha256_bytes(source.read_bytes())
    except (OSError, TypeError, ValueError):
        return None


def _checkout_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(*, repo_root: Path, runner: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.vision_utils import get_vision_position_ids

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError("unexpected installed transformers version")
    source_bytes = LEGACY_SOURCE.read_bytes()
    legacy_source_hash = sha256_bytes(source_bytes)
    if legacy_source_hash != LEGACY_SOURCE_SHA256:
        raise ValueError("legacy source hash mismatch")
    legacy_method = extract_legacy_get_rope_index(
        source_bytes, expected_sha256=LEGACY_SOURCE_SHA256
    )
    with torch.device("cpu"):
        installed_model = _simple_model(torch, Qwen2_5_VLModel)
        cases = [
            _case_geometry(case, legacy_method=legacy_method, installed_model=installed_model, torch=torch)
            for case in FIXED_CASES
        ]
    model_source_hash = _source_sha256_for_object(Qwen2_5_VLModel.get_rope_index)
    vision_utils_hash = _source_sha256_for_object(get_vision_position_ids)
    return {
        "schema_version": 1,
        "status": "ok",
        "requested_cases": len(FIXED_CASES),
        "effective_cases": len(cases),
        "failures": 0,
        "purpose": "fixed_synthetic_qwen_rope_version_geometry_cpu_comparison",
        "scope": "position_id_helpers_only_no_weights_no_media_no_processor_no_forward_no_generation",
        "device": "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "sources": {
            "legacy_v4_49_source_url": LEGACY_SOURCE_URL,
            "legacy_v4_49_source_bytes": len(source_bytes),
            "legacy_v4_49_source_sha256": legacy_source_hash,
            "installed_v5_17_model_source_url": INSTALLED_MODEL_SOURCE_URL,
            "installed_modeling_source_sha256": model_source_hash,
            "installed_v5_17_vision_utils_source_url": INSTALLED_VISION_UTILS_SOURCE_URL,
            "installed_vision_utils_source_sha256": vision_utils_hash,
            "runner_sha256": sha256_bytes(runner.read_bytes()),
            "config_sha256": sha256_json(CONFIG_SPEC),
        },
        "configuration": {
            "spatial_merge_size": SPATIAL_MERGE_SIZE,
            "tokens_per_second": TOKENS_PER_SECOND,
            "vision_start_marker_precedes_video_placeholder_run": True,
            "prefix_token_count_includes_vision_start_marker": True,
            "cases_are_fixed_synthetic_and_not_outcome_selected": True,
        },
        "cases": cases,
        "interpretation_limit": (
            "These are CPU geometry-helper comparisons on three fixed synthetic grids. They do not establish model behavior, "
            "visual content effects, QA or explanation validity, or a universal runtime difference. The candidate modifies only "
            "video temporal IDs and intentionally leaves installed text/spatial positions unchanged."
        ),
        "run": {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "checkout_head": _checkout_head(repo_root),
            "seed": None,
        },
    }


def safe_output_path(value: str | None, repo_root: Path) -> Path:
    if value is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = repo_root / "results" / f"qwen_rope_version_geometry_cpu_{timestamp}.json"
    else:
        output = Path(value)
        if not output.is_absolute():
            output = repo_root / output
    output = output.resolve()
    results = (repo_root / "results").resolve()
    if output.parent != results or output.suffix.lower() != ".json":
        raise ValueError("output must be a JSON file directly under repository results/")
    return output


def write_report(report: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Unique JSON path directly under repository results/")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        output = safe_output_path(args.output, repo_root)
        if output.exists():
            raise FileExistsError
        report = build_report(repo_root=repo_root, runner=Path(__file__).resolve())
        write_report(report, output)
    except FileExistsError:
        print("RoPE geometry CPU check: output already exists; refusing overwrite", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, ImportError, TypeError, AttributeError, KeyError):
        print("RoPE geometry CPU check: source or helper comparison failed", file=sys.stderr)
        return 2
    print(
        "RoPE geometry CPU check: "
        f"cases={report['effective_cases']}/{report['requested_cases']} | "
        f"candidate reproduces full legacy positions="
        f"{sum(case['candidate_temporal_only_matches_full_legacy_positions'] for case in report['cases'])}/"
        f"{report['effective_cases']} | artifact=results/{output.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
