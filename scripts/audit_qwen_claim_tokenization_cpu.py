"""Audit Qwen chat-prefix/claim token boundaries on fixed synthetic text only.

CPU/tokenizer-and-preprocessor diagnostic. It loads the pinned processor from
the local Hugging Face cache and generates three tiny synthetic video frames;
downloads, model weights, real/private videos, QA data, and model generation
are disabled. The fixed prompt and targets are prospective fixtures, not
selected from model outcomes. Output contains case indices, lengths, equality
flags and hashes only; it never emits fixture text, frame paths, or token IDs.

Usage::

    .venv/bin/python scripts/audit_qwen_claim_tokenization_cpu.py
    .venv/bin/python scripts/audit_qwen_claim_tokenization_cpu.py \
      --output results/qwen_claim_tokenization_cpu_20261013_01.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from PIL import Image


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_CACHE = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
    / "snapshots" / MODEL_REVISION
)
FIXED_PROMPT = "Describe the visible scene briefly. Use only visual information."
FIXED_TARGETS = (
    "A",
    " A",
    "The object moves.",
    " The object moves.",
    "3",
    " 3",
    "0.5",
    "0.5 seconds.",
    "No.",
    "\nThe object moves.",
    "水中の物体。",
    "é",
)
CHOICE_LABELS = ("A", "B", "C", "D", "E")
CHOICE_PROMPT = (
    "Synthetic question?\nOptions:\nA. option one\nB. option two\n"
    "C. option three\nD. option four\nE. option five\n"
    "Choose the best answer. Respond with one letter only.\nAnswer:"
)
SYNTHETIC_VIDEO_FRAME_COUNT = 16
SYNTHETIC_VIDEO_FRAME_SIZE = 224
SYNTHETIC_MASKED_FRAME_COUNT = 4
SYNTHETIC_SELECTED_MASK_INDICES = tuple(range(SYNTHETIC_MASKED_FRAME_COUNT))
SYNTHETIC_CONTROL_MASK_INDICES = tuple(
    range(SYNTHETIC_VIDEO_FRAME_COUNT - SYNTHETIC_MASKED_FRAME_COUNT,
          SYNTHETIC_VIDEO_FRAME_COUNT)
)
FIXTURE_SPEC = {
    "prompt": FIXED_PROMPT,
    "targets": list(FIXED_TARGETS),
    "cases_are_synthetic_and_fixed_before_outcomes": True,
    "choice_prompt": CHOICE_PROMPT,
    "choice_labels": list(CHOICE_LABELS),
    "synthetic_video_frame_count": SYNTHETIC_VIDEO_FRAME_COUNT,
    "synthetic_video_frame_size": SYNTHETIC_VIDEO_FRAME_SIZE,
    "synthetic_video_cap_pixels_per_frame": False,
    "synthetic_mask_fill": "black",
    "synthetic_selected_mask_indices": list(SYNTHETIC_SELECTED_MASK_INDICES),
    "synthetic_control_mask_indices": list(SYNTHETIC_CONTROL_MASK_INDICES),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def ids_list(value: Any) -> list[int]:
    """Normalize a single tokenizer row without leaking values to output."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        if "input_ids" not in value:
            raise ValueError("tokenizer output missing input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("unexpected tokenizer output shape")
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if any(type(token_id) is not int for token_id in value):
        raise ValueError("tokenizer IDs must be integer values")
    return list(value)


def _token_hash(ids: list[int]) -> str:
    return sha256_json(ids)


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer(text, add_special_tokens=False)
    if isinstance(result, dict):
        result = result["input_ids"]
    else:
        result = result.input_ids
    return ids_list(result)


def _template_ids(processor: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    result = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    return ids_list(result)


def audit_fixed_cases(processor: Any) -> dict[str, Any]:
    """Compare separate, literal-concatenation, and complete-chat token paths."""
    tokenizer = processor.tokenizer
    messages = [{"role": "user", "content": FIXED_PROMPT}]
    rendered_prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered_prefix, str):
        raise ValueError("chat template did not render a string prefix")

    prefix_ids = _template_ids(processor, messages, add_generation_prompt=True)
    prefix_retokenized_ids = _tokenize(tokenizer, rendered_prefix)
    prefix_encoding_equal = prefix_ids == prefix_retokenized_ids
    records = []
    for index, target in enumerate(FIXED_TARGETS, start=1):
        standalone_target_ids = _tokenize(tokenizer, target)
        literal_concat_ids = _tokenize(tokenizer, rendered_prefix + target)
        direct_concat_ids = prefix_ids + standalone_target_ids

        completed_messages = messages + [{"role": "assistant", "content": target}]
        completed_chat_ids = _template_ids(
            processor, completed_messages, add_generation_prompt=False
        )
        complete_has_prefix = completed_chat_ids[:len(prefix_ids)] == prefix_ids
        contextual_suffix_ids = completed_chat_ids[len(prefix_ids):] if complete_has_prefix else []
        records.append({
            "case_index": index,
            "standalone_target_token_count": len(standalone_target_ids),
            "literal_concat_token_count": len(literal_concat_ids),
            "prefix_plus_standalone_token_count": len(direct_concat_ids),
            "literal_concat_equals_prefix_plus_standalone": literal_concat_ids == direct_concat_ids,
            "literal_concat_token_ids_sha256": _token_hash(literal_concat_ids),
            "prefix_plus_standalone_token_ids_sha256": _token_hash(direct_concat_ids),
            "completed_chat_prefix_stable": complete_has_prefix,
            "completed_chat_suffix_token_count": len(contextual_suffix_ids),
            "completed_chat_token_ids_sha256": _token_hash(completed_chat_ids),
        })
    return {
        "fixed_case_count": len(FIXED_TARGETS),
        "template_prefix_token_count": len(prefix_ids),
        "rendered_prefix_retokenized_token_count": len(prefix_retokenized_ids),
        "template_prefix_equals_retokenized_rendered_prefix": prefix_encoding_equal,
        "template_prefix_token_ids_sha256": _token_hash(prefix_ids),
        "rendered_prefix_retokenized_token_ids_sha256": _token_hash(prefix_retokenized_ids),
        "literal_concat_match_count": sum(
            record["literal_concat_equals_prefix_plus_standalone"] for record in records
        ),
        "completed_chat_prefix_stable_count": sum(
            record["completed_chat_prefix_stable"] for record in records
        ),
        "cases": records,
    }


def audit_fixed_choice_labels(processor: Any) -> dict[str, Any]:
    """Verify A-E next-token IDs at the exact chat generation boundary."""
    tokenizer = processor.tokenizer
    messages = [{"role": "user", "content": CHOICE_PROMPT}]
    prefix_ids = _template_ids(processor, messages, add_generation_prompt=True)
    records = []
    standalone_ids = []
    for label in CHOICE_LABELS:
        ids = _tokenize(tokenizer, label)
        standalone_ids.append(ids)
        complete = _template_ids(
            processor,
            messages + [{"role": "assistant", "content": label}],
            add_generation_prompt=False,
        )
        stable_prefix = complete[:len(prefix_ids)] == prefix_ids
        suffix = complete[len(prefix_ids):] if stable_prefix else []
        records.append({
            "label_index": len(records),
            "standalone_token_count": len(ids),
            "standalone_token_ids_sha256": _token_hash(ids),
            "completed_chat_prefix_stable": stable_prefix,
            "completed_chat_starts_with_standalone_label": bool(ids) and suffix[:len(ids)] == ids,
            "completed_chat_suffix_token_count": len(suffix),
            "completed_chat_token_ids_sha256": _token_hash(complete),
        })
    distinct_single_tokens = (
        all(len(ids) == 1 for ids in standalone_ids)
        and len({ids[0] for ids in standalone_ids if len(ids) == 1}) == len(CHOICE_LABELS)
    )
    gate_pass = (
        distinct_single_tokens
        and sum(row["completed_chat_prefix_stable"] for row in records) == len(CHOICE_LABELS)
        and sum(row["completed_chat_starts_with_standalone_label"] for row in records)
        == len(CHOICE_LABELS)
    )
    return {
        "label_count": len(CHOICE_LABELS),
        "distinct_single_token_labels": distinct_single_tokens,
        "gate_pass": gate_pass,
        "generation_prefix_token_count": len(prefix_ids),
        "generation_prefix_token_ids_sha256": _token_hash(prefix_ids),
        "stable_prefix_count": sum(row["completed_chat_prefix_stable"] for row in records),
        "contextual_label_match_count": sum(
            row["completed_chat_starts_with_standalone_label"] for row in records
        ),
        "labels": records,
    }


def audit_video_choice_labels(processor: Any) -> dict[str, Any]:
    """Check label boundaries after the production video-input processor expands frames."""
    from activation_patching_sweep import build_inputs, write_frame_set
    from necessity_choice_margin_utils import apply_mask_fill
    from qwen_vl_utils import process_vision_info
    import torch

    prompt = CHOICE_PROMPT
    with tempfile.TemporaryDirectory(prefix="synthetic_choice_video_") as directory:
        frame_paths = []
        frame_images = []
        for index in range(SYNTHETIC_VIDEO_FRAME_COUNT):
            color = (index * 13 % 256, index * 53 % 256, index * 97 % 256)
            path = Path(directory) / f"frame_{index}.jpg"
            frame = Image.new(
                "RGB", (SYNTHETIC_VIDEO_FRAME_SIZE, SYNTHETIC_VIDEO_FRAME_SIZE), color
            )
            frame.save(path, format="JPEG")
            frame_paths.append(str(path))
            frame_images.append(frame)

        user_message = {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": prompt},
            ],
        }
        base_inputs = build_inputs(processor, "cpu", frame_paths, prompt)
        base_ids = ids_list(base_inputs["input_ids"])

        mask_audit = {}
        base_grid = ids_list(base_inputs["video_grid_thw"]) if base_inputs.get("video_grid_thw") is not None else []
        base_pixels = base_inputs.get("pixel_values_videos")
        for condition, masked_indices in (
            ("selected_black_mask", set(SYNTHETIC_SELECTED_MASK_INDICES)),
            ("control_black_mask", set(SYNTHETIC_CONTROL_MASK_INDICES)),
        ):
            masked_frames = [
                apply_mask_fill(frame, "black") if index in masked_indices else frame.copy()
                for index, frame in enumerate(frame_images)
            ]
            mask_dir = Path(directory) / condition
            masked_paths = write_frame_set(masked_frames, str(mask_dir))
            masked_inputs = build_inputs(processor, "cpu", masked_paths, prompt)
            masked_ids = ids_list(masked_inputs["input_ids"])
            masked_grid = (
                ids_list(masked_inputs["video_grid_thw"])
                if masked_inputs.get("video_grid_thw") is not None else []
            )
            masked_pixels = masked_inputs.get("pixel_values_videos")
            same_pixel_shape = (
                base_pixels is not None and masked_pixels is not None
                and tuple(base_pixels.shape) == tuple(masked_pixels.shape)
            )
            mask_audit[condition] = {
                "masked_frame_count": len(masked_indices),
                "input_ids_equal_to_full": masked_ids == base_ids,
                "video_grid_equal_to_full": masked_grid == base_grid,
                "pixel_values_shape_equal_to_full": same_pixel_shape,
                "pixel_values_content_changed": (
                    same_pixel_shape and not torch.equal(base_pixels, masked_pixels)
                ),
                "masked_input_token_ids_sha256": _token_hash(masked_ids),
            }
        mask_gate_pass = all(
            row["input_ids_equal_to_full"]
            and row["video_grid_equal_to_full"]
            and row["pixel_values_shape_equal_to_full"]
            and row["pixel_values_content_changed"]
            for row in mask_audit.values()
        )
        records = []
        for label in CHOICE_LABELS:
            messages = [user_message, {"role": "assistant", "content": label}]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            images, videos = process_vision_info(messages)
            completed_inputs = processor(
                text=[rendered], images=images, videos=videos,
                padding=True, return_tensors="pt",
            )
            completed_ids = ids_list(completed_inputs["input_ids"])
            stable_prefix = completed_ids[:len(base_ids)] == base_ids
            suffix = completed_ids[len(base_ids):] if stable_prefix else []
            label_ids = _tokenize(processor.tokenizer, label)
            records.append({
                "label_index": len(records),
                "expanded_video_prefix_stable": stable_prefix,
                "expanded_video_target_starts_with_label": (
                    bool(label_ids) and suffix[:len(label_ids)] == label_ids
                ),
                "completed_input_token_count": len(completed_ids),
                "completed_input_token_ids_sha256": _token_hash(completed_ids),
            })
        gate_pass = all(
            row["expanded_video_prefix_stable"]
            and row["expanded_video_target_starts_with_label"]
            for row in records
        )
        return {
            "synthetic_frame_count": len(frame_paths),
            "synthetic_frame_size": SYNTHETIC_VIDEO_FRAME_SIZE,
            "cap_pixels_per_frame": False,
            "base_expanded_video_input_token_count": len(base_ids),
            "base_expanded_video_input_token_ids_sha256": _token_hash(base_ids),
            "label_count": len(records),
            "expanded_video_prefix_stable_count": sum(
                row["expanded_video_prefix_stable"] for row in records
            ),
            "expanded_video_contextual_label_match_count": sum(
                row["expanded_video_target_starts_with_label"] for row in records
            ),
            "mask_input_stability": {
                "selected_frame_count": SYNTHETIC_MASKED_FRAME_COUNT,
                "control_frame_count": SYNTHETIC_MASKED_FRAME_COUNT,
                "conditions": mask_audit,
                "gate_pass": mask_gate_pass,
            },
            "gate_pass": gate_pass and mask_gate_pass,
            "labels": records,
        }


def _source_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        value = result.stdout.strip()
        return value if value else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _local_tokenizer_source_hashes(snapshot: Path) -> dict[str, str]:
    hashes = {}
    for basename in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "preprocessor_config.json", "processor_config.json",
    ):
        source = snapshot / basename
        if source.is_file():
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[basename] = digest.hexdigest()
    return hashes


def _installed_implementation_hashes(processor: Any, tokenizer: Any) -> dict[str, str | None]:
    """Hash installed implementation files without returning their local paths."""
    from qwen_vl_utils import process_vision_info
    from necessity_choice_margin_utils import apply_mask_fill

    objects = {
        "processor_class_source": type(processor),
        "tokenizer_class_source": type(tokenizer),
        "chat_template_method_source": getattr(processor, "apply_chat_template", None),
        "video_preprocessor_source": process_vision_info,
        "mask_fill_implementation_source": apply_mask_fill,
    }
    try:
        objects["chat_template_utils_module_source"] = import_module(
            "transformers.utils.chat_template_utils"
        )
    except (ImportError, ModuleNotFoundError):
        objects["chat_template_utils_module_source"] = None
    hashes: dict[str, str | None] = {}
    builder_path = Path(__file__).resolve().with_name("activation_patching_sweep.py")
    try:
        hashes["production_video_prompt_builder_source"] = sha256_bytes(builder_path.read_bytes())
    except OSError:
        hashes["production_video_prompt_builder_source"] = None
    for name, obj in objects.items():
        if obj is None:
            hashes[name] = None
            continue
        try:
            source = Path(inspect.getfile(obj))
            hashes[name] = sha256_bytes(source.read_bytes())
        except (OSError, TypeError, ValueError):
            hashes[name] = None
    return hashes


def build_report(
    processor: Any, *, runner: Path, repo_root: Path, snapshot: Path
) -> dict[str, Any]:
    import tokenizers
    import transformers

    tokenizer = processor.tokenizer
    backend_hash = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        backend_hash = sha256_bytes(backend.to_str().encode("utf-8"))
    chat_template = getattr(processor, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_hash = (
        sha256_bytes(chat_template.encode("utf-8")) if isinstance(chat_template, str) else None
    )
    comparison_started = time.perf_counter()
    observations = audit_fixed_cases(processor)
    choice_labels = audit_fixed_choice_labels(processor)
    video_choice_labels = audit_video_choice_labels(processor)
    comparison_elapsed = time.perf_counter() - comparison_started
    result = {
        "schema_version": 1,
        "status": "ok",
        "requested_cases": len(FIXED_TARGETS),
        "effective_cases": len(observations["cases"]),
        "failures": 0,
        "purpose": "synthetic_qwen_claim_tokenizer_and_video_boundary_cpu_diagnostic",
        "scope": "tokenizer_chat_template_and_synthetic_video_preprocessor_only_no_model_weights_no_real_video_or_QA_or_generation",
        "model_revision": MODEL_REVISION,
        "versions": {
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "processor_class": type(processor).__name__,
        },
        "source_sha256": {
            "runner": sha256_bytes(runner.read_bytes()),
            "tokenizer_backend_serialization": backend_hash,
            "chat_template": chat_template_hash,
            "local_tokenizer_files": _local_tokenizer_source_hashes(snapshot),
            "installed_implementation_files": _installed_implementation_hashes(processor, tokenizer),
            "fixed_fixture_spec": sha256_json(FIXTURE_SPEC),
        },
        "observations": observations,
        "choice_label_boundary": choice_labels,
        "video_choice_label_boundary": video_choice_labels,
        "timing": {
            "synthetic_boundary_check_core_seconds": comparison_elapsed,
            "scope": "fixed_synthetic_boundary_checks_excludes_imports_processor_loading_and_artifact_write",
        },
        "interpretation_limit": (
            "A tokenizer/preprocessor boundary result on these fixed synthetic text/video cases only; it does not show a prior QA/claim score was mis-tokenized, "
            "does not evaluate model behavior or semantics, and does not establish faithfulness or efficacy. "
            "The revision-named local snapshot and listed source-file hashes record provenance but do not authenticate every snapshot file."
        ),
        "run": {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "checkout_head": _source_commit(repo_root),
            "runner_sha256": sha256_bytes(runner.read_bytes()),
            "fixture_spec_sha256": sha256_json(FIXTURE_SPEC),
        },
    }
    return result


def safe_output_path(value: str | None, repo_root: Path) -> Path:
    if value is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = repo_root / "results" / f"qwen_claim_tokenization_cpu_{timestamp}.json"
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o600)
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
    parser.add_argument("--model-snapshot", type=Path, default=MODEL_CACHE,
                        help="Cached pinned processor snapshot; loading is local-only")
    parser.add_argument("--output", help="Unique JSON path directly under repository results/")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        output = safe_output_path(args.output, repo_root)
        if output.exists():
            raise FileExistsError
        snapshot = args.model_snapshot.expanduser().resolve()
        if not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
            raise FileNotFoundError
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        report = build_report(
            processor, runner=Path(__file__).resolve(), repo_root=repo_root, snapshot=snapshot
        )
        write_report(report, output)
    except FileExistsError:
        print("claim-tokenizer diagnostic: output already exists; refusing overwrite", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
        print("claim-tokenizer diagnostic: local tokenizer audit failed", file=sys.stderr)
        return 2
    observations = report["observations"]
    choice_gate_pass = (
        report["choice_label_boundary"]["gate_pass"]
        and report["video_choice_label_boundary"]["gate_pass"]
        and report["video_choice_label_boundary"]["mask_input_stability"]["gate_pass"]
    )
    print(
        "Claim tokenizer boundary CPU check: "
        f"cases={observations['fixed_case_count']} | "
        f"literal-concat matches={observations['literal_concat_match_count']}/"
        f"{observations['fixed_case_count']} | "
        f"completed-chat prefix stable={observations['completed_chat_prefix_stable_count']}/"
        f"{observations['fixed_case_count']} | "
        f"chat-boundary A-E={report['choice_label_boundary']['contextual_label_match_count']}/5 "
        f"video-input-boundary A-E="
        f"{report['video_choice_label_boundary']['expanded_video_contextual_label_match_count']}/5 "
        f"mask-input-stability="
        f"{'pass' if report['video_choice_label_boundary']['mask_input_stability']['gate_pass'] else 'fail'} "
        f"gate={'pass' if choice_gate_pass else 'fail'} | "
        f"artifact=results/{output.name}"
    )
    return 0 if choice_gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
