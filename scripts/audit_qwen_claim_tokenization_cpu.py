"""Audit Qwen chat-prefix/claim token boundaries on fixed synthetic text only.

CPU/tokenizer-only diagnostic. It loads the pinned processor from the local
Hugging Face cache; downloads, model weights, videos, QA data, and generation are
disabled. The fixed prompt and 12 targets are prospective fixtures, not selected
from model outcomes. Output contains case indices, lengths, equality flags and
hashes only; it never emits fixture text or token IDs.

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
import time
from typing import Any


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
FIXTURE_SPEC = {
    "prompt": FIXED_PROMPT,
    "targets": list(FIXED_TARGETS),
    "cases_are_synthetic_and_fixed_before_outcomes": True,
    "choice_prompt": CHOICE_PROMPT,
    "choice_labels": list(CHOICE_LABELS),
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
    objects = {
        "processor_class_source": type(processor),
        "tokenizer_class_source": type(tokenizer),
        "chat_template_method_source": getattr(processor, "apply_chat_template", None),
    }
    try:
        objects["chat_template_utils_module_source"] = import_module(
            "transformers.utils.chat_template_utils"
        )
    except (ImportError, ModuleNotFoundError):
        objects["chat_template_utils_module_source"] = None
    hashes: dict[str, str | None] = {}
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
    comparison_elapsed = time.perf_counter() - comparison_started
    result = {
        "schema_version": 1,
        "status": "ok",
        "requested_cases": len(FIXED_TARGETS),
        "effective_cases": len(observations["cases"]),
        "failures": 0,
        "purpose": "synthetic_qwen_claim_tokenizer_prefix_boundary_cpu_diagnostic",
        "scope": "tokenizer_and_chat_template_only_no_model_weights_no_generation_no_video_or_QA",
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
        "timing": {
            "tokenizer_comparison_core_seconds": comparison_elapsed,
            "scope": "audit_fixed_cases_only_excludes_imports_processor_loading_and_artifact_write",
        },
        "interpretation_limit": (
            "A tokenizer-boundary result on these fixed synthetic cases only; it does not show a prior QA/claim score was mis-tokenized, "
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
    choice_gate_pass = report["choice_label_boundary"]["gate_pass"]
    print(
        "Claim tokenizer boundary CPU check: "
        f"cases={observations['fixed_case_count']} | "
        f"literal-concat matches={observations['literal_concat_match_count']}/"
        f"{observations['fixed_case_count']} | "
        f"completed-chat prefix stable={observations['completed_chat_prefix_stable_count']}/"
        f"{observations['fixed_case_count']} | "
        f"chat-boundary A-E={report['choice_label_boundary']['contextual_label_match_count']}/5 "
        f"gate={'pass' if choice_gate_pass else 'fail'} | "
        f"artifact=results/{output.name}"
    )
    return 0 if choice_gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
