#!/usr/bin/env python3
"""Audit CLEVRER-Humans JSON and trusted official CEG pickle schema.

Pickle loading can execute code. By default this script refuses to unpickle; pass
--trust-official-pickle only for the archive obtained from the official Dryad
record. The script reports metadata and schema, not annotation text or video data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
from typing import Any


PICKLE_FILES = ("train_ceg_data.p", "valid_ceg_data.p")
JSON_FILES = ("causal_cloze.json", "train_question.json", "valid_question.json")
TIME_KEYS = {
    "start", "end", "start_frame", "end_frame", "frame", "frame_id",
    "timestamp", "time", "interval", "temporal_interval",
}
REPORT_SCHEMA_VERSION = 7
EXPLICIT_TIME_PATTERNS = {
    "frame_number": re.compile(r"\b(?:frame|frames|frm)\s*#?\s*\d+\b", re.IGNORECASE),
    "clock_timestamp": re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b"),
    "seconds_unit": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b", re.IGNORECASE
    ),
}


def normalize_scene_id(value: Any) -> str:
    """Normalize unpadded question JSON IDs to CLEVRER's five-digit scene IDs."""
    text = str(value)
    return text.zfill(5) if text.isdigit() else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_shape(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"top_level_type": type(value).__name__}
    if isinstance(value, dict):
        result["record_count"] = len(value)
        result["sample_keys"] = [str(key) for key in list(value)[:10]]
        counts: Counter[str] = Counter()
        question_count = 0
        question_fields: Counter[str] = Counter()
        choice_count = 0
        choice_labels: Counter[str] = Counter()
        for record in value.values():
            if isinstance(record, dict):
                counts.update(str(key) for key in record)
                questions = record.get("questions")
                if isinstance(questions, list):
                    question_count += len(questions)
                    for question in questions:
                        if isinstance(question, dict):
                            question_fields.update(str(key) for key in question)
                            choices = question.get("choices")
                            if isinstance(choices, list):
                                choice_count += len(choices)
                                for choice in choices:
                                    if isinstance(choice, dict) and "answer" in choice:
                                        choice_labels[str(choice["answer"]).lower()] += 1
        result["common_record_keys"] = counts.most_common(30)
        if "questions" in counts:
            result["nested_question_count"] = question_count
            result["nested_question_fields"] = question_fields.most_common(30)
            result["nested_choice_count"] = choice_count
            result["choice_answer_labels"] = dict(sorted(choice_labels.items()))
    elif isinstance(value, list):
        result["record_count"] = len(value)
        result["sample_record_keys"] = (
            sorted(str(key) for key in value[0])[:30]
            if value and isinstance(value[0], dict)
            else []
        )
    return result


def scan_time_keys(value: Any, *, max_items: int = 1_000_000) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    stack: list[tuple[Any, str]] = [(value, "$")]
    seen: set[int] = set()
    visited = 0

    while stack and visited < max_items:
        current, path = stack.pop()
        if isinstance(current, (dict, list, tuple, set, frozenset)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
        visited += 1

        if isinstance(current, dict):
            for key, child in current.items():
                key_text = str(key)
                if key_text.lower() in TIME_KEYS:
                    counts[key_text.lower()] += 1
                    if len(examples) < 30:
                        examples.append({"key": key_text, "path": path})
                stack.append((child, f"{path}.{key_text}"))
        elif isinstance(current, (list, tuple, set, frozenset)):
            for index, child in enumerate(current):
                stack.append((child, f"{path}[{index}]"))

    return {
        "time_key_counts": dict(sorted(counts.items())),
        "time_key_examples": examples,
        "objects_visited": visited,
        "scan_truncated": bool(stack),
    }


def graph_summary(value: Any) -> dict[str, Any] | None:
    nodes_method = getattr(value, "nodes", None)
    edges_method = getattr(value, "edges", None)
    if not callable(nodes_method) or not callable(edges_method):
        return None
    try:
        nodes = list(nodes_method(data=True))
        edges = list(edges_method(data=True))
    except (TypeError, AttributeError):
        return None
    temporal = scan_time_keys({"nodes": nodes, "edges": edges})
    node_attribute_keys: Counter[str] = Counter()
    edge_attribute_keys: Counter[str] = Counter()
    node_attribute_value_types: dict[str, Counter[str]] = {}
    edge_attribute_value_types: dict[str, Counter[str]] = {}
    for _, attrs in nodes:
        if not isinstance(attrs, dict):
            continue
        for key, item in attrs.items():
            key_text = str(key)
            node_attribute_keys[key_text] += 1
            node_attribute_value_types.setdefault(key_text, Counter())[type(item).__name__] += 1
    for _, _, attrs in edges:
        if not isinstance(attrs, dict):
            continue
        for key, item in attrs.items():
            key_text = str(key)
            edge_attribute_keys[key_text] += 1
            edge_attribute_value_types.setdefault(key_text, Counter())[type(item).__name__] += 1
    node_id_types = Counter(type(node).__name__ for node, _ in nodes)
    string_node_labels = [node for node, _ in nodes if isinstance(node, str)]
    explicit_time_pattern_counts = Counter(
        pattern_name
        for label in string_node_labels
        for pattern_name, pattern in EXPLICIT_TIME_PATTERNS.items()
        if pattern.search(label)
    )
    return {
        "class": f"{type(value).__module__}.{type(value).__name__}",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_attribute_keys": dict(sorted(node_attribute_keys.items())),
        "edge_attribute_keys": dict(sorted(edge_attribute_keys.items())),
        "node_attribute_value_types": {
            key: dict(sorted(types.items()))
            for key, types in sorted(node_attribute_value_types.items())
        },
        "edge_attribute_value_types": {
            key: dict(sorted(types.items()))
            for key, types in sorted(edge_attribute_value_types.items())
        },
        "node_id_value_types": dict(sorted(node_id_types.items())),
        "string_node_label_count": len(string_node_labels),
        "string_node_label_length_chars": {
            "min": min(map(len, string_node_labels), default=None),
            "max": max(map(len, string_node_labels), default=None),
        },
        "explicit_time_locator_pattern_counts": dict(
            sorted(explicit_time_pattern_counts.items())
        ),
        **temporal,
    }


def summarize_ceg(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"top_level_type": type(data).__name__, "schema_error": "expected dict"}

    video_names: set[str] = set()
    graph_fields: Counter[str] = Counter()
    graph_counts: dict[str, dict[str, int]] = {}
    graph_attribute_keys: dict[str, Counter[str]] = {
        "node": Counter(),
        "edge": Counter(),
    }
    graph_attribute_value_types: dict[str, dict[str, Counter[str]]] = {
        "node": {},
        "edge": {},
    }
    node_label_audit: dict[str, dict[str, Any]] = {}
    time_scan: dict[str, Any] = {"time_key_counts": {}, "time_key_examples": []}
    scene_fields: Counter[str] = Counter()

    for scene_id, scene in data.items():
        if not isinstance(scene, dict):
            continue
        scene_fields.update(str(key) for key in scene)
        video_name = scene.get("video_filename")
        if isinstance(video_name, str):
            video_names.add(video_name)
        for field, graph in scene.items():
            summary = graph_summary(graph)
            if summary is not None:
                graph_fields[str(field)] += 1
                key = str(field)
                counts = graph_counts.setdefault(key, {"graphs": 0, "nodes": 0, "edges": 0})
                counts["graphs"] += 1
                counts["nodes"] += summary["node_count"]
                counts["edges"] += summary["edge_count"]
                label_counts = node_label_audit.setdefault(
                    str(field),
                    {
                        "node_id_value_types": Counter(),
                        "string_node_label_count": 0,
                        "string_node_label_length_chars_min": None,
                        "string_node_label_length_chars_max": None,
                        "explicit_time_locator_pattern_counts": Counter(),
                    },
                )
                label_counts["node_id_value_types"].update(summary["node_id_value_types"])
                label_counts["string_node_label_count"] += summary["string_node_label_count"]
                label_counts["explicit_time_locator_pattern_counts"].update(
                    summary["explicit_time_locator_pattern_counts"]
                )
                length_range = summary["string_node_label_length_chars"]
                if length_range["min"] is not None:
                    current_min = label_counts["string_node_label_length_chars_min"]
                    current_max = label_counts["string_node_label_length_chars_max"]
                    label_counts["string_node_label_length_chars_min"] = (
                        length_range["min"] if current_min is None else min(current_min, length_range["min"])
                    )
                    label_counts["string_node_label_length_chars_max"] = (
                        length_range["max"] if current_max is None else max(current_max, length_range["max"])
                    )
                for scope in ("node", "edge"):
                    graph_attribute_keys[scope].update(
                        summary[f"{scope}_attribute_keys"]
                    )
                    for attribute, types in summary[f"{scope}_attribute_value_types"].items():
                        graph_attribute_value_types[scope].setdefault(
                            attribute, Counter()
                        ).update(types)
                for time_key, count in summary["time_key_counts"].items():
                    time_scan["time_key_counts"][time_key] = (
                        time_scan["time_key_counts"].get(time_key, 0) + count
                    )
                for example in summary["time_key_examples"]:
                    if len(time_scan["time_key_examples"]) < 30:
                        time_scan["time_key_examples"].append(
                            {"key": example["key"], "path": f"scene[{scene_id}].{field}.{example['path']}"}
                        )

    return {
        "top_level_type": "dict",
        "scene_count": len(data),
        "scene_key_sample": [str(key) for key in list(data)[:10]],
        "scene_fields": scene_fields.most_common(),
        "video_filename_count": len(video_names),
        "video_filename_sample": sorted(video_names)[:20],
        "graph_fields": dict(sorted(graph_fields.items())),
        "graph_totals_by_field": graph_counts,
        "graph_attribute_key_occurrences": {
            scope: dict(sorted(counts.items()))
            for scope, counts in graph_attribute_keys.items()
        },
        "graph_attribute_value_type_counts": {
            scope: {
                attribute: dict(sorted(types.items()))
                for attribute, types in sorted(attributes.items())
            }
            for scope, attributes in graph_attribute_value_types.items()
        },
        "graph_node_label_audit_by_field": {
            field: {
                "node_id_value_types": dict(sorted(values["node_id_value_types"].items())),
                "string_node_label_count": values["string_node_label_count"],
                "string_node_label_length_chars": {
                    "min": values["string_node_label_length_chars_min"],
                    "max": values["string_node_label_length_chars_max"],
                },
                "explicit_time_locator_pattern_counts": dict(
                    sorted(values["explicit_time_locator_pattern_counts"].items())
                ),
            }
            for field, values in sorted(node_label_audit.items())
        },
        **time_scan,
    }


def causal_edge_pairing_summary(data: Any) -> dict[str, int]:
    """Count same-target positive/negative CEG edge opportunities without labels."""
    totals = {
        "scenes_analyzed": 0,
        "scenes_with_positive_and_negative_edges": 0,
        "scenes_with_a_target_having_both_edge_types": 0,
        "positive_edges": 0,
        "negative_edges": 0,
        "targets_with_positive_incoming_edges": 0,
        "targets_with_negative_incoming_edges": 0,
        "targets_with_both_edge_types": 0,
        "targets_with_exactly_one_positive_and_one_negative_incoming_edge": 0,
        "scenes_with_target_having_exactly_one_positive_and_one_negative_incoming_edge": 0,
        "same_target_positive_negative_pair_combinations": 0,
    }
    if not isinstance(data, dict):
        return totals

    for scene in data.values():
        if not isinstance(scene, dict):
            continue
        positive_graph = scene.get("CEG_pos")
        negative_graph = scene.get("CEG_neg")
        if positive_graph is None or negative_graph is None:
            continue
        try:
            positive_edges = list(positive_graph.edges(data=True))
            negative_edges = list(negative_graph.edges(data=True))
        except (AttributeError, TypeError):
            continue

        positive_by_target = Counter(target for _, target, *_ in positive_edges)
        negative_by_target = Counter(target for _, target, *_ in negative_edges)
        common_targets = positive_by_target.keys() & negative_by_target.keys()
        totals["scenes_analyzed"] += 1
        totals["positive_edges"] += len(positive_edges)
        totals["negative_edges"] += len(negative_edges)
        totals["targets_with_positive_incoming_edges"] += len(positive_by_target)
        totals["targets_with_negative_incoming_edges"] += len(negative_by_target)
        totals["targets_with_both_edge_types"] += len(common_targets)
        exact_binary_targets = [
            target
            for target in common_targets
            if positive_by_target[target] == 1 and negative_by_target[target] == 1
        ]
        totals["targets_with_exactly_one_positive_and_one_negative_incoming_edge"] += len(
            exact_binary_targets
        )
        totals["same_target_positive_negative_pair_combinations"] += sum(
            positive_by_target[target] * negative_by_target[target]
            for target in common_targets
        )
        if positive_edges and negative_edges:
            totals["scenes_with_positive_and_negative_edges"] += 1
        if common_targets:
            totals["scenes_with_a_target_having_both_edge_types"] += 1
        if exact_binary_targets:
            totals[
                "scenes_with_target_having_exactly_one_positive_and_one_negative_incoming_edge"
            ] += 1
    return totals


def audit(annotation_dir: Path, *, trust_pickle: bool) -> dict[str, Any]:
    annotation_dir = annotation_dir.expanduser().resolve()
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "annotation_dir_name": annotation_dir.name,
        "files": {},
    }
    required = (*JSON_FILES, *PICKLE_FILES)
    missing = [name for name in required if not (annotation_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing expected annotation files: " + ", ".join(missing))

    json_values: dict[str, Any] = {}
    for filename in JSON_FILES:
        path = annotation_dir / filename
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
        json_values[filename] = value
        report["files"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **json_shape(value),
        }

    ceg_values: dict[str, dict[Any, Any]] = {}
    for filename in PICKLE_FILES:
        path = annotation_dir / filename
        entry: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "loaded": False,
        }
        if trust_pickle:
            with path.open("rb") as source:
                value = pickle.load(source)
            if isinstance(value, dict):
                ceg_values[filename] = value
            entry.update({"loaded": True, **summarize_ceg(value)})
            report.setdefault("causal_edge_pairing_by_split", {})[
                filename.removesuffix("_ceg_data.p")
            ] = causal_edge_pairing_summary(value)
        report["files"][filename] = entry

    if trust_pickle:
        split_pairs = (
            ("train_question.json", "train_ceg_data.p"),
            ("valid_question.json", "valid_ceg_data.p"),
        )
        alignment: dict[str, Any] = {}
        for question_file, ceg_file in split_pairs:
            questions = json_values[question_file]
            graphs = ceg_values.get(ceg_file, {})
            question_ids = (
                {normalize_scene_id(key) for key in questions}
                if isinstance(questions, dict)
                else set()
            )
            graph_ids = {normalize_scene_id(key) for key in graphs}
            alignment[question_file.removesuffix("_question.json")] = {
                "question_scene_count": len(question_ids),
                "graph_scene_count": len(graph_ids),
                "matched_scene_count": len(question_ids & graph_ids),
                "question_scene_without_graph_count": len(question_ids - graph_ids),
                "graph_scene_without_question_count": len(graph_ids - question_ids),
            }
        all_graph_ids = {
            normalize_scene_id(key) for data in ceg_values.values() for key in data
        }
        cloze = json_values["causal_cloze.json"]
        cloze_ids = (
            {normalize_scene_id(key) for key in cloze} if isinstance(cloze, dict) else set()
        )
        alignment["causal_cloze"] = {
            "record_scene_count": len(cloze_ids),
            "matched_train_or_valid_graph_count": len(cloze_ids & all_graph_ids),
            "record_without_graph_count": len(cloze_ids - all_graph_ids),
        }
        report["cross_file_alignment"] = alignment
        report["timing_scan_scope"] = (
            "Named temporal keys in CEG graph node/edge attributes only; this does not "
            "search event-description text or establish human temporal grounding."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trust-official-pickle",
        action="store_true",
        help="allow pickle.load; use only for the official Dryad archive you verified",
    )
    args = parser.parse_args()
    if not args.trust_official_pickle:
        parser.error(
            "pickle files are executable serialization; verify this is the official Dryad "
            "package, then pass --trust-official-pickle to inspect graph schemas"
        )

    report = audit(args.annotation_dir, trust_pickle=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote={output}")
    print("Interpretation note: presence of timing keys is structural evidence only, not proof of valid human temporal grounding.")


if __name__ == "__main__":
    main()
