#!/usr/bin/env python3
"""CPU-only, privacy-preserving schema audit for official CLEVRER QA JSON.

The audit reports counts and field coverage only. It never prints question text,
answers, programs, scene identifiers, or full local paths. It does not establish
explanation faithfulness; it only checks whether the data can support a later
structural-rationale feasibility study.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


QUESTION_KEYS = {"question", "question_text", "query"}
ANSWER_KEYS = {"answer", "correct_answer", "answer_idx", "answer_index"}
PROGRAM_KEYS = {"program", "functional_program", "programs"}
TYPE_KEYS = {"question_type", "question_family", "type", "family"}
SCENE_KEYS = {"scene_index", "scene_id", "video_filename", "video_id"}
DEFAULT_NAMES = {
    "train.json", "val.json", "validation.json", "train_questions.json",
    "val_questions.json", "validation_questions.json",
}

# Operator tokens registered by the official CLEVRER executor. Its multiple-
# choice runner stores programs as string-token lists: registered module names
# are operators and other strings are values/control tokens.
# Source: https://github.com/chuangg/CLEVRER/blob/master/executor/executor.py
CLEVRER_OPERATOR_TOKENS = frozenset({
    "objects", "events", "unique", "count", "exist", "negate", "belong_to",
    "filter_color", "filter_material", "filter_shape", "filter_resting",
    "filter_moving", "filter_stationary", "filter_start", "filter_end",
    "filter_in", "filter_out", "filter_collision", "filter_order",
    "filter_before", "filter_after", "query_color", "query_material",
    "query_shape", "query_direction", "query_frame", "query_object",
    "query_collision_partner", "filter_ancestor", "unseen_events", "all_events",
    "counterfact_events", "filter_counterfact",
})


def discover_json_files(roots: Iterable[Path]) -> list[Path]:
    """Find likely official CLEVRER question files without opening other data."""
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".json":
            found.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                if filename.lower() in DEFAULT_NAMES:
                    found.add((Path(directory) / filename).resolve())
    return sorted(found)


def _walk_dicts(value: Any, inherited_scene_link: bool = False):
    if isinstance(value, dict):
        has_scene_link = inherited_scene_link or any(
            _normal_key(key) in SCENE_KEYS for key in value
        )
        yield value, has_scene_link
        for child in value.values():
            yield from _walk_dicts(child, has_scene_link)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child, inherited_scene_link)


def _normal_key(key: Any) -> str:
    return str(key).strip().lower()


def _is_question(obj: dict[str, Any]) -> bool:
    return (
        any(key in obj for key in QUESTION_KEYS)
        and (any(key in obj for key in ANSWER_KEYS)
             or any(key in obj for key in PROGRAM_KEYS)
             or "choices" in obj)
    )


def _is_counterfactual(question: dict[str, Any]) -> bool:
    type_values = [
        str(value).lower() for key, value in question.items()
        if _normal_key(key) in TYPE_KEYS and isinstance(value, (str, int, float))
    ]
    return any("counterfactual" in value or value in {"cf", "what_if"}
               for value in type_values)


def _choice_programs(question: dict[str, Any]) -> list[Any]:
    """Return non-empty program payloads attached to answer choices."""
    choices = question.get("choices")
    choice_programs = []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_keys = {_normal_key(key): key for key in choice}
            choice_program_key = next(
                (choice_keys[key] for key in sorted(PROGRAM_KEYS) if key in choice_keys),
                None,
            )
            if choice_program_key is not None:
                program = choice[choice_program_key]
                if program not in (None, [], {}, ""):
                    choice_programs.append(program)
    return choice_programs


def _question_level_programs(question: dict[str, Any]) -> list[Any]:
    """Return non-empty question-level program payloads, if present."""
    keys = {_normal_key(key): key for key in question}
    program_key = next((keys[key] for key in sorted(PROGRAM_KEYS) if key in keys), None)
    if program_key is not None:
        program = question[program_key]
        if program not in (None, [], {}, ""):
            return [program]
    return []


def _question_programs(question: dict[str, Any]) -> tuple[list[Any], str | None]:
    """Use choice-specific programs first; question-level data is a fallback."""
    programs = _choice_programs(question)
    if programs:
        return programs, "choice_level"
    programs = _question_level_programs(question)
    if programs:
        return programs, "question_level"
    return [], None


def _program_signature(question: dict[str, Any]) -> str | None:
    """Hash a question's program structure, never exposing the program itself."""
    programs, _ = _question_programs(question)
    if not programs:
        return None
    canonical_programs = sorted(json.dumps(
        program, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ) for program in programs)
    canonical = json.dumps(canonical_programs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _operation_trace(program: Any) -> tuple[str, ...]:
    """Extract official CLEVRER operators from node or string-token programs.

    In the official token-list format, registered module names are operators;
    all other strings (values and boundary markers) are discarded.
    """
    trace: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            token = value.strip().lower()
            if token in CLEVRER_OPERATOR_TOKENS:
                trace.append(token)
        elif isinstance(value, dict):
            normalized = {_normal_key(key): key for key in value}
            op_key = next(
                (normalized[key] for key in (
                    "type", "op", "function", "operation", "module",
                    "module_name", "function_name", "op_name", "operation_name", "name",
                )
                 if key in normalized),
                None,
            )
            op_value = value.get(op_key) if op_key is not None else None
            if isinstance(op_value, (str, int, float)):
                trace.append(str(op_value).lower())
            elif isinstance(op_value, list) and op_value and all(
                isinstance(item, (str, int, float)) for item in op_value
            ):
                trace.extend(str(item).lower() for item in op_value)
            else:
                for child in value.values():
                    visit(child)

    visit(program)
    return tuple(trace)


def _program_skeleton_signature(question: dict[str, Any]) -> str | None:
    """Hash sorted operator traces, ignoring program arguments and node links."""
    programs, _ = _question_programs(question)
    if not programs:
        return None
    traces = sorted(_operation_trace(program) for program in programs)
    if not any(traces):
        return None
    canonical = json.dumps(traces, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _question_tokens(question: dict[str, Any]) -> tuple[str, ...] | None:
    """Return normalized question tokens for local-only aggregate comparison."""
    text_value = next(
        (question[key] for key in sorted(QUESTION_KEYS)
         if isinstance(question.get(key), str) and question[key].strip()),
        None,
    )
    if text_value is None:
        return None
    return tuple(token.lower() for token in re.findall(r"\w+|[^\w\s]", text_value))


def _token_edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Levenshtein distance over tokens; callers publish only coarse counts."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def _program_schema_counts(questions: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(programs: list[Any]) -> dict[str, Any]:
        root_types: Counter[str] = Counter()
        field_type_counts: Counter[str] = Counter()
        list_item_type_counts: Counter[str] = Counter()
        dict_node_count = 0

        def visit(value: Any) -> None:
            nonlocal dict_node_count
            if isinstance(value, dict):
                dict_node_count += 1
                for key, child in value.items():
                    key_name = str(key)
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", key_name):
                        key_name = "<other_key>"
                    field_type_counts[f"{key_name}:{type(child).__name__}"] += 1
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    list_item_type_counts[type(child).__name__] += 1
                    visit(child)

        for program in programs:
            root_types[type(program).__name__] += 1
            visit(program)
        return {
            "root_type_counts": dict(sorted(root_types.items())),
            "field_type_counts": dict(sorted(field_type_counts.items())),
            "list_item_type_counts": dict(sorted(list_item_type_counts.items())),
            "dict_node_count": dict_node_count,
        }

    question_programs: list[Any] = []
    choice_programs: list[Any] = []
    source_counts: Counter[str] = Counter()
    empty_question_program_with_choices = 0
    for question in questions:
        choice_level = _choice_programs(question)
        question_level = _question_level_programs(question)
        if choice_level:
            source_counts["choice_level"] += 1
            choice_programs.extend(choice_level)
            has_question_program_field = any(
                _normal_key(key) in PROGRAM_KEYS for key in question
            )
            empty_question_program_with_choices += int(
                has_question_program_field and not question_level
            )
        elif question_level:
            source_counts["question_level_fallback"] += 1
            question_programs.extend(question_level)
        else:
            source_counts["no_nonempty_program"] += 1
    return {
        "question_level": summarize(question_programs),
        "choice_level": summarize(choice_programs),
        "question_program_source_counts": dict(sorted(source_counts.items())),
        "empty_question_program_with_choice_program_count": empty_question_program_with_choices,
    }


def audit_payload(payload: Any) -> dict[str, Any]:
    """Count QA/program structure while intentionally excluding content values."""
    dictionaries = list(_walk_dicts(payload))
    questions = [(obj, linked) for obj, linked in dictionaries if _is_question(obj)]

    type_counts: Counter[str] = Counter()
    question_with_answer = 0
    question_with_program = 0
    question_with_choices = 0
    choice_total = 0
    choice_with_answer = 0
    choice_with_program = 0
    question_with_scene_link = 0
    counterfactual_n = 0
    counterfactual_with_program = 0
    counterfactual_with_nonempty_program_payload = 0
    counterfactual_with_answer = 0
    question_with_nonempty_program_payload = 0
    choice_with_nonempty_program_payload = 0

    for question, inherited_scene_link in questions:
        keys = {_normal_key(key) for key in question}
        choices = question.get("choices")
        choice_dicts = [choice for choice in choices if isinstance(choice, dict)] \
            if isinstance(choices, list) else []
        has_answer = bool(keys & ANSWER_KEYS) or any(
            {_normal_key(key) for key in choice} & ANSWER_KEYS for choice in choice_dicts
        )
        has_program = bool(keys & PROGRAM_KEYS) or any(
            {_normal_key(key) for key in choice} & PROGRAM_KEYS for choice in choice_dicts
        )
        has_nonempty_program = bool(
            _choice_programs(question) or _question_level_programs(question)
        )
        question_with_answer += int(has_answer)
        question_with_program += int(has_program)
        question_with_nonempty_program_payload += int(has_nonempty_program)
        question_with_scene_link += int(inherited_scene_link or bool(keys & SCENE_KEYS))

        type_values = [
            str(value).lower() for key, value in question.items()
            if _normal_key(key) in TYPE_KEYS and isinstance(value, (str, int, float))
        ]
        is_counterfactual = _is_counterfactual(question)
        if is_counterfactual:
            counterfactual_n += 1
            counterfactual_with_program += int(has_program)
            counterfactual_with_nonempty_program_payload += int(has_nonempty_program)
            counterfactual_with_answer += int(has_answer)
        for value in type_values:
            type_counts[value[:48]] += 1

        if isinstance(choices, list):
            question_with_choices += 1
            choice_total += len(choices)
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                choice_keys = {_normal_key(key) for key in choice}
                choice_with_answer += int(bool(choice_keys & ANSWER_KEYS))
                choice_with_program += int(bool(choice_keys & PROGRAM_KEYS))
                choice_with_nonempty_program_payload += int(any(
                    _normal_key(key) in PROGRAM_KEYS and value not in (None, [], {}, "")
                    for key, value in choice.items()
                ))

    # CLEVRER's observed official format is a list of scene records. Count
    # within-record pairability, using only hashes of program structures.
    # This is a structural proxy, not proof that two questions target distinct
    # or semantically matched interventions.
    per_scene_cf: list[list[tuple[
        dict[str, Any], str | None, str | None, tuple[str, ...] | None
    ]]] = []
    if isinstance(payload, list):
        for scene in payload:
            scene_questions = [obj for obj, _ in _walk_dicts(scene) if _is_question(obj)]
            cf_questions = [q for q in scene_questions if _is_counterfactual(q)]
            per_scene_cf.append([
                (q, _program_signature(q), _program_skeleton_signature(q), _question_tokens(q))
                for q in cf_questions
            ])
    cf_scene_counts = [len(rows) for rows in per_scene_cf if rows]
    cf_scenes_with_pairs = sum(count >= 2 for count in cf_scene_counts)
    cf_scenes_with_distinct_program_pairs = sum(
        len({signature for _, signature, _, _ in rows if signature is not None}) >= 2
        for rows in per_scene_cf
    )
    all_cf_questions = [
        obj for obj, _ in questions if _is_counterfactual(obj)
    ]
    program_schema_counts = _program_schema_counts(all_cf_questions)
    exact_pairs = [rows for rows in per_scene_cf if len(rows) == 2]
    distinct_program_pairs = [
        rows for rows in exact_pairs
        if all(row[1] is not None for row in rows) and rows[0][1] != rows[1][1]
    ]
    same_skeleton_pairs = sum(
        rows[0][2] is not None and rows[0][2] == rows[1][2]
        for rows in distinct_program_pairs
    )
    different_skeleton_pairs = sum(
        rows[0][2] is not None and rows[1][2] is not None
        and rows[0][2] != rows[1][2]
        for rows in distinct_program_pairs
    )
    text_edit_bins: Counter[str] = Counter()
    for rows in distinct_program_pairs:
        if rows[0][2] is None or rows[0][2] != rows[1][2]:
            continue
        left_tokens, right_tokens = rows[0][3], rows[1][3]
        if left_tokens is None or right_tokens is None:
            text_edit_bins["unavailable"] += 1
            continue
        distance = _token_edit_distance(left_tokens, right_tokens)
        if distance <= 1:
            text_edit_bins[str(distance)] += 1
        elif distance <= 3:
            text_edit_bins["2-3"] += 1
        else:
            text_edit_bins["4+"] += 1

    return {
        "top_level_type": type(payload).__name__,
        "top_level_records": len(payload) if isinstance(payload, (dict, list)) else 0,
        "question_count": len(questions),
        "question_with_answer": question_with_answer,
        "question_with_program": question_with_program,
        "question_with_nonempty_program_payload": question_with_nonempty_program_payload,
        "question_with_choices": question_with_choices,
        "choice_count": choice_total,
        "choice_with_answer": choice_with_answer,
        "choice_with_program": choice_with_program,
        "choice_with_nonempty_program_payload": choice_with_nonempty_program_payload,
        "question_with_scene_link": question_with_scene_link,
        "question_type_counts": dict(sorted(type_counts.items())),
        "counterfactual_count_by_type_field": counterfactual_n,
        "counterfactual_with_answer": counterfactual_with_answer,
        "counterfactual_with_program": counterfactual_with_program,
        "counterfactual_with_nonempty_program_payload": counterfactual_with_nonempty_program_payload,
        "counterfactual_program_schema": program_schema_counts,
        "counterfactual_scene_count": len(cf_scene_counts) if isinstance(payload, list) else None,
        "counterfactual_scenes_with_2plus_questions": (
            cf_scenes_with_pairs if isinstance(payload, list) else None
        ),
        "counterfactual_scenes_with_2plus_program_signatures": (
            cf_scenes_with_distinct_program_pairs if isinstance(payload, list) else None
        ),
        "paired_scenes_distinct_programs_same_operator_trace": (
            same_skeleton_pairs if isinstance(payload, list) else None
        ),
        "paired_scenes_distinct_programs_different_operator_trace": (
            different_skeleton_pairs if isinstance(payload, list) else None
        ),
        "paired_scenes_operator_trace_unavailable": (
            len(distinct_program_pairs) - same_skeleton_pairs - different_skeleton_pairs
            if isinstance(payload, list) else None
        ),
        "same_operator_trace_pairs_question_token_edit_distance": {
            "candidate_pairs": same_skeleton_pairs,
            "bins": {key: text_edit_bins[key] for key in (
                "0", "1", "2-3", "4+", "unavailable"
            )},
        },
        "max_counterfactual_questions_per_scene": (
            max(cf_scene_counts, default=0) if isinstance(payload, list) else None
        ),
    }


def _coverage(count: int, total: int) -> str:
    return f"{count}/{total}" if total else "n/a"


def format_summary(reports: list[tuple[Path, dict[str, Any]]]) -> str:
    lines = [f"CLEVRER CPU data audit: {len(reports)} candidate JSON file(s)"]
    for path, report in reports:
        # A basename/split label is sufficient; avoid leaking a machine-specific path.
        lines.append(
            f"{path.name}: questions={report['question_count']}, "
            f"counterfactual-type={report['counterfactual_count_by_type_field']}, "
            f"question answer={_coverage(report['question_with_answer'], report['question_count'])}, "
            f"program field/nonempty={_coverage(report['question_with_program'], report['question_count'])}/"
            f"{_coverage(report['question_with_nonempty_program_payload'], report['question_count'])}, "
            f"choices={report['choice_count']} "
            f"(answer={report['choice_with_answer']}, "
            f"program field/nonempty={report['choice_with_program']}/"
            f"{report['choice_with_nonempty_program_payload']}), "
            f"scene-link={report['question_with_scene_link']}"
        )
        if report["counterfactual_count_by_type_field"]:
            lines.append(
                "  counterfactual-type rows with answer/program-field/nonempty-payload: "
                f"{_coverage(report['counterfactual_with_answer'], report['counterfactual_count_by_type_field'])}/"
                f"{_coverage(report['counterfactual_with_program'], report['counterfactual_count_by_type_field'])}/"
                f"{_coverage(report['counterfactual_with_nonempty_program_payload'], report['counterfactual_count_by_type_field'])}"
            )
            if report["counterfactual_scene_count"] is not None:
                lines.append(
                    "  same-scene structure: "
                    f"{report['counterfactual_scenes_with_2plus_questions']} scenes with 2+ CF questions; "
                    f"{report['counterfactual_scenes_with_2plus_program_signatures']} with 2+ distinct program signatures; "
                    f"max CF questions/scene={report['max_counterfactual_questions_per_scene']}"
                )
                if report["paired_scenes_distinct_programs_same_operator_trace"] is not None:
                    lines.append(
                        "  among exactly-two-question scenes with distinct programs: "
                        f"same operator trace={report['paired_scenes_distinct_programs_same_operator_trace']}; "
                        f"different trace={report['paired_scenes_distinct_programs_different_operator_trace']}; "
                        f"unavailable={report['paired_scenes_operator_trace_unavailable']}"
                    )
                    text_edits = report["same_operator_trace_pairs_question_token_edit_distance"]
                    lines.append(
                        "  same-trace pairs question-token edit distance bins "
                        "(descriptive; no text published): "
                        f"0={text_edits['bins']['0']}, 1={text_edits['bins']['1']}, "
                        f"2-3={text_edits['bins']['2-3']}, 4+={text_edits['bins']['4+']}, "
                        f"unavailable={text_edits['bins']['unavailable']}"
                    )
    lines.append(
        "Interpretation: schema feasibility only; program/rationale agreement is not internal faithfulness."
    )
    return "\n".join(lines)


def safe_report(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    """Build a publishable summary that excludes source paths and raw text."""
    allowed_metrics = (
        "top_level_type", "top_level_records", "question_count",
        "question_with_answer", "question_with_program", "question_with_nonempty_program_payload",
        "question_with_choices", "choice_count", "choice_with_answer", "choice_with_program",
        "choice_with_nonempty_program_payload",
        "question_with_scene_link", "counterfactual_count_by_type_field",
        "counterfactual_with_answer", "counterfactual_with_program",
        "counterfactual_with_nonempty_program_payload",
        "counterfactual_program_schema",
        "counterfactual_scene_count", "counterfactual_scenes_with_2plus_questions",
        "counterfactual_scenes_with_2plus_program_signatures",
        "paired_scenes_distinct_programs_same_operator_trace",
        "paired_scenes_distinct_programs_different_operator_trace",
        "paired_scenes_operator_trace_unavailable",
        "same_operator_trace_pairs_question_token_edit_distance",
        "max_counterfactual_questions_per_scene",
    )
    files = []
    for path, report in reports:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({
            "file_name": path.name,
            "sha256": digest,
            "counts": {key: report[key] for key in allowed_metrics},
        })
    return {
        "audit": "clevrer_counterfactual_program_schema",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Schema feasibility only; program/rationale agreement is not internal faithfulness."
        ),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", default=[], type=Path,
        help="explicit JSON file; repeat for train/validation files",
    )
    parser.add_argument(
        "--root", action="append", default=[], type=Path,
        help="search for likely CLEVRER QA filenames under this directory; repeatable",
    )
    parser.add_argument("--output", type=Path,
                        help="optional path for a sanitized counts-only JSON report")
    args = parser.parse_args()
    candidates = {path.resolve() for path in args.input}
    candidates.update(discover_json_files(args.root))
    paths = sorted(candidates)
    if not paths:
        print("No candidate CLEVRER QA JSON found. Specify --input or search roots with --root.")
        return 2

    reports = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        reports.append((path, audit_payload(payload)))
    print(format_summary(reports))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"refusing to overwrite existing report: {args.output}")
        args.output.write_text(json.dumps(safe_report(reports), indent=2) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
