#!/usr/bin/env python3
"""Aggregate-only audit of a frozen NExT-GQA source-row binding.

Reads local source files and performs no writes, model loads, or scoring. The
summary omits QA text, row identities, paths, and hashes. It verifies internal
consistency with caller-supplied lineage; it does not authenticate how that
lineage was produced or establish human claim faithfulness.

Example::

    python scripts/audit_nextgqa_actual_qa_source_binding.py \
      --csv NExT-GQA/datasets/nextgqa/test.csv \
      --lineage data/private_nextgqa/claim_lineage_freshcohort_20261016_03.json \
      --worksheet data/private_nextgqa/claim_mapping_freshcohort_20261016_20261016_01.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import unicodedata

from nextgqa_actual_qa_source_adapter import build_actual_qa_source


def summarize(csv_path: Path, lineage_path: Path, worksheet_path: Path) -> dict:
    csv_raw = csv_path.read_bytes()
    lineage_raw = lineage_path.read_bytes()
    worksheet_raw = worksheet_path.read_bytes()
    lineage = json.loads(lineage_raw)
    worksheet = json.loads(worksheet_raw)
    records = lineage.get("records") if isinstance(lineage, dict) else None
    sheet_rows = worksheet.get("records") if isinstance(worksheet, dict) else None
    if (not isinstance(lineage, dict) or not isinstance(worksheet, dict)
            or not isinstance(records, list) or not isinstance(sheet_rows, list)
            or len(sheet_rows) != len(records)
            or any(not isinstance(row, dict) for row in sheet_rows)):
        raise ValueError("source_records_invalid")
    selection = {"records": [{
        "video_id": row["video_id"], "claim_id": row["claim_id"],
        "dataset_video_id": row["dataset_video_id"],
        "dataset_question_id": row["dataset_question_id"],
    } for row in records]}
    target = {"records": [{"video_id": row["video_id"], "claim_id": row["claim_id"]}
                          for row in records]}
    qa, manifest = build_actual_qa_source(
        csv_raw=csv_raw, selection=selection,
        target_manifest_raw=json.dumps(target).encode("utf-8"),
        lineage_manifest_raw=lineage_raw)

    normalize = lambda value: re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if (None in row or any(value is None for value in row.values())
                    or any(not isinstance(value, str) for value in row.values())):
                raise ValueError("csv_row_width_invalid")
            choices = [row[f"a{i}"].strip() for i in range(5)]
            matches = sum(normalize(choice) == normalize(row["answer"])
                          for choice in choices)
            rows.append(matches)
    sheet_by_video = {row.get("video_id"): row for row in sheet_rows
                      if isinstance(row, dict)}
    if len(sheet_by_video) != len(sheet_rows):
        raise ValueError("worksheet_cohort_invalid")
    qa_rows = qa["records"]
    question_matches = sum(
        isinstance(sheet_by_video.get(row["video_id"]), dict)
        and row["question"] == sheet_by_video[row["video_id"]].get("question")
        for row in qa_rows)
    unique = sum(matches == 1 for matches in rows)
    ambiguous = sum(matches > 1 for matches in rows)
    missing = sum(matches == 0 for matches in rows)
    return {
        "status": "local_source_binding_passed_unvalidated",
        "scope": "actual_local_csv_and_caller_supplied_lineage_consistency_only",
        "csv_rows": len(rows), "gold_unique_rows": unique,
        "ambiguous_gold_rows": ambiguous, "missing_gold_rows": missing,
        "requested_records": len(records), "effective_records": len(qa_rows),
        "question_matches_worksheet": question_matches,
        "source_row_bindings": len(manifest["source_row_bindings"]),
        "files_written": 0, "model_forwards": 0, "score_calls": 0,
        "execution_authorization": "not_granted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--worksheet", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.csv, args.lineage, args.worksheet)
    except OSError:
        print(json.dumps({"status": "blocked", "failure_code": "input_unavailable",
                          "files_written": 0, "score_calls": 0,
                          "execution_authorization": "not_granted"}))
        return 2
    except (ValueError, TypeError, KeyError, AttributeError,
            json.JSONDecodeError) as exc:
        # Failure output is deliberately reduced to a stable code only.
        code = str(exc) if str(exc) and " " not in str(exc) else "audit_failed"
        print(json.dumps({"status": "blocked", "failure_code": code,
                          "files_written": 0, "score_calls": 0,
                          "execution_authorization": "not_granted"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
