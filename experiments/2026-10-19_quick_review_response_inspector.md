# Quick-check response inspector (2026-10-19)

## Purpose and implementation

Prepare local handling for the three-item nickname-first exploratory review without requesting or echoing personally identifying/free-text fields. `scripts/inspect_nextgqa_quick_review_response.py` checks the exact v2 schema and purpose, joins each ordered response to the opaque item ID embedded in the selected HTML artifact, rejects missing/duplicate/reordered/unknown items, and the CLI displays only aggregate response counts. Its in-memory summary retains item ordinals and category labels but no item IDs or free text. The input nickname, approximate time, and notes are validated but never printed or copied into the summary. It rejects any output use as a faithfulness estimate or scoring authorization. The reviewer workflow itself remains simple: nickname, then three direct questions; no file picking was added.

The Japanese runbook now links directly to this quick-check flow and clearly separates it from the signature- and timing-gated formal review, preventing the exploratory launcher from being mistaken for the full audit workspace. It also states that the three exploratory items are the same formal-review candidates, so someone who sees them cannot later serve as their blinded formal reviewer. This disclosure was added to the launcher and generated artifact without adding reviewer steps. The private v2 HTML was rebuilt through a temporary path and atomically replaced only after its three opaque item bindings matched the previous artifact; final permissions are `0600` and size is 30,645,813 bytes. No response JSON was present.

## Validation

Seven synthetic inspector tests cover item-ID extraction from the HTML, valid mapping, omission of nickname/free text, wrong artifact IDs, incomplete/duplicate/reordered answers, unknown judgment, unhashable malformed field types, duplicate JSON keys, and forged purpose. Together with the five builder safety tests, `PYTHONPATH=scripts .venv/bin/python -m unittest test_inspect_nextgqa_quick_review_response test_build_nextgqa_quick_review_artifact` passed **12/12**. The exact local v2 artifact was read-only parsed as 3 unique opaque IDs (no ID values printed). A synthetic end-to-end CLI run against those IDs returned the expected 3/3 format summary and did not echo synthetic nickname, time, or note strings. `py_compile`, `--help`, and `git diff --check` also passed. No private review JSON was found or read for this implementation, and no human response or GPU run occurred.

## Interpretation and limits

The inspector can establish only that a local JSON has the expected schema and maps to all three item IDs in the chosen v2 artifact. It cannot establish who answered, whether the videos were watched, whether a judgment is correct, or whether the answers measure explanation faithfulness. These answers remain informal observations and cannot unlock scoring.

## Next action

After a reviewer completes the local artifact, run `.venv/bin/python scripts/inspect_nextgqa_quick_review_response.py /path/to/downloaded-review.json` from the repository root while keeping that JSON local. This real-response command has not been run because no response JSON exists. Analyze only the returned counts as exploratory impressions; keep them separate from signed claim review, prediction-link judgments, direct scores, and `ΔF`. If no response arrives, continue with justified independent handoff engineering rather than repeat completed GPU proxy runs.
