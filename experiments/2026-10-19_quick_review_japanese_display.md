# Simplify Japanese quick-review prompts (2026-10-19)

## Purpose

Reduce the remaining language friction in the three-item exploratory video check. The existing nickname-to-question flow was already simple, but the displayed questions and generated claims remained English. This is a usability change for the informal impression check; it does not change the formal review protocol or create a faithfulness measurement.

## Change and provenance

- The builder now displays Japanese question and claim text first and keeps the original English wording in a collapsed, optional disclosure for translation checks.
- Translation text lives in a mode-0600 JSON file under the ignored `data/private_nextgqa/` directory. Entries are keyed by SHA-256 of the source question and claim together, so protected question/claim text is not embedded in tracked source code or this log and stale translations fail closed if either changes.
- Generated a new private artifact, `data/private_nextgqa/quick_review_ja_video_only_20261019_itemid_v3.html`; kept v2 and the older artifact unchanged. v3 and the translation file are both mode 0600. The artifact embeds the same three videos and retains the existing nickname, one-question-at-a-time, three-choice response schema, local JSON download, network-deny CSP, and limitation notice.
- Updated the Japanese launcher, runbook open command, and local response inspector default to v3.

## Checks and observed outcome

- Builder and response-inspector unit tests: **16/16 passed**.
- Python compilation, inline JavaScript syntax check, and scoped `git diff --check`: passed.
- Generated HTML structural inspection: one embedded video element, zero file inputs, Japanese primary prompt/claim targets present, and optional source-text targets present.
- Local output sizes: v3 is 30,652,700 bytes; v2 remains 30,645,813 bytes. Both have mode 0600. No response JSON was created or found by this change.

## Interpretation and limitations

This removes the English-reading obstacle while preserving access to the original wording. The Japanese text is a human translation and may still lose nuance; the optional English disclosure lets a reviewer notice that. Three broad judgments remain informal impressions, not complete hallucination labels, independent human attestation, claim-to-frame evidence, prediction-link judgments, or a scoring authorization. No GPU run was warranted: changing display language does not create a discriminating faithfulness experiment.

## Faithfulness progress

- Direct explanation-faithfulness scoring: **0/3 claims**; no new direct score, effect estimate, confidence interval, or test.
- Formal claim review: **0/3**; player-clock/decoder-frame correspondence: **0/15**; prediction-link review: **0/3**.
- The earlier 30-video annotation-anchor intervention remains an exploratory proxy only (mean ΔF +0.00833 nat; SD 0.14284, mixed signs); it is not a generated-explanation faithfulness result and is not reopened here.
- GPU/CUDA is available in the individually elevated environment (RTX 5070 Ti, CUDA-enabled `.venv`), but no valid new GPU score run is justified until the relevant review inputs and phase gates are available.

## Next action

The reviewer can open v3 from the repository root with:

```bash
xdg-open ./data/private_nextgqa/quick_review_ja_video_only_20261019_itemid_v3.html
```

If an exploratory response JSON is produced, validate it locally with `scripts/inspect_nextgqa_quick_review_response.py`. Keep that JSON local; it contains a nickname and optional free text. Separately continue the CPU-only scorer/receipt integration audit. Do not interpret the informal response as a faithfulness score or use it to bypass the formal evidence and scoring gates.
