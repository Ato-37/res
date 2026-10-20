# Current v2_short claim-review handoff (2026-10-21, local-only)

## Purpose

The prior Japanese launcher targeted an earlier validation cohort. This update
creates a simplified local artifact for the three current `v2_short` generated
claims, so any reviewers who choose to respond see the exact claims used in the
latest exploratory GPU work. The user's waiver means review collection is not a
prerequisite for exploratory local work; it does not mean a review occurred.

## Artifact and provenance

- Source packet: `data/private_nextgqa/claim_generation_val_exploratory_v2short_20261021_01.json`
- Artifact: `data/private_nextgqa/quick_review_v2short_20261021_01.html`
- Artifact SHA-256: `329a99fe4965655e827ca4067f909c6b010283166a0e04b8e7875a9f257dd0ec`
- Artifact mode: `0600`; package directory mode: `0700`; package files: `0600`.
- Three questions and claims are displayed in Japanese, with English source
  wording available in the expandable panel. Videos are embedded in the page.
  No network connection is needed or permitted by its content-security policy.
- Responses are collected locally and downloaded as one schema-v6 JSON with
  artifact fingerprint, per-reviewer nickname, and all three ratings per item.
  This is a short exploratory impression check, not a truth adjudication or
  formal faithfulness result.

Primary verification decoded the embedded item array in memory and confirmed
for all three rows that the displayed question and claim match the source
packet, each opaque item ID matches its review token, and each embedded video
byte sequence matches the corresponding browser-compatible packaged video. The
HTML fingerprint recomputes from the page with its fingerprint placeholder
restored. No question, claim, reviewer identity, or response was printed in the
verification summary.

## Browser verification

The exact new artifact passed the Windows Chrome E2E through WSL:

```text
QUICK_REVIEW_ACTUAL_ARTIFACT=data/private_nextgqa/quick_review_v2short_20261021_01.html \
PYTHONPATH=scripts .venv/bin/python -m unittest test_quick_review_artifact_browser_e2e -v
Ran 1 test; OK
```

The test exercised three reviewer sessions, individual nickname entry, video
decode/readiness across item transitions, and combined JSON export bound to the
artifact fingerprint. The exported answers were synthetic test values; this is
workflow evidence only and no human response was created. A first run in the
normal sandbox failed the Windows Chrome process/profile cleanup check; the
same bounded test passed under individually elevated WSL/Windows execution.

An independent read-only audit compared the artifact's current SHA-256 to this
log, confirmed the E2E command explicitly targets that exact filename, and
verified that the test's actual-artifact branch opens the supplied HTML, waits
for metadata and decoded frames across all three items, runs three reviewer
sessions, and binds the JSON export to the loaded page's fingerprint and item
order. The audit found no item/video/claim binding mismatch. This supports the
browser-workflow claim above, not any human judgment or faithfulness result.

The response inspector was also exercised with a synthetic schema-v6 response
bound to this exact artifact. It accepted the fingerprint and 3-item order,
validated 9 judgments from 3 synthetic respondents, redacted nicknames in its
summary, and retained `faithfulness_inference=not_performed` and
`scoring_authorization=not_granted`. This validates artifact-to-intake wiring;
synthetic values contribute no human-review count.

## Research status and next action

The artifact now offers a current, simple handoff, but **actual human review is
still 0/3**. Direct faithfulness remains **0/3 scored**, verified clock/frame
matches **0/15**, and Stage-2 prediction links **0/3**. The recent original vs
gray and original vs frame-mean experiments remain exploratory conditional-
likelihood sensitivity measures, not direct faithfulness. No formal review or
prediction-link receipt was produced. Rights status remains unverified; the
artifact, videos, and any future response must stay local.

Next: if a reviewer voluntarily completes this artifact, inspect the downloaded
JSON against this exact fingerprint and keep the response private. Continue
local exploratory work independently, while preserving the distinction between
user-waived collection, actual observations, and formal review evidence.

## Launcher correction after handoff audit

The Japanese launcher and runbook still targeted the superseded 2026-10-19
artifact even though the current v2_short artifact was already built and passed
browser E2E. Updated both redirect/fallback targets and both documented local
open commands to `quick_review_v2short_20261021_01.html`. Metadata-only
assertions verified the two launcher targets resolve to the present owner-only
artifact, both runbook commands point to the same file, and no stale filename
remains in either document. The private HTML was not modified or staged.

This is a handoff repair only. The user's instruction waives human review as a
prerequisite for local exploratory experiments; it does not create human
judgments. Actual reviews remain 0/3, and direct faithfulness remains 0/3.
Continue exploratory experiments with explicit unreviewed/model-proxy labels;
do not fabricate review receipts or bypass data/source provenance checks.
