# Formal review export to Stage-1 bundle CLI composition (2026-10-19)

## Purpose

Check that the actual claim-review page export handler produces the exact JSON
and integrity-index bytes consumed by the private Stage-1 bundle CLI, without
manual schema translation. Exercise both structural acceptance and rejection
after the exported payload is changed.

## Setup

Extended `scripts/test_audit_qta_stage1_bundle_cli.py` with a synthetic-only
integration test. It renders the existing formal claim-review page from the
Stage-1 audit fixture, extracts and executes that page's actual export and
index-download handlers using Node's VM and a minimal synthetic DOM state,
captures the exact review JSON/index download bytes, signs those bytes using
the fixture's generated test key,
and sends the bundle through `audit_qta_stage1_bundle_cli.main`. It then adds a
byte to the review export and checks that the same CLI rejects the changed
bundle. All videos, review records, keys, identities, and answers are
synthetic. The test is not a real-browser interaction and does not access
protected data.

## Verified results

- The actual rendered handlers generated two JSON downloads; the index-button
  bytes exactly matched the handler's pending index and its
  `review_payload_sha256` matched the review download bytes.
- The bundle CLI accepted the exact generated export as
  `stage1_structurally_valid_non_authorizing`, with 3 structurally valid rows,
  `model_load_permitted=false`, `score_ready=false`, and
  `execution_authorization=not_granted`.
- Appending one byte to the exported review caused CLI exit 2 and a sanitized
  `blocked` report with `stage1_audit_contract_invalid`; no review token was
  printed.
- `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_audit_qta_stage1_bundle_cli -v`: **12/12 passed**.
- Combined renderer, handoff audit, reviewer attestation, bundle CLI, and
  Stage-1 audit suites: **83/83 passed**. The unrelated lightweight-artifact
  builder/inspector suite also passed **17/17**; its synthetic Chrome flow
  passed **1/1** after the save-disclosure update.

## Interpretation and limits

This closes a concrete software handoff seam: the rendered page's export
schema and integrity index compose with the existing Stage-1 bundle audit
without a translation step, and post-export mutation is caught. Node VM
execution does not establish real browser download behavior, reviewer
identity, actual video viewing, claim truth, independent trust-root
verification, release custody, clock/frame correspondence, Stage-2
prediction-link review, or authorization. The synthetic fixture must not be
treated as a human review.

## Faithfulness progress and next action

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. No empirical `ΔF`, confidence interval, or test exists. This
integration test improves the formal handoff only. Next, verify the synthetic
Stage-2 binding against the same source rows and keep all score calls denied
until eligible human review and independently pinned trust/release evidence
exist. Rights status remains unverified; protected artifacts stay local.
