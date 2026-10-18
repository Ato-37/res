# QTA QA-source contract inventory closure (2026-10-18)

## Purpose

Register the new CPU-only `qta_actual_qa_source_contract` in the local
pre-review candidate inventory. An independent audit found that its source
bytes were not covered even though the contract is intended to bind actual
NExT-GQA question/choice rows for later QTA preparation.

## Implementation and verified result

Added the source contract as an explicit code role in
`scripts/build_nextgqa_pre_review_byte_inventory.py` and extended its test to
require the role. This is an inventory closure change only; it grants no
review, phase, model, or scoring authority.

Checks:

- Inventory builder synthetic suite: **19/19 passed**.
- Python compilation and `git diff --check`: passed.
- A new local candidate inventory was built at
  `data/private_nextgqa/pre_review_inventory_qta_20261018_0912_01/` using the
  existing default inputs. It reports **74 artifacts, 0 missing required
  slots**, includes the QTA QA-source contract, and its registered source hash
  matches the current file.
- The new output directory/file modes are **0700/0600**. Only a sanitized
  aggregate was inspected; private inventory bytes, hashes, filenames, and
  media were not published.

The candidate still reports `review_started=false`, `review_completed=false`,
`joint_freeze=false`, `score_ready=false`, and `execution_authorization=not_granted`.
Rights status remains `unverified_local_only`.

## Interpretation and limits

The inventory now covers this source-contract implementation in the candidate
byte set. `candidate_only` is not a witnessed release, a completed review, a
joint freeze, a phase receipt, or a prebaseline pass. No reviewer judgment,
player-to-decoder correspondence, baseline prediction, model forward, GPU
scoring, `DeltaF`, confidence interval, or test was produced. The actual
faithfulness status therefore remains **0/3** independent claim reviews,
**0/3** prediction-link judgments, **0/15** timing correspondences, and **0/3**
QTA scores.

## Provenance and next action

- Inventory builder SHA-256:
  `94cb0ea23cca2b9f0c13c3f520614c3ad6d0097eeb8f8b345859df30567c6e5b`
- Synthetic test SHA-256:
  `f63e303661fb0891424c957d9af60566ad50cb66370bae535f8a62ccf2900baa`
- QTA source-contract SHA-256:
  `3583031504ed30af05b16a4efc5cb24a53cf51b9e1f064708982abc3723f6609`

Continue implementing a raw-artifact, fixed-three-row, non-authorizing
prebaseline diagnostic only where it can rerun leaf validators and use explicit
independent pins. Do not infer semantic human validation, identity, timing
correspondence, score freeze, or phase authorization from this inventory.
