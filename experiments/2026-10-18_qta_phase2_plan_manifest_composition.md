# QTA phase-2 plan-manifest composition

## Purpose

Bind each of the three phase-2 nine-branch plan rows to the exact signed phase-1
cohort record, its phase-1 structural row-binding digest, and the corresponding
baseline p/q manifest row. This closes a composition gap: the phase receipt
verifier signs artifact bytes but intentionally treats the plan artifact as
opaque. The new checker remains non-authorizing.

## Setup

- Date: 2026-10-18.
- Inputs: deterministic synthetic clock/processor adapter fixtures, the actual
  three-row synthetic cohort preflight, synthetic five-option baseline scores,
  and temporary Ed25519 trust rosters created by the receipt tests.
- Phase 1 fixes the ordered three-row cohort. Phase 2 must carry the exact same
  receipt bindings and `freeze_id`.
- The row adapter's review token is a 24-character opaque token, while signed
  receipt bindings require a 64-hex opaque token. The composition therefore
  binds `SHA256(ASCII(review_token))` to the receipt token and never emits the
  source review token in its report.
- No protected video, live reviewer identity, production trust root, model, or
  GPU was used.

## Implementation

`scripts/qta_phase2_plan_manifest.py` verifies the signed phase-2 receipt and
its recursively signed phase-1 receipt before trusting the plan bytes. It then
requires: (1) the same freeze ID and ordered record/batch bindings across
phases; (2) three unique phase-1 row hashes from the atomic preflight; (3) a
canonical baseline p/q manifest with three records, valid finite log
probabilities, and recomputed deterministic p/q indices; and (4) each plan's
row hash, batch hash, receipt token, baseline row digest, and p/q indices to
match those upstream artifacts. The complete nine-branch plans are passed
through the existing validator and checked against the frozen layers, logit
bias, and sham seed from the existing three-row aggregator. Parser inputs are
size-capped. Output explicitly denies model loading, scoring, and execution.

The follow-up also upgrades `qta_baseline_pq_candidate` to emit manifest v2.
Each row now contains a canonical score-binding descriptor and its SHA-256,
covering A–E token IDs, choice-token binding digest, prompt/input hashes,
score position and input token, processor batch, log probabilities, p/q, and
five run digests. The phase-2 plan's `score_binding_sha256` must equal this
descriptor hash. The prediction-link renderer and validator accept both v1
and v2 manifests; only the phase-2 composition gate requires v2.

## Verified results

- Focused baseline candidate, prediction-link review, renderer, phase receipt,
  schedule, adapter-composition, and cohort-preflight checks: **59/59 passed**;
  candidate/review/receipt core including output-size limit: **40/40 passed**.
- `py_compile` passed for the v2 builder, phase-2 verifier, review validator,
  renderer, and affected tests.
- Full repository test discovery from the repository root with `PYTHONPATH=.`:
  **1,922/1,922 passed** in 63.176 seconds after all changes. Earlier full-suite
  runs passed 1,920 and 1,921 tests at intermediate revisions. An initial
  discovery attempt from inside `scripts/` was invalid (12 path/import
  failures); it was discarded and root-based discovery was used.
- Positive synthetic end-to-end composition verified both signatures, the
  recursive phase link, all three plan rows, and the fixed 24-forward phase-2
  schedule (27 including phase 1).
- Negative tests rejected a changed signed baseline artifact, a different
  phase-2 freeze ID, and a plan whose layer/bias/sham seed differs from the
  frozen QTA constants.
- Independent audit confirmed the first composition change. A new independent
  audit confirmed v2 hashing, phase-2 score-hash matching, and review-page
  compatibility; it found no critical issue within the non-authorizing scope.
  A further check verified that the reviewer HTML displays the protocol hash
  but omits score-binding field names and model/tokenizer/scorer/processor
  hashes. The output-size cap also has a dedicated failure test.
- These are structural synthetic checks only; they do not establish real-data
  validity, human judgment correctness, reviewer identity, or model faithfulness.

## Source hashes

- `scripts/qta_baseline_pq_candidate.py`:
  `a5e9af0259814fcf2300f389eb14f3d7094b6bff131156ee6108fecaa5fe3ede`
- `scripts/qta_phase2_plan_manifest.py`:
  `9cc53e234771ad43c200df4725743a2837c4d8640780b893e8795086fbf24719`
- `scripts/nextgqa_prediction_link_review.py`:
  `036842ea7d528dbf7f8fe0ad4cbfc180386412f16d4c9783cf4c0bbe3e03655a`
- `scripts/render_nextgqa_prediction_link_review.py`:
  `426e45ee1093d5248b5f674c30cb7dae16e94aab6e0c9054a0329f53736333a2`
- `scripts/test_qta_baseline_pq_candidate.py`:
  `4012d78aa3c60efee09b375370975c86022d5a1d244df298348a609d00370908`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `2299db3102fdb9da60d065cf2f26602074a2bc019977b508f0a89d39bf141e93`

## Limitations and current faithfulness progress

This composition checker does not authenticate the semantic correctness of a
plan, the origin of a score descriptor, human review, or the mapping from
browser-clock footage to decoder PTS. A hash proves internal byte consistency,
not that the log probabilities came from an actual Qwen forward. The production
trust root, reviewer identities and signatures,
review evidence, and prediction-link judgments remain missing. Verified direct
faithfulness outcomes remain **0/3**; capture/player-to-decoder clock pairs
remain **0/15**; prediction-link judgments remain **0/3**; and there is no
direct score, effect estimate, confidence interval, or significance test. This
work improves protocol wiring, not empirical faithfulness evidence.

Independent source audit found that target/sham group indices are still only
shape-checked: the plan does not recompute top-k selection from baseline
question-attention scores. Existing phase-1 prompt/token artifact bytes are
not yet parsed by the phase-2 verifier, and the current phase-2 artifact roles
do not include a baseline-attention capture or forward receipt. Recomputing a
selection from caller-supplied vectors would establish arithmetic consistency
only; it would not authenticate those vectors as Qwen attention. The next
protocol step is to specify a post-baseline attention artifact with exact batch,
question-position, model/runtime, layer, and output bindings, then test its
selector recomputation without granting execution authority. The independent
human intake still lacks a separately pinned trust root and actual reviewer/key
assignments, so no review package has been opened.

## Next action

Keep this candidate out of production execution gating until the unresolved
selection, stage-1 receipt, and baseline-forward provenance joins are specified
and independently checked. Continue the human-free
protocol and measurement work while the assigned reviewers prepare the local
review handoff. Do not run model scoring until the upstream review and trust
gates are actually satisfied.
