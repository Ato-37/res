# QTA pre-baseline runner gate audit (2026-10-18)

## Purpose

Determine whether the next real-video baseline-only QTA run can be implemented
and executed safely before the current human/trust gates are complete. The
selected thesis measure is explanation-indicated video-evidence dependence;
the candidate primary outcome is improvement under a fixed internal QTA-Bias
intervention.

## Verified source findings

- The human-review runbook says a production pre-baseline gate and baseline-only
  runner do not exist. It requires stage-1 review exports, clock/frame evidence,
  source targets, prompt/answer token IDs, sampler/runtime/scorer configuration,
  one immutable unmodified full-video batch per record, and all-three-row atomic
  preflight before a phase-1 receipt may authorize exactly three baseline
  forwards. It also states that neither phase-receipt schema nor signature
  verifier exists (`docs/nextgqa_v6_human_review_runbook.md`, lines 230–256).
- The frozen QTA draft requires the complete input/trust gates before model load,
  then exactly one untreated full-video pass per row with no treatment/control
  branches. Any baseline failure stops the cohort. The later prediction-link
  review and second gate precede the remaining 24 forwards
  (`docs/qwen_question_attention_faithfulness_protocol_v1_draft.md`, lines
  301–332).
- `scripts/nextgqa_atomic_batch_freeze.py` is expressly a non-authorizing
  integration primitive and freezes all nine condition batches; its
  `authenticated_gate_check` is a caller-provided callable, not an authenticated
  verifier. It cannot serve as the three-forward phase-1 gate.
- `scripts/smoke_qwen_question_selector_intervention.py` is explicitly a
  synthetic smoke. It runs one baseline pass and then all eight additional
  treatment/sham branches. It is not a real-cohort baseline-only runner.
- The available prebaseline diagnostic and baseline-p/q candidate builder are
  CPU-only/non-authorizing and accept caller pins; neither can authorize model
  loading. The stage-1 source-pin adapter added in this cycle improves exact
  byte/field joins but explicitly does not authenticate review identity or
  semantic correctness.

An implementation-support review and a separate protocol audit independently
reached the same conclusion; the primary agent verified the cited source text.
No protected video or review content was accessed for this audit, and no GPU
work was run.

## Decision and research status

Do not create a production-capable runner that can reach model loading yet. It
would have to invent a missing authorization contract or trust a caller-provided
success boolean. A CPU-only fail-before-weight scaffold could be built, but
without a frozen receipt schema and trusted phase-owner role it would not resolve
the current blocker and would be speculative plumbing.

Direct faithfulness remains **0/3** independently reviewed claims, **0/15**
verified browser/player-to-decoder correspondences, **0/3** prediction-link
judgments, and **0/3** claim-linked QTA scores. No empirical `ΔF`, confidence
interval, or test exists. GPU execution is technically available, but there is
no valid score-bearing input authorization. Rights status remains unverified;
all protected materials and derived review outputs remain local.

## Next action

Complete the external review/release roles and actual stage-1/frame-correspondence
evidence. Once those artifacts and an independently pinned phase-owner trust
root are available, freeze and independently audit the phase-1 receipt schema
and integrated all-row verifier before any model-loader path is wired. Then run
at most three baseline forwards, preserve the outputs privately, and require
separate post-baseline prediction-link review and phase-2 authorization before
any treatment or sham forward.
