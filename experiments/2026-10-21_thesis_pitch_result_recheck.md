# Thesis pitch result recheck (2026-10-21)

## Purpose

Create a concise advisor-facing Japanese thesis pitch from already completed
internal-intervention results, then independently recheck the numeric claims
against private receipts and existing analyzers. This is not a new experiment
and contributes no new efficacy evidence.

## Rechecked evidence

- The completed interval-generation raw-logit receipt is
  `data/private_nextgqa/interval_generation_cohort30_logits_full_20261021_01.json`
  (SHA-256
  `807c77719878165ac95832a6e1fa44d260b55084d6360876452f12dc7c28fb41`,
  owner-only mode 0600). The runner's `validate_receipt` accepted all three
  frozen rows when bound to the exact local packet SHA. Independently
  recomputed per-video mean selected-minus-baseline token log-probability
  shifts were `+0.00582963`, `+0.01053816`, and `+0.02474871` nat/token
  (mean `+0.01370550`). Selected-minus-sham values were
  `−0.00114691`, `−0.00003081`, and `+0.02375332` (mean `+0.00752520`).
  Nevertheless, baseline, selected, and sham intervals were identical on all
  three rows; each condition had mean tIoU `0.08300`; paired IoU differences
  were `[0, 0, 0]`. Token steps remain nested within three videos, so no
  inferential test or interval was computed.
- Re-ran `scripts/analyze_qta_full_video_accuracy_diagnostic.py` on its
  default private receipt. It verified 27/27 records and 81/81 forwards;
  all conditions had 15/27 correct, predictions changed on 0/27 rows, and
  maximum absolute gold-margin changes were 0.375 nat under both QTA and sham.
  Receipt SHA-256:
  `f17764b6503d3a499d6acc95cca27c02fbe823c1a1f7298b272d0aac09767bfe`.
- Re-ran the head-confirmation CPU analyzer on the committed JSON and
  manifest. It reproduced 79/79 complete rows, selective effect `−0.0756`
  nat, 95% t CI `[−0.1578,+0.0065]`, and failure of the preregistered
  `+0.25 nat` success criterion. This estimand is separate from generated
  explanation faithfulness.

## Deliverable and interpretation

Added the short advisor-facing pitch at
`docs/thesis_pitch_internal_intervention_ja.md`. Its recommended thesis angle
is the measurable gap between internal-score movement and external prediction
or interval behavior under baseline/sham controls. This is a feasible negative
or inconclusive measurement-validation study; it is not evidence that
faithfulness improved. Current direct claim-support evaluation remains 0/3.

These experiments span different cohorts and endpoints and were not pooled.
The interval study uses NExT-GQA QA-time annotations rather than independently
verified support for a generated explanation. The 27-video study is post-hoc
and reused rows. The head result is scoped to its fixed preregistered
estimand. None establishes that all internal interventions fail or that model
explanations are unfaithful in general.

## Verification and next action

The two existing CPU analyzers completed; the private interval receipt passed
the current runner's schema/arithmetic validator and exact packet-hash binding;
the advisor document has no trailing whitespace and all five referenced local
files exist. No GPU job, human review, publication, or source-code change was
performed in this recheck. Git publication is unavailable in this workspace's
read-only `.git` mount, so the new advisor document and this note remain local.

To advance a direct improvement claim, run the frozen support-key versus
matched-sham fixed-claim intervention only after a usable claim-to-video
support mapping is available. Until then, the thesis should present the score
versus behavior dissociation as the principal measured result and mark direct
faithfulness as unmeasured.
