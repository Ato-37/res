# Thesis framing checkpoint: internal intervention and faithfulness (2026-10-21)

## Purpose

Convert the existing real-video internal-intervention evidence into a thesis
question with a defensible fallback result, while retaining the requested
scientific target: an internal intervention should improve dependence of an
explanation on the video evidence it describes. This is an evidence synthesis
and writing change, not a new model experiment.

## Evidence inspected

- `docs/thesis_topic_faithfulness_ja.md` and
  `experiments/2026-10-21_thesis_progress_internal_intervention_synthesis.md`.
- `experiments/2026-10-21_qta_full_video_accuracy_diagnostic.md`: 27 fixed
  real-video rows, 81/81 forwards, baseline/QTA/sham each 15/27 correct,
  identical choices on 27/27 rows, and 0 wrong-to-correct or correct-to-wrong
  flips. This was post-hoc on a reused cohort, not a direct faithfulness test.
- `experiments/2026-10-21_qta_cohort30_internal_key_ablation.md`: three
  post-result rows; QTA-minus-baseline fixed-prediction `DeltaF` mean
  +0.08333 nat, median +0.125 nat, but only 2/3 material positives. QTA-minus-
  matched-sham mean +0.04167 nat is driven by one row; the other two contrasts
  are numerical zero. The selected interval is machine-proposed and unreviewed.
- `experiments/2026-10-21_thesis_progress_internal_intervention_synthesis.md`:
  prior small cohorts include a +0.03927 nat/token mean interaction against a
  severe frame-mean video control. Temporal-order prompt-bound reruns are
  mixed across small cohorts after correcting a legacy-prompt scoring bug;
  see the repair log. The source also labels the claim-conditioned cohorts as
  non-authorizing historical diagnostics. They are not direct efficacy results.
- `docs/head_faithfulness_confirmation_results.md` and
  `experiments/2026-10-09_faithfulness_threshold_and_literature_review.md`:
  the closed prespecified head confirmation was 79/79, mean −0.0756 nat,
  95% t CI [−0.1578,+0.0065], below its +0.25 nat success criterion. This
  result is scoped to that estimand; the historical source/run crosshash and
  sample-independence limitations remain disclosed.
- Formal current direct claim-support score remains 0/3 for the present
  handoff (and formal review workspace score remains 0/25); no direct
  claim-specific effect or interval has been measured.
- Re-ran the privacy-preserving legacy-review inspector on the user's saved
  v5 JSON against its matching old v14 item artifact. It reproduced 3
  respondents × 3 ratings: answer fit 5 match / 3 mismatch / 1 unsure, and
  evidence sufficiency 2 sufficient / 5 insufficient / 2 unsure. The current
  three generated claims have 0 matching IDs; the v5 export has no artifact
  fingerprint. This is prior exploratory feedback, not a current claim label.

## Decision and implementation

The strongest thesis framing is now: **“Does internal visual intervention
improve a video VLM explanation's evidence dependence? A matched-control study
of score/behavior dissociation on NExT-GQA.”** The thesis may report the
exploratory finite-cohort evidence that internal scores moved while choices or
time intervals did not, and the preregistered head result missed its practical
criterion. It must not claim semantic faithfulness improved or that all
internal interventions fail.

This checkpoint independently reran the existing CPU analyzer on
`results/head_faithfulness_confirm_20261009.json` and
`results/head_faithfulness_confirm_20261009_run_manifest.json`. It reproduced
79/79 records, failures 0, mean target margin drop −0.0601 nat, matched control
+0.0155 nat, selective effect −0.0756 nat (SD 0.3667; 95% t CI
[−0.1578,+0.0065]), p versus zero 0.0706, and p versus +0.25 nat
`1.52e−11`; the preregistered success rule remained false. The result JSON SHA
is `993acd16e62c1e8cf8aafe075674cc1e649761f2e1ca5bb05be83814a2e6ac36` and
the run-manifest SHA is
`d3c59d49c3609a03d868592d188507f3df5fc1cd1408612bfb98c315d63e1b16`.
The analyzer reported unique videos as unverified in the current local join;
the separately recorded identity reconciliation still does not establish
result-to-run-manifest crosshash binding or statistical independence.

Added a Japanese advisor-facing abstract, three research questions, an explicit
future primary estimand, and a contribution boundary to
`docs/thesis_topic_faithfulness_ja.md`. The proposed direct primary requires
frozen claim-support spans and an evidence-key vs overlap-matched-sham contrast.
The support review response is still absent, so the formal direct result remains
unmeasured. The prior human feedback motivates separating answer correctness
from explanation sufficiency, but cannot be transferred to the current claims.
No authorization state was altered and no GPU experiment was run.

## Limitations and next action

The positive proxy signals are small, heterogeneous, post-hoc or otherwise
limited, and several historical claim-conditioned runs are explicitly
non-authorizing. The source cohorts are not pooled. Current evidence is
appropriate for an exploratory measurement-validity/negative replication
thesis only. The next evidence-bearing step is to bind the current review
response to its exact artifact and, if valid claim supports are produced,
execute the separately frozen support-key vs matched-sham experiment. If no
response is available, continue with independent audit and thesis-ready
reanalysis of already authorized results; do not label proxy estimates as
direct faithfulness.

At this checkpoint, a no-content scan of **1,079** JSON files across the
private workspace and Downloads found zero responses bound to the current
artifact fingerprint `abaa3b8dee8f4fc9cbd2031d1772d2912502a9c1bafb1f2b03dac90365d8a1d4`.
The exact local handoff is
`data/private_nextgqa/claim_support_frame_review_test_20261021_06.html`.
An individually elevated hardware check found the RTX 5070 Ti available with
15,472 MiB free VRAM and the repository `.venv` reporting PyTorch 2.11.0+cu128
and CUDA available. The direct score runner still denies model loading and
scoring in the absence of authenticated claim/timing inputs. The independently
audited fixed continuation on the default local video root had only 1 test
candidate and 0/33 validation candidates after its known-history exclusions;
the full-archive candidate census is a different, incomplete exposure view.
The earlier QA-gold-mask endpoint is closed due to heterogeneous results, so
neither candidate count justifies extending it. No new GPU run was justified
by this recheck.

## Literature check

Recent primary preprints already describe layer-aware spatiotemporal evidence
intervention for VideoLLM hallucination mitigation ([STEAR, arXiv:2604.03045](https://arxiv.org/abs/2604.03045)),
temporal/spatial contrastive decoding ([SEASON, arXiv:2512.04643](https://arxiv.org/abs/2512.04643)),
and heterogeneous causal faithfulness of visual attention in VLMs
([arXiv:2609.00830](https://arxiv.org/abs/2609.00830)). Therefore, this project
must not claim novelty for generic visual-key steering or causal attention
intervention. The contribution needs to be established through the
claim-level, evidence-validated matched-control evaluation; literature novelty
remains provisional.
