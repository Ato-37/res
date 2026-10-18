# STEAR primary-source reproducibility follow-up (2026-10-18)

## Purpose

Resolve whether the published STEAR method can be frozen as the internal
intervention for the thesis outcome selected by the user: improved selective
prediction dependence on evidence stated in an explanation. This is a
primary-source methods audit, not a model experiment.

## Verified source details

The fixed source is arXiv v1, submitted 2026-04-03:
<https://arxiv.org/abs/2604.03045v1>.

- The 10-page main paper describes uncertainty-triggered operation using
  normalized entropy over layers 5–16, threshold 0.85, and KES selection of the
  top 10% attention-salient patches. It specifies temporal homogenization
  coefficient `gamma=0.80` and contrastive strength `alpha=0.75`.
- Its reinjection equations reuse a backbone visual projection, form local
  key-value memory, apply layer-specific query/key/value/output transforms,
  and add a scaled retrieval vector to intermediate token states. The main
  paper does not report a numerical reinjection scale `lambda` or an
  operational boundary/width for the trigger-layer neighborhood. It says code
  is in supplementary material. The arXiv TeX-source endpoint was inaccessible
  through the available source browser; no claim is made about what the
  supplement contains.
- The Qwen2.5-VL-7B row reports NExT-QA official accuracy of 59.63% baseline
  and 60.87% with STEAR (absolute `+1.24` percentage points). The paper's
  headline comparison is benchmark accuracy; this is not the thesis's
  fixed-prediction claim-evidence dependence estimand or `DeltaF`.
- In the pinned local Transformers Qwen2.5-VL implementation, the language
  decoder uses self-attention with interleaved video tokens and has no separate
  cross-attention sublayer. Using answer-to-visual-token self-attention as the
  paper's cross-attention signal, or reusing Qwen's self-attention projections
  for its local memory, would therefore be a Qwen-specific adaptation, not a
  faithful reproduction established by the paper.

## Decision and limits

STEAR remains prior art and a possible separately named comparator. Do not
freeze it as the primary treatment from the main paper alone: key operator
details are unspecified there, the supplementary source was not obtained, and
the proposed Qwen mapping changes the method. Do not tune missing parameters on
the current outcome cohort. A reproduction claim requires the supplement or
author implementation plus a source-level Qwen mapping; a new adaptation would
need an explicit operator, fixed control/sham, and outcome-blind freeze before
any efficacy score.

## Faithfulness progress

`未実施`: this audit produced no direct model/dataset/intervention/metric
measurement. The active claim-linked QTA branch remains at 0/3 independently
reviewed claims, 0/15 verified browser-to-decoder correspondences, 0/3
stage-2 prediction-link judgments, and 0/3 scores; no empirical `DeltaF`,
uncertainty interval, test, or multiplicity adjustment exists. The audit is
literature/protocol support only. Human/trust/capture/calibration/production
phase gates remain unresolved. Rights status is unverified; protected data
remain local.

## Next action

Continue the fixed claim-linked QTA branch; obtain actual independent review,
capture and calibration evidence before using GPU scoring. Revisit STEAR only
if its supplementary source becomes available independently of the outcome
cohort.
