# Visual-faithfulness literature check for the next review-gated study

## Purpose

Check whether current evidence supports extending the claim-level study toward
internal prediction-dependence measures, without retrofitting the already-run
mask-specificity result.

## Primary sources reviewed

- Song et al., *Visual Attention Faithfulness in Vision-Language Models is
  Heterogeneous*, arXiv:2609.00830v1 (submitted 2026-09-01; arXiv page lists
  EMNLP 2026): https://arxiv.org/abs/2609.00830
- Wei et al., *Visual Causal Scene Refinement for Video Question Answering*,
  ACM Multimedia 2023: https://arxiv.org/abs/2305.04224

## Findings from the papers

Song et al. intervene on Qwen2.5-VL and InternVL2.5 visual tokens, measuring
both comprehensiveness (effect of removing a selected region) and sufficiency
gap (effect of retaining only that region), with equal token budgets for their
human-region and attention-ranked comparisons. They report heterogeneous
sample-level modes and that human regions satisfy comprehensiveness in about
60% of cases; on their VQAv2 subset, mean comprehensiveness was 0.505 for
human-grounded regions and 0.813 for attention top-k under the same token
budget. Their other datasets are document tasks (VRDU and ChartQA), not video.
These are the authors' reported findings, not a reproduction by this project.

Wei et al. study temporal and frame-level causal-scene discovery for VideoQA
through a trained question-guided refiner and causal-scene separator. That is
relevant VideoQA context, but an intervention-based training architecture and
benchmark accuracy do not by themselves establish that a natural-language
explanation faithfully identifies the evidence used by a frozen VLM.

## Implication for this project

The completed local mask-specificity diagnostic compares the prediction-margin
effect of one unreviewed annotation union with one fixed translated placebo.
It does not measure sufficiency, use human-reviewed claims, or show that the
model's internal attention is a faithful explanation. The recent VLM results
support treating faithfulness as sample- and intervention-specific and treating
human evidence regions as a reference that can diverge from model reliance.

For a future study after the review gates pass, preregister paired
claim-linked comparisons at an equal visual-token budget: reviewed claim/frame
regions, attention-ranked regions, a deterministic translated or random control,
plus both removal and retention interventions with fixed answer scoring. Report
both necessity-like and sufficiency-like behavior and keep the claim-level
prediction-link judgment separate from model-internal attention. This is a
design proposal only; it must be frozen before outcomes and cannot be used to
reinterpret the completed 147-row exploratory run.

## Limits and next action

The cited VLM paper uses image/document tasks and its own intervention point;
the VideoQA paper uses a separately trained architecture. Neither validates
the current QTA estimator or Qwen2.5-VL video-token treatment. The next
confirmatory step remains blocked on signed human-reviewed claims and evidence
correspondence. Continue local software and synthetic validation while that
input is unavailable; do not run another outcome-adaptive score on the same
cohort.
