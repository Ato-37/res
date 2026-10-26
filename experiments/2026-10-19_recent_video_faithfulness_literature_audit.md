# Recent video-faithfulness literature check — 2026-10-19

## Purpose

Check whether recent primary studies suggest a justified new GPU branch or change the thesis estimand, following the completed synthetic QTA construct run and the blocked formal review handoff. This is a scoped literature audit, not a project experiment.

## Primary-source observations

- Liu et al., [On the Faithfulness of Visual Thinking: Measurement and Enhancement](https://arxiv.org/abs/2510.23482) (arXiv v1, 2025-10-27) study agentic image LVLM multimodal CoT with inserted visual observations. They intervene on textual or visual thought components and separately evaluate reliability and sufficiency; the visual component is replaced with noise in their intervention. The paper's evidence concerns image-based agentic MCoT, not free-text explanations for passive video QA, and its sufficiency judgments use an external LVLM judge. This supports separating whether an explanation is factually supported from whether it contains enough evidence, but does not supply an operational substitute for claim-linked human review in this project.
- Yang et al., [INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs](https://arxiv.org/abs/2603.11481) (arXiv v1, 2026-03-12) report clean, visual-degradation, evidence-corruption, and temporal-intervention conditions. Their temporal shuffle/reversal score tests order sensitivity on order-dependent items; it does not localize a particular explanation claim to a reviewed interval. Their appendix reports a three-annotator audit of label preservation for selected invariant-label perturbations. This is relevant as a possible later secondary robustness check, not as an explanation-faithfulness score or replacement for current interval review.
- Huang et al., [EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidence](https://arxiv.org/abs/2606.24797v2) (v2 dated 2026-10-04) evaluate answer-plus-evidence outputs with temporal-semantic matching. This addresses evidence localization and answer correctness together, but the project has no local EG-VQA media and its metric would not test whether an internally intervened model relies on a reviewer-validated explanation.

These are author-reported preprint results and protocol descriptions. This audit did not independently reproduce their experiments.

## Decision and next action

The project's current question remains distinct: after independent claim-to-video review and verified clock/frame mapping, does a frozen internal intervention increase the fixed baseline prediction's selective dependence on the reviewer's evidence interval versus a matched control? The recent papers do not justify reopening closed head, QTA-proxy, or synthetic cohorts, nor starting a new dataset run without local data and a separately frozen estimand. Keep temporal-order robustness as a possible secondary outcome only if its item subset and answerability are fixed before outcomes and data are available. The immediate evidence-bearing action remains completion of the local multi-review handoff, followed by the trust/correspondence gates; no new GPU run is justified by this literature audit.

## Current evidence boundary

The local GPU synthetic-event cohort completed 40/40 pairs, 80/80 clips and 720/720 forwards without failures; its equally weighted primary was -0.01719 nat (95% CI [-0.05415, +0.01978], p=0.35055). It is a synthetic construct check, not evidence for natural-video faithfulness. The quick-review JSON inventory contains one unique informal respondent (a duplicate export is deduplicated); it has three claim impressions and no structured sufficiency field. Formal progress remains 0/3 claim reviews, 0/15 clock/frame correspondences, 0/3 prediction-link reviews, and 0/3 direct scores. No direct DeltaF or interval exists.
