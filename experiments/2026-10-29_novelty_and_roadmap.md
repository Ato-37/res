# Novelty map and roadmap after the Map-the-Flow transfer (2026-10-29)

Scope: planning document, no new measurement. Everything below is within the tested scope (Qwen2.5-VL-7B/3B, NExT-GQA, TVBench 4 tasks, attention knockout); none of it is an explanation-faithfulness verdict.

## 1. Where the project stands relative to the base paper
Paper: attention-knockout map of information flow in VideoLLMs (cross-frame -> video->question -> question->last), effective-pathway pruning, logit lens.
Our reproduction ladder (own re-implementation of the official protocol; official code not run):
- Holds on Qwen2.5-VL / TVBench: cross-frame, video->text, text->last dependence (WP0), Table 2 first half, Figs 2, 3, 7, window ablation.
- Does not hold: Table B second-half drop (Qwen 1.6-27.9% vs paper <=1.7%), Fig 9 (rise only at layers 24-27). So Qwen2.5-VL keeps cross-frame dependence late in the stack where the paper's models do not; this is a real model difference, not a bug (positive controls pass).
- Pending: pruning (Table 3), logit lens (Figs 4-5), attention maps (Fig 6), other models/benchmarks.

## 2. Where novelty can honestly be claimed ("not found in the literature we searched", never "first")
N1 evidence-specific pathway dependence: knockout of video->text on the annotated evidence frames vs sham (NExT-GQA): -0.68 nat, dose-response, replicated across cohorts. The paper blocks whole classes of tokens; evidence-specific blocking is ours.
N2 the model's own cited interval as the intervention target (necessity / sufficiency / input-level / stability, WP2-WP7): effect concentrates where the citation overlaps the annotation, not on start-anchoring.
N3 a negative result: annotation-free correction (CSL) only sharpens confidence; evidence-focusing was not confirmed.
N4 cross-model check (3B): same direction, weaker.
N5 (pending) Qwen-specific late cross-frame dependence (Table B) as a characterization contrast to the paper's models.
N6 (pending) pathway pruning on a model family the paper did not test, with a matched random control.
N7 (pending, extension) causal test of the paper's logit-lens claim: does cross-frame blocking remove temporal-keyword emergence in video tokens?

## 3. What makes it interesting (one sentence each, for the talk)
- The paper says WHERE information flows; we ask whether the flow is used on the RIGHT evidence, and where the model's own citation disagrees with the annotation.
- Qwen2.5-VL differs from the paper's models in late-layer cross-frame use, so "effective pathway" is model-specific and pruning ranges cannot be copied (tested by P1/P2 of the pruning run).
- Negative results are informative: annotation-free correction does not help; selective prediction is inconsistent.

## 4. Roadmap (ordered by expected information, each with a prespecified gate)
R1 pruning (Table 3 transfer) -> run now; P1/P2 in experiments/2026-10-29_map_the_flow_pruning_plan.md.
R2 logit lens + cross-frame-blocked extension -> experiments/2026-10-29_map_the_flow_logit_lens_plan.md.
R3 attention maps (Fig 6): video->question attention mass on annotated evidence vs cross-frame-blocked (needs eager attention on one layer band; CPU analysis of saved maps). Gate: only if R1 or R2 reveals a pathway difference worth localizing.
R4 conditional on R1 (if Qwen's pathway differs): pathway-aware ranges derived from a DEVELOPMENT split of TVBench (disjoint from the test split), then one confirmatory pruning run on the held-out split. Never tune ranges on the evaluated cohort.
R5 connect to faithfulness: use the pruned/knockout model as a counterfactual model for the cited-interval metrics (does necessity of the citation survive when only the effective pathway is intact?). This is the thesis-level link; it needs R1 to pass P1.
R6 correction line (the user's "補正"): only prespecified, annotation-free, one primary endpoint; the earlier CSL null means a new mechanism is required (e.g. late-layer cross-frame reinforcement guided by Table B), with a positive control; otherwise stop the branch.
R7 reproducibility: official code + official checkpoint run (needs the user's permission; permission layer denied cloned-code execution earlier); until then results are labelled "own re-implementation".

## 5. Presentation structure (interim talk next week)
1 goal and position; 2 the base paper in 3 slides (what they measured, three findings, why it matters); 3 what "reproduction" means here (ladder table: holds / differs / pending); 4 our extension N1-N2 with the evidence-vs-sham figure; 5 negative results; 6 limits and next steps R1-R7. Speak slowly: one message per slide, no best-in-table claims.
