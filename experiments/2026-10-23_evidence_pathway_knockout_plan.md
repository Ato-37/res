# Prespecified plan: evidence-specific pathway knockout (WP1) on Qwen2.5-VL-7B (written 2026-10-23, before any WP1 run or cohort selection)

Base paper: Kim, Kim & Han, *Map the Flow* (ICLR 2026, arXiv:2510.13251): video information reaches the answer through video -> question/option rows (middle layers) and question -> last (middle-to-late layers), with
little direct video -> last flow. WP0 (separate plan) replicates the pathway profile on Qwen2.5-VL by blocking edges for the WHOLE video. The base paper does not ask whether the information that matters comes from the **evidence frames**
(NExT-GQA's annotated time interval). WP1 does: block only the keys of the annotated evidence groups on a pathway, versus an equal-count disjoint hash sham. Label-free, no human review. Scope: pathway-level dependence of the
answer on the annotated interval; not explanation faithfulness.

## Design (fixed now)
- Cohort: NExT-GQA test split, outcome-blind hash-ranked candidates 301-500 (ranks 1-300 belong to WP0 and the evidence-focusing Stage 2), same structural screens as before; videos whose annotation covers more than half of the 16
  temporal groups (no disjoint equal-count sham) are recorded as excluded, not replaced. N = number of eligible videos (expected about 150-190). Model, readout, frames, seed as in WP0 (Qwen2.5-VL-7B-Instruct, 4-bit/BF16, fp32 readout, 32 frames).
- Per video: unmodified forward, then knockouts (attention-mask cells set to the dtype minimum, audited) in two layer sets: BAND = layers 6-20 (the base paper's video->question range, clipped to this model's 28 layers) and ALL = layers 1-28.
  Arms: (1) V2T_evidence: text rows T cannot attend to the video keys of the annotated groups; (2) V2T_sham: same for the equal-count sham groups; (3) V2L_evidence and (4) V2L_sham: the last position cannot attend to those keys;
  (5) V2T_all: T cannot attend to ALL video keys (positive control). Eleven forwards per video.
- Endpoints per video: change in the fixed gold-minus-baseline-foil margin (nat) and, for baseline-correct videos, the relative change of the predicted option's probability (%p, as in WP0).
- Practical threshold: delta_min = 0.3 nat (justified in experiments/2026-10-23_practical_threshold_justification.md for this margin endpoint).

## Primary hypotheses and decision rules (prespecified)
- P1 (BAND): contrast = margin change under V2T_evidence minus under V2T_sham. Effect if two-sided t-test p < 0.05, sign-flip p < 0.05 and mean <= -0.3 nat (evidence keys matter more than sham keys on the video->text route).
- P2 (BAND): the same contrast for V2L (the base paper predicts little direct video->last flow): reported with TOST equivalence at +/-0.3 nat.
- Multiplicity: Holm over P1 and P2 for the effect tests. Secondary, descriptive: the ALL-layer versions, %p contrasts, and the arm decomposition (evidence - baseline, sham - baseline).
- Positive control (must pass for the nulls to be informative): mean %p of V2T_all (BAND) <= -20% on baseline-correct videos (WP0's pilot profile shows large drops on this edge). If it fails, all V2T nulls are uninformative.
- Reporting regardless of outcome: requested/effective N and exclusions, means with 95% intervals, medians/trimmed means, Wilcoxon p-values, heavy-tail caveats, prediction-change rates.

## Limits
Annotation intervals are uncalibrated to decoded PTS and may not contain all evidence; the sham differs in position and content (not a pure null); knockout cells are large (two sets of layers only, no layer search); one model, one split.
Test videos were used by unrelated earlier experiments (ASSUMED-CLEARED); cohort independent of WP0 and Stage 2 by rank.
