# Practical threshold (delta_min) justification for the answer-margin endpoint (2026-10-23)

Earlier plans used delta_min = 0.025 nat, chosen from a power calculation rather than from the meaning of the effect. This note derives a justified value.

## Evidence (345 videos from the two multi-query cohorts; baseline margins from the receipts)
- Baseline gold-minus-foil margin: mean 1.80, SD 3.50, median 1.75 nat; baseline "accuracy" (margin > 0) 69.9%.
- Decision relevance: share of videos whose baseline margin lies within +/-delta of 0 (a shift of delta could flip the gold-vs-foil decision):
  0.025 -> 0.0%, 0.05 -> 0.9%, 0.1 -> 1.4%, 0.2 -> 2.3%, **0.3 -> 5.5%**, 0.5 -> 9.9%, 1.0 -> 18.8%.
- Standardized effect: pooled SD of the per-video annotation-minus-sham contrast = 1.69 nat, so a small effect (d_z = 0.2) = 0.34 nat; 0.025 nat is d_z = 0.015.
- Noise floor: the fp32 readout removed the 0.125 nat bf16 quantization (smallest nonzero |contrast| about 2e-4 nat); the dominant noise is between-video, not numerical.

## Decision
delta_min = **0.3 nat** (two independent anchors agree: ~5% of decisions within reach, and d_z ~ 0.18-0.2). 0.025 nat had no practical meaning (0.0% of decisions) and is retired as a
practical threshold; earlier prespecified verdicts that used it stay as recorded, and are reported alongside this sensitivity analysis.

## Post-hoc sensitivity of the completed multi-query contrasts (not prespecified; TOST p at +/-delta, annotation-minus-sham)
| run | mean [90% CI] | TOST p, delta 0.3 | delta 0.5 |
|---|---|---|---|
| cohort 1 -ln256 | -0.515 [-0.771,-0.259] | 0.917 | 0.539 |
| cohort 2 -ln256 | -0.235 [-0.415,-0.056] | 0.276 | 0.008 |
| cohort 1 +ln4 | +0.207 [+0.030,+0.385] | 0.195 | 0.004 |
| cohort 2 +ln4 | +0.124 [-0.039,+0.287] | 0.038 | 0.000 |
| cohort 1 +ln16 | +0.449 [+0.119,+0.779] | 0.772 | 0.399 |
| cohort 2 +ln16 | +0.304 [+0.040,+0.568] | 0.510 | 0.111 |
Reading: at +ln4 the contrast is small relative to 0.3 nat (equivalent in cohort 2 only); +ln16 and the cohort-1 suppression are not within +/-0.3. This is a sensitivity analysis on
already-analysed data; it does not change any prespecified decision.
