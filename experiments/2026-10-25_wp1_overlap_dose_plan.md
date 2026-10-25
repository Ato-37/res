# Prespecified plan: WP1D, overlap dose-response of the evidence-specific video->text knockout (written 2026-10-25, before any WP1D run)

## Question
WP1 compared only the two ends: block the annotated evidence groups (100% overlap) versus an equal-count disjoint sham (0% overlap) on the video->text edge, layers 6-20 (evidence minus sham -0.68 nat, N=112). Is the effect graded in the overlap? A straight dose-response would strengthen "the answer depends on the annotated interval" over alternatives where any 2 blocked groups of a certain kind hurt equally; a flat-then-jump or flat curve would say the end contrast is carried by something other than overlap per se. Label-free; scope = answer margin on NExT-GQA, not explanation faithfulness.

## Design (fixed now)
- Cohort: the WP1 cohort (NExT-GQA test, outcome-blind hash-ranked candidates 301-500, same structural and sham screens; N = 112 eligible; same manifest). Same videos as WP1: the 0%/100% arms reproduce WP1's primary contrast within video (a consistency check, not an independent replication).
- Per video: baseline; video->text knockout (text rows cannot attend to the keys of the blocked groups, layers 6-20, mask -inf) of k = |annotation groups| groups whose overlap with the annotation is m = round(f*k) for f in {0, .25, .5, .75, 1}: the m groups from the annotation are hash-first, the k-m others hash-first outside the annotation (m=0 = the WP1 sham, m=k = the WP1 evidence set); duplicate m (small k) run once. Whole-video video->text positive control. Up to 7 forwards/video.
- Endpoint: margin change (fixed gold-minus-baseline-foil margin, nat) per arm minus baseline.
- Primary (one analysis, video = unit): per-video least-squares slope of the margin change on the achieved overlap fraction m/k; effect = -slope (nat per 0->100% overlap). Supported iff t, sign-flip and Wilcoxon p < 0.05 and mean effect >= 0.3 nat (delta_min justified in experiments/2026-10-23_practical_threshold_justification.md); else TOST at +/-0.3 nat or inconclusive.
- Secondary (descriptive): mean margin change per overlap level; 0% vs 100% contrast (WP1 P1 on these videos); middle-level (50%) deviation from the line between the ends (curvature; videos with k>=2 only); %p change by level.
- Positive control (must pass for a null to be informative): whole-video video->text mean relative change <= -20% on baseline-correct videos.
- Reporting regardless of outcome: requested/effective N, exclusions, per-level counts, intervals, heavy-tail caveats.

## Limits
One group = 2 sampled frames, annotation boundaries may be uncalibrated, so "100% evidence" contains boundary blur; levels are discrete (k small -> few levels; k=1 only 0/100%); N=112 limits power for curvature; one band (6-20), one model, one split; same videos as WP1.

## Workload
About 112 x 7 = 780 forwards, about 1 hour; queued after the CSL run. Runner scripts/run_wp1_overlap_dose.py, set builder scripts/qta_overlap_dose.py (+ test), analysis scripts/analyze_wp1_overlap_dose.py (+ test). Not yet run on GPU.
