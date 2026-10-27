# Plan (not yet implemented or run): WP3 sufficiency of the model's own cited interval (written 2026-10-27, before any WP3 run)

## Why
WP2 (val inconclusive +0.298 nat, test supported +0.307 nat) shows necessity: removing the interval the model cites from the video->text pathway hurts the forced-choice answer more than an equal-count sham. Faithfulness metrics come in pairs (comprehensiveness/necessity and sufficiency); a citation that is necessary but not sufficient may be only one of several carriers. WP3 adds the missing half with the existing machinery: keep only the cited interval (keep-mode attention bias, as in the evidence-focusing/SGC/CSL runners) versus an equal-count sham keep, and compare to the full-video baseline.

## Design (fixed now)
- Cohort: the WP2 test cohort (278 recorded; analysed set = the 265 with a parsed, valid citation) and the WP2 val cohort for a second, separate analysis; each analysed once, no pooling. The citation is re-generated with the same prompt and decoding as WP2 (or read from the WP2 receipts if the receipt stores the parsed groups; to be decided at implementation by verifying the stored fields).
- Arms per video: baseline (full video); K_cited = keep only the cited groups (bias -ln256 on all other groups at all post-video rows, all 28 layers); K_sham = keep an equal-count sham set (same sham construction as WP2); optionally K_rand4.
- Endpoint: answer margin (gold minus baseline-best-foil, nat, fp32 readout). Primary contrast: K_cited minus K_sham (a larger margin under the cited keep means the citation carries more of the answer-relevant information than an equal-count sham). Secondary: K_cited minus baseline (does the cited interval alone preserve the answer; negative = loss of information), accuracy flips with exact McNemar.
- Decision rule: supported iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat (delta_min unchanged), positive control = the WP2 positive control logic (oracle-keep minus sham-keep >= 0.3 nat in the cohort, as in the SGC plan) must hold, otherwise nulls are uninformative; otherwise TOST at +-0.3 nat or inconclusive.
- Workload: 3-4 forwards/video (no knockout sweep) plus the generation: ~278 x 4 forwards, about 1.5-2 h per cohort. Smoke n=3 first; new output tag.

## Limits
Keep-mode is a strong out-of-distribution intervention (about 12 of 16 groups suppressed); sham differs in position and content; citation times may be uncalibrated; one model, one citation format, forced-choice answer; sufficiency of an interval for a forced-choice answer is not sufficiency of an explanation; label-free endpoint, measures dependence on the cited interval, not a faithfulness verdict.

## Next
Implement `scripts/run_wp3_cited_sufficiency.py` by reusing `qta_cited_evidence.py` and the keep-mode hooks, add a CPU test and an analyzer, smoke, then queue test then val.
