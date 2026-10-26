# Prespecified plan: progressive knockout (signal-leakage check of the pathway band) (written 2026-10-26, before any run)

## Why (reviewer concern of the base paper, applied to our band choice)
In the OpenReview discussion of Map the Flow (ICLR 2026, Accept Poster) reviewer NrDF noted that blocking a pathway in only some layers risks leaking the information through other layers or edges, and asked for cumulative knockouts (layers 1..N); the authors added such a progressive knockout. Our WP0/WP1 use two bands (6-20 and 1-28) chosen from a window profile, with no cumulative check, so the same concern applies. This experiment checks on Qwen2.5-VL-7B whether the video->text effect is carried by the identified middle band.

## Design (fixed now)
- Cohort: the WP0 cohort (NExT-GQA test, outcome-blind hash-ranked candidates 1-80, same screens and manifest); baseline-correct videos only, as in WP0 and the base paper. Model, precision, fp32 readout, 32 frames as in WP0.
- Edges: video->text and text->last (the two edges that carry information to the answer in WP0). Whole-video blocks (no evidence/sham split; this is a pathway-level check).
- Conditions per edge: forward cumulative layers 1..N and reverse cumulative layers N..28 for N in {4, 8, 12, 16, 20, 24, 28} (N=28 forward and N=4 reverse are all-layer / last-layers references). 28 forwards per correct video; about 55 correct videos expected, about 1,550 forwards, about 2.5 h.
- Endpoint: relative change (%) of the probability of the originally predicted option (as in WP0), mean per condition with a bootstrap 95% interval (videos resampled).
- Prespecified descriptive criteria for video->text (not significance tests; any failure is reported and the 6-20 band is then flagged as not validated): C1 forward N=28 mean <= -20% (the edge matters); C2 |mean(N=28) - mean(N=20)| < 10 points (no extra dependence in layers 21-28); C3 reverse N=24 (layers 24-28) mean >= -10%; C4 forward N=4 (layers 1-4) mean >= -10%.
- Not run: video->last and cross-frame (cost); they were not the claims of WP1.

## Limits
80 videos, baseline-correct subset only; relative-change endpoint (not the margin); one model; thresholds are descriptive (taken from WP0's effect scale), so the criteria describe consistency with the band and are not a test.

## Workload
Runner scripts/run_progressive_knockout.py (not yet run on GPU; smoke n=3 first), analysis scripts/analyze_progressive_knockout.py (+ test). Queued after all other queued experiments (user request); then the reviewer-concern follow-up below.

## Follow-up after the whole queue (user request 2026-10-26): remaining reviewer concerns
The concerns raised in the discussion of the base paper that this project has not yet addressed, with the planned resolution (CPU analyses on existing receipts first, GPU only if needed):
1. Reliance on knockout only: compare the attention-mass group scores (SGC receipts) with the causal group scores (CSL receipts, same videos) per group and per video (rank correlation); a non-knockout view of the same pathway. CPU, existing data.
2. Selection bias (only baseline-correct videos): re-report the primary endpoints split by baseline correct versus wrong from existing WP1/SGC/CSL receipts. CPU.
3. Failure-mode analysis: where baseline-wrong videos lie in the confidence distribution and how crops/knockouts move them (the base paper finds overconfidence in false options and over-reliance on static frames). CPU on existing receipts.
4. Open-ended generation: WP2 (self-cited interval) is the first step; already queued.
5. Long videos: not tested (32 frames only); stated as a limitation unless time allows a 16-vs-32-frame comparison on the WP0 cohort.
6. Fine-grained feature interpretation: out of scope for this thesis; stated as a limitation.
