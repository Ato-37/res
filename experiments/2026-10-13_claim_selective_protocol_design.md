# 2026-10-13: claim-selective intervention protocol design

## Purpose

Define the next distinct faithfulness protocol after the closed head-selective
confirmation. This is a design and contract artifact; no claim-selective
real-data efficacy run is reported here.

## Fixed estimand

For each video, let `S` be the frozen-answer teacher-forced log-margin for the
preselected claim token span. The video-level selective dependence is

`d = (S(full) - S(claim-mask)) - (S(full) - S(matched-control-mask))`.

The primary estimate is the mean of `d` over a baseline-correct cohort fixed
before inference. The matched control has the same number of masked frames and
the closest available temporal rank distance outside the claim interval.

## Required pre-run gates

1. The claim span and temporal evidence interval must be independently recorded
   before model scoring; post-hoc span selection is excluded.
2. Claim and control interventions must use the same processor settings,
   model revision, precision, seed, frame sampling, and answer target hash.
3. The runner must reject duplicate IDs, missing spans, nonfinite scores,
   target-hash mismatches, and incomplete triplets. The CPU contract is covered
   by `claim_selective_dependence_utils.py`.
4. Rights and annotation gates must be resolved for the chosen cohort. The
   current NOAA gate remains `no_go` because human attestation is missing, and
   NExT-GQA clip-level rights are not yet verified.

## Analysis and stopping rules

Report requested/effective counts, failures, baseline-correct count, mean,
standard deviation, confidence interval, and the fixed practical threshold.
Masking can create out-of-distribution inputs, so a positive `d` alone cannot
establish semantic faithfulness. A failed gate or missing attestation stops the
dependent branch; it does not justify changing the metric, span, cohort, or
threshold after outcomes.

## Current decision

Do not run this protocol on the local untracked candidates yet. The high-
resolution GPU work in this cycle established that processor cap settings
change `video_grid_thw` and measured margins materially; therefore the next
claim-selective run must freeze and publish those settings in its manifest.
