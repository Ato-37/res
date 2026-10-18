# QTA–gsub finite-cohort leave-one-video-out check (2026-10-19)

## Purpose

Check whether the existing near-zero matched-minus-cross question-span alignment contrast changes materially when one eligible video pair is omitted. This is a robustness diagnostic for an exploratory selector-localization proxy; it does not test generated-explanation faithfulness.

## Setup and provenance

- Reused the fixed 80-video, 160-question test-associated cohort and the saved QTA selector output. No model inference or GPU work was run.
- Recomputed the same 32-frame / 16-temporal-group mapping against the local NExT-GQA `gsub`, `frame2time`, `upbd`, and video-map files, with the existing clock, media, and processor-grid checks. No rows were replaced.
- Of the 80 requested video pairs, 60 were eligible. Exclusions were 11 source-clock-invalid, 5 annotation-outside-media, 3 processor-grid mismatch, and 1 nonlocalized pair. There were 0 failed rows; all 160 questions joined and 162 intervals were checked.
- For each eligible video, averaged its two question-level matched-minus-cross IoUs, then calculated the 60 leave-one-video-out means. Since every eligible video contributes exactly two rows, the full-cohort mean equals the 120-question-row mean.
- Output is local-only: `data/private_nextgqa/qta_question_swap_gsub_alignment_local_20261019_0340_loo.json`, mode `0600`, SHA-256 `ce0e41dcf99948831b04cd45b4bbe6728341bd057d4c17aabf53e67588fb01e9`. The report stores aggregate influence summaries only, without video identifiers or per-video contributions.
- Analysis source SHA-256: `02fe31326213e24729015a597bee7381dba2c9a3272c355c5ed17fc466d0c348`. Selector result provenance in the report: `44b79f7170b1a05b6199524e9d8f2baef85ada08aae10deb6cdbeb838cd3e4d3`.

## Results

- Full-cohort matched-minus-cross IoU mean: **−0.0038924** over 120 question rows, unchanged from the prior aggregate report. Median was 0.
- Across the 60 leave-one-video-out estimates, the minimum was **−0.0061107**, the maximum **+0.0002789**, and the largest absolute shift from the full-cohort mean was **0.0041713**.
- No hypothesis test or population confidence interval was calculated. No p-values or multiplicity adjustment apply.
- The influence helper's two unit tests and the existing 14 gsub-alignment tests passed (16/16 total); Python compilation and scoped `git diff --check` passed.
- An independent code/output audit confirmed the 60-pair / 120-question weighting and exact agreement of the old and new aggregate means, counts, and exclusions. It also confirmed the reported LOO range is arithmetically plausible within the `[-1, 1]` per-pair delta bounds. Because neither artifact retains per-pair values, the extrema could not be independently recomputed from saved rows; treat them as runner-reported and rerun the input-bound analysis to reproduce them.

## Interpretation and limitations

The descriptive contrast is near zero, and its sign can become slightly positive after omitting one eligible video pair. The direction is therefore not stable to every single-pair omission. The LOO range describes sensitivity within this finite, reused cohort only; it is not a sampling interval or evidence of population effect. NExT-GQA `gsub` labels question-relevant spans, not evidence selected by a generated explanation. The analysis also reuses a test-associated selector cohort, excludes 20/80 pairs, uses 16 coarse temporal groups, and has no independent human browser-clock-to-decoder review. The aggregate-only output prevents later independent recomputation of the individual LOO extrema from stored contributions; the run is reproducible only by re-running the input-bound analysis.

## Faithfulness progress and next action

- Current NExT-GQA direct explanation-faithfulness evidence remains **0/3 claims scored**; formal claim review **0/3**, clock/frame correspondence **0/15**, and prediction-link review **0/3**. No direct `ΔF`, confidence interval, or test was added.
- This is an auxiliary selector-localization proxy only. Historical direct head-intervention evidence remains a separate, negative confirmation and is not updated by this diagnostic.
- No new justified GPU run is available: the human-reviewed evidence and independently verified trust/role materials required by the frozen direct-score branch remain absent, and existing human-free GPU proxies are complete.
- Next: obtain exploratory responses separately if available, then complete the actual independent formal reviewer/trust assignments and evidence review. Once the frozen gates pass, run the predeclared direct-score smoke before the bounded full GPU evaluation. Do not use these overlap results to tune the selector or redefine the endpoint.
