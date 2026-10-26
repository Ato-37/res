# QTA validation annotation-proxy study (2026-10-18)

## Purpose and scope

Build a human-review-independent exploratory GPU study of whether the fixed
Qwen answer margin depends selectively on NExT-GQA's question-specific temporal
annotations, and whether the fixed QTA-Bias intervention changes that selective
dependence. This is a benchmark-annotation proxy. It does not evaluate a
model-generated explanation, human-reviewed claim evidence, annotation
correctness, or explanation faithfulness.

This is a separate, explicitly exploratory branch authorized by the user to
advance while claim-level human review remains incomplete. The claim-linked
primary GPU gate in `2026-10-18_qta_treatment_selection.md` remains closed;
this proxy does not satisfy or waive any part of it. Never pool these results
with direct claim-faithfulness scores or describe them as explanation
faithfulness.

## Cohort and selection

Use the complete, already-fixed 30-video NExT-GQA validation pilot manifest
from 2026-10-08, whose bytes are checked against that run's retained manifest
digest before use. The 30-row manifest's exact source digest was verified and
its bytes copied into `data/private_nextgqa/` with mode `0600`; the original
file's permissions were tightened to `0600` because it contains raw QA text,
paths, and identifiers. There is one question per video and 30 unique videos.
An exact media-byte comparison found zero overlap between these 30 videos and
the three videos in the current protected v6 review package. The runner must
repeat that check before scoring. Do not
sample a subset, replace rows, or filter on baseline correctness, answer margin,
QTA/evidence overlap, or any new model output. The exact cohort was previously
used in a separate head-attribution pilot; this reuse and its prior outcomes
make the new analysis exploratory, not independent or confirmatory. The
three-video review cohort is excluded by media-byte hash and is never opened by
this branch.

The local annotation sources are frozen to these SHA-256 values before the new
score run: `val.csv` `ac5a1eb1d6dee9cd84ab7f9c94ab345b2404646f24b79ac182cd2aa9e2d424c1`,
`gsub_val.json` `4feb23e53102cb2d4779e857ff8f9fd9b552fc38435621406b2ed9b0bf76204b`,
and `map_vid_vidorID.json`
`1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685`.

## Frozen measurement

Use the pinned local Qwen2.5-VL-7B-Instruct checkpoint in 4-bit with BF16
compute, eager attention, and the frozen 32-frame endpoint-inclusive sampler
per video. The historical head pilot used 16 frames; this QTA study follows
the currently frozen 32-frame QTA sampler. From the
untreated full-video forward, select QTA temporal groups using average
question-query attention over decoder layers 5–16, top `max(1, ceil(0.10*T))`
complete temporal groups, and earliest-index tie breaking. Freeze this
selection across every branch. Apply the fixed `ln(2)` visual-key attention
logit bias at layers 5–16.

Map every NExT-GQA temporal annotation interval onto complete Qwen temporal
groups through the existing processor-batch geometry helper. Before loading
the model, require a nonempty evidence set and an equal-count, disjoint
rank-matched control set for every row. This mapping uses the dataset's
annotation timestamps and the decoder-relative PTS origin; no browser/player
clock calibration or independent human validation is claimed.

Keep the dataset's `gsub` duration as the annotation-validity bound. For
midpoint-cell geometry, use the explicit sampled-support endpoint
`max(gsub_duration, final_sampled_relative_PTS)`, because a dataset duration
can precede an actual decoded final sample. Validate every annotation interval
against the original `gsub` duration before mapping, preserve its original
values, and use the extended endpoint only to clip the sampled-frame support
cells and choose controls. Record both values and their difference. This is a
uniform, outcome-blind rule for the entire fixed cohort, not a per-row repair.

For each row, run the baseline full input, baseline evidence-key ablation,
baseline control-key ablation, QTA full input, QTA evidence-key ablation, and
QTA control-key ablation. Reuse the baseline-full pass to freeze `p` (highest A–E score) and
`q` (highest-scoring remaining A–E choice). Let `m` be the fixed `p` versus `q`
next-token log-probability margin. The row-level outcome is:

`DeltaF = [m(QTA, control-ablated) - m(QTA, evidence-ablated)] - [m(baseline, control-ablated) - m(baseline, evidence-ablated)]`.

Positive values mean the QTA treatment increased the fixed prediction's
selective dependence on the benchmark-annotated temporal groups under these
key-ablation conditions. The evidence/control overlap with selected QTA groups
is descriptive only and must not determine eligibility or subgroup analysis.

## Execution and analysis gates

First run a one-row real-video wiring smoke from the frozen cohort, with all
six forwards and exact output/provenance checks. If it passes, run all 30
frozen rows (180 requested forwards) without changing settings or replacing
failures. Based on the prior 32-frame rate of 50 forwards in 236 seconds, the
full cohort is roughly 14 minutes of model-body time; model loading, cohort
preparation, and video variation add uncertain overhead. Recheck current VRAM
and host-memory guards immediately before
loading and before each forward. Keep all raw inputs and results under the local
private directory.

Report requested/effective rows, completed/failed forwards, finite-cohort mean,
sample SD, observed range, positive/negative/zero counts, and full-input
restricted-choice accuracy. Do not report a confirmatory p-value, population
confidence interval, or causal claim. If any row or branch fails, retain the
failure denominator and do not present a complete-cohort effect estimate.

## Human-review handoff status

The separate protected claim-review handoff remains a candidate only. The
private input inventory contains 77 roles and has zero missing slots; a
companion private provenance note binds its exact package, staged-media, and
standalone-HTML paths to the inventory bytes. Review has not started, the
review is incomplete, no joint freeze exists, and execution authorization is
not granted. Reviewer identities, trust root, witness, and signed receipt are
still needed for that separate branch.

Direct generated-claim faithfulness remains **0/3 independently reviewed,
0/15 browser/player-to-decoder correspondences, 0/3 prediction-link judgments,
0/3 claim-linked scores, and no direct `DeltaF`, uncertainty interval, or
test**. This annotation-proxy study will not change those counts.

## Limitations and next action

The validation cohort was previously exposed to model inference and analyzed
under a different intervention. The new QTA outcome is post hoc relative to
the project as a whole and cannot establish independent replication. Temporal
annotations may be noisy or semantically incomplete, and the timestamp origin
is not browser-calibrated. Attention bias and key ablation may alter the model
in ways beyond evidence use. Even a positive result supports only this
finite-cohort benchmark-annotation proxy under the stated model, sampler,
prompt, and intervention. Rights remain unverified; no material or derived
per-item information may be published.

The original all-annotation-groups policy is now **preflight no-go** for its
complete-cohort run: the corrected full CPU scan found five rows without an
equal-count disjoint control. Do not run a feasible subset or treat the earlier
row-one engineering smoke as evidence. A separately versioned, one-group
anchor proxy is being implemented and evaluated in
`2026-10-18_qta_annotation_anchor_proxy_v1.md`; its outcomes must remain
separate. Continue the separate reviewer handoff preparation in parallel.

## Execution record (2026-10-18)

The first attempted one-row smoke used the documented defaults and stopped in
`source_preflight` before processor construction, video decode, model load, or
any forward: requested/effective rows 1/0, preflight attempted/completed 0/0,
and all six branch counts were attempted/completed 0/0. The protected failure
receipt is `data/private_nextgqa/qta_annotation_proxy_smoke_20261018_0215.json`
(mode `0600`, SHA-256
`3a6a856f94d195bfcf4a9904269f579e179ee5b737c7136e9a33d3953944d780`). It
records `FileNotFoundError` without a score. Independent path inspection found
that the runner incorrectly defaulted to `~/data/private_nextgqa/…`
while the frozen manifest is in this checkout at
`data/private_nextgqa/faithfulness_validation_pilot_20261008.raw.json`. The
runner now resolves its default relative to the checkout; an explicit path
still takes precedence. Its focused tests pass **16/16**, Python compilation
passes, and the focused worktree passes `git diff --check`. The default resolver
was independently checked against the actual local manifest (15,579 bytes).
The narrow fix is awaiting the independent code audit before retrying the
smoke. No GPU/model inference ran in this failed attempt; it contributes no
faithfulness or proxy estimate.

After that path correction, the second smoke also stopped before video decode
or model load. It exposed an overstrict cohort validator requiring all 30
question IDs to be globally distinct. The frozen manifest has 30 distinct
videos but 11 distinct question IDs; NExT-GQA IDs are scoped by video. The
validator now requires 30 distinct videos and 30 distinct `(video_id, qid)`
pairs. A synthetic test allows repeated question IDs across videos and rejects
a repeated video. The suite passes **17/17**, compilation and `git diff
--check` pass; this second correction is pending independent audit. The private
receipt `data/private_nextgqa/qta_annotation_proxy_smoke_20261018_0218.json`
(mode `0600`, SHA-256
`1b2f7662572267170198661ffc1b7f1563a493b376df7d44af35d06efadaab2b`)
records `source_preflight`, `ValueError`, zero preflight rows, zero forwards,
and no estimate. The manifest digest remained the frozen
`968ed130684510de8c6fe7e91a4c0ed90db2cac90a8b2200ff3b55254cd77875`; no
cohort rows or outcomes were changed.

The third smoke passed all fixed-source and one-row CPU preflight checks and
loaded the pinned model. Its baseline-full forward completed, then the
baseline-evidence-ablation forward raised an uncaptured `RuntimeError`;
requested/effective rows remained 1/0 and no score was retained. The private
receipt SHA-256 is
`0d701c8e448565c6c6f321d4c9161b19736735a9d311773e288a78ac13e5f791`.
Peak VRAM and full runtime were not recorded in this failure receipt. To
diagnose the next bounded retry, the runner now stores the first 240 characters
of `RuntimeError` text only in the owner-only private receipt (nothing is
printed or published). Its focused suite passes **17/17**, including a
300-character synthetic truncation-boundary test; Python compilation and
`git diff --check` pass. The independent auditor approved this constrained
diagnostic change and the cohort-identity correction. The next action is a
fourth one-row smoke with a unique output name; do not start 30 rows unless all
six forwards complete and the smoke receipt validates.

The fourth smoke reached the same pre-forward VRAM guard after baseline-full
completed. Its private receipt
`data/private_nextgqa/qta_annotation_proxy_smoke_20261018_0228.json` (mode
`0600`, SHA-256
`8516e8af993549433b98a8d0263f3e00badf76851f9cad5bf6bba78181a7ec3a`)
identifies `per_forward_free_vram_guard_failed`; effective rows stayed 0 and
only 1/6 forwards completed. This is a conservative guard trip, not evidence
that the CUDA forward itself failed. The runner now releases unused CUDA
caching-allocator blocks after each successful CPU-logit return and retains
the same 2-GiB free-VRAM floor before every forward; if it still trips, the
private failure detail will include free, allocated, and reserved byte counts.
The focused suite remains **17/17**, compilation and whitespace checks pass.
This cache/guard diagnostic change is pending independent audit before a new
smoke attempt.

The subsequent complete-cohort attempt stopped at `full_cpu_preflight` after
1 of 30 rows completed and row 2 was attempted. No model was loaded and no GPU
forward ran. Its mode-0600 failure receipt
`data/private_nextgqa/qta_annotation_proxy_30_20261018_0240.json` has SHA-256
`98adc897b1ea3da62b1db629c26afad2ee378011c20942e56da4601891c4f380`; it
records requested 30, effective 0, and zero branch attempts. The receipt's
original schema omitted `ValueError` messages, so the exact fixed-code cause
was unavailable. The runner now admits only lowercase underscore-form
`ValueError` codes (and suppresses general ValueError text that could contain a
question or path); focused tests pass **18/18**, py_compile and
`git diff --check` pass. This diagnostic is pending independent audit before
repeating full preflight. No rows were replaced or dropped.

## Outcome-blind timing-bound correction (2026-10-18)

The failed 30-row CPU preflight exposed a time-domain mismatch before model
construction or any cohort forward. On row 2, `gsub` duration was 50 s while
the last sampled decoder-relative PTS was 50.417033 s; its only annotated
interval ended at 17.5 s. The row failed solely because one mapper parameter
was being used both as the annotation-validity bound and as the final sampled
support-cell boundary. This did not inspect or use a model outcome for row 2.

The shared interval mapper now has an optional separate annotation duration;
omitting it preserves all previous callers' semantics. The proxy runner uses
`max(gsub duration, final sampled PTS)` for support geometry, while `load_fixed_rows`
continues to reject any interval outside its original `gsub` duration. The
source intervals are not modified. Geometry hashes bind both durations, and
each private row record reports the support-minus-annotation duration. The
new CPU tests verify that a 17.5 s interval maps with a 50.417 s support end
while an interval beyond the original 50 s annotation bound is still rejected.
Focused interval/runner tests pass **38/38**; this change is pending independent
code review and a one-row row-2 preflight. The prior successful one-row smoke
included an engineering score for row 1; that value did not inform this rule.
All work remains exploratory and non-confirmatory, and no 30-row score was
produced.

## Complete-cohort CPU preflight result (2026-10-18)

After the outcome-blind support-duration correction passed the focused suite,
the corrected original `all_annotations` preflight was run across the fixed
30-row cohort, without model loading. The cohort-wide call stopped at row 9
after 8 rows completed with `equal_count_temporal_control_unavailable`. A
separate ordinal-only scan isolated all rows: 25/30 completed geometry and
five failed the frozen equal-count disjoint-control requirement at ordinals
9, 15, 16, 20, and 28. No row was dropped or replaced. No checkpoint model was
loaded, no GPU forward ran, all six branch counts remained 0/30, and no score
or effect estimate exists for this attempt. The ordinals are local diagnostic
positions only and are not published.

This result closes the original full-cohort policy before scoring; running only
the 25 geometrically feasible rows would violate its locked all-or-none
denominator. It does not alter the one-row engineering smoke, whose output is
not an efficacy observation. The new single-anchor policy is a different
exploratory estimand and is versioned separately; its implementation and
one-row real-video smoke must pass their own independent gates before any
30-row score.
