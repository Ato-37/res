# Ten-video test-split gold-interval margin pilot (2026-10-21)

## Purpose

The user waived human-review collection as a prerequisite for local research.
The waiver was applied to this exploratory measurement; it does not make any
human judgment complete. A preceding three-video test-split pilot had a mixed
black-mask contrast (+0.5833 nat), while an earlier validation cohort had
−0.2500 nat. This fixed follow-up asks whether the gold-interval QA-margin
contrast is stable on a separate larger, outcome-blind cohort. It does not
measure generated-claim support or claim faithfulness.

Primary contrast:
`black equal-count control gold-vs-best-foil margin − black QA-gold-interval
mask gold-vs-best-foil margin` (nats). Blur-mask contrasts are the specified
sensitivity condition. No cohort, metric, threshold, or subgroup was selected
after examining scores.

## Outcome-blind selection and failed preflight

A private path-only history manifest combined 17 original sample manifests,
10 claim-generation packets, and the 56 media files in all previously built
local candidate pools (1,447 path entries / 457 normalized video keys). Its
SHA-256 is `d6b4e1d1fda9f41b6eade844e0ca533495b9b342214f1253958c99e0ce7869d9`,
mode 0600. The pool builder records the hash-of-single-manifest-hash-set
`ccc42f7e850291e76a2b81346c50669cd2e3b29ff78153572e389e0b5c8198d5`, which
was independently recomputed from the manifest SHA according to the builder's
source implementation.

The first new pool attempt contained 10 deterministic hash-ranked test videos,
but only 9 met the fixed decode/timing/control eligibility criteria. Its
preflight stopped before model load and produced **0 forwards**. No score was
observed. To preserve the pre-score eligibility rule, that entire pool was
added to the path history and excluded. A second outcome-blind pool then
ranked 20 new videos; 111 eligible QA pairs were retained across those 20
videos. Across the complete test split, 2,741 rows were excluded by known
sample paths, 594 by prior-media HMAC matches, and 14 had invalid intervals;
2,204 eligible pairs across 456 media remained before the 20-video cap.

The second pool's selected-pair allowlist is bound by SHA-256
`335716b9ade96f82e1959dffeb9c7732743e15fc22d4bf3bddc8aec8f3beaa38`; its
summary is bound by `c84936f98dc3894b0b03556146e99756b1ae762747840c7078bb1846fff3b2f1`.
An independent auditor verified that the 10 locked videos are distinct,
disjoint from earlier candidate-pool media and the prior-media exclusion set,
and each exact media/QID assignment joins to the selected allowlist. The
manifest, split files, three HMAC indexes, source commit, and runner/source
hashes match the recorded bindings.

### Later lineage correction

A follow-up audit found that the above disjointness statement was too broad.
All 20 media in the second pool are present in the older local
`zip_gold_candidates_20261017_01` 490-candidate inventory, and the 10-video
lock06 overlaps 10/20 media in that pool. The prior inventory was documented
as availability/candidate reconciliation rather than scoring, but its full
exposure history is not authenticated. Thus this n=10 study's exact lock and
results remain verified, while **independence from all earlier candidate-pool
enumerations is not established**. Its estimates are deterministic finite-
cohort descriptions and should not be described as an independent sample from
the full earlier candidate universe. A later claim-conditioned pilot initially
reused 3/10 exact media+QID rows; it was excluded and not scored. The replacement
claim-conditioned cohort explicitly excludes all 10 lock06 rows. See
`2026-10-21_claim_conditioned_temporal_internal_intervention.md` for the
provenance correction and replacement cohort. This correction changes the
selection-history limitation, not the recorded 10-video arithmetic.

Outcome-blind preflight passed for 10/10 videos: 32 frames each, 10 timing
passes, no timing rejection, and no missing matched control. It scanned the
first 10 of 20 candidates and left 10 unscanned. The five fixed conditions
were full video; black QA-interval mask; black equal-count temporal-control
mask; blur QA-interval mask; and blur equal-count temporal-control mask.
Preflight SHA-256 is
`537cc54ae239f1631b29ab18f75387b6483b8b54ffdb19bda8a2cbfdc11de4ef`; lock
SHA-256 is `e4159231b08544abd201c6ef665817afb6d0f4e0ef0aa22d2770cc345f7541a6`.

## GPU execution and results

The fixed cohort completed **50/50 forwards and 10/10 full condition sets**,
with zero failures and final CUDA synchronization passed. It used
Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, RTX 5070 Ti, 4-bit weights,
BF16 compute, eager attention, 32 frames per video, and seed 20261015. Free
VRAM before model load was 14.657 GiB, available host RAM 12.974 GiB, peak
allocated VRAM 12.185 GiB, and elapsed time 232.21 seconds.

| Fixed row | Black control − QA interval | Blur control − QA interval | Full-video answer correct |
|---|---:|---:|---|
| 0 | +0.5000 | −0.1250 | no |
| 1 | 0.0000 | 0.0000 | yes |
| 2 | −3.2500 | −0.7500 | yes |
| 3 | −0.6250 | −0.2500 | yes |
| 4 | +0.7500 | +1.8750 | yes |
| 5 | −0.1250 | −0.1250 | yes |
| 6 | −0.7500 | −1.2500 | yes |
| 7 | 0.0000 | −0.3750 | yes |
| 8 | +1.2500 | +0.8750 | yes |
| 9 | −0.3750 | 0.0000 | no |
| **Mean** | **−0.2625** | **−0.0125** | **8/10** |

Primary black-mask sample SD is 1.2183 nats, range [−3.2500,+1.2500]; three
rows are positive and seven negative (two are numerically near zero). Blur
contrast sample SD is 0.8589, range [−1.2500,+1.8750]; its mean is near zero.
Black-minus-blur gold-interval-mask margin averages −0.1500 nat (sample SD
0.4441; range [−1.1250,+0.6250]). No inferential test or population
confidence interval was calculated: this is a deterministically selected,
non-probability cohort, and the receipt reports `ci=null`, `p_value=null`.

The owner-only result is
`data/private_nextgqa/gold_interval_cohort_20261021_06_result.json`, mode
0600, SHA-256
`7f8026adc2b40ccbd133360ffcb3a0fd0ca45cf0e13495de58d99823a6c394f4`.
Independent audit recomputed all 10 black and blur contrasts, the black/blur
condition difference, all three summary statistics, 8/10 accuracy, and the
result/preflight/lock/input/source bindings. `test_run_nextgqa_gold_interval_margin_pilot`
passed 21/21 earlier in this turn.

## Interpretation and faithfulness status

The ten-video black-mask mean (−0.2625) is close to the earlier validation
pilot mean (−0.2500), while the separate three-video test pilot was +0.5833.
The within-cohort signs and magnitudes vary substantially. The blur sensitivity
mean is almost zero in this ten-video cohort. Together these observations do
not support a stable, robust answer-margin advantage for masking benchmark QA
intervals. Black masking can introduce an out-of-distribution artifact; blur
also alters the visual input. The primary model answered correctly on 8/10
full videos, but the measurement still compares the dataset gold answer against
the strongest foil, not the model's generated claim.

This establishes only finite-cohort QA answer-margin sensitivity under these
masking interventions. It does not show that an explanation is true, that its
claims map to the dataset QA intervals, or that explanation faithfulness
improved. Relative video PTS was not independently calibrated to the annotator
playback clock. Known-history indexes remain incomplete and unauthenticated;
dataset/model rights remain unverified. Protected inputs and detailed receipts
remain local and unpublished.

Actual current-claim human review remains **0/3**, verified clock/frame pairs
**0/15**, Stage-2 prediction links **0/3**, and direct generated-claim
faithfulness scores **0/3**. The user's waiver removes review collection as a
blocker for these local exploratory measurements; it does not alter these
observed counts. Treat the benchmark gold-interval endpoint as closed for
further cohort expansion. The next research step should implement a distinct,
claim-conditioned prediction-dependence experiment with its model-derived
evidence anchors explicitly labeled unreviewed and exploratory. Do not reuse
the failed timestamp-localization branch, which is already closed.
