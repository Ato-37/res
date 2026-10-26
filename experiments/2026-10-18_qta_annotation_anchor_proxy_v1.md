# QTA single-anchor annotation proxy v1 (2026-10-18)

## Status and rationale

This is a separately versioned, exploratory finite-cohort benchmark-annotation
proxy. It is not a repair or successful execution of the all-annotated-groups
proxy in `2026-10-18_qta_validation_annotation_proxy.md`, and it is not direct
explanation faithfulness. That frozen design's all-or-none CPU preflight found
that equal-count, disjoint controls cannot be formed for five of the 30 fixed
rows. No model was loaded and no full-cohort score was produced.

The new estimand is the QTA-versus-baseline change in fixed-prediction margin
dependence on **one deterministic representative group covered by the
benchmark's temporal annotations**, compared with one disjoint non-annotation
group. It deliberately gives up full-span coverage. It is post hoc in the
project history and uses a reused, previously exposed validation cohort, so
all results are finite-cohort descriptive and exploratory. Keep them separate
from the failed all-groups protocol and all direct-claim faithfulness counts.

## Frozen representative and control selection

1. Validate every original annotation interval against its unchanged `gsub`
   duration. Use the existing outcome-blind sampled-frame midpoint-cell
   geometry and its separate support end `max(gsub_duration, final sampled
   relative PTS)`.
2. Merge overlapping or touching annotation intervals for overlap arithmetic.
   Keep the source intervals unchanged. Map all intervals through both actual
   sampled PTS and the processor's scalar-FPS implied grid and require identical
   group membership for each interval.
3. Among covered Qwen temporal groups, select the group with the greatest exact
   duration of overlap between its support cell and the merged annotation
   union. Resolve equal overlap by the earliest temporal group index. The
   selector receives no attention, logits, answer, or model outcome.
4. Select one control group from all other temporal groups by minimizing
   absolute temporal-group-index distance from the representative group; ties
   go to the earlier group. Require at least two temporal groups and exactly
   one evidence and one disjoint control group for every row.
5. Preserve the same fixed cohort of 30, six forwards per row, fixed baseline
   full-input `p/q`, QTA selector/treatment, local-only outputs, and all-or-none
   30-row denominator. Any row failure means effective `n=0` and no cohort
   summary. Do not replace rows or run a post-selected feasible subset.

The score formula remains the branch interaction defined in the all-groups
proxy, but now its ablation operands each contain exactly one temporal group.
It therefore estimates dependence on the chosen anchor group under this exact
masking operation, not dependence on all annotated evidence.

## Required reporting and limits

Report requested/effective rows, failures, branch counts, finite-cohort
descriptive statistics, and the selected group's overlap fraction and temporal
position summaries. Do not report a population confidence interval, p-value,
causal claim, or direct faithfulness conclusion. Interpretations are limited by
annotation quality, uncalibrated browser/player-to-decoder clock origin,
one-model/one-sampler specificity, key-ablation distribution shift, the
representative group's possible location/length bias, and historical cohort
reuse. Rights remain unverified; keep all inputs and derived outputs local.

## Implementation and execution gate

The implementation must expose this policy explicitly while preserving the
old all-groups policy as its default. Add deterministic unit tests for interval
union, exact-overlap selection/ties, nearest-control/ties, boundary groups,
old-design infeasibility with new-design feasibility, invalid intervals,
and all-or-none denominator enforcement. Independent code/protocol audit and
a one-row real-video six-forward smoke are required before the 30-row run. The
full run is authorized only if the smoke, all-30 CPU preflight, runtime VRAM
guard, pinned checkpoint audit, and privacy-safe local-output checks pass.

The explicit `single_overlap_anchor` policy is implemented in the shared
interval mapper and real-video runner; the legacy `all_annotations` remains
the default. Its geometry fingerprint binds canonical interval endpoints,
annotation/support durations, all annotation-covered groups, selected and
control groups, policy, and exact anchor-overlap diagnostics. Success and
failure receipts both bind the chosen policy. Unit tests cover interval union,
ties, support edges, complete evidence-set validation, one-control feasibility
when the all-groups design fails, unchanged default behavior, deterministic
mapper output, and rejection of unknown policies.

Before the new real-video smoke: focused interval/runner tests passed **46/46**;
the full `scripts` unittest suite passed **1,713/1,713**; `py_compile`, CLI
`--help`, and `git diff --check` passed. An independent source audit accepted
the new selection logic and policy binding. The independent audit's fingerprint
diagnostic-linkage suggestion was implemented by hashing exact interval
endpoints and overlap fields. The full-suite run also exposed stale test
fixtures after the independently added `capture_operator` role and a missing
explicit test classification for this local-only runner; both were updated,
and the full suite then passed. No model was loaded or GPU run as part of these
checks.

Implementation hashes before the smoke:

- interval mapper: `9d4e65443db1a0462aa742a8cdf7fd0b534bfb4d90bfbe75148bd85055701b56`
- runner: `9a3a48bcba203355aa3c05b48780dccba55658a4de2c2bae7bb247de8ad57843`
- interval mapper tests: `ae2e6dd60e9f8167c54ec9d13f47702a61f7b06aee94ebd522d38ca9104e41c4`
- runner tests: `96a63770e024ff9381cc33c6c2aa8f32eb962aaf8b32ef1e36d893b9e4d0ca6e`

The next run is the first real-video smoke under this new policy, using one
fixed cohort row, six requested forwards, and no raw media/question/logit
output. The previous successful real-video smoke under the old policy took
64.415 s and peaked at 11.093 GiB allocated VRAM; that is a planning reference,
not a runtime guarantee. If the smoke and all-30 CPU preflight pass, the
30-row/180-forward run is expected to require roughly 10–30 minutes including
model load, decode, and bookkeeping; actual elapsed time will be recorded.

## Final-schema real-video smoke (2026-10-18)

The first anchor-policy smoke completed 1/1 fixed rows and all 6/6 requested
branches in 56.596 seconds. That receipt had the correct policy and branch
counts, but omitted the exact annotation/evidence/control group-index arrays
needed for direct local inspection. The runner was extended to retain those
private geometry arrays, without changing selection, scoring, cohort, or
forward behavior; focused tests remained 46/46 and `py_compile` plus
`git diff --check` passed.

The final-schema smoke was then rerun under the same policy and fixed first
cohort row. Receipt:
`data/private_nextgqa/qta_annotation_anchor_smoke_20261018_1133.json`, mode
`0600`, 9,654 bytes, SHA-256
`c0450fe2ca0ef28a7ac57ccee6753ca978ece52f6c38ec59c825669b7e9cdf2e`.
Observed: status `engineering_smoke_only`; requested/effective rows 1/1;
preflight 1/1; 6 requested, attempted, and completed forwards; failures 0;
elapsed 62.980 s; RTX 5070 Ti, CUDA 12.8, pinned model revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`; free VRAM before load 14.657 GiB,
peak allocated 11.093 GiB. The private record has 16 total temporal groups,
annotation-covered groups `[0, 1]`, selected evidence group `[0]`, and disjoint
uncovered control `[2]`; selected overlap is `87087/20000` seconds, or
`2233/4000` of the merged annotation union. This confirms the smoke used one
covered anchor and one group outside the full annotation union.

The one-row `DeltaF` is `-0.2499999553` nat and baseline-full A–E accuracy is
1/1. These are engineering-smoke outputs only, with no uncertainty estimate,
not an efficacy result and not faithfulness evidence. The single row and its
outcome were not used to choose or revise the frozen selection rule. Rights
remain unverified and the receipt remains local-only.

At the final-schema smoke, the runner SHA-256 was
`cfbffce9224a4eb2917c41b3b38dc78efa701cebae4ee08a863a4bfbe454b93d`; the
interval-mapper SHA was
`9d4e65443db1a0462aa742a8cdf7fd0b534bfb4d90bfbe75148bd85055701b56`, and the
anchor-protocol SHA bound by the run was
`ec6a4c423fbc6251302d9432c2d00ed0991df683980925b4c427a773e4cd5e9d`.

## Full exploratory finite-cohort run (2026-10-18)

The fixed all-or-none preflight passed 30/30 rows. The pinned Qwen2.5-VL-7B
checkpoint byte audit passed all 5/5 shards, then the single-anchor policy ran
the six frozen branches on all 30 rows. The command was:

```text
.venv/bin/python scripts/run_nextgqa_validation_qta_annotation_proxy.py --n 30 --output-name qta_annotation_anchor_30_20261018_1133.json --evidence-selection-policy single_overlap_anchor --user-authorized-local-only
```

Private receipt:
`data/private_nextgqa/qta_annotation_anchor_30_20261018_1133.json`, mode
`0600`, 56,649 bytes, SHA-256
`286a10899754ae9cb7a49e986d6c822f7268a313621c25e96293ba7c48f5be9a`.
It records local-only execution and unverified rights. Requested/effective rows
were 30/30, failures 0, CPU preflight 30/30, and forwards
requested/attempted/completed 180/180/180 (30/30 in every branch). Elapsed
runtime was 892.942 s. Hardware was RTX 5070 Ti, CUDA 12.8, 4-bit Qwen with
BF16 compute at pinned revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
peak allocated VRAM was 11.714 GiB. The run used seed 20261018 and 32 frames.
All 14 run-bound source-file hashes, the fixed cohort and dataset hashes, and
the checkpoint audit were independently checked against the receipt. The
separate protocol auditor verified all 30 geometry records and recomputed the
stored aggregate from the private rows without disclosing per-row values.

Observed finite-cohort descriptive results: mean ΔF `+0.0083333` nat, sample
SD `0.1428366`, range `[-0.2500001, +0.2500000]` nat; 17 positive, 13 negative,
0 zero. Baseline full-input restricted-choice accuracy was 23/30 (76.7%). The
selected anchor's annotation-overlap fraction averaged 0.5181 (SD 0.2754,
range 0.0808–1.0); its normalized temporal position averaged 0.58 (SD 0.3436,
range 0–1), and the full annotation union covered an average of 4.23 groups
(SD 3.93, range 1–14). A local aggregate analyzer independently recomputed
these statistics from the private receipt and verified the source hashes. The
analyzer also verifies the all-or-none denominator, branch counts, selected
and control membership, fixed answer/foil margin keys, per-row ΔF recomputation
from six stored margins, and stored aggregate agreement. Focused
analyzer/mapper/runner tests passed 51/51.

Interpretation: the mean is close to zero and individual directions are mixed.
This run does not provide evidence of a robust improvement from the internal
intervention. No p-value, confidence interval, or population inference is
reported. These are descriptive measurements on a reused, historically exposed
validation cohort under one model/sampler and a single representative annotation
group; they do not establish direct generated-claim faithfulness, causal
efficacy, or performance on a fresh cohort. Attention-key ablation may shift
inputs out of distribution, and the anchor selector can favor interval length
and temporal location. The result and all underlying records remain private;
rights status is unverified.

Reproduction of the aggregate-only audit (prints no individual record values):

```text
.venv/bin/python scripts/analyze_nextgqa_qta_annotation_proxy.py data/private_nextgqa/qta_annotation_anchor_30_20261018_1133.json
```

Next action: do not tune this proxy against the observed outcomes. Advance the
human-review handoff by obtaining the pending role/key assignments and
independent trust-root pin. In parallel, use a genuinely fresh, predeclared
cohort and a direct claim-linked outcome protocol before treating any follow-up
as confirmatory. Until the human review and prediction-link gates pass, report
direct claim-faithfulness coverage as zero.

Analyzer source hashes after verification:

- `scripts/analyze_nextgqa_qta_annotation_proxy.py`:
  `1854d7e41638daebde2ac5ed94ec1361e39d01f896aad47c0597d4de347484bd`
- `scripts/test_analyze_nextgqa_qta_annotation_proxy.py`:
  `11f228476868928ed224e1723618091b11726df94d9d068c2846249557bf8349`

## Initial feasibility observation (2026-10-18)

After the timing-bound correction passed 38 focused unit tests, the fixed
30-row CPU preflight stopped at row 9 with
`equal_count_temporal_control_unavailable` after 8 rows completed. A separate
per-row geometry scan found the same control-unavailable code at row ordinals
9, 15, 16, 20, and 28; the other 25 rows completed processor/video geometry.
These are geometry-only observations, not outcomes. They were obtained before
model loading or cohort scoring. No rows are excluded from v1.

Resolution: the fixed all-groups design remained closed at 25/30 feasible.
The separately versioned `single_overlap_anchor` protocol completed the
all-30 preflight and full run documented above; it does not rehabilitate the
all-groups estimand. Do not rerun either branch on this cohort. The next
scientific work is an independent fresh-cohort design after human review and
prediction-link protocol gates are satisfied.
