# Benchmark-interval GPU pilot on generated-claim cohort (2026-10-21)

## Purpose and preregistered interpretation

The separate same-model localization pilot on these same three development
items produced three localization generations, but failed interval validation
on item 3 before any of the nine scoring forwards. It produced no score. This
follow-up therefore uses the dataset's supplied QA gold intervals directly,
without attempting to rescue or repeat the failed machine localization. The
human-review collection was waived by the user; no review is recorded as having
occurred.

The fixed estimand is still the three-row descriptive mean of
`gold-vs-best-foil margin under equal-count black control mask` minus that under
`black mask of benchmark QA gold interval(s)`, in nats. Positive values mean
the benchmark interval mask reduced the answer margin more. This is an
auxiliary QA-evidence-dependence measure on the exact three validation items
that received generated explanations. It does not test whether the generated
claim is true, whether its text is supported by these intervals, or whether
model explanations are faithful. No inferential test or population interval
will be reported for `n=3`.

## Fixed inputs and workload

Use the existing two local claim-generation packets, de-duplicate by source
video SHA-256, require exactly three unique videos and unset review fields,
decode 32 frames per video, and use the NExT-GQA gold intervals associated
with those exact QA records. The runner constructs full, black gold-interval,
and equal-count temporal-control inputs and calls the pinned Qwen2.5-VL-7B
scorer nine times total. Expected GPU work: one offline 4-bit BF16 model load
and nine one-row forwards, approximately 1–3 minutes after load. Resource
minimums are 12 GiB free VRAM and 8 GiB available WSL memory.

## Limits and faithfulness progress

The three-item validation cohort is reused and is not confirmatory. This
measurement is benchmark-interval prediction dependence only; it must remain
`補助診断のみ`. Formal generated-claim faithfulness remains **0/3** scored,
with no direct `DeltaF`, CI, or test. The semantic claim/frame mapping and
Stage-2 review are not established. User rights status remains unverified and
all raw media, QA text, and detailed receipts remain local.

## First execution attempt and implementation repair

Attempt `data/private_nextgqa/benchmark_interval_claim_cohort_gpu_20261021_0345_01.json`
(SHA-256 `7d8bdabba4099615883a370cf5118b8a603d5d289d4872b64f64a329348f4ae3`,
mode 0600) loaded the pinned model and completed one of nine scoring forwards.
At the next root entry the live resource callback rejected the call: free VRAM
had fallen to **1.523 GiB**, below the runner's 2-GiB per-forward floor.
The root-boundary counters were 2 entries / 1 verified / 1 completed; this
shows the second model body did not run. Peak allocated VRAM was 11.040 GiB,
elapsed time 30.19 s, and no aggregate `D3` was produced. The private receipt
contained one partial row; it is excluded from every estimate and will not be
used as a surviving subset.

Source inspection found the scoring core kept the previous condition's model
output/logits and transferred input batch live while entering the next
condition. The core now drops those GPU references and empties unused CUDA
cache after extracting each CPU score. The retry also checks free VRAM before
transferring each next batch and again at model root entry. These changes affect
resource lifetime and guards only; they do not change inputs, interval
membership, score, metric, or cohort. The runner propagates a safe failure
code and discards partial rows from its result summary.

## Verified retry result

After **43/43 CPU regression tests**, the same fixed nine-call experiment was
retried once as
`data/private_nextgqa/benchmark_interval_claim_cohort_gpu_20261021_0412_02.json`
(mode 0600, 7014 bytes; private artifact SHA-256
`4e8736558c639204fd7a7c34d36343db2dab6ab24c503e8313493492d9a6a0cf`). It
completed **9/9 forwards and 3/3 triplets**, with no failure code. This was
one bounded retry of the same cohort, metric, interval source, and frame count;
no threshold or sample selection changed.

Observed row-level `control margin - benchmark interval-mask margin` values
were **+0.875**, **0.000**, and **−1.625 nat**. Their independently recomputed
arithmetic mean is **−0.250000 nat**, matching the receipt. The sign is opposite
the descriptive direction: on these three items, temporal-control masking
reduced the answer margin more on average than masking the benchmark QA
interval. Row 2 ties; row 3 dominates the negative mean. No uncertainty
interval or p-value is reported for this fixed `n=3` descriptive cohort.

An additional arithmetic decomposition rejoined each private row's gold label
locally, without emitting labels. Mean `gold log-probability (control -
interval-mask)` was **−0.003630 nat**; mean `best-foil log-probability (control
- interval-mask)` was **+0.246370 nat**. Their difference is the same −0.250000
nat margin contrast. Thus the negative mean here comes almost entirely from a
larger best-foil log-probability under the temporal control, not a systematic
drop in the gold-choice log-probability under the interval mask. This is a
description of these three rows, not a mechanistic explanation or population
effect.

The retry used RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128, Qwen2.5-VL-7B
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights and BF16
compute, seed 20261021, and 32 frames/video. Free VRAM before model load was
14.657 GiB; minimum free VRAM was **8.669 GiB before batch transfer** and
**8.595 GiB at model-call entry**. The run took 55.5 seconds. The saved receipt
matches both private input-packet hashes and records 9/9 root calls verified
and completed. Its output is mode 0600 and remains local.

Interpretation: this is evidence only about dependence of the fixed QA answer
margin on black-masking dataset-provided QA intervals versus equal-count
temporal controls in these three reused validation records. It does not test
generated-claim support, explanation faithfulness, causal mechanism, or a
population effect. The sign does not support the claim-evidence-specificity
hypothesis for this cohort. Human-validated faithfulness remains **0/3**; the
user waived review collection, but no human review was performed. Rights status
remains unverified. Keep all data-derived results local.

The next useful action is to audit whether the generated claim text is
structurally tied to the benchmark QA record and whether a non-human, fixed
claim-to-interval construction can be validated without using this same model
twice. Any such endpoint must remain separately labeled and chosen before
observing further scores. Keep the failed same-model localization and this
benchmark-interval endpoint distinct. Do not repeat or expand this three-row
cohort based on its outcomes.

## Cohort30 follow-up (2026-10-21)

As an exploratory replication of the same auxiliary QA-margin endpoint, a
separately frozen cohort30 completed after a three-condition smoke. The smoke
completed 3/3 forwards on its first row; the full run completed 9/9 forwards
for 3/3 rows with no failure. The full owner-only receipt is
`data/private_nextgqa/claim_likelihood_benchmark_interval_cohort30.json`,
mode 0600, SHA-256
`3a94442d…78093c6`. Independent audit verified all four input hashes and
eleven source/helper hashes against the local inputs, all row bindings and
forward counters, and the arithmetic. Recomputed `control − benchmark
QA-interval-mask` margins were approximately −0.00000013, −0.6250, and
−0.7500 nat (mean **−0.4583 nat**, median −0.6250; 0 positive, 3 negative,
one numerically near zero). The independent audit also confirmed smoke/full
agreement on their shared first row.

Interpretation remains strictly auxiliary: this cohort shows no positive
finite-cohort direction for QA-margin sensitivity to benchmark intervals,
but it does not measure predicted interval accuracy, generated-claim support,
or explanation faithfulness. Human review is still unperformed; its waiver
only removed review collection as a prerequisite for exploratory local runs.
Rights remain unverified and the data and receipt remain local. This result
does not justify repeating or expanding the endpoint. The next direct
engineering step is a frozen, question-conditioned temporal-key selector
applied during interval generation, with baseline and treated intervals
scored against QA time labels; that would be an exploratory QA-localization
proxy, not a claim-support result. Claim-level explanation validity still
needs claim-specific support labels.
