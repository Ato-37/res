# Synthetic transient-event QTA GPU smoke (2026-10-19)

## Purpose

Validate the frozen nine-forward-per-clip synthetic QTA scoring path on the
reserved engineering seed before any 40-pair cohort run. This is a synthetic
mechanism/runner check, not a faithfulness or explanation-quality result.

## Setup

- Protocol: `experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md`
- CPU preflight: `data/private_nextgqa/qta_synthetic_event_construct_preflight_20261019_v3.json`
- Runner: `scripts/run_qta_synthetic_event_construct_gpu.py`
- Checkpoint: pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights, BF16 compute,
  eager attention; RTX 5070 Ti, CUDA 12.8, torch 2.11.0+cu128.
- Smoke uses one reserved scene pair, two generated clips, and exactly 18
  forwards. Its seed is not part of the 40-pair cohort.
- Before the successful run, three attempts stopped before any model forward:
  an incorrectly sourced attention aggregation import, then an unimported
  `Counter` used for overlap totals. Each wrote a distinct owner-only failure
  receipt with zero completed forwards and no estimate. Both code issues were
  repaired and covered by focused CPU tests before retrying.
- The final run was individually elevated only to expose WSL GPU access. The
  pre-run check saw 14,750 MiB free GPU memory and about 14.5 GiB host
  `MemAvailable`; smoke pre-load guards also passed.

## Verified result

The final reserved-seed smoke passed: **1/1 pair, 2/2 clips, 18/18 attempted
and completed forwards, 0 failures**. It took 91.985 seconds including model
startup. The owner-only receipt is
`data/private_nextgqa/qta_synthetic_event_construct_gpu_smoke_20261019_v4.json`,
mode `0600`, SHA-256
`33c401083ceb40ae87ca3ad213eabc8f27a70a99d5d67d0b491dd10d1ed641d5`.
It marks `cohort_included=false` and `faithfulness_result=false`.

An independent local verification recomputed and matched all 11 checks:
runner/source hashes, source commit, CPU preflight digest and its source subset,
assignment hash, reserved seed hash, all five checkpoint shard records and
aggregate digest, model-config hash, processor fingerprint, and tokenizer
fingerprint. The preflight digest bound by the smoke is
`be20dc82b715fffc59ab1c2dca17d9070f54ae2defe7116ab5ec75b399231286`.
Peak CUDA allocation was 12,254,939,648 bytes and peak reservation was
14,338,228,224 bytes; the per-forward memory guards passed throughout. GPU
memory returned to approximately 1.25 GiB used after model cleanup.

The separate all-cohort CPU processor preflight passed **40/40 pairs and
80/80 clips** in 12.206 seconds. It checked 40 matched question/choice/render
geometries, two event groups and two disjoint nearest temporal controls for all
80 clips, and exact onset/orientation/answer-position balance. It did not load
weights or run a forward.

## Interpretation and limits

The smoke confirms that the local pinned model can execute the frozen attention
capture, selector, baseline, QTA, sham, and internal key-ablation branches for
the reserved synthetic pair under current resource limits. It provides no
effect estimate: smoke results are excluded from the cohort by design. The
three pre-forward failures also show why a successful CPU fixture alone is not
evidence that the model scoring path works.

Even a successful full synthetic result cannot establish explanation
faithfulness, NExT-GQA behavior, or natural-video evidence dependence. The
mechanism uses synthetic clips, question-token attention, fixed gold-versus-
foil scoring, and key ablation with potential distribution shift. Rights status
is unverified; all receipts remain local and owner-only.

## Analysis readiness and next action

Added `scripts/analyze_qta_synthetic_event_construct.py` to implement the
protocol's equally weighted four-stratum estimate, stratified variance,
Welch–Satterthwaite degrees of freedom, two-sided 95% t interval and test, with
the frozen zero-variance rule. Hash-sham is reported as descriptive means only,
as required by the protocol. The analyzer validates the full denominator,
schema, pinned model/precision, current full source hash set, assignment hash,
checkpoint-shard digest, preflight/smoke digests, and source commit before
analysis. It rejects overflowed moments and writes owner-only output without
overwriting. Tests cover standard and zero-variance estimates, incomplete or
unbound receipts, and inconsistent moments. After independent review found
and prompted those provenance/reporting hardenings, the focused research/UI
suite passed **50/50 tests** and compile checks passed. A final independent
audit of the latest changes is pending.

The smoke receipt must be rerun after the runner was changed to bind its digest
into the full-run receipt. After that new exact smoke and one final live resource
check, the next justified action is the full frozen cohort: **40 pairs, 80
clips, 720 forwards**, expected to take about 52 minutes including setup.
Require the current source-bound CPU preflight and passing smoke receipt. Any
guard failure or incomplete denominator invalidates the cohort and must not
produce an estimate. Direct real-video faithfulness remains unchanged at **0/3** claim reviews, **0/15**
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores; no direct `DeltaF`, interval, or test exists.

## Source-bound smoke v6 and analysis gate repair (2026-10-19)

The full runner initially omitted the mandatory `expected_source_commit` when
validating its smoke receipt. It now captures `git rev-parse HEAD` before smoke
validation and requires the smoke to carry that exact commit. The analyzer now
independently re-runs both strict runner validators against its directly read
preflight and smoke receipts, in addition to binding each raw file digest to the
full receipt. It explicitly checks that the smoke actually loaded weights and
used the GPU. Added tests cover those two flags, stale upstream digests, source
commit mismatch, and invalid preflight binding.

A fresh GPU smoke against the current runner passed: **1/1 pair, 2/2 clips,
18/18 forwards, 0 failures** in 94.116 seconds. Its owner-only receipt is
`data/private_nextgqa/qta_synthetic_event_construct_gpu_smoke_20261019_v6.json`
(mode `0600`, SHA-256
`e622ef1fcffeae82ea0dfe4b801f16791340fe6d7c51290aee6e482f64da3094`). It is
explicitly excluded from the cohort and is not a faithfulness result. The smoke
records peak allocation/reservation of 12,254,939,648 / 14,338,228,224 bytes
(runtime manifest); immediately afterward the GPU had 14,749 MiB free, 0%
utilization, and host `MemAvailable` was about 14.2 GiB. Source hashes,
protocol, preflight digest, assignment manifest, smoke seed, and source commit
recompute correctly. The CPU preflight remains 40/40 pairs and 80/80 clips.

Focused analyzer+runner tests pass **15/15**, py_compile passes, and
`git diff --check` passes. The independent auditor verified the runner commit
binding and smoke v6 before the analyzer's final strict-validator change; a
second read-only audit of that final analyzer change is pending. The ordinary
sandbox cannot see NVML, so GPU availability was verified with an individual
elevated `nvidia-smi` call; this is not evidence of GPU unavailability.

The full run remains a synthetic mechanism check, not real-video evidence or
direct faithfulness. If the independent audit clears the strengthened analyzer
and a fresh immediate resource guard passes, proceed with the frozen full
cohort (40 pairs, 80 clips, 720 forwards; estimated about 52 minutes). Keep all
receipts local and owner-only. A successful synthetic estimate must not change
the direct faithfulness counts above.

## Frozen synthetic construct full-run result (2026-10-19)

The independently reviewed, source-bound run completed on the pinned local
Qwen2.5-VL GPU path: **40/40 scene pairs, 80/80 clips, 720/720 forwards, 0
failures**, in 2,633.197 seconds (43m53s). The full receipt is
`data/private_nextgqa/qta_synthetic_event_construct_gpu_full_20261019_v1.json`
(mode `0600`, SHA-256
`375a255e7be2f6b94f35a475df26b8066a78eb9a3cf31e00a8cdc5aeb3f13bb7`); the
analysis is
`data/private_nextgqa/qta_synthetic_event_construct_gpu_full_analysis_20261019_v1.json`
(mode `0600`, SHA-256
`b27e712e13a5a05739cc7e3d83e5a61ce33540e8d1c3b5ddc1c90a264b25c823`).
The workspace was dirty at run time; the receipt is bound to source commit
`407d7f4c8a4dfa016cdbb0edd2e31b7eac32856b` and its recorded source hashes.

The frozen primary endpoint is the equally weighted mean across four event-onset
strata of scene-pair `DeltaF_QTA`. It was **−0.01719 nat** (SE 0.01814; approximate
stratified Welch–Satterthwaite df 31.68; 95% t CI [−0.05415, +0.01978]; two-sided
one-sample t test p=0.35055). The interval crosses zero; this run supplies no
evidence of positive synthetic construct effect. The protocol reports the sham
mean (−0.02031 nat) descriptively only, without inferential testing. An
independent rerun of the analyzer against the full, preflight, and smoke receipts
reproduced the estimate, CI, and p-value exactly; its output is the separate
owner-only `..._recheck_v1.json`. The frozen runner, preflight, and analyzer test
suites passed **24/24** in this checkpoint.

This is still synthetic construct validation only. It did not evaluate a
human-reviewed explanation or natural video and cannot establish faithfulness or
natural-video evidence dependence. Key ablation may shift the input distribution,
and question-token attention is only the tested selector. Rights remain
unverified; receipts and derived analysis stay local. Formal direct-faithfulness
progress remains **0/3** claim reviews, **0/15** clock/frame correspondences,
**0/3** prediction-link reviews, and **0/3** direct scores; there is no direct
`DeltaF`, interval, or test.

Next: do not resume the closed head search or repeat this synthetic cohort.
Continue the separately scoped actual-media source/annotation gate and
measurement work; only run the prespecified real-video smoke once its own
cohort/source prerequisites are met. A synthetic result does not waive those
requirements.
