# PDCA restart checkpoint: batch-compatibility diagnosis and claim-mask protocol v2

## Purpose and window

This resumed work window began at 2026-10-14 13:32:51 JST and is scheduled
through 14:32:51 JST. The prior 9-hour request had already passed its deadline
without completion; this entry makes no claim about work in that gap. The current
work has two bounded engineering aims: diagnose the 32-frame batch-shape output
mismatch, and align the next claim-linked protocol with the user's selected
estimand/cohort while keeping data gates closed.

## Batch-shape diagnostic

The 32-frame, 16-cached-decode B1/B2 comparison stopped on a greedy-output
mismatch. A single controlled replay added only scalar top-1 margin, top-2 set
overlap, and first divergence step. The output is
`results/qwen_video_batch_throughput_gpu_smoke_20261014_b1_b2_32f_16decode_margin_01.json`
(SHA-256 `95863d18d5b26a6b6ce71d2908be0338c0cdb046c1db01dbd720b39b56f36be8`,
29,940 bytes, mode 0600); source commit `2a45fab50cb5bc2f24b64d3637f988070e08603f`,
runner SHA-256 `da8378a4b81c9630fe6d76559b888534262c7130ef5cdf990eff798d954c91d9`.

Independent artifact audit verified 38 calls (34 measured plus 4 warmup), fixed
32-frame token/grid geometry, duplicate-item correspondence, cache positions,
step-7 first divergence, and aggregate-only privacy. At the first mismatch,
contexts were still equal, the B1 top-1 margin was 0.125 logit units, B2's was
0.000 (a tie), and top-2 sets matched. Duplicate B2 items remained identical.
The mismatch recurred on the same clip; the single-clip result is consistent
with sensitivity to near-tie argmax under batch-shape numerical variation, but
cannot estimate prevalence. B1 took 2.4288 s; B2 took 4.0304 s for two duplicate
copies. Because output compatibility failed, the apparent 1.205 ratio is not a
valid speedup. Sparse GPU-wide utilization/memory-busy samples do not measure
sustained use or bandwidth. Stop the 32-frame B2 throughput branch pending a
pre-outcome protocol revision; do not proceed to 128 decode steps on this path.

## Claim-mask protocol design update

The user selected the existing-plan alignment: fixed answer log-margin,
claim-linked evidence-frame mask, all eligible videos primary, and
baseline-correct videos secondary. Design review found the old draft used a
baseline-correct primary cohort, called a frame intervention a token mask, and
left the log-margin target ambiguous. The dated design and contract now specify
one five-choice QA per eligible video; fixed gold label and fixed five choices;
per-condition gold next-token log-probability minus the highest of the four
fixed foil-label log-probabilities; and
`d=(S(full)-S(selected))-(S(full)-S(control))`. The claim's independently
reviewed temporal interval maps to sampled evidence frames. The control selects
equal-count non-evidence frames by minimum total temporal-rank distance, with a
lexicographic tie rule; insufficient controls are pre-scoring ineligible with
no fallback. Baseline correctness is unmasked greedy==gold and is secondary.

Schema v2 now records the eligible denominator and pre-run exclusions, binds
answer-candidate and gold-choice manifests with distinct hash roles, records
the review timing and rights status, and hard-codes the selected
estimand/population labels. Candidate count reconciles with eligible plus
pre-scoring exclusions. The timestamp mapping now states inclusive
`frame_index / decoded_fps` interval membership, and the control cost is the
one-to-one sorted-rank sum used by the existing temporal helper. The structural
validator only accepts the exploratory stage because SESOI and
secondary-family multiplicity remain unfixed; it always returns
`execution_authorization: not_granted`. It does not authenticate review
signatures, rights, hash targets, mapping semantics, per-item mask cardinality,
or runtime behavior. The arithmetic helper now permits distinct per-video
target hashes and reports only a unique-hash count, not target hashes.

Verification: 51 CPU tests passed across the claim-mask protocol, claim-selective
contrast arithmetic, temporal control utilities, and the batch runner. All
changed Python files byte-compiled and `git diff --check` passed. Independent
audit of the protocol found no blocker for an exploratory structural contract;
the control semantics were compared directly with the existing helper. No
private candidate data or GPU inference was used for this protocol work.

The next bounded GPU diagnostic changes the batch question from throughput to
same-context numerical behavior: let B=1 generate its greedy prefix, then force
both B=2 rows to consume those tokens while retaining only scalar summaries of
the raw per-step logits. The purpose is to see whether argmax differences
persist under a common prefix beyond the first shared-context divergence. The
planned 32-frame/16-step run makes 34 measured plus 4 warmup forwards (38
total), with no speed comparison. CPU and independent code audits must pass;
fresh GPU/WSL preflight and an unused output path are required first.

## Faithfulness status and gates

This window adds 0 direct faithfulness samples and 0 new faithfulness effect
estimates. The closed head confirmation remains 79/79, −0.0756 nat (95% t CI
[−0.1578,+0.0065]), below the old +0.25 nat rule; head/layer/alpha search stays
closed. NOAA human review remains incomplete and NExT-GQA item-level rights
remain unresolved. Claim-mask efficacy execution is therefore still `no_go`.
The batch test is an engine-compatibility diagnostic only and is not evidence of
explanation faithfulness.

## Next actions

1. Complete the independent code/protocol audit and repair any blocker.
2. Publish only privacy-safe derived artifacts and this experiment record with
   scoped protocol/code changes, then fetch and verify remote hashes.
3. Continue CPU-only estimand and statistical-gate design. Do not call the
   protocol confirmatory or run gated real data until SESOI/multiplicity,
   human-review attestation, and item-level rights are resolved.
4. The same-context diagnostic is engineering-only; do not add artificial load,
   claim throughput improvement, or count it as faithfulness progress.
