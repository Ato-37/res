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
schema v3 validator only accepts the exploratory stage because SESOI remains unfixed. It now checks the single-primary/Holm declarations and rejects malformed SESOI status values.
Secondary multiplicity now follows the existing plan: Holm within predeclared
baseline-correctness and question-type families, with one primary contrast. It
always returns
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

The follow-up GPU diagnostic held autoregressive context constant: B=1 generated
its 17-token prefix, then both B=2 rows were forced to consume exactly that
prefix; scalar summaries of raw per-step logits were compared. The run completed
on one fixed 32-frame NOAA clip, with 34 measured and 4 warmup model calls (38
total), 16 cached decode steps per measured condition, and no throughput claim.
Result: `results/qwen_video_batch_throughput_gpu_smoke_20261014_b1_b2_32f_16decode_sharedctx_01.json`
(SHA-256 `9a1625ced7a19cac80596c4358a28d4c9503cae883fa8bc5c7ee6d17d6a5f9fa`, 29,743 bytes, mode 0600).
Source commit was `ea0121d1bf6fb0bea1d8219c38a2c2e043109962`; the recorded
runner SHA-256 `feb55724609ff9f7d4cd1aa306cd8f3c33d33182a428a9561a65c0f867dcd31d`
matches the runner inspected locally.

All 17 context prefixes and forced sequences matched, and duplicate B=2 rows
were identical. Nevertheless, B=1 versus B=2 raw-logit top-1 differed at steps
7 and 8 (first divergence 7); both steps had a near-tie margin of 0.125 versus
0.000, then 0.000 versus 0.125, and the top-2 sets matched. Thus free-running
context divergence is not needed to explain these two observed differences in
this one clip. The `shared_context_top1_mismatch` gate failed. This does not
estimate mismatch prevalence, establish answer/semantic impact, or support a
faithfulness claim. B=1 generation took 1.8938 s and B=2 forced generation
3.4804 s; because B=2 tokens were constrained, these timings are not a valid
throughput comparison. Peak allocated memory was 6.7274/7.6202 GiB and reserved
7.6328/9.2871 GiB, respectively. An independent auditor verified the 38-call reconciliation, cache geometry,
all 17 identical context prefixes, forced-sequence equality, raw-logit
collection before the forcing processor, runner/helper/runtime hashes, and
privacy boundaries. It confirmed top-1 differences at steps 7 and 8 for both
B=2 rows. The auditor noted warm-up speed fields could be misread as throughput;
the published derived summary removes only `clips_per_second` and
`seconds_per_clip` from warm-up rows. Its SHA-256 is
`6e14bad29baf087bda6051b6e13d76015c92380d049c3cb6e5cdca0638be2f06`; the
runner-native source result remains unchanged locally under the hash above.
The then-current 51-test CPU suite, byte-compilation, and `git diff --check` passed.

After publication, one independently launched repeat tested stability of this
fixed-clip observation. It completed 38/38 calls and reproduced exactly the
same scalar diagnostics: all 17 context prefixes and forced sequences matched;
raw top-1 mismatches again occurred only at steps 7–8, with margins 0.125 vs
0.000 and 0.000 vs 0.125. B=2 duplicate rows matched exactly. An independent
artifact auditor verified provenance, configuration, geometry, hashes, and
privacy directly. This repeat uses the same clip, checkpoint, GPU, seed, and
configuration; it is not an independent video sample or population estimate.
The runner-native repeat is
`results/qwen_video_batch_throughput_gpu_smoke_20261014_b1_b2_32f_16decode_sharedctx_rep02.json`
(SHA-256 `02e652cc19b0a4bfd80793ac92888fd126ea8c550e2badb62a318b444c985df3`);
the warm-up-rate-free summary is
`results/qwen_video_batch_throughput_gpu_smoke_20261014_b1_b2_32f_16decode_sharedctx_rep02_summary_01.json`
(SHA-256 `464b326058a3a7c2df486a97beee9e69db0493efba46913d43bfdf5f28a39b54`).
The repeat's logged diagnostic time was 25.739 s; B=1/B=2 generation took
1.985/3.467 s, with peak reserved VRAM 7.633/9.287 GiB. These are diagnostic
timings, not comparable throughput because B=2 is constrained.

## Faithfulness status and gates

This window adds 0 direct faithfulness samples and 0 new faithfulness effect
estimates. The closed head confirmation remains 79/79, −0.0756 nat (95% t CI
[−0.1578,+0.0065]), below the old +0.25 nat rule; head/layer/alpha search stays
closed. NOAA human review remains incomplete and NExT-GQA item-level rights
remain unresolved. Claim-mask efficacy execution is therefore still `no_go`.
The batch test is an engine-compatibility diagnostic only and is not evidence of
explanation faithfulness.

## Next actions

1. Publish the independently audited sanitized summary and protocol/log changes
   with scoped staging, then fetch and verify remote hashes.
2. Continue CPU-only estimand and statistical-gate design; the SESOI needs an
   independent, pre-outcome basis and is the remaining statistical design gate.
   Do not call the protocol confirmatory or run gated real data until SESOI,
   human-review attestation, and item-level rights are resolved.
3. The same-context diagnostic is engineering-only; do not add artificial load,
   claim throughput improvement, or count it as faithfulness progress.


## Validator hardening follow-up

The existing plan already fixed Holm adjustment within each predeclared secondary family. The claim-mask validator previously documented this as unresolved without enforcing it. Schema v3 now requires one primary contrast with no within-study adjustment, the two declared secondary families (`baseline_correctness_strata`, `question_type_strata`) with Holm adjustment within each, and an explicit SESOI status (`unresolved` or `preoutcome_fixed`). Confirmatory stage remains rejected; a structural pass cannot authenticate review, rights, or authorize inference. Added tests reject alternate multiplicity, multiple primary contrasts, post-result SESOI labels, unhashable malformed status, and schema-v2 input. The targeted suite passes 25 tests. Independent audit found no authorization bypass or remaining malformed-input crash path; it noted that `preoutcome_fixed` is only a caller declaration because the schema lacks SESOI value/rationale/timing evidence. The contract now says this explicitly. Syntax and diff checks pass. This documentation/schema update resolves the multiplicity-declaration gap only; SESOI and external gates remain open.


## Item-level rights provenance check

Read-only source review found the local official NExT-GQA checkout at commit `63772e0256ad9e34b83d6a108f1ba2041b924607` with root MIT text (local LICENSE SHA-256 `c5bd6fa091c4c3997377f0254e1ae75e56baa452597671a68c1e2f3f6304ff40`). The NExT-GQA data README describes QA/temporal annotation files; the repository license text alone does not resolve rights for the underlying videos. Official VidOR documentation says its videos come from YFCC100M and presents per-video YFCC ID/hash fields ([VidOR dataset documentation](https://xdshang.github.io/docs/vidor.html)); the NExT-QA README identifies its source videos as VidOR ([NExT-QA README](https://github.com/doc-doc/NExT-QA/blob/main/README.md)). The NExT-GQA root license is available at [its official LICENSE](https://github.com/doc-doc/NExT-GQA/blob/main/LICENSE). This means the code/annotation package and source video rights must be tracked separately; no legal determination is made here.

A private, local-only structural check inspected candidate manifests without printing sample IDs or question/answer text: the 80-row necessity-choice candidate set and 3-row smoke set each had video IDs that joined 80/80 and 3/3 to the checked-out NExT-GQA map; neither manifest had license/rights/source-URL fields. The 25-row confirmation manifest has only `video`, `question`, `answer`, and `grounding` fields and no item-rights fields. No private manifest or raw item identifier was staged. The Multimedia Commons YFCC100M documentation states that original media use follows creator-selected Creative Commons licenses and that per-file license information is in the YFCC100M metadata ([YFCC100M Core Dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)). That same official page currently says Yahoo Webscope is not offering the metadata access route. Therefore a NExT-GQA-to-VidOR ID join is necessary provenance but does not resolve item-level video permission; item-specific source metadata or rights-holder confirmation and review remain needed.

Schema v4 now requires a hash-bound item-rights manifest, covered/unresolved eligible-video counts that reconcile to the frozen eligible denominator, and zero unresolved items whenever `status=verified`. “Covered” means item-specific permission evidence for the intended use was reviewed and recorded; a video-ID map join alone is insufficient. The validator checks only declared counts and hash shape, not license truth; execution authorization remains hard-coded not granted. Independent audit verified the count logic, no authorization bypass, and fail-closed malformed inputs, and prompted this wording clarification. The targeted suite passes 26 tests.

Independent design review further specified the private manifest: one row per eligible video, a local-only crosswalk, item license evidence/source and review provenance, and separate intended-use decisions for inference, retention, aggregate publication, and redistribution. Annotation and source-media terms must remain separate; missing/ambiguous rows or use-scope conditions stay unresolved. This is recorded as a future evidence format, not as a claim that the current candidate set has the required rights evidence.

The independent audit refined “conditional” handling: a condition remains unresolved until its applicability and fulfillment are reviewed for the declared use. A separate search found a [Yonsei YFCC100M video metadata database](https://mcml.yonsei.ac.kr/downloads/yfcc100mvideos), but its described fields are video hash, geometry, duration, frame count/rate, and visual features; its page does not describe per-video license metadata. It may help technical provenance, but it cannot close the item-rights gate. The official [Multimedia Commons page](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) still identifies the original YFCC metadata as the source of per-media license information and currently notes Webscope access is unavailable. No metadata archive was downloaded, no item IDs were disclosed, and no faithfulness run was added.

### SESOI declaration gate hardening (in progress)

The schema-v4 validator accepted `sesoi_status=preoutcome_fixed` without a threshold value, explicit units, rationale, date, or independent-review record. This was only a caller declaration, but its shape did not protect against accidentally carrying over the old head result's +0.25 nat criterion or treating status alone as a completed decision. Schema v5 now requires mutually exclusive unresolved and fixed SESOI records. A fixed declaration must include a positive nat threshold, the primary 95% CI lower-bound decision rule, a hash bound to the declared primary claim-mask estimand, a rationale category/hash, UTC fixed time, and a separate independent-review record hash marked completed/before-scoring. The exploratory stage remains the only accepted stage and execution authorization remains false.

The validator still cannot authenticate referenced files, the reviewer, or the truth of timing and no-outcome-access declarations; it does not choose or justify a threshold. Independent code audit found that equality between two caller-supplied estimand hash strings was too weak, so schema v5 now recomputes the hash from canonical JSON containing model identity, final eligible cohort, sampling, claim/mapping/review record hashes, scoring, target, answer-candidate, gold-choice, triplet assignment, and intervention/control definitions. Changing an input while retaining the old hash now fails. Candidate/exclusion provenance hashes remain separately recorded in the cohort declaration; the estimand hash binds the final eligible manifest. It also does not accept old-head results as evidence for this different fixed-answer frame-mask estimand. Claim-mask SESOI remains unresolved. After adding claim/mapping/triplet binding and rejecting malformed Unicode scalar text, the targeted schema-v5 suite and power-planning tests pass 36 tests, including unresolved-value leakage, primary-hash invalidation, bool/non-finite/non-positive thresholds, invalid UTC time, review timing, and malformed category cases. Python byte-compilation and `git diff --check` pass. No GPU/data inference occurred; this changes protocol structure only and contributes 0 direct faithfulness samples and 0 effect estimates.

### Conditional power sensitivity (planning only)

A CPU-only utility computes exact noncentral-t power for the planned rule “two-sided 95% t interval lower bound exceeds SESOI” when the independent video-level paired differences are normally distributed with common variance. SciPy's documented noncentral-t distribution is used for the test statistic. The test statistic has `df=n−1`, noncentrality `sqrt(n) × (true_mean−SESOI)/SD`, and rejects when it exceeds the 97.5th percentile t critical value. The grid varies effective `n` over 20/40/60/80 and hypothetical distance of the true mean above SESOI over 0.1/0.2/0.3/0.5 paired SD; no outcomes, empirical variance, or threshold are supplied. Every `n`, including 80, is a hypothetical scenario, not a verified eligible sample count; current rights/review gates still block efficacy work.

At hypothetical `n=80`, power is 0.424 for an excess of 0.2 paired SD and 0.755 for 0.3 SD; at `n=40`, corresponding power is 0.234 and 0.456. A follow-up grid also calculates minimum n for a hypothetical 80% target: 787, 199, 90, and 34 at excesses 0.1, 0.2, 0.3, and 0.5 SD, respectively. Independent audit checked that each listed minimum reaches 80% while the preceding n falls below it. Thus, under these assumptions, even 80 effective videos do not reach 80% power when the true mean is only 0.3 SD above SESOI. These estimates describe design sensitivity under the stated distributional assumption and two-sided CI rule. They do not justify an SESOI, show that any effect exists, validate normality, or guarantee that the candidate cohort yields those effective sample sizes. Artifact `results/claim_mask_power_sensitivity_20261014_01.json` contains the initial 16-cell grid (SHA-256 `d955a91f46f9ea8601f3400f534310e90090b6ba3247b74bb2beca89d05dac77`). The final extended artifact `results/claim_mask_power_sensitivity_20261014_03.json` includes the 80%-power sample-size sensitivity and records Python/SciPy versions (3.12.14/1.18.1; SHA-256 `b431d6e465a684259fe3c1370a81f24225d69baabd3ed8b968b50933b06174e0`). Reproduce the versioned artifact with:

```sh
.venv/bin/python scripts/analyze_claim_mask_power_sensitivity.py --sample-sizes 20 40 60 80 --excesses 0.1 0.2 0.3 0.5 --confidence 0.95 --target-power 0.8 --output results/claim_mask_power_sensitivity_20261014_03.json
```

The `nct` numerical method is documented by [SciPy's official reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html). Six power-utility tests pass for boundary alpha, monotonicity, assumptions, invalid input handling, and the minimum-n calculation. The combined schema-v5 and power-planning suites pass 36 tests. No GPU or restricted dataset was accessed.

### Human SESOI decision aid

Created `docs/claim_mask_sesoi_decision_template.md` and linked it from the protocol contract. It requires the decision-maker to explain what results below/above a proposed threshold would mean, separate practical interpretation from same-estimand measurement evidence and power planning, record an independent review and pre-score status, and leave schema fields null if no threshold is defensible. It explicitly excludes transfer of the old head `+0.25 nat`, states the exact control/evidence-mask log-margin ratio and geometric-mean interpretation, and prevents it from being mislabeled as an answer-probability change. This worksheet intentionally requires SESOI fixed before any claim-mask outcome score, including development pilot; after fixation, pilot variance/runtime may inform sample planning but pilot effect direction/significance cannot alter SESOI. It states that SESOI review does not clear human-claim, dataset-rights, runtime, or power gates. This is an empty worksheet only; no threshold was chosen and no human review was fabricated. It remains a document-only aid, not execution authorization. No data or GPU inference occurred and direct faithfulness remains 0 samples / 0 new effects.
