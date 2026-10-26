# Local exploratory claim/video content-dependence on an overlap-screened cohort (2026-10-21)

## Purpose and scope

After the earlier three-row diagnostic reached a closed endpoint, this work tested whether a separate validation cohort could be selected outcome-blind and scored without collecting additional human review. The user's instruction waives review collection as a prerequisite for local exploratory experiments. It does not change review records: no human judgment was performed for this cohort. This is a conditional-likelihood sensitivity diagnostic, not a direct faithfulness score.

## Cohort construction and duplicate detection

The CPU-only ZIP candidate-pool builder scanned the local validation split and local archive, applying the three supplied prior-exposure HMAC indexes and existing sample-manifest paths. The first pool selected 10 media files from 534 unique annotation-eligible videos. Its 3 generated claims turned out to be exact matches to all three earlier development rows by both source-video bytes and generated-claim text. The initial pool therefore failed independence, and its repeated smoke score is excluded from all evidence and estimates.

To repair selection, a private mode-0600 path-only manifest was created from the union of existing `nextgqa_samples*.json` manifests and local claim-generation packet video paths (431 unique path entries). The builder then selected a second 10-video pool using the same deterministic media-hash ranking and annotation eligibility rules. The new generation packet contains three distinct source-video byte hashes and has zero video overlap with the three prior development rows checked. This is a known-history overlap check, not a certification of freshness: the prior-exposure ledger remains incomplete and unauthenticated.

The new Qwen generation packet is `data/private_nextgqa/claim_generation_val_exploratory_20261021_02.json`, SHA-256 `34231eaadee610bcfe7b81c5eeaef43553ee3d11122e064e490ca8fb2ce0a533`, mode 0600. It records 3/3 generated videos but only 2/3 structurally valid standalone claims; the remaining row failed the sentence-shape contract. No row was replaced. The packet's cohort digest and source/input digests remain inside the protected receipt.

## GPU execution and observations

The successful new claim rows were scored with the existing pinned Qwen2.5-VL-7B content-dependence runner. Each condition uses the original 16 decoded RGB frames or same-dimension constant `(127,127,127)` frames, deterministic tokenization of the saved sentence, identical question/choice prompt, and exactly two teacher-forced forwards per row. Model revision was `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute, eager attention, seed 20261021, RTX 5070 Ti. Both receipts show distinct processor-pixel hashes, equal timestamps, 2/2 completed forwards, peak allocation about 7.34 GiB, and 14.657 GiB free VRAM before load.

| Frozen row | Claim tokens | Original minus gray mean log-likelihood (nat/token) | Status |
|---|---:|---:|---|
| row 0 | 11 | +0.977361 | 2/2 forwards |
| row 1 | — | — | claim sentence-shape validation failed; 0/2 scoring forwards |
| row 2 | 10 | +0.793603 | 2/2 forwards |

The two successful row values are reported separately. **No cohort mean, CI, hypothesis test, or population inference is reported**, because one of the three frozen rows has no valid target claim. The two positive contrasts indicate that, for these two model-generated sentences, Qwen assigns higher conditional likelihood under the sampled original video than under constant-gray frames. The contrast remains sensitive to severe out-of-distribution image replacement and says nothing about semantic support, truth, causal grounding, or improved explanation faithfulness. The saved sentence was re-tokenized; original generation token IDs were not retained.

An initial attempt to pass the `partial` generation packet failed before GPU execution because the reusable reader accepted only packets whose entire cohort was valid. The reader now has an explicit opt-in path that accepts a partial packet only when the requested row is marked valid, its nonempty claim is present, and the declared structured-row count matches a full packet recount. The ordinary frame-permutation caller retains strict all-pass behavior. Regression tests for the reader, scorer, and analyzer passed **17/17**; Python compilation passed. The failed zero-forward receipt remains local and is excluded.

Successful receipt SHA-256 values:

- row 0: `37f43d6cdb19ac901680d47fda3c3fa8e7248ef2df390e1d561f4c0cfef6d24a`
- row 2: `f20116b045c38fd2a4ac5c1f456c4c91a64981e9677487eb1e709e401014239f`

## Faithfulness progress and next action

This advances exploratory evidence of **visual-input likelihood sensitivity on two non-overlapping known-history rows**. It does **not** increase direct faithfulness progress: direct `DeltaF` scores remain 0/3, actual reviews 0/3, verified clock/frame correspondences 0/15, and Stage-2 prediction links 0/3. No treatment improvement has been established. Rights remain unverified. Videos, claim text, question text, local paths, and all derived receipts remain local and unpublished.

Next, retain the failed row as a failure and do not compute a survivor-only mean. Before any larger GPU cohort, improve the generation validity rate with a protocol fixed before new outcomes, then independently audit a new cohort's overlap and generated-target contract. Keep direct faithfulness status separate unless an actual claim-to-evidence mapping and direct evidence/control contrast are measured.

## Independent audit and reader hardening

An independent read-only audit confirmed the two packet/row/video/claim hash
joins, distinct row tokens and videos, score arithmetic, condition pixel hashes,
frame/timestamp geometry, model/revision, 2/2 forward counts, zero attention
hooks, and owner-only receipt modes. It found one reader-contract edge case:
Python treats `True` as integer `1`, and a selected `valid=true` row could lack
nonempty claim text while another row satisfied the declared structured count.
The reader now requires `type(declared_count) is int` and verifies the selected
row's `valid is True` plus a nonempty string claim. The tests reject both cases
and still pass 17/17. This is a validation-only hardening; it does not alter
scoring or the two receipts. Reversing only this post-run validation hardening
reconstructs the helper source exactly at the SHA-256 recorded in both run
receipts (`0edba95d102c8c3cd4626c25d199349a828c69be0dc246510b4ebf5430b19b98`).
The scoring runner hash in the receipts matches the current scorer source.

## Prompt-format amendment and independent cohort (2026-10-21)

The preceding fixed three-row generation had one claim fail a format-only
sentence-shape check. Before selecting new outcomes, the generator gained an
explicit `v2_short` prompt version requiring at most 25 words, one simple
sentence, and one main action. The original v1 suffix remains unchanged and
is still the default for old packets. New rows store their prompt version, and
the content scorer resolves that exact recorded version; missing version means
legacy v1, while unknown versions fail closed. The versioned-prompt and scoring
regressions passed within the combined **28/28** unit tests; compilation and
scoped whitespace checks passed.

A third outcome-blind pool was built after excluding 434 recorded local video
paths. Its generation packet is mode 0600, SHA-256
`3c39125c1f5ec5dc36e29468d0009858ff0de65947eba60eb6ec722a246866f0`, with
cohort digest `fe38150029e50c9ad37684deeb70a9744928e89b3b5ec5416330fe3984194893`.
It generated three structurally valid claims in 17.27 seconds (3/3), all on
unique video bytes with 0/3 overlap against the three prior development rows
checked. These are not certified fresh against the complete history; event
identity and exposure sources remain incomplete.

Each of the three rows then completed the original-versus-gray score pair on
the pinned RTX 5070 Ti setup, for **6/6 forwards** total. The analyzer and an
independent primary arithmetic/provenance check agree:

| Row | Re-tokenized claim tokens | Original−gray (nat/token) | Peak allocated VRAM |
|---|---:|---:|---:|
| 0 | 11 | +1.557732 | 7.383 GiB |
| 1 | 10 | +1.312235 | 7.253 GiB |
| 2 | 10 | +0.465541 | 7.387 GiB |
| **finite-cohort mean** | — | **+1.111836** | — |

Median is +1.312235 nat/token; range is [+0.465541,+1.557732], with all
three rows positive. This is a fixed three-row description only. No CI,
hypothesis test, or population inference was calculated. Every receipt binds
the same packet hash and `v2_short` generation prompt; the scoring runner
hashes match the current source. Video/claim hashes do not match the old three
rows, row tokens are unique, all rows use 16 frames with matching timestamps,
and the processor-pixel hashes differ between original and gray conditions.
Each receipt is owner-only mode 0600 and records 2/2 forwards, fixed model
revision, and zero attention intervention hooks. Receipt SHA-256 values:

- row 0: `9d52d147ec79aa41fa547a4f5ddf474d9cf9d2b6fa191c05c9bf782f6e38fce3`
- row 1: `010fc24c4c3b9f6fc4bfd78cbe2425c083ff403b2698a31a829faf18f67fbe63`
- row 2: `f01becd4ad144efcbe870762091d45019879fe24c95e7278d6e9e908fa3d55e1`

The v2 format check was 3/3 versus 2/3 in the previous separate three-row
v1 attempt. The cohort, sampled videos and generated claims differ, and each
sample is tiny, so this is only a feasibility observation; it does not show
the prompt improves validity in general. The score itself only shows higher
conditional claim likelihood under original frames than a severe constant-gray
replacement for these three rows. It does not show truth, evidence support,
causal grounding, or improved faithfulness. Actual human reviews remain 0/3;
direct faithfulness scoring remains 0/3 with no `DeltaF`, interval, or test.

Next action: evaluate a content-preserving control on a separately fixed
cohort or stop pursuing gray-mask sensitivity as a primary research endpoint.
Keep the direct faithfulness endpoint and its evidence/clock/link counts
separate from all conditional-likelihood diagnostics.

## Post-result frame-mean sensitivity control on the v2_short rows

To check whether the original-minus-gray contrast could be explained only by
the severe constant-gray replacement, the saved-claim runner was extended to
restore the generation prompt version recorded in each packet row. This keeps
the question/choice text, claim tokenization, timestamps and 16-frame count
fixed while replacing each frame with its rounded RGB channel mean. CPU tests
for the frame transform and versioned prompt resolver passed 6/6; compilation
and scoped `git diff --check` passed.

The same three already-selected v2_short validation rows were scored on the
local RTX 5070 Ti, using the pinned Qwen2.5-VL-7B revision, 4-bit weights,
BF16 compute, eager attention and two forwards per row. All 6/6 forwards
completed. Each receipt matches the current source hashes, packet hash and
`v2_short` prompt; row tokens are unique; processor pixel hashes differ across
conditions; receipt files are mode 0600. Receipt SHA-256 values:

- row 0: `0a51fb0462a0236ccad56d4fcb67a1a37ed573299cd26b237b585f9d87e380db`
- row 1: `73ddf7bc1d812edeab5f4ac8d2c5da677bfda78ef213c76bc40ada8c977ecee8`
- row 2: `e4ee03619aeca8f51441562c25ca57882720ec2d598d762ea5a9c9a675082515`

| Row | Original minus frame-mean log likelihood (nat/token) | Original minus gray (nat/token; prior run) |
|---|---:|---:|
| 0 | +1.522251 | +1.557732 |
| 1 | +1.394775 | +1.312235 |
| 2 | +0.499409 | +0.465541 |
| **Finite-cohort mean** | **+1.138812** | **+1.111836** |

All three frame-mean contrasts are positive. Their paired difference from the
previous gray contrasts averages +0.026976 nat/token. This small selected
cohort does not support a meaningful ranking between the two controls and has
no confidence interval or hypothesis test. It does show that, in these three
rows, the conditional-likelihood drop also occurs when spatial layout is
removed but per-frame average color is retained. Both transformations are
distribution shifts, and neither estimates claim truth, semantic support,
causal grounding, or explanation faithfulness. This is post-result analysis on
the same selected claims, not an independent replication.

Human review collection was waived for this exploratory run, but actual human
review remains 0/3; the receipts explicitly say review was not performed.
Direct faithfulness scores remain 0/3, with verified clock/frame matches 0/15
and prediction links 0/3. The next useful experiment must avoid another
perturbation-only likelihood endpoint: construct an outcome-blind, directly
scorable prediction-link/evidence cohort, retaining the review waiver as an
execution permission while clearly labeling all unreviewed claims exploratory.
