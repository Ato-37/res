# Known-history-screened test-split gold-interval margin replication (2026-10-21)

## Purpose and decision

The user waived human review as a prerequisite for local experiments. This is
recorded as a review waiver, not as completed human review. We ran one new,
outcome-blind, known-history-screened test-split cohort through the existing
benchmark gold-interval answer-margin diagnostic. The endpoint is deliberately
kept separate from generated-claim faithfulness: it tests whether masking
dataset-provided QA intervals affects the model's gold-vs-best-foil answer
margin more than masking an equal number of control frames.

## Cohort and provenance

The private path-only history manifest combined all 17 existing
`nextgqa_samples*.json` manifests and all 10 local claim-generation packets:
1,391 path entries / 437 distinct normalized video keys. Its SHA-256 is
`d7ff4cb61dfa98e27f7817a614825c8f67b3d6d452254609527dc658ac6921b4`; mode
0600. The pool builder's `sample_manifest_set_sha256` is
`c22c3609ed0c88a273e4a3f93518d1396a5bb52f464c298d4413b09bcfae74c6`. Since
the invocation used an exact pattern matching only that one manifest, this
equals SHA-256 of the manifest's ASCII SHA-256 string. The builder source
computes this field from the sorted manifest file hashes and excludes matching
normalized video keys before HMAC filters and deterministic media ranking.

The first pool attempt used the validation split, but the fixed scorer uses the
test split. Its preflight failed at candidate selection with zero forwards; no
scores were produced from that attempt. The test-split pool was then rebuilt
with the same manifest and three existing HMAC exposure indexes. It scanned
5,553 rows: 2,677 were excluded by sample paths, 594 by prior-media exposure,
14 had invalid intervals, and 2,268 eligible pairs remained. The fixed
SHA-256 media ranking selected three unique media and 18 eligible question
pairs; exactly three media were extracted. The pool's input summary hash is
`35bba605139b0554ae113afdbc60f5b081cb97778080fcd29a9efba150bb643b`.

The outcome-blind GPU preflight passed for all three videos, with 32 frames
each and five fixed conditions. Timing passed 3/3; decode failures, timing
rejections, and missing control pairs were all zero. The lock contains three
unique media/QID assignments. The preflight SHA-256 is
`d57b2141352551243ed37f3937e0038c06f3ab01783a2a4ee216b94b5d6681f7`; lock
SHA-256 is `2abe093ac9cb764f561b5a34f6f964a66174e61e8617a1dd08efd2ab17c6cd1d`.
An independent auditor verified the runner and source-file hashes against the
committed source, input split hashes, selected-pair filter, unique assignments,
and preflight-to-lock bindings. The auditor initially questioned the history
manifest binding because the pool stores a digest of the manifest-hash set,
not the raw manifest hash; source inspection and independent recomputation
confirmed the single-manifest set digest matches exactly.

## GPU run and observations

The completed run scored 3 videos x 5 conditions = 15/15 forwards. It used the
pinned Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, RTX 5070 Ti, 4-bit quantization,
BF16 compute, eager attention, 32 frames, and seed 20261015. Initial free VRAM
was 14.657 GiB; available host memory was 13.233 GiB; peak allocated VRAM was
11.428 GiB; elapsed time was 73.44 seconds. No forward failures or final CUDA
synchronization failures occurred. The owner-only result JSON is
`data/private_nextgqa/gold_interval_cohort_20261021_04_result.json`, mode 0600,
SHA-256 `53d17bd72fee6ac0f92bfcf44388a1baa5372bafe3948995887feb1eb8260767`.

Primary contrast, in nats:
`black equal-count control gold-vs-best-foil margin - black gold-interval-mask
gold-vs-best-foil margin`.

| Fixed row | Black control − gold-interval mask | Blur control − gold-interval mask |
|---|---:|---:|
| 0 | +1.8750 | +0.2500 |
| 1 | −0.2500 | −0.2500 |
| 2 | +0.1250 | +0.1250 |
| **Mean** | **+0.5833** | **+0.0417** |

The black-condition sample SD is 1.1342 nats (range −0.2500 to +1.8750); the
blur-condition sample SD is 0.2602 (range −0.2500 to +0.2500). No inferential
test or population confidence interval is reported for this deterministically
selected, non-probability sample of three. The model's unmasked gold-choice
accuracy was 0/3, so these margin contrasts do not represent successful answer
predictions in this cohort.

## Interpretation, limits, and next action

This cohort does not reproduce the previous test-split smoke's black-mask mean
of −0.2500 nats: the new mean is +0.5833, with mixed row signs. The blur
contrast mean is near zero (+0.0417). The gap between the black and blur
averages is consistent with sensitivity to the black-fill intervention, but
three items cannot identify its cause. The all-wrong unmasked predictions also
limit interpretation of a gold-label margin as prediction dependence.

This is only an exploratory measurement of model dependence on dataset-provided
QA intervals under black/blur masking. It does not evaluate generated
explanation claims, claim-to-evidence alignment, truth, causal use of evidence,
or improved faithfulness. The relative presentation-time origin is not
independently calibrated to the annotation playback clock. The cohort is
deterministically selected rather than representative, the prior-exposure
indexes are incomplete and not externally authenticated, and dataset/model
rights remain unverified. All videos, QA text, private manifests, detailed
receipts, and score rows remain local and unpublished.

Human reviews actually completed remain 0/3; direct generated-claim
faithfulness scores remain 0/3. This run does not change either count. Do not
expand this endpoint based on these observed values. A useful next research
step is to build a direct prediction-dependence design around a predeclared
model-selected evidence interval, report it explicitly as unreviewed and
exploratory under the user's waiver, and measure the prediction change against
an equal-duration control. It must not be called human-validated claim
faithfulness.

## Validation

- Outcome-blind GPU preflight: passed, three videos, five conditions.
- Score runner: passed, 3/3 complete condition sets, 15/15 forwards.
- Independent result-binding arithmetic assertions: passed for result,
  preflight, lock, pair-filter and exclusion-file hashes; equal evidence/control
  frame counts; and recomputed black/blur means and accuracy.
- `test_run_nextgqa_gold_interval_margin_pilot`: 21/21 passed.
