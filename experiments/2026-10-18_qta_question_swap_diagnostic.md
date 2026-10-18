# QTA same-video question-swap selector diagnostic (2026-10-18)

## Purpose and status

Test one mechanism premise in the question-token temporal selector: when the
video pixels and processor geometry stay fixed, does changing the question
change selected temporal groups by more than repeated forwards of the same
question? This is an exploratory selector diagnostic only. It is not a test of
whether selected groups contain relevant evidence, whether predictions depend
on them, or whether explanation faithfulness improves.

The target construct remains prediction dependence on video evidence indicated
by an explanation, with internal intervention as the proposed way to improve
it. The question-swap diagnostic is secondary and cannot satisfy the direct
claim-review, timing-correspondence, prediction-link, or scoring gates.

## Frozen cohort and protocol

- Population: local NExT-GQA test-split videos available under the existing
  local media root and not recorded in the 14 registered `data/nextgqa_samples_*.json`
  exclusion manifests. This is a previously studied, test-associated
  population; it is not a fresh or independent cohort.
- Candidate eligibility for this diagnostic reads only the `video_id`, `qid`,
  and `question` CSV columns, the video-ID-to-path map, local media, and the
  registered sample manifests. It requires at least two distinct exact
  question strings. The historical deterministic selector is used only to
  reconstruct and exclude its three prior repeatability/wiring videos; that
  reconstruction uses the old validity rule. No answers, choices, gold spans,
  generated text, or prediction scores enter question-pair selection or model
  forwards.
- Cohort: all eligible local candidates after exclusion of the three known
  prior QTA repeatability/wiring videos; no replacement. The outcome-blind
  inventory found 83 eligible candidates before this explicit three-video
  exclusion and 80 after it. The exact cohort, selected questions, local paths,
  and media hashes are held only in a mode-0600 manifest under the owner-only
  private data root.
- Per video, one pair of distinct question strings is chosen by minimum
  SHA-256 rank over protocol ID, opaque video ID, and the pair's QIDs; identical
  question strings under different QIDs are deduplicated. No question meaning,
  length, token count, annotation, or model output affects pair selection.
- Freeze the endpoint-inclusive 32-frame decoded sample and materialize a
  processor batch independently for each question. The pixel tensors, frame
  metadata, and visual grid must match exactly. Reuse each question's exact
  processor batch for two separate untreated forwards: A1, A2, B1, B2.
- Selector: average question-token attention across heads and layers 5–16;
  choose `ceil(0.10*T)` complete temporal groups, with the existing stable
  lower-index tie rule. The model remains in `eval()`; 4-bit weights use BF16
  compute, BF16 model dtype, eager attention, and the pinned local checkpoint.
- For each video, `J_repeat = (J(A1,A2)+J(B1,B2))/2` and `J_cross` is the
  mean Jaccard of all four A-to-B combinations. Primary descriptive contrast
  is `d = J_repeat - J_cross`; positive values mean the selected-group sets
  differ more across the two questions than under same-question repeat noise.
  Report the finite-cohort mean, sample SD, median, range, and sign counts.
  No p-value, population CI, multiplicity search, or efficacy conclusion is
  allowed because source/event clustering is unresolved and the cohort is
  outcome-exposed/test-associated.
- All per-video media identities, question text, token positions, selected
  groups, and individual metrics remain local. The shareable experiment log
  contains only aggregate counts and estimates after an explicit privacy
  review. Rights status remains unverified; no protected material is published.

## Expected workload and execution gates

The full diagnostic requires **320 attention-capture forwards** (80 videos × 4
forwards), plus a four-forward smoke on the first frozen video. A prior real
QTA capture took about 6 seconds per forward on the RTX 5070 Ti, but this is
only a planning reference; the total estimate is roughly 35–50 minutes plus
model load, checkpoint hashing, decode, and preprocessing. Check current GPU
free memory and WSL memory immediately before each invocation. Run the smoke
first and require exact A/B pixel-tensor equality, finite selector scores,
verified capture-call receipts, and no output overwrite. Do not start the full
run if any smoke condition fails. Keep the GPU job in the existing `.venv` and
the local checkpoint; do not change or rebuild the environment.

## Interpretation limits

A positive contrast only shows question-conditioned selector sensitivity
beyond repeat variation for these fixed test videos. It can arise from wording,
question length, tokenization, or other lexical effects. It does not establish
semantic relevance, correct evidence selection, claim support, answer
dependence, intervention benefit, source/event independence, or faithfulness.
A null contrast would weaken the assumption that this selector responds to
question content under this setup, but would not prove generic visual salience
is the only mechanism.

The 80-video finite cohort is not an independent confirmation: it comes from
the same local NExT-GQA test-associated pool, and exact-byte historical exposure
coverage is incomplete. Related source events and prior model exposure remain
unknown. Any failed decode/forward remains a failure within the frozen 80-row
denominator; no replacement is allowed.

## Verified results and next action

No question-swap GPU run has been performed at the time this protocol was
frozen. Direct faithfulness remains 0/3 independently reviewed claims, 0/15
verified player/browser-to-decoder correspondences, 0/3 prediction-link
judgments, and 0/3 direct scores; there is no direct `DeltaF`, interval, or
test. Next: independently audit the freeze/runner code and private manifest,
run the four-forward real-video smoke, and proceed to the fixed 320-forward
run only if its checks pass.
