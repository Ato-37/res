# 16-frame reviewed-claim QTA sensitivity protocol (exploratory)

Status: **post-hoc development protocol; not confirmatory or preregistered**.
It matches the existing cohort11-v3 claim packet and the current 16-frame
review artifact. It supersedes the 32-frame draft only for this exact local
feasibility cohort; it does not alter prior runs or their estimands.

## Question and scope

For each frozen Qwen2.5-VL-7B-Instruct claim, does question-attention temporal
key steering increase the fixed claim's differential log-likelihood
sensitivity to reviewer-selected support groups versus an equal-count nearby
temporal control, compared with no steering?

This is conditional probability of a re-tokenized claim string under one fixed
prompt and a particular attention intervention. It does not test claim truth,
whether the text explains a predicted answer, or general faithfulness. Claims
are re-tokenized from saved parsed strings; the original generation token IDs
were not retained. Review labels are self-reported and limited to 16 sampled
frame thumbnails.

## Frozen mapping and source checks

- Use only the existing cohort11-v3 packet and exact fingerprinted `_08`
  review artifact. Require exact artifact SHA/fingerprint, embedded packet SHA,
  source record index, video/QID/candidate-pair hashes, question/claim hashes,
  and selected-row order to agree. Rehash packet and source video locally.
- Require exactly 16 ordered frame times in packet and artifact to agree within
  `1e-6` seconds. Re-decode the same 16 source RGB frames and check their
  timestamps/pixel verification against the packet. Frame indices refer to
  these sampled frames, not continuous video intervals or processor patches.
- Map reviewed frame index `i` to complete Qwen temporal group `floor(i/2)`;
  require processor temporal patch size 2, 8 groups, matching `video_grid_thw`,
  and exact visual-token positions. Deduplicate selected groups. Never infer
  support between selected frames.
- A row is score-eligible only if every submitted reviewer chose `supports`
  and at least one frame index was selected by every reviewer (the
  intersection, not the union of individual selections). Preserve all requested rows and every
  non-support, disagreement, decode, geometry, and runtime failure in the
  denominator. No replacements. This conservative rule is exploratory and
  does not authenticate reviewers.

## Intervention and outcome

Build the exact generation prompt recorded for each row (`v3_sequence_complete`)
using the saved question, five options, 16 decoded frames, sample FPS, pinned
processor, and chat template. Re-tokenize the frozen parsed claim once. Capture
question-token attention on the unmasked full-video prompt, average heads and
layers 5–16 over complete temporal groups, and choose the top
`max(1, ceil(0.10 * 8)) = 1` group (12.5% of groups; earlier index breaks ties).
On final claim-token prediction rows in layers 5–16, add `ln(2)` to those
selected visual-key logits.

Mask complete support groups or equal-count control groups with black fill in
the pixels while keeping the prompt, processor geometry, token target and
nonvisual inputs fixed. Do not delete visual attention keys in any condition.
The control is disjoint from support and from QTA keys;
among feasible group sets, minimize the sum of absolute group-rank distances
paired in sorted order, with lexicographically earliest tie break. If no such
control exists, keep the row as a pre-score failure. QTA may overlap support.
Record overlap but do not condition eligibility on outcomes; no branch deletes
visual keys, so overlap cannot silently cancel the QTA treatment in one branch.

For steering condition `g` in `{baseline, QTA}`, define
`D_g = mean_token_logP(claim | control-group mask, g) -
mean_token_logP(claim | support-group mask, g)`. The primary finite-cohort
contrast is `DeltaD = D_QTA - D_baseline`. Positive `D` indicates more
conditional claim-likelihood loss under support masking than under the matched
control; positive `DeltaD` indicates that the specified QTA bias increases
that contrast on this cohort. Report all per-row values, requested/effective
counts, failure stages and key overlaps. With at most three reused,
post-hoc rows, report descriptive values only: no population CI, significance
test, or confirmatory claim.

## Resource and privacy plan

One-row technical smoke: one selector attention forward plus four
support/control × baseline/QTA score forwards. Full cohort: up to 15 forwards
(5 per row), plus model load. Prior adjacent 16-frame runs used about 7–9 GiB
peak VRAM and minutes rather than hours; check free VRAM and host memory before
each run. Smoke uses only synthetic frame-group mapping and must be explicitly
marked as such; it is engineering validation, not a human review or faithfulness
result. It validates artifact fingerprint/schema without recording synthetic
decisions as human evidence. The receipt records package versions, source
hashes, pinned revision, config hash, and a checkpoint file-inventory hash
(names/sizes; weight bytes are not rehashed). All packets, videos, responses
and receipts remain local mode 0600; rights status remains unverified.

## Remaining prerequisite

The actual current-artifact human response is absent. Do not fabricate support
labels or use a synthetic smoke receipt as the real-data result. Once a valid
response arrives, validate it with the adapter, freeze a new receipt/source
hash manifest, run the one-row human-mapped smoke, inspect it, then score all
three requested rows once. This protocol change was made before any score from
this proposed estimand was computed; the cohort is nevertheless post-hoc and
must remain exploratory.
