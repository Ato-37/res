# NExT-GQA finite-cohort claim-aligned input-dependence protocol v2

Status: **design draft; not frozen, not scoreable, and not an execution
authorization**. This pre-score proposal supersedes v1 only after a complete,
valid claim packet, target manifest, code, cohort membership and this protocol
are jointly hash-frozen. The recorded cohort digest freezes membership only;
it does not freeze usable claim text, a valid claim-input packet, or the final
frame-sampling plan. The current format-valid candidate cohort is the three-item
`sentence_exploratory` set with digest
`162fda51377768ae4e488fbfa1a8377f903ead3af7b277a17780b53e67b79a95`. If any
input differs, stop and version a new protocol before inspecting scores.
Protected data and derived artifacts remain local; rights status is unverified.

## Protocol routing (do not mix gate versions)

This draft is the candidate protocol for a **finite three-record descriptive
feasibility study**. It is separate from the schema-v13 all-eligible candidate
estimand/contract. `scripts/preflight_claim_mask_gates.py` validates schema v13
and intentionally returns blocked because its inferential and SESOI items are
unresolved and confirmatory readiness is not granted; that report is not a v2
feasibility-protocol audit. Conversely,
`scripts/claim_score_preflight_rehearsal.py` only joins synthetic review and QA
target fixtures, reports nine forwards as planned for the three-row contract,
and always records zero score calls and `execution_authorization=not_granted`.
It does not validate this draft's freeze state or authenticate real inputs.

No production atomic preflight is currently connected to the v2 protocol. A
future v2 runner must validate this exact protocol version and all three rows
atomically before model load, retain the denominator of three, and make zero
score calls for any preflight failure. It must not inherit v13's unresolved
population-inference/SESOI requirements or silently borrow any v13 pass
semantics. This
routing note does not freeze v2, satisfy any review/clock/target gate, or grant
scoring authority; the complete protocol, claim packet, target, cohort, frame
sampler, and code still require joint hash-freeze before any score is opened.

## Estimand and scope

The estimand is a finite-cohort descriptive contrast of **model next-token
answer margins under two masking interventions** for these exact three
video/claim/question records, using the pinned Qwen2.5-VL-7B-Instruct
checkpoint. It is not a population estimand, confirmatory test, causal claim,
or threshold-based efficacy decision. The result may describe how the fixed
gold-versus-best-foil margin changes under the human-reviewed claim-mask
condition relative to the fixed temporal-control mask for these items. It does
not establish that the model used the reviewed evidence, that the generated
claim is faithful or true, the model's internal reasoning, or generalization
to other videos, questions, models, or interventions.

The cohort is all three records, one fixed benchmark question per unique video.
Question selection, claims, masks, controls, scoring code, and model inputs
must be frozen without seeing intervention scores. Keep the requested
denominator at three. Any unsupported claim, failed review, missing interval,
source/clock mismatch, invalid control, or failed target row blocks the whole
cohort; do not replace or drop records after inspecting outcomes.

## Per-item contrast

For each item, construct the user text exactly as
`question + "\nOptions:\n" + "\n".join(f"{label}. {choice}" for the five
ordered choices labeled A–E) + "\nChoose the best answer. Respond with one
letter only.\nAnswer:"`. Put this text after the video content item in one user
message and render it with the pinned processor's
`apply_chat_template(..., tokenize=False, add_generation_prompt=True)`. The
scoring prefix ends at that rendered generation prefix, immediately before the
assistant's first answer token. Tokenize each of `A`–`E` with the exact pinned
`processor.tokenizer(label, add_special_tokens=False)` call; require one token
per label and five distinct token IDs. Score each single next-token label from
the same rendered prefix using natural-log probabilities in nats. Bind the
rendered prefix bytes, token IDs, tokenizer, processor, prompt builder and all
code hashes; independently recompute them before scoring. For condition `z`,
define

`S_i(z) = log P(gold_i | z) - max_{foil != gold_i} log P(foil | z)`.

Conditions are (1) full video, (2) black-filled frames overlapping the union
of all distinct human-reviewed support-occurrence intervals for the one frozen
claim, and (3) black-filled equal-count temporal-control frames. Preserve each
reviewed occurrence as its own interval; distinct intervals may overlap or
share endpoints, and the union counts any sampled frame only once. Do not
choose one occurrence or merge disjoint intervals after review. Hold the question,
ordered choices, gold index, prompt, checkpoint,
tokenizer, processor, frame sampler, and every non-mask input fixed. For item
`i`, the paired contrast is

`d_i = S_i(control) - S_i(claim_mask)`.

For each private item/condition record, retain the five A–E next-token
log-probabilities and identify the condition's best foil, in addition to the
gold-versus-best-foil margin. The best foil can change across conditions, so
the contrast can move because the gold score changed, the leading foil changed,
or both. This decomposition is diagnostic only; it does not change `d_i` or
the primary cohort summary. Keep choice-level values and labels local in the
mode-0600 result; public-safe aggregates must not expose the protected answer.

Report each `d_i` separately. Report `D3 = (d_1 + d_2 + d_3) / 3` only when all
three full/claim-mask/control triplets complete. If any triplet fails, report
completed triplets and sanitized failure codes, set `D3=null`, and do not
average survivors. The full-video condition is descriptive context and is not
part of `d_i`. A complete run requires nine forwards.

**SESOI status: `not_applicable_to_finite_descriptive_estimand`.** This is not a
zero SESOI and does not imply success or failure against a practical threshold.
Do not report a confidence interval, standard error, p-value, population
estimate, subgroup comparison, multiplicity-adjusted inference, or
threshold-pass claim. Do not transfer the prior head-study +0.25 nat threshold.

## Frozen intervention and controls

Use 32 endpoint-inclusive uniformly sampled decoded-frame indices. The final
sampler and its source/configuration hash must be frozen before scoring.
Evidence ranks are sampled frames whose decoder PTS lies inside any
human-reviewed interval, inclusive of endpoints. For `k` evidence ranks,
require `k >= 1` for every row after applying the frozen 32-frame sampler and
the calibrated PTS guard. Record the sampled overlap count and ranks in the
private audit record. If `k == 0`, report
`zero_sampled_evidence_overlap` and block the complete cohort before model
load; do not score no-op masks or replace the row. Otherwise, select exactly
`k` distinct non-evidence ranks minimizing the sum of absolute rank distances;
ties go to the lexicographically smallest ordered control-rank tuple. If fewer
than `k` controls exist, block the cohort before model load.
Set every pixel channel of selected complete RGB frames to zero before Qwen
resizing/normalization. Only selected frame pixels may vary across conditions.
Bind the transform code/configuration and exact processor inputs.

Black masking can create out-of-distribution inputs. Rank matching controls
sampled position distance and frame count, not visual information or semantic
content. These limits must accompany any result.

## Required gates before any score call

All three items must have a completed independent review of the full source
video, following every requirement in the frozen
`docs/nextgqa_claim_review_policy_v5.json` policy. The complete, valid
claim-input packet and displayed claim text must match byte/hash-bound inputs;
the claim must be supported by visible pixels and all relevant occurrences
annotated under the frozen occurrence policy. The reviewer must be independent
of claim generation, scoring, and custody, and must not access dataset indexes
or external answer sources or see gold answers, seed intervals, or outcomes.
Verify opaque token-to-media custody and exact staged media bytes. Any review
failure blocks all items.

Calibrate the actual HTML5 review-player clock against decoder PTS using the
exact staged media bytes. Each anchor must identify the compositor-presented
frame using `requestVideoFrameCallback` metadata, and separately verify
correspondence to the decoder frame/PTS. The current worksheet captures only
the latest callback and does not verify that it is fresh at click time;
`video.currentTime` alone does not identify the displayed frame. The current
converter pairs `mediaTime` to the frozen decoder PTS but still returns an
unvalidated bundle. Freeze the browser and decoder versions,
stream/timebase, PTS-origin rule, and beginning/interior/end anchor coverage.
Keep predeclared fit and held-out anchors disjoint; report maximum held-out
residual and use a predeclared empirical guard to reject any frame whose
sampled-frame classification can change within that guard. The guard is an
empirical check on captured anchors, not an uncertainty bound, confidence
interval, or future-error guarantee. Frame2time/PTS residuals and ffplay
observations are not substitutes. Browser playback and capture must be
exercised in the actual review environment. If frame correspondence is missing
or ambiguous, the clock gate fails.

An empirical residual check is not an independently verified clock pass. The
calibration helper reports only `empirical_residual_within_tolerance`; the
assignment planner additionally requires a caller declaration that independent
review is claimed and hashes for capture and decoder-frame-correspondence
evidence. The planner checks those fields structurally only; fabricated hashes
and declarations can pass this planning seam. A signed, evidence-reading audit
manifest is still needed for authentication. The planner remains unvalidated
and cannot authorize scoring.

Before model load, build an exact source-backed target for all three rows. Bind
the full raw `test.csv` digest plus a per-row digest over all parsed CSV fields,
and cross-check each unique video/QID pair against lineage. Bind the exact
question, ordered choices, gold index, and claim-input digest. Recompute
`A`–`E` token IDs and one-token counts with the pinned local tokenizer using the
production scoring path; bind the precise rendered scoring-prefix digest and
all model, model-config, tokenizer, processor, prompt-builder and scoring-code
hashes. The current structural target contract, actual-CSV adapter, and
experimental structural target wrapper do not alone authenticate all upstream
declarations, tokenize prefixes, or authorize inference. A production atomic
preflight is still to be implemented and audited; it must require this
protocol's 32-frame endpoint-inclusive sampler and verify its frozen
sampler/configuration and code hashes (the current planner accepts a
caller-specified frame count and does not bind those hashes), check every
upstream artifact, decoder PTS/pixel identity, frame assignment and score-input
binding, and return zero score calls for any failure.

## Reporting and data handling

Record requested/effective/reviewed/supported/assigned/scored counts, all nine
requested/attempted/completed forwards, failures, duration, peak VRAM, seed,
software/hardware/precision, and exact source/configuration hashes. Keep raw QA
and per-item derived scores in mode-0600 local artifacts; aggregate output must
omit QA text and stable identities. Do not publish protected-data-derived
artifacts while the local-only restriction applies.

## Current readiness

Direct claim faithfulness remains **0/3 human-reviewed and 0/3 scored**. A
complete valid claim-input packet for this frozen cohort is still required. No
actual HTML5-player clock capture or production atomic launch gate has passed.
A CPU diagnostic now checks the QA-margin runner's full unmasked prompt and
processor path on these source-bound rows, including A–E label-token shape;
it does not validate the final claim-bearing prompt, masked inputs, or atomic
target launch gate. The finite-cohort descriptive
estimand avoids unjustified population inference, but it does not remove these
operational gates. Status remains **no-go for scoring** until all gates above
are verified and the complete protocol/artifact set is frozen before outcomes
are accessed.
