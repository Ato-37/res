# NExT-GQA three-video claim-linked input-dependence feasibility protocol

Status: **design draft; not frozen for scoring and not execution-ready**. This
document defines a bounded descriptive target for the three existing review
candidates. It does not authorize review, inference, or claim-level conclusions.
Rights remain unverified; any protected data and derived results stay local.

## Target and gate

The target is exactly the existing frozen three-video candidate manifest and
its one model-generated claim per unique video; record and verify the exact
manifest hash before review. Use one benchmark QA per unique video, selected
without inspecting review or score outcomes. Keep all three in the denominator. Before loading the model, all
three must have a completed independent review, a `supports_claim` decision,
verified token-to-media custody, exact media/source lineage, a validated
HTML5-player-to-decoder-PTS calibration, valid claim interval masks, and valid
equal-count disjoint controls. If any item fails, score none of the cohort and
do not replace it. A different subset would require a separately specified
target and protocol before outcomes are inspected.

The proposed score has three conditions per video: full video, black-filled
claim-evidence frames, and black-filled rank-matched control frames. Hold model,
checkpoint, prompt, question, gold answer, four foils, frame sampler, processor,
and every non-mask input fixed. The full condition is descriptive context, not
part of the primary contrast. This is nine model forwards for a complete
three-video cohort.

## Primary descriptive contrast

For each of the five fixed answer choices, score its single next-token label
(`A`–`E`) using the same model/tokenizer prefix and the label's natural-log
probability (nats). For condition `z`, let

`S(z) = log P(gold answer | z) - max_foil log P(foil | z)`

using the same five answer candidates and fixed gold index in all conditions;
the best foil may differ by condition. For video `i`, define

`d_i = S(control_i) - S(claim_mask_i)`

and, only if all three paired scores complete,

`D3 = (d_1 + d_2 + d_3) / 3`.

A positive `d_i` means the claim-mapped mask lowers this benchmark-answer margin
more than the matched control mask for that item. Report `D3` only as an exact
finite-cohort description. Do not report a confidence interval, p-value,
standard error, population estimate, subgroup test, or SESOI pass claim. If any
condition fails after scoring starts, retain requested denominator 3, report
the completed triplet count and sanitized failure codes, and set `D3=null`; do
not average the successful subset.

The current implementation proposal uses 32 endpoint-inclusive, uniformly
sampled decoded-frame indices. The final sampler and its source/configuration
hash must be frozen before scoring. Evidence frames are sampled frames whose
PTS lies within any reviewed interval, with inclusive endpoints. For `k`
evidence ranks, choose exactly `k` distinct non-evidence ranks minimizing the
sum of absolute rank distances; resolve ties by the lexicographically smallest
ordered control-rank tuple. If fewer than `k` controls exist, the full cohort
fails before model load. Apply the same exact black transform to complete RGB
frames (all channels zero) before Qwen resizing/normalization; only the selected
frame pixels may differ among conditions. Bind this transform's code/config
hash and the exact preprocessor inputs in the assignment manifest.

This measures answer-margin dependence under a specific claim-linked input
masking operation. It does not score the generated claim's own likelihood,
establish claim truth, prove internal reasoning or causation, or establish broad
explanation faithfulness. Black masking can create out-of-distribution inputs;
rank-matching equalizes sampled position distance and frame count, not visual
information or semantic content. The gold answer and foils define the score and
must never be shown to the reviewer.

## Required evidence and provenance before scoring

The preflight must bind exact bytes/hashes for the candidate and eligible
manifests, claim-generation output, review policy/export/index, custody/source
crosswalk, calibration artifacts, decoder PTS and frame-pixel identities,
sampled processor inputs, mask/control assignments, scoring implementation,
model/tokenizer/checkpoint/configuration, and all imported timing helpers. It
must verify all three rows atomically before model load and produce zero model
calls on any preflight failure. A calibration report must use the exact staged
media bytes and the actual HTML5 player clock, keep held-out anchors separate
from fit, report the maximum held-out residual, and reject any assignment whose
sampled-frame classification changes within a predeclared empirical guard
derived from those residuals. This guard is an empirical check over the captured
anchors, not an uncertainty bound, statistical confidence interval, or
guaranteed future-error bound. It must be reported with that limit. Prior
frame2time-versus-PTS residual
summaries and `ffplay` clocks do not meet this requirement.

The frozen QA-target manifest must separately bind each claim row to its exact
benchmark question and ordered five choices, the fixed gold index, dataset-QID
digest, benchmark source-manifest and source-row digests, rendered
scoring-prefix digest, and the exact claim-input-manifest
digest. It also binds the scoring protocol, checkpoint, model config,
tokenizer, processor config, and prompt-builder hashes. Each row records the
`A`–`E` label order, corresponding token IDs, and one-token counts. Those counts
are only declarations until an independent builder reruns tokenization with
the exact checkpoint tokenizer and scoring prefix; a structural manifest
validator must never present them as tokenization evidence. Changes to choice
order, gold index, prefix, tokenizer, or model config invalidate the frozen
target binding.

The current `claim_answer_target_contract.py` is only a synthetic-tested
schema/hash validator for this future handoff. It does not create target
manifests from the benchmark, inspect tokenizer behavior, authenticate source
artifacts, or authorize inference. A production builder and independent
tokenizer audit remain required. `claim_score_preflight_rehearsal.py` connects
this structural contract to the PTS assignment planner, a hash-bound lineage
manifest, and synthetic QA source rows. It checks that question/QID/choice/gold
and source-row digests agree with the worksheet and supplied source rows, tests
all-or-none failure, and never calls the model. Separately,
`nextgqa_actual_qa_source_adapter.py` builds hash-bound QA source rows from the
actual local NExT-GQA CSV for the three selected rows; the aggregate-only
`audit_nextgqa_actual_qa_source_binding.py` checks those rows against the CSV,
worksheet, and caller-supplied lineage. A local run verified 3/3 selected source
rows and 3/3 worksheet question matches (5,553 CSV rows; 5,552 uniquely
matching gold choices, one ambiguous row), with zero files written and zero
model/score calls. This is a source-consistency check, not independent
authentication of the caller-supplied lineage, tokenization evidence, human
claim review, or execution authorization. Because the other upstream reports
remain caller-supplied, this success state is not a production preflight or
execution authorization.

The result manifest records requested/reviewed/supported/assigned/scored counts,
all nine requested forwards and attempted/completed forwards, sanitized
failures, runtime, peak VRAM, seed, software versions, hardware, precision,
and exact source/configuration hashes. Local artifacts use mode 0600 and unique
exclusive-create paths. Summary output contains only aggregate estimates,
counts, safe hashes, and failure codes. Per-video derived scores remain in a
separate local mode-0600 artifact. Do not publish these protected-data-derived
results to GitHub while the local-only restriction applies.

## Current readiness

No item has a completed human review (**0/3**). The available review HTML does
not record or validate the player-to-PTS calibration, and no browser is
currently available in the execution environment. The CPU assignment planner
is an unvalidated planning component only; its caller-supplied audit/calibration
reports are not an inference authorization. There is no production claim-linked
scorer or validated all-cohort launch gate. Therefore this protocol is not yet
scoreable, and the current answer-margin runs that mask NExT-GQA QA-gold spans
remain a different, auxiliary estimand.
