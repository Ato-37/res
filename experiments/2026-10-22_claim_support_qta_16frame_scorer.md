# Claim-support QTA 16-frame scorer GPU wiring smoke (2026-10-22)

## Purpose

Prepare a bounded, post-hoc exploratory scorer for the current 3-video,
16-frame generated-claim review artifact. The scorer must bind a future local
review response to the exact artifact and source packet, then compare claim
mean log probability after equal-budget reviewed-support versus matched
non-support frame-group pixel masking, with and without the frozen QTA
attention bias. No visual attention keys are deleted. A synthetic frame
mapping is allowed only to validate GPU wiring; its
scores are not evidence.

## Frozen design and implementation

The protocol is recorded in
[`../docs/claim_support_qta_16frame_protocol_v1_exploratory.md`](../docs/claim_support_qta_16frame_protocol_v1_exploratory.md).
It matches the current artifact's 16 sampled frames and v3 sequence-complete
claim-generation packet. The primary descriptive contrast is
`DeltaD = (LP(control-pixel-mask,QTA)-LP(support-pixel-mask,QTA)) -
(LP(control-pixel-mask,baseline)-LP(support-pixel-mask,baseline))`, in nats per claim
token. QTA selects the top one of eight temporal groups from question-token
attention averaged over layers 5–16 and applies `ln(2)` visual-key bias at
those groups. The support and control masks have equal frame-group/key budgets;
the control excludes both support and QTA groups and minimizes rank distance.
Rows without a feasible control remain failures; there is no replacement.

Implementation is in
[`../scripts/run_claim_support_qta_16frame_exploratory.py`](../scripts/run_claim_support_qta_16frame_exploratory.py).
It source-binds the artifact, packet, frozen manifests, dataset inputs, video
digests, frame timestamps and claim/question hashes before loading the model;
re-decodes exact media before inference; checks prompt, processor geometry and
the actual intervention masks; and writes only a new owner-only local receipt.
The full run requires a response bound to the exact current artifact. A
bounded no-follow reader accepts local Windows Downloads/DrvFS permission
modes while retaining regular-file, size, stable-read, and downstream
fingerprint checks. Output receipt permissions remain `0600`.

## Initial smoke and independent audit correction

Normal Python: repository `.venv`, torch 2.11.0+cu128, CUDA 12.8. An
individually elevated preflight saw an NVIDIA GeForce RTX 5070 Ti with 14.29
GiB free VRAM and 13.71 GiB host-available memory. The first implementation
smoke `_02` completed mechanically in 43.602 s, with 5/5 forwards and 7.469
GiB peak allocated VRAM, but independent source audit found an estimand bug:
all four score branches applied black pixel masking and `-inf` visual-key
deletion. Its `DeltaD=+0.010701` nat/token belongs to a different composite
intervention and is invalid for the frozen pixel-mask-only protocol. Do not
use it as a result. Receipt SHA-256
`9fadbf96e8f08852bad5456294570f6b721d94921cb867ce3dad60d3aaab3ebe` was
preserved locally with mode `0600`.

The code was corrected so baseline uses no attention-mask edit and QTA adds
only the frozen `ln(2)` key bias; neither score branch deletes keys. The adapter
now uses the intersection of frame choices across unanimous `supports`
reviewers, and rows with no shared frame remain ineligible. The synthetic
smoke constructs an in-memory synthetic response solely to validate exact
artifact fingerprint/schema; it is never written or counted as human review.
Before model loading, CPU preflight re-encodes each thumbnail from the exact
decoded model-input RGB frame and requires exact JPEG-data equality.

The corrected `_03` GPU smoke completed the selector plus four baseline/QTA ×
pixel-mask forwards on one real local video in 38.354 s. QTA branches biased
266 keys, all branches deleted zero keys, and each branch passed the layer
5–16 forward-hook audit. Peak allocated VRAM was 7.468 GiB. The private
receipt `data/private_nextgqa/claim_support_qta16_synthetic_smoke_20261022_03.json`
has SHA-256
`9fcaca3c5cbe25cc515470808ae95f30507dd492b0d6cc424e000f05d96f3927` and mode
`0600`. It records source hashes, Python/package versions, model config hash,
pinned checkpoint revision, and a snapshot file-inventory hash (names/sizes;
model weight bytes are not rehashed).

The `_03` smoke's synthetic group overlaps QTA. Unlike `_02`, the QTA bias
remains active in both pixel-mask branches. Its synthetic `DeltaD=+0.005756`
nat/token is only a wiring diagnostic and is not a faithfulness result.

An initial smoke receipt (`data/private_nextgqa/claim_support_qta16_synthetic_smoke_20261022_01.json`,
SHA-256 `beb1cf7b77395d9d26d3a5d02a6d9f6b12988b3847886e3eb70dfcfd586cccb0`)
failed in a forward-hook audit because the hook deleted its positional-args
reference before returning it. The failure was preserved, the hook corrected,
and the second smoke used a new output path and passed.

## Checks and remaining evidence gaps

Fourteen focused `unittest` cases passed, covering the response adapter,
fingerprint and timestamp validation, cross-reviewer frame intersection,
equal-budget control selection, failure retention, contrast arithmetic, and
safe response-file reading.
`py_compile` and scoped `git diff --check` passed. No external dependency was
installed. The current artifact still has 0/3 human-reviewed claims and no
artifact-bound response was found in the prior local inventory. Therefore the
full real-label score was not run: direct claim-frame dependence remains
unmeasured (0/3); formal direct faithfulness review remains 0/25. The GPU
smoke is not a substitute for either.

## Final receipt metadata correction (2026-10-22)

Independent audit then found two receipt-level omissions: `_03` still described
the outcome as key ablation and its source list omitted five runtime helpers.
The runner now records the pixel-mask estimand explicitly and hashes 14 source
files, including prompt, masking, thumbnail, response, and output-guard helpers.
An immutable new smoke `_04` supersedes `_03` for provenance. It completed 5/5
forwards on RTX 5070 Ti in 37.970 s; peak allocated VRAM was 7.468 GiB. The
receipt SHA-256 is
`77440199f1393622c1976c3f257a67672de46ce55123cc59baf2bf766777bb84`, mode
`0600`. Independent receipt inspection recomputed that all 14 source hashes
match current files, verified the pixel-mask-only estimand string, 0 deleted
keys, package/config/snapshot-inventory metadata, and false direct-faithfulness
flags. The synthetic `DeltaD` repeats +0.005756 nat/token and remains purely a
plumbing check. The checkpoint snapshot inventory records 16 relative file
names and sizes; weight bytes are not rehashed, so exact checkpoint content is
supported by the pinned model revision/config and inventory but not a complete
per-weight digest.

## Final decode-runtime manifest extension

A follow-up audit noted that the decoder's PyAV/Numpy runtime versions and the
presentation-time helper source were not yet listed. The runtime manifest now
includes those versions and source. A final new receipt `_05` supersedes `_04`
for the current code: SHA-256
`ccb20fb5263e3ed539b44e37ba43acc7fc1db3d3e61c0a047537b7998153c787`, mode
`0600`. It completed 5/5 forwards in 34.519 s, peak allocated VRAM 7.468 GiB;
all 15 source hashes independently match, all eight listed runtime package
versions are present, the emitted estimand says pixel masking with no visual
key deletion, and all four branches report zero deleted keys. The repeated
synthetic `DeltaD=+0.005756` nat/token remains only a deterministic wiring
diagnostic. Weight bytes remain unhashed; see the inventory limitation above.

## Next action

When an exact-artifact response JSON is locally available, run the full
three-row cohort using `--mode full --response <local-response.json>` and a
fresh unique private output name. Keep all failures and review-ineligible
rows in the denominator; score only unanimously support-marked rows. Report
finite-cohort paired values only; three previously explored videos do not
support population inference, and even a successful result is exploratory,
limited to model dependence on selected displayed frames. Do not publish the
private video-derived receipt. Rights status remains unverified; all
materials and outputs remain local-only under the user's instruction.
