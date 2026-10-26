# Human-review waiver and next experiment selection (2026-10-21)

## User direction

The user instructed that human checks are excessive and asked to proceed as if
passed. For execution planning, collection/attestation is waived for exploratory
local work. The underlying facts remain unchanged: no current-claim human review,
frame/clock verification, or prediction-link review was performed. Do not encode
an inferred human judgment as an observed review or as evidence of claim truth.
Protected inputs and derived scores remain local; rights status is unverified.

## Current evidence and endpoint audit

The current review artifact contains three generated claims, but the two available
Downloads exports bind only to the older v14 artifact (0/3 item IDs match current
claims). Existing current-cohort GPU branches have no new run available:

- Benchmark-QA interval versus equal-count temporal mask: 3/3 triplets, 9/9
  forwards; finite-cohort mean control-minus-interval gold-vs-best-foil margin
  **−0.250 nat**. This is auxiliary QA-interval dependence, not generated-claim
  faithfulness. The branch is complete and closed.
- Same-model claim interval localization: localization validation failed before
  scoring; 0/9 scoring forwards and no effect estimate. The frozen branch is
  closed to retries.
- Cross-scale 3B localization: final one-item smoke yielded 0/1 valid intervals
  and 0/9 scoring forwards. The feasibility branch is closed.
- QTA generated-text v2: its finite three-row proxy run is complete and closed;
  it measures dependence on benchmark timing, not claim support or faithfulness.

No new GPU experiment was run at this checkpoint. The existing RTX 5070 Ti and
CUDA-enabled `.venv` were verified through individually elevated read-only
execution: PyTorch 2.11.0+cu128, CUDA available, 15.74 GiB free VRAM, and
approximately 13.97 GiB host memory available. The normal sandbox still cannot
see the GPU.

## Faithfulness progress

**Direct faithfulness: 0/3 claims scored; no direct effect estimate.** Formal
review counts remain 0/3 claims, 0/15 frame/clock pairs, and 0/3 prediction links.
There is no direct score, confidence interval, or hypothesis test. The benchmark
interval result above remains auxiliary only; do not count it as claim faithfulness.

## Next experiment to implement

Create a separately versioned, explicitly exploratory visual-dependence test
that scores a deterministic re-tokenization of each saved generated-claim
string under the reconstructed question/video prompt. The original emitted
token IDs were not saved, so this is not the exact generation token sequence.
Compare natural video order with a deterministic temporal permutation that
preserves the same decoded frame pixels and frame count. Apply
a fixed internal visual-attention intervention and its baseline to both inputs,
then estimate the finite-cohort difference-in-differences in mean claim-token
log likelihood. This removes the need for human-selected claim intervals and
avoids reusing the closed QA-margin endpoint. Freeze the cohort, prompt, frame
permutation, key mapping, intervention strength, tokenization, and failure rules
before any scores. Require an independent implementation audit and one-item GPU
smoke before scoring the full fixed cohort; retain all three rows with no
survivor-only mean.

Interpretation must remain narrow: the result would test whether a fixed Qwen
internal visual-attention intervention changes the model's sensitivity to frame
order when predicting a re-tokenized saved sentence. It would not show that the text is
true, that its claims are grounded in the changed frames, or that general
explanation faithfulness improved. Temporal permutation can itself produce an
out-of-distribution video. If the source prompt cannot be reconstructed and
hash-bound, or the saved claim cannot be re-tokenized deterministically under
the pinned tokenizer, do not substitute a different prompt or claim.

## Next action

Implement and CPU-test the new scorer/protocol, audit its visual-key mapping and
teacher-forced target alignment independently, then run one preselected real-video
GPU smoke. Only proceed to the three-row exploratory estimate if the smoke
completes all conditions and the frozen inputs match. Do not rerun a closed branch
or reinterpret its outcome as direct faithfulness.

## Implementation and pre-smoke check

Added `scripts/run_saved_claim_frame_permutation_attention_smoke.py` and its
focused tests in `scripts/test_run_saved_claim_frame_permutation_attention_smoke.py`.
The runner fixes the first row of the existing smoke packet, verifies the source
video digest and all 16 saved sample PTS values, reconstructs the video-first
question-plus-five-choice generation prompt, and scores the saved parsed claim
string after deterministic retokenization. It reverses temporal-patch-sized
blocks of decoded PIL frames, rebuilds both processor batches, and checks that
text IDs, video grid, timestamps, shapes, and nonvisual tensors remain equal
while processed visual pixels differ. Four teacher-forced forwards compare
natural/permuted order with baseline and a fixed `ln(2)` bias on all visual keys
in Qwen layers 5–16. The receipt explicitly records that the re-tokenized IDs
are not original emission IDs and that no human review occurred. It writes a
new mode-0600 private receipt and excludes raw claim text, token IDs, question
ID, and paths.

Primary-agent verification: `py_compile` passed, focused tests **7/7** passed,
CLI help passed, and scoped `git diff --check` passed. The initial unittest
invocation used the wrong module search path and failed to import the test; the
corrected invocation used `PYTHONPATH=scripts` and passed all seven tests. This
was an invocation error, not a test failure. At that pre-smoke checkpoint, no
GPU forward had run yet.

The pre-smoke runtime check through individually elevated read-only execution
found CUDA available on the RTX 5070 Ti with 15.74 GiB free VRAM and about
13.97 GiB host memory available. The planned workload was one offline 4-bit
BF16 Qwen2.5-VL-7B model load and exactly four one-row teacher-forced forwards.
Guards require 12 GiB free VRAM and 8 GiB host memory before load and 2 GiB
free VRAM before each forward.

## Verified one-row GPU smoke

The first run completed 4/4 forwards in 21.86 seconds and produced an
interaction of **+0.01277978 nat/token**. Independent audit reproduced all four
branch means: original/baseline −0.28595468, permuted/baseline −0.30040835,
original/bias −0.27778989, and permuted/bias −0.30502334 nat/token. The
natural-minus-permuted gap was +0.01445367 under baseline and +0.02723345 with
visual bias. The first receipt did not include runtime hook-entry counters, so
it is retained as a preliminary run and not the final audited artifact.

To close that provenance gap, the runner now observes the attention mask after
each bias hook and checks every selected visual-key/query cell equals `ln(2)`.
The revised focused suite passes **8/8**. A second run completed 4/4 forwards
and reproduced the same interaction exactly. Its owner-only receipt is
`data/private_nextgqa/saved_claim_frame_permutation_attention_smoke_20261021_02.json`,
SHA-256
`ef928137172477dde4f061d9ad7c34e2e168163d6f8ecae4d9c7177065eadec3`.
It records 2,112 visual-token positions, one invocation per layer in all 12
biased layers, zero bias-hook invocations in baseline branches, the current
input-packet hash, and matching runner/helper/model-config hashes. Independent
arithmetic gives the same +0.012779780362117354 nat/token interaction. Current
packet and source/config hashes were independently recomputed; the receipt is
mode 0600 and 7,620 bytes.

This remains a **single-row sensitivity smoke**, not direct faithfulness. The
target consists of seven tokens deterministically re-tokenized from the saved
parsed sentence; the packet does not preserve original emission IDs. Reversing
temporal blocks changes order while preserving pixels and regular timestamps,
which is a deliberate chronology perturbation and may be out of distribution.
The intervention biases all 2,112 visual-token positions by `ln(2)`, so it is a
coarse global treatment, not selective claim evidence. There is no CI, test, or
population estimate for one row. Human review remains unperformed; the user's
waiver applied to this exploratory run. Rights remain unverified and all
protected artifacts remain local.

## Next action

Do not call the positive one-row interaction a faithfulness improvement. The
next useful step is an independently audited multi-row extension over the
remaining two fixed candidates under the same frozen text-re-tokenization,
frame permutation, and global visual-key bias, with no row replacement. Report
all requested/effective counts and each finite-cohort row; do not compute a
population interval or significance claim from this three-item development
cohort. Direct claim-grounding/faithfulness evidence remains 0/3.

Before examining the remaining rows, the CLI was extended with an explicit
`remaining` packet and zero-based row index. It does not inspect outcomes to
choose the row. Each invocation requests one row/four forwards; the two fixed
remaining rows will be run separately and combined only if both receipts
complete. The focused suite now passes **9/9**, including exact row selection
and out-of-range rejection. The first audited runner bytes were preserved at
`scripts/run_saved_claim_frame_permutation_attention_smoke_20261021_v1.py`
(SHA-256 `cc4b956713acec60707ce36d57d34350b19e221fbeacda82c31b430f63c33b2b`),
matching the source hash in the already audited 02 receipt. The extended runner
SHA-256 is `df8ddf525b44fc19aed29f199e656607d4394511ae092299547d2f73f8f17ba6`.
Planned remaining workload is two sequential local 4-bit BF16 model loads and
eight teacher-forced forwards in total; live GPU/RAM guards are checked before
each process and each model call. The two results, if complete, remain a
post-hoc exploratory finite-cohort description because the smoke was already
observed; no confirmatory or population claim will be made.

The first remaining-row attempt stopped during processor preflight before any
model load or scoring forward (0/4). A CPU reproduction isolated the strict
metadata check: Qwen returned a floating frame count of
`15.999999999999998` for 16 actual decoded frames. This was floating-point
rounding, not a frame or video mismatch. The validator now accepts only a
finite count within `1e-6` of 16, while still requiring the exact frame-index
sequence; the receipt also preserves the observed floating value instead of
truncating it. The focused suite passes **10/10**, including an explicit
roundoff-acceptance and wrong-count-rejection test. No score was inspected or
used to change the protocol. Retry this same fixed row once to a fresh output;
the failed 0/4 receipt remains preserved and is excluded from every estimate.

## Remaining-row results

The fixed retry for remaining packet row 0 and the fixed run for row 1 both
completed 4/4 forwards. Each receipt verifies exactly one invocation in each
of the 12 biased layers and the expected `ln(2)` values across every selected
visual-key/query mask cell. The failed first attempt remains 0/4 and is not
counted. The three valid rows (original smoke row plus these two preselected
remaining rows) have interactions:

| Fixed row | Re-tokenized target length | Bias × frame-order interaction (nat/token) |
|---|---:|---:|
| Smoke row | 7 | +0.01277978 |
| Remaining row 0 | 13 | +0.02259141 |
| Remaining row 1 | 15 | −0.03043045 |

The exact finite-cohort arithmetic mean is **+0.00164692 nat/token**, median
**+0.01277978**, and range **[−0.03043045, +0.02259141]**. Two rows are
positive and one negative; no inferential test or interval is appropriate for
this selected three-row development cohort. The near-zero mean and opposing
row signs do not support a consistent intervention effect even on this narrow
text-likelihood proxy. The independent audit confirmed all three receipts,
token-logprob arithmetic, per-layer hook counts, visible-zero-to-`ln(2)` mask
change, packet/source/checkpoint hashes, source video digests, and local video
bytes. It reproduced the finite mean **+0.0016469158664221089 nat/token**.
The failed first attempt was 0/4 and is excluded; its one-time retry binds the
identical row token. No audit issue remains for this descriptive proxy result.
Direct claim-faithfulness remains **0/3**; this experiment neither completes
human reviews nor evaluates semantic support.
