# Exploratory saved-claim video-content dependence (2026-10-21)

## Question and rationale

The preceding three-row diagnostic found near-zero average interaction between
global visual-key attention bias and temporal-block reversal (+0.00165
nat/token, mixed signs). It tests a treatment interaction, not whether claim
likelihood depends on the video's visual content at all. A separate, fixed
exploratory contrast will compare each saved sentence's teacher-forced mean
token log likelihood under its original sampled video and a same-shape,
constant mid-gray video. The question is whether these model-generated
sentences are conditionally sensitive to the sampled image content.

## Frozen design

- Cohort: all three rows in the two already frozen local validation claim
  packets used by the 2026-10-21 temporal-permutation diagnostic; no row
  replacement. The first smoke row and two remaining rows are joined by their
  fixed packet hashes and row tokens.
- Input: same decoded 16 frames, source prompt, model/tokenizer revision,
  deterministic re-tokenization, frame timestamps, and processor geometry.
- Treatment: replace each decoded RGB frame with constant `(127,127,127)`
  pixels at the original dimensions, then run the same pinned video processor.
- Conditions: original video and constant-gray video, both with unmodified
  attention; exactly two teacher-forced forwards per row.
- Primary finite-cohort description: per-row `mean_logprob(original) -
  mean_logprob(gray)` in nat/token, plus all three row values, arithmetic mean,
  median, and range. No population interval or significance test for this
  selected three-row development cohort.
- No outcome-dependent changes, subgroup search, retries with substituted
  rows, or head/layer/alpha tuning. Preserve failed receipts and report failures.

## Interpretation and limits

Positive contrasts mean the model assigns greater conditional likelihood to
the re-tokenized saved sentence with the original sampled video than with a
constant-gray video. This measures sensitivity to image content under a
severe, out-of-distribution input replacement. It does not verify that the
claim is true, that the relevant evidence supports the claim, that the
generated text used the evidence causally, or that explanation faithfulness
improved. Saved original emission token IDs were not retained. The cohort was
selected for validation/development and has only three rows. Human review is
waived as a collection step for this exploratory local experiment, but no
human judgment will be recorded or inferred. Rights remain unverified; all
inputs and receipts stay local.

## Execution gate and next action

Reuse the existing private-packet, prompt reconstruction, exact video digest,
PTS checks, frozen model revision, resource guards, and target-scoring helper.
First add CPU tests for the gray-frame transform and condition invariants,
then run one fixed-row GPU smoke (2 forwards). Only if both conditions score
and all nonvisual prompt/metadata fields match, run the two remaining fixed
rows (4 forwards). Record actual runtime, resource measurements, output
hashes, and failures. This does not unlock the separately gated formal
human-validated faithfulness baseline.

## Execution and verified result (2026-10-21)

Implementation: `scripts/run_saved_claim_video_content_dependence.py`; summary
analysis: `scripts/analyze_saved_claim_video_content_dependence.py`. The runner
reuses the frozen packet reader, exact prompt reconstruction, video byte/PTS
checks, pinned checkpoint revision, visual-token alignment and per-forward
resource guards. The gray branch reconstructs all 16 RGB frames at the same
dimensions, then sends them through the same processor. It asserts matching
prompt IDs, video grid/timestamps, tensor shapes and every nonvisual tensor;
processor-expanded video pixels must differ. Only baseline attention is used.

CPU validation passed **6/6** tests across the gray transform, contrast
arithmetic, receipt completeness, unique-row requirement, and rejection of
arithmetic mismatches. `py_compile`, CLI help, and scoped whitespace checks
passed. The fixed three rows ran on the RTX 5070 Ti using the existing pinned
Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
4-bit weights, BF16 compute, and eager attention. Each invocation completed
exactly 2/2 forwards; total **6/6**, no failed row or substitution. Elapsed
per row was 17.36, 21.75, and 22.40 seconds; peak allocated VRAM was 7.33,
7.23, and 7.33 GiB. Each run recorded approximately 14.66 GiB free VRAM and
13.69–13.75 GiB available host memory before loading.

| Fixed row | Claim tokens after deterministic re-tokenization | Original minus gray (nat/token) |
|---|---:|---:|
| Smoke row | 7 | +0.64580144 |
| Remaining row 0 | 13 | +0.39277174 |
| Remaining row 1 | 15 | +1.62991402 |

The finite-cohort mean is **+0.88949573 nat/token**, median **+0.64580144**,
range **[+0.39277174, +1.62991402]**; all three row differences are positive.
There is no population confidence interval or hypothesis test: these are three
selected development rows, not an independent representative sample. The
runner-produced receipt hashes are respectively:

- Smoke: `219528e0d10c78cb46ed73fd94a85e7a0613d270bcae9cec67cd7fb614aaaf7d`
- Remaining 0: `27f295f94eb3b57279fab60111e90f1de1194847b3e581660ebd13308c1467ed`
- Remaining 1: `2ddb914560b2a08360957d357fa79cf8d30c2108641a9d9b31564dd393fd9e4c`

The summary is reproducible with:

```bash
.venv/bin/python scripts/analyze_saved_claim_video_content_dependence.py
```

Primary-agent provenance checks independently recomputed all three receipt
contrasts, verified distinct row tokens, packet SHA-256 values, row and saved
claim digests, and actual local video-byte hashes against both the frozen
packets and receipts. An independent auditor reproduced all branch token means
and contrasts, confirmed the 2/2 forward counts, 2,112 visual keys and expected
zero baseline intervention hooks, and found no execution or arithmetic issue.
That audit notes that the receipts bind the model by pinned revision but do not
include hashes of each weight shard; the result is therefore not described as a
byte-level checkpoint audit. Inputs and receipts are mode `0600` and remain
local; no raw claim text, question IDs, media, or responses were published.

## Interpretation, faithfulness status, and next action

Observation: for these three fixed saved sentences, Qwen assigned greater
conditional likelihood under the original video than under same-size,
constant-gray frames, across all three rows. This is evidence of model
sensitivity to the sampled visual input under this intervention. The gray
replacement is a severe out-of-distribution change; pixel statistics and the
video representation also change. The saved claim strings were deterministically
re-tokenized because original generation token IDs were not retained. The
claims and cohort are model/development-selected. Therefore the result does not
establish truth, semantic support, causal use of a particular evidence frame,
or improved explanation faithfulness. It does not alter the earlier
near-zero/mixed global-attention-bias × temporal-order interaction.

Faithfulness progress remains **0/3 direct claim-faithfulness scores**. Human
reviews were not performed and are not represented as complete; the user's
instruction waived them only as a collection prerequisite for local exploratory
experiments. Verified clock/frame pairs and prediction-link reviews remain
0/15 and 0/3. There is still no direct `DeltaF`, interval, or confirmatory test.
Dataset/model rights remain unverified; all protected inputs and derived scores
remain local under the user's local-only authorization.

Next, keep the formal direct-score branch from inferring semantic alignment
from this result. The scientifically useful follow-up is a separately frozen
content-preserving control that can distinguish dependence on visual structure
from the distribution shift caused by blanking every frame. Predefine the
condition and estimand before looking at its scores, retain the same three rows
with no replacement, and label it exploratory. In parallel, the simplified
local review artifact remains available if a reviewer later chooses to provide
actual judgments; it is not a blocker for this exploratory diagnostic.
