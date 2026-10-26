# Local-history-screened val claim/QA-interval key-bias pilot (2026-10-21)

## Research question and status

On a new, locally history-screened three-video validation cohort, does a fixed
internal visual-key bias change saved-claim likelihood sensitivity to the
machine-annotated NExT-GQA QA interval more than at baseline, and more than an
equal-budget disjoint key sham? This is an exploratory proxy run on
model-generated, unreviewed claims. The target endpoint is not direct semantic
faithfulness: the dataset interval is annotated for the question/answer and
has not been verified as support for the generated claim.

## Frozen sample, implementation and checks

The outcome-blind val candidate lock
`data/private_nextgqa/qta_candidate_lock_val_20261021_1038z_01.json` (SHA-256
`fad5997af14b8cf50415de7f6bcd5cc69681c378441660aecb1854c5872633f4`)
contained 30 unique candidate media rows after exclusion of 1,528 media
hashes in the available private JSON history. The claim generator also
excludes local `data/nextgqa_samples_*.json` media and requires a valid
equal-count non-overlapping interval/control geometry. In the frozen
outcome-blind order, four rows were scanned to obtain three that passed this
CPU geometry check: one had no valid control pair, with zero decode failures
and zero timing rejections. The three selected hashes were absent from the
lock's known-prior set. This only establishes exclusion against the available
local inventory; history is incomplete, so exhaustive novelty is not claimed.

The three fixed rows were generated using NExT-GQA val, 16 frames and
`v2_short`, with Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights and BF16 compute.
The one-row generation smoke and three-row packet each produced structured
claims for all requested rows; the smoke claim exactly matched full row 0.
Full generation peak allocated VRAM was 7.49 GiB. The private full packet
`data/private_nextgqa/claim_generation_val_preflight_full_20261021_01.json`
is mode 0600, SHA-256
`77c4c0c8ac4a9a88bbacadb5fc941bade3b70f49f84a3e722d111e472a2ad0fa`.
Dataset hashes are recorded in that packet. The generator runner hash is
`30b76f7ed27083d6540bf97afc9fa6f3754528917a7dd624ef8cf5d135053c8c`.

The existing six-branch scorer was generalized to accept either a `val` or
`test` packet and to join against the packet's split-specific CSV and gsub
files. The frozen estimand and branch construction did not change. Before
GPU execution, the actual val packet passed a CPU-only three-row preflight:
source video bytes and decoded PTS matched, CSV and gsub records joined,
QA/control/sham group budgets were equal within each row, QA/control mask
processor pixels differed, and all six branch geometries were prepared.
Focused tests passed **10/10**. Python compilation and `git diff --check`
passed. Scoring runner SHA-256:
`46c236779440503607c0c3b0307ab7363b89a09f906a85efc653cdc8f1b396a5`.

Each sample has six teacher-forced forwards: baseline, QA-interval key bias,
and same-size disjoint sham bias crossed with QA-interval versus equal-count
control pixel masking. The fixed contrast is `F = LL(control mask) − LL(QA
mask)`; primary `DeltaF = F_QA − F_baseline`; secondary specificity is
`F_QA − F_sham`. Bias is `ln(2)` on selected visual keys in layers 5–16.
The smoke completed 6/6 forwards. One initial full attempt wrote a private
failure receipt because the smoke path was passed relative to the workspace
instead of the runner's private directory; it failed at receipt validation
before any forward (0/0). Retrying with the smoke basename passed the
hash-bound smoke check and completed all **18/18** full forwards on 3/3 rows.
No row was replaced. The successful receipt binds all five model-weight shard
hashes and the smoke, source, packet and dataset hashes. It is mode 0600.

## Results

| Row | `F_baseline` | `F_QA` | `F_sham` | `DeltaF` (QA−baseline) | QA−sham |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.41109 | −0.35063 | −0.38680 | +0.06046 | +0.03617 |
| 1 | +0.42970 | +0.46498 | +0.38799 | +0.03528 | +0.07699 |
| 2 | −0.05946 | −0.10070 | −0.09084 | −0.04124 | −0.00986 |
| **Mean** | **−0.01362** | **+0.00455** | **−0.02988** | **+0.01817** | **+0.03444** |

The finite-cohort `DeltaF` median is +0.03528, range [−0.04124, +0.06046],
with 2/3 positive rows. The QA-minus-sham median is +0.03617, range
[−0.00986, +0.07699], also 2/3 positive. These are descriptive summaries of
this fixed n=3 sample only: no confidence interval, hypothesis test, or
population inference is justified. The full private receipt
`data/private_nextgqa/claim_text_qa_interval_key_bias_freshval_full_retry01_20261021_01.json`
has SHA-256
`fa55671d2dd449b755d8ea131b222f88c7196e0116798b4c5f7a55b185ccb815`; the
smoke receipt SHA-256 is
`5447dcaa7eda0e8d439fc4749a9e4c37ccfa59a989b336478912032d5df8e65d`.
The primary agent independently recomputed all 18 branch token means and
both contrasts from stored token-level scores, matched row/video/claim hashes
to the exact packet, and confirmed the smoke/full row-0 identity and all five
model shard hashes. Runtime was 122.72 seconds for full scoring; peak
allocated VRAM was 7.39 GiB, with 14.66 GiB free VRAM and 13.35 GiB available
host memory before load. Hardware was an RTX 5070 Ti with CUDA 12.8 and
PyTorch 2.11.0+cu128.

## Interpretation and limitations

The result gives a small, mixed positive proxy signal: for two of the three
newly selected rows, the internal QA-window bias increased the fixed claim's
QA-mask sensitivity relative to both baseline and disjoint sham. The third
row moved in the opposite direction. The mean is +0.01817 nat/token versus
baseline and +0.03444 versus sham; the effect is heterogeneous at n=3. The
same-size control mask and equal-budget sham reduce two simple alternatives,
but do not validate the interval as evidence for the claim. Black masks can
create out-of-distribution inputs, the claim is generated by the evaluated
model, claims were re-tokenized, and global key bias is not claim-specific.
The interaction with temporal-block order on this same cohort was near zero
and mixed (mean −0.00353 nat/token), so a positive QA-mask proxy does not imply
improved temporal evidence dependence. This combined pattern is useful for a
thesis discussion of metric/control dependence, but not evidence that
explanations became more truthful or faithful.

## Faithfulness progress and next action

- Direct semantic claim-to-frame evaluation: **0/3 reviewed, 0/3 direct
  intervention scores**, no direct `DeltaF`, confidence interval or test.
- This GPU result is a machine-annotated QA-interval claim-likelihood proxy,
  not direct faithfulness evidence. Human review remains unperformed despite
  the user's waiver of review as an exploratory run prerequisite; rights
  remain unverified. Inputs and row-level artifacts stay local.
- This is post-hoc relative to other exploratory temporal-order analysis on
  the same three generated claims, and the local-history ledger cannot prove
  all prior exposure. Do not retune the bias on this cohort.
- The next direct step still requires actual claim-specific support spans. A
  useful proxy follow-up, if justified, is an independent same-protocol
  validation cohort with a larger frozen n and no outcome-based selection;
  it must keep the machine-QA interval label explicit and treat the present
  n=3 as a pilot rather than a confirmatory estimate.
