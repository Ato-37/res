# QTA v4 mask-specificity implementation and execution

## Purpose and design

This is a post-result exploratory diagnostic for the test-associated 147-row
QTA v4 cohort. It compares the annotation-union mask's fixed answer/foil margin
contrast to one preselected nonwrapping translated mask with the same number
and run shape of temporal groups and the same number of processor visual-token
positions. It is not a direct measure of explanation faithfulness, does not
use reviewed claims, and cannot establish causal prediction use or
generalization. Full protocol: `2026-10-18_qta_v4_mask_specificity_design.md`.

## Implementation and checks

Implemented `scripts/run_nextgqa_qta_v4_mask_specificity.py`,
`scripts/qta_v4_mask_specificity_utils.py`, and two focused test files. The
runner binds the frozen v4 score/plan/census/preflight/smoke, ordered row
bindings, dataset and source hashes, processor manifest, and this protocol.
All row-level receipts remain under `data/private_nextgqa/` at mode 0600.

The current source passed `py_compile`, `git diff --check`, and 15/15 focused
tests. Independent source audits found and closed row-binding and smoke-gate
issues; an additional audit approved the list/tuple normalization fix. No
protected inputs were read by the auditors.

While the full run is executing, a separate audit utility,
`scripts/audit_qta_v4_mask_specificity_result.py`, was implemented to recompute
row-level contrast arithmetic and finite-cohort summaries independently. It
also verifies provenance, complete denominators, exact branch counts, and
owner-only receipt properties. Its tests cover arithmetic corruption, missing
placebo values, Boolean counters, missing summary/denominator mismatch,
negative absolute errors, symlink and permissions, and same-file-snapshot hash
verification. The independent auditor approved this utility; the combined
focused test suite now passes 22/22 tests. This audit tool is separate from the
already-running scorer and does not affect its frozen source hashes.

## Verified preflight

CPU preflight `_02` completed **147/147** rows in **529.4 seconds**. It found
**140** eligible translations and **7** structurally unavailable rows. An
independent verifier reproduced source/data hashes, frozen v4 receipt
validation, plan-order identity bindings, and 147 unique row bindings. The
private receipt is mode 0600, SHA-256
`c7bf4a8ef382efe69e7150d485ec3f25e19b1e120c81351c4d79b18138463dba`.

The earlier `_01` preflight had the same counts, but a smoke attempt using that
code version failed closed before any forward due to JSON-list/runtime-tuple
representation mismatch. That failed smoke receipt records 6 requested, 0
attempted, and 0 completed; it produced no estimate. The corrected implementation
normalizes only the sequence fields and has a regression test. It writes new
`_02` receipts rather than overwriting the earlier outputs.

## Verified GPU smoke and full run

The corrected six-forward smoke completed **6/6** branch forwards on one
eligible row. Answer/foil matched the frozen v4 receipt and the baseline-full
margin absolute error was **0.0**. The private smoke receipt is mode 0600,
SHA-256 `faedae20b8847f78c2125e9b720e419a71627cd42cea1ef397bd6cf17bce36ae`.

The full GPU run began 2026-10-18 16:18 JST on the RTX 5070 Ti after observing
1,188 MiB / 16,303 MiB GPU memory use and 14 GiB host memory available. The
planned workload is **868 forwards**: four annotation conditions for all 147
rows plus two translated-mask conditions for each of 140 eligible rows. Prior
v4 execution (588 forwards in 2,539 seconds) gives an approximate 3,749-second
GPU computation estimate, excluding model load and CPU decoding. This remains
an estimate. The full run completed all **868/868** forwards for **147/147**
rows, with **140** placebo-eligible and **7** structurally unavailable rows.
The runner-reported elapsed time was **3,456.826 seconds** (57m36.8s).

The independent result auditor passed all frozen plan/data/source/receipt,
permission, denominator, branch-count, row-binding, and arithmetic checks. The
private receipt is mode 0600 with SHA-256
`8aeae3e9da1ed1257d02fa3a3304eacbfd61023f8714367e0a76976a08ffa0c3`.
It records RTX 5070 Ti, CUDA 12.8, 14.657 GiB free VRAM before model load, and
11.929 GiB peak allocated VRAM. The auditor marked `faithfulness_result=false`.

Finite-cohort descriptive results (no inferential test or population interval):

| Measure | n | Mean (nat) | Sample SD | Range (nat) | Positive / negative / zero |
|---|---:|---:|---:|---:|---:|
| `D_annotation_union`, all rows | 147 | +0.006803 | 0.137073 | [−0.375000, +0.375001] | 74 / 68 / 5 |
| `D_annotation_union`, eligible rows | 140 | +0.008929 | 0.137947 | [−0.375000, +0.375001] | 72 / 63 / 5 |
| `D_placebo_translation`, eligible rows | 140 | −0.016071 | 0.153528 | [−0.625000, +0.375000] | 65 / 68 / 7 |
| Paired difference `D_annotation_union − D_placebo_translation` | 140 | +0.025000 | 0.119200 | [−0.250001, +0.250001] | 69 / 49 / 22 |

The observed paired mean is small relative to the rowwise spread and has no
predeclared practical threshold in this exploratory diagnostic. It does not
establish a nonzero effect, annotation specificity beyond the selected
translation, or faithfulness. The cohort is test-associated with unknown prior
exposure; annotations are unreviewed; the placebo can overlap real evidence and
location/OOD effects remain plausible. Rights remain unverified and all
protected material stays local.

## Limits and next action

The independent human-free falsification checks are complete in
`scripts/audit_qta_v4_mask_specificity_exhaustive.py` and
`scripts/test_qta_v4_mask_specificity_position_wiring.py`; they use synthetic
geometry/processor fixtures and do not alter the frozen scorer. Human-review
progress remains separate: 0/3 reviewed claims, 0/15 verified frame pairs, and
0/3 prediction-link judgments. A fresh non-authorizing candidate inventory is
documented separately in `2026-10-18_human_review_candidate_inventory.md`.
Until trusted role assignments and the witnessed pre-open handoff exist, the
review branch remains at preparation only. Rights remain unverified; protected
materials and run receipts remain local-only.

## Primary follow-up arithmetic audit (2026-10-18)

The primary agent independently read the mode-0600 aggregate receipt and
recomputed all three contrasts directly from its stored condition margins,
without using the project summary helper. The receipt hash still matches the
value above. It contains 147/147 rows and 868/868 requested, attempted, and
completed forwards; all six branch counters independently match 147 rows for
the four base branches and 140 rows for the two translated-mask branches.
Maximum absolute difference between stored and independently recomputed
`D_annotation_union`, `D_placebo_translation`, and paired specificity values
was **0 nat** for each. The independently recomputed means and sample SDs
match the reported values: all-row annotation-union D **+0.006803 / 0.137073
nat** (n=147), translated-mask D **−0.016071 / 0.153528 nat** (n=140), and
paired specificity difference **+0.025000 / 0.119200 nat** (n=140).

A full current-tree provenance audit was attempted but correctly failed closed:
the current worktree no longer matches the run-time source freeze for
`scripts/run_nextgqa_qta_v4_mask_specificity.py` and
`scripts/run_nextgqa_validation_qta_annotation_proxy.py`; all four pinned data
files still match. The `_04` private source archive matches the historical
validation-helper hash, but it does not contain the specificity runner, so the
complete source set cannot be reconstructed from that archive. The score
receipt and logged run-time source hashes remain intact, and the earlier
independent run-time audit is recorded above, but this follow-up is explicitly
an aggregate arithmetic/count audit, not a fresh end-to-end provenance audit.
No uncertainty interval or test was added. Direct generated-claim faithfulness
remains **0/3 reviewed, 0/15 verified frame correspondences, 0/3
prediction-link judgments, and 0/3 direct scores**. Keep this note and all
derived scores local-only.

## Historical source archive coverage audit (2026-10-18, local-only)

To resolve the noted runner-source gap, the primary agent checked the private
aggregate receipt and each of the four mode-restricted source archives without
printing any record-level content. The aggregate receipt SHA-256 is
`8aeae3e9da1ed1257d02fa3a3304eacbfd61023f8714367e0a76976a08ffa0c3`; its
27 recorded run-time source hashes remain intact. Current-tree comparison
matches **25/27** sources; the two nonmatching files are
`run_nextgqa_qta_v4_mask_specificity.py` and
`run_nextgqa_validation_qta_annotation_proxy.py`.

Historical source-hash coverage by archive was `_01` **11/27**, `_02` **21/27`,
`_03` **21/27**, and `_04` **24/27**. All four archives contain the historical
validation-helper source, but none contains the historical
`run_nextgqa_qta_v4_mask_specificity.py` runner. The `_04` archive itself has
SHA-256
`cd2b20b71c99d28ce65a4c375568d3a999ebe8c357bc800bec8f1372c245fbd8`.
Therefore the scoring runner source cannot currently be reconstructed from
these archives. The run-time hash ledger and aggregate result remain available,
and the earlier independent arithmetic/count audit still reproduces all
reported contrasts exactly, but a fresh full source-set provenance audit is
not possible. This is an audit limitation, not evidence that the score values
are numerically wrong; keep the result exploratory and do not elevate its
claims. Private receipt/archive paths and derived row data remain local-only.
