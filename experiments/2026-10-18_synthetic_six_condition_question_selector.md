# Synthetic six-condition question-selector integration (2026-10-18)

## Purpose

Verify that the processor-derived question-token temporal selector, frozen
attention-key bias, baseline-prediction/foil scorer, and `ΔF` arithmetic run
together through the pinned Qwen2.5-VL model. This is a synthetic integration
test, not a real-data or faithfulness efficacy experiment.

## Setup

- Three deterministically generated 32-frame synthetic videos; one fixed
  synthetic QA prompt per row; A–E answer-token IDs re-resolved from the pinned
  tokenizer and checked as distinct single tokens.
- Six branches per row: baseline and key-bias intervention, each with full,
  claim-evidence-black, and matched temporal-control-black inputs. The selector
  is measured once from untreated full-video attention over exact question
  token queries, averaged over layers 5–16, and frozen for all three treatment
  branches.
- The selector chooses `ceil(10% of 16 temporal groups) = 2` complete groups
  per row. The selected keys receive an additive `ln(2)` attention-logit bias
  at the separate final answer-score query in layers 5–16.
- Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights, BF16 compute,
  eager attention. Seed: `20261018`. Hardware: RTX 5070 Ti.
- Requested workload: 3 rows, 18 forwards. The runner writes aggregate-only
  JSON under a mode-0700 directory as mode 0600 and refuses overwrite or a
  parent directory visible to group/other users.

## Verified result

The source-bound run completed **3/3 rows and 18/18 forwards** with finite
outputs. It took **69.94 s**, used **10.83 GiB peak allocated** and **12.44 GiB
peak reserved** VRAM. Each row mapped 9 question tokens and 16 temporal groups;
selected group indices were `[0, 5]`, with 512/4096 visual keys per row.
This is 12.5% of groups/keys after the ceiling rule, not exactly 10%.

The synthetic `ΔF` mean was **−0.1667 nat** (sample SD **0.0722 nat**, n=3).
No confidence interval or hypothesis test was computed. This value only
confirms that the arithmetic path emitted a finite contrast for the synthetic
fixture; it has no research or efficacy interpretation.

The private result JSON has SHA-256
`84654dc3ec53f7f6ad90910e361e89df5d109d3b323bebd49406a8dae01fc02a`.
Independent recomputation matched all **11/11** source hashes, the pinned
checkpoint config hash, and the recorded mode-0600 file/mode-0700 directory.
The exact source hashes, seed, processor fingerprint, model config hash,
requested/effective counts, resource use, and elapsed time are in the local
result at
`data/private_nextgqa/synthetic_six_condition_question_selector_20261018_0457/summary.json`.
That JSON remains local; this log contains only synthetic aggregate values.

Focused selector/hook/output tests passed **18/18** before the final output
cleanup regression was added; the final cleanup suite passed **4/4**. The full
repository script suite passed **1,462/1,462**. Python compilation and
`git diff --check` passed. A prior attempt stopped before any model forward on
a missing synthetic-record variable; it is retained as a separate mode-0600
failure artifact and was not pooled. An intermediate successful run lacked
source hashes and is superseded by the source-bound run above.

## Interpretation and limits

The run verifies the synthetic question-token-to-temporal-group selection,
fixed selector reuse, key-bias hook, six-branch A–E score extraction, and
`ΔF` arithmetic. It does not test the selected groups against a human-reviewed
claim, establish that a masked frame is necessary for the model's complete
prediction, or measure whether faithfulness improves. The selected operation
is a question-conditioned attention intervention; if later validated, a
positive claim must be restricted to increased dependence on independently
reviewed claim-indicated evidence under this exact model, selector, hook, and
mask contrast. Rights remain unverified for the separate protected-data work;
this run itself used synthetic inputs only.

Direct claim-linked faithfulness remains **0/3 independently reviewed,
0/3 scored, and 0/15 browser-to-decoder correspondences**, with no real `ΔF`
estimate, interval, or test. Human review and clock/decoder correspondence
remain external prerequisites. No real-data score was produced.

## Next action

Use the completed CPU/GPU integration to remove selector/scorer wiring as a
future blocker. Continue the human-review and 15-point player-clock/decoder
correspondence handoff. Before any real score, independently freeze the exact
intervention, ablation/control operations, complete cohort, and analysis rule;
then require a run-bound preflight over all review/provenance inputs and every
condition batch before model loading. Do not treat this synthetic run as a
pilot effect or use its `ΔF` value to choose parameters.

## Estimand alignment repair (2026-10-18)

Independent protocol audit found that the earlier attention-selector draft
used black-frame input masks, which did not match the thesis's selected
internal-intervention estimand or the existing internal visual-key ablation
implementation. That draft has been replaced with an internal-ablation protocol:
all conditions use one original full-video processor batch, while evidence and
matched-control temporal groups are ablated only on the final answer-query row
at fixed layers. The protocol now specifies half-open reviewed intervals,
positive-overlap midpoint support cells, multi-occurrence union, deterministic
equal-count temporal control, and a version-independent SHA-256 sham selector.

This is a **protocol repair only**, not a new experiment. The combined baseline,
QTA-Bias, sham, and internal-ablation scorer remains unimplemented and the
protocol is still a draft pending independent audit. Direct faithfulness
status remains 0/3 human-reviewed, 0/3 scored, 0/15 verified browser-to-decoder
correspondences, and no real-data `ΔF`, interval, or test. Existing separate
synthetic GPU smokes are engineering diagnostics only; the previous
black-frame-mask smoke is not evidence for the revised internal estimand.

Next: close the independent audit findings, then implement and test the
combined synthetic-only internal-ablation runner. Do not score protected real
data before the human and clock gates and atomic runner preflight pass.

### Hash-ranked sham integration GPU smoke (2026-10-18)

Replaced the combined synthetic hook smoke's Python PRNG sham selection with
the protocol's stable SHA-256 ordering (`qta-sham-v1`, seed
`qta-sham-seed-20261018-v1`, opaque synthetic token `synthetic-record-0`).
CPU selector tests passed 5/5 and the full repository suite passed 1,463/1,463
in 30.6 s. The initial system-Python focused-test attempt failed because that
interpreter lacks PyTorch; rerunning under the repository `.venv` passed.

The existing 9-condition synthetic Qwen hook smoke then passed: 1/1 synthetic
records, 9/9 forwards, finite outputs and hook cleanup. The target groups were
`[0,5]`; the SHA sham groups were `[5,15]`, with one overlapping target group.
Runtime 40.19 s; peak allocated/reserved VRAM 10.75/12.44 GiB on the RTX 5070
Ti. Private result is mode 0600 at
`data/private_nextgqa/synthetic_qta_hash_sham_20261018_0508/summary.json`,
SHA-256 `7de487300c64e01aeaf41b4e2632b9b8598f7453d9c7200f3c19c22a2a64925c`.
This smoke uses synthetic group assignments, does not compute answer scores,
and does not implement the real claim-mapped cohort preflight or scorer.

Direct faithfulness remains **0/3 reviewed, 0/3 scored, 0/15 verified clock
correspondences; no real `ΔF`, CI, or test**. The protocol draft is still under
independent audit and confirmation design decisions remain open.

### Fixed-prediction score-path integration (2026-10-18)

Extended the synthetic 9-condition runner to extract full-vocabulary
log-softmax values for five distinct single-token A–E labels from the final
prompt position, freeze `p/q` from baseline-full, and calculate the same
`F`/`ΔF` arithmetic for QTA and hash-ranked sham. Focused composition/scorer
checks passed 6/6; the full repository suite passed 1,465/1,465 in 30.1 s;
`py_compile` and `git diff --check` passed.

A fresh final-source GPU run passed 1/1 synthetic record and 9/9 forwards in
39.79 s; peak allocated/reserved VRAM was 10.75/12.44 GiB. It froze option
indices D/E and token IDs `[32,33,34,35,36]`. Synthetic `ΔF_QTA` and
`ΔF_sham` were both approximately `-5.96e-8` nat and their difference was 0.
These numbers only verify finite score extraction and arithmetic on a
synthetic clip. They have no faithfulness interpretation and no uncertainty
analysis. Result is mode 0600 at
`data/private_nextgqa/synthetic_qta_scoring_20261018_0521/summary.json`,
SHA-256 `99e2f52eb2788effc9d54002a5cd5cd76ea176a788735e12d84f6929746038a9`.
The primary agent independently recomputed all 15 local repository-source
hashes and the SHA sham assignment; they match the artifact.

Direct faithfulness remains **0/3 reviewed, 0/3 scored, 0/15 verified clock
correspondences; no real `ΔF`, CI, or test**. Human review and timing remain
the main external gates. The protocol draft's earlier “no scorer exists” text
was narrowed: a synthetic score-arithmetic smoke now exists, but no
claim-mapped, atomic-preflight real-data scoring runner exists.

### Final provenance-complete synthetic score-path run (2026-10-18)

The independent code audit found that the prior 0521 artifact omitted the
fixed-prediction scoring helper from its source map and had inaccurate runner
wording. The runner now hashes `fixed_prediction_selective_dependence.py`,
records synthetic A–E diagnostics accurately, and calls the run a score-path
integration smoke rather than claiming it performed no scoring. Focused tests
passed 6/6; after the final code edits, the full suite passed **1,465/1,465**
in 30.55 s, with compile and whitespace checks clean.

The provenance-complete rerun passed **1/1 synthetic record, 9/9 forwards**.
The primary agent recomputed all **16/16** repository source hashes, the SHA
sham group assignment, output hash, and private modes (directory 0700, file
0600). Result SHA-256:
`84d3e9360d617b4cb15d0885b4f8b7b5fa2c6ef9e8a09b09a14a0cee8a10637d` at
`data/private_nextgqa/synthetic_qta_scoring_20261018_0530/summary.json`.
An independent auditor verified the previous 0521 code path and confirmed its
numerical score calculation; the final run adds the missing scorer source hash
and corrects wording. Model weight bytes remain un-hashed; the pinned cache
revision is recorded.

The final numerical diagnostics are unchanged: target ΔF and sham ΔF each
`−5.960464477539063e−8` nat (floating-point residue near zero), difference 0.
They are synthetic-only and have no inferential or faithfulness meaning. The
protocol audit found the corrected internal-ablation semantics and gate design
coherent; the protocol remains an engineering draft, not a preregistration.

Next implement the claim-mapped real-data runner with the three cohort-wide
gates in the protocol, but keep score branches disabled until all human review
and clock-correspondence receipts are validated. Direct evidence is still
**0/3 reviewed, 0/3 scored, 0/15 clock correspondences, no real ΔF/CI/test**.
