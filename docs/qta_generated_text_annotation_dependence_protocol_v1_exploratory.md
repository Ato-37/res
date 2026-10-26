# QTA-generated text / benchmark-annotation dependence (exploratory v1)

## Status and scope

This is a separately named exploratory endpoint. It is **not** the frozen
direct faithfulness endpoint, a hallucination detector, or an estimate of
whether a generated claim is true. It asks whether the exact tokens of a
model-generated explanatory sentence have greater conditional likelihood
sensitivity to removing NExT-GQA `gsub`-anchored visual attention keys than to
removing a rank-distance-matched control, and whether a fixed QTA key-bias
intervention changes that contrast.

The annotation target comes from benchmark QA timing, independently of the
QTA-selected keys. `gsub` does not independently establish that a particular
generated sentence is supported by those frames. Any result must be called a
**benchmark-annotation dependence proxy** and kept separate from human claim
review, direct faithfulness scores, and the preregistered primary.

## Frozen estimand

For each eligible video-question row, generate one explanation sentence once
from the unmodified baseline model. Preserve the exact raw continuation token
IDs, the processed prompt boundary `P`, generation termination state, and the
generation configuration before scoring any condition. No decode/re-tokenize
step is permitted for the scored target.

Map the official question's `gsub` intervals to complete temporal visual-key
groups using the existing processor/PTS geometry mapper. Define the matched
control with `select_equal_count_rank_distance_control_groups`. Before any
model score is observed, require the evidence and control sets to be disjoint,
nonempty, equal in group count, and exactly equal in flattened processor visual
key count. Record their overlap with the QTA selector; do not select or exclude
rows based on outcomes or on the overlap.

From one unablated baseline attention pass, select
`max(1, ceil(0.10 * T))` temporal groups using question-token attention and the
existing frozen QTA selector. Reuse this exact key set in all QTA conditions.
Apply the fixed `ln(2)` positive attention-logit bias in layers 5–16 at every
target-token prediction query `P−1 … P+L−2`, where `L` is the number of frozen
raw continuation IDs. Under both baseline and QTA, remove the annotation or
control keys from those same target-query rows. If bias and ablation overlap,
ablation takes precedence; the overlap count is reported.

For each model condition `m`, use the **mean** target-token log probability as
the primary score (sum is a length-sensitive secondary diagnostic; EOS status
is recorded, not silently inferred). Define:

```text
A_m = mean_logp(target | control-key ablation, m)
    - mean_logp(target | annotation-key ablation, m)
ΔA  = A_QTA - A_baseline
```

Positive `A_m` means removing annotation-anchored keys lowers the frozen
sentence's conditional likelihood more than removing the matched control keys.
Positive `ΔA` means that difference is larger under QTA. Neither sign means the
sentence is true, that the annotation supports its specific claim, or that
faithfulness improved.

## Required execution and reporting

- Use a new, outcome-blind, video-disjoint cohort for any real-video pilot.
  Compare video IDs and byte hashes with the known local prior-use inventory;
  if historical freshness cannot be certified, label the cohort reused or
  history-unknown and do not describe it as independent confirmation.
- Preserve a new source/inputs manifest; keep raw text, IDs, media paths,
  generations, and per-row scores mode 0600 under the private data root.
- CPU/synthetic tests must establish raw target-ID freezing, exact prompt
  boundary, causal logit-row alignment, exact query-row hook coverage, exact
  flattened key-budget equality, overlap/ablation precedence, finite score
  arithmetic, and complete failure accounting before a GPU smoke.
- A GPU smoke is engineering-only. A three-row run is a feasibility and
  heterogeneity description, with no significance test or population interval.
  If both are run, freeze four geometry-eligible candidates in one order before
  any scoring: the first is reserved for the engineering smoke, and the next
  three form the descriptive feasibility cohort. Do not replace a candidate
  after seeing any generated text score.
  A full result requires a separately justified cohort and a frozen plan before
  scores are observed. Do not change the prompt, selector, layers, bias,
  interval mapping, control, or endpoint after outcomes and then reuse those
  results as confirmatory evidence.
- Every artifact and report must set `direct_faithfulness=false` and
  `faithfulness_result=false`; separately report formal human-review status.
  Rights remain unverified; all protected inputs and derived artifacts remain
  local and unpublished.

## Current go/no-go

The multi-query attention-mask and compact target-logit-window primitives,
candidate lock builder, and geometry preflight have CPU tests and a real-media
CPU preflight. They remain unaudited as an end-to-end GPU runner. The
2026-10-16 sentence pilot's text packet does not retain the original generated
continuation IDs and must not be retrofitted as if those IDs were available.
The new lock found 417 local test-video candidates after excluding 83 matches
to the then-current local prior-media hash inventory; a first 3 candidates
passed CPU temporal geometry and equal-key control checks. Historical exposure
is still unknown. Rerun a four-row preflight after the scorer is frozen, audit
it independently, then consider one engineering smoke plus three descriptive
rows as a proxy-only GPU pilot.
