# QTA generated-text / benchmark-annotation dependence proxy v2 (exploratory)

## Status

This is a prospective design revision after the v1 finite-cohort output showed
complete QTA/control-key overlap in two of three rows. It does not reinterpret,
filter, or replace any v1 row. The v1 endpoint is closed for additional scoring.
The v2 selector-only engineering smoke and fixed-pool feasibility pass have
completed, followed by a separately implemented and independently audited
scorer, a successful scoring smoke, and the fixed three-row descriptive run.
Protected receipts remain local under `data/private_nextgqa`. The descriptive
run closes this finite exploratory proxy cohort; it does not measure direct
faithfulness or justify cohort expansion.

## Question and estimand

For each eligible real-video question, generate one explanation continuation
once from baseline and freeze its exact raw token IDs. Let `L_m(S)` be the mean
conditional log probability of those same IDs when visual groups `S` are
ablated under model condition `m`. Define:

```text
A_m = L_m(control ablation) - L_m(gsub-evidence ablation)
ΔA  = A_QTA - A_baseline
```

This is a benchmark-annotation dependence proxy. `gsub` is question timing,
not independently verified support for each generated sentence. It cannot
establish sentence truth, hallucination, or explanation faithfulness.

## Outcome-independent group selection

For each geometry-valid row, use the fixed question-attention QTA selector
before generation. Let `E` be the complete processor temporal groups mapped
from the official QA interval and `Q` the fixed top `max(1,ceil(T/10))` QTA
groups. Require `E ∩ Q = ∅`. Among groups outside both `E` and `Q`, choose a
control set `C` with `|C|=|E|` that minimizes the sum of absolute rank distances
between sorted `E` and sorted `C`; resolve equal-cost ties by the
lexicographically smaller index tuple. Fail closed if no such `C` exists.
Flattened processor visual-key positions must also be equal-budget and pairwise
disjoint across `E`, `Q`, and `C`.

The treatment selector and all three group sets must be computed and frozen
before generation, raw target inspection, or any likelihood scoring. Do not
replace groups after observing an output. Record selector scores/indices,
evidence/control indices, key counts, distance cost, failures, and the exact
decision receipt. Candidate eligibility is defined by this rule before scoring;
report attempted and excluded denominators. Do not compare eligible and
ineligible rows as if randomized.

## Cohort and execution sequence

1. Use the already frozen deterministic candidate order. V1-scored indices
   0–3 are unavailable for v2 outcomes. The fixed v2 scan pool is exactly
   zero-based lock indices **4–15** (12 candidates maximum); these candidates
   are not certified fresh because historical exposure records are incomplete.
2. In original order, CPU-preflight each candidate's source/data/media hashes
   and processor geometry. A geometry failure is recorded and consumes that
   index; do not backfill from beyond index 15.
3. Run one engineering smoke on already-used lock index 0, which is excluded
   from all v2 eligibility and outcomes. It may compute only the selector,
   candidate control, and key-set validity; no generation or text scoring.
4. Run the selector-only GPU feasibility pass over indices 4–15, stopping at
   the fourth eligible row or after index 15. It may compute `Q`, `C`, and
   eligibility only; it must not generate text or score an output. The first
   four eligible candidates in original order become one reserved scoring
   smoke and three non-overlapping finite descriptive rows. If fewer than four
   qualify, record the shortfall and do not expand the pool or score any rows.
   Record pool size, index attempted/pass/failure counts, selector failures,
   each exclusion reason, accepted candidates, and unattempted indices after
   early stop.
5. The scoring runner binds this immutable eligibility receipt and its own
   successful smoke before scoring the three descriptive rows. The completed
   run is recorded in the [2026-10-19 experiment log](../experiments/2026-10-19_qta_v2_selector_feasibility.md).
6. Report per-row deltas, signs, range, overlap checks, sample failures, runtime,
   and VRAM. No p-value, population interval, or efficacy claim is planned for
   three rows. A larger study requires a separately justified, pre-outcome plan.

## Interpretation and stop conditions

This design removes direct key-set collision among QTA treatment, evidence
ablation, and control ablation. It does not make QA timing a semantic claim
annotation, eliminate mask-induced distribution shift, or prove that likelihood
changes represent faithful reasoning. The control set is selected conditional
on the QTA set; this may alter its rank-distance distribution and must be
reported. Eligibility can select a restricted subset. If too few candidates
meet the frozen rule, stop this proxy rather than relaxing it after observing
scores.

All artifacts remain local-only, mode 0600 under the protected data root.
Rights remain unverified. Every output must state
`direct_faithfulness=false` and `faithfulness_result=false`.

## Observed finite-cohort result (2026-10-19)

Smoke index 7 completed one generation and 4/4 scoring branches, yielding
`ΔA=+0.013513 nat/token`. Descriptive indices 8–10 completed 3/3 generations
and 12/12 scoring branches with zero failures. Their `ΔA` values were
`+0.033096`, `+0.004754`, and `+0.044470 nat/token` (mean `+0.027440`, median
`+0.033096`, range `[+0.004754,+0.044470]`). This is a selected n=3 finite
cohort; no CI, p-value, or population inference is reported. Two rows had
negative `A` under both model conditions. The positive deltas therefore only
describe how this intervention changed the relative evidence-vs-control
ablation margin; they are not an absolute evidence-dependence gain or
faithfulness effect. The fixed endpoint is closed to post-result expansion or
tuning. Direct claim review and prediction-link adjudication remain outstanding.
