# Qwen question-attention internal intervention and evidence-dependence protocol (draft)

Status: **engineering protocol draft; not preregistered and not authorized for
real-data scoring.** A combined synthetic selector, internal-ablation, sham,
and score-arithmetic smoke exists and has passed independent code/result audit.
The claim-mapped real-data scorer and cohort preflight remain unimplemented and
unaudited. Human review and player-clock/decoder-PTS gates are incomplete.

## Research question and scope

For Qwen2.5-VL-7B-Instruct on a fixed NExT-GQA question, does a fixed
question-conditioned internal attention-bias intervention increase the fixed
baseline prediction's selective dependence on independently reviewed
claim-indicated video evidence, measured by internal visual-key ablation and
compared with an equal-budget temporal control ablation?

This is an operational score-dependence measure under a particular internal
attention intervention and a fixed-query access ablation. It cannot establish
claim truth, complete internal reasoning faithfulness, or necessity along every
model pathway. The selector reads question-token attention only; it does not
read the generated claim, reviewed interval, answer, options, gold label, or
output scores. A positive result would support only increased dependence under
this operation on independently reviewed claim-indicated evidence. It would
not show that the intervention itself selected the claim's frames or followed
the explanation.

## Fixed treatment candidate

The candidate is **Qwen Question-Temporal Attention Bias (QTA-Bias)**, a
project-specific adaptation, not a STEAR reproduction.

1. Run the untouched full-video input with the fixed five-choice prompt and
   identify the exact rendered question character span and corresponding
   processor `input_ids`.
2. From untreated full-video attention, average question-token query rows,
   heads, and decoder layers 5–16 (1-based). Sum scores over complete temporal
   groups defined by `video_grid_thw` and processor merge size.
3. Select `k=max(1, ceil(0.10*T))` groups from `T`, breaking ties toward the
   earlier group. Freeze the selection for the record and reuse it in every
   treatment branch.
4. At the final prompt-token answer-score query in layers 5–16, add `ln(2)` to
   selected visual-key logits. No other activation or model-weight changes.

Report `T`, `k`, actual fraction, selected temporal indices, and visual-key
count. Since `ceil` can exceed 10% (e.g. 2/16=12.5%), never call it exactly
10%. The selector is an engineering candidate and must be implemented in the
same runner as the fixed-query outcome ablation before real scoring.

## Review eligibility and time-to-token mapping

The unit is one frozen generated claim per video-question record. Bind it by
the exact claim text digest, claim ID, source prompt/answer digest, opaque
record token, and review-policy digest. A human reviewer who declares
independence from generation, scoring, and custody must judge the whole claim
`supports_claim`, mark every distinct sufficient occurrence, and provide a nonempty rationale for each interval. The
review export contains self-attested independence, judgments, intervals, and
rationales. Its exact-byte adoption signature can be integrity-checked under
the pinned reviewer roster, but the current scheme does not authenticate human
identity or independently prove that the review occurred or the judgments are
semantically correct; report that limit.
Other judgments, missing intervals, failed source-clock checks, or claim-digest
mismatch make the row ineligible; do not infer a replacement interval. If
multiple occurrences are marked, map each separately and take the union of their temporal groups; do
not fill gaps between occurrences. Duplicate intervals are retained in the
review record but contribute the same groups once to the union.

Intervals are seconds on the HTML5 review player's media-time clock, recorded
at millisecond precision with inclusive endpoints as in review policy v6. The
frozen calibration maps `player_time = slope * decoder_PTS + intercept`.
Before interval-to-group mapping, convert both endpoints exactly as
`decoder_PTS = (player_time - intercept) / slope`; never pass player times
directly to a PTS-based mapper. `map_annotation_intervals_to_pts()` implements
this inverse with rational arithmetic. Bind the calibration artifact and
converted intervals into each row's source manifest. Apply the predeclared
empirical held-out residual guard in PTS seconds to the mapped boundaries; if
the union of groups touched by intervals expanded by that guard differs from
the union touched by intervals contracted by that guard, fail the row before
model loading. `map_review_intervals_to_qwen_groups()` enforces this
possible-versus-certain group-set equality when given the guard. This is a
conservative stability check under the observed residual radius; it is not a
future-error bound or statistical uncertainty interval. Every mapped interval
endpoint must lie within the observed PTS span of independently matched
anchors; do not extrapolate beyond it. If an endpoint is outside the anchor
span, that row fails this feasibility protocol.

The
mapper requires positive duration and positive-duration overlap, so a shared
single endpoint has zero measure. A sampled temporal group's support cell is bounded by
midpoints between strictly increasing decoded PTS values, clipped to
`[0, verified_duration]`; group support is the span of its temporal-patch
frames. A group is evidence when its support has positive-duration overlap
with any reviewed interval. This is the existing
`scripts/qwen_review_interval_token_groups.py` rule. Review-to-decoder clock
correspondence must be independently established before mapping. Zero mapped
groups, mapping to every group, invalid/non-increasing PTS, or inability to
form an equal-size disjoint control is a row-level failure, not an invitation
to alter boundaries or sampling.

The matched control has the same number of complete temporal groups and is
disjoint from the union of evidence groups. Sort evidence indices and candidate
non-evidence indices ascending. Pair the j-th selected control index with the
j-th evidence index and minimize the sum of absolute index distances over all
subsets of the required size; ties resolve to the lexicographically earliest
control tuple. This dynamic-programming rule is implemented in
`select_equal_count_rank_distance_control_groups()` and fails closed when too
few candidates exist. The control may be noncontiguous. Record evidence/control
indices, temporal support bounds, token positions, and intersection counts
with the frozen QTA selector. This is a rank-distance control; it does not
guarantee semantic exchangeability or equal temporal-support duration.

## Frozen prediction, conditions, and outcome

From the untreated full-video next-token distribution over A–E, freeze `p` as
the highest-log-probability option and `q` as the highest-scoring different
option, ties in A–E order. Resolve all five option token IDs with the pinned
tokenizer. Score the same `p`, `q`, and final prompt-token position in every
branch. Gold answer margin and accuracy are secondary only: define
`M_gold(g)=log P_g(gold|full)-max_{j != gold} log P_g(j|full)` over A–E and
record full-video accuracy for baseline and QTA-Bias. Treat both as descriptive
outcomes for this `n=3` feasibility cohort; they do not replace the fixed
baseline `p/q` primary estimand.

The current claim generator creates a separate answer-like sentence without
binding it to the later model prediction. Claim support in the video alone
therefore does not show that the reviewed claim explains the scored `p`. Add a
**prediction-link adjudication** after every row's baseline `p/q` is frozen and
before any treatment or ablation branch. This is a second-stage textual
discrimination judgment, conditional on a completed first-stage review in
which a reviewer viewed the video and recorded `supports_claim` plus all
reviewed intervals. It does not independently verify video grounding. The
second-stage reviewer must be independent of the first-stage video reviewer
and blind to gold labels and all branch outcomes. They see the exact question,
options, `p`, `q`, frozen claim, the first-stage decision, and
the interval digest; they judge whether the claim text distinguishes choosing
`p` over `q` given that frozen video review. Record `supports_p_over_q`,
`compatible_not_discriminating`, `unrelated`, `contradicts_p`, or `ambiguous`,
with rationale and digests of all displayed fields. Only
`supports_p_over_q` is eligible; all other outcomes are row failures. This is
not a gold-answer judgment. It prevents calling an unrelated claim interval
“explanation evidence” for the scored prediction. The existing v6 claim-review
export has no such field, so a separately versioned supplemental receipt and
independent audit are required; the current review package does not satisfy
this gate. A CPU structural validator now binds the receipt to exact baseline
manifest bytes, externally frozen cohort/source digests, and `p/q` recomputed
from baseline A–E log-probabilities. It remains non-authorizing: its expected
hashes and token list must come from independently verified frozen inputs.
The baseline manifest and second-stage receipt bind the first-stage receipt
hash; each row also binds the first-stage `supports_claim` decision and
reviewed interval-list digest into the exact fields shown to the second-stage
reviewer. This establishes byte-level linkage between stages but does not
verify the first-stage receipt's trust/release chain. The validator cannot
authenticate who reviewed the material or whether the judgment is semantically
correct. A static offline packet renderer and local structural
validator CLI now exist, but no real baseline/receipt/source bundle exists and
the validator is not connected to the production scorer or a cohort-wide
release/review/clock preflight.

All outcome branches use the **same original full-video processor batch**.
`evidence_ablation` and `control_ablation` are not black-frame input masks:
they set the chosen visual-key entries to `-inf` only on the fixed answer-query
row in decoder layers 5–16 (1-based), leaving other mask entries unchanged.
The baseline operation applies no positive bias; the QTA operation applies
the frozen `ln(2)` bias at those same layers 5–16. Evidence/control ablation
takes precedence if a selected QTA key overlaps an ablated key. The implementation must record that overlap by condition and test this
composition in the combined runner.

For operation `g` in `{baseline, QTA-Bias}`, and `z` in
`{full, evidence_ablation, control_ablation}`, define
`m_g(z)=log P_g(p|z)-log P_g(q|z)`, using the same teacher-forced option
tokens and score position. Then

`F_g = m_g(control_ablation) - m_g(evidence_ablation)`

`ΔF_QTA = F_QTA-Bias - F_baseline`.

Positive `ΔF_QTA` means QTA-Bias increased the fixed prediction's differential
score sensitivity to evidence-key ablation relative to matched-control key
ablation under this exact intervention. The full score is retained as a
diagnostic; it cancels algebraically from `F`. This is not a pixel-input
masking estimand and must not reuse or relabel black-mask results.

An internal random-key-bias sham is secondary only. It applies the same
`ln(2)` additive key bias at the same final score query and layers 5–16, with
the same ablation-precedence rule as QTA-Bias. Rank all `T` temporal groups by
SHA-256 of the UTF-8 byte string
`qta-sham-v1\0{protocol_seed}\0{opaque_record_token}\0{decimal_group_index}`,
ascending digest then ascending index; take the first `k`. Freeze
`protocol_seed="qta-sham-seed-20261018-v1"` now; each real record uses its
pre-existing opaque `review_token` from the frozen candidate inventory as
`opaque_record_token`, bound into the cohort manifest before score access. This
defines a stable deterministic selection without PRNG-version dependence.
Reuse it in all sham branches. Report sham/evidence and sham/control overlaps
explicitly; sham may overlap either set, so sham contrasts do not isolate a pure
generic-bias effect. `ΔF_sham` and `ΔF_QTA-ΔF_sham` remain secondary and may
not be promoted after outcomes. There are 9 model forwards per record: baseline and QTA across
full/evidence/control (six forwards), plus sham across those three conditions.
The baseline-full forward is shared with selector and `p/q` binding; it is not
duplicated.

## Stages, analysis, and fail-closed gates

The present 3-record inventory is a descriptive feasibility cohort only, and
still requires human review and timing verification. If all gates later pass,
report per-record scores, mean, sample SD, range, requested/effective counts,
and failures. Do not compute a confirmatory p-value or efficacy claim from
`n=3`. Do not inspect treatment scores to choose selector, layer, strength,
cohort, metric, or threshold. Any later confirmation requires a newly frozen,
video-ID-disjoint cohort; independently justified SESOI; sample-size method;
paired video-level estimator and CI/test; missingness/failure rules; and
multiplicity plan, all fixed before confirmation outcomes are accessed. These
confirmatory choices are unresolved, so this draft cannot authorize a
confirmatory run.

Use three cohort-wide stages. **Pre-load authorization/input gate:** verify
exact source/question/claim and cohort digests; a review record for every row
containing the reviewer's self-attested independence and judgments
(`supports_claim`, all occurrence intervals, rationales, policy digest), with
exact-byte integrity and trusted-roster adoption receipt verified;
player-clock/decoder-PTS correspondence receipts for every interval anchor;
pinned model/tokenizer/processor versions and protocol settings; no duplicate or previously used video IDs; and processor
geometry, PTS, evidence/control assignments, and input batches. Any failure
means zero model loads. **Baseline preparation:** after that gate, perform one
untreated full-video forward per requested row to produce selector attention,
the frozen A–E `p/q`, and baseline-full A–E log probabilities needed as the
baseline/full condition in the later estimand. Keep these outputs private; do
not run evidence/control or treatment/sham branches and do not compute or expose
cohort effects. The baseline/full forward is reused later, never repeated.
Record baseline preparation successes/failures privately. Any baseline-
preparation failure is a cohort failure; stop baseline preparation at the first
failure, retain the private failure/count record, and run zero branch forwards.
**Cohort-wide
post-baseline gate:** only after all requested rows have valid baseline outputs,
complete and independently audit the prediction-link adjudication for every
row, then verify all rows' A–E token IDs, `p/q`, score positions,
selector/sham positions, branch input hashes, hook targets, and ablation/bias
composition. Any failed or missing link judgment means zero branch forwards
for the cohort and no ΔF estimate. Other failed post-baseline checks likewise
mean zero branch forwards. If all pass, run the eight additional forwards per
row (baseline evidence/control, QTA full/evidence/control, and sham
full/evidence/control), then release a cohort estimate only after every
requested row is complete. The complete design therefore uses nine forwards per
row including the baseline preparation, not nine new forwards after the
prediction-link gate. A byte
inventory or synthetic callback is not human attestation, timing evidence, or
execution authorization.

## Known limitations and present status

The attention selector may not overlap the reviewed evidence. Report its
intersection and union with evidence/control groups; overlap analyses remain
descriptive, not post-hoc subgroups. The review receipt verifies exact bytes
and roster-key adoption but does not establish that a review occurred or
cryptographically verify reviewer identity, viewing, or semantic correctness. Other visual/text states and
attention paths remain available after fixed-row ablation. Attention-mask `-inf`
ablation is a mechanistic access intervention under eager attention, not
removal of all information about a frame. Bias can change logits through
non-evidence routes, and model-internal interventions may have distributional
effects. Claims are limited to the checkpoint, prompt, sample, processor,
layers, mask backend, and intervention tested.

Direct evidence status: **0/3 reviewed, 0/3 prediction-link adjudications,
0/3 scored, 0/15 verified player-to-decoder correspondences; no real `ΔF`,
CI, or test**. Existing
question-selector GPU and internal-ablation GPU outputs are separate synthetic
engineering checks; neither tests faithfulness. The final nine-forward combined
smoke tests synthetic group assignments and fixed-prediction arithmetic, but it
does not implement claim-mapped real evidence/control groups, cohort preflight,
or authorized real-data scoring; no compliant real-data scorer exists. Rights
status remains unverified; protected data/results remain local.

## Next action

The corrected estimand and exact gate contract have passed an independent
protocol audit. The combined synthetic baseline/QTA/sham runner and score
adapter have also passed an independent code/result audit. Next bind the
reviewed-claim mapper and three-stage cohort preflight into a real-data runner,
with CPU tests before any authorized real smoke. Keep real-data scoring blocked
until human and clock gates pass and the runner receives independent audit.
Do not use synthetic results as faithfulness evidence.
