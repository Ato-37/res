# Synthetic oracle-claim evidence-dependence study v1 (protocol draft)

## Status and question

**Design draft only. No cohort is frozen and no model outcome has been
observed under this protocol.** This proposes a new, human-free construct test
of whether a fixed internal visual-key intervention increases the model's
dependence on known video evidence when judging a single atomic claim. It is
not a rerun/reanalysis of the closed synthetic QTA study and is not a
natural-video or free-form explanation-faithfulness result.

The prior synthetic QTA study (40 pairs/80 clips/720 forwards) is closed. It
used one circle/triangle color-change family, question-token selection, a
five-choice answer margin, and a QTA-vs-baseline contrast. Its primary result
was −0.01719 nat (95% CI [−0.05415,+0.01978]). This proposal changes all four:
new event families and seeds; claim-span selection; binary true/false claim
judgment; and a claim-conditioned intervention-vs-matched-sham estimand.
Reusing the previous aggregate receipts cannot answer this question because
they contain no claim scores or claim-span attention.

## Stimuli and truth source

- Generate **48 independent 32-frame, 1-fps clips**, 16 in each of three new
  procedural event families: (1) one object briefly crosses a fixed gate and
  returns to its starting side while a foil object stays fixed; (2) one object
  briefly rotates to a specified orientation and returns while the foil
  object stays fixed; (3) one object briefly expands and returns to its
  initial size while the foil object stays fixed. Each event occupies exactly
  four consecutive frames, with identical object state before and after. This
  makes the transient event frames necessary to distinguish true from false
  statements; static pre/post state cannot reveal the event.
- Use visually distinct, counterbalanced objects and exact generator-state
  truth. Do not reuse the old red/blue transient-color family. The true and
  false claims within a clip differ only in object identity, e.g. “The star
  briefly crosses the gate and returns” versus “The diamond briefly crosses
  the gate and returns.” One object performs the family event; the other does
  not.
- Freeze two asymmetric renderers (five-point star and kite-like diamond), a black
  gate, star/diamond colors `#3366CC` and `#CC6633` (randomly swapped), and
  the in-memory RGB renderer. Each scene seed independently draws gate x from
  `{112,128,144}`, a shared track-y offset from `{-8,0,8}`, base size from
  `{24,28,32}`, and background shade from `{#F5F5F5,#EBEFF3,#FAF4EA}`.
  Randomly assign the star to the upper/lower track (track centers at y=60/132
  plus one shared offset). Freeze the PRNG algorithm
  and draw order in the generator manifest.
  Reject no valid draw; these nuisance values alter pixel scenes but not
  event truth. The balanced target-object × answer-label assignment is
  shuffled with the frozen `random.Random` PRNG onto the four independently generated scene seeds in each
  family × onset cell. Record every seed and nuisance draw before rendering
  model outcomes. The event target and foil occupy distinct tracks. During the
  four event frames `s..s+3`, use these state trajectories: gate family
  center-x `(80, g+12, g+48, 80)` for gate x `g` (the object center crosses
  the gate line; the full shape need not clear it);
  rotation family angle degrees `(0,90,90,0)`; expansion family radius
  multiplier `(1.0,1.5,1.5,1.0)`. Outside the event window both objects remain
  frame-identical; the foil never changes. The true claims are respectively
  “briefly crosses the gate and returns,” “briefly rotates by a quarter-turn
  and returns,” and “briefly grows to 1.5 times its initial size and returns.”
  The false foil substitutes only the other object's identity. Assert from
  generator states that the target satisfies the predicate (crosses both
  sides of gate, reaches 90 degrees, or reaches the 1.5 scale ratio) and the
  foil does not; also assert rendered event-only differences and unchanged
  pre/post frames. Fail closed on any violation.
- For each family, cross four event-onset strata (frames 4, 10, 18, 24) with
  four independent scene seeds per onset. Within each family × onset cell,
  cross the two event-target objects and the two answer-label orders exactly
  once, using `random.Random(20261021 + 10000 + 4*family_index + onset_index)`
  to shuffle assignments onto the four scene seeds.
  Seed-dependent nuisance variation above makes the four scenes distinct.
  Generate
  the analysis seeds from a versioned random stream and explicitly reject any
  overlap with the closed color-change cohort and reserved smoke seed. Use a
  versioned deterministic hash ranking over all unordered pairs of distinct
  temporal groups that exclude event groups and their immediate neighbors.
  Hash `control-v1|scene-seed|opaque-clip-token|i|j` with SHA-256, canonicalizing
  `i<j`, and choose the lexicographically smallest digest (tie by `(i,j)`).
  Freeze the exact control hash algorithm and seed before rendering outcomes.
- Build one input per claim: the video plus a question asking whether that
  single supplied atomic statement is supported. Present two answer choices,
  `A. Supported` and `B. Not supported`, or the reversed mapping; balance the
  supported-to-A/B mapping across true and false claims. The programmatically
  generated true and false-foil statements have matching syntax and differ
  only in object identity. Truth labels are derived directly from generator
  state, not model predictions or human review.
- Freeze the rendered text template to `Does this video support the following
  statement?\nStatement: {claim}\n{choice_A}\n{choice_B}\nAnswer:` inside
  one video-first user message, with `choice_A/B` set by the randomized label
  order. Resolve the claim's exact character offsets with the fast tokenizer's
  offset map after video expansion and capture only the last token wholly
  inside the claim. Require one literal claim occurrence, one video placeholder
  expanded to one contiguous visual-token run, and exact processor IDs equal to
  raw rendered IDs with only that placeholder expanded. Ignore only leading or
  trailing whitespace attached to boundary tokens; reject non-whitespace
  boundary crossings, ambiguous spans, A/B retokenization, or processor A/B
  continuation mismatch before model loading.
- Before any model load, validate all 48 clips, truth labels, true/false
  statement differences, rendered text, A/B single-token IDs, processor frame
  geometry, onset/family/object/label balance, exact event-to-key groups, and
  non-overlapping seed lineage. Fail the full-cohort run on any failed row;
  no replacements or outcome-dependent exclusions.

## Frozen scoring and internal intervention

For each clip and each of its two statements, score the model's supported vs
not-supported answer-label log-probability margin. Use the same pinned local
Qwen2.5-VL-7B-Instruct checkpoint, 4-bit weights, BF16 compute, eager
attention, visual grouping derived from the actual processor geometry, layers
5–16, and requested `ln(2)` visual-key logit bias as the closed QTA study.

The selector is claim-span-conditioned rather than question-token-conditioned:
on the unmodified full-video forward for that particular statement, capture
the **last tokenizer token wholly contained in the exact claim substring** at
layers 5–16 and aggregate its visual-key attention to temporal groups. Earlier
causal query rows cannot see the claim suffix, so do not average all claim
tokens. Select the top `max(1, ceil(0.10*T))` groups with ties to the earlier
group. Use that frozen selector only for the same statement's intervention
branches. This claim-token row is used only to select keys and is not itself
intervened on. Apply the `ln(2)` bias and
event/control key ablations on the final prompt query row that predicts the
first A/B answer token, using keys selected from the claim-token row. Record
selector, intervention, and scored-logit positions separately; verify the
actual additive-mask delta at every intervention-query × selected-key cell
and verify scoring reads the corresponding next-token logits. Do not use the
answer-label position to select keys.

For every claim, run three selector conditions (unmodified baseline,
claim-selected visual-key bias, deterministic hash-matched sham bias) under
three video-key conditions (full video, event-key ablation, matched-control
key ablation). The sham uses the exact number of temporal groups as the
claim-selected intervention and a frozen seed/opaque clip token. For each
claim, rank every unordered pair `0≤i<j<T` by SHA-256 of
`sham-v1|scene-seed|opaque-clip-token|claim-kind|i|j`, using lexicographic
digest order then `(i,j)` as a tie break; take the first pair. This sham
ranking is independent of selected attention and outcome. Do not match sham overlap to evidence/control
windows or require the sham to differ from the selected set: those rules
condition the sham on treatment selection and can erase selector/evidence
alignment. An accidental exact sham/QTA match is allowed and reported.
Event-key ablation removes the two groups
intersecting the exact four-frame transient event. Control-key ablation
removes two non-event groups excluding the event groups and their immediate
neighbors; select from all eligible pairs by a frozen deterministic hash
rank. Freeze exact event trajectories, group mapping, control/sham seeds, and
overlap rules before looking at model outcomes. Since ablation takes
precedence over bias where sets overlap, record effective biased-group counts
for each arm/mask. Differences in effective dose after masking are part of
this joint intervention-by-ablation estimand, not evidence for a dose-matched
pure attention effect. Both interventions and shams use the same video
frames, prompt, processor, query/key geometry, and attention hook contract.
Masking remains a possible out-of-distribution intervention.

This is 2 statements × 3 selector arms × 3 video-key conditions = **18
forwards/clip**, **48 clips**, **864 forwards**. A separate engineering
smoke uses one reserved out-of-cohort clip and 18 forwards. The exact
implementation and prompt must be CPU-tested first; use the existing
claim-span attention capture and multiquery key-bias hook only after the
current unsupported assumption (arbitrary contiguous claim-token query rows)
has a tested binding. No Qwen score is allowed until the CPU preflight and
smoke pass.

## Primary estimand and inference

For clip `i`, claim `x`, selector arm `t`, and video-key condition `c`, use
processor-bound single-token answer IDs to compute
`L[i,x,t,c] = logP(Supported | x,i,t,c) -
logP(NotSupported | x,i,t,c)`. Sign the A/B label-token difference according
to the frozen answer order; do not use raw Supported log-probability alone.
Define `M[i,t,c] = L[i,true,t,c] - L[i,false-foil,t,c]`, using each claim's
own rendered prompt and claim-span selector. Define selective evidence
dependence `D[i,t] = M[i,t,control-ablation] -
M[i,t,event-ablation]`. The single primary outcome is the equally weighted
mean of `D[i,bias] - D[i,sham]` over the 12 family × onset cells (four
independent clips per cell), equally weighting each family and onset stratum.
Positive values mean the claim-selected bias increases the true-vs-false
claim-margin response to event-window retention relative to the equal-size
sham, within this procedural benchmark. Baseline is reported descriptively,
not as another primary contrast.

Compute the stratified standard error, Welch–Satterthwaite degrees of freedom,
two-sided 95% t interval and one-sample test of `theta=0` over independent
scene seeds within the 12 fixed family × onset cells. Use only this one inferential
endpoint; report the estimate, CI, p-value, all family means, direction counts,
full-video true/false accuracy, requested/effective rows, failures, and
selector/evidence/control overlap. Zero-variance cells contribute zero to the
stratified variance; if the total variance estimate is zero or the
Welch–Satterthwaite denominator is zero, report the estimate without a finite
CI or p-value and do not add a variance correction. Do not tune or replace the primary claim,
intervention, dose, family, masks, or practical threshold after observing
results. Zero-variance cells contribute zero; only a total variance estimate
of zero removes the interval and test. Any failed row or branch invalidates
the full-cohort primary estimate; do not pool or replace rows.

## Main validity threats and interpretation ceiling

This study uses oracle-generated short claims, not free-form explanations.
The model is asked to judge synthetic statements rather than explain a natural
video; success would validate only claim-conditioned evidence sensitivity on
these procedural families. It cannot establish that Qwen generates faithful
explanations or that internal steering improves natural-video grounding.
Additional threats are language priors, object/action salience, family-specific
rendering cues, answer-label bias, detector difficulty, leakage from statement
construction, key-ablation distribution shift, and imperfect temporal controls.
Counterbalance labels/objects/onsets, use true/false pairs differing in one
object term, report baseline accuracy and every family, and retain all clips.
Even so, masking out-of-distribution inputs and a three-family generator leave
external validity limited. Do not describe this as human-validated or as a
replacement for the current review handoff.

## Resource plan and gates

The completed 720-forward synthetic QTA run took 2,633 seconds (43m53s), but
this protocol changes prompt length, captures claim-token rows, and scores two
prompts per clip; that historical time is only a rough lower-bound reference.
Expected full run is approximately 53–65 minutes plus preflight/model startup,
subject to the 18-forward smoke. Before GPU execution, check live GPU VRAM and
host memory; require 13.5 GiB free VRAM before model load and at least 3 GiB
before each forward, preserving the existing strict guards. One smoke and one
full run only; no repeated same-cohort score collection. Keep aggregate
receipts local and mode 0600; rights status for the pinned model remains
unverified and no media/results may be published.

## Next work before execution

1. Independent review this estimand against the closed QTA cohort and inspect
   the current hook's query-row semantics.
2. Implement a versioned three-family synthetic generator, true/false prompt
   binder, claim-span position mapper, label-token scorer and aggregate-only
   receipt/analyzer. Add CPU tests for span ambiguity, token-label binding,
   mask-cell exactness, selector determinism, randomization/balance, failure
   handling and estimator arithmetic.
3. Freeze seed assignment and run a streaming CPU processor preflight. If its
   design or processor geometry fails, repair before looking at GPU outcomes.
4. Run the reserved one-clip engineering smoke under current live resource
   guards; independently audit its source/config/geometry bindings. Only after
   pass launch the full 864-forward cohort.

No human review is assumed or imputed for this synthetic oracle study. Current
real-video direct-faithfulness counts remain unchanged and must be reported
separately: 0/3 reviewed claims, 0/15 clock/frame correspondences, 0/3
prediction-link judgments, and 0/3 direct intervention scores.
