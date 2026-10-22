# Synthetic transient-event QTA construct validation v1 (2026-10-19)

## Purpose and scope

This prospectively specified, local synthetic study asks whether the frozen
question-temporal attention bias (QTA) changes a gold-answer-versus-fixed-foil
margin's measured sensitivity to known transient event keys relative to
equal-count, nearest temporal-control keys. It is a mechanism/measurement
construct check for the existing internal attention-key ablation, not an
explanation-faithfulness or hallucination experiment: the fixture has no
model-generated explanation or claim, and QTA is selected from question-token
attention rather than an explanation. Keep it separate from every NExT-GQA
result and direct faithfulness count.

The preceding two-video readability smoke remains closed at its original
denominator and will not be expanded or used to tune this study. No outcomes
from this cohort may be used to change the intervention, group mapping,
controls, score, or analysis.

## Frozen cohort and stimuli

- Generate **40 scene-seed pairs (80 clips)** from the
  circle/triangle transient-event family. Within each pair, geometry, question,
  and the identical ordered A–E answer choices are shared; only which shape
  briefly changes color differs. Each clip is 32 RGB frames at 1 fps.
- Each target changes red→blue for exactly four frames and then returns to red.
- Use four equally weighted onset strata at frames `[4, 10, 18, 24]`; each
  leaves at least four frames in both the pre-event and post-event red states.
  There are exactly 10 scene pairs per onset stratum.
- Sample **41** unique scene seeds without replacement from the full uint32
  seed space using Python `random.Random(20261019).sample(range(2**32), 41)`.
  Reserve the first seed solely for the engineering smoke; the remaining 40
  are the analysis cohort. This keeps smoke outcomes outside the cohort and
  its denominator. Shuffle those 40 seeds with `random.Random(20261020)` and
  assign consecutive groups of ten to onset strata in ascending onset order
  `[4, 10, 18, 24]`. Enumerate orientation × rotation combinations first by
  `circle_left=True, False`, then by cyclic rotation `0..4`; within each onset
  stratum, shuffle those ten combinations with
  `random.Random(20261021 + stratum_index)`, where stratum indices follow the
  ascending onset order above. This defines a random sample of procedural
  scenes, stratified equally by onset, with randomized balanced orientation
  and answer ordering. Record Python version and hash the exact assignment
  manifest before model scoring. The engineering smoke may validate only
  pipeline mechanics; its scores cannot tune stimuli, treatment, score,
  controls, or analysis and never enter the cohort estimate.
- The ten combinations per onset stratum are circle-left/circle-right crossed
  with the five cyclic rotations of a frozen list of five choices. Both clips
  in a pair share orientation, question, choice order, and scene seed. Across
  all 40 pairs, each onset has exact 5/5 orientation balance and each answer
  rotation occurs exactly twice. For each target object separately, correct
  answer labels occur 8 times at each A–E position across the 40 pairs.
  Seeded small position jitter is shared within pair.
- Use the pinned local Qwen2.5-VL-7B-Instruct checkpoint at revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
  eager attention, and the existing 32-frame video processor path.

## Frozen measurement and analysis

For each clip, derive complete temporal groups from the actual processor
geometry. Each four-frame event must map to exactly two groups; otherwise the
preflight fails before model loading. The evidence mask uses those two groups.
Choose exactly two disjoint control groups by ranking all non-evidence groups
by absolute distance between group center and the event-interval center, then
earlier group index; no outcome or attention value enters this control rule.
The CPU-only preflight checks evidence/control budgets and disjointness but does
not claim to measure the later model-derived selector overlap. During scoring,
record the frozen QTA and hash-sham selector overlap with evidence/control
groups descriptively and retain every fixed clip, including overlap cases.

Use nine ordered forwards per clip: baseline, frozen QTA, and deterministic
hash-sham, each under full input, evidence-key ablation, and control-key
ablation. Derive QTA groups once from the untreated baseline-full
question-token attention (mean layers 5–16, top `max(1,ceil(0.10*T))`, ties to
the earlier group); apply `ln(2)` visual-key bias at layers 5–16 and reuse the
same selector for all QTA branches. Set `p` to the known generated gold choice.
Freeze `q` as the highest-scoring non-gold A–E choice from that same clip's
baseline-full pass and reuse the same `p/q` across every branch. This measures
gold-answer score dependence against a fixed baseline foil; it does not
evaluate whether a possibly wrong model-generated prediction is faithful. Do
not condition cohort inclusion on baseline correctness, selector overlap, or
score.

For treatment `t`, let `m_t(c)` be the fixed gold-versus-baseline-best-foil
log-probability margin under condition `c`:

`F_t = m_t(control-ablation) - m_t(evidence-ablation)`

`DeltaF_QTA = F_QTA - F_baseline`

The primary unit is the **scene pair**: average the two clip-level
`DeltaF_QTA` values within each pair. For the four equally weighted onset
strata (`n_h=10`), estimate `theta=(1/4)*sum_h(mean_h)`. Use the stratified
variance `V=sum_h((1/4)^2*s_h^2/n_h)` and Welch–Satterthwaite degrees of
freedom `df=V^2/sum_h((((1/4)^2*s_h^2/n_h)^2)/(n_h-1))` for the two-sided 95%
t interval and one-sample test of `theta=0`, using the independent
scene-seed pairs within each fixed onset stratum as sampling units. If `V=0`,
the t degrees of freedom and test are undefined: report the estimate and
stratum values, with no t interval or p-value, rather than inventing a test.
This is an approximate model-based t inference; it ignores the finite-
population correction from sampling without replacement, which is negligible
at 40 draws from 2^32 seeds. It targets only the procedural scene-seed
population under the frozen seed-sampling and counterbalancing design, not
natural videos. There is one primary endpoint and
no multiplicity adjustment. A lower 95% interval bound above zero supports
greater gold-margin sensitivity to event keys than to nearest temporal controls
only for this synthetic task/model/intervention/mask operation. Otherwise
report an inconclusive or negative result for this fixed design. There is no
predeclared practical-effect threshold; do not borrow the separate head
study's +0.25-nat threshold. Hash-sham contrast, choice accuracy, and selector
overlap are secondary descriptive diagnostics, not additional significance
searches.

## Denominator, cost, and stop rules

- Requested denominator: 80 clips, 40 pairs, 720 model forwards (nine per
  clip). No salvage subset, replacement, or outcome-dependent exclusion.
- A non-finite score, source/configuration hash mismatch, incomplete branch,
  invalid event-to-group geometry, or failed all-cohort CPU preflight makes the
  full score run invalid; retain the failure receipt and produce no cohort
  estimate.
- CPU preflight must stream one clip/pair at a time and retain hashes/aggregate
  geometry only, not all processor tensors. The GPU runner must free each
  clip's outputs/input references before the next clip and recheck live VRAM.
- Require exact balance in preflight: 40 pairs/80 clips; 10 pairs per onset;
  5/5 orientations within each onset; every orientation × answer-rotation
  combination once per onset; each target object's gold label exactly 8 times
  at A–E across the cohort. Within each pair, question/options, geometry,
  onset, evidence groups, and control groups must match; only the changed
  object identity and corresponding gold label may differ.
- Based on the previous 588-forward QTA execution (42.32 min), expected
  model-body time is **about 52 minutes** for 720 forwards, plus startup and
  preflight. Prior
  related peak reservation was 12.44 GiB; the recent long-prompt smoke peaked
  at 13.354 GiB reserved. Before any full run, require the verified checkpoint,
  a passing CPU preflight/smoke, at least 13.5 GiB free VRAM before loading and
  3 GiB before every forward, and a live host-memory check. Stop safely if any
  guard fails; do not lower thresholds to force completion.
- The output remains local under `data/private_nextgqa/`, mode 0600, aggregate
  only. Never write prompts, answers, frames, per-clip scores, or absolute
  paths to a shared/published artifact. Rights remain unverified.

## Interpretation limits and next action

Even a positive result would establish neither that generated explanations
identify these events nor that QTA makes explanations more faithful. The
single geometric scene family, deterministic choice construction, synthetic
visual distribution, question-conditioned selector, and key-ablation
distribution shift restrict interpretation. The control is a nearest temporal
non-event key group, not a visual placebo. It would provide a controlled
mechanism/measurement validation, not direct thesis efficacy evidence. No
outcome from this cohort exists at protocol freeze.

Next: implement and independently audit a new fixture, CPU-only 40-pair
processor preflight, and nine-branch GPU runner; the existing two-clip runner
cannot execute this design. Require exact cohort/stratum balance, two event
groups and two disjoint controls per clip, stable source/configuration hashes,
and bounded memory before any GPU smoke. Then run the reserved first seed as
one paired two-clip engineering smoke: **18 forwards** (nine per clip),
explicitly outside the 40-pair cohort.
Use the existing SHA-256 sham selector contract with frozen seed
`qta-sham-seed-20261018-v1`, opaque token
`qta-synth-event-v1:{scene_seed}:{changed_object}`, group-index ordering,
and earlier-index tie break; record this configuration in the manifest. Only
if the external smoke passes and live resource
guards allow it should the full 720-forward cohort run. Formal claim-linked
real-video scoring remains separately blocked
on independently trusted human/release inputs and all 15 clock/frame
correspondences; this study does not satisfy them.
