# STEAR-to-Qwen adaptation audit (2026-10-18)

## Purpose

Assess whether STEAR is ready to serve as the internal intervention for the
study's primary outcome: whether an explanation's claimed video evidence
selectively supports the fixed prediction, and whether an internal treatment
increases that selective dependence. This is a method/reproducibility audit,
not a new experiment.

## Sources and inspected implementation

- Primary paper: [STEAR arXiv HTML v1](https://arxiv.org/html/2604.03045v1),
  especially §§3.2–3.4 and 4.1, and Table 1.
- Installed Transformers Qwen2.5-VL implementation and the repository's
  `scripts/smoke_qwen_stear_trigger_feasibility.py` synthetic hook smoke.
- Research estimand and gates in
  `docs/internal_intervention_grounding_faithfulness_plan.md` and the current
  open-ended PDCA log.

## Verified observations

The paper reports Qwen2.5-VL-7B results: NExT-QA accuracy 59.63% baseline and
60.87% with STEAR (+1.24 percentage points); EventHallusion 63.81% to 67.48%,
VidHalluc 81.70% to 83.84%, and MVBench 58.74% to 61.58%. The paper's protocol
states risk layers 5–16, normalized-entropy threshold 0.85, top-10% salient
patches, temporal homogenization coefficient 0.80, and contrastive strength
0.75. These are benchmark accuracy results; they do not estimate this study's
fixed-prediction explanation-evidence dependence or its intervention effect
`ΔF`.

The accessible main paper does not numerically specify reinjection strength
`λ`, the extent/boundary rule for the trigger-layer neighborhood, or enough
detail to reproduce the on-the-fly query/key/value/output mappings and their
Qwen insertion point. The paper says code is in supplementary material; that
material was not available in the inspected page during this audit. This is an
unresolved source-access gap, not evidence that the supplement contains no
specification.

The installed Qwen2.5-VL decoder has self-attention, MLP, and normalization,
with no separate cross-attention sublayer. The repository smoke captures
intermediate entropy logits and self-attention rows, then applies a top-10%
visual-token selector with a radius-one neighborhood as an engineering
fixture. Its docstring and output explicitly exclude reinjection,
counterfactual decoding, and efficacy. It uses synthetic video. Thus the
existing smoke verifies limited hook/shape feasibility only; it is not a
STEAR implementation or a real-video faithfulness result.

## Interpretation

STEAR is a relevant prior-art intervention candidate and a useful comparator,
but its published Qwen accuracy gain cannot establish the selected thesis
outcome. Directly porting the method is currently under-specified and its
attention/evidence selector is not the same as the independently reviewed
claim interval used by the thesis outcome. Any simplified implementation must
be named as a new Qwen adaptation, specify which paper components are omitted,
freeze its selection and treatment semantics before examining outcomes, and
compare it against a matched internal sham/control. A treatment that selects
or recalculates its target separately in each ablation branch would change
what is being tested; the currently planned selector is fixed from the
untreated full-video pass across branches.

## Direct faithfulness progress and limits

Direct claim-linked video-faithfulness progress remains **0/3 independently
human-reviewed, 0/3 scored, and no `ΔF` estimate, uncertainty interval, or
statistical test**. Verified browser-to-decoder frame correspondences remain
**0/15**. The latest review handoff repair made the local packet's field
disclosure complete but did not produce a review. The production six-condition
GPU scorer and a frozen intervention are still absent. No GPU job was run in
this audit; repeating the existing generic synthetic trigger smoke would not
answer the intervention question.

## Next action

Retrieve and audit the paper supplement if an accessible primary-source copy
is available. Then specify one explicitly named Qwen adaptation and matched
control/sham that are independent of reviewed outcomes, and implement the
full fixed-label, fixed-input six-condition scorer with immutable bindings.
Continue human-review and player-clock/decoder-PTS calibration work in
parallel. Run an actual-video GPU pilot only after those measurement and
protocol gates pass. Protected materials and derived review data remain local;
rights status is unverified.
