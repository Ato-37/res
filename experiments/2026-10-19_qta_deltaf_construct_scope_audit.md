# QTA ΔF construct-scope audit (2026-10-19)

## Question

Does the current primary estimator support the thesis target—whether an
internal intervention increases prediction dependence on video evidence
indicated by a generated explanation—or only a narrower behavior?

## Audited implementation and protocol

- Protocol: `docs/qwen_question_attention_faithfulness_protocol_v1_draft.md`
  SHA-256 `533cfb31e37be0dcf3a498e273741305981f802d09dfd62a8642f04edaaa1721`.
- Baseline scoring primitive: `scripts/nextgqa_baseline_only_scoring.py`
  SHA-256 `b8a5de1bb8a983d1b1ecec3f185ea411bce5cbd09adab9b5fa5dffcb67a4f34d`.
- Actual-input bridge: `scripts/qta_actual_candidate_input_bridge.py`
  SHA-256 `ee9658afe3d3152acec2ee351f5bb9ffc26035edc47be0fe02491ebcdf266905`.
- Source commit: `407d7f4c8a4dfa016cdbb0edd2e31b7eac32856b`.

The protocol freezes the untreated full-video answer options `p` and `q`, then
defines each condition's score as their log-probability margin. Its primary
contrast is the difference between the control-ablation and
claim-evidence-ablation margins; `ΔF` is that contrast under QTA-Bias minus
the same contrast under baseline. Explicitly, with
`m_g(z)=log P_g(p|z)-log P_g(q|z)`, the frozen definition is
`F_g=m_g(control_ablation)-m_g(evidence_ablation)` and
`ΔF=F_QTA-Bias-F_baseline`. The operation masks selected visual-key entries
on the final answer-query row in decoder layers 5–16 while reusing the original
processor batch. It is therefore a targeted internal-access intervention,
not literal removal of video frames from the input. QTA-Bias adds the
protocol's fixed `ln(2)` bias to visual-key entries at those same layers; this
is part of the intervention dose, not a measured effect.

The referenced protocol file is explicitly a **draft**, and I found no
preregistration record in the inspected source/log set. Thus `ΔF` is a drafted
primary estimand, not a verified preregistered endpoint. Any confirmatory
follow-up must freeze and independently timestamp the protocol before exposing
new outcomes.

The older exploratory generated-text proxy v1 document calls this direct
endpoint “the preregistered primary.” That phrase conflicts with the direct
protocol's explicit draft/not-preregistered status and is stale wording; it
should not be used as evidence of registration. I leave that older protocol
byte-for-byte intact because it documents a completed, separately named proxy
endpoint; this audit records the correction without rewriting its historical
source.

Stage 1 is intended to independently review whether the claim is supported
and where it occurs; a separate Stage-2 reviewer judges whether that claim
distinguishes the frozen `p` from `q`. The protocol explicitly says this
second judgment is not a re-verification of video grounding. Both review
stages remain uncompleted. The planned three rows are a descriptive
feasibility cohort: the protocol forbids a confirmatory p-value or efficacy
claim from `n=3` and requires a separate video-disjoint confirmation plan.

## Primary-source context

- Turpin et al. show in text-model tasks that input biases can change answers
  while generated chain-of-thought rationalizes those answers without naming
  the bias. This establishes a reason not to infer model reliance from a
  plausible explanation alone; it is not a video-Qwen result:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf
- Jain and Wallace report, for NLP attention models, that attention weights
  can diverge from importance measures and distinct attention patterns can
  yield equivalent predictions. This is a caution about reading the QTA
  selector as an explanation; the present estimator instead tests a specified
  intervention and remains model/task specific:
  https://aclanthology.org/N19-1357/
- ERASER distinguishes human-rationale alignment from whether a provided
  rationale influenced its prediction. This supports keeping Stage-1 claim
  support separate from prediction-dependence measurement:
  https://aclanthology.org/2020.acl-main.408/
- Siegel et al.'s CCT work motivates measuring probability-distribution
  changes rather than only discrete label flips. It studies text LLMs and does
  not validate this video-token intervention:
  https://aclanthology.org/2024.acl-short.49/
- A May 2026 preprint, BonaFide, reports that many tested chain-of-thought
  faithfulness metrics were near chance on its constructed ground-truth tasks.
  It is not a test of QTA or this video protocol; it motivates treating
  construct validation as a separate requirement:
  https://arxiv.org/abs/2605.25052

## Finding and limits

**Inference from the frozen operational definition:** a positive `ΔF` would
support the narrow statement that, for the tested fixed predictions and
reviewed cohort, QTA-Bias increased margin sensitivity to ablation of
reviewer-indicated visual keys relative to a rank-distance control, compared
with baseline. It would not by itself show that the model generated or used
the explanation as its reason, that all video access routes were removed, or
that the explanation is fully truthful. The Stage-2 human judgment improves
claim-to-prediction linkage but is not a causal test of the model's explanation
generation process. Rank proximity also does not guarantee semantic or visual
exchangeability of control keys, and attention/evidence/control overlap and
mask-induced distribution shift must remain visible in the result.

This finding does **not** change the pre-outcome primary metric, subgroup,
threshold, or closed exploratory branches. The real-data direct endpoint has
not been run, so any future protocol revision can still be versioned before
outcomes. The intended graduation-thesis target remains broader than the
evidence presently available.

## Useful next validation

If the current `ΔF` is retained, preregister it before new outcomes as a narrow
mechanistic outcome. Before a broader claim that explanation faithfulness
improved, design a separately
frozen validation with controlled video pairs that change a reviewed
claim-bearing event and matched placebo edits, measured with and without the
internal intervention. Test selective fixed-`p/q` changes on evidence edits,
not generic margin collapse; report explanation consistency separately. This
is a proposal, not an executed experiment. It requires a source-video-level
cohort, an independently justified practical effect threshold, adequate power,
and new human review; the present `n=3` feasibility plan cannot supply those.

## Current direct progress

No direct faithfulness evidence is added by this construct audit. Formal claim
review is **0/3**, verified clock/frame correspondence is **0/15**,
prediction-link review is **0/3**, direct scoring is **0/3**, and there is no
real `ΔF`, interval, or test. Current GPU availability does not satisfy the
missing review/trust/scoring gates. The first permitted model activity remains
the prespecified three untreated baseline forwards once the authentic
review-only receipt, root/roster/witness, Stage-1 review, and clock/frame inputs
are verified. Phase-1 transaction composition is currently unit-tested only;
it is not a production CLI or runtime path.

## Follow-up checkpoint (2026-10-19 11:58 JST)

- Re-read the current formal-workspace controller and Japanese runbook. The
  localhost Stage-1/Stage-2 structural audit exists, uses external pins, and
  remains fail-closed/non-authorizing; its focused synthetic controller suite
  passed **8/8** and `py_compile` passed in this turn.
- Browser E2E still has no successful browser assertion. The recorded Windows
  Chrome route cannot reach the WSL loopback server, and this Linux environment
  has no Chromium/Chrome/Firefox binary or Playwright/Selenium/Pyppeteer package.
  No unsafe non-loopback workaround or package installation was attempted.
- A fresh private filename-only scan found **0** quick-review response JSONs in
  `data/private_nextgqa`; there is no informal human impression to summarize.
- The active simple artifact remains nickname-first, three questions, no file
  picker or timestamp entry. That satisfies the user's lightweight usability
  request, but is not the formal review UI and cannot produce faithfulness
  evidence.
- Last individually elevated GPU check (11:54 JST) found 15,134/16,303 MiB
  free on the RTX 5070 Ti and CUDA-enabled PyTorch. No model forward was run.
  GPU inference is still gated on authentic external trust/reviewer/handoff,
  semantic claim review, and clock/frame correspondence; available hardware
  does not replace those inputs.

Direct faithfulness remains **0/3 scored**: formal claim review **0/3**,
clock/frame correspondence **0/15**, prediction-link review **0/3**, with no
real `ΔF`, confidence interval, or test. The next material step is either
successful human input into the simplified local check (informal only), or
completion of the independently trusted formal handoff. The Windows-to-WSL
browser route is an engineering blocker for browser E2E only; it does not block
the already-verified localhost controller tests or independent CPU work.

## Follow-up checkpoint (2026-10-19 12:02 JST)

An independent provenance audit confirmed that the direct QTA protocol is
explicitly an engineering draft, not preregistered. It found one stale use of
“preregistered primary” in the older, separate generated-text proxy v1 document;
that wording refers to this direct endpoint and is inconsistent with the
protocol header. The historical v1 proxy specification was left byte-for-byte
unchanged; this audit records the correction so that phrase is not treated as
registration evidence.

For the separate generated-text proxy v2, the dated design log supports that a
fixed candidate pool and selector-only eligibility rule preceded generation
and scoring. The realized three rows were selected by that rule, rather than
named individually before the selector pass. No external registration record,
immutable pre-score protocol hash, or freeze receipt was found. The record now
calls this “prospectively specified exploratory,” not preregistered; this
supports temporal ordering of the procedure but does not prove the exact
pre-score document bytes were frozen. The v1/v2 finite proxy cohorts remain
closed and are not promoted to direct-faithfulness evidence.

The simple local review path was revalidated with a synthetic-only Chrome E2E
(1/1 under individual WSL/Windows interop access) and builder/inspector/importer
CPU tests (25/25). The default sandbox's Chrome cleanup failure repeated before
the elevated run; the elevated test passed and used only synthetic video. A
filename-only scan still found zero exploratory response JSONs. No human review
or direct score was added; direct counts remain **0/3, 0/15, 0/3, 0/3**, with no
real `ΔF`, confidence interval, or test. The GPU remains available per the
11:54 JST check, but no direct GPU score is allowed until authentic review and
clock/frame evidence exist.

## Follow-up checkpoint (2026-10-19 12:06 JST)

The local response scan still found **0** quick-review answer JSONs. A fresh
individually elevated, read-only WSL check found the RTX 5070 Ti available with
**15,122/16,303 MiB free**, **0% GPU utilization**, CUDA 12.8, and
PyTorch `2.11.0+cu128`; WSL reports 13 GiB available out of 15 GiB. No model
was loaded. Hardware is not the limiting factor, but the formal direct-score
inputs have not appeared: independent trusted human/release evidence, claim
judgments, and clock/frame correspondence remain absent. No new real-data GPU
forward is justified under the frozen direct protocol. The concise local
artifact is ready for an informal impression response, which would not replace
those formal inputs.

## Current PDCA checkpoint: GPU experiment eligibility (2026-10-19 12:11 JST)

An independent branch audit searched for actual-data GPU work that could
advance the direct-faithfulness question before the missing human/trust/clock
inputs arrive. The primary cross-checked the finite-cohort logs. No unclosed,
non-duplicate actual-video GPU experiment was identified:

- The fixed `single_overlap_anchor` annotation-proxy cohort completed **30/30
  videos and 180/180 forwards**; its descriptive mean ΔF was **+0.00833 nat**
  (SD **0.14284**). The all-groups variant closed at **25/30 feasible** and was
  not run as a partial denominator.
- The question-swap diagnostic completed **80/80 videos and 320/320 forwards**;
  it measured selector overlap behavior, not claim support or prediction
  faithfulness.
- The generated-text v2 proxy completed **3/3 rows and 12/12 forwards**, with
  descriptive mean ΔA **+0.027440 nat/token**; this finite endpoint is closed
  and is not direct evidence.
- The v4 mask-specificity analysis completed **868/868 forwards** across
  147/147 rows (140 placebo-eligible). It is post-result exploratory and does
  not replace claim-linked review.

These logged experiments are auxiliary or exploratory and remain closed; no
cohort, endpoint, subgroup, or threshold was changed here. A current
individually elevated check confirms the GPU is available (RTX 5070 Ti,
15,133/16,303 MiB free, CUDA 12.8, PyTorch 2.11.0+cu128), so the reason not to
launch a new model job is experimental eligibility, not hardware access. The
prespecified next direct GPU activity is **exactly three untreated baseline
forwards**, but only after trusted Stage-1 review/release and all **15/15**
clock/frame correspondences pass. After freezing `p/q`, obtain separate
Stage-2 prediction-link judgments before any treatment run.

Faithfulness status remains `no_go` for a new direct GPU score: **0/3** claims,
**0/15** clock/frame pairs, **0/3** prediction links, **0/3** direct scores,
and no direct ΔF/interval/test. No new GPU workload was run in this checkpoint.
