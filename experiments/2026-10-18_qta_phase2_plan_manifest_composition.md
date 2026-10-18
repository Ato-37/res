# QTA phase-2 plan-manifest composition

## Purpose

Bind each of the three phase-2 nine-branch plan rows to the exact signed phase-1
cohort record, its phase-1 structural row-binding digest, and the corresponding
baseline p/q manifest row. This closes a composition gap: the phase receipt
verifier signs artifact bytes but intentionally treats the plan artifact as
opaque. The new checker remains non-authorizing.

## Setup

- Date: 2026-10-18.
- Inputs: deterministic synthetic clock/processor adapter fixtures, the actual
  three-row synthetic cohort preflight, synthetic five-option baseline scores,
  and temporary Ed25519 trust rosters created by the receipt tests.
- Phase 1 fixes the ordered three-row cohort. Phase 2 must carry the exact same
  receipt bindings and `freeze_id`.
- The row adapter's review token is a 24-character opaque token, while signed
  receipt bindings require a 64-hex opaque token. The composition therefore
  binds `SHA256(ASCII(review_token))` to the receipt token and never emits the
  source review token in its report.
- No protected video, live reviewer identity, production trust root, model, or
  GPU was used.

## Implementation

`scripts/qta_phase2_plan_manifest.py` verifies the signed phase-2 receipt and
its recursively signed phase-1 receipt before trusting the plan bytes. It then
requires: (1) the same freeze ID and ordered record/batch bindings across
phases; (2) three unique phase-1 row hashes from the atomic preflight; (3) a
canonical baseline p/q manifest with three records, valid finite log
probabilities, and recomputed deterministic p/q indices; and (4) each plan's
row hash, batch hash, receipt token, baseline row digest, and p/q indices to
match those upstream artifacts. The complete nine-branch plans are passed
through the existing validator and checked against the frozen layers, logit
bias, and sham seed from the existing three-row aggregator. Parser inputs are
size-capped. Output explicitly denies model loading, scoring, and execution.

The follow-up also upgrades `qta_baseline_pq_candidate` to emit manifest v2.
Each row now contains a canonical score-binding descriptor and its SHA-256,
covering A–E token IDs, choice-token binding digest, prompt/input hashes,
score position and input token, processor batch, log probabilities, p/q, and
five run digests. The phase-2 plan's `score_binding_sha256` must equal this
descriptor hash. The prediction-link renderer and validator accept both v1
and v2 manifests; only the phase-2 composition gate requires v2.

## Verified results

- Focused baseline candidate, prediction-link review, renderer, phase receipt,
  schedule, adapter-composition, and cohort-preflight checks: **59/59 passed**;
  candidate/review/receipt core including output-size limit: **40/40 passed**.
- `py_compile` passed for the v2 builder, phase-2 verifier, review validator,
  renderer, and affected tests.
- Full repository test discovery from the repository root with `PYTHONPATH=.`:
  **1,922/1,922 passed** in 63.176 seconds after all changes. Earlier full-suite
  runs passed 1,920 and 1,921 tests at intermediate revisions. An initial
  discovery attempt from inside `scripts/` was invalid (12 path/import
  failures); it was discarded and root-based discovery was used.
- Positive synthetic end-to-end composition verified both signatures, the
  recursive phase link, all three plan rows, and the fixed 24-forward phase-2
  schedule (27 including phase 1).
- Negative tests rejected a changed signed baseline artifact, a different
  phase-2 freeze ID, and a plan whose layer/bias/sham seed differs from the
  frozen QTA constants.
- Independent audit confirmed the first composition change. A new independent
  audit confirmed v2 hashing, phase-2 score-hash matching, and review-page
  compatibility; it found no critical issue within the non-authorizing scope.
  A further check verified that the reviewer HTML displays the protocol hash
  but omits score-binding field names and model/tokenizer/scorer/processor
  hashes. The output-size cap also has a dedicated failure test.
- These are structural synthetic checks only; they do not establish real-data
  validity, human judgment correctness, reviewer identity, or model faithfulness.

## Source hashes (before stage-1 source-pin follow-up)

- `scripts/qta_baseline_pq_candidate.py`:
  `a5e9af0259814fcf2300f389eb14f3d7094b6bff131156ee6108fecaa5fe3ede`
- `scripts/qta_phase2_plan_manifest.py`:
  `649eae33578db02551a38986b2ee6b3c765581154819216b9daa07c981de45a0`
- `scripts/nextgqa_prediction_link_review.py`:
  `036842ea7d528dbf7f8fe0ad4cbfc180386412f16d4c9783cf4c0bbe3e03655a`
- `scripts/render_nextgqa_prediction_link_review.py`:
  `426e45ee1093d5248b5f674c30cb7dae16e94aab6e0c9054a0329f53736333a2`
- `scripts/test_qta_baseline_pq_candidate.py`:
  `4012d78aa3c60efee09b375370975c86022d5a1d244df298348a609d00370908`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `2299db3102fdb9da60d065cf2f26602074a2bc019977b508f0a89d39bf141e93`

## Limitations and current faithfulness progress

This composition checker does not authenticate the semantic correctness of a
plan, the origin of a score descriptor, human review, or the mapping from
browser-clock footage to decoder PTS. A hash proves internal byte consistency,
not that the log probabilities came from an actual Qwen forward. The production
trust root, reviewer identities and signatures,
review evidence, and prediction-link judgments remain missing. Verified direct
faithfulness outcomes remain **0/3**; capture/player-to-decoder clock pairs
remain **0/15**; prediction-link judgments remain **0/3**; and there is no
direct score, effect estimate, confidence interval, or significance test. This
work improves protocol wiring, not empirical faithfulness evidence.

Independent source audit found that target/sham group indices are still only
shape-checked: the plan does not recompute top-k selection from baseline
question-attention scores. Existing phase-1 prompt/token artifact bytes are
not yet parsed by the phase-2 verifier, and the current phase-2 artifact roles
do not include a baseline-attention capture or forward receipt. Recomputing a
selection from caller-supplied vectors would establish arithmetic consistency
only; it would not authenticate those vectors as Qwen attention. The next
protocol step is to specify a post-baseline attention artifact with exact batch,
question-position, model/runtime, layer, and output bindings, then test its
selector recomputation without granting execution authority. The independent
human intake still lacks a separately pinned trust root and actual reviewer/key
assignments, so no review package has been opened.

## Next action

Keep this candidate out of production execution gating until the unresolved
selection and baseline-forward provenance joins are specified and independently
checked. The stage-1 source-pin digest is now joined, while review authenticity
and upstream source-audit provenance remain separate gates. Continue the human-free
protocol and measurement work while the assigned reviewers prepare the local
review handoff. Do not run model scoring until the upstream review and trust
gates are actually satisfied.

## Follow-up: stage-1 source-pin linkage

The next source audit showed that the v2 score descriptor carried option and
scoring bindings but did not carry all source pins needed to link the
baseline manifest to the phase-1 source-pin artifact. The descriptor now also
contains question/options/claim hashes, source-video hash, reviewed-interval
hash, and the stage-1 support decision. The phase-2 candidate builder and
verifier bind these fields, row order, 24-character review-token hashes, and
the prior video-review receipt digest to the exact phase-1
`stage1_review_and_lineage` artifact. The verifier still labels semantic review
unauthenticated and grants no scoring authority.

The focused builder, phase-2 receipt, prediction-link, and renderer suite passes
**63/63** after schema hardening. Negative tests reject a changed stage-1
receipt digest, a changed claim source pin, a boolean schema version, and an
unknown top-level key. Full repository discovery passes **1,925/1,925** in
65.592 seconds. These checks close byte/field mismatch checks only; they do
not authenticate people, review truth, or model-forward provenance.

The phase-2 parser now requires exact parity with the stage-1 producer's
top-level key set, including its private-QA/source-binding fields and explicit
denial fields. It consumes only the review receipt digest and ordered source
rows for this join; exact schema parity detects producer drift without treating
the other fields as reviewed evidence. Test fixtures model the complete key
set, but their non-join payload values remain synthetic. An independent auditor
compared the producer's complete 26-key output (including the three denial
fields supplied by `DENIAL`) against the parser set and found no mismatch. The
auditor reviewed the fixtures and negative tests; its focused suite passed
**51/51**.

Source hashes after the stage-1 source-pin follow-up (before the selection
arithmetic follow-up below):

- `scripts/qta_baseline_pq_candidate.py`:
  `ff30e8455a4dca60d3f64209e9efaac570ab5984d8e0284778d7ba857a2a6799`
- `scripts/qta_phase2_plan_manifest.py`:
  `4e37eab6f3384132756feb35448820d3a350416baf20d754f9002eae30000d15`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `e911d10cd189adbf7e3a6de6b9c95fe6928bc8392b8520d7c8d19f3520f370fa`

## Follow-up: selection arithmetic binding

The remaining target/sham plan gap has been narrowed with
`scripts/qta_selection_provenance.py`, and its record is now included in the
hash-bound nine-branch plan and required by phase-2 composition. It binds each
row to the phase-1 opaque record token, row and processor-batch hashes, the
baseline score-binding and p/q row hashes, prompt/input hashes and score
position, ordered temporal groups, and the frozen selector source SHA, layers
5–16, tie rule, top-k budget, sham selector, and seed. Per-layer group masses
are encoded as big-endian IEEE-754 binary32 hex; aggregate scores use binary64
hex. The verifier independently recomputes group means, target ranking/ties,
sham digest ranking, and consistency with the canonical plan and biased key
positions. The plan binds the complete record into its hash, and the phase-2
candidate rejects a changed prompt binding.

This is intentionally an arithmetic and cross-artifact check. The score masses
remain caller-supplied and are not authenticated as output from a real Qwen
baseline forward. Question-token positions are also self-declared in this
record; phase-2 binds them to prompt/input hashes but does not independently
derive them from rendered text, tokenizer offsets, or a processor mapping.
Processor temporal-group positions are compared to the plan's signed row
binding but are not re-derived from a separately parsed processor artifact.
The artifact explicitly states `forward_source_authenticated=false`,
`attention_capture_authenticated=false`, `model_load_permitted=false`, and
`scoring_authorized=false`. The CPU protocol tests did not load a model; a
separate synthetic GPU smoke is recorded below.

Verification: selector, plan, baseline candidate, and phase receipt focused
tests passed **40/40**. An independent audit of the integrated composition and
synthetic adapter suite passed **33/33**, and separately confirmed the fixed
selector-source/layer/seed checks with the **9/9** selector unit tests. Root
full-suite discovery passed **1,935/1,935** in 66.308 seconds. Expected
fail-closed diagnostics printed by subprocess tests did not fail the suite.

This adds reproducible selection arithmetic but no new direct faithfulness
measurement. Status remains **0/3** reviewed claims, **0/15** verified
browser-clock-to-decoder correspondences, **0/3** prediction-link judgments,
and **0/3** direct scores; no effect estimate, interval, or test exists.

Next: independently parse and bind the actual processor mapping and the
phase-1 prompt/token artifact, then define a post-baseline forward/capture
receipt whose underlying output bytes—not only a declared hash—can be checked.
Keep real scoring behind the unresolved human review, trust, timing, prediction
link, and capture gates. Continue CPU protocol checks in parallel with the
local human-review handoff preparation.

Selection arithmetic revision source hashes:

- `scripts/qta_selection_provenance.py`:
  `c9d6adf7ee3ce097a57688a4ec26e16d5249e3fd1d344323e2c3c5f6cbf4a3b4`
- `scripts/qta_nine_branch_plan.py`:
  `bcaeefd9bdd95e1a41312c4ca7f924b43a70139a741e7d5efcef29d8318e4376`
- `scripts/qta_phase2_plan_manifest.py`:
  `771de3c9433b44be41439934703f2e3f16edc82354ff9c13da4e7813ad9a706f`
- `scripts/test_qta_selection_provenance.py`:
  `2b91de1f1a800374a2c0c609b30339f4cfc7730226cd67b8e68d51cfde41d73d`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `2f19a0e7154c0df31b69ff686003517d96b623513d398eff59183aa1693eff1f`

## Synthetic GPU integration smoke

The first local GPU attempt loaded the pinned model and completed its baseline
forward, then failed closed before the treatment branches because the legacy
smoke's raw synthetic sham token differed from the new provenance record's
SHA-256 phase-row token. The r2 repair made those paths internally consistent,
but used the wrong raw-versus-hashed token convention relative to the frozen
protocol as described below. No result from that incomplete attempt was
treated as an experiment.

That r2 command completed **1 synthetic record and 9/9 forwards**, but the
independent audit then found that the protocol specifies the raw 24-character
`review_token` as the sham hash input, while the r2 smoke used its SHA-256
binding. Its sham selections are therefore protocol-incompatible and r2 is
retained only as an engineering diagnostic, not as the corrected validation.
The implementation and tests were changed to carry the raw opaque token from
the locally supplied baseline row and compare it against the phase-2
provenance.

The protocol-aligned r3 command
`.venv/bin/python scripts/smoke_qwen_question_selector_intervention.py --output data/private_nextgqa/qta_selection_provenance_smoke_20261018_r3/summary.json`
completed **1 synthetic record and 9/9 forwards** on the local RTX 5070 Ti,
CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 5.18.0, BF16 4-bit weights,
and the pinned checkpoint revision. Runtime was **52.33 s**, peak allocated
VRAM **10.755 GiB**, peak reserved **12.439 GiB**, with 1.911 GiB free after
the run. The output is local-only under a mode-0700 directory, file mode 0600;
its SHA-256 is `9db55b76acdd2fe2c7298b8f034e3e34aa259072ec436dfc81086d15c2817405`.
The 16 synthetic temporal groups produced protocol-aligned raw-token sham
selection; all nine branch logits passed finite-input/batch checks, and
selection arithmetic reproduced successfully. A regression independently
compares raw-token ranking with hashed-token ranking and confirms they differ.

This is synthetic wiring evidence only: it is not a real NExT-GQA run, has no
human-reviewed evidence intervals, and reports `faithfulness_result=false`,
`attention_capture_authenticated=false`, and `treatment_frozen=false`. It
provides no direct faithfulness score, effect estimate, interval, or
significance test. Model/attention capture provenance remains unauthenticated.
The phase-2 parser also still takes `question_positions` from the selection
record itself as its expected value, rather than independently deriving that
span from the rendered prompt/tokenizer artifact. This remains an explicit
selection provenance gap; the candidate is not execution-ready.

## Local real-video sampler eligibility preflight

Using the refreshed hash-only candidate inventory, the CPU-only
`scripts/run_qta_real_video_sampler_preflight.py` decoded the three frozen
local candidate videos without displaying frames, loading a model, or scoring.
The command requested 32 endpoint-inclusive uniform decoded-frame samples per
video. It reported decoded frame counts **510, 1,140, and 1,258**, 32 unique
sampled PTS and 32 unique sampled RGB hashes for each video, and maximum grid
deviations **16.145 ms, 16.145 ms, and 16.355 ms**, all within the frozen
50-ms tolerance. Reported effective sample rates were 1.825287, 0.815690, and
0.739134 fps. The local mode-0600 manifest is
`data/private_nextgqa/qta_real_video_sampler_preflight_20261018_1948.json`,
SHA-256 `181f605d6093186f561f92075688ef0b94a52d654d51a86ef0031bd9cd195728`.

This establishes only local decoder/sampler feasibility and sample timing
geometry. It does not match browser/player clock footage to decoder PTS, audit
the sampled pixels against reviewed evidence, or establish semantic claim
support. The 15 browser-clock/decoder-PTS correspondences and all human review
judgments remain **0**. The artifact declares `rights_status` as
`unverified_local_only`, `execution_authorization=not_granted`,
`model_loaded=false`, and `faithfulness_scored=false`; it remains private and
must not be published.

## Verification after protocol alignment

After adding the raw-token cross-artifact binding and its independent
raw-versus-hashed ranking regression, the focused suite passed **42/42** tests.
Root full-suite discovery passed **1,937/1,937** in 68.954 seconds. `git diff
--check` passed. These tests verify structural arithmetic and artifact joins;
they do not resolve the question-token position or real attention-capture
authenticity gaps above.

Current source SHA-256 values for this selection candidate:

- `scripts/qta_selection_provenance.py`:
  `d5b3a7582e2c1561c0eb480046f3ee80cd3cdf5801bee01ff1fa77d5bd07e1db`
- `scripts/qta_nine_branch_plan.py`:
  `bcaeefd9bdd95e1a41312c4ca7f924b43a70139a741e7d5efcef29d8318e4376`
- `scripts/qta_phase2_plan_manifest.py`:
  `8ff0368f2d9c2da2d9e045e2dd5cb060e5b2725c24feb8299375301a6b7930f2`
- `scripts/smoke_qwen_question_selector_intervention.py`:
  `da41b0a29f3d5aeeb625181b88ccff62c3f19f57abd14957c6f0649b12f93058`
- `scripts/test_qta_selection_provenance.py`:
  `c795f770ac87d3ac591233c1e0e42bf53fb5d7ffd0fc764a984f004e96483ff9`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `cbc96582c21a7b741b799243ca51388f60f229f7fbf0081eba11d58f89a57ea7`
- `scripts/qta_choice_token_binding.py`:
  `45f7cfce20fd0fcb8fc7709d60f2638fe6e098d38f29b581f6a4c082611552ef`
