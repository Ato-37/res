# Test-split frame-mean content-sensitivity check (2026-10-21)

## Question and status

The fixed validation cohort showed positive saved-claim likelihood contrasts
for original video versus both constant-gray and per-frame RGB-mean controls.
This is a post-result, exploratory split check: does the same predeclared
original-minus-per-frame-RGB-mean contrast recur on a distinct outcome-blind
test-split cohort? This is not a confirmatory or independent replication:
validation outcomes motivated the design, prior model exposure is incompletely
known, and the same model family generates and scores claims.

## Frozen estimand and cohort

- Select the first three distinct eligible test-split videos in deterministic
  loader order after excluding all media found in local claim-generation and
  scoring artifact families. Use the local sample-exclusion manifest as an
  additional exclusion. Selection is outcome-blind; preserve failures and do
  not replace rows. Current CPU feasibility scan: the first 16 candidate
  videos included 13 excluded media and 3 timing-eligible unique videos, with
  no decode failures or timing rejections. This scan did not run the model.
- Freeze media/QID assignments and selected pair hashes before generation.
  Bind the selected pair-filter bytes by SHA-256 in the lock and bind each
  generation batch's filter bytes in its packet.
- Generate one `v2_short` claim per locked row with 16 sampled frames using
  the pinned Qwen2.5-VL-7B-Instruct revision. No human review is required for
  this exploratory local run under the user's waiver; no human review is
  encoded as performed.
- For each saved claim, score exactly two teacher-forced inputs: original
  video and per-frame uniform RGB equal to rounded per-channel frame means
  (`floor(mean + 0.5)`, clipped to `[0,255]`). Keep question, choices, prompt,
  timestamps, target tokenization, frame count, and processor fixed.
- Primary row endpoint: original-video minus per-frame-RGB-mean mean
  target-token log likelihood, nat/token. Report all locked rows and finite
  cohort mean/median/range only. No CI, hypothesis test, threshold search, or
  population claim at `n=3`; do not pool with validation.

## Meaning and limits

This can show only whether the generated sentence's conditional likelihood
differs between the original video and a uniform per-frame color control on
these three test-split rows. The transform changes spatial layout, local
variance, edges, texture, and other pixel statistics, and may be out of
distribution. It cannot establish claim truth, semantic evidence support,
causal evidence use, hallucination rate, or direct explanation faithfulness.
Direct faithfulness remains 0/3. Historical model exposure is incomplete;
the test split is not certified fresh. Rights remain unverified and all raw
videos, questions, claims, and receipts remain local.

## Workload and execution gate

Expected bounded workload: one pinned 4-bit/BF16 GPU load for three short
claim generations, followed by a second pinned 4-bit/BF16 load and six scoring
forwards. Estimate several minutes including model loads. Require the existing
`.venv`, CUDA/BF16, at least 12 GiB free VRAM and 8 GiB host memory before each
load; use a one-row generation/scoring smoke before the two remaining rows.
Verify source hashes, lock/filter membership, packet-to-lock media/QID
assignments, video byte hashes, exact timestamps, nonvisual tensor equality,
and both completed forwards before retaining any score. The local GPU/host
preflight on 2026-10-21 found 15.28 GiB free VRAM and 14 GiB host memory
available; recheck immediately before model load. Do not publish protected
inputs, claims, questions, or per-row receipts.

## Execution and verified results

The freezer's first lock (`_01`) was rejected before generation after a wider
audit found that its filename-only prior-score scan missed some locally scored
JSONs. It was not used for claim generation or inference; an initial attempted
process was interrupted during CPU video scan with zero model forwards. The
freezer now also detects nested `completed_forwards > 0` and explicit scoring
markers and binds the selected pair-filter hash. This remains a marker-based
heuristic, not proof of complete historical exposure. The corrected `_02`
lock excluded 40 locally discoverable media digests and selected three test
rows; selected-filter SHA-256 is
`51658b1e778e91966d0b893c5ffe80727c67ff000e82e7807bc7e531300b7c5b`.
The smoke and remaining claim packets bind to their respective frozen batch
filters; all three source-video byte hashes and source-code hashes matched.

Claim generation completed **3/3** test-split videos with 16 sampled frames
each. A first scoring smoke exposed a positional-argument mismatch before model
load (0 forwards); its failed receipt was preserved. After correcting the
caller, the one-row smoke passed, followed by the two untouched locked rows.
Across three separate 4-bit/BF16 scorer loads, **3/3 rows and 6/6 forwards**
completed. Receipt reconciliation verified for every row: test split,
input-packet SHA-256, row token, saved-claim SHA-256, source-video SHA-256,
batch-filter membership, and explicit
`human_review.review_actually_performed=false`. All outputs remain private
under `data/private_nextgqa`.

Observed per-row original-minus-frame-mean target log-likelihood differences
were **+0.480102, +2.214195, +0.466227 nat/token**. The finite-cohort mean was
**+1.053508**, median **+0.480102**, and range **[+0.466227, +2.214195]**; all
three rows were positive. These describe this selected n=3 cohort only. No CI,
significance test, or pooling with validation was used. This is an exploratory
conditional-likelihood sensitivity proxy—not a direct faithfulness result—and
the row range and out-of-distribution control limit interpretation. Human
review was waived as a run prerequisite but was not performed; direct
faithfulness remains **0/3**.

Focused CPU tests passed **9/9**; `py_compile` and `git diff --check` passed.
The packet reader now accepts explicit packet paths under the private root and
an expected split while retaining owner/file-type/mode checks.

## Next action

Do not repeat these three perturbation measurements or tune the contrast. The
next useful step is an independent audit of this proxy against alternative
nonsemantic controls and claim-length/tokenization effects, using already
available private receipts only. Then decide whether a separately frozen
cohort would materially discriminate the content-sensitivity hypothesis.
Any further experiment remains exploratory until a direct, reviewed
prediction-evidence-link endpoint exists.
