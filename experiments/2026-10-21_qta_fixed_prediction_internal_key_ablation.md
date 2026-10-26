# Exploratory fixed-prediction internal key ablation (2026-10-21)

## Question and scope

Does a fixed global visual-key attention bias increase the tested 7B model's
selective dependence on an already machine-selected temporal window when the
same full-video prediction and foil are compared under evidence-window and
matched-control visual-key ablations? This is a post-hoc exploratory run on
cohort 12, which was selected and analyzed in sequential development work. It
cannot provide confirmatory or population evidence.

The user waived human review as a prerequisite for local exploratory work.
This does not imply review occurred. The claim-to-window relation remains
unreviewed and the result must not be called human-validated faithfulness.
Protected videos and detailed receipts remain local; rights are unverified.

## Frozen design

- Inputs: the exact three rows from cohort 12's existing selected-window
  receipt, packet, cohort lock, and pair filter. No row substitution.
- Model: pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
  eager attention, offline local checkpoint.
- CPU preflight: decode the same 32 sampled frames and verify source-video and
  PTS hashes; reconstruct the exact test QA prompt; verify A–E continuation IDs
  against the multimodal processor suffixes; bind processor grid and whole
  temporal patch groups. The selected window is the frozen contiguous 8-frame
  chunk. Its control is the nearest disjoint contiguous group window of equal
  size, ties to the lower group index. Reject nonaligned or unequal mappings.
- Per row, first run one unmodified full-video forward. From that same pass,
  freeze the full-video top A–E label and runner-up foil, and select the top
  `ceil(0.10*T)` temporal groups by question-token attention aggregated over
  layers 5–16 (ties follow the existing selector's fixed order).
- Then run exactly four final-answer-query forwards with unchanged full-video
  pixels and prompt: baseline evidence-key ablation, baseline control-key
  ablation, `+ln(2)` bias on selected QTA keys plus evidence-key ablation, and
  the same bias plus control-key ablation. The intervention and ablation are
  applied to the eager attention mask in layers 5–16. Verify hook values and
  one call per layer.
- For each condition, calculate the fixed-prediction margin as `LP(p)-LP(q)`.
  Define `F_base = margin(control ablation) - margin(evidence ablation)` and
  `F_bias` analogously; primary descriptive quantity is `DeltaF = F_bias -
  F_base` in nats. Positive DeltaF means the tested bias increased selective
  answer-margin dependence on the machine-selected window versus its control.
- Requested workload: 3 rows, 15 forwards total, one model load. Require at
  least 12 GiB free VRAM and 8 GiB available host memory before loading, and at
  at least 2 GiB VRAM / 3 GiB host memory before each CPU-to-GPU batch transfer
  and every forward. The first row is
  the in-process smoke; continue to the remaining two only if all five first-
  row branches pass their input, logit, and hook checks.

## Interpretation limits

This is a mechanistic sensitivity contrast for one Qwen checkpoint, a coarse
global intervention, one machine-selected unreviewed window, and three
sequentially analyzed test rows. The key ablation is an internal attention-mask
intervention, not pixel deletion. It does not establish claim truth, semantic
support, causal generation-time use, improvement in general faithfulness, or
thesis-level efficacy. The cohort is post-hoc; no CI, significance test, or
population estimate is planned. Compare the result with the near-zero
cohort-12 post-hoc full-prediction pixel-mask diagnostic without treating the
different interventions as directly interchangeable.

## CPU implementation and preflight

Runner: `scripts/run_qta_fixed_prediction_key_ablation.py`. Geometry helper:
`scripts/qwen_frozen_frame_window_groups.py`. The helper requires exact complete
temporal-patch alignment, equal frame/key budget, disjoint windows, valid PTS
length/order, and exact video placeholder token geometry. The runner verifies
the input packet/selection/lock/pair hashes against the completed cohort-12 QA
receipt before scoring.

Focused CPU suites passed **37/37** across new arithmetic/control and frame-to-
temporal-key tests plus existing attention capture, multiquery intervention,
and selector tests.
Python compilation and scoped whitespace checks passed. Actual local processor
preflight passed all three rows: 32 sampled frames, temporal patch size 2,
16 temporal groups, exact 8-frame selected/control windows mapped to 4 groups
each, and all five A–E processor suffix checks per row. No model weights or GPU
forwards were used in this preflight.

## GPU execution and result

The final successful local run used
`data/private_nextgqa/qta_fixed_prediction_key_ablation_cohort12_20261021_03.json`
(SHA-256 `ab79530682de6b78e56b8d6d73651f7a1f5e7e07c6bb0d8079916917a33bfe09`,
mode 0600). It recorded 3/3 rows and 15/15 attempted/completed forwards on an
RTX 5070 Ti with 4-bit weights/BF16 compute; peak allocated VRAM was 11.17 GiB
and elapsed time was 97.43 seconds. The receipt's 12 source hashes and prior
QA receipt hash were independently checked against current local files. An
independent arithmetic recomputation from the five-choice branch vectors
matched the saved `DeltaF` values and mean.

| Frozen row | F baseline (nat) | F with +ln(2) QTA bias (nat) | DeltaF (nat) |
|---|---:|---:|---:|
| 1 | −0.1250 | approximately 0 | +0.1250 |
| 2 | −0.1250 | approximately 0 | +0.1250 |
| 3 | −0.1250 | −0.2500 | −0.1250 |

Finite-cohort mean DeltaF was +0.04167 nat (median +0.1250; range −0.1250 to +0.1250;
2/3 rows positive). This is mixed descriptive evidence that the
global visual-key bias changed the fixed prediction's margin dependence on
this machine-selected window relative to its matched control for these rows.
The result does not establish a reliable increase: the sign differs in one
row, the cohort is tiny and post-hoc, and no population inference is warranted.
No gold answer was used to define the fixed prediction or contrast.

Two earlier attempts were preserved as mode-0600 failure receipts and not
overwritten. Attempt 1 stopped after one completed full-video forward because
the runner passed compact captured rows to the full-matrix selector. Attempt 2
stopped during its fourth attempted forward (three completed) because the hook
checker compared a BF16-rounded mask value to exact FP32 `ln(2)`. The runner
was repaired, covered by 37 focused CPU tests, and independently audited
before attempt 3. These failed attempts are not included in the successful
15-forward cohort result.

Human-review collection was waived as a prerequisite by the user, but the
receipt explicitly records that no review occurred. This remains an
unreviewed machine-selected interval proxy; direct, human-validated
faithfulness evidence is still 0/3. Detailed protected receipts remain local
and unpublished; dataset rights status remains unverified. This exploratory
result alone does not justify a thesis-level efficacy claim.

## Known-exposure-filtered cohort 14 follow-up (2026-10-21)

Following the user's waiver of human-review collection as an exploratory-run
precondition, the existing fixed-prediction key-ablation design was applied to
cohort14. No judgment was imputed: actual claim/window review remains 0/3.
The same frozen 3B-selected 8-frame claim window, nearest disjoint equal-group
control, QTA top-10%-group selection, layers 5–16, and fixed `ln(2)` visual-key
bias were used. A full-video pass fixed each row's prediction and foil; four
ablation passes measured baseline and QTA-biased dependence.

The runner now accepts explicit selection/lock/pair/packet/prior-QA paths under
`data/private_nextgqa/`, validates their owner-private bindings, and supports a
one-row 5-forward smoke with no cohort aggregate. Focused tests cover private
path validation and intervention arithmetic. The CPU preflight checked all
three rows; the smoke completed 5/5 forwards; the full GPU run completed
15/15. Hardware was an RTX 5070 Ti, 4-bit Qwen2.5-VL-7B with BF16 compute,
11.12 GiB peak VRAM allocation, and 317.1 seconds elapsed.

| Row | F baseline (nat) | F QTA (nat) | DeltaF QTA − baseline (nat) |
|---:|---:|---:|---:|
| 1 | +0.125000 | +0.250000 | +0.1249999 |
| 2 | 0.000000 | 0.000000 | 0.0000000 |
| 3 | +0.125000 | 0.000000 | −0.1250000 |

Mean DeltaF was **−0.000000045 nat**, median 0, range **[−0.125,
+0.1249999]**, and 1/3 rows were positive. The independent auditor verified
all seven input hashes, 12 source hashes, five forwards per row, each row's
fixed prediction/foil against its full-video vector, branch margins, and
aggregate arithmetic. The private receipt is
`qta_fixed_prediction_key_ablation_cohort14_20261021_01.json`, SHA-256
`b9573dd3c6f7dafad14b62d397066d8f7c6503686f5ca83ec427fd3cb715038d`.

This shows no average internal-QTA effect on the prespecified finite-cohort
DeltaF here. It is distinct from the pixel-mask QA-margin estimand in the
claim-likelihood log; the two should not be pooled. It is not direct
faithfulness because the selected window is automated and unreviewed, the
global key bias is not claim-specific, and n=3 supports no population
inference.

### v2_short prompt cohort 29

This is a new claim-generation prompt stratum and should not be pooled with
the preceding v1 cohorts. QTA completed 15/15 with DeltaF [0, approximately
0, approximately 0] nat (mean 0); matched sham completed 6/6 with QTA-minus-
sham [+0.125, −0.125, 0] nat (mean 0). Receipts:
`qta_fixed_prediction_key_ablation_cohort29.json`, SHA-256
`5e3dabda72b092efaedb4e0f583c244175fa59def3616a1b13a6eaa1ef7b2c11`; sham
`qta_hash_matched_sham_ablation_cohort29.json`, SHA-256
`57a143421eabd79c1011bcfa9f5eafc2c848701fb0e7d55bdd43846a22b40a21`. Both
are owner-only mode 0600. Independent audit verified all receipt hashes,
bindings, forwards, and arithmetic. The audit did not inspect videos or claim
truth; human review remains unperformed. This is exploratory machine-score
evidence and does not show internal faithfulness improvement.

### Overlap-matched hash-sham comparison

To test whether the QTA groups were more selective than equally sized
arbitrary keys, a deterministic SHA-256-ranked sham was chosen for each row,
matching the QTA group's overlap counts with the evidence/control groups and
excluding the QTA set itself. The runner was parameterized to bind the cohort14
anchor and received a one-row 2-forward smoke before the full run. The full
comparison completed 6/6 forwards in 52.7 seconds, at 11.12 GiB peak allocated
VRAM. The independent audit verified three row bindings, all seven upstream
input hashes, all 14 source hashes, one call per layer in the required layer
set, sham cardinality, and matched overlap counts.

| Row | DeltaF QTA (nat) | DeltaF sham (nat) | QTA − sham specificity (nat) |
|---:|---:|---:|---:|
| 1 | +0.1249999 | +0.1249995 | +0.000000377 |
| 2 | 0.0000000 | +0.000000060 | −0.000000060 |
| 3 | −0.1250000 | −0.1250000 | +0.000000030 |

QTA-minus-sham specificity averaged **+0.000000116 nat**, median
+0.000000030, range **[−0.000000060, +0.000000377]**, positive in 2/3 rows.
This is effectively zero at this numeric scale: the sham recreates the QTA
DeltaF almost exactly on all three items. It is evidence against specificity
for these rows under this intervention, not a universal null or confirmatory
claim. Receipt: `qta_hash_matched_sham_ablation_cohort14_20261021_01.json`,
SHA-256 `c2b81c29f5ccab113d28fd5f68c1498becc370e8ed2bc25a57a394a8353b9509`.

The combined focused CPU suite passed **33/33** after these changes. Both
private result receipts were audited but not staged or published. Current
GPU-based evidence on cohort14 is convergent but exploratory: pixel-mask
selected-window D3 ≈0, QA-interval-control D3 ≈0, internal QTA DeltaF ≈0, and
QTA-versus-sham specificity ≈0. The next useful step is to freeze the same
endpoints and parameters on a separate known-exposure-filtered cohort before
claim generation/scoring, then report it as a sequential exploratory follow-up
without tuning from these outcomes. Human review remains waived as a gate but
unperformed; rights remain unverified; no protected artifacts may be shared.

### Cohort 15 claim-generation outcome

The frozen cohort's pinned 7B `v2_short` generator processed 3/3 requested
videos from 16 frames but produced structurally valid claims for only 2/3.
The failed row's parser code is `claim_missing`. This is a data-generation
failure, separate from the waived human-review collection gate. No row was
replaced, and no 3B window likelihood, QA-margin, QTA, or sham scoring was run
for cohort15. The owner-only partial packet is
`claim_likelihood_claims_remaining_20261021_15.json`, SHA-256
`5f997b5fd9227bd67bdd158f625b6dfb05e421b9a631938b4f11a1dc3164b691`, mode
0600. Review remains unperformed and no faithfulness score was computed.

The frozen cohort is now exposed for subsequent sampling even though this
generation failed. Next action: add this packet to the known-local exposure
union, freeze a new outcome-blind cohort under the unchanged protocol, and
continue only if a complete 3/3 claim packet is produced. Do not salvage a
partial-cohort mean or change prompts on these already observed rows.

## Cohort 15 sequential follow-up plan (frozen before claim generation, 2026-10-21)

This is a sequential exploratory follow-up after inspecting cohort14; it is not
a confirmatory replication. A new exposure union scanned all 825 local private
JSON files, including 203 recognized claim-generation/scoring receipts and 74
unique media digests. Its inventory SHA-256 is
`443d326eb768993188dfda65c02a8d852e3bd1b3d66bc8f50b309e4bfb62f801` and
manifest SHA-256 is
`99a8915d2f61b82d131025bf68b3db45366ab9882d35b4210311efb1af042937`.
Before claim generation or scoring, the freezer scanned 29 candidates and
selected 3 videos with zero overlap against that known local union. The
owner-only cohort lock SHA-256 is
`2133c8bb5105194deb0e0c96b974a9c392d20c2064ea9b026002d1ba49e052b0`;
pair-filter raw SHA-256 is
`56a483bf1ab78ef1b6d436df93cb97896de138d0c26f0448149099036ed8a062`.
These hashes and all three lock-source hashes were independently recomputed
before generation.

Keep the cohort14 procedures fixed: pinned 7B `v2_short` claim generation on
16 frames; pinned 3B teacher-forced likelihood over four equal 8-frame chunks,
choosing earliest maximum; 7B QA margins for full, selected-window mask, and
equal-count control (9 forwards); same-cohort benchmark QA-interval control (9
forwards); fixed full-video-prediction internal QTA key ablation at layers
5–16 with `ln(2)` bias (15 forwards); and overlap-matched hash sham (6
forwards). Smoke one row before each full scorer. Require all 3 structured
claims and complete cohort receipts; do not replace or average incomplete rows.
Keep every detailed receipt owner-private, preserve the review status as
unperformed, and report n=3 descriptively without inference. This plan and
cohort were frozen after cohort14 results, so even matching outcomes remain
sequential exploratory evidence. Rights status remains unverified.

## Cohort 16 sequential follow-up (2026-10-21)

The frozen cohort16 generation and GPU results are also summarized in
`2026-10-21_claim_likelihood_selected_window_prediction_dependence.md`. This
is an unchanged-protocol sequential exploration, not a confirmatory replication.
Claim generation was structured 3/3; the 3B selector completed 12/12 window
forwards. The 7B QA-mask endpoints each passed a one-row smoke and completed
9/9 forwards. The 7B fixed-prediction QTA passed a 5-forward smoke and
completed 15/15 forwards; hash-sham passed a 2-forward smoke and completed
6/6 forwards.

QTA DeltaF (QTA key-intervention F minus baseline F) was +0.125,
approximately 0, and +0.250 nat. Finite-cohort mean +0.125, median +0.125,
range approximately 0 to +0.250; 2/3 positive. The QTA-minus-overlap-matched
sham specificity contrast was +0.250, approximately 0, and +0.125 nat; mean
+0.125, median +0.125, range approximately 0 to +0.250; 2/3 positive. The
sham therefore does not falsify all internal sensitivity, but the finite
three-row set also does not show stable separation from sham. These are
descriptive values only; no interval or population effect is estimated.

The separate QA-margin endpoints were +2.083 nat mean for selected-window
mask versus matched temporal control (3/3 positive), and +1.208 nat mean for
benchmark QA-interval mask versus control (1/3 positive). These are not the
QTA outcome and cannot be used as corroborating measurements of the same
construct. All detailed receipts are mode 0600 under `data/private_nextgqa/`;
review remains unperformed, rights unverified, and nothing was published.
Independent audit is pending.

Next action: accept the independent hash/arithmetic audit, then continue the
fixed parameters on the next frozen cohort. No prompt, layer, bias, window,
subgroup or endpoint tuning is allowed based on these results.

### Cohort 17 sequential outcome

The fixed-prediction QTA run passed the 5-forward smoke and completed all
15/15 forwards. Per-row DeltaF was −0.125, approximately 0, and +0.125 nat;
the mean was +0.000000145 nat and 1/3 rows positive. The 6-forward
overlap-matched sham run passed its 2-forward smoke; QTA-minus-sham contrasts
were 0, +0.125, and +0.000000236 nat (mean +0.041667; 2/3 positive). This
near-zero small-cohort result does not support a stable positive effect, and
the comparator does not resolve specificity. Full values and hashes are in
`2026-10-21_claim_likelihood_selected_window_prediction_dependence.md`.
Independent audit verified all nine cohort17 artifacts' modes, hashes,
bindings, forward counts and numerical summaries. It did not reopen the video
bytes; human review remains unperformed and rights status unverified.

### Cohorts 20 and 21

Cohort20's QTA mean DeltaF was −0.0417 nat (0/3 positive); its QTA-minus-sham
contrast was −0.0417 nat (1/3 positive). Independent audit passed the full
receipt chain and arithmetic. The failed initial selector output had zero
forwards but lacked its own source/input hash map; downstream used the
separately successful 12/12 retry receipt. Cohort21 QTA completed 15/15 with
mean DeltaF −0.0833 nat (0/3 positive); sham specificity mean +0.0417 nat
(1/3 positive), completed 6/6. Cohort21 audit is pending. Both cohorts remain
local, exploratory, and unreviewed; neither supports an SSS faithfulness claim.

### Cohort 22

The unchanged protocol completed 15/15 QTA forwards and 6/6 matched-sham
forwards after smoke. DeltaF values were approximately 0, approximately 0,
and +0.000000294 nat (mean +0.000000098 nat; the positive value is at
floating-point scale). QTA-minus-sham specificity was −0.125, approximately
0, and approximately 0 nat (mean −0.0417; 0/3 positive). Receipts are
`qta_fixed_prediction_key_ablation_cohort22.json` (SHA-256
`c6dd515638f1410403a17ea2845c00376c96a3338adfc680bb7bf2077ebd19e2`) and
`qta_hash_matched_sham_ablation_cohort22.json` (SHA-256
`7e6ec1dd221bd6f1a48ae4be570a51fd861e4b6102ec4c7219ea6ac6f25f7418`), both
mode 0600. Claim truth and selected evidence remain unreviewed. This does not
support a meaningful internal-faithfulness improvement; it is a three-item
exploratory machine-score proxy. Independent audit is pending.

Independent audit subsequently verified cohort23's receipt bindings, 15/15
QTA and 6/6 sham forwards, arithmetic, and owner-only modes. Per-row DeltaF
was +0.125, approximately 0, and −0.125 nat (mean −0.000000201 nat); the
QTA-minus-sham contrasts were +0.125, +0.125, and −0.250 nat (mean
−0.000000160 nat). Both averages cancel to numerical zero, with no consistent
positive finite-cohort effect. The lock and packet cohort fingerprints differ
because the lock is built on a 16-frame temporal grid and packet generation on
32 frames; media/question/pair identities match for all rows. Video bytes and
claim truth were not independently inspected, and the human review remains
unperformed. These results do not establish faithful explanations.

### Cohort 24

QTA completed 15/15 and sham completed 6/6 after their respective smoke runs.
DeltaF was +0.125, 0, and −0.125 nat (mean −0.000000007 nat); QTA-minus-sham
was approximately 0, +0.125, and −0.125 nat (mean +0.000000002 nat). Receipts:
`qta_fixed_prediction_key_ablation_cohort24.json`, SHA-256
`d9495ea8bf58db81db24d8b54cf9ddb464ced3450f24ab7f05fe5764fa296737`; and
`qta_hash_matched_sham_ablation_cohort24.json`, SHA-256
`fb79f6ba4993bfd4ba6aaa6a1d913e2bf90a29e3fcdad5ab064ec0d5e3b904f8`.
Both are mode 0600. The positive selected-window QA D3 in this cohort
(+1.292 nat) did not translate to a positive QTA ΔF or sham-specificity
contrast. Across eight complete cohorts (16–18, 20–24), the descriptive mean
of cohort-level QTA DeltaF means is approximately zero; only cohort16 has a
meaningful positive mean, with negative means in 20–21 and numerical zero in
the other five. No population inference is warranted. Independent audit passed
mode, hash, row-binding, count, and arithmetic checks; claim truth remains
unreviewed and the receipt explicitly records review collection as waived
rather than completed.

### Cohort 26

QTA completed 15/15 and the overlap-matched sham 6/6 after smoke. DeltaF was
+0.125, +0.250, and approximately 0 nat (mean +0.125; 2/3 positive).
QTA-minus-sham was approximately 0, +0.125, and +0.125 nat (mean +0.083;
2/3 positive). QTA receipt SHA-256
`cbf4c811cb13428298b6f489b8011980d83e6fe70c8359cfa592586b0c60abe0`; sham
SHA-256 `d55a63567e736ca2a9403b4739200b6c664f828746a408a02a24550919aa4bd6`.
Both are mode 0600. Independent audit verified the full receipt chain,
rowwise hashes, forwards, and arithmetic; it did not inspect video bytes or
claim truth. This is a positive 3-item exploratory result of small magnitude,
not confirmation of internal faithfulness. Across nine complete cohorts
(16–18, 20–24, 26), only cohorts16 and 26 have meaningfully positive QTA
means; the other results are negative or numerically zero. The simple
descriptive average of cohort means is about +0.014 nat, with no population
inference.

### Primary-agent row-level synthesis of cohorts 16–18, 20–24, 26

To quantify the pattern without relying on the cohort-level prose above, the
primary agent independently recomputed the fixed-prediction margins from each
QTA receipt's five-choice likelihood vectors and recomputed sham specificity
from each matched-sham receipt. All nine QTA receipts had 3/3 rows and 15/15
forwards; their sham receipts had 3/3 rows and 6/6 forwards. QTA and sham
row-token plus media digests matched within each pair, all receipts were mode
0600, and the 27 row tokens were unique across these nine cohorts.

Across these **27 sequential exploratory rows**, QTA `DeltaF` had mean
`+0.01389` nat, median approximately `0`, range `[-0.1250,+0.2500]`, and 7/27
values greater than `+0.001` nat. The paired QTA-minus-sham specificity
contrast had mean `+0.02315` nat, median approximately `0`, range
`[-0.2500,+0.2500]`, and 11/27 values greater than `+0.001` nat. These are
descriptive only: the cohorts were collected sequentially after repeated
development analyses, the QTA/sham outcome itself is quantized at about
0.125-nat increments in many rows, and the underlying evidence windows and
claim-to-frame support were machine-selected/unreviewed. No p-value or
population confidence interval is warranted. This synthesis strengthens the
current negative thesis lead: a coarse global temporal-key bias has not shown
a stable or sham-specific increase in selective prediction dependence under
this setup. It is still not a direct, semantically validated faithfulness
measurement.

An independent audit then verified the complete receipt chains: requested
input/source hashes matched current files; each sham receipt bound to its QTA
anchor; all 27 paired rows matched on row token, video digest, input-batch
digest, and frozen prediction/foil; and all 27 source videos were distinct.
The auditor recomputed `F_baseline`, `F_QTA`, and `F_sham` from the five-choice
branch vectors and reproduced both effect contrasts. Its three-way sign
classification (treating `|DeltaF| <= 1e-6` nat as numerical zero) was QTA:
7 positive, 6 negative, 14 near zero; QTA-minus-sham specificity: 11
positive, 7 negative, 9 near zero. No arithmetic, binding, or mode issue was
found. The independent auditor did not inspect claim truth or video semantics.
