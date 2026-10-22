# Visible-object generated explanation v2 (2026-10-22)

## Purpose and fixed protocol

The v1 three-video fresh cohort produced no completed generated-explanation
contrast: two videos failed exact rationale-time/mask-frame alignment and the
one CPU-eligible video generated the permitted uncertainty answer `Unclear.`
instead of an object movement sentence. Its 0/3 coverage remains a recorded
failure, not a zero effect. This v2 is an outcome-informed **new exploratory**
feasibility design; it is not a confirmatory test of v1.

The v2 prompt asks Qwen2.5-VL-7B for one short, directly visible observation
of the named object at the nominal sampled anchor grid time, relevant to the question.
It asks the model to name the object and forbids unseen causes and the answer
letter. If the object is not clearly visible, the model may say `Unclear.`;
that exact sentinel is a **generation attrition** and receives zero likelihood
scores. Other scored sentences must pass the unchanged short, single-sentence,
object-name structural rules. This broadens the target from motion to visible
object observation; it may produce a static or trivial sentence and cannot
by itself establish semantic support or explanation faithfulness.

The object-bbox mean-fill versus equal-size non-overlapping spatial-sham fill,
Qwen temporal-pair mapping, fixed +ln(2) key bias at layers 5–16, same frozen
target token IDs in six scores, and metric are unchanged. For each eligible
row, `D_key = meanLP(sham-fill,key)-meanLP(object-fill,key)` in nat/token;
the primary exploratory interaction is `D_bbox_bias-D_spatial_sham_bias`,
secondary `D_bbox_bias-D_no_bias`. Report row values and attrition; n=3 allows
descriptive estimates only, no significance claim. The generated sentence
requires separate frame-grounded human review before any direct faithfulness
interpretation. Bbox is not segmentation, mean-fill may be OOD, and Qwen fuses
the edited frame with its unchanged neighbor.

## Frozen candidate cohort and gates

The metadata-only v2 selector first requires an action-recognition,
non-counterfactual, single-word object and a supported rationale line. It
requires an annotated object mask whose `frame_number/30` proxy time lies
strictly inside the first supported rationale interval (0.05 s edge margin)
and inside its object track. The proxy matched official time-map PTS to floating
precision in six locally available videos, but this does **not** prove it for
new files. Official MD5-checked video/time-map retrieval and the existing
source-frame/PTS CPU preflight are mandatory before inference. A frozen
selected row that fails remains in the denominator; there is no replacement.

Private cohort `minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json`
was frozen before v2 model output (SHA-256
`750af9b4303b4576954061e377bf6d16ef2f640f7a6b6027fc071c6c5ea0e678`,
mode 0600). Metadata counts: 1160 total rows; 267 one-reference, 266 exact
name joins, 168 action-recognition/non-counterfactual, 63 single-word object,
46 supported rationale lines, 23 strict proxy anchor rows. Eight of the 23
were removed by prior GPU-exposure video exclusions, leaving 15 rows in 15
videos and seven participant prefixes. Three deterministic videos were chosen
with one per participant prefix (P05, P06, P08). Their private row IDs and
annotations remain in the cohort file. The earlier v2 candidate files `_01`
and `_02` were superseded before media retrieval or v2 GPU output: `_01` used
a status incompatible with the runner; `_02` did not enforce participant
diversity. Neither is a result cohort.

Selector source SHA-256 is
`a8b8ee8700cff647196f511f18a7d9e590eef1ed8bb720e8064eb0163b389104`;
the initial v2 runner source SHA-256 was
`64eacfe3c9d9e5c20dbe7d92673cc239f30cba24369fc32748b047124b20e391`.
The v1 diagnostic runner was copied to
`scripts/snapshots/run_minerva_generated_object_pixel_pilot_20261022_v1.py`
before changing the prompt. The current source records
`prompt_protocol=visible_object_observation_v2_exploratory` in CPU/GPU
receipts. Focused selector/runner/pixel tests passed 23/23; compile check and
scoped diff check passed. No fresh-cohort v2 media or GPU run had occurred
at that freeze checkpoint.

An independent pre-GPU audit reproduced the cohort/source hashes and all
eligibility counts, but found a strict-proxy binding gap: the metadata freezer
chose the best mask across the first rationale line's intervals, while the
production anchor helper chooses the first interval with a qualifying actual
PTS mask. Before any new-cohort GPU work, the runner was amended to require
that the official-PTS anchor frame and interval equal the selected proxy
frame and interval, and that the actual PTS retain the frozen 0.05-second
interior margin. A mismatch is an explicit CPU-gate failure with no
replacement. The audited current runner SHA-256 is
`316fe7e0ad46243921c683b43c04307d4957216b91d68a0ed82da19ca67b86f3`;
focused selector/runner/pixel tests now pass 24/24, compile and scoped diff
checks pass. The frozen cohort remains unchanged. Two selected participants
have other exposed videos, so the cohort is fresh by video, not entirely by
participant.

While the official new media downloaded, an older exposed video was run as a
separate v2 wiring smoke. Its CPU pixel gate passed, but the baseline GPU
generation produced a malformed multi-line response rather than one short
object sentence (`generated_target_not_short_single_line`); no likelihood
score was attempted. The private CPU and failure receipt SHA-256s are
`1ce5994fa5b350696ad06de4300fbab32142a6a0aa37c32e99657979d3758252`
and `22dca807d3a061e826262f27eaf452ce68c7ec921a48911f24812857b82e1b3e`.
The exact rejected text is retained only in the local mode-0600 receipt.
This one post-exposure failure is a feasibility warning, not a v2 cohort
outcome and not a reason to change its frozen target rule after seeing it.

The nominal grid time in the prompt may differ from the decoded source-frame
PTS. For fresh row 0, the prompt said approximately 30.5 s in the clip while
the anchor frame's actual clip-local PTS was about 30.2667 s (difference
0.2333 s, within the predeclared 0.25 s grid-error gate). Keep this wording
and gate fixed for the remaining v2 cohort rows; do not describe the prompt
time as an exact PTS or change it after observing row-0's score.

Expected work before GPU: download exactly six official files for the three
selected videos, likely a few GiB and potentially tens of minutes based on
the prior three-video transfer, with 860 GiB local free disk. After the
three-row source/PTS preflight, run full-prompt CPU pixel gates only for
passing rows. A passing row's GPU run is one greedy generation plus six
teacher-forced scores, previously ~10.6 GiB peak allocated VRAM and ~35 s
after model load; allow a separate unique failure receipt for a sentinel or
other generation attrition. All media, generated text, receipts, and review
pages remain private and local; rights status is unverified under the user's
local-only authorization.

## Results and next action

An early preflight while the remaining media downloaded requested all 3 frozen
rows: row 0 passed and rows 1–2 were `video_file_missing`, which is pending
transfer rather than final attrition. Row 0's full-prompt CPU pixel gate passed
with a `[32,22,22]` Qwen grid, 25 bbox visual keys, 25 spatial-sham keys,
three distinct processed pixel-tensor hashes, and the exact selected proxy
frame/interval matching the official-PTS anchor. The existing 0.05-second
safe-interior rule was enforced before GPU. Private receipt paths are
`minerva_visible_object_v2_cohort_cpu_preflight_20261022_01.json` and
`minerva_visible_object_v2_fresh_row0_cpu_20261022_01.json` (SHA-256s
`467bdae10d4e0370d525ffd2960a39a5b373e91a6bb2c6efbf28f740e6c4eaba`
and `b6be257080d71a61c1a3814ccd3848cb16745235175a68cbcd4f740f284f84b6`).

Fresh row 0 then completed one greedy generation and all six teacher-forced
GPU score branches. Private mode-0600 GPU receipt
`minerva_visible_object_v2_fresh_row0_gpu_20261022_01.json` has SHA-256
`916f39e824846fdd80de8f485ebcf6151850759630f807a96d4afff38445c99e`;
all runs used the same eight-token generated continuation. The unchanged
`D = meanLP(sham-fill)-meanLP(object-fill)` values (nat/token) are:

| Internal key condition | Object fill | Spatial-sham fill | D |
|---|---:|---:|---:|
| No bias | −0.849599 | −0.816838 | +0.032761 |
| Bbox-key +ln(2) | −0.861549 | −0.819576 | +0.041973 |
| Spatial-sham-key +ln(2) | −0.852659 | −0.825768 | +0.026891 |

The primary bbox-minus-spatial-sham interaction is **+0.015082 nat/token**;
secondary bbox-minus-no-bias is **+0.009212 nat/token**. Peak allocated VRAM
was 10.65 GiB. This is one video, selected in a new exploratory cohort after
v1 failures, and the generated sentence is grammatically malformed. No human
support judgment exists yet. Consequently these numbers are conditional-
likelihood sensitivity diagnostics only, not a demonstrated grounding or
faithfulness improvement. No CI, test, or multiplicity claim is justified at
1/3 completed rows. An independent read-only audit reproduced the CPU/GPU,
cohort, preflight, selected-mask, target-token, hook-count, and six-score
bindings and arithmetic; it did not rehash the full raw media. No discrepancy
was found beyond the disclosed nominal-grid versus actual-PTS time difference.

The frozen six-file transfer completed (official MD5 checks, 2,576,180,367
bytes, no reused files). Final cohort source/PTS preflight `_02` passed 3/3;
all three full-prompt CPU pixel gates passed with 25, 4, and 6 matched
bbox/sham visual keys, respectively. Row 0's completed GPU receipt is bound
to the earlier partial preflight `_01` (1/3 passed while the other files were
missing); rows 1–2 bind final `_02`. The row-0 record, including source hashes,
selected mask, PTS, bbox mapping, and sampling, is bytewise equivalent in
both preflight JSON records. The different whole-receipt hashes remain in
each GPU/CPU provenance and must not be silently conflated.

Row 1 completed one nine-token generation and 6/6 scores. Row 2 generated a
malformed multi-line response, failed the fixed
`generated_target_not_short_single_line` rule and attempted 0/6 scores.
Thus GPU contrast coverage is **2/3 requested**, with one structural
generation failure retained in the denominator. Private row-1 GPU receipt
SHA-256 is `b009e0ed32eea1db1b05f9af30b75f6f586bf5abe643e63cbcdd21d275778d0f`;
row-2 failure receipt SHA-256 is
`3ec7c7fb68187c9050705095c0cfe3574649fa476842fb0bed27e91188f95811`.

| Frozen row | D no bias | D bbox-key | D spatial-sham-key | Primary interaction | Secondary interaction |
|---|---:|---:|---:|---:|---:|
| 0 | +0.032761 | +0.041973 | +0.026891 | +0.015082 | +0.009212 |
| 1 | −0.013739 | +0.000734 | +0.011261 | −0.010527 | +0.014474 |
| 2 | — | — | — | — | — |

All numbers are nat/token. Among the two completed rows, the primary mean is
**+0.002278** with a purely descriptive 95% t interval
**[−0.160419,+0.164974]** (df=1); the secondary mean is **+0.011843** with
interval **[−0.021581,+0.045267]**. The primary sign is 1 positive/1
negative. No confirmatory test or multiplicity-adjusted inference is reported:
n=2 makes the normality assumption and interval unstable, and v2 was designed
after observing v1 feasibility failures. The primary interval spans both
benefit and harm; it does not support a reliable improvement.

An independent read-only audit reproduced row-1 score arithmetic and all
source/receipt/proxy-anchor joins, confirmed 6/6 common-target forwards and
layers 5–16 hook counts, and classified row 2 as zero-score attrition. Row 0's
generated sentence is grammatically malformed; row 1's sentence is a short
static object observation whose support and question relevance remain
unreviewed. Neither score certifies claim truth or explanation faithfulness.
Human support review is **0/2 scored claims** and direct faithfulness progress
is **0 scored/reviewed v2 claims with validated semantic support**.

Next, build and verify one private Japanese 64-frame review page for the two
successful generated sentences, preserving row 2 as attrition and both
row-specific preflight hashes. Obtain human frame-grounded judgments without
showing the intervention scores in the page. Then decide whether a larger,
prospectively fixed follow-up is worthwhile; do not tune this two-row outcome
into a confirmatory claim. The old `_06` NextGQA response remains separate.

The private Japanese review page has now been built at
`data/private_minerva/minerva_visible_object_v2_review_ja_20261022_01.html`
(mode 0600; SHA-256
`eb6187c69893657f1ce440087d6437ed66a2bf119cb2f3eef8a0de5632eb8662`).
It displays two successful claims with 64 sampled source-frame thumbnails each,
retains row 2 as structural attrition, asks one nickname per reviewer, and
exports one combined local JSON after consecutive reviewers. The canonical
artifact fingerprint is
`38f682b967f0ead5e5fbc45817f899715c8e4320fda2873c5523507706966036`;
the builder SHA-256 is
`e74dcf1b8463516871df38a06463747d9d44b69c72f380d3014c49779c77437f`.
The builder binds row 0 to early preflight `_01` and rows 1–2 to final `_02`
only after checking each selected row record against final `_02`. Source,
frame-image, claim, question, receipt, and page hashes are embedded in its
private provenance. Focused builder tests passed 9/9; the emitted page's
fingerprint and JavaScript syntax were separately checked. An independent
read-only audit verified the page bytes, all 128 embedded thumbnail hashes,
two claim/receipt/row-specific-preflight joins, and row-2 failure binding;
it found no material binding gap. The page summarizes the row-2 failure as
the general exception class while the private receipt retains its exact
failure code. No actual reviewer has answered, so this handoff
does not increase the direct faithfulness count.

## Prospective exploratory generation comparison (fixed before new GPU outcomes)

The conditional-likelihood pilot above does not compare explanations produced
under intervention. A new local-only runner,
`scripts/run_minerva_visible_object_generation_bias_v1.py`, compares three
greedy generations on the **same unfilled 64-frame input** for every row in
the already frozen v2 cohort: no bias, bbox visual-key +ln(2), and equal-dose
spatial-sham visual-key +ln(2). The fixed layers are 5–16, the seed is
20261022, and the generation budget is 64 new tokens. The DynamicCache hook
must verify the prefill/decode mask geometry and exact bias on all selected
cells in every targeted layer; any failed branch remains recorded as failure.
The selected-key sets and all source/PTS/annotation bindings are rebuilt and
checked against the existing row-specific CPU receipts before model loading.

The first endpoints are mechanical validity (the previously fixed one-line,
object-name rule), exact text/token hashes, and branch attrition. Later blinded
human judgment of source-frame support is required for any semantic conclusion.
Because this is an outcome-informed exploratory cohort of three videos,
generation differences cannot be promoted to a confirmatory faithfulness
effect. No generated text or private annotation is eligible for publication.
Expected workload is three generation calls per video, roughly 9 total,
with approximately 11 GiB peak model/input VRAM from the preceding GPU run;
run row 0 as smoke before rows 1–2. GPU outcomes had **not** been collected
at the time this protocol was written.

### GPU outcome and interpretation

All three frozen videos completed all three greedy generation branches
(9/9 calls; 3/3 row receipts). The local mode-0600 receipts have SHA-256
`7b1548c0655ff280137a25a3fc26ef718df57bffa7714750cc1d0bee079c31e2`
(row 0), `27f8f57d721526a815128ec3641142e540264e34301ecf7d8cc6ee3b7e479221`
(row 1), and `09bde4b7beddd304d7ffbc36bf4fe70d2f15a2f53b147e0e02b231a41f9df67a`
(row 2). Each peak allocation was 10.60–10.67 GiB. The nominal +ln(2)
attention-mask edit is stored as an effective +0.69140625 in bfloat16;
the deviation is numerical rounding, not a different selected dose.

| Frozen row | Baseline mechanical status | Bbox-key mechanical status | Spatial-sham-key mechanical status | Text hashes: bbox vs sham |
|---|---|---|---|---|
| 0 | valid (8 tokens) | valid (11 tokens) | valid (11 tokens) | identical; both differ from baseline |
| 1 | valid (9 tokens) | valid (9 tokens) | valid (9 tokens) | identical; both equal baseline |
| 2 | invalid multiline (9 tokens) | invalid multiline (9 tokens) | valid (11 tokens) | different; bbox equals baseline |

The mechanical rule checks a single-line sentence and target-object mention;
it does not judge truth, relevance, or grammatical quality. Row 0's baseline
contains a grammatical defect; both biased branches produce the same revised
sentence. Row 2's spatial-sham output meets the mechanical rule but contains
another grammatical defect. These texts and original question/annotation
remain private. The number of mechanically valid generated branches is 2/3
baseline, 2/3 bbox-key, 3/3 spatial sham. Relative to baseline, bbox changed
text in 1/3 rows and spatial sham in 2/3; bbox and spatial sham differed in
1/3. There is **no observed bbox-specific validity gain**: the only row with
an increased validity is the spatial-sham branch. These are descriptive
within three already exposed exploratory rows, with no significance test.
Every treatment had 12 layer-hook calls per generated-token forward: row 0
132 calls and 3,300 verified mask cells in each treatment, row 1 108 calls
and 432 cells, row 2 bbox 108 calls/648 cells and sham 132 calls/792 cells.
All matched the effective bfloat16 delta. The independent source/receipt
audit verified all three rows, the hook implementation, and the aggregate
arithmetic. No human frame-grounded judgment has been
made for these new texts, so direct faithfulness remains unmeasured.

The selected bbox keys occupy only 25/4041, 4/4017, and 6/4046 positions
in the three prefill prompt sequences (approximately 0.62%, 0.10%, 0.15%).
As a **toy uniform-attention calculation only**, doubling their unnormalized
attention weight would change their total mass by approximately +0.61,
+0.10, and +0.15 percentage points, respectively. Actual attention is not
uniform and was not measured; this arithmetic is a dose-geometry hypothesis,
not an explanation of the observed null result or a warrant to tune bias
strength on these outcomes.

A separate CPU verifier recomputed the earlier two-row conditional-likelihood
summary from the private v2 cohort, official media/time maps, row-specific
preflights, CPU/GPU receipts, current source hashes, six branch audits, and
row-2 zero-score failure. Its privacy-minimized version wrote a derived
7,041-byte JSON to
`results/minerva_visible_object_v2_verified_summary_20261022_03.json`
(SHA-256 `c0b5404896fcf3d07b465a71b270313f4ccf0a2436c35e1c79ad1f81c6466bd1`).
The primary mean independently recomputed as +0.002277713 nat/token with
descriptive 95% t interval [−0.160418809,+0.164974234]. It contains hashes,
counts, and effects, with no raw claim text, question text, video ID, path, or
frame. The first real-media verifier run found a mistaken required field:
the completed GPU receipt does not repeat `manifest_sha256`; that hash is
instead bound in the CPU/preflight/official-media checks. The verifier was
corrected before the first successful output. Independent audit then found
that this `_01` summary did not bind each row's annotation-source hashes back
to the frozen cohort or record the summarizer source hash. Both provenance
checks were added before `_02`; a mutation test for row-2 source drift passes,
making the focused suite 6/6. The `_02` version was independently verified,
then `_03` removed linkable per-row video/time-map/row-key hashes from the
public summary while retaining their private verification and receipt hashes.
Both `_01` and `_02` are superseded; only `_03` should be published.

Next, independently verify the three-row receipts and produce a blinded
local review artifact for all distinct generated sentences, retaining the
condition mapping privately. A fresh prospective cohort would then be
needed to test a revised intervention; this three-row pattern argues against
claiming the current bbox-key bias improves explanation faithfulness.

The separate local CPU verifier
`scripts/summarize_minerva_generation_bias_v1.py` checked all three generation
receipts against the frozen cohort, matching row-specific CPU/preflight hashes,
current runner/hook source hashes, branch text hashes, mechanical classifications,
and per-token/layer/key mask-audit counts. It emitted a privacy-minimized
4,342-byte summary at
`results/minerva_generation_bias_v1_safe_summary_20261022_02.json`
(SHA-256 `119be12535dcbdc78b100a2591699eedcafd6797bd2f02e69f2e1bd8a592f0cc`).
It contains no raw text, video IDs, or media paths. Its 2/3, 2/3, 3/3
mechanical-validity counts and zero bbox-specific validity gains match the
private receipt audit. This verifier binds to previously checked media hashes
in CPU/GPU receipts; it does not itself rehash the 2.5 GB media files.
An independent audit of `_01` recomputed every count and identified missing
preflight-record/source checks in the reusable verifier. Those checks were
added before `_02`: each selected record must equal the final 3/3 preflight
record and pass, preflight source fields must match, and every CPU source hash
must match the current source file. The bbox-only validity count is explicitly
named as a **mechanical** diagnostic. `_01` is superseded and remains local.

One metadata-only follow-up cohort of three videos was frozen privately at
`data/private_minerva/minerva_metadata_proxy_anchor_cohort_v2_followup_20261022_01.json`
(SHA-256 `99044f12cfe6affca3371fd9192317b43dce5d0f83d39517c6af5148b1a4a44a`).
After exposure exclusions, 12 eligible videos from seven participant prefixes
remained. Its three selected videos are previously unexposed as videos, but
all three participant prefixes occur among older exposed videos; they are
therefore **not participant-fresh**. The cohort is still a frame-number/30
metadata proxy and has not passed official-media CPU preflight. It is a
candidate for a later prospectively fixed method comparison, not new efficacy
evidence. Do not download/run it merely to increase sample count under a
method that has not shown selective benefit.
The local annotation/association/mask source hashes match the frozen cohort;
rerunning the metadata selector read-only reproduced the exact counts,
selection, and exposure-exclusion set.

### Blinded local support-review handoff for generation branches

The private page
`data/private_minerva/minerva_visible_object_v2_generation_bias_review_ja_20261022_01.html`
was built from the three completed generation receipts (6,762,268 bytes,
mode 0600; SHA-256
`2eb5fa8912e510e0d641433d241a6c7812b33c8fc0883bc71817d5665a60a967`).
It displays each distinct generated text once per row: 2/1/2 items, five in
total. Each item has 64 exact sampled source-frame JPEGs, giving 320 embedded
images; duplicated texts across intervention branches are not presented twice.
The page does not reveal which attention condition produced a text or any
model score. Item-to-branch correspondence is confined to the separate private
mode-0600 mapping
`data/private_minerva/minerva_visible_object_v2_generation_bias_review_mapping_20261022_01.json`
(SHA-256 `13d680cb622ac47c2fd95268c0677b8ac881fa7f4c4ffba44f28ac82cfb7dbe5`).
Its canonical page fingerprint is
`01940e12cad3abefc56d685956642f701e899470928729f1e6e0c09f5fd28108`.
The builder source SHA-256 is
`4e03d9d5aa9577e7be991bad55689bd75321dbe27c2b7daedd6f668841611b21`.
Focused synthetic tests passed 4/4, Python compilation passed, and the
actual page's fingerprint, five-item/320 embedded image hashes, item/mapping
IDs, hidden condition labels, and JavaScript syntax were checked. A separate
independent read-only audit then reproduced the page/mapping/source-manifest
hashes, all five item/320 frame hashes and receipt joins, condition blinding,
and four focused tests. It found no critical integrity issue. No reviewer
responses have been collected;
direct semantic support and faithfulness remain unmeasured.

The local read-only response inspector
`scripts/inspect_minerva_generation_bias_review.py` is ready for a browser
download from any local directory. It verifies the page fingerprint, embedded
frame hashes, exact five-answer completeness, private mapping, and now all
mapping condition labels against the three original GPU generation receipts.
An independent audit found that a coverage-preserving branch-label swap in the
private mapping would previously pass; the receipt cross-check closes that
misattribution path. It passed against all five actual items. The page allows
reviewers to reuse a nickname, so the inspector retains each response session
and separately reports the number of distinct case-folded nicknames; neither
count proves distinct human identity. Focused builder/inspector tests passed
12/12. No response JSON for this page has been observed, so no new human
support counts or direct faithfulness result are claimed. To inspect a later
local export, use the private page and mapping paths above and the three
`minerva_visible_object_v2_generation_bias_row{0,1,2}_gpu_20261022_01.json`
receipts with `--gpu-receipt` in row order. Keep the response and private
inputs local; the inspector prints aggregate counts only.

The next internal-representation branch is a VISTA-inspired **exploratory**
single-block residual intervention described in
`docs/image_to_video_intervention_options_20261022.md`. A source-level
feasibility audit found that Qwen2.5-VL's decoder block output is a residual
Tensor and that only the final prompt-position vector needs capturing. It also
found that changing only the last block output at that position cannot affect
all tokens of a multi-token object name in a parallel teacher-forced pass.
Therefore the prototype must pre-fix one interior block and its vector dose
before any outcome inspection. First use an already-local row for a technical
GPU smoke; the three unexposed candidate videos remain unrun. Neither the
literature proposal nor the technical audit is efficacy evidence.
