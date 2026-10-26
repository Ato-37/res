# Generated-claim object-region pixel intervention (2026-10-22)

## Purpose and frozen design

Test whether a fixed Qwen2.5-VL-generated sentence about an annotated object is
more sensitive to removing that object's annotated image region than to removing
an equal-shape region elsewhere in the same sampled frame, and whether a fixed
internal bbox-key bias changes that selectivity. This is an exploratory
generated-text dependence experiment. Independent semantic support review is
needed before calling it direct explanation faithfulness.

The metadata-only cohort was frozen before any outcomes at
`data/private_minerva/minerva_ego_object_pixel_generated_cohort_fresh3_20261022.json`
(SHA-256 `97edad28501d9a1ca8b134f80f485fe30bdd0cf0f4b62a0e9b481e01c41bd6cc`).
It requests three distinct videos from 30 eligible after excluding nine video
IDs used in earlier Minerva model outputs or initial smoke. All three selected
rows remain in the denominator; a failure is recorded, never replaced after
looking at outcomes. The media and annotations remain private and local. Rights
status is unverified under the user's local-only authorization.

Freeze these choices before GPU scoring:

- Use the same fixed Qwen2.5-VL-7B checkpoint, BF16/4-bit compute, 64 frames at
  nominal 2 fps, and annotation-centered clip sampling. A CPU gate must verify
  exact source frame/PTS, Qwen temporal grid, source bbox clipping, and distinct
  processor pixel tensors for all conditions. If nonuniform source sampling
  makes Qwen's synthetic `sample_fps` clock inaccurate, stop and use the
  already-audited uniform focal-clip sampler rather than pretending timestamps
  are exact.
- Generate one deterministic short movement sentence from the full unmodified
  clip with no key intervention. The prompt names the exact sampled annotation
  frame's clip-local time, so the requested movement is at that moment. Freeze
  its token IDs and text hash before any ablation score. Report empty, invalid,
  or non-object outputs as failures; do
  not select a better candidate after seeing scores. An automated structural
  check does not certify the sentence's truth.
- On the exact sampled annotation frame, replace the clipped bbox pixels with
  that frame's mean RGB. Use an equal-width/height, non-overlapping sham region
  selected deterministically from annotation metadata before scoring. The
  other member of the Qwen temporal pair stays unchanged. If no sham is
  feasible or if support/sham processed pixel tensors are identical, fail the
  row before model scoring.
- Score the same frozen sentence under support-fill and sham-fill inputs,
  crossed with three internal conditions: no key bias, +ln(2) on bbox-linked
  visual keys, and +ln(2) on equal-dose same-pair spatial sham keys at layers
  5–16. No visual-key deletion. For each internal condition, define
  `D = meanLP(sham-filled input) - meanLP(object-filled input)` in nats/token.
  The primary exploratory interaction is `D_bbox_bias - D_spatial_sham_bias`;
  `D_bbox_bias - D_no_bias` is secondary. Positive values would indicate
  greater selective target dependence under the bbox bias, within this exact
  proxy and cohort. All choices remain fixed regardless of the sign.
- Start with one row as a GPU wiring smoke after CPU validation. Expected work:
  one greedy generation plus six teacher-forced score forwards, about 10–12 GiB
  peak allocated VRAM based on adjacent runs and roughly one minute of scoring
  after model load. Use unique mode-0600 private receipts. Then run the remaining
  frozen rows only if the smoke proves the intended pixel/key interventions and
  provenance; do not tune the intervention based on smoke scores.

The GPU runner now writes a private failure receipt if generation or any of
the six score branches raises after the CPU gate. It records attempted and
completed call counts and the frozen target-token hash when one exists, without
publishing generated text. A failed row remains in the fixed denominator. The
CPU gate retains distinct failure modes separately in its cohort preflight.

## Status

The cohort is frozen. The generated-target runner and separate wiring smoke
passed their CPU and GPU gates. Official media download and frozen-cohort CPU
preflight are complete; the new cohort had one CPU-eligible row, whose baseline
generation was `Unclear.` and therefore failed the fixed target rule before
scoring. Human support review is 0/3 for this cohort and no direct faithfulness
score exists.

An early preflight while the transfer is incomplete requested all 3 frozen rows
and passed 0/3. Receipt
`minerva_ego_object_pixel_newcohort_cpu_preflight_20261022_01.json` has SHA-256
`de338efd3fd99641763ce4ae42e7327ed1ca01dafbc7ae5abfd900107c58b1d9`.
Rows 0 and 1 failed `video_file_missing`; row 2, whose official media and time
map were present, failed `rationale_mask_frame_missing_in_timestamp_bin`.
For row 2, all three supported rationale intervals had zero annotated mask
frames inside the interval. The nearest mask for the first interval was about
1.13 s before its start. The existing exact-anchor rule is therefore applied
as written: row 2 is excluded from GPU scoring and remains a recorded failure,
not silently moved to another frame or replaced by another video. This is a
source/annotation alignment failure, not a model result. Once transfer finishes,
rerun the same preflight to determine whether rows 0 and 1 are eligible.

The official six-file download later completed with MD5 checks
(2,077,108,254 total bytes; no reused files). The full new-cohort CPU
preflight receipt `_03` (SHA-256
`92fc60427b1d3515d86991c9ceb74fbd53743e94bb221e330b2d7322a6c208f5`)
requested 3 and passed 1: row 0 passed, while rows 1 and 2 had the same
`rationale_mask_frame_missing_in_timestamp_bin` failure. Row 0's full-prompt
CPU pixel receipt `_01` (SHA-256
`8e96f1e231d944844974099265e173b80d7c20db29f364d13d732ee676c2b7d4`)
passed with a `[32,22,22]` processor grid, distinct base/object-fill/sham-fill
pixel tensor hashes, and 6 bbox versus 6 sham visual keys. Its first GPU
attempt produced a continuation but failed the predeclared object-name
structural check before any of the six scores (`generated_target_missing_object_name`).
The mode-0600 failure receipt SHA-256 is
`f18bd0dba26458ede59ab43ab56579784748ce14fbc962ca9ed0ca9daf050f31`.
Thus this fresh cohort yielded 0/3 completed likelihood contrasts, not a
positive or negative effect estimate.

For a separate post-exposure feasibility extension of the old focal32s cohort,
rows 1 and 2 both passed full-prompt CPU pixel gates (6 and 4 matched
bbox/sham visual keys, respectively), but each GPU generation failed the same
object-name check before scoring. Together with the earlier successful row-0
wiring smoke, this old cohort has 1/3 completed exploratory contrasts. The
extension was chosen after observing the row-0 smoke sign, so it cannot be
called confirmatory. Failure receipts are `oldcohort_row1_gpu_20261022_01.json`
(SHA prefix `4a602425ea44a6eb`) and `oldcohort_row2_gpu_20261022_01.json`
(SHA prefix `f43efbb4469b6381`); both record one completed generation and zero
teacher-forced scores.

The fixed prompt permits a literal `Unclear` when movement is not visible,
whereas the target validator requires the object name in every answer. The
three target failures could be this protocol conflict or model-generated
pronouns/other wording; the original failure receipts retained only token
hashes, so the text is unknown. The runner now keeps rejected generated text
and the exception code in its local mode-0600 failure receipt for diagnosis;
this does not alter the success score definition. Focused unit tests pass
22/22 and compile checks pass. Any rerun uses a new CPU receipt bound to the
new source hash and a unique GPU output, and remains explicitly diagnostic.

The diagnostic rerun of fresh row 0 confirmed the generated text was exactly
`Unclear.`; the private failure receipt `_02` has SHA-256
`b445bca9a74e01bc3c0a99bb68933fe8fe1aa4f519f5199f46967cfd1e1db014`
and records zero score calls. This resolves that row's structural failure as
the prompt's permitted uncertainty answer, not a validator crash. It does not
justify scoring `Unclear.` as an object explanation. A source-only audit found
the success scoring path unchanged and the diagnostic text confined to the
mode-0600 private receipt. A restricted failure-code field was added to avoid
copying arbitrary exception messages; the focused runner tests passed 17/17.

A separate private Japanese 64-frame review HTML was built for the old cohort's
one successful generated sentence, with its other two rows explicitly marked
GPU-generation attrition. It is
`minerva_ego_object_pixel_oldcohort_review_ja_20261022_01.html`, mode 0600,
SHA-256 `c522f7f30f8e7fe53fc9989c51fa55ce229ebab41f71d9bfcdd4ff02890ccb4c`,
canonical artifact fingerprint
`7a391c956c68bb21580dbbd12562cf098e8bd870ab4a876034cec5352d8532f3`.
The builder binds the rendered claim, question, 64 JPEG frames, source indices
and PTS to the GPU/CPU receipts and official media; it was tested 6/6 and its
page fingerprint was independently recomputed. No human response has been
collected, so this is review readiness, not direct faithfulness evidence.

Because the official transfer may take time, a separate **wiring smoke** may
use row 1 of the earlier locally available focal32s cohort
(`minerva_ego_generated_grounding_cohort_fresh3_20261022.json`). That video
already has a dataset-rationale likelihood outcome, so its new generated-text
pixel result is post-exposure technical evidence only and is excluded from the
new three-row analysis. The smoke tests the exact runner and resource path;
its score sign cannot change the frozen cohort, intervention, or metric.

The separate smoke cohort's current CPU source/media/bbox preflight completed
3/3 in 89.04 s, receipt
`minerva_ego_object_pixel_smoke_cohort_cpu_preflight_20261022_01.json`
(SHA-256 `1e55d9616d19d5ec718d48a1e201069fc6b99daaeecbdc0db2ca3d10fca6e1e4`,
private mode 0600). All three had an exact annotated source frame in the
64-frame input and a same-time Qwen temporal group. The stored maximum source
clock deviations were 0, ~1.8e-15, and ~1.4e-14 s, and the bbox mapped to
9, 6, and 4 merged visual keys respectively. This establishes source-frame
geometry only; the support/sham pixel-tensor gate and model scoring remain
pending.

The updated full-prompt CPU pixel gate for smoke row 0 is receipt `_03`
(SHA-256 `b9a8d0ee2753a816eba64de885fe311035ee0d39b712e2ce3e1dcaf0debba715`,
mode 0600), bound to current runner SHA-256
`ccc617fc34ea139faa67f73deb235d684d852beb60ac0071c2162d9876e80319`.
The first two CPU receipts were superseded by the pre-outcome anchor-time and
failure-ledger source fixes. `_03` records anchor clip time 16.0 s, a
`[32,22,22]` video grid, 9 bbox keys and 9 spatial-sham keys, equal 70,587
source pixels in each box, three distinct processed pixel tensor hashes, and
identical nonvisual inputs and prompt token IDs. It is an engineering gate,
not a semantic support label.

### Wiring-smoke GPU result (separate, post-exposure video)

One local GPU smoke completed with the pinned Qwen2.5-VL-7B checkpoint, 4-bit
BF16 compute and eager attention. Receipt
`minerva_ego_object_pixel_smoke_row0_gpu_20261022_01.json` has SHA-256
`48aea122f6f272173d78423c223b20a188ab7ac9ac0d7acc07679ab147dc7635`,
mode 0600, and binds the source runner and CPU receipt digests. One greedy
generation produced a 9-token continuation, fixed before all six teacher-
forced scores; all 6/6 branches completed. The mean log-probabilities and
predeclared differences (nat/token) are:

| Internal key condition | Object-box fill | Spatial-sham fill | D = sham − object |
|---|---:|---:|---:|
| No bias | −0.911056 | −0.918019 | −0.006963 |
| Bbox-key +ln(2) | −0.906768 | −0.853381 | +0.053387 |
| Spatial-sham-key +ln(2) | −0.927385 | −0.932376 | −0.004991 |

The primary interaction is `+0.058378 nat/token`; secondary bbox-minus-no-bias
is `+0.060350 nat/token`. Peak allocated VRAM was 10.63 GiB and scoring plus
generation took 25.6 s after model load. These are one-video technical and
exploratory observations. This video was used in an earlier rationale-likelihood
run and is excluded from the new cohort. A generated short sentence is still
not independently verified as true or as supported by this one frame; the
sham matches shape/area and processor-key count, not visual salience. No CI or
test is reported. An independent read-only audit checked the source/output and
CPU bindings, all six frozen-target branch hashes, hook counts, and recomputed
contrasts; no arithmetic or provenance discrepancy was found. Direct faithfulness
progress remains **0 reviewed generated claims / 0 direct scores**.

## Next action

The current fresh cohort is closed at 0/3 completed contrasts; do not erase
its attrition or quietly reinterpret its missing score. Build a new cohort
with a pre-outcome mask-frame/time-bin eligibility gate, then verify exact
official time maps before GPU. Revise the generated-target prompt as a clearly
new exploratory protocol only after checking whether visible descriptions
are feasible at the selected annotated frame. Collect independent human
support judgments for successfully generated sentences using the private
64-frame page. Do not count the old `_06` NextGQA review response toward the
Minerva cohort.
