# Saved-claim likelihood under QA-interval key bias (exploratory plan)

## Question and estimand

Does a fixed positive internal bias on visual tokens aligned to a machine-readable
NExT-GQA QA interval increase the saved claim's conditional token likelihood
sensitivity to that QA interval's pixels, relative to baseline and a disjoint
equal-budget key-bias sham? The target is the already frozen `v2_short` claim
from `data/private_nextgqa/claim_likelihood_claims_remaining_20261021_30.json`.
It is re-tokenized deterministically from parsed claim text because generation
token IDs were not stored.

For each row, use six teacher-forced conditions: three key-bias arms (none,
QA-window, deterministic sham-window) crossed with two black-pixel inputs (QA
window masked, equal-count temporal-control window masked). In layers 5–16,
add `ln(2)` to visual-key attention logits for every target-claim prediction
row in the relevant biased arm. There is no attention-key ablation. Define
`F_X = mean logP(claim | control-pixel-mask, X bias) - mean logP(claim |
QA-pixel-mask, X bias)`. The primary finite-cohort contrast is
`F_QA_bias - F_baseline`; the predeclared specificity diagnostic is
`F_QA_bias - F_sham_bias`. Positive values indicate a larger QA-mask penalty
under that bias arm.

The QA time span is expanded outward to complete Qwen temporal groups (the
pinned processor uses two sampled frames per group). A same-length contiguous
control group window is selected by nearest temporal-center distance among
disjoint windows, with lower starting index breaking ties. A third same-length
contiguous sham window is selected by SHA-256 rank using a fixed protocol seed;
it must be disjoint from both pixel-mask windows. The identical expanded QA
and control windows define black pixel masks and key-group positions. Rows with
more than one QA interval, missing sampled frames, unavailable controls, or
unavailable disjoint shams fail closed.

## Frozen inputs and verified CPU preparation

The private packet is frozen at three distinct test videos, Qwen2.5-VL-7B
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 32 saved PTS per row,
and `v2_short`; `claim_review_completed` is false. Rights remain unverified.
The runner verifies owner-only input mode, packet and dataset hashes, row/video
joins, QA intervals against `gsub_test.json`, question/options/answer against
`test.csv`, source-video SHA-256, exact 32-frame decoded PTS, and equal Qwen
temporal-group/key geometry across both masked pixel batches. Outputs are
mode 0600 in the ignored private data directory; overwrites are refused.

CPU-only geometry check on the actual frozen packet and pinned local processor
passed for all 3 rows: 32 frames became 16 temporal groups; expanded QA group
windows were `[9,10]`, `[1,2]`, `[9,10]`, nearest disjoint control windows
were `[7,8]`, `[3,4]`, `[7,8]`, and deterministic shams were `[3,4]`, `[8,9]`,
`[1,2]`. All three had distinct QA/control black-pixel tensors, equal key
budgets within each row, and shams disjoint from both mask windows. Focused
CPU tests passed 6/6; syntax compilation and scoped whitespace checks passed.
### GPU smoke 01 and independent audit

The first GPU smoke completed on one cohort30 row: 6/6 teacher-forced forwards,
14 claim tokens, 87.45 seconds end-to-end, RTX 5070 Ti / CUDA 12.8, and 10.86
GiB peak allocated VRAM. The private receipt is
`data/private_nextgqa/claim_text_qa_interval_key_bias_cohort30_smoke_20261021_01.json`
(mode 0600; SHA-256 `4e53f24b…`). An independent audit rechecked all five
input hashes, all 12 code-source hashes, each branch's 14 finite token scores,
the means and contrasts, and each hook observation for layers 5–16.

Observed on this one prior-selected row: `F_baseline = −0.00623` nat,
`F_QA-bias = +0.02787` nat, `F_sham-bias = +0.03101` nat; the primary
`F_QA-bias − F_baseline = +0.03411` nat, while specificity
`F_QA-bias − F_sham-bias = −0.00314` nat. This is an inconclusive single-row
exploratory signal; the QA-key bias did not outperform the sham bias here.
It does not establish improved faithfulness.

The independent audit found that full-mode smoke-receipt validation was too
shallow. The runner now requires a matching smoke-mode receipt, correct first
row/video/claim/token identity, exactly six attempted/completed branches, all
finite token scores, self-consistent means and primary arithmetic, the expected
hook count in every selected layer, and explicit non-faithfulness/review flags.
Two negative-path validator tests were added; focused tests now pass 8/8. This
source change invalidated smoke 01 as a full-run gate. A fresh final-source
smoke (`…smoke_20261021_02.json`, mode 0600) passed; its 6/6 forwards and matching
source/input/first-row hashes were independently checked before full mode.

### Full fixed-cohort GPU run

Full mode completed the frozen 3-row cohort: 18/18 forwards, RTX 5070 Ti,
CUDA 12.8, BF16 compute with 4-bit weights, 169.57 seconds end-to-end,
11.32 GiB peak allocated VRAM. The private result is
`data/private_nextgqa/claim_text_qa_interval_key_bias_cohort30_full_20261021_01.json`
(mode 0600; SHA-256 `b04a68dd…`). I independently recomputed all six branch
means from each saved token-level score list and recomputed every contrast;
all 18 hook records show 0 calls in baseline and exactly one call per selected
layer 5–16 in each biased branch. There were no failed forwards.

| Row | `F_baseline` | `F_QA-bias` | `F_sham-bias` | `Delta(QA−baseline)` | `QA−sham` |
|---:|---:|---:|---:|---:|---:|
| 1 | −0.00623 | +0.02787 | +0.03101 | +0.03411 | −0.00314 |
| 2 | −0.09623 | −0.02130 | −0.06605 | +0.07493 | +0.04475 |
| 3 | −0.23160 | −0.20942 | −0.23462 | +0.02218 | +0.02520 |

The fixed-cohort mean `Delta(QA−baseline)` is **+0.04374 nat per claim token**
(median +0.03411; range +0.02218 to +0.07493; positive 3/3). The mean
specificity contrast `F_QA-bias−F_sham-bias` is **+0.02227 nat/token**
(median +0.02520; range −0.00314 to +0.04475; positive 2/3). These are
descriptive values only; n=3 is previously selected development data, so no
population CI, significance test, or confirmatory claim is warranted. The
smoke row is repeated in the full run and is not an extra independent item.

This is a promising internal-intervention effect on a machine-aligned QA-window
conditional-likelihood proxy, not a direct faithfulness result. The QA span is
not claim-specific reviewed support, the cohort is post-selected, and the
black mask can be out of distribution. Direct human-validated faithfulness
remains 0/3 for this cohort. A thesis can report a bounded exploratory finding
that fixed +ln(2) key bias in layers 5–16 increased QA-window mask sensitivity
relative to baseline on all three items, alongside weaker/unstable sham
specificity; it cannot claim semantically faithful explanations or general
improvement.

### Same-resolution cohort29 exploratory replication

An additional CPU audit found cohort29 is a disjoint, 3-row `v2_short`, 32-frame
packet compatible with this frozen geometry. It had prior outcomes on other
metrics, so this is a retrospective exploratory replication, not confirmation.
The runner was generalized only to accept an owner-only private packet path and
to stream SHA-256 over all five pinned model weight shards. One first attempt
stopped before model load because the Hugging Face cache stores shard symlinks
in a sibling `blobs/` directory; it wrote a 0600 failure receipt and ran zero
forwards. The path check was corrected and unit tests still passed 8/8.

Cohort29 smoke then passed 6/6 forwards. The full run completed 18/18 forwards
in 181.3 seconds, with peak allocated VRAM 12.17 GiB. Its private receipt is
`data/private_nextgqa/claim_text_qa_interval_key_bias_cohort29_full_20261021_01.json`
(mode 0600; SHA-256 `aeee74cd…`); all five shard hashes, all code/input hashes,
token-level scores, hook counts, and arithmetic are recorded. The cohort values:

| Row | `F_baseline` | `F_QA-bias` | `F_sham-bias` | `Delta(QA−baseline)` | `QA−sham` |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.05158 | +0.03379 | +0.07591 | −0.01778 | −0.04211 |
| 2 | −0.04308 | +0.00581 | −0.06337 | +0.04889 | +0.06918 |
| 3 | +0.20532 | +0.08618 | +0.18358 | −0.11913 | −0.09739 |

The cohort29 mean `Delta(QA−baseline)` is **−0.02934 nat/token** (positive
1/3), and mean `QA−sham` is **−0.02344 nat/token** (positive 1/3). Across the
two non-overlapping 3-video packets (n=6 descriptive only), pooled mean
`Delta(QA−baseline)` is **+0.00720 nat/token**, range −0.11913 to +0.07493,
positive 4/6; pooled `QA−sham` is **−0.00059 nat/token**, range −0.09739 to
+0.06918, positive 3/6. The cohort30 pilot's 3/3 positive shifts did not
replicate in cohort29, and pooled specificity is essentially zero. This is
evidence against a reliable benefit from this generic intervention on this
proxy, not evidence against all internal interventions or faithfulness methods.
No CI or significance test is justified for these post-selected retrospective
development cohorts.

## Run protocol and cost

The completed run used one-row smoke validation and then 18 forwards over all
three rows. The frozen 32-frame cohort
is preprocessed on CPU before scoring. A comparable prior 32-frame QTA run
completed six short-target forwards in 52.7 seconds and used 11.12 GiB peak
allocated VRAM; this runner scores multi-token claims and builds paired masked
batches, so treat approximately 1–3 minutes scorer time for smoke and 3–8
minutes for full scoring as a planning range only, plus model loading and CPU
preparation. The runner requires at least 12 GiB free VRAM and 8 GiB host RAM
before loading; each forward also checks at least 2 GiB free VRAM and 3 GiB
host availability. Runtime, peak memory and effective counts are written to
the private receipt.

Both smoke runs and the full command executed successfully. Protected inputs
and derived receipts remain local; no raw artifacts were published.

## Scope and limitations

This is post-result exploratory reuse of six test videos across two packets. A QA interval is
not a human-reviewed support interval for the generated claim. A positive
contrast would show only an increase in this model's conditional likelihood
sensitivity to black-pixel removal from an annotation-aligned window under a
fixed internal bias, on this cohort and protocol. It would not establish claim
truth, semantic evidence support, improved grounding accuracy, or direct
explanation faithfulness. Black fill can be out of distribution; the group
expansion coarsens annotation boundaries; the text likelihood target is
re-tokenized; and the finite cohort supports descriptive values only, with no
population inference or significance claim. Human review was waived as a run
prerequisite, not completed. Rights remain unverified.

## Next action

The broader pool does not provide a larger identical-resolution confirmatory
sample: exact 32-frame eligibility gives only these six rows from cohorts29–30;
16-frame packets require a different sampling-resolution stratum and cannot be
pooled as the same protocol. Next, use a fresh preselected cohort and
claim-specific support intervals. Without those annotations, future runs are
QA-window proxy replications only. Do not tune target, mask, bias dose, sham
rule, or metric from these outcomes.

### Separate 16-frame exploratory stratum prepared

To extend the same six-condition estimand without mixing 16- and 32-frame
sampling, all v2_short 16-frame claim packets 06–09 and 14–22 were screened
using only claim validity, one QA interval, sampled-frame coverage, same-length
disjoint control, available deterministic disjoint sham, and unique media.
Across 35 packet records, 32 had valid parsed claims; 19/32 met the full
geometry, with exclusions of 6 no-sham, 1 no-control, and 6 invalid/multiple
interval or no sampled hit. The 19 selected rows have 19 distinct video and
video/question hashes. This is still retrospective: 18 rows have related
prior outcome exposure, and one row from attrited cohort08 was previously
unscored; the new key-bias endpoint was not consulted.

The mode-0600 local packet
`data/private_nextgqa/claim_text_qa_internal_bias_16frame_cohort_20261021_01.json`
has SHA-256 `171022f5…` and binds hashes for all 13 source packets and four
dataset inputs. The 16-frame geometry uses 8 temporal groups and remains a
separate stratum from 32-frame cohorts. CPU decode and processor preflight
passed for 19/19 rows, verifying each QA/control/sham key geometry and mask
pair; peak process RSS was 2.53 GiB. The complete preflight output remains
local and contains row-level protected metadata.

Next: run a 1-row GPU smoke (6 forwards) and, only if all score/hook/hash gates
pass, the full stratum (114 forwards). Based on the earlier 3-row runs, budget
approximately 12–15 minutes for full mode, with peak VRAM expected below the
measured 12.17 GiB 32-frame run due to half as many sampled frames; this is an
estimate, not a measurement. Keep all outputs local and do not pool this stratum
with the 32-frame estimates.

The smoke passed and the full run completed all 19 rows / 114 forwards. It took
592.46 seconds end-to-end on RTX 5070 Ti / CUDA 12.8, peak allocated VRAM 7.57
GiB, host-available memory before load 12.08 GiB. The mode-0600 result
`data/private_nextgqa/claim_text_qa_interval_key_bias_16frame_full_20261021_01.json`
has SHA-256 `ad0f1f61…`; it includes all five pinned weight-shard hashes.
Initial independent recomputation confirms all 19 unique videos, 114 finite
token-score vectors with means matching their branches, 114/114 expected
forward completions, and one hook per selected layer 5–16 for all biased
branches (zero hooks in baseline).

Observed 16-frame results: `Delta(QA-bias−baseline)` mean **−0.00261
nat/token**, median −0.01088, range [−0.05520,+0.11348], positive 6/19 and
negative 13/19. The QA-bias minus sham specificity contrast mean is **−0.00901
nat/token**, median −0.00104, range [−0.16646,+0.09775], positive 9/19 and
negative 10/19. The corresponding first-row smoke values were negative, but
the full cohort was run as frozen. This stratum does not support a stable
benefit: the primary average is near zero/slightly negative, and the sham
contrast is near zero/slightly negative. These are fixed, retrospective
descriptive values, not confidence intervals or significance tests; the pool
has prior related outcome exposure and uses QA intervals rather than
claim-specific reviewed support.

The 16-frame cohort's within-stratum summary is not combined with the separate
32-frame n=6 retrospective results. Together they show that the initial
cohort30 positive pattern is not robust across cohorts/resolutions. A useful
thesis result is the negative/inconclusive finding that generic positive QA-key
steering does not reliably improve this measurable claim-likelihood dependence
proxy under matched sham control. Direct semantic faithfulness remains
unmeasured (formal claim-support reviews 0/25 across these machine cohorts).

## New 32-frame local-history-screened exploratory cohort (sham protocol v2)

### Frozen plan and feasibility repair

To continue without reusing the scored cohort29/30 or the 16-frame 19-row
stratum, a new 3-video identity lock was frozen before claim generation from
the existing 20-media local test pool, after unioning the owner-only local
exposure manifest (`claim_exposure_union_20261021_01.json`: 321 classified
source files, 122 media hashes) and current local scored-claim artifacts. The
lock SHA-256 is `7fec3b3210c20ca627f65618cd963b6d0c9ec6df16a370c008e39218c0d0e416`;
the selected-pair filter SHA-256 is
`831b301fd049ed28172182e72cc1f7338dbbc757c061ce3dafa0419cc47dcfcf`. The
lock reports 3/3 unique videos and zero overlap with known local scored-claim
digests. It also explicitly says that all candidates appeared in prior
candidate metadata and full history is unavailable. This is sequential
exploratory evidence, not a certified fresh or confirmatory cohort.

Claim generation completed on the frozen videos with Qwen2.5-VL-7B-Instruct,
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16 compute, v2_short
prompt, seed 20261015, and 32 decoded frames: structured claims 3/3. The
16-frame attempt failed during CPU geometry preparation because a disjoint
contiguous sham window was unavailable (0 model forwards). A 32-frame attempt
with the old contiguous-window sham also failed before any forward. After both
failures, the sham protocol was changed before scoring: v2 selects a
deterministic hash-ranked set of equal-count temporal groups disjoint from both
the QA and pixel-mask control windows. This preserves equal visual-key budget
and explicit non-overlap while dropping the unnecessary contiguity condition.
The 32-frame CPU geometry then passed for all 3 rows (evidence/control/sham
group counts 1/1/1, 4/4/4, and 5/5/5). Focused tests passed 9/9; no scored
outcome had been observed when this design repair was made.

### GPU run and result

On the RTX 5070 Ti with CUDA 12.8 and the pinned 4-bit/BF16 model, the matching
one-video smoke passed 6/6 forwards. Independent audit recomputed its hashes,
finite token scores and branch arithmetic; each intervention layer 5–16 fired
once, and QA/sham both biased 252 visual keys on the smoke row. The smoke
receipt is mode 0600, SHA-256
`f7bd5f92fddffe45e8166eac172c5710957ca50c0664ce9b3e31f99ea66786fd`.

The frozen 3-row full run completed all 18/18 forwards, with no failures,
10.87 GiB peak allocated VRAM, and 224.54 seconds elapsed. The owner-only
result is `data/private_nextgqa/claim_text_qa_exploratory_32f_shamv2_full_retry01_20261021_01.json`,
SHA-256 `466b59622d5427ed81ea286a35b9e8892d9cab595547abf0a974f9582b97fede`.
Per-row primary contrasts (nat/token) were:

| Row | QA-bias − baseline | QA-bias − matched sham |
|---|---:|---:|
| 1 | +0.03940 | +0.01952 |
| 2 | −0.03682 | −0.02271 |
| 3 | −0.03884 | +0.03943 |
| Mean | **−0.01209** | **+0.01208** |

The descriptive 95% t intervals are `[-0.12288,+0.09870]` and
`[-0.06674,+0.09090]`, respectively (n=3; not confirmatory population
inference). The primary QA-bias effect was positive on 1/3 rows; the
matched-sham specificity contrast was positive on 2/3. Thus this cohort does
not show a stable cohort-level increase in QA-window claim-likelihood
sensitivity. A small positive matched-sham mean is heterogeneous and its
interval is broad around zero. Neither contrast is a direct faithfulness
score: the QA interval answers the benchmark question and was not validated as
support for each generated claim; black masks can be out-of-distribution; the
cohort is small and sequential; and semantic review was not performed. The
runner records `direct_faithfulness=false`.

### Reproducibility and next action

Private inputs, claim text, row identities, and detailed receipts remain local;
rights remain unverified. No result was published. The earlier v1 19-row
results and this v2 3-row run are separate protocol strata and must not be
pooled. The useful thesis outcome is now stronger negative-replication evidence
that a generic positive visual-key bias does not reliably increase this
benchmark-interval claim-likelihood proxy relative to baseline or matched
sham. Direct semantic faithfulness remains 0/25 formal scores, and the current
human-review artifact remains unanswered (0/3).

Next, stop repeating this QA-interval proxy unless a genuinely new cohort and
a pre-outcome reason for another intervention are established. To test actual
faithfulness improvement, use claim-specific reviewed evidence frames and a
held-out cohort with an unchanged support-key versus matched-sham protocol;
the current result motivates that direct outcome, but does not establish it.

Aggregate reproduction:
`PYTHONPATH=scripts .venv/bin/python scripts/analyze_claim_text_qa_interval_key_bias_summary.py --input data/private_nextgqa/claim_text_qa_exploratory_32f_shamv2_full_retry01_20261021_01.json`
recomputes token means, branch contrasts, descriptive t intervals, and source
hash checks while omitting raw claim/question/video identifiers. The runner and
summary suites passed **12/12** tests; bytecode compilation and `git diff
--check` passed.
