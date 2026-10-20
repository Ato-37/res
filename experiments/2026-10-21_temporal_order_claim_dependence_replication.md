# Test-pool temporal-order dependence follow-up (2026-10-21)

## Purpose and estimand (frozen before inference)

Test whether the fixed global visual-key attention bias increases the
conditional likelihood gap between a saved Qwen-generated claim under its
natural sampled-video order and under deterministic temporal-block reversal.
This is a new exploratory follow-up on a separately materialized, unscored
candidate pool, after earlier three-row interaction results were observed. It
is not an independent confirmatory study.

For each of three fixed rows, the primary descriptive interaction is
`(original_biased - reversed_biased) - (original_baseline - reversed_baseline)`
in mean claim-token log likelihood (nat/token). Positive values mean the fixed
bias increased this particular temporal-order sensitivity diagnostic. Each
row has four teacher-forced forwards. The attention bias remains the existing
fixed `ln(2)` on all expanded visual-key positions in layers 5–16; the control
reverses complete temporal-patch-sized blocks while preserving within-block
order. No metric, cohort, row, or threshold changes after scores are observed.

## Cohort, source, and limits

The source is the local NExT-GQA test-split candidate pool
`data/private_nextgqa/test_claim_pool_local_20261021_04`, containing 20
hash-ranked videos and 99 eligible question-pair hashes after its recorded
exposure scan. Select the first three eligible rows in the frozen selector's
deterministic order, with exactly one question per distinct media digest. A
CPU decode/timing preflight and frozen media/QID hashes bind the selected rows
before claim generation. No score values are used in selection.

A search of current local generated-claim packets and scoring receipts found no
matching media digest for the selected rows. The candidate videos do appear in
older split/candidate metadata and later broad exclusion ledgers; those records
do not prove prior model inference. The prior-exposure inventory is incomplete,
so these rows are described as a separately materialized, locally unscored
candidate pool with possible unrecorded exposure—not as certified fresh.

Generate one concise `v2_short` claim per selected video with the pinned
Qwen2.5-VL-7B model, then score the same saved sentence after deterministic
re-tokenization under four conditions. Human review is waived as a prerequisite
for this exploratory local run but is not claimed as completed. Claim truth,
claim-to-frame support, and clock/frame agreement are unreviewed. Rights remain
unverified; all source packets, media, and detailed receipts stay local.

This measures conditional likelihood sensitivity to temporal order and a
coarse global attention-logit intervention. Reversal may be out of distribution,
the claim generator and scorer share a model, and the bias is not claim-specific.
The result cannot establish claim truth, semantic evidence support, hallucination
rate, or direct explanation faithfulness. For n=3, report all rows and finite-
cohort mean/median/range only; no population CI or significance test.
Failures are retained, no rows are replaced, and no estimate is reported unless
all three locked rows complete.

## Workload and gate

Expected workload: three claim generations plus twelve teacher-forced scoring
forwards across three sequential model invocations, roughly several minutes
including checkpoint loads. Require at least 12 GiB free VRAM and 8 GiB host
memory before each load and a 2 GiB free-VRAM floor before each forward. Use the
existing `.venv`, offline pinned checkpoint, 4-bit weights and BF16 compute.
Run CPU tests and one fixed-row four-forward GPU smoke first; only after that
row and receipt checks pass, score the remaining two locked rows. Do not
continue if the output packet fails the pre-frozen cohort binding.

## Verified execution and provenance audit

Local GPU execution used the existing offline `.venv`, Qwen2.5-VL-7B-Instruct
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights and BF16
compute on the RTX 5070 Ti. The three locked rows were selected before generation
and scoring. Claim generation returned 2 structured claims out of 3; locked row
1 failed with `claim_missing`. No replacement was made. The two valid rows each
completed four scoring forwards. The recorded row interactions were +0.035984
and +0.062075 nat/token, but the first row's receipt records runner hash
`d55efb8d…` while the currently available runner is `8420f406…`; the exact
old source snapshot could not be recovered from the workspace. Therefore row 0
is visible as a stored receipt value but fails source attribution and is excluded
from the analyzer's verified `scored_rows` count. Row 2's runner hash matches.
The analyzer reports 1 provenance-verified scored row, 1 failed claim row, and
1 provenance-unverified row; cohort mean, CI, and test remain null. This is an
incomplete exploratory run, not evidence of an intervention effect.

Independent arithmetic/hook checks and local analyzer validation reproduced the
stored values and expected 0/1 hook counts for baseline/biased branches. The
analyzer now verifies the recorded runner hash against the available source and
keeps mismatched rows out of the verified count. Focused tests passed (14 tests)
for the analyzer, cohort freezer, and scorer; the system-Python attempt was not
usable because it lacked the project dependencies, so validation used `.venv`.
Rights remain unverified, inputs/results remain local, and no human review was
performed. This conditional likelihood/order-sensitivity metric is not direct
faithfulness, semantic support, truth, or hallucination measurement.

## Next action

Close this fixed cohort: its failed row cannot be replaced, and the unavailable
runner snapshot prevents verified attribution for another row. Do not calculate
a cohort effect or resume head/layer/alpha search. Continue with a separately
prespecified and source-pinned experiment only if an eligible local cohort and
reproducible implementation are available; otherwise improve direct-faithfulness
measurement design and identify the concrete data/annotation inputs needed for
that endpoint. Human review is waived only as a prerequisite for exploratory
local work; it remains factually incomplete (0/3 reviewed).
