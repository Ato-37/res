# Validation-split temporal-order claim dependence replication (2026-10-21)

## Prospective exploratory protocol

This is a new exploratory split-specific cohort after observing the incomplete
three-row test-pool run. It is not confirmatory and will not be pooled with that
run. Human-review collection is waived only as a prerequisite for this local
exploration; no human review is claimed or encoded as completed.

Before claim generation or scoring, select the first three distinct eligible
validation videos in the deterministic `freeze_cohort` order from the full local
NExT-GQA validation data and available local videos. Exclude any media digest
present in the current local claim-generation/scoring artifact families. Freeze
the media/QID hashes and source/config hashes in a mode-0600 local lock, and
freeze selected candidate-pair hashes in separate mode-0600 filter files bound
by the generation packets. The lock does not itself bind the filter-file hash
(its `candidate_pair_filter_sha256` is null for full-split selection); preserve
this as a provenance limitation. The historical exposure inventory is
incomplete, so these items are not certified fresh. If three rows cannot be
frozen, stop this cohort without reducing N or substituting another split.

For each locked row, generate one `v2_short` Qwen2.5-VL-7B claim using 16
sampled frames and the exact eligible QA pair selected by the frozen selector.
Score the deterministically re-tokenized saved claim string under natural frame
order and deterministic temporal-block reversal, each with baseline and the
fixed `ln(2)` visual-key attention bias at layers 5–16. The per-row descriptive
interaction is `(original_biased - reversed_biased) - (original_baseline -
reversed_baseline)` in nat/token. This tests temporal-order sensitivity of the
re-tokenized sentence under a coarse internal intervention; it does not test
claim truth or semantic evidence support. No CI or significance test will be
reported for n=3. No row replacement or post-result parameter/cohort changes.

Run a one-row GPU smoke first, then score the remaining locked rows only if the
claim-generation packet and smoke receipt bind to the frozen lock and current
source. Preserve failures. No cohort estimate unless all 3 generation rows and
all 12 scorer forwards have source-verified completed receipts. Rights remain
unverified and raw media/claims/questions/receipts remain private.

## Preflight evidence and next action

A CPU-only feasibility scan of the full local validation split, excluding 24
unique media digests found in current local claim artifact families, returned
three eligible unique videos: 3 scanned, 3 timing passes, no decode failures
or timing rejections. Selection used no model outcomes. This was a pre-run
snapshot; the lock and pair-filter files were subsequently written before GPU
claim generation.

## Execution results (primary analysis and independent audit)

The frozen validation cohort was `4fbd70fe…` (lock SHA-256
`c11351aa…`). The pinned Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` generated structured `v2_short`
claims for all 3/3 videos. The scorer completed 12/12 requested forwards across
the three receipt rows. Each scorer receipt records the same runner SHA-256 as
the available source; the analyzer also rechecked packet SHA-256 bindings,
media/QID assignment hashes, pair-filter digests, branch arithmetic and
baseline/biased attention-hook counts. All source and packet records remained
mode 0600 under the private data root.

The preregistered per-row interaction values were −0.005938, +0.046906, and
+0.074168 nat/token. The finite-cohort mean is +0.038378, median +0.046906,
and range [−0.005938, +0.074168] nat/token (2/3 positive). No CI, hypothesis
test, or multiplicity adjustment is reported at n=3. These values describe the
fixed validation cohort only. The sign pattern is mixed, the mean is small, and
the data do not establish an attention-intervention benefit. This is still a
re-tokenized claim likelihood sensitivity to temporal order, not direct
faithfulness, claim truth, hallucination, or semantic evidence support. Actual
human review and direct faithfulness remain 0/3; collection was waived for this
exploratory run only.

An independent artifact audit reproduced all three row values and the aggregate,
verified the three receipt-to-packet SHA-256 bindings, current scorer source
hashes, 12/12 forward counts, and the packet row-to-lock/pair-filter mapping.
That audit identified a missing automated check of the claim generator and its
pilot dependency. The analyzer now checks both packet-recorded source hashes
against the available source before accepting packet rows; rerunning it retained
3/3 valid rows and the values above. The pair-filter files are bound by the
generation packets and row-level pair hashes, but the frozen lock itself has a
null `candidate_pair_filter_sha256`; this remains a provenance limitation and
is not retroactively represented as a pre-run lock binding. Future freezes
should include a non-null pair-filter hash. Row-to-pair identity was separately
recomputed during the independent audit from the local split/question mapping;
the analyzer itself currently checks filter membership but does not derive
pair identity from each packet row.

## Next action

Keep this validation cohort separate from the incomplete test cohort;
do not combine them or infer a population effect. The local historical exposure
inventory is incomplete, so these items are not a fresh confirmation holdout.
A further GPU run needs a separately frozen cohort and a distinct, justified
estimand; do not keep repeating temporal perturbations solely to accumulate
positive values. The existing local review artifact is bound to different
claims and does not validate this cohort.
