# Same-model localized claim-mask pilot (2026-10-21, local exploratory)

## Purpose and fixed question

The user waived collection of the outstanding human checks for this run. This
does not make those checks factually complete. To advance an exploratory
faithfulness measurement without fabricating attestations, this pilot uses
Qwen itself to locate a time span for each already-frozen generated claim,
then asks whether the same Qwen's fixed QA answer margin depends more on that
span than on an equal-count temporal control.

Primary finite-cohort contrast is the mean across the fixed three validation
items of `margin(temporal-control black mask) - margin(machine-localized claim
span black mask)`, in nats. Each margin is the gold-choice next-token log
probability minus the best of the four foil-choice log probabilities. A
positive value means the model-localized-span mask reduced this margin more.
The planned inputs are exactly the de-duplicated three records in the existing
smoke and remaining claim-generation packets, matched by source-video digest.
There are three localization generations and nine scoring forwards; no
subgroup search, threshold selection, p-value, or population confidence
interval is planned.

## Implementation and pre-run checks

New local runner: `scripts/run_machine_localized_claim_mask_pilot.py`. It pins
the existing Qwen2.5-VL-7B-Instruct revision, 4-bit weights, BF16 compute,
eager attention, 32-frame endpoint-inclusive decode, exact decoded-video
digest match, strict JSON timestamp validation, nearest-rank matched control,
frozen CPU batches, exact model-call boundary check, and per-forward memory
guards. It writes only an exclusive mode-0600 local aggregate under
`data/private_nextgqa/`; raw text, paths, IDs, and source videos are not put in
the result object.

CPU interval-parser tests pass **4/4**. The fixed source join found three
unique video digests across the two private packets; all three paths exist and
the human-review fields are still null. An independent local decode verified
all three packet media hashes and constructed equal-count benchmark-interval
mask/control geometry on 32 frames per clip (5/5, 9/9, and 16/16 sampled
frames/controls). These benchmark intervals are used only as a CPU geometry
smoke and were not substituted for machine-localized intervals in the first
GPU branch. The first machine-localized run completed three generations but
rejected item 3 at timestamp/interval validation before scoring. The saved
private receipt reports the stage and a no-score state only; it does not retain
the generated interval text or the more specific parser rejection. Therefore,
the exact invalidity cause cannot be reconstructed without repeating the
localization, which this experiment does not justify.

## Interpretation limits and gate status

This is a new post-hoc exploratory estimand on a reused validation/development
cohort, not the preregistered human-reviewed primary. The same model generated
the claim, localizes its supporting interval, and receives the mask; this
creates circularity. No human checked the claim or localized interval, no
independent claim/frame attestation is made, and black fill may shift the input
distribution. A result can show only the tested model's response to this
specific automated intervention on these three items. Rights status remains
unverified; all source material and detailed outputs stay local.

The user instruction is recorded as a waiver of collecting human review for
this exploratory pilot, not as evidence that a person reviewed anything. The
formal trust-root, reviewer identity, Stage-2, and production atomic-preload
records remain absent; this exploratory runner does not claim or emit formal
authorization. Direct formal human-validated faithfulness remains 0/3 scores.

## Run and next action

The machine-localized branch produced **0/3 scored rows** and no estimate. Its
third interval failed validation after three localizer generations. This
branch is closed; do not repeat it or enlarge the cohort from these outcomes.
The separate benchmark-interval branch is reported in
`2026-10-21_benchmark_interval_claim_cohort_gpu_pilot.md`; it is an auxiliary
QA-margin diagnostic, not a rescue score for this branch.

Human-validated generated-claim faithfulness remains **0/3**. The user's waiver
removed the obligation to collect human review for these exploratory runs; it
does not change the observed review count or validate claim truth/localization.
