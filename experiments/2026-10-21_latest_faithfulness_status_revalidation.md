# Latest faithfulness status revalidation (2026-10-21)

## Purpose

Reconcile the current worktree's thesis-facing status with the three latest
fresh-validation QA-interval claim-mask cohorts and the current human-review
handoff. This is a provenance/status audit and documentation update, not a new
model experiment.

## Verified state

The three fixed-specification exploratory cohorts requested 3, 10, and 7 rows;
they scored 3, 6, and 7 rows respectively, with 18/18, 36/36, and 42/42
completed forwards. Their per-row `DeltaF = F_QA-bias - F_baseline` means are
+0.01817, -0.00355, and -0.00368 nat/token. The 16-row finite-cohort
descriptive recomputation from logged row contrasts is mean +0.00046375,
median +0.00475, range [-0.05460,+0.06046], and 8/16 positive. The
QA-bias-minus-sham cohort means are +0.03444, -0.00878, and +0.01474
nat/token; weighted over the rows these are +0.009614815, with 9/16 positive.
The 16 rows belong to sequential exploratory cohorts, so this pooled
arithmetic is descriptive only: it is not a confirmatory pooled estimate, and
no interval, test, or population inference is attached.

The full receipt hashes recorded in the corresponding experiment logs were
recomputed locally:

- n=3 successful retry: `fa55671d2dd449b755d8ea131b222f88c7196e0116798b4c5f7a55b185ccb815`
- n=6: `2c16d5ac6fbbda353370b1bba3f4690ef9c8b045398fa2dd59d2697f5ae23b94`
- n=7: `9ff2683a1ee2cd8d76c9d8d2b3f82d22116909bb0c9abcd802ca53a127914900`

The current frame-review HTML is present locally with mode 0600 and SHA-256
`309739d3aacb39ea46745c41893fbc2736661e421886e228922f1690faa52715`.
The latest handoff log reports a synthetic two-reviewer browser/download E2E
pass, while explicitly stating that reviewer identity and semantic support
were not validated. A fingerprint-bound response scan found 0/52 candidate
JSONs; the 52 are files scanned, not response count. Current direct reviewed
claims and direct intervention scores remain 0/3; formal review inventory is
0/25, with clock/frame pairs 0/15 and prediction links 0/3.

## Interpretation and action

The GPU proxy runs are valid as local exploratory computations but do not
measure claim support or human-validated explanation faithfulness. Their
primary direction varies across cohorts and is near zero in the later two.
The direct evaluation remains blocked on actual human claim-to-frame review;
the user declined the most recent review request, so no judgment is inferred.
No new same-estimand GPU run is justified by these outcomes. The thesis-facing
overview and pitch have been updated to include the three-cohort result and
keep the direct-faithfulness limitation explicit.

## Checks and limitations

`git diff --check` passed for the updated overview and pitch. Receipt hash and
finite-cohort arithmetic were independently checked locally; a separate audit
also checked row-level branch arithmetic and packet-to-receipt media binding.
The primary agent reran the direct-preload, phase-1 adapter, prompt-token
binding, and actual-processor candidate regression command; all 49 tests passed
in 12.9 seconds. These synthetic/structural tests establish fail-closed wiring
only and do not authorize scoring or substitute for review.
No model or GPU forward ran during this status revalidation. Rights status
remains unverified and all private data/results stay local. The worktree has
extensive unrelated modifications and untracked files, so changes have not
been staged, committed, or published.

## Next action

When an actual reviewer response becomes available, validate it against the
exact artifact fingerprint, then continue the direct claim-review and
prediction-link gates. Until then, do not call the QA-interval proxy direct
faithfulness or claim that internal intervention improved explanation
grounding. Continue independent protocol/readiness audits that do not require
imputing human judgments.

## Same-day follow-up: prompt repair and native interval audit

After this checkpoint, a source audit found that the temporal-order scorer had
used its legacy v1 suffix on four packets of `v2_short` claims. All 11 valid
previously scored rows were rerun (44/44 forwards) with the recorded v2_short
suffix and source hashes. The four separate cohort means are +0.021637
(n=3), +0.053469 (n=3), +0.041990 (n=3), and −0.018895 nat/token (n=2 valid
rows). The old v2_short estimates are invalid and retained only as
wrong-prompt diagnostics. These repairs are outcome-exposed likelihood
sensitivity, not confirmation or direct faithfulness. An independent receipt
audit reproduced every row and mean. See
`2026-10-21_temporal_order_prompt_binding_repair.md`.

A separate aggregate audit of the local NExT-GQA annotations confirms that
all 3,358 validation and 5,553 test QA rows join to QA-level `gsub` time-span
annotations. It found 3 validation spans extending beyond duration and 36
test spans with negative starts. On the exact three-item current review
artifact, 15/48 sampled timestamps fall inside four QA reference intervals,
with at least one such timestamp for each item. These annotations are not
claim-specific; they do not change the direct score. The current handoff is
still 0/3 reviewed, 0/15 clock/frame correspondences, 0/3 prediction links,
and 0/3 direct intervention scores. The new aggregate-only audit is
`2026-10-21_nextgqa_native_interval_coverage_audit.md`; reproduce it with
`PYTHONPATH=scripts .venv/bin/python scripts/audit_nextgqa_native_interval_coverage.py --include-current-review-packet-overlap`.

## Same-day follow-up: additional prompt-version invalidation

A whole-runner prompt audit found two more completed v2_short score branches
that used the legacy v1 suffix: QA-interval × attention-bias (3 rows/12
forwards) and video-content × attention-bias (3 treatment rows/3 new forwards,
with reused baselines). The corresponding estimates −0.018260 and −0.003816
nat/token are not valid for the logged v2_short scoring condition. Keep the
receipts as historical wrong-prompt diagnostics; the video-content DID is
invalid because its treatment and reused baseline suffixes differ. The
independent original-versus-gray baseline chain is outside this invalidation.
Both runners now resolve the packet-recorded version and bind version plus
suffix digest into future outputs. Four runner/analyzer test modules passed
14/14. No rerun was made on the post-result n=3 cohort. Details:
`2026-10-21_prompt_version_scoring_audit.md`.

The current private review artifact still has zero fingerprint-bound real
responses. The older Downloads JSON is legacy v14 and does not bind to the
current frame-review artifact. Browser E2E has since verified the actual
filesystem download of a synthetic two-reviewer export; this is workflow
validation only, not evidence that a real review occurred. Formal direct counts
remain 0/3 claim reviews, 0/15 clock/frame pairs, 0/3 prediction links, and
0/3 direct intervention scores.
