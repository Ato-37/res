# Direct faithfulness checkpoint (2026-10-22, local)

## Purpose

Revalidate whether the current three-item claim-support artifact now has
fingerprint-bound responses, audit the production scoring path, and decide if a
nonduplicate real-video score or GPU experiment is justified. This checkpoint
does not run inference.

## Verified current state

- Latest local handoff:
  `data/private_nextgqa/claim_support_frame_review_cohort11_v3_20261021_08.html`,
  SHA-256 `e4222baf2620fe8e04f30d4f8f7af830e30a878ed1aec436fcc53bbcfe88e150`;
  its embedded artifact fingerprint is
  `98be38623100f6554820d104b1e7d575fac8954c2224a0e933d1042ecc97f577`.
  Also checked the earlier `_06` handoff (SHA-256
  `309739d3aacb39ea46745c41893fbc2736661e421886e228922f1690faa52715`,
  fingerprint `abaa3b8dee8f4fc9cbd2031d1772d2912502a9c1bafb1f2b03dac90365d8a1d4`).
- Ran `scripts/audit_claim_review_response_inventory.py` against both exact
  artifacts and `<windows-user>/Downloads`. It scanned 2 JSON candidates;
  each artifact has 0 fingerprint-bound responses, 0 reviewer entries, and 0
  item decisions. Total invalid bound responses: 0. This is a structural
  inventory only; it does not authenticate identity or establish semantic
  support.
- Ran the browser E2E on the latest `_08` artifact with individual elevated
  WSL-to-Windows execution after ordinary interop was unavailable. Windows
  Chrome passed **2 synthetic reviewers × 3 items**, verified the exported
  Blob and actual downloaded JSON against the artifact fingerprint, and
  confirmed the prior export link was revoked between reviewers. This confirms
  the current handoff's mechanics only; it generated no human responses and
  validates neither identity nor semantics. The artifact SHA-256 remained
  `e4222baf2620fe8e04f30d4f8f7af830e30a878ed1aec436fcc53bbcfe88e150`.
- Found and repaired a stale handoff target: both Japanese HTML launchers, the
  Japanese runbook's target/inspector command, and the browser E2E default had
  still pointed at `_06`. They now point to the latest cohort11 v3 `_08`
  artifact. A path-consistency check passed 4/4 files, changed-document
  whitespace checks passed 5/5, and `py_compile` passed. Re-running the
  browser E2E with no artifact argument then passed again on the latest file:
  2 synthetic reviewers × 3 items, actual JSON download verified, stale export
  revoked. No human judgment was made.
- Updated `docs/research_overview.md` and the 2026-10-22 checkpoint to record
  the current artifact, response inventory and verified launcher destination.
- An independent source audit found that the formal direct-faithfulness
  pre-load gates and baseline scorer intentionally remain non-authorizing.
  Passing structural checks does not permit model loading or scoring. The
  existing response/link inputs do not provide the actual claim-support,
  frame-time, and prediction-link judgments for this cohort. Flipping a test
  flag or recording a declined review as complete would not remove that gap.
- No distinct, prespecified GPU endpoint is available on the current locked
  rows: the nearby real-video fixed-claim, gray/frame-mean, QA-interval,
  temporal-order and global-bias proxy branches have already been run and
  closed or are heterogeneous. Another run on those same proxies would not
  improve direct-faithfulness evidence.

## Faithfulness progress

- **Directly evaluated:** 0/25 formal claim-support decisions; current handoff
  0/3 claim reviews, 0/15 verified frame-time pairs, 0/3 prediction-link
  decisions, and 0/3 direct intervention scores.
- No direct `DeltaF`, confidence interval, hypothesis test, or multiplicity
  adjustment exists for this cohort.
- **Auxiliary diagnostics only:** prior real-video internal steering changed
  some conditional likelihood values, but effects were small/mixed across
  cohorts and controls; the 27-row QA diagnostic had identical 15/27 accuracy
  and 0/27 choice flips under baseline, QTA and sham. These do not increment
  direct-faithfulness counts.
- Rights status remains unverified. Private media, responses, and receipts
  remain local.

## Interpretation and next action

Current state is `no_go` for direct scoring on this cohort, while the local
exploratory review workflow is available. Do not manufacture review labels,
reuse legacy responses for the current artifact, or reopen closed proxy
branches. The next direct-evidence step is an actual response bound to this
artifact; after it is inspected, freeze the reviewed support anchors and run
the smallest exploratory fixed-claim baseline/support-key/matched-sham
comparison with all failures retained. Until then, continue only independent
measurement/readiness work that changes this path rather than another proxy
rerun.

No model, CUDA process, or GPU forward ran in this checkpoint.

## Claim-review frame-time adapter readiness follow-up

An independent source audit confirmed a remaining engineering gap: the current
simple response format (`artifact_fingerprint`, per-reviewer decisions, and
frame indices) is not accepted by the separate signed/formal QTA input path,
and no production runner consumes it for a claim-frame intervention score.
Existing QTA scorers use other cohorts/estimands and mark direct faithfulness
false. This means a valid response alone would not yet be sufficient to run the
intended direct experiment.

Added [`adapt_claim_support_frame_review_response.py`](../scripts/adapt_claim_support_frame_review_response.py)
as a private exploratory bridge from the exact current artifact/response pair
to frame-index/time mappings. All requested rows are retained; only unanimous
`supports` rows are marked eligible, and non-support/disagreement rows remain
in the denominator. The output contains hashes and selected indices/times, not
reviewer nicknames or raw text, and cannot authorize model loading. Five
standard-library `unittest` cases pass; byte-compilation and scoped whitespace
checks pass. Pytest is unavailable in both `.venv` and system Python, so the
pytest suites were not rerun. No dependency was installed.

This adds zero human-reviewed rows and zero direct scores. The current response
inventory remains 0/3, 0/15 verified frame-time pairs, 0/3 prediction links,
and 0/3 direct scores. Next, after a matching response arrives, run the adapter
and implement a dedicated frozen-claim scorer for baseline, reviewed-support
ablation, and an equal-budget matched non-support control. That scorer does not
exist yet; do not substitute a previously closed QA proxy or call readiness
work a faithfulness effect. No GPU job ran in this follow-up.

## Cross-cohort internal-steering synthesis (2026-10-22 follow-up)

Recomputed the existing QTA and overlap-matched-sham selective-ablation
receipts for nine cohorts (16–18, 20–24, 26), checking 27 paired row tokens,
27 distinct source-video digests, all 189 completed forward counts, fixed
prediction/foil pairing, and the rowwise saved-margin formulas. The mean of
cohort means was +0.013889 nat for QTA−baseline (median +0.000000036, range
[−0.083333,+0.125000]; 3 positive, 3 negative, 3 near-zero cohorts) and
+0.023148 nat for QTA−sham (median +0.000000016, range
[−0.041667,+0.125000]; 4 positive, 3 negative, 2 near-zero). Near-zero is
defined as absolute value ≤1e-7 nat. Four unit tests, byte-compilation, and
whitespace checks passed. Reproduction and provenance are in
[`2026-10-22_qta_crosscohort_sensitivity_synthesis.md`](2026-10-22_qta_crosscohort_sensitivity_synthesis.md).

This strengthens the descriptive conclusion that generic internal steering
effects are inconsistent across reused development cohorts. It adds no direct
faithfulness score and no new GPU evidence: formal review remains 0/25, current
handoff remains 0/3, and direct `DeltaF` remains unavailable. Human review was
not performed; rights status is unverified. No additional GPU run was justified
on these closed, outcome-exposed cohorts.

## NOAA real-video exploratory GPU follow-up

Ran a separate fixed QTA answer-distribution sensitivity experiment on all 6
locally available NOAA question rows across 3 videos. The pinned Qwen2.5-VL-7B
model completed 18/18 baseline/QTA/disjoint equal-budget-sham forwards; peak
allocated VRAM was 9,289,169,920 bytes and runtime 139.034 s. Mean five-choice
distribution TVD was 0.007412 for QTA vs baseline, 0.000830 for sham vs
baseline, and 0.007547 for QTA vs sham; neither intervention changed the top
choice on any of the 6 rows. These are exploratory descriptive numbers with
only three video clusters and no CI/test. The sham is not evidence-overlap-
matched. Gold and review materials were not read, human review was not
performed, and rights status remains unverified. Therefore this adds no direct
faithfulness/grounding evidence and does not change the 0/25 formal review or
0/3 current-artifact review counts. Full protocol, receipt hash, limitations,
and reproduction details are recorded in
[`2026-10-22_noaa_qta_distribution_exploratory.md`](2026-10-22_noaa_qta_distribution_exploratory.md).

On 2026-10-22, re-ran the structural response inventory against the exact
current cohort11 v3 `_08` artifact and local Downloads/private-nextgqa roots.
It scanned 1,167 candidate JSON files and found 0 responses bound to this
artifact (0 reviewer entries, 0 item decisions, 0 invalid bound files). This
inventory does not authenticate identity or judgment semantics, but confirms
there is still no returned response JSON for the current handoff in those local
roots. The artifact SHA-256 remains
`e4222baf2620fe8e04f30d4f8f7af830e30a878ed1aec436fcc53bbcfe88e150`.

A post-hoc CPU analysis joined the NOAA QTA groups to the local annotation
audit's QA evidence timestamps. QTA hit 2/6 rows, its disjoint sham 3/6, and
the row-specific uniform-group expectation was 1.75/6; paired outcomes were
QTA-only 1, sham-only 2, both 1, neither 2. The annotation method is mixed
across only three videos, and the time-to-sampled-frame mapping is coarse.
This weak grounding-proxy audit is not claim-support or direct faithfulness;
review remains 0/25 and 0/3. See
[`2026-10-22_noaa_qta_annotation_time_alignment.md`](2026-10-22_noaa_qta_annotation_time_alignment.md).

## NOAA annotation-matched QA margin sensitivity (2026-10-22 follow-up)

Ran a distinct post-hoc GPU sensitivity test on the same six NOAA QA rows. For
each row, the baseline top choice and strongest foil were frozen. The same
`ln(2)` visual-key bias at layers 5–16 was applied to all timestamp-mapped QA
evidence groups or to an equal-size, temporally nearest non-evidence control
group set; both conditions had identical biased-key counts. The fixed-pair
margin difference (evidence bias minus matched-control bias) was
`−0.125, 0, −0.125, 0, −0.125, 0` nat, mean `−0.06250` nat, 4/6 negative,
1/6 numerically near zero positive, and 1/6 zero. All 18/18 forwards completed
on RTX 5070 Ti, with 6.85 GB peak allocated VRAM and 132.148 s elapsed. An
independent receipt audit reproduced the mean, row arithmetic, matching
budgets, 12-layer hooks, and source/manifest/smoke hashes. No confidence
interval, test, or population inference is warranted.

This does not support a positive local sensitivity advantage for annotation-
mapped groups over the proximity-matched controls. It remains a reused,
post-hoc QA timestamp diagnostic; the annotations are not reviewed support
spans for generated claims, and the forced key bias is not natural evidence use.
Direct faithfulness remains 0/25 formal claim-support reviews and 0/3 for the
current claim/frame/prediction-link/direct-score handoff. The experiment and
local receipt hash are documented in
[`2026-10-22_noaa_annotation_matched_group_margin.md`](2026-10-22_noaa_annotation_matched_group_margin.md).

## 16-frame claim-support scorer wiring smoke and independent correction (2026-10-22)

The prior statement that no production scorer consumed the lightweight
response format has been superseded by a local exploratory bridge and scorer.
The current three-item 16-frame artifact/packet source binding is checked
before model load; full mode still requires an exact-artifact response. Since
no such response is currently available, so only a synthetic-support mapping
smoke was run on one actual local video. Independent audit found that the first
smoke `_02` applied both pixel masks and visual-key deletion in all branches;
its +0.010701 nat/token is invalid for the stated estimand and must not be
used. The code was corrected to use pixel masking only, with key bias only in
QTA branches. The corrected `_03` completed 5/5 forwards in 38.354 s; each
branch passed layer-5–16 hook counts, QTA branches biased 266 keys, and no
branch deleted keys. Peak allocated VRAM was 7.468 GiB. The synthetic
`DeltaD=+0.005756` nat/token remains an execution diagnostic only. Smoke now
validates artifact fingerprint/schema and exact JPEG bytes regenerated from
the decoded model-input frames; it does not create human evidence.

The response adapter uses the intersection of each unanimous-support
reviewer's frame choices; no common frame makes the row ineligible. The
corrected receipt records runtime/source/config provenance (checkpoint weight
bytes are not rehashed). Fourteen focused unit tests, `py_compile`, and scoped
whitespace checks pass.
The implementation and private receipt hash are documented in
[`2026-10-22_claim_support_qta_16frame_scorer.md`](2026-10-22_claim_support_qta_16frame_scorer.md).
Human-reviewed rows remain 0/3, verified frame-time pairs 0/15, prediction
links 0/3, and direct scores 0/3; the formal count remains 0/25. Therefore
the original response gate still blocks a reviewed-support direct score, but
the scorer wiring gap is no longer the blocker. The current status is no
direct claim-specific faithfulness effect measured; user-authorized review
waiver does not create support-frame labels. Rights status remains unverified,
and the video-derived smoke receipt stays local-only.

## Receipt and estimand audit follow-up

Independent review found that the first smoke `_02` added internal visual-key
deletion to the intended pixel-mask comparison; its synthetic score was
invalid for the stated estimand. That code path was fixed and `_02` explicitly
excluded from evidence. `_03` executed the corrected pixel-mask-only path but
its receipt still mislabeled the estimand and omitted five helper source
hashes. `_04` was run with corrected receipt metadata and a complete 14-file
runtime source manifest; all 14 digests were independently recomputed and
matched. `_04` again completed 5/5 synthetic-map forwards (37.970 s, 7.468
GiB peak allocated VRAM), with zero visual-key deletions. Its repeated
synthetic `DeltaD=+0.005756` nat/token is only a deterministic wiring check.
The private receipt SHA is
`77440199f1393622c1976c3f257a67672de46ce55123cc59baf2bf766777bb84`.
Fourteen focused tests pass. The current artifact still has no human response,
so direct support-dependent faithfulness remains unmeasured at 0/3 and formal
review remains 0/25. Full model weight bytes are not individually hashed by
the local receipt manifest.

The runtime manifest was extended once more to include PyAV/Numpy package
versions and `presentation_time_utils.py`; final smoke `_05` supersedes `_04`
for the current source manifest. Its receipt SHA-256 is
`ccb20fb5263e3ed539b44e37ba43acc7fc1db3d3e61c0a047537b7998153c787` (mode
`0600`). All 15 source hashes independently match, and it completed the same
5/5 synthetic-map forwards in 34.519 s with 7.468 GiB peak allocated VRAM.
This closes that receipt-level dependency omission but does not change direct
faithfulness evidence; weight bytes remain unhashed and the human response
remains absent.

## Latest artifact inventory revalidation (2026-10-22)

Re-scanned 1,172 local JSON candidates across `data/private_nextgqa` and the
local Downloads directory against both scorer artifact `_06` and the latest
launcher artifact `claim_support_frame_review_cohort11_v3_20261021_08.html`.
Both exact HTML fingerprints verified; neither had a bound response (0/3
self-reported item decisions on each artifact), and there were zero invalid
bound files. The scan is structural only and does not validate semantics or
reviewer identity.

Re-ran the focused scorer/adapter suite: 14/14 tests passed. This validates
binding, frame intersection, mask-budget and contrast arithmetic behavior; it
adds no reviewed examples or direct model scores. Direct claim-support
faithfulness remains 0/3 on the current handoff and 0/25 in the formal review
inventory. The next direct measurement still requires a response bound to the
scorer's exact artifact; this inventory does not justify another proxy run.
The separate current-local exposure census likewise found no development-ready
cohort; see
[`2026-10-22_local_exposure_census_dev_cohort.md`](2026-10-22_local_exposure_census_dev_cohort.md)
for source counts, the test-split exclusion, and the external annotation-source
lead.
