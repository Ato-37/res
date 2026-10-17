# QTA checkpoint hash smoke and construct-link audit (2026-10-17)

## Purpose

Close the checkpoint-byte provenance gap in the synthetic internal-score-path
smoke, and respond to an independent construct audit asking whether the human
reviewed claim actually explains the answer prediction used in the faithfulness
score. No real-data branch was run.

## Setup and verified result

The existing synthetic one-record QTA-Bias/internal visual-key-ablation path
was rerun with five local Qwen2.5-VL-7B-Instruct checkpoint shards hashed by
streaming SHA-256 before model load. Pinned revision:
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. All five computed digests equal
the content-addressed Hugging Face cache blob names. The run completed 1/1
synthetic record and 9/9 forwards in 47.239 seconds on an RTX 5070 Ti,
bf16/4-bit with bf16 compute; peak allocated/reserved VRAM was 10.7549/12.4395
GiB. The private run artifact SHA-256 is
`7dd89c84d799b49e1749255d53f1ffa35e4c07c8f60d014926893dca7fa69933`.

The synthetic target and sham `ΔF` values were each approximately
`−5.96e−8` nat, with a zero difference. They are floating-point residue on
synthetic groups and have no scientific interpretation. The sanitized,
content-hash summary is
[`results/synthetic_qta_checkpoint_hash_smoke_20261018.json`](../results/synthetic_qta_checkpoint_hash_smoke_20261018.json), SHA-256
`9e2e29f9d146a639a9e0b7383d112adbf78270040d8abf295324740f1f4d6cdf`.

The evidence-to-token mapper also received a CPU regression test for an
interval whose legacy contiguous control does not exist but whose union-level
rank-distance control does. The multi-occurrence mapping path now maps each
occurrence without prematurely requiring the legacy control, then applies the
protocol's union-level rank-distance control. The focused mapper suite passed
12/12 tests and the focused QTA composition suite passed 6/6. The final
repository suite passed 1,479/1,479 after the dirfd output hardening and
prediction-link validator changes; Python compilation and `git diff --check`
also passed.

An independent protocol auditor found a construct-link gap: the reviewed
answer-like claim was not bound to the fixed baseline `p/q` predictions later
scored. The draft now requires a post-baseline, pre-treatment adjudication that
the reviewed claim supplies a video-grounded reason for choosing `p` over `q`.
It also harmonizes interval endpoint wording with review policy v6. The
current review packet does not contain the supplemental judgment, and no
real-data score is authorized by this engineering work.

To make this future gate executable, `scripts/nextgqa_prediction_link_review.py`
now hashes and parses the exact baseline-manifest bytes, verifies the expected
cohort token sequence and externally pinned question/options/claim digests,
recomputes `p/q` from the five baseline A–E log probabilities, and binds the
receipt to the displayed question/options/claim/p/q digest and protocol hash.
It fails closed on missing, extra, reordered, or changed rows; its result only
reports the count of supporting self-reported judgments and whether every
expected row has one. Six CPU tests pass. The caller must source expected
hashes/tokens independently;
the validator does not authenticate that source, render a reviewer packet,
authenticate reviewer identity, verify semantic correctness, or authorize scoring.

The independent code audit also found that the smoke output was temporarily
world-readable during write and could chmod a shared parent. The runner now
opens the verified `data/private_nextgqa` root and a fresh run directory via
directory descriptors, creates the run directory as 0700, and uses exclusive
0600 file creation with cleanup of partial output. Seven focused tests cover
modes, overwrite, symlink/outside paths, shared-directory protection, open
failure, and serialization cleanup. This change affects local result handling
only; no protected review or score output was generated.

## Faithfulness progress and limits

Direct real-data faithfulness progress remains **0/3 reviews, 0/3 prediction-link
adjudications, 0/3 scored, and 0/15 verified player-to-decoder correspondences**.
There is no real `ΔF`, confidence interval, or hypothesis test. The rerun
improves checkpoint reproducibility and synthetic execution coverage only; it
does not improve empirical faithfulness evidence. The reviewed-claim mapping
and predictor-link construct gate are prerequisites, not evidence of an effect.
Rights status remains unverified; protected inputs and the private artifact
remain local.

## Offline prediction-link review packet

The function-only renderer was not operational for a reviewer, so it now has a
CLI that reads an exact manifest file and creates an exclusive new HTML file
directly inside a mode-0700 private root. It opens the directory and children
with directory-relative, no-follow operations; refuses non-regular or oversized
input, refuses overwrite, writes output as mode 0600, and removes a partial
output on write failure. The page is static, uses a restrictive CSP with no
network/image/media access, HTML-escapes source text, and exports only the
self-reported receipt with `execution_authorization=not_granted`. The browser's
download flow for that receipt still needs an operator to verify and secure its
destination; the page itself does not guarantee that downloaded receipt's mode.

Five synthetic CPU regression tests passed: hostile markup escaping and frozen
P/Q display, hash/schema/prediction fail-closed behavior, private output mode and
exclusive creation, symlink/unsafe-root rejection, and partial-write cleanup.
The renderer also passed `node --check` on its emitted inline JavaScript. Full
repository tests subsequently passed 1,488/1,488 in 29.593 seconds; `py_compile` and
`git diff --check` passed. The CLI pins its root to
`data/private_nextgqa`, requires current-user ownership and private root/input
modes, and accepts only direct-child names. It has not been exercised in a
browser and no real manifest or reviewer receipt was created. Independent audit
found that the original function-only implementation lacked a practical safe
CLI; after the additions, it confirmed the path, mode, no-follow, size, escaping,
and cleanup controls. Residual limitations are that power loss/process kill can
leave an incomplete exclusively-created HTML file, and the browser's receipt
download destination/mode is not controlled by the page; the reviewer must
verify and secure that local file before custody validation.

## Faithfulness progress and next action

Direct empirical faithfulness remains **0/3 video-claim reviews, 0/3
prediction-link adjudications, 0/3 scored records, 0/15 player-to-decoder
correspondences, and no real-data ΔF, CI, or test**. The renderer and its tests
improve the readiness of the review handoff only. They do not establish that an
explanation predicts or supports a model answer, nor that the internal
intervention improves faithfulness. Rights status remains unverified; the
renderer is designed to keep its HTML inside local protected storage, and no
protected content was published.

Next, build fail-closed preflight binding the existing release/trust-root
ceremony, first-stage human video review, browser-to-decoder clock receipts, and
post-baseline prediction-link receipt. Do not score real data before all these
gates and cohort bindings pass.

The receipt validator received an independent robustness audit. It found that
Python treated JSON `true` as schema version `1`, and that an enormous integer
or unpaired Unicode surrogate could escape the intended stable validation
error. Both manifest/receipt schema version checks now require an actual JSON
integer; non-finite/out-of-range conversions and invalid UTF-8 encodings fail
closed. New synthetic regressions cover all three classes. The independent
auditor also confirmed the core cohort/order/source/display hash checks and
that the result remains explicitly self-reported and non-authorizing.

Added `scripts/validate_nextgqa_prediction_link_review.py` as a local private-file
entry point. It reads the baseline manifest, reviewer receipt, and a separately
frozen cohort/source-digest file as exact mode-0600 direct children of
`data/private_nextgqa`; requires pinned bytes for baseline, source manifest, and
protocol; and writes a mode-0600 structural summary. Its synthetic integration
tests check a successful structural receipt, changed cohort pin and receipt
binding failures, direct-child path restrictions, private mode, and refusal to
overwrite. A structural pass still sets `other_score_gates_verified=false` and
`execution_authorization=not_granted`. It does not verify the source file's
signing/trust-root chain; that must be independently frozen and audited by the
eventual cohort-wide production preflight.

The independent CLI auditor verified the expected source pin, baseline and
protocol bindings, exact token order, private reads/writes, and non-authorizing
summary. It noted that path replacement by a same-user concurrent process is
not fully eliminated on input reads; this is outside the normal local threat
model but should be closed if same-UID adversarial concurrency is included.

The updated focused tests passed 7/7 for the structural validator, 5/5 for the
review page, and 3/3 for the private validation CLI. The complete scripts suite
passed 1,488/1,488 in 29.593 seconds. `py_compile` and `git diff --check` passed.
The emitted review HTML's inline JavaScript also passed `node --check` on a
synthetic manifest. No real receipt, baseline, or source bundle was read or
created.

Current source hashes: validator
`69cea1e22af129268951aaebec693d4cd79c2cfb67a16c018f3ee9d274d8da61`, validator
tests `a7df1b5bb16024c3d980c54d0e457d564894344f7bd093d02f5dfb57d2024507`, CLI
`f32cae3d129ab1b0bcd7a58aced0a6c7e2e6d979c4d55d220f791df59c12606e`, CLI tests
`4a23dbb170b701436b4dee57cc1b8f08b5fa4037f57e020c68c407291faaae4a`.

The future local invocation (not run because no frozen baseline manifest exists)
is:

```bash
.venv/bin/python scripts/render_nextgqa_prediction_link_review.py \
  --manifest-name <frozen-baseline-manifest.json> \
  --output-name <new-private-review-page.html> \
  --expected-manifest-sha256 <64-hex-frozen-manifest-hash> \
  --protocol-sha256 <64-hex-frozen-protocol-hash> \
  --video-claim-review-receipt-sha256 <64-hex-verified-stage1-receipt-hash>
```

After the reviewer returns a secured receipt, the structural validator's
future invocation (also not run; no real receipt or independent source bundle
exists yet) is:

```bash
.venv/bin/python scripts/validate_nextgqa_prediction_link_review.py \
  --baseline-name <frozen-baseline-manifest.json> \
  --receipt-name <secured-human-receipt.json> \
  --cohort-source-name <independently-frozen-source-digests.json> \
  --output-name <new-private-validation-summary.json> \
  --expected-baseline-sha256 <64-hex-baseline-hash> \
  --expected-cohort-source-sha256 <64-hex-source-bundle-hash> \
  --expected-protocol-sha256 <64-hex-protocol-hash>
```

## Stage-link closure after independent construct audit

A separate construct auditor correctly identified two remaining ambiguities:
the second reviewer was not explicitly distinct from the video reviewer, and
the HTML described the prior receipt as displayed although only a verdict and
hashes were shown. The protocol now defines stage 2 as an independent person
making a text-only `claim`-to-fixed-`p/q` discrimination judgment conditional on
the trusted stage-1 video review. It explicitly does not re-verify video
grounding. The renderer now states that the receipt bytes are not displayed or
re-reviewed; only the hash-bound `supports_claim` decision and interval digest
are shown, with trust verification assigned to the not-yet-implemented
pre-score gate. The rendered receipt, validator, and tests require the
stage-2 self-attestation to be independent of the video reviewer as well as
generation, scoring, and custody.

The baseline manifest binds the first-stage receipt hash. Each row binds the
first-stage support decision and exact reviewed-interval digest; both are
included in the fields digest displayed to stage 2. The stage-2 receipt repeats
the prior receipt hash. The validator checks equality across the externally
pinned expected hash, baseline bytes, receipt, and independently frozen source
inventory. Independent code audit confirmed this binding and that a structural
pass remains explicitly non-authorizing. It does not authenticate the prior
receipt or reviewer identity; production trust/release preflight remains
unimplemented and is a hard blocker to real scoring.

For `n=3`, the protocol now defines the secondary full-video gold margin and
accuracy descriptively. They do not replace or broaden the fixed-baseline
`p/q` primary outcome. A positive primary result would still only show a change
in fixed prediction margin sensitivity to claim-indicated key ablation versus
rank-matched control under the specified QTA intervention, not general
explanation faithfulness or improved answer accuracy.

After this change, the three focused suites passed 16/16 tests. The independent
code auditor also ran those 16 tests and `py_compile`/`git diff --check`
successfully, and verified that UI copy, emitted attestation, validator
constant, and fixtures match. The independent construct auditor confirmed the
stage-2 construct is coherent under the conditional text-only definition and
reaffirmed that `n=3` is descriptive only. A later full-suite run is required
after these final wording and regression assertions. No real-data review,
receipt, score, or faithfulness estimate was created.

An independent interface-design audit also identified an execution-count
ambiguity in the three-stage draft: baseline/full is both required to freeze
selector and `p/q`, and is later one of the nine total condition/record
forwards. The protocol now requires retaining that baseline-full five-choice
distribution privately and reusing it, so only eight additional forwards per
record occur after the prediction-link gate (24 total additional forwards for
the three-record cohort). This corrects accounting only; no model run followed.


## Final post-change verification

`PYTHONPATH=scripts .venv/bin/python -m unittest discover -s scripts -p 'test_*.py'` passed 1,489/1,489 in 30.245 seconds after the stage-link changes. The focused three modules passed 16/16, and `compileall` passed. The full suite includes intentionally failing-path tests that print argparse or synthetic diagnostic messages; the process exit code was 0 and unittest reported `OK`. Tests used synthetic fixtures; no model inference or protected input was used.

## Player-clock to decoder-PTS interval mapping

The QTA protocol previously said the review-to-decoder correspondence must be
established but did not state the actual conversion at the mapper boundary.
The existing calibration convention is
`player_time = slope * decoder_PTS + intercept`; the temporal mapper consumes
decoder PTS. The protocol now requires the exact inverse
`decoder_PTS = (player_time - intercept) / slope` for every reviewed endpoint,
using rational arithmetic, and binds converted intervals and calibration
bytes in each frozen source row. It also requires a conservative assignment
stability check: the union of temporal groups touched after expanding each
interval by the empirical held-out PTS residual must equal the union touched
after contracting by it. If not, the row is ambiguous and fails before model
loading. The helper `map_annotation_intervals_to_pts()` implements the inverse;
`map_review_intervals_to_qwen_groups(...,
interval_boundary_guard_seconds=...)` now enforces possible/certain group-set
equality. The guard remains an observed residual radius, not a confidence or
future-error bound.

Targeted clock/calibration/group-mapper tests passed 28/28, including affine
offset/drift conversion, malformed values, a stable boundary assignment, and a
guard large enough to change group assignment. The complete scripts suite then
passed 1,493/1,493 in 28.966 seconds; compilation and `git diff --check` passed.
This is synthetic CPU validation only. The tolerance and held-out anchor set
must still be frozen independently before real captures; no actual clock
capture, 15/15 frame correspondence, interval assignment, or real score exists.

Independent follow-up found the new guard parameter had not yet been threaded
through the processor-batch adapter, a concrete float midpoint could turn an
exact zero-overlap touch into a tiny positive overlap, and the standalone
interval converter did not enforce the v6 millisecond endpoint precision. The
adapter now requires an explicit calibrated guard (omission fails), carries
exact rational PTS, intervals, duration, and guard in its digest, and forwards
the guard into the group mapper. Group membership and possible/certain guard
sets now use `Fraction` comparisons through classification; float conversions
occur only for human-readable diagnostics. The converter rejects endpoints
off the 1 ms grid. Synthetic regressions cover an exact rational midpoint touch,
guard propagation/omission, and sub-millisecond endpoints.

The affected clock, group, and processor-batch suites passed 35/35; the entire
scripts suite passed 1,495/1,495 in 30.567 seconds, with exit code 0. This
closes the code-level clock-axis/guard path for future callers; it does not
create a calibration, authenticate anchors, set an acceptable tolerance, or
establish evidence for any real video. Anchors must cover the reviewed time
range; max held-out residual remains descriptive and is not a future-error
bound. A production preflight still must bind the exact calibration artifact,
convert and store the rational intervals, pass the guard to this adapter, and
verify the 15 human-browser/decoder correspondences before model loading.

An independent final audit then found that the interval converter emits
canonical rational strings while the processor-batch adapter initially
accepted only numeric values. The adapter now parses only canonical rational
strings (and continues to reject malformed forms); a direct converter-to-
adapter regression test covers the handoff. After this correction, the focused
clock/calibration/group/adapter tests passed 36/36 and the complete scripts
suite passed 1,496/1,496 in 29.789 seconds. `compileall` and
`git diff --check` also passed. These remain synthetic CPU checks. The 15
browser-to-decoder correspondences, calibration anchors, reviewer decisions,
and any real-data score remain absent, so this code does not authorize model
loading or support an empirical faithfulness claim.

## Three-row pre-score mapping join hardening (2026-10-18)

A CPU-only non-authorizing join was added in `scripts/qta_prescore_mapping_binding.py` to bind the existing review, calibration, processor-mapping, and baseline audit outputs for the fixed three-row cohort. It retains all three requested records on failure and always returns zero branch forwards and `execution_authorization=not_granted`.

The join now recomputes the canonical ordered interval digest and player-to-PTS transformation; requires the externally pinned review-payload and review-receipt hashes; verifies a canonical rational calibration payload digest over source-video hash, slope/intercept, anchor span, and guard against both independent calibration pins and scalar clock values; checks exact processor batch and sampling-metadata pins; checks temporal indices against grid bounds and exact independently pinned evidence/control mappings; and compares question/options/claim digests with their per-token pins. `audit_nextgqa_review_handoff.py` now uses the same strict decimal millisecond-grid interpretation as the join, rejecting near-grid floats such as `1.0010000000000001`. Exact duplicate intervals remain recorded in order and are counted once by the downstream interval union mapper.

Focused review/clock/batch-join tests passed 33/33. The full scripts suite passed 1,506/1,506 in 30.043 seconds; `compileall` and `git diff --check` passed. An independent code audit verified the calibration payload hash/value linkage and the externally pinned sampling/group mapping checks; the supplied pins themselves remain unauthenticated by this helper. These are structural regression results only.

Faithfulness progress is unchanged: **0/3** independent video-claim reviews, **0/3** prediction-link adjudications, **0/3** scored, **0/15** player-to-decoder correspondences, and no real-data `ΔF`, confidence interval, or test. The local synthetic GPU runs remain numerical/engineering diagnostics and do not count as evidence of faithfulness improvement. Rights status remains unverified; protected inputs and derived outputs remain local.

Next, integrate these joins into one fail-closed cohort preflight only after the exact trust-root/release and clock-receipt schemas are bound. Separately, the existing three video rows still require the independent human review and frame correspondences before any real-data baseline or treatment forward can be considered.

## Prescore row-adapter integration and audit (2026-10-18)

Added three CPU-only projections to reduce field-assembly drift before the fixed-cohort QTA mapping join: `qta_prescore_review_row_adapter.py` binds worksheet/review object bytes and row order to the bundle review hash; `qta_prescore_processor_row_adapter.py` allowlists the serializable mapper fields and omits its private processor snapshot; and `qta_prescore_clock_row_adapter.py` binds the exact upstream calibration artifact, its raw JSON, capture hash/schema, fixed fit/held-out anchor IDs, canonical rational times, and empirical-only uncertainty declarations into the join payload.

An independent audit found that the first clock projection version accepted a self-consistent artifact with missing builder fields. The adapter now requires the exact bound artifact key set and fixed builder declarations (`clock_scope`, calibration policy/thresholds, empirical-only validation scope, observed-residual guard source, not-evaluated uncertainty, and the wrapper's limitations). A synthetic integration test runs the actual upstream `build_bound_html5_clock_calibration` then adapts its returned bytes/object. The independent re-audit found the reported gap closed and no new concrete issue in this delta.

Observed checks at the initial adapter commit: focused adapter suite **18/18 passed**; complete scripts suite **1,524/1,524 passed** in 30.321 seconds; `compileall`, `py_compile`, and `git diff --check` passed. All adapter inputs and test examples are synthetic. A successful projection authenticates neither human review, reviewer identity, upstream bundle digests, browser telemetry, nor player-to-decoder frame correspondence. All outputs continue to declare correspondence unverified and `execution_authorization=not_granted`.

Faithfulness progress remains **0/3** independent claim reviews, **0/3** prediction-link adjudications, **0/3** scored, **0/15** player-to-decoder correspondences, and no real-data `ΔF`, interval estimate, or test. This work improves structural handoff reliability only; it does not change treatment efficacy evidence. Next: bind the remaining trust-root/release and frame-correspondence receipts into a cohort-wide preflight, while keeping real inference blocked until the independent reviews and all upstream gates exist.

## Review-adapter source-audit rerun hardening (2026-10-18)

Independent audit identified a bypass in the caller contract: the review-row adapter accepted a caller-constructed dictionary that looked like a successful handoff-audit report. Replaced that parameter with the exact worksheet, lineage, review, and index raw byte bundle plus the frozen lineage hash; the adapter now strict-parses those bytes and reruns `audit_nextgqa_review_handoff.audit` internally before projecting rows. A regression confirms that a forged summary is not an accepted argument and changed or incomplete review bytes fail closed. This does not establish a human review, custody, key-holder identity, signed release chronology, or clock/frame correspondence; those remain separate upstream components for the future cohort audit.

The independent audit found the caller-summary bypass closed, with no remaining concrete adapter mismatch. It noted an unhashable `support_decision` could throw `TypeError` inside the underlying handoff validator. The validator now type-checks before set membership, and regression tests confirm list/dict values return the normal invalid-report status. Focused handoff plus review-adapter tests passed **25/25**; focused review/processor/clock adapter tests passed **17/17**. Final post-edit full suite passed **1,524/1,524** in 29.492 seconds. `compileall` and `git diff --check` passed. Faithfulness effect evidence remains unchanged at **0/3** reviewed, adjudicated, or scored; **0/15** human frame correspondences; no empirical `ΔF` or uncertainty estimate.

## Three-row stage-1 bundle join and independent security audit (2026-10-18)

Added `scripts/qta_stage1_bundle_audit.py`, a CPU-only, non-authorizing aggregate that reruns the source-byte handoff, custody, signed release-witness, and reviewer-key-adoption validators. It binds the exact worksheet, lineage, custody, both standalone and packaged review HTML, and all three staged/package media roles to the signed inventory, and rerenders both HTML artifacts from the same source snapshots. The output separates `structurally_valid_records` from human validation and always leaves model loading and scoring disabled; browser-clock, frame-correspondence, and calibration gates are explicitly outside this stage.

Independent adversarial review found that the initial implementation could open a media path derived from an unvalidated worksheet token before custody rejected it. Fixed this by validating exactly three unique 24-character tokens before any media access, opening only a private owner-only media directory with `O_DIRECTORY|O_NOFOLLOW`, requiring an exact filename set, and opening canonical leaf names relative to that directory descriptor. Added regressions proving a `../sentinel` token never opens its sibling file and a symlinked media directory is rejected before any media open. The earlier role-mismatch, standalone HTML, post-snapshot media mutation, non-disclosure, and non-authorizing status cases remain covered.

Observed checks after the security fix: focused stage-1/handoff/render tests passed **28/28**; the complete scripts suite passed **1,533/1,533** in **32.599 seconds**. `py_compile` and `git diff --check` also passed. All fixtures are synthetic. No protected video, human review, or browser capture was consumed; the join creates no faithfulness score and does not authenticate identity or semantic review. Current direct evidence remains **0/3** claim reviews, **0/3** prediction-link judgments, **0/3** scored, **0/15** player-to-decoder correspondences, and no real `ΔF`, CI, or test. Next work is the independently reviewed clock/frame evidence and calibration chain; its browser screenshots and 15 blinded pair judgments do not exist yet, so real scoring remains blocked.
