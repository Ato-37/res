# Open-ended faithfulness PDCA — 2026-10-17

Requested scope: continue useful faithfulness research until the user explicitly
says `stop`; no finite duration or deadline was specified. Accounting began
conservatively at **2026-10-17 04:04:59 JST** (**2026-10-16 19:04:59 UTC**),
the first observed clock after the instruction. Rights status remains
unverified; protected inputs and derived records remain local.

## Cycle 1 — repair the private human-review handoff

**Purpose.** Make the existing three-video exploratory review package usable
without misdescribing what its browser exports. This prepares a human gate; it
does not produce faithfulness measurements or authorize model scoring.

**Changes.** Kept frozen `docs/nextgqa_claim_review_policy_v5.json` unchanged
and added a versioned implementation disclosure in
`docs/nextgqa_claim_review_export_contract_v5_renderer_v1.md`. The renderer now
lists the full review JSON, review index, optional clock payload, clock-event,
and clock-index fields before the review rows. It explicitly states that the
frozen policy's shorter export sentence does not match the renderer. Export is
disabled until the reviewer checks an acknowledgment; the boolean is recorded
as `export_disclosure_acknowledged`. The page still says that the acknowledgment
and browser timestamps are self-reports and that no score authorization is
granted. The overview now keeps review held until such a newly disclosed page is
used.

**Checks and observed results.** The focused renderer suite passed 9/9 and the
full script suite passed 1,074/1,074 (`.venv`, Python 3.12). `compileall` and
`git diff --check` passed. A new offline page was rendered exclusively from
the reconstructed 3-row exploratory snapshot and local manifests, with mode
0600 and SHA-256
`e833adaaf81794e2214bfca7dc6e44e7092eec077034bb8696e0e2cec5c7fae1` at
`data/private_nextgqa/claim_review_reconstructed_disclosed_20261017_0412.html`.
Static checks confirmed the full export inventories and acknowledgment appear
before the question/claim rows, export requires and records the acknowledgment,
the manifest and sidecar digest metadata are present, and the page has no
network endpoint or scoring authorization. An independent metadata-only audit
reproduced the page hash and checked its disclosure, mode, 3-record binding,
manifest/sidecar hashes, frozen policy hash, and held `review_completed=false`,
`scoring_ready=false`, `execution_authorization=not_granted` states. No claim
text or video bytes were read during that independent audit.

**Faithfulness progress.** Direct claim-to-video faithfulness remains **0/3
human-reviewed and 0/3 scored**. There is no direct effect estimate, uncertainty
interval, or test. This cycle made the human-review package handoff-ready; it
did not advance the direct empirical estimate. The three-item manifest is a
new reconstructed exploratory snapshot, not recovery of the historical frozen
cohort. Its pinned cohort digest is declaration-based, not independently
recomputed from the full planner inputs. Rights status remains unverified.

**GPU decision.** No GPU experiment was run in this cycle. Human review,
player-to-decoder PTS calibration, a frozen finite target/protocol, and a
validated atomic all-item launch gate remain prerequisites for a meaningful
claim-mask score. The earlier GPU QA-gold interval margin runs are a different
auxiliary estimand, and rerunning them would not resolve these blockers. The
first direct GPU experiment must cover the complete three-item finite cohort
atomically (nine forwards); the current protocol does not permit a one-item
score or survivor averaging. A smaller scored smoke would need its own
predeclared, outcome-excluded target. Use the pinned local checkpoint.

**Next action.** Continue implementing and testing the real-data target builder,
review/calibration handoff, and atomic scorer prerequisites while preserving
zero-score behavior on any gate failure. As soon as independent human review
and actual player-clock/decoder-PTS validation are available, freeze the exact
input bundle and run the complete three-item GPU experiment. Do not reopen the closed confirmatory head search; if no more useful
claim-review work remains, the user's latest instruction permits a separately
predeclared exploratory head study, clearly separated from the failed
confirmatory branch.

## Cycle 2 — keep the new acknowledgment consumable downstream

An independent static audit found that the renderer's new top-level
`export_disclosure_acknowledged` field would make the existing raw handoff audit
and assignment planner reject every completed review, because both enforce an
exact v5 key set. Added this field to both exact schemas and require the JSON
boolean `true`. Missing or false values fail closed. Added fixtures and negative
tests for both conditions. The audit still marks the handoff as structurally
unvalidated and the assignment planner still returns zero model calls and
`execution_authorization=not_granted`; disclosure acknowledgment alone cannot
authorize inference.

The two focused suites (`test_audit_nextgqa_review_handoff` and
`test_claim_review_assignment_plan`) pass **42/42** including the new cases.
Independent audit confirmed false/missing acknowledgments fail closed, while
true remains structurally unvalidated with zero calls and no authorization.
The full script suite then passed **1,076/1,076**; `compileall` and
`git diff --check` passed. This is an integration repair, not a new empirical
result. The next implementation target is the real source-backed QA
target/tokenizer builder, followed by the smallest honest clock-calibration
path. No installed browser or browser automation package is available in this
WSL environment, so actual player-to-decoder calibration remains a runtime
gate rather than something to replace with synthetic evidence.

Further integration review found the private package CLI rerendered pages
without the optional v2 claim manifest/sidecar, so it would reject the new
bundle-bound page. Added paired CLI inputs and exact renderer-byte comparison
with both artifacts included; incomplete pairs fail before copying. The
synthetic positive CLI path now packages a bundle-bound page, and negative
tests cover a missing pair and a schema-valid changed bundle with the old page.
The latter confirms rejection at exact page-byte comparison before creating
output. Focused package/renderer tests pass **15/15**; an independent auditor
confirmed the changed bundle remains structurally valid. The actual private
package at
`data/private_nextgqa/reviewer_package_reconstructed_disclosed_20261017_0420/`
contains only `index.html` and three token-named videos. Independent
metadata-only verification confirmed directory mode 0700, file modes 0600,
and page hash `e833adaaf81794e2214bfca7dc6e44e7092eec077034bb8696e0e2cec5c7fae1`.
This closes handoff plumbing only; no reviewer opened the page and no model/GPU
score was run. After this regression, the full scripts suite passed
**1,087/1,087**; `compileall` and `git diff --check` passed. No synthetic test
was interpreted as a real review or GPU result.

## Cycle 3 — bind the actual QA target candidate to the reviewed claim input

The existing 32-frame diagnostic target builder bound actual NExT-GQA CSV rows,
Qwen's prompt/processor path, and A–E tokenizer declarations, but explicitly
marked its claim-input manifest absent. Extended it to accept the v2
claim-manifest/sidecar pair, verify exact manifest bytes, policy/model/cohort
declarations, worksheet/lineage/CSV/generation hashes, and the per-video claim
and source-row joins. It fails closed on an incomplete pair, stale digest,
source-hash mismatch, or claim-row mismatch. The resulting artifact remains
diagnostic-only and has no score or authorization fields set to success.

Focused synthetic tests pass **15/15**; they cover positive binding and pair,
manifest digest, worksheet hash, and claim-text failures. A real local CPU-only
run bound the reconstructed claim manifest to **3/3** source QA rows and
processed **32 frames per video** through the pinned Qwen processor/tokenizer.
All five answer labels were one token with distinct IDs; prefix and
processor-input-ID hashes were produced for each row. The private artifact
`data/private_nextgqa/diagnostic_target_claimbound_20261017_0425.json` is mode
0600, SHA-256
`6219bb9b98b7bc66f038f9ccf58f13e361eedb58064d2f2dd9fa4a379cf92e7d`. It
records the exact claim-manifest digest
`7ebc1dffe883cc2a3ccf248b6c53a03271c129a5aed0adb6a0bdb429cd9d7743`, while
remaining `diagnostic_candidate_unvalidated`, `worksheet_review_completed=false`,
`model_weights_loaded=false`, `cuda_used=false`, zero network/model/score calls,
and `execution_authorization=not_granted`. No question, answer, claim, or video
content was printed. This is real-source target/prompt plumbing, not
faithfulness evidence.

An independent metadata-only source and artifact audit confirmed the joins but
found that the artifact limitation incorrectly said no claim manifest existed.
Cycle 4 below records the correction and rebuild. The
remaining score blockers are the still-pending human review, real HTML5-player
to decoder-PTS calibration (no browser is installed here), a jointly frozen
finite-cohort protocol/inputs, and an audited production atomic scorer. The
first valid direct GPU run remains the complete three-row cohort with nine
forwards; a one-row score is not authorized by this design. Continue resolving
independent CPU/source/protocol gates and offer the already prepared local
review page to an independent reviewer; do not spend GPU time on another
QA-gold interval proxy that cannot satisfy the claim-specific gate.

## Cycle 4 — correct audit-discovered limitation and revalidate

The independent audit verified the target candidate's source joins and
artifact metadata, then found a contradiction: a claim manifest was present
and hash-bound, while the artifact still said it was absent. Changed the
limitation to appear only when no manifest was supplied, and added regression
assertions for both manifest-present and manifest-absent cases. Rebuilt the
candidate to a new exclusive private path
`data/private_nextgqa/diagnostic_target_claimbound_disclosed_20261017_0435.json`
with mode 0600 and SHA-256
`917d52269b14ef0a49b07e46d770c993e694ce0c53a431eff0b6dde48d456b86`.

The focused builder suite passed **15/15**; the complete script suite passed
**1089/1089**; `compileall` and `git diff --check` passed. The independent
auditor then noted the artifact did not record the exact sidecar bytes. Added
`claim_input_sidecar_sha256`, tested both presence/absence cases, and rebuilt
again to
`data/private_nextgqa/diagnostic_target_fully_bound_20261017_0427.json`, mode
0600, SHA-256
`e43bf9abc553c094570c5da49c3f47284c7400a5519775889bfa808af8282795`.
The final independent metadata-only audit confirms exact local manifest and
sidecar hashes, three-way row joins, and all source hash bindings. This remains CPU-only target
construction: 3/3 source rows and claims bound, 32 frames per row, zero model
weights, CUDA, network, forwards, or score calls. It is not a faithfulness
result. The artifact still uses declared upstream eligibility/cohort digests;
those are not independently reconstructed from all source records.

The safe source implementation was committed as `207e43aff930b8a723dd9fa751f8eb1d480d409f`
and live-verified on GitHub `main`. The staged/pushed scope contained only
review-page implementation/disclosure, hash-bound target code, and synthetic
regression tests; local videos, outputs, and descriptive results stayed out of
the commit.

Direct faithfulness remains **0/3 reviewed and 0/3 scored**. The first
claim-specific GPU run is still blocked by human review, real player-clock to
decoder-PTS calibration, and a production all-record atomic scoring gate.
Accordingly no GPU job was launched: the limited GPU resource is reserved for
the complete informative three-row comparison once those gates pass. Next,
finish an independently auditable clock/calibration pathway and inspect the
atomic scorer interface, then hand the local review package to an independent
human reviewer. Keep all protected data and artifacts local.

## 04:42 JST — bind clock capture to the anchor click

The independent audit of the first fresh-callback renderer found a timing race:
although stale cached metadata was eliminated, a video that continued playing
could advance between the anchor click and callback. The previous check compared
`mediaTime` only with `currentTime` read at callback, so it did not establish
which requested anchor the callback represented.

Updated the renderer to record player time and monotonic request time at the
click, require callback delay and callback-to-handler age <=250 ms, and require
callback `mediaTime` within 0.1 s of both click-time and callback-time player
positions. A 1 s wait remains a timeout only; callbacks beyond the 250 ms
request-delay bound fail closed. The converter and calibrator now require schema
v3, the fixed `rvfc_click_bound_v1` identifier, the exact threshold object, and
independently recheck click-time, request-delay, and callback freshness. The
protocol draft was updated and its exact SHA-256 is
`98c113102530f9ca7ca37607c4bfbf411339f64ec9f1f3957ecd2829fe3b0543`; both
candidate target and synthetic-preflight pins were updated to that digest.
Synthetic regressions cover late request, click-time mismatch, and threshold
mutation.

Root checks: renderer **9/9**, clock converter **16/16**, calibrator **6/6**,
complete script suite **1104/1104**, Python `compileall`, `git diff --check`,
and Node 24.18 JS syntax check all pass. A new private click-bound reviewer
package is at
`data/private_nextgqa/reviewer_package_clickbound_20261017_0443/` (4 files;
private directory mode 0700; each file mode 0600; HTML SHA-256
`aed73b06f3e7a76ad1763c66fb0042419b4e8eb3bf3713ef4b9ef15a308bb78d`). Its
review disclosure lists the actual review/clock outputs and all threshold
values. The package is rebuilt from the existing local custody-checked videos;
no human opened it, no browser is installed in this WSL environment, and no
actual HTML5/decoder correspondence or PTS calibration was observed. The
second independent audit of the click-bound code is pending.

Direct claim-faithfulness remains **0/3 reviewed, 0/3 scored**, with no effect
estimate or uncertainty interval. GPU access is available, but a claim-specific
GPU run remains blocked by independent human review, real browser-to-decoder
clock calibration, and a production atomic all-three-row scorer. This cycle
made the handoff and calibration telemetry stricter; it did not create a
faithfulness result. No GPU job was launched because another smoke would repeat
the already logged real-video/Qwen CUDA path checks without answering the
claim-faithfulness question. Next: receive the independent code audit, then
continue the smallest unresolved scorer/provenance gate while keeping the
review package ready for an independent human.


## 04:46 JST — implement the nine-forward margin kernel

To reduce integration time after the review/clock gates clear, added a small
inference/math core for exactly three rows × (`full`, `claim_evidence_black`,
`temporal_control_black`). It validates the entire 3×3 batch container and
five distinct label-token IDs before the first forward, collects only those
five next-token log-probabilities, stores each condition's best-foil margin,
computes `d_i = control - claim_mask`, and emits `D3` only after nine completed
forwards and three complete triplets. A failure stops at that point, preserves
the frozen denominator of three, gives sanitized failure type, reports partial
completed triplets and leaves D3 null; no CI or p-value is generated.

A synthetic CPU-logits test suite checks the known +3.0 nat per-row contrast,
exactly nine success forwards, rejection of an incomplete ninth condition
with zero calls, CPU materialization of all nine batches before the first
forward, duplicate answer-token rejection, non-finite logits, sanitized
resource-guard and device-token tensor failures, and null D3 after a failure on
forward eight. Focused checks pass **8/8** and Python compilation/diff-check
pass. Independent audit confirmed the previous version; this final failure
path still needs a follow-up check. This module is only
the scoring kernel; it does not authenticate input batches, review, clock,
cohort, target, code freeze, model loading, or authorization. The production
atomic all-nine preflight remains a separate major blocker; no GPU inference
was run.

## 04:49 JST — scorer kernel audit and runtime check

Independent audit of the current scorer core verified the fixed margin formula,
contrast sign, all-three-row equal-weight D3, and failure behavior. The focused
scorer suite passed **8/8**; the full script suite passed **1112/1112**. The
auditor found no material math or tested failure-semantics issue. The core's
`.to("cpu")` calls are only a materialization check: they discard the returned
objects and do not authenticate, hash, or freeze their tensor contents. The
core deliberately accepts already-preflighted batches, so it is not a
production scorer and must not be called on the diagnostic target artifact.

An individually elevated runtime check in the repository `.venv` confirmed
PyTorch **2.11.0+cu128**, CUDA build **12.8**, and an RTX 5070 Ti with 16,303
MiB total and 966 MiB in use at 04:49 JST. Ordinary sandbox `nvidia-smi`
remains blocked by OS visibility. This confirms runtime availability, not
permission to bypass scientific gates. No GPU forward was run.

Independent production-path review established the shortest valid next path:
an all-three-row atomic CLI must authenticate the complete review and timing
evidence, exact source/cohort and protocol freeze, decoder frame/PTS identity,
and all nine post-mask Qwen processor inputs before loading model weights. It
must persist/recheck hashes for all tensor fields and conditions, then invoke
the audited math core exactly nine times; any missing gate must yield zero
model loads and zero forwards. Reusable source, tokenizer, decoder, and mask
helpers exist, but no signed/evidence-reading review audit, real browser/decoder
calibration, exact nine-batch freeze, or production caller exists yet. Actual
three-video human review and browser observations require an independent human
and reviewer browser; these cannot be generated by synthetic or GPU work here.

The overview was corrected from 1104 to the verified 1119-test count and now
states the independent math audit and CUDA check without claiming efficacy.
Direct evidence remains **0/3 reviewed, 0/3 scored**, with no effect estimate,
interval, or test. The cohort remains an exploratory snapshot. Local protected
materials remain uncommitted; rights status is unverified.

**Next action.** Implement the fail-closed all-row exact-input bundle verifier
and production orchestration in bounded steps, beginning with byte/tensor
hashing and required evidence manifest validation. Run it only in its blocked,
zero-model-load state until genuine human and browser/decoder evidence exists.
Once all gates pass, expected workload is 3 rows × 3 conditions = 9 Qwen
forwards, using the pinned model; estimate runtime/VRAM after a non-scoring
resource check and verify available memory immediately before that run.

## 04:54 JST — exact batch digest primitive

Added `scripts/nextgqa_batch_digest_freeze.py` as a narrow integrity primitive
for the future exact input bundle. It requires three ordered rows and all three
fixed conditions per row, identical keys across all nine batch mappings, and
records gold index, condition/row order, nested mapping/container structure,
tensor dtype/shape/byte hashes, plus typed primitive fields. It rejects
non-finite tensors/scalars and arbitrary objects. Its output explicitly says
`unvalidated_digest_manifest_only`, `source_authentication=not_performed`,
and `execution_authorization=not_granted`; it does no I/O, model loading, or
scoring.

The independent implementation audit found an important integration edge:
pinned Transformers 5.18.0 returns `video_metadata` as a non-tensor `Mapping`
by default. Primary-source inspection confirmed `VideoMetadata` contains only
finite scalar/list fields for this path and is intentionally excluded from
tensor conversion. The helper was amended to encode these fields with explicit
type tags (including float64 hexadecimal values), while still rejecting
arbitrary objects. A mutation of metadata now invalidates the digest. Focused
tests pass **7/7** and the complete script suite passes **1119/1119**; module
compilation and `git diff --check` pass. No real Qwen batch was digested, and no
GPU forward was run. A content digest is not provenance, a signature, a protocol
freeze, or scoring authorization.

Direct claim-faithfulness is still **0/3 reviewed, 0/3 scored**, with no effect
estimate, interval, or test. The fastest remaining route is human review plus
browser/decoder evidence while implementing the caller that checks all source,
protocol, review, clock, and condition-input gates before model load. The
digest primitive reduces one part of that caller's verification work but does
not authorize the existing processor batches. Keep protected artifacts local;
rights status remains unverified. The privacy-safe source/test/overview scope
was committed as `60104727869816f3e10d00e9c5adad29f1bfe78f` and the live GitHub
`main` reference was independently verified at the same SHA. The local
experiment log and all protected artifacts were excluded from that commit.

## 05:01 JST — construct and exercise the complete synthetic 3 × 3 input path

Added `scripts/nextgqa_condition_batch_builder.py`. Given three already
decoded records and caller-provided exact PTS/intervals, it first validates all
three frame assignments, then constructs full, black claim-evidence, and
rank-matched black temporal-control frames for each. It routes all nine
conditions through the existing Qwen prompt/processor path, checks prompt and
non-image processor inputs stay identical within a row, and returns an
in-memory digest bundle explicitly marked unvalidated with no source/review/
clock authentication, model load, forward, or authorization. Any malformed
third row fails before vision/processor calls; errors do not return partial
triplets.

Independent audit first found the inherited RGB-mean check could miss spatial
pixel changes. Replaced it with exact reconstruction of Qwen's installed
`fetch_image` plus video resize path and `torch.equal` against the emitted
vision tensor. Synthetic tests now reject both reversed-frame and spatial-flip
pixels before the processor runs. Focused builder tests pass **6/6**. A local
offline integration check instantiated the pinned Transformers **5.18.0**
processor and `qwen_vl_utils` on three synthetic 32-frame records: it created
**9/9** condition batches, independently rehashed the complete digest, and
reported `model_loaded=false`, `model_forwards=0`, and
`execution_authorization=not_granted`. The observed model batch keys were
`input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values_videos`,
`second_per_grid_ts`, and `video_grid_thw`. This integration used synthetic
frames, not local dataset videos, and is not a faithfulness experiment.

After the exact-pixel refinement, the complete script suite passed
**1125/1125** and the independent implementation audit confirmed the exact
frame-order/content comparison against installed `qwen_vl_utils` source.
The verifier uses the output spatial size for its reconstruction, so it does
not independently establish whether Qwen chose the policy-correct resolution;
library/source version authentication is also a separate gate. Direct
claim-faithfulness remains **0/3 reviewed and 0/3 scored**, with no effect
estimate/interval/test. The new helper closes condition-to-processor plumbing
for a supplied cohort but does not authenticate that cohort, human intervals,
decoder clock, or source frames. The next actual-data work still requires the
independent three-video human review and real browser-to-decoder calibration;
then freeze all inputs/protocol and run the one complete nine-forward GPU
comparison. Keep rights status unverified and protected inputs local.

## 05:11 JST — snapshot all nine post-mask processor batches before inference

An independent implementation task added
`scripts/nextgqa_atomic_batch_freeze.py`. It enforces the exact three record
IDs/order, three fixed conditions per row, five distinct answer token IDs,
CPU-materializes all nine batches, clones tensor storage and nested containers,
rejects non-finite or unsupported values, and compares each batch against an
expected digest map before calling an upstream validation hook. It returns a
scorer-compatible CPU snapshot with `execution_authorization=not_granted`; it
has no model loader or forward path. This closes the source-alias gap where a
CPU `.to("cpu")` could return the original tensor and a later hook could mutate
it after hashing.

Focused tests pass **7/7**. Independent audit confirmed the detached snapshot
survives mutation of the source tensor and that incomplete/mismatched/nonfinite
inputs fail before the hook. The same audit found the return value remains
mutable after return: a caller can change a tensor and invalidate its stored
digest. Thus this is a hash-checked snapshot, not an immutable freeze; any
future production caller must rehash against the recorded values immediately
before model execution and connect the hook to concrete reviewed validators.
An arbitrary callable returning `True` is accepted in test and does not
authenticate anything. `py_compile` and `git diff --check` passed. The complete
repository scripts suite passed **1132/1132** in 14.945 seconds. The initial
test invocation omitted `PYTHONPATH=scripts` and failed import; the corrected
repository-standard invocation passed all tests.

**Faithfulness progress:** direct claim-linked evaluation remains **0/3
reviewed, 0/3 scored**, with no effect estimate, interval, or test. This is an
engineering/provenance integrity diagnostic only. No GPU forward occurred.
The earlier QA-gold evidence-mask GPU outcomes remain a separate estimand and
are not counted as generated-claim faithfulness. Human review and actual
browser-to-decoder PTS/frame correspondence remain outstanding; protocol and
all-source evidence are not frozen. The three candidate videos remain an
exploratory finite pilot only and are quarantined from confirmatory scale-up.

**Next action:** bind this snapshot to an independently reviewed all-source
and timing preflight and add a scorer-side expected-digest recheck immediately
before the nine forwards. Do not load model weights or score while reviewer
and clock evidence are absent. Keep all protected artifacts local and rights
status unverified.

## 05:16 JST — recheck hashes on the exact device batch before each forward

The scorer now requires hashes for all nine exact condition batches and the
ordered record IDs from the snapshot. It hashes the full cohort before any
forward, then hashes each transferred device batch again immediately before
its model call. It accepts only exact `FrozenBatch` instances, with the type
check before generic validation invokes `.to()`. Tests cover changed later and
current inputs, a mapping with arbitrary `.to()`, and a `FrozenBatch` subclass
whose overridden `.to()` would have side effects; changed or substituted
inputs stop before forwarding. The independent audit confirmed the freezer's
exact output remains accepted. Focused snapshot+scorer tests pass **19/19**;
the complete scripts suite passes **1136/1136** in 14.563 seconds. `py_compile`
and `git diff --check` pass.

A bounded elevated CUDA check transferred a **4,000,012-byte** synthetic
`float32` video tensor plus integer token/grid tensors to the RTX 5070 Ti and
recomputed the digest. The CPU and CUDA batch digests matched exactly. This
confirms value/dtype/shape preservation on that synthetic transfer only; it
loaded no model, ran no model forward, and touched no video or QA content.

**Faithfulness progress:** still **0/3 independently reviewed and 0/3 scored**;
no direct effect estimate, interval, or test. This GPU diagnostic is strictly
engineering validation. The scorer remains an inference core without a
production caller or authenticated gate. A GPU digest pass does not authenticate
media, human intervals, browser clock, source decoder frames, or the draft
protocol. No valid faster faithfulness experiment exists under the current
protocol. The next useful action is still to complete the real independent
review and browser-to-decoder calibration, then bind concrete audit validators
and frozen input hashes before exactly nine forwards. Rights status remains
unverified; protected materials and derived outputs stay local.

The privacy-safe implementation handoff is live on GitHub `main`: the batch
snapshot primitive was published in `24417abd528cf2735762c080124d82c5dfdfd4d5`,
and scorer-side digest verification in `ce7b8ed2a50e7a6c7f8cb5a4e7ba7617a9dd069e`.
A live `git ls-remote` confirmed `main` at the latter SHA. Only source/tests
and the aggregate research overview were published; private review package,
protected data, manifests, and this local PDCA log were excluded.

## 05:18 JST — exercise the freezer-to-scorer chain as one CPU integration

Added a synthetic integration test that builds all nine toy condition batches,
computes the expected digest map, passes them through `freeze_all_batches`,
and directly supplies its returned batches/token IDs/record IDs/hashes to the
scorer. It asserts exactly 9/9 calls, a complete D3 value matching the known
synthetic logits, and no authorization. The test docstring explicitly labels
that D3 assertion as software validation only. It does not invoke Qwen or use
any real video/QA. The independent auditor confirmed the integration boundary
and the tests now pass **20/20** across the freezer and scorer suites. The full
scripts suite passes **1137/1137** in 14.751 seconds; compilation and diff
checks pass. Earlier bounded GPU hash test remains a 4 MB synthetic transfer;
no model forward occurred.

**Faithfulness progress remains unchanged:** `未実施`, **0/3 reviewed and 0/3
scored**, with no generated-claim effect estimate, uncertainty interval, or
test. Engineering execution readiness is better, but no production caller
connects the synthetic gate hook to independent human review/custody, actual
browser-to-decoder correspondence, or a frozen protocol/source bundle. No
model weights should be loaded for this comparison until those gates pass.
The review HTML path remains
`data/private_nextgqa/reviewer_package_clickbound_20261017_0443/index.html`
(local only, directory 0700 and files 0600). An independent reviewer and
separate custodian must complete their recorded handoff; actual browser capture
and decoder-frame/PTS matching must also be independently audited. Keep rights
status unverified and all protected data/results local.

The end-to-end synthetic integration test and current aggregate overview are
live on GitHub `main` at `df780f7cd54f9cbdb332310322abfe590356c30e`; live
`git ls-remote` matched the local HEAD. Together with `24417ab` and `ce7b8ed`,
these scoped commits contain only source/tests and aggregate engineering
status. The private local log, review package, videos, QA inputs, and score
artifacts remain excluded.

## 05:21 JST — reduce reviewer handoff loss risk

An independent workflow audit found that the offline reviewer page needs two
separate downloads for the review payload and its integrity index. After the
first download the page held the index only in memory; closing/reloading the
page made it impossible to finish the handoff without repeating the review.
The page also gave no clear post-download instruction for returning both files
privately. Updated the page to show the required 1/2 and 2/2 steps, state that
browser download initiation is not proof of disk persistence, keep the page
open until the index download is initiated, disable the index button after its
single use, and direct both files (plus clock payload/index when collected) to
the private custodian. No review payload is persisted to localStorage because
it contains reviewer identifiers and free-text video-derived rationales. This
change does not alter review JSON/index schemas, semantic review status, or
scoring authorization.

The focused renderer suite passes 9/9; Python compilation and `git diff
--check` pass. The page is still offline and all five tests protecting private
fields and non-authorizing state remain intact. The visible action text does
not independently prove that files reached disk; the reviewer must verify the
downloads before closing the page.

An independent gate audit also confirmed a separate unresolved integration
issue: the clock converter/binder output lacks the planner's caller-supplied
clock attestation and frame-correspondence evidence digests, while the current
capture format explicitly says decoder-frame identity is unverified. Therefore
no browser payload can be treated as calibrated until an independent,
evidence-reading correspondence audit exists; hand-editing hashes is not a
valid repair. This repair remains ahead of model scoring, alongside human
review and production preflight.

**Faithfulness progress:** `未実施`, 0/3 human-reviewed and 0/3 scored; no
generated-claim effect estimate or uncertainty interval. The change reduces
handoff friction only. No GPU job was justified or run: current v2 still
requires the entire three-record, nine-forward cohort atomically, and human
review/clock/source gates remain unsatisfied. Next: audit the UI changes,
render a new private review package from the same local pinned inputs, then
continue closing the clock evidence/planner schema and production preflight
without weakening the v2 gates. Rights status remains unverified; protected
inputs and outputs stay local.

## 05:23 JST — independent audit and refreshed private reviewer package

A second audit found and we fixed two reviewer-page edge cases. The export
button now locks synchronously and `checkReady()` respects that lock throughout
the asynchronous payload digest, preventing a second export while review state
is changing. If payload/index generation throws, the lock is released and the
reviewer can retry. Once generated, separate retry buttons redownload the exact
same in-memory JSON and matching integrity index, so a failed browser download
does not require repeating the review or produce a new payload digest. No
browser-local persistence was added. An independent auditor verified the lock,
exception recovery, retry identity, unchanged schema/hash/status/authorization,
and private-only handling; no material issue remains. Actual browser save
completion still cannot be established by the page, which says so explicitly.

The focused renderer tests pass 9/9. The full scripts suite passes **1137/1137**
(14.735 s); `py_compile`, generated synthetic-page `node --check` on Node
24.18.0, and `git diff --check` pass. Re-rendered the actual local package from
the same reconstructed worksheet, lineage, custody, claim manifest/sidecar and
staged videos. New private package:
`data/private_nextgqa/reviewer_package_handoff_retry_20261017_0524/`;
HTML SHA-256 `33a3bca42e23ff28ad68d18a51fbcdd88451dd7f3a276f5a82c064fadf4348f9`;
directory mode 0700 and all four files 0600. Package creation verified custody
and exact renderer bytes; the contents remain local and were not opened by a
human. This package is ready for the independent reviewer and separate
custodian, but is not a completed review.

**Faithfulness progress:** `未実施`, direct claim-linked review/scoring 0/3,
with no effect estimate, uncertainty interval, or significance test. There was
no GPU run. The direct-v2 runner is still absent, and scoring remains blocked
by actual human review, evidence-backed compositor-frame/decoder-PTS
correspondence, protocol/input freeze, and production atomic preflight. The
clock converter/planner seam still needs a typed evidence adapter: current
converter output cannot directly satisfy the planner's extra unverified
attestation/evidence-digest fields, and adding caller-supplied hashes alone
would not establish correspondence. Next work is to define and test the
actual per-anchor frame evidence record and its independent audit binding,
then connect it consistently to calibration/planning and build the atomic
preflight. No head search is warranted while these more direct gates remain.
Rights remain unverified; protected assets stay local.

The reviewer handoff usability fix and contract update were committed as
`1ef234c82e83c6a6db9feefdf3739ec6936ca3ce`; live `git ls-remote` verified
GitHub `main` at the same SHA after push. Only the renderer, its synthetic
regression test, and the public export-contract note were committed. The local
PDCA log, protected package, and unrelated user changes remain uncommitted.

## 05:24 JST — start the clock-evidence interface audit

Next branch: resolve the structural seam between calibration output and
assignment planning without fabricating or weakening evidence. The converter
and binder produce empirical residual artifacts but do not themselves provide
the planner's extra evidence-attestation fields; the capture contract also
explicitly marks decoder-frame identity unverified. Independent support and
audit roles are checking whether a concrete per-anchor browser screenshot /
decoder-frame evidence record can validate exact correspondence, what a local
auditor would need to inspect, and where cryptographic or human identity
attestation remains outside software verification. No artifact has been
accepted as calibrated, and no GPU work is authorized by this design branch.

## 05:26 JST — connect bound calibration artifacts by opaque token

Independent source audit found the assignment planner required calibration
rows keyed by `video_id`, while the bound clock-calibration producer carries
only the opaque `review_token`. The worksheet already binds each token to one
video, so this was a repairable structural join gap. The planner now accepts
legacy video-ID-only rows or token-only bound rows, requires one consistent
identity shape for the full cohort, requires exact token coverage, and maps
through the worksheet without inserting an unhashed video ID into the
calibration artifact. Rows carrying both identities must agree. The existing
media SHA comparison remains the final cross-row guard, while artifact bytes,
content hashes, and all not-authorized states remain unchanged.

New synthetic tests cover successful token-only joining without scoring,
identity-free blocking, partially mixed identity shape blocking, and swapped
but fully-covered tokens failing at the media SHA gate. Independent audit
confirmed the hash-bound artifact comparisons remain intact and no authorization
or scoring path was added. Focused assignment tests pass 30/30. The complete
scripts suite is being re-run after the final assertions; no GPU run occurred.
This closes one data-flow defect only: it does not generate or validate the
planner's separate clock-evidence attestation/digests, prove screenshot to
decoder-frame correspondence, or make an empirical residual a calibration
pass. Direct faithfulness remains 0/3 reviewed and 0/3 scored.

The token-join change is committed and live-verified on GitHub `main` at
`1162fdc518b26f423d28428157fb73721d25b410`. The exact final rerun passed
**1141/1141** repository script tests in 14.337 s; targeted planner tests were
30/30. `py_compile` and `git diff --check` passed. The independent audit found
and closed a mixed identity-shape edge and confirmed a fully covered swapped
review-token mapping is rejected specifically by the calibration-to-media
hash gate. Only planner source/tests and the aggregate overview were committed;
private local logs and protected artifacts remain out of GitHub.

## 05:27 JST — define the evidence-reading frame correspondence gate

Independent design and source audits agree that an rVFC timestamp plus decoder
PTS fit does not establish the actual displayed image equals a frame decoded
from the staged source. The new evidence route must bind each predeclared anchor
to (a) exact capture-event and staged-media hashes, (b) an OS/compositor
screenshot crop of the player viewport, (c) decoder candidate frame index/PTS
and RGB bytes from those exact media bytes, and (d) a frozen, ambiguity-rejecting
comparison reviewed by a separate evidence reader. A verified hash list without
inspection or signer provenance is insufficient. Canvas-only capture is also
insufficient because it may not capture compositor output. Evidence records and
all images remain private with mode 0600.

The current capture page does not emit screenshot crops or geometry, the repo
has no comparator/auditor for these artifacts, and local WSL still has no
working reviewer browser. Therefore this specification work cannot create a
real clock pass, and the scoring branch remains gated. Next implementation
should start with a strict, non-authorizing audit-input schema and tests that
prove exact byte/hash joins and fail on ambiguous correspondence; visual
inspection/signature and real browser evidence must remain explicit external
requirements. Do not accept caller-entered evidence hashes as validation.

## 05:31–05:34 JST — build and verify decoder-PTS clock anchor candidate

An independent audit of the new fixed-fraction clock-anchor builder found four
fixable issues. I narrowed the downstream `predeclared_frozen` label to mean
only outcome-blind anchor-schedule predeclaration, documented that it does not
freeze the full v2 scoring protocol, and made the hash and decode consume the
same in-memory media bytes. Zero time-base denominators now fail as the
sanitized `anchor_time_base_invalid`; stdout no longer emits a private digest.
The independent auditor rechecked the current source and confirmed those fixes.

Focused tests pass 6/6 and the full repository script suite passes 1147/1147
(14.824 s). The local private builder validated current worksheet, lineage,
custody, and media byte bindings and generated a 0600 candidate with three
records/fifteen anchors; the artifact is
`data/private_nextgqa/clock_anchor_plan_candidate_20261017_0533.json`. Its
authorization remains `not_granted`. No anchor points, tokens, media hashes,
or claims are copied to this log or public Git. This tests private artifact
construction only: no browser capture, human evidence audit, model forward,
GPU work, or faithfulness score occurred. The script suite prints expected
failure-path diagnostics from negative tests while reporting all tests passed.

The public builder experiment note records code/tests and only aggregate
protected-run counts. A provenance manifest is still missing; before capture,
we must bind the exact plan and inputs to builder/runtime versions and the
private joint-freeze receipt. Next task is to implement that manifest or a
stricter evidence-reader gate, then establish whether a browser capture can run
in the current Windows/WSL setup. Direct generated-claim faithfulness remains
0/3 human-reviewed and 0/3 scored. No reason exists to reopen the previously
closed head search yet.

## 05:34–05:43 JST — bind candidate plan to private generation provenance

Independent design/audit review confirmed that the HTML5 renderer records only
self-reported `currentTime`/rVFC data and no OS/compositor screenshot or video
viewport geometry. The converter labels decoder identity unverified, and the
planner accepts caller-supplied evidence hashes structurally without reading
evidence. Reviewers independently found no browser executable or automation
package available in this WSL runtime. The minimum real correspondence test
requires an actual review browser plus OS-level screenshot capture; headless
DOM screenshots or synthetic tests cannot pass that gate. A single live anchor
would only smoke the capture method; the full clock plan needs all 15 anchors.

The first manifest run failed safely before output creation because PyAV's
stream time-base value is a `Fraction`, which JSON does not serialize. The
builder now canonicalizes that exact field as a string. Its rerun validated all
three local staged videos and created paired 0600 plan/provenance artifacts:
`clock_anchor_plan_candidate_20261017_0542.json` and
`clock_anchor_plan_provenance_20261017_0542.json`. Independent local checks
verified three records, fifteen anchors, three PTS provenance records, mode
0600, and authorization `not_granted`. The sidecar binds worksheet, lineage,
custody, builder source, exact plan, ordered decoder PTS digests, stream/pixel
metadata, PyAV/FFmpeg library versions, and leaves the joint-freeze receipt
null. No protected digest, token, timestamp, claim, or media was published.

This closes plan-generation provenance only. It does not establish browser
capture, screenshot-to-decoder-frame identity, human review, clock pass, or
faithfulness. Direct claim faithfulness remains 0/3 human-reviewed and 0/3
scored. Current WSL cannot perform the required browser capture; continue with
a useful evidence-reader implementation/specification and look for an approved
actual browser route. Keep head search closed while these direct gates remain
actionable. Rights remain unverified and all protected artifacts remain local.

## 05:36–05:38 JST — close manifest audit, verify browser route, rerun suite

Latest independent audit checked the private plan/manifest pair without
printing private values. Plan SHA matches the sidecar; source hashes for the
builder, decoder, custody auditor, and output policy match current files; all
three exact input byte hashes, three token/media rows, and fifteen anchor
assignments align. Both files are 0600 under a 0700 current-user private
folder. The manifest remains `candidate_not_jointly_frozen`, has a null joint
freeze receipt, and grants no execution. Output targets are restricted to that
private directory. The paired-write code uses fsynced 0600 temporary files,
no-replace hard links, and rollback for ordinary write errors; a process/power
failure between file installs can still leave an orphan, so consumers must
require both files and verify the plan hash. The audit found no public leakage.

One earlier run used a manually future-dated filename; I renamed that same
private byte pair to the actual 05:37 JST stamp. Contents and digests were not
changed. The most recent generated pair is
`data/private_nextgqa/clock_anchor_plan_candidate_20261017_0537.json` and
`data/private_nextgqa/clock_anchor_plan_provenance_20261017_0537.json`.

Windows Edge 155 is installed. Ordinary sandbox WSL interop fails, but a
per-command elevated read-only query can access the WSL UNC path. I attempted
to open the existing local review page in Edge; process inspection did not
confirm a page-specific process, so I do not count the page as opened. This
only establishes that an elevated Windows browser route exists, not actual
playback or screenshot evidence. The full script suite passes 1149/1149 in
14.595 s after the latest source changes; the eight focused tests also pass.

Direct claim faithfulness remains 0/3 human-reviewed and 0/3 scored. The actual
RTX 5070 Ti and CUDA-enabled venv are available under individual elevation, but
the protocol gates still block the nine-forward run. Next: make the Windows
review-page launch reliable without exposing or substituting for independent
human review, then build a strict evidence-reader/audit path. Do not reopen
head search while these actionable direct-faithfulness gates remain.

## 05:38–05:41 JST — final provenance hash test and actual browser handoff

The provenance helper now accepts raw dependency source bytes and computes
those SHA-256 values itself; its test verifies computed digests, exact worksheet,
lineage, custody, and serialized-plan hashes, candidate status, null joint
freeze receipt, and not-granted authorization. This replaces the earlier test
that only passed precomputed dependency digests. Focused tests pass 9/9. The
latest private plan/manifest was regenerated from the latest builder source as
`clock_anchor_plan_candidate_20261017_0541.json` and
`clock_anchor_plan_provenance_20261017_0541.json`; local checks verified the
current builder and dependency code hashes, plan hash, 3 records/15 anchors,
0600 modes, candidate status, and non-authorization. No private values were
printed or published.

Windows Edge 155 and the exact local review page are readable through elevated
WSL interop. A fresh Edge profile process was launched with the file URL and
its process arguments confirmed the handoff page path. This opens the local
review interface for a human; no human response or review record has been
created, and no browser playback or compositor screenshot was captured. It is
not a clock pass. A previous launch attempt was not confirmed; the dedicated
fresh-profile launch resolved that process-detection issue.

The latest full script-suite run is in progress. Direct claim faithfulness
remains 0/3 human-reviewed, 0/3 scored; the nine-forward GPU experiment is
blocked by joint protocol freeze, human review, decoder correspondence, and
production atomic preflight. The GPU itself is available under individual
execution, as confirmed separately. Continue with evidence-reader engineering
and browser capture viability; do not run a proxy experiment or resume head
search while these prerequisites remain actionable.

Final verification at 05:41 JST: 1150/1150 repository script tests passed in
14.629 seconds; focused clock-builder tests 9/9; `py_compile` and `git diff
--check` passed. The suite's expected negative-path diagnostics were emitted by
failure-case tests; overall test status is OK.

## 05:41–05:44 JST — actual Edge control-route smoke

Windows Edge 155 loaded the local review page sufficiently to expose a local
DevTools target. I tried attaching the exact staged video files to the three
review file inputs through CDP, but the WebSocket control connection failed
before a media-readiness probe. I closed the temporary Edge debug process and
verified its local debug endpoint was closed. I do not count this as playback,
clock capture, screenshot correspondence, or reviewer action. The separate
non-debug Edge review page remains available, but a dedicated window/tab display
was not confirmed. This reveals the next technical blocker: DevTools control
from the elevated WSL-to-Windows interop route is unreliable. Avoid further
CDP attempts unless a more reliable documented control channel is found. The
local plan and provenance pair remain valid; direct faithfulness remains
0/3 reviewed and 0/3 scored.

Additional read-only check: Windows reports a visible Edge window with the
page's generic title `Private offline video review`. This confirms the review
page is actually open in a window, while still not establishing video playback,
human review, or screenshot correspondence.

## Follow-up — close converter provenance bypass

The independent pipeline audit found that the CLI required the plan provenance
sidecar but the Python converter API allowed both provenance arguments to be
omitted, and the binder still accepted the legacy bundle shape. Both paths are
now closed: the Python API requires sidecar bytes plus the expected SHA-256;
the binder requires that digest and copies it to the calibration record. Added
tests for omitted API arguments, legacy bundle rejection, digest mismatch, and
replay against a changed plan. Focused tests pass 21/21, full scripts suite
1155/1155, `py_compile` passes, and `git diff --check` passes. This establishes
only structural byte continuity; sidecar declarations are not independently
authenticated, and no browser screenshot/frame evidence, human review, model
forward, or faithfulness score was produced. Direct claim-faithfulness remains
0/3 reviewed and 0/3 scored.

## Follow-up — candidate signed clock-frame evidence reader

Implemented a CPU-only reader for the three-video/fifteen-anchor candidate.
It reuses the strict clock observation converter, verifies signed per-anchor
bindings across the plan, raw capture event, screenshot and viewport crop,
exact staged media bytes, and independently decoded display-order frame/PTS/RGB
digest. It rejects incomplete/duplicate anchors, mismatched/tampered artifacts,
non-`match` decisions, and screenshot/event timestamps more than a candidate
250 ms apart. The reviewer signature now includes a pseudonym, independent
reader role, independence attestation, explicit visual-comparison attestation,
and creation time. Focused synthetic tests pass 8/8; full repository suite
passes 1163/1163 in 15.807 s; `py_compile` passes.

Independent review found the intended trust boundary: signature/key identity
must be frozen out of band, both timestamps and viewport geometry remain
self-reported, and the reader does not compare screenshot pixels to the decoder
frame. A signed false `match` can pass the byte-consistency audit; this is
tested explicitly and documented. Output remains
`candidate_signed_evidence_bytes_consistent`, `joint_freeze_status` remains
`not_jointly_frozen`, and authorization remains `not_granted`. No real
screenshot/browser capture, human review, GPU/model forward, or faithfulness
score was produced. Direct generated-claim faithfulness remains 0/3 reviewed
and 0/3 scored.

## Follow-up — WSL permission smoke and protocol-pin regression repair

The Windows helper's non-capture `PermissionSmokeOnly` branch passed through
the Windows-to-WSL bridge: it created a unique empty probe, set and verified
Linux directory mode 0700 and file mode 0600, then removed the probe. The
`ValidateOnly` branch also passed against the current candidate plan and
provenance sidecar, returning before any output directory or capture. Neither
check captured a screen or video, and neither establishes browser-to-decoder
frame correspondence. A read-only GPU check found RTX 5070 Ti, 14,974 MiB free
of 16,303 MiB, 1% GPU utilization, and the `.venv` has PyTorch 2.11.0+cu128
with CUDA 12.8. The device is available; the direct scoring gate remains closed.

The first full test rerun had 11 failures among 1167 tests, all from the
synthetic v2 preflight's stale protocol SHA pin. Updating the protocol's
reference to review policy v6 changed its byte digest, while the synthetic
fixture continued to pin the previous digest and consequently rejected valid
fixtures at `protocol_v2_identity_mismatch`. Updated that pin to the current
protocol SHA-256 `db71bc5017d0f85aeb58506480edf97f2cbf8ba5d9d56b5ea22fc2fb8f811973`.
Added an exact test for the v6 review JSON top-level export fields. The focused
renderer/preflight tests pass 19/19 and the complete scripts suite passes
1167/1167 in 17.888 seconds. The failure text printed during the suite is
expected from negative-path tests; final status was `OK`.

An independent audit checked the 0635 v6 package's policy digest across
worksheet, sidecar, custody record, and HTML; it found the artifacts aligned
and the package modes private (directories 0700, files 0600). Its initial
reported omission of `export_disclosure_acknowledged` was withdrawn after
rechecking the same policy bytes; the field is present in the policy and exact
field assertion now guards it. No human review, screenshot capture, decoder
alignment, model forward, or score occurred. Direct faithfulness remains
**0/3 human-reviewed, 0/3 scored**, with no effect estimate, uncertainty
interval, or test. The nine-forward GPU experiment remains blocked by actual
independent human review, 15 frame correspondences, and a real-data atomic
production gate. Rights status remains unverified; protected material stays
local.

## 06:27–06:31 JST — independently audit the remaining production gate

**Purpose.** Check whether existing validators can be safely assembled into a
real-data runner now, rather than assuming that the synthetic runner is close
to production readiness.

**Observed source state.** `run_nextgqa_claim_mask_feasibility_v2.py` is
explicitly a synthetic-only skeleton and its module entry point exits without
a production CLI. The review auditor returns
`unvalidated_handoff_structurally_valid` with
`semantic_review_authenticated=false`; the clock/frame reader verifies
signatures against a caller-selected key but reports the key trust as
`expected_hash_supplied_by_caller_not_joint_freeze_authenticated`, leaves
`joint_freeze_status=not_jointly_frozen`, and returns
`execution_authorization=not_granted`. The target and actual-QA validators also
remain unvalidated. Thus a wrapper that simply combines their “pass” fields
would turn declarations into authorization and is not a sound implementation.

**Checks.** Independent design and implementation audits reached the same
conclusion without opening protected video/QA contents or editing files. The
current gate/scorer/review/clock regression set passed **62/62**; `compileall`
and `git diff --check` passed. This verifies existing fail-closed behavior,
not production readiness. No model loaded and no forward, screenshot capture,
human review, or faithfulness score occurred.

**Faithfulness progress.** Direct claim-linked evidence remains **0/3
human-reviewed and 0/3 scored**, with no estimate, interval, or test. The GPU
is available but was not used for inference because no valid real-data atomic
preflight exists. Rights status remains unverified; local-only operations are
permitted, and rights are not the current blocker.

**Next action.** The shortest remaining path still needs actual independent
review exports and all 15 screenshot-to-decoder correspondences, plus a
pre-outcome trusted key/joint-freeze receipt that pins the reviewer/auditor
trust roots, exact protocol/cohort/target/sampler/scorer, and validators. Until
those evidence inputs exist, keep model loads and score calls at zero. The
user has been asked whether the three-video review can happen now; continue
independent preparation while awaiting that answer. Do not resume head search
while the direct-evidence branch has these actionable gates.

## 06:36–06:38 JST — independently reproduce closed head-confirmation provenance

**Purpose.** Verify the historical quantitative result independently while the direct claim-review inputs remain outstanding; this is a result audit, not a reopening of head search.

**Observed artifacts and checks.** The committed result JSON SHA-256 is `993acd16e62c1e8cf8aafe075674cc1e649761f2e1ca5bb05be83814a2e6ac36`; its run-manifest SHA-256 is `d3c59d49c3609a03d868592d188507f3df5fc1cd1408612bfb98c315d63e1b16`. The analyzer source is byte-identical to its checked-in version at `6ab0e09` (SHA-256 `03fb95d6b089b956117e309816606982e330eed64e13f1511ce34a20c508e29e`). Re-running it reproduced 79/79 records, 0 failures, selective effect `−0.075633 nat`, SD `0.366724`, 95% t CI `[−0.157775,+0.006509]`, p=`0.0706` vs zero and p=`1.5211e−11` vs the predeclared `+0.25 nat` boundary; success remains false.

**Provenance gap.** The result itself and current analyzer reproduce, but exact raw input-manifest identity does not: the run manifest pins sample SHA-256 `2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`, which does not match the available `data/nextgqa_samples_confirm_heldout.json` (`7bf254…`) or smoke manifest (`7ada7c…`). A local metadata-only scan of JSON files under the repository and `/tmp` found no raw file with the pinned digest. A separately tracked, hash-linked identity audit and its reconciliation artifact do report the same pinned sample hash and 79 unique normalized IDs, paths, and video contents, with zero exact duplicates; the current reconciliation explicitly sets `result_run_manifest_crosshash_binding=false` and `independence_established=false`. Thus the cohort identity summary is supported by recorded audit artifacts, but the raw input cannot be rejoined locally and strict end-to-end SSS provenance remains qualified rather than fully closed.

**Checks.** The identity-reconciliation and head-analyzer test suites passed 24/24. One initial unittest invocation omitted `PYTHONPATH=scripts` and failed module import; the corrected invocation passed. `git diff --check` passed for this local log update.

**Faithfulness status.** Historical head evidence is still 79/79, with no support for the preregistered practical threshold; this is scoped to that intervention/model/dataset/score and does not establish general explanation faithfulness. The direct claim-linked real-video branch remains 0/3 human-reviewed and 0/3 scored, with no estimate, CI, or test. No new GPU inference ran: the unresolved review, capture geometry, and joint trust/preflight gates remain material. Next action is to obtain a valid independent review/capture package and exact pinned source inputs; do not resume head/layer/alpha tuning.


## 06:39–06:47 JST — GPU boundary and synthetic browser-frame capture feasibility

**Purpose.** Recheck the live WSL GPU before any model run and test whether the click-bound browser frame can yield a clean, content-only image for a later independent decoder correspondence review.

**Environment.** Normal sandbox inspection reported `nvidia-smi` blocked by the OS and `.venv` `torch.cuda.is_available()=False`; this matches the known sandbox visibility boundary. The individually elevated read-only check succeeded: RTX 5070 Ti, 14,968/16,303 MiB free, 1% utilization; `.venv` PyTorch 2.11.0+cu128 / CUDA 12.8 reports CUDA available and names the same GPU. WSL reports 15 GiB total RAM, approximately 13 GiB available. No model loaded and no CUDA scoring ran.

**Synthetic browser test.** On Windows Edge 155.0.4283.45, a locally generated 160×90, 2-frame H.264 `testsrc` clip was wrapped in a same-page `File`/Blob URL. In Edge new-headless mode, `requestVideoFrameCallback` followed by synchronous `drawImage(video, ...)` and PNG export succeeded (`image/png`, 160×90, 12,726 bytes, callback mediaTime 0). The exported PNG was compared locally to PyAV RGB decodes of that same synthetic clip: against frame 0, mean absolute channel error 5.0597, exact-pixel fraction 0.2637, maximum channel error 139; against frame 1, mean error 10.0013, exact fraction 0.2607, maximum 160. This confirms technical export feasibility but rejects exact RGB-hash equivalence across browser/PyAV conversion. The sample is synthetic, one codec/size/browser, and says nothing about the protected dataset.

A separate Chrome DevTools Protocol screenshot prototype did not yet isolate the video element: its clip included adjacent page text, and a subsequent viewport-only readiness harness hung. The confirmed synthetic Edge process (PID 7772) was stopped and verified by its launch time; no existing user Edge process was touched. This approach is not accepted as a capture path.

**Interpretation and next action.** The browser canvas route could remove arbitrary screen-crop selection for a *revised* estimand—visual comparison of a browser media-element frame associated with rVFC metadata against an independently decoded candidate. It does not prove physical monitor/compositor display, and its pixel conversion differs from PyAV. The existing candidate protocol still requires OS screenshot evidence, so this result alone must not substitute for it. Shortest useful follow-up is to version and audit an offline, blinded 15-pair viewer that presents a custodian-selected video-only crop plus the candidate decoded frame; the reviewer must attest to crop purity and choose match/ambiguous/mismatch. Any protocol switch to browser-element frames requires a new pre-outcome capture policy and independent freeze. No protected video or question content was opened, no actual frame was captured, and no model/faithfulness score was produced.

**Faithfulness progress.** Direct claim-linked real-video faithfulness remains 0/3 human-reviewed and 0/3 scored, with no effect estimate, interval, or test. GPU availability is now reverified; the nine-forward run is still blocked on review, frame evidence, and authenticated atomic preflight. Rights status is unverified; all protected files remain local.

## 06:49–06:50 JST — bounded real-video Qwen CUDA path smoke

**Purpose and expected workload.** Use GPU time for the smallest useful real-data demonstration while the score-bearing branch remains gated: one locally held-out NExT-GQA test video, four PTS-derived frames, one pinned Qwen2.5-VL-7B 4-bit BF16 CUDA forward. No QA text, answer, interval, generated text, or score is used. Expected duration was about 30 seconds after model/cache readiness; only aggregate diagnostics are written to a mode-0600 local result.

**Verified result.** The runner inspected three deterministic unused valid test-video candidates for its technical sampling constraint; two were rejected for timing, then one video passed. It decoded 1,402 frames and sampled 4/4; Qwen preprocessing and the single CUDA forward completed with finite logits. The runner-measured interval (candidate decode through forward, excluding candidate discovery/import startup) was 25.9 s and forward time 3.09 s. Hardware was RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 5.18.0, bitsandbytes 0.50.2, qwen-vl-utils 0.0.14. Peak PyTorch allocated VRAM was 6.04 GiB, peak reserved 6.29 GiB, and CUDA driver free memory after run was 8.30 GiB. The privacy-minimized output is `results/nextgqa_real_video_gpu_smoke_20261017_0710.json`; it contains no video identity/path, raw media, QA, or generated text. No attempt was made to publish it; its sampled-frame digests remain local.

**Interpretation and limits.** This is a successful real-video-to-Qwen CUDA pipeline demonstration and provides an observed cost estimate for one forward. It is not a faithfulness experiment: no answer or explanation was scored, no annotation interval was used, and no effect estimate or uncertainty interval exists. The score-bearing claim-mask run remains blocked on independent 3-video human claim review, 15 screenshot-to-decoder correspondences, trusted pre-outcome freeze, and production preflight. Rights status remains unverified; all data and derived digests remain local. The direct faithfulness status is still 0/3 human-reviewed and 0/3 scored. The next action is to complete a strictly non-authorizing, blinded visual correspondence handoff and independently verify its limits; do not resume head search while this actionable branch remains.

## 06:53–06:55 JST — rebind decoder-anchor plan to the current v6 cohort

**Purpose.** Before preparing any reviewer frame pairs, verify that the existing five-anchor plan actually belongs to the current 06:35 v6 review cohort; avoid joining stale decoder frames to newer claims/videos.

**Observed result.** A hash-only cross-check of the earlier candidate plan against the current v6 staged media matched **0/3** source videos, so that plan is excluded from the current handoff. Rebuilt a new outcome-blind candidate plan with `build_nextgqa_clock_anchor_plan.py` from the current v6 worksheet, lineage manifest, custody manifest, and staged media. The builder verified its source joins and returned three records with `execution_authorization=not_granted`. An independent local integrity check confirmed **3/3** exact video SHA matches, **15** planned anchors, all five distinct prespecified anchor IDs per video, and mode `0600` for plan and provenance files. The plan and sidecar are local private artifacts under `data/private_nextgqa/clock_anchor_plan_candidate_v6_20261017_0654.json` and the corresponding provenance filename.

**Interpretation and limits.** This repairs the stale-cohort hazard and establishes a current-cohort anchor candidate for later frame preparation. It does not jointly freeze the scoring protocol, prove screenshot correspondence, constitute human review, calibrate clocks, or authorize scoring. No screenshots or decoder-frame PNGs were generated in this cycle. Next action is to finish independent review of the synthetic-only blinded-pair builder, then use the exact plan bytes to prepare and audit decoder candidates without opening them as a scoring result.

## 06:55–06:57 JST — repair and independently audit blinded-pair handoff

**Implementation.** The first synthetic-only prototype passed six basic tests, but independent audit found it could accept a changed reviewer page, accept any 15-item anchor label grid, and did not bind the input rows to actual candidate-plan bytes. The builder was tightened: it reads the private plan with no-follow/mode checks, verifies the exact plan SHA-256 and candidate/no-authorization status, derives the 3×5 token/anchor assignments from the plan, and requires exact equality with the custodian manifest. It binds a random per-package nonce, each embedded normalized PNG hash, and a normalized exact HTML-template hash; `audit-export` reads and verifies the actual reviewer HTML before evaluating exports. The reviewer export now carries pseudonym, independent-role and independence self-attestation, UTC time, nonce and package-manifest hash, while explicitly declaring unauthenticated provenance and no signature.

**Verification.** Primary rerun and independent audit both pass **10/10** synthetic tests, including wrong-plan hash, swapped token/anchor assignment, incomplete/duplicate rows, package/page/image tampering, impure/ambiguous/mismatch blocks, symlinks, modes, and invalid image dimensions. `py_compile` and `git diff --check` pass. No real screenshots/PNGs were read by this builder, and no GPU was used.

**Limits and next action.** This is still a presenter and internal-consistency auditor, not an authenticated human gate: the pseudonym/time/independence are self-reported, nonce reuse for the same package is not prevented, and the operator-supplied plan hash needs an upstream frozen trust root. It cannot authenticate crop purity or the screenshot/media/frame/PTS correspondence. The reviewer may infer image source from content/dimensions; color-profile conversion and browser scaling can alter visual appearance. Keep any resulting status non-authorizing. Next action is to export decoder PNGs from the now-current v6 plan in a single pass per video, verify the PTS/source joins, and then wait for genuinely captured video-only screenshots before assembling a review package. Direct faithfulness remains 0/3 human-reviewed and 0/3 scored, with no effect estimate, interval, or test.


## 06:58–07:03 JST — export and verify current candidate decoder anchors

**Purpose and boundary.** After an independent metadata-only audit matched the current candidate plan/provenance to PyAV, FFmpeg libraries, all pinned source hashes, and the staged-media cohort, export the prespecified 15 decoder-side frames needed for the correspondence handoff. The source was the current staged cohort; this did not capture screenshots or authorize any score.

**Verified result.** `scripts/export_clock_anchor_decoder_pngs.py` completed with status `candidate_artifact_export_only`, `execution_authorization=not_granted`, and 15 records. An independent local integrity pass found exactly 15 valid RGB PNGs, all mode `0600` under a mode `0700` directory, plus a mode `0600` crosswalk. PNG byte total was 4,067,917. The crosswalk declares 15 effective of 15 requested anchors. Synthetic exporter tests pass 5/5; blinded reviewer-builder tests pass 10/10; compilation and whitespace checks pass.

**Interpretation and limit.** This is real-media decoder preparation, not screenshot evidence, human review, calibration, or claim-faithfulness measurement. Direct claim-linked progress remains 0/3 human-reviewed and 0/3 scored; no faithfulness effect/CI/test. The next useful path is to establish a real OS/compositor capture and crop-to-decoder correspondence for all 15 anchors; the Windows helper requires an interactive private Edge session and human hotkey at each planned event. Existing evidence validation checks a declared rectangle and recomputed crop hash, but does not prove that the rectangle excludes browser controls or surrounding page content.


## 07:03–07:07 JST — repair capture-row order and validate Windows/WSL setup

**Observed issue.** An independent source audit found the Windows helper enumerated grouped fit anchors before held-out anchors, yielding per-video 5%, 50%, 95%, 25%, 75% instead of the planned chronological 5%, 25%, 50%, 75%, 95%. The prompt also did not identify the current assigned token/anchor. This could misdirect capture work.

**Change and checks.** The helper now constructs each row in explicit chronological anchor-ID order, verifies exact fit and held-out membership separately, rejects duplicate IDs and repeated review tokens, and displays token, anchor ID, and planned decoder PTS on each prompt. The doc now states the limit: these prompts reduce row confusion but do not verify browser state. Independent source audit found no remaining ordering/misbinding issue within scope. Synthetic source-contract suite passes **4/4**; evidence-reader tests pass **12/12**; clock-input conversion tests **21/21**; blind-pair tests **10/10**; exporter tests **5/5**; scoped `git diff --check` passes. The source-contract suite models mapping in Python and checks source markers; it is not a PowerShell runtime test. The modified script itself passed the Windows PowerShell language parser.

**Windows/WSL check.** Normal sandbox invocation of Windows interop failed with `UtilBindVsockAnyPort`; a one-command elevated read-only query succeeded. The candidate plan/provenance passed the helper's Windows PowerShell `ValidateOnly` branch. `PermissionSmokeOnly` also verified Windows-to-WSL mode 0700 for the disposable directory and 0600 for the disposable probe file, and removed the probe. No videos, screenshots, browser, or human review were accessed in these Windows checks; no actual capture occurred.

**Limits and next action.** This removes a software ordering hazard and verifies a necessary capture prerequisite. The current OS helper still needs a human in the Edge review page to position the player and press the hotkey 15 times; the current sandbox cannot do that interaction on its own. It captures a full foreground window, after which a video-only crop must be separately produced and reviewed. Direct claim-linked faithfulness remains 0/3 human-reviewed and 0/3 scored; no effect estimate, interval, or inferential test.


## 07:08–07:09 JST — verify reviewer handoff is the same current anchor cohort

**Purpose.** Confirm the already-rendered current-v6 human-review page, packaged copy, staged media tokens, and refreshed 0654 clock plan refer to the same three records before further capture preparation. This check read local HTML/JSON bytes for token fields and hashes only; it did not display or print questions, claims, video content, or token values.

**Verified result.** The plan has 3 unique tokens; staged media has 3 filenames and its token set exactly matches the plan. The private HTML contains exactly 3 explicit `review_token` fields, all unique and exactly equal to the plan token set. Its hash equals the packaged `index.html` hash. Both HTML copies are mode 0600.

**Limit.** This verifies cohort binding and private handoff packaging only. It does not show that a person opened the page, reviewed any claim, or recorded a correct interval. Human review remains 0/3.


## 07:09–07:11 JST — prepare and harden receipt-bound screenshot cropper

**Purpose.** The OS helper captures a complete Edge window, but the existing blind frame-pair reviewer requires a video-area crop. Implement a local-only transform so future real captures can be cropped reproducibly without treating a supplied rectangle as proof of crop purity.

**Implementation and audit.** `scripts/crop_nextgqa_clock_screenshots.py` accepts a strict crop map for exactly the 15 current plan anchors, hash-bound to exact candidate plan, provenance, and capture receipt bytes. It checks private modes/owners, receipt screenshot byte hashes, full plan/provenance/runtime/source/media provenance (PyAV, FFmpeg, Python), exact token/anchor coverage, integer bounds, and cleans up only its own partially created directory. It exports RGB PNGs and a private crosswalk with source/crop hashes, keeping `execution_authorization=not_granted`; it explicitly records video-only purity as unestablished and requiring human review. Independent review found initial flaws with JSON identity types, internal provenance validation, and race cleanup; those were repaired and retested.

**Verified checks.** Final synthetic suite **9/9** passed, including exact 15 filenames/crosswalk joins, provenance and runtime drift, malformed identity values, screenshot tampering, bbox bounds, coverage, permissions, and simulated output-directory collision/race. `py_compile` and `git diff --check` pass. Independent auditor confirmed all identified issues resolved and no false video-purity claim. No real screenshot was processed and no actual crop was generated.

**Limit and next action.** The crop map still needs a human to specify the video rectangle and visually attest each crop; coordinates/hash/bounds alone cannot prove the rectangle excludes controls/page content. The actual OS screenshot step still requires interactive Edge playback. Direct claim-faithfulness remains 0/3 reviewed and 0/3 scored, with no effect/CI/test.


## 07:11–07:16 JST — measure real-video 32-frame CUDA resource envelope

**Purpose and expected workload.** Estimate whether the pinned Qwen2.5-VL-7B local-video path can process the intended 32-frame temporal input on the available RTX 5070 Ti. This was a single-video, one-forward engineering smoke, with no QA, answer, interval, generated text, or faithfulness score.

**Verified result.** Before run, the GPU showed 14,863 MiB free; the repository `.venv` had Torch 2.11.0+cu128 with CUDA enabled. The runner checked seven deterministic candidates under the fixed timing criterion; six were rejected, one selected video decoded to 32/32 frames. The Qwen input had 4,064 tokens and grid `[16, 28, 36]`; the one CUDA forward returned finite logits. The runner-measured decode-through-forward segment was 34.5865 s, including a 4.2495 s forward. Peak PyTorch allocated VRAM was 10.4977 GiB; CUDA driver free memory after forward was 2.2232 GiB of 15.89 GiB. After process exit/model release, a separate check saw 14,846 MiB free and 1% utilization. The mode-0600, 2,548-byte privacy-minimized output is `results/nextgqa_real_video_gpu_smoke_32f_20261017_01.json`.

**Interpretation and limits.** This confirms a 32-frame real-video CUDA path for one selected video and gives a resource datapoint. Peak memory is close enough to device capacity that each exact claim video and each condition batch must be preflighted; this example does not guarantee the cohort will fit. It is not a faithfulness result and has no effect estimate, interval, or test. Direct claim-faithfulness remains 0/3 human-reviewed and 0/3 scored. The score-bearing branch remains blocked on real OS/compositor screenshots and 15 independently reviewed screenshot-to-decoder correspondences, human claim review, trusted pre-outcome freeze, and production atomic preflight.

## 07:18–07:42 JST — full-cohort gold-interval geometry census at 16/32 frames

**Question.** Does the local NExT-GQA test-video candidate pool contain enough sampled gold-interval ranks to construct an equal-count, rank-matched temporal control under the intended 32-frame sampler? This is an outcome-blind input-geometry feasibility check, not a claim-faithfulness or answer-margin experiment.

**Exclusion audit.** The first 16-frame run used the local 118-entry `exclude_prior_all_20261016_02.json`. All 85 candidate video rows were counted as examined and all 85 were excluded before decode, leaving zero decode attempts. The denylist provenance remains unresolved, so this is not evidence that all 85 were in fact previously seen; it does show that treating this legacy set as authoritative leaves no candidates.

**Implementation repair.** Review of the empty run exposed ambiguous output names: `effective_scanned_videos=85` counted rows inspected, although no video was decoded. The runner now reports `candidate_rows_examined`, `candidates_excluded_before_decode`, and `decode_attempts` separately, and validates their partition. The focused synthetic suite passes **3/3**, including a malformed-accounting rejection; compilation and scoped `git diff --check` pass. An independent audit confirmed the partition matches `freeze_cohort` control flow when no cohort lock is supplied, as in this census.

**Verified local results.** To assess geometry alone, reran with the explicit empty exclusion manifest, retaining the same deterministic 85-row candidate pool. At 16 frames, all **85/85** candidates passed timing and decoded; all 85 had at least one tested QA interval yielding nonempty evidence ranks and equal-count distinct rank-matched controls. The selector tested 103 questions to choose one eligible question per video. At 32 frames, the same **85/85** candidates passed, again with no equal-count-control failures; 96 questions were tested. The cohort fingerprints are stable across the pre- and post-accounting-fix repeats for each frame count. Runtime was 326.15 s at 16 frames and 329.942 s at 32; peak RSS was 1,438.3 MiB and 2,827.8 MiB. Both final artifacts are mode 0600 and bind matching current dataset, selector, census-runner, and exclusion-manifest hashes. Independent checks confirm the aggregate counts, accounting invariants, privacy mode, and code/input hashes.

**Interpretation and limits.** This removes one geometry feasibility uncertainty for the existing 85-video local candidate pool: under this sampler, at least one QA gold interval per video can produce an equal-count control at both tested frame counts. It does not show that a generated claim is supported, that human-reviewed intervals have the same geometry, that masked model inputs differ only as intended after the Qwen processor, or that any score changes. The 85 candidate videos all appear in the provenance-unresolved 118-entry prior-exposure list, so they are not a validated fresh cohort. Rights remain unverified; all derived aggregates stay local. Direct generated-claim faithfulness remains **0/3 reviewed, 0/3 scored, no effect/CI/test**. The nearest required action for the direct branch is still independent human review of the prepared three-video handoff and genuine 15-anchor OS/browser capture plus correspondence audit. No head search is justified while those direct gates and their prepared workflow remain actionable.

## 07:41–07:47 JST — separate historical head evidence from claim-faithfulness progress

**Issue and change.** The existing progress analyzer placed the closed 79-video head intervention under `direct_faithfulness`, while its scope caveat excluded generated-claim faithfulness and it omitted the current 0/3 review/score state. Schema v2 now reports the head result separately as `historical_head_confirmation` and generated-claim status separately, with declared counts explicitly unverified, no score artifact, and null effect/interval/p-value. It never infers claim evidence from the head result. The head input is pinned to the verified artifact SHA-256, 79/79/0 accounting, exact +0.25 nat threshold, and the observed t-CI upper bound below that threshold; otherwise analysis fails closed. CLI requires the current operator-declared claim review/scored counts.

**Verification.** Focused tests pass **6/6**, including declared 3/3 counts still producing no verified claim score or estimate, impossible count combinations, substituted head-artifact rejection, and the expected current 0/3 snapshot. CLI produced a mode-0600 temporary snapshot with schema v2, historical head result −0.07563 nat (95% t CI [−0.15777, +0.00651]), and generated-claim status `not_evaluated_no_verified_direct_score_artifact`; the temporary was removed. Compilation and scoped `git diff --check` pass. An independent audit found no remaining material contradiction. Old schema-v1 result JSONs and dated 2026-10-12 prose remain historical; they were not rewritten.

**Limits and next action.** This repairs progress reporting only and supplies no new faithfulness evidence. The latest verified status is still generated-claim review 0/3 and score 0/3, with no direct estimate, uncertainty interval, or test. The human review and browser/compositor evidence gates remain outstanding; the analyzer's count inputs are self-reported declarations, not authentication. Next work remains the actual independent review and 15-anchor browser-to-decoder capture/audit, followed by protocol freeze and a production atomic preflight. No head search is warranted.

## 07:47–07:55 JST — exact three-video 32-frame CUDA capacity check

**Question and scope.** Can each of the three caller-pinned, currently staged review videos complete one sequential, unmasked, 32-frame Qwen2.5-VL-7B forward on the RTX 5070 Ti? This is an engineering capacity diagnostic. It does not use questions, answers, generated claims, human labels, or intervals and does not calculate faithfulness.

**Implementation and checks.** Added `scripts/run_nextgqa_review_cohort_gpu_capacity_smoke.py` and seven synthetic validation tests. Before model loading, the runner verifies pinned manifest bytes and permissions, the plan-to-provenance hash link, exact 3-row token sets, unique media identities, staged media SHA-256 and size, then snapshots the validated bytes into private temporary files so the decoder consumes the same bytes. It rejects symlinks; failure logs contain fixed codes only; output requires a private directory and contains opaque ordinals. An independent code audit prompted fixes for hash/open races, duplicate-video acceptance, and failure-detail leakage. The runner also records SHA-256 and byte size for all 16 locally cached checkpoint/processor assets (16,595,981,281 total bytes); these local hashes do not independently authenticate the official upstream checkpoint. The initial 1 ms run rejected the first video's PTS grid before any forward. The existing full-cohort engineering audit had already used 50 ms as an exploratory decoder tolerance; a CPU preflight then verified 32/32 frames for all three current videos at that fixed tolerance. Synthetic tests pass **7/7**, Python compilation and `git diff --check` pass.

**Verified GPU result.** Preflight found CUDA/BF16 available on the RTX 5070 Ti and 14.66 GiB free VRAM. The bounded run loaded local revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` in 4-bit with BF16 compute, completing **3/3** sequential 32-frame inputs with finite logits and **0 failures** in 27.84 seconds. Input token counts by opaque ordinal were 4,256, 4,592, and 4,064; peak PyTorch allocations were 10.94, 11.74, and 10.50 GiB; peak reserved memory reached 13.83 GiB; the minimum recorded CUDA driver free memory after a forward was 0.66 GiB. The result is mode 0600 at `data/private_nextgqa/gpu_capacity_32f_three_review_video_20261017_02.json`, SHA-256 `56408999da3570fa93b751d16907ee658b938eb977b4ec9b516cc47c0c0c8485`. The output's runner hash matches the runner bytes used for the run and its cohort identity is explicitly caller-pinned/local, not authenticated. Independently checked decoder helper SHA-256 is `bef8a096c4de423a98ae5590a55f3e564f9398ec692ebd03343e4b207fea7351`; the result JSON did not yet carry that helper hash. After the run, the runner was hardened to clean up temporary snapshots on exceptional exit and to include decoder/PyAV/FFmpeg provenance on future runs; current runner SHA is therefore different from the one recorded in this artifact. The earlier incomplete 1 ms artifact remains separate and is not counted as a GPU forward.

**Interpretation and limits.** This establishes that each of these exact three local byte snapshots can individually pass one 32-frame unmasked forward under the recorded local model/config, and gives a per-video resource envelope. It does not establish that a nine-forward claim/mask/control workload or batched run fits, that browser timestamps correspond to decoder PTS, that claims are valid, or that any explanation is faithful. The 0.66 GiB minimum headroom means any later score run must remain sequential and monitor memory. Rights remain unverified; all media-derived results stay local. **Faithfulness progress: `補助診断のみ`; direct generated-claim evidence remains 0/3 reviewed, 0/3 scored, 0 effect estimates, with no CI or test.** Next action: independently audit this output, then reduce the human-review/correspondence and atomic score-preflight burden; do not reopen the closed head search while this direct branch has actionable prerequisites.

## 07:55–08:00 JST — refresh actual CSV/QA target against current v6 claim bundle

**Purpose.** The prior target candidate was built against an earlier claim-manifest/sidecar pair. Its three video/claim identities and claim-text hashes matched the present v6 handoff, and its source CSV hash matched, but its exact claim manifest and sidecar hashes did not. Rebuild a current-byte-bound unmasked target candidate before attempting any future preflight.

**Verified result.** Ran `scripts/build_nextgqa_diagnostic_target_candidate.py` with the current v6 claim input manifest and sidecar, matching generation packet, lineage, worksheet, actual `test.csv`, v2 draft, local video root, pinned processor revision, and 32 frames. It returned `diagnostic_candidate_unvalidated`, with **3/3** source rows bound and **32 frames per item**. Independent verification recomputed the candidate JSON SHA-256 as `3885fd8a6029c483f1628a67710f6f8f5898472a4d14138cabb89dab73b1d6c7`, checked mode `0600` under a `0700` directory, and matched current claim-manifest, sidecar, CSV, lineage, worksheet and protocol hashes. Every A–E label was a distinct single token for all three rows. Model weights loaded=false, CUDA=false, forwards=0, score calls=0, authorization=`not_granted`. The builder's focused tests passed **15/15**, actual-CSV target integration tests passed **3/3**, and compilation/whitespace checks passed. The candidate remains local because it contains protected-data-derived target hashes and gold indices.

**Limits and next action.** This closes the stale claim-input binding discrepancy for the current candidate and confirms source/tokenization shape; it does not authenticate reviewer judgments, support intervals, clock/frame correspondence, target semantics, or a joint pre-outcome freeze. Direct faithfulness remains **0/3 human-reviewed, 0/3 scored, no effect/CI/test**. The shortest remaining path to a score is still independent human review and real browser-to-decoder evidence, followed by the missing production v2 atomic preflight and exact all-nine-batch freeze. Keep all derived data local. No further GPU job is justified until those scoring gates pass; the completed 3-video GPU work remains capacity-only.

## 08:00–08:03 JST — verify all-nine condition construction with the pinned Qwen processor

**Purpose.** The existing 3×3 batch builder had synthetic unit coverage but no run using the real cached Qwen2.5-VL processor. Validate the processor/vision preprocessing path now, with synthetic frame pixels and invented QA only, so a processor integration defect can be found before the human-gated data run.

**Implementation and verified result.** Added `scripts/smoke_nextgqa_qwen_condition_batch_builder.py` (SHA-256 `b5b403f0a5d250f10e3abe476429fdd2e2b4ba9a1a64fb741ae6a627e75d8ca9`). It disables CUDA and network access, loads only the locally cached pinned processor revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, creates three deterministic 32-frame synthetic RGB sequences and dummy five-choice items, and calls the existing exact condition builder. The real Transformers 5.18.0/Qwen processor constructed all **9/9** condition batches; exact reconstructed Qwen frame pixels matched, and the builder's typed batch manifest verified. Aggregate synthetic batch-manifest SHA-256: `c2988b6917e38a04df00fb40c0fbf10238d12a15e5f938c5ba587bc47e3afee9`. The report showed `model_loaded=false`, `model_forwards=0`, `cuda_initialized=false`, and `execution_authorization=not_granted`; source/review/clock authentication and protocol freeze remain `not_performed`. No dataset, real QA, claim, media, or private artifact was read or written.

**Checks.** The real processor smoke completed successfully. CPU condition-builder tests passed **6/6** and atomic freeze/scorer integration tests passed **12/12**; script compilation and `git diff --check` passed. The freeze integration is synthetic and does not authenticate upstream evidence or constitute a real score.

**Interpretation and next action.** The exact local Qwen processor can construct all nine masked/unmasked inputs in memory from correctly shaped inputs, reducing one implementation uncertainty before the first gated run. It does not verify real-video decoding, review intervals, player/decoder correspondence, source target semantics, or nine-forward GPU capacity. Direct generated-claim faithfulness remains **0/3 human-reviewed, 0/3 scored**, with no direct estimate, interval, or test. The head-confirmation branch remains closed below its preregistered +0.25 nat threshold; no head search was reopened. The next substantive path is still the actual independent review and 15-point browser/compositor-to-decoder evidence, then joint freeze and production atomic validation. The artifacts and this detailed local log remain local; rights are unverified.

**Post-audit hardening.** Static review of commit `b788f85` found the smoke's success assertion inferred nine effective batches from the manifest row count and did not assert every non-authorizing status. Tightened it to verify the exact 3×3 condition keys from returned batches, independently recompute the digest manifest, and assert zero model load/forwards/CUDA plus every authentication/freeze field and `execution_authorization=not_granted`. Reran the final script successfully: 9/9 batches, same manifest SHA, elapsed **8.37 s**, process peak RSS **1,936,408 KiB**. Final script SHA-256 is `45a104b01d3e23f023b810a92fffc15ecbfaa972de6cd58b13d8a2bdc9a491f9`; compilation and whitespace checks pass. The earlier static audit's stated no-execution scope was accurate at audit time; this final execution is the independently measured synthetic CPU run.

## 08:04–08:09 JST — synthetic full-cohort nine-forward GPU capacity and scorer-path test

**Purpose.** Determine whether the exact existing 3×3 scorer can complete nine sequential Qwen forwards at 32 frames per condition on the RTX 5070 Ti, independent of unresolved human/data authentication gates. This is an engineering capacity test only.

**Setup and code.** Added `scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py` (current SHA-256 `d2464b2281a51382b9868b2f1df2284397064c5eb20edc2eb65030cf14ea6433`). It creates deterministic synthetic RGB frames and invented 5-choice QA rows, uses the pinned local processor/checkpoint revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, builds and hash-freezes all nine condition batches on CPU before model-weight loading, then calls the shared scorer sequentially. Network is disabled; it uses local BF16 + 4-bit BF16-compute weights. A scorer-only test review fixed a CUDA-context initialization assertion before execution, moved the 12 GiB check after CPU input freeze but before weight load, added a 512 MiB free-VRAM check after device transfer and immediately before each model call, separated scorer attempts from actual model-call counts, labeled scoring-phase-only VRAM metrics, emitted null for unobserved resource values, and sanitized post-load exceptions. The independent code audit found no remaining material blocker; it did not run the GPU test.

**Verified GPU result.** Immediately before the run, the RTX 5070 Ti had 14,851 MiB free of 16,303 MiB; `.venv` reported CUDA 12.8, PyTorch 2.11.0+cu128, BF16 supported. The final synthetic run completed **9/9 Qwen model calls**, **9/9 scorer attempts**, and **3/3 triplets** in **31.46 s scoring time** with no failure. Scoring-phase peak allocated VRAM was **10.74 GiB**, peak reserved **12.47 GiB**, minimum free immediately before a model call **1.93 GiB**, and free at scoring exit **1.95 GiB**. Outputs contained aggregate counters only; per-item synthetic scores and D3 were discarded. A prior attempt stopped before batch building or weight loading because the precheck initialized CUDA and a later assertion incorrectly required it to remain uninitialized; after removing that contradictory assertion and moving the VRAM query, the final run passed. No real videos, QA, claims, labels, human decisions, or derived protected data were read or written.

**Interpretation and limits.** This verifies the nine-forward software path and a synthetic 32-frame GPU resource envelope. It is not an NExT-GQA score, does not establish the actual three videos' masked/control workload, and does not imply authorization. The earlier actual-video unmasked capacity run had a lower observed minimum free VRAM (0.66 GiB), so the synthetic 1.93 GiB headroom must not be substituted for a real-cohort memory guarantee; any actual run still needs sequential loading/monitoring and the preflight. Direct faithfulness remains **0/3 human-reviewed and 0/3 scored**, with no effect, interval, or test. The head confirmation remains closed below +0.25 nat; no head search was reopened. Rights remain unverified; all private artifacts stay local. Next work: finish the independently audited clock-evidence manifest builder/preflight plumbing, while the human review and actual 15-anchor capture/correspondence remain necessary before any real score.

## 08:15 JST — strengthen evidence-manifest candidate prevalidation

**Purpose.** Close remaining deterministic mismatches between the offline candidate builder and the downstream clock/frame evidence auditor, so malformed local capture declarations and equivalent-but-noncanonical PTS spellings fail before signing rather than during later audit.

**Changes and checks.** The builder now requires the exact capture-receipt top-level schema, timestamp, caller-pinned capture-helper SHA-256 and method, and the downstream auditor's window-process/PID/title, uint32 tick-wrap and 250 ms limit, and in-screen geometry rules. It explicitly checks exact token/anchor key sets across receipt, crop, and decoder maps, returning stable `ValueError` failure codes for incomplete joins. Decoder PTS must both equal the planned rational value and use its canonical `str(Fraction)` representation, matching the downstream auditor's byte-level rule. Added regression cases for equivalent `0/1` vs canonical `0`, helper hash-pin mismatch, duplicate/missing expected anchors, and an over-limit capture delay.

**Verification.** Focused builder plus existing evidence-auditor suites passed **25/25** using synthetic fixtures, including the integration path in which the produced Ed25519 manifest reaches the actual auditor and 15/15 anchors are structurally accepted with execution still `not_granted`. Additional adversarial JSON-type tests fail closed. The primary and independent auditor both reran the final suite successfully; `py_compile` and scoped `git diff --check` passed. No private claim/video/evidence input was opened and no GPU operation occurred in this cycle.

**Limits and next action.** This reduces candidate-builder/downstream-auditor schema mismatch; it does not authenticate a real person's review, prove screenshot crop purity, establish screenshot-to-decoder visual correspondence, jointly freeze the plan/key, or authorize calibration/scoring. No actual review export or screenshot capture exists. Direct generated-claim faithfulness remains **0/3 reviewed, 0/3 scored, no effect/CI/test**. Next: complete independent review of this hardening, then publish only these generic code/tests if the scoped audit remains clean; keep all video-derived evidence local. The shortest path to a real score still requires the prepared independent human claim review and 15 actual capture/correspondence decisions, followed by a trust-root freeze and production atomic preflight. No head search is warranted while these direct-branch actions remain useful.

**Audit/publication.** Independent audit confirmed the final 25/25 synthetic suite and no remaining material issue in the hardening. Committed only the two generic source/test files as `84c58c0` (`Harden clock evidence manifest prevalidation`), pushed without logs or private artifacts, and read back live GitHub `refs/heads/main` as the exact same commit. The detailed PDCA log remains local.

## 08:20 JST — malformed cohort-ID fail-closed check

**Purpose.** A separate independent design audit found that the draft-v2 synthetic contract checker could leak a `TypeError` while building its clock-config map if either row `video_id` or the caller's `expected_video_ids` contained a JSON list/dict. Close this error path so invalid synthetic input yields a bounded blocker report instead of aborting the CLI/API.

**Change and verification.** `_clock_config_expected` now validates expected IDs as a unique list of nonempty strings and validates each row/row ID before dictionary/set operations; malformed input returns the existing `clock_mapping_config_content_mismatch` path. Added cases for list/dict expected IDs and row IDs. Primary and independent auditor reruns both pass **12/12** focused synthetic tests, including the unchanged positive fixture that remains explicitly `synthetic_only`, `real_media_eligible=false`, 0 score calls, and `execution_authorization=not_granted`. Scoped diff check passes. The independent audit found the expected-ID hole after its first probe; the follow-up closes it. No private data/GPU used.

**Interpretation and next action.** This only improves failure reporting in a synthetic structural checker. It does not connect production validators, authenticate human review, freeze the protocol, or advance direct claim-faithfulness evidence. Keep it separate from the upcoming production-preflight design; direct status remains 0/3 reviewed and 0/3 scored, no effect/CI/test.

**Publication.** Only the two generic source/test files were committed as `b3f9a52` and pushed. A live `git ls-remote` verified origin/main at `b3f9a52de27092f97129ebe3ebaf38d7e0626c87`. The experiment log and all private artifacts remain local.

## 08:25–08:33 JST — local pre-review byte inventory (candidate only)

**Purpose.** Reduce the time between completion of the three human reviews and a later joint freeze by pinning which currently staged bytes are present, while avoiding content inspection or prematurely treating a reviewer package as authenticated evidence.

**Implementation and verification.** Added `scripts/build_nextgqa_pre_review_byte_inventory.py` and focused synthetic tests. It hashes bytes only; output has fixed role, size, and SHA-256 fields, no paths, filenames, tokens, QA, claims, video content, or signatures. Private input permissions and directory topology are checked; the package and staged media must each contain exactly three MP4s, with matching basenames and bytes. Missing/mismatching inputs remain `candidate_only`, never jointly frozen or score-ready, and produce CLI exit 2. Independent review identified a same-size concurrent-update race; the builder now compares inode, size, mtime, and ctime before/after hashing. Negative tests now cover extra package files, invalid private-directory mode, and output symlink preservation. Final synthetic suite passed **8/8** in both primary and independent runs, with compilation and scoped diff check passing. The production CLI help was inspected before use.

**Verified local result.** Hash-only inventory of the current v6 staged reviewer package and fixed source/code dependencies recorded **50 roles, 0 missing slots**. Final receipt: `data/private_nextgqa/pre_review_inventory_v6_20261017_0830_02/pre_review_candidate_byte_inventory.json` (directory 0700, receipt 0600). Status is `candidate_only`; review started/completed=false, joint freeze=false, score ready=false, execution authorization `not_granted`, rights `unverified_local_only`. No video frame was decoded and no claim/question text was read or emitted. The independent audit confirmed the final 8-test implementation, while retaining the explicit limitation that inventory does not authenticate custody-to-media identity and assumes the private directories are quiescent during the scan. Generic code can be published; the receipt and log remain local.

**Interpretation and limits.** This verifies byte presence, permissions, and package-to-staging media equality at one local point in time. It does not prove that the three media correspond to the intended custody rows, that a human has reviewed them, that screenshots match decoder frames, or that inputs are frozen. Direct generated-claim faithfulness is still **0/3 reviewed, 0/3 scored**, with no effect estimate, CI, or test. Next shortest path remains the user's promised later human review plus real 15-anchor capture/correspondence audit; this receipt can detect byte drift when those steps resume but cannot unblock GPU scoring. No head search is justified while this independent gate-preparation work remains useful.

**Publication.** Committed only the generic builder and synthetic tests as `2efc06d51e21581bfba09117f0fd5b317ff6bb77`; pushed to `origin/main`, then live `git ls-remote` returned the same hash. The private inventory receipt and detailed log were not staged or pushed.

## 08:34–08:38 JST — reviewer handoff order and role runbook

**Purpose.** Make the next human-dependent steps executable without accidentally opening the package before pre-outcome freezing or mistaking self-report for verified evidence.

**Change and review.** Added `docs/nextgqa_v6_human_review_runbook.md`. It points to the current private page and candidate inventory, explicitly labels both as preparation only, defines pre-outcome freeze owner/custodian/claim reviewer/frame-pair reviewer roles, specifies the hidden information and all-occurrence interval policy, details the 15-anchor OS/browser/crop/correspondence sequence, and requires nonzero evidence overlap plus equal-count disjoint controls for all three rows before model loading. It also states missing/mismatched/ambiguous evidence blocks the cohort and that the next score requires a production atomic preflight. Independent review found omissions about explicit freeze-owner assignment, control eligibility, and disclosure acknowledgment/withdrawal; all were repaired and then confirmed against the v6 policy and v2 protocol. `git diff --check` passes. The runbook contains no private video, QA, claim, or reviewer data.

**Verified current gate state.** A fresh metadata/name-only audit found no review export, screenshot, capture receipt, or crop artifact since the candidate inventory. The inventory remains 50/50 present candidate roles, but direct evidence remains **0/3 reviewed and 0/3 scored**, with no effect/CI/test. No GPU run is scientifically justified while the pre-outcome freeze, human review, 15 real captures/correspondence, and production preflight are absent. The separate 79-item head confirmation remains closed below its preregistered practical threshold; no head search is reopened.

**Next action.** Because the protocol remains a draft and reviewer trust keys are not assigned, do not open the package yet. The next external gate is to assign the freeze owner and two independent reviewers, bind their keys and the exact v2 inputs/code before viewing, then obtain the three claim-review exports and 15 capture/frame-pair decisions locally. Reassess production-gate implementation against those now-frozen input contracts and run every existing audit before any masked inference.

**Publication.** Committed only the generic runbook as `f9675af78db5af3d405ad5dc63b9b69894b0b83c`, pushed it, and verified the live `origin/main` ref matches. The private inventory and PDCA log were not staged or published.

## 08:39–08:43 JST — reject upstream failures before nine-batch CPU snapshots

**Purpose.** The all-nine condition bundle can occupy substantial host memory. If a concrete upstream gate rejects the cohort, avoid cloning and hashing all nine already-built processor batches; retain the exact hash-bound final gate after materialization.

**Measurement.** On the pinned local Qwen processor with deterministic synthetic frames and invented QA only, the exact 3×3 condition builder produced nine batches with **694,534,644 tensor bytes total** (maximum one batch 77,170,516 bytes). The measured extra snapshot-copy upper estimate is about **0.647 GiB**. The run took 3.17 seconds inside the build segment, process maximum RSS was 1,940,148 KiB, CUDA stayed uninitialized, and no model loaded. This quantifies only the synthetic processor path; real-video batch bytes may differ.

**Implementation and audit.** Added required `upstream_precheck` before snapshots, but retained `authenticated_gate_check` after all nine observed hashes equal the pre-frozen expected hashes. Cheap nested-schema and digest-format checks run first. The early hook is explicitly read-only/side-effect-free and cannot write a freeze receipt. Before invoking it, the function snapshots token IDs, record IDs, batch mapping shells, and the expected hash map; adversarial tests mutate the source batch and caller hash map during precheck and confirm the old expected digest still fails closed. Tests also verify precheck exception/False causes zero `.to('cpu')` calls, malformed condition schemas skip the hook, final gate runs only after nine snapshots and hashes, and final gate exception/False never returns frozen inputs. Focused suite passed **18/18** in both primary and independent runs; compilation and scoped diff check pass. Synthetic scorer integration still uses fabricated data and returns `execution_authorization=not_granted`.

**Limits and next action.** This saves up to the observed snapshot volume only when the upstream precheck rejects; successful precheck leaves the same peak snapshot memory. The generic module does not authenticate a real precheck—production must own the closure and revalidate time-sensitive evidence in the final post-hash gate. This neither implements the production orchestrator nor creates direct faithfulness evidence. Latest direct status remains **0/3 human-reviewed and 0/3 scored**, no estimate/CI/test, with no review exports or browser captures. The next external gate remains pre-outcome joint freeze and independent reviewers; once stable inputs exist, wire the early read-only validators and final exact-hash gate in the real production entry point.

**Publication.** Committed the freezer, synthetic smoke call-site, and tests only as `6e6434e3ff51aa47e94bd91a05b01b24d585a72f`, pushed them, and verified live `origin/main` at that exact hash. The synthetic measurement output and detailed PDCA log remain local.

## 08:46 JST onward — remove redundant CPU copies from score preflight

**Purpose.** Reduce avoidable host-memory traffic in the first gated 3×3 score run without changing the protocol, authorization, condition batches, or model forwards.

**Finding and change.** Independent source audit found that scoring validation called `.to("cpu")` on all nine freezer-produced `FrozenBatch` snapshots, then the hash check called `.to("cpu")` on the same nine again. `FrozenBatch.to` recursively clones tensor leaves, so this creates redundant full-batch copies before inference. The scorer now verifies the exact `FrozenBatch` type, recursively rejects non-CPU tensor leaves without copying, and hashes those existing CPU snapshots directly. Per-forward transfer and post-transfer hash validation remain in place.

**Checks.** Focused scoring tests pass **14/14**. New tests prove a bad ninth hash fails before any `.to` call or model forward, a successful score performs exactly one transfer per batch (nine transfers for nine forwards), and a non-CPU tensor is rejected without an attempted copy. Existing mutation-detection, non-finite-logit, exact-forward-count, and incomplete-cohort tests pass. `py_compile` and `git diff --check` pass. Independent audit found no material blocker. No private data, model, GPU, or scoring outcome was accessed in this cycle.

**Synthetic integration measurement.** The pinned local Qwen processor built the same deterministic synthetic 3×3 bundle, then the generic freezer and a fake CPU model exercised all nine score calls. The scorer reported 9/9 completed forwards, invoked `.to(device)` exactly 9 times (once per forward), completed its score loop in 0.769 s, and left CUDA uninitialized. `execution_authorization=not_granted`; fake logits and the resulting synthetic D3 are not a model or research result. Based on the prior measured batch sizes, the removed validation/hash passes would each clone 694,534,644 tensor bytes cumulatively across the cohort (about 1.39 GB total clone traffic), with a largest single-batch clone of 77,170,516 bytes; this is byte-count arithmetic, not an observed RSS reduction. The measurement establishes copy-call reduction and successful 9-forward wiring only, not wall-time improvement on GPU.

**Interpretation and limits.** This removes two unnecessary CPU tensor-copy passes from the preflight path; it does not establish a particular wall-time or peak-RSS reduction. Direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored with no effect estimate, interval, or test. The protocol remains draft; production trust/freeze gates and the human/browser evidence are still absent. Do not use head exploration as evidence for this direct branch.

**Next action.** The code and synthetic copy-count measurement are complete and published. Direct work now needs the externally performed pre-outcome freeze and independent human review/capture; once their exact artifacts arrive, audit them and implement the production all-three gate against the frozen bytes. The GPU remains available, but no score should run before these prerequisites.

**Audit and publication update.** Independent reviewer found no material correctness issue and confirmed the pre-forward nine-hash gate, transfer-time hash check, and incomplete-cohort behavior remain intact. The relevant integration suites passed **62/62**, followed by compilation and `git diff --check`. Synthetic 3×3 processor exercise completed 9/9 fake-model forwards with exactly nine `.to(device)` calls and no CUDA initialization. Committed only `scripts/nextgqa_claim_mask_scoring.py` and its test as `1937da1ad92db480d18f8c62b36553b69ba4aa5a`; push succeeded and a live `git ls-remote origin refs/heads/main` returned the same hash. The detailed work log and unrelated existing worktree files were not staged.

**Current readiness refresh.** The local pre-review inventory still has 50/50 required roles present, but `joint_freeze=false`, `review_started=false`, `review_completed=false`, `score_ready=false`, and `execution_authorization=not_granted`. A fresh metadata/name-only scan still finds no human review export, real capture, signed evidence, or correspondence decision. The RTX 5070 Ti is currently visible to the authorized elevated command with 14,896 MiB free; `.venv` is Torch 2.11.0+cu128 and CUDA 12.8. No browser executable or Playwright/Selenium package is installed in WSL, and no browser process is running. This confirms GPU capacity is not the current score blocker; human review/freeze and actual browser/capture evidence are.

**Pre-review freeze design audit.** A separate read-only assessment found that a signed local receipt could bind candidate bytes but cannot, by itself, release the package under the current runbook: the runbook freezes v2 and reviewer public-key hashes before opening, v2 is still a draft, and a key whose hash is supplied inside its own receipt is self-authenticating. A local timestamp would be a signed machine-clock assertion, not proof of ordering. A distinct reviewer-handoff freeze could be designed, but it would require an explicit runbook/protocol decision and an independently trusted signer/reviewer roster. No such identities or trust anchor are currently in the worktree, so no authorizing receipt or release was fabricated.

**Current faithful-progress accounting.** Direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored, with no direct effect/CI/test. The existing auxiliary QA-gold-interval results remain n=10 at 16 frames (mean −0.1375 nat, SD 1.5969, 4/10 positive) and the same assignments at 32 frames (mean −0.3375 nat, SD 1.9085, 6/10 positive); these are finite exploratory answer-evidence diagnostics, not claim-faithfulness estimates. They do not justify repeating that branch or bypassing the human gate.

## 08:54 JST onward — separate human-review release from score freeze

**Purpose.** Reduce avoidable delay before the first valid human review while keeping the scoring protocol and GPU execution closed until their own prerequisites pass.

**Evidence and decision.** Fresh local inventory and two independent design audits confirm 50/50 candidate package roles exist, but the inventory remains candidate-only: `review_started=false`, `review_completed=false`, `joint_freeze=false`, `score_ready=false`, `execution_authorization=not_granted`. The v2 draft already fixes the finite three-record estimand, denominator/failure rules, three conditions, 32-frame sampler, and mask/control rule; policy v6 fixes review/display/export rules. Its exact text and those commitments can be pinned for review without declaring the draft score-frozen, provided no score/model access is permitted and any later change to the cohort, claims, review rule, estimand, metric, conditions, or eligibility/failure/control rules invalidates affected reviews. Later scoring still requires a separate freeze over human exports, clock/frame evidence, source targets, code, exact nine batches, and the final v2 protocol.

**Change.** Updated `docs/nextgqa_v6_human_review_runbook.md` to distinguish a review-only release receipt from the later score freeze. It requires an independently established roster/key trust root, exact byte bindings for package/claims/policy/renderer/capture plan and the fixed v2 estimand, witnessed ordering if only local timestamps exist, and explicit `review_only`, `score_ready=false`, `model_load_permitted=false`, `execution_authorization=not_granted` states. The existing prohibition against self-attested trust roots remains. No actual release receipt was fabricated and no protected package was opened.

**Independent audit and limits.** A separate audit found the split can preserve the finite-cohort estimand if all stated invariants are frozen and later scoring binds the completed review/calibration outputs. It also confirmed the split does not by itself unblock review: no independent human assignments/trusted key root exist yet, no actual browser captures or 15 pair decisions exist, and the score-freeze receipt/production atomic gate is still absent. GPU re-runs would add no direct faithfulness evidence under these conditions. The documented auxiliary QA-gold interval result remains exploratory n=10 (16f −0.1375 nat; same items 32f −0.3375 nat) and is not repeated. The closed head confirmation remains n=79, −0.0756 nat, 95% t CI [−0.1578,+0.0065], below its fixed +0.25 nat practical threshold; no head search is resumed.

**Faithfulness progress.** `直接評価済み`: 0/3 generated claims human-reviewed, 0/3 scored; no direct effect, interval, or test. `補助診断のみ`: prior real-video capacity and exploratory QA-interval runs described above; neither measures generated-claim faithfulness. Rights status remains unverified; protected artifacts remain local.

**Next action.** Freeze and release the v6 package through the review-only process once an independent freeze owner/custodian and the two reviewers are assigned and their key root is independently pinned. Then complete 3 claim reviews and 15 browser-to-decoder pair decisions; in parallel, use the frozen outputs to finish and audit the score-only production preflight before any model loading.

**Publication.** The generic runbook change only was committed as `48a0e398601648db69104169085af9550c061618` and pushed. A live `git ls-remote` confirmed `origin/main` at the same commit. The private inventory, staged videos, and detailed work log were not staged or published.

## 09:02 JST — audit a faster auxiliary GPU experiment against prior exposure

**Question.** A paired real-video versus text-only Qwen baseline could quantify whether adding sampled video changes NExT-GQA fixed-choice answers/margins, independently of the delayed claim-review branch. It would be an auxiliary QA visual-input contribution diagnostic, not generated-claim faithfulness. A primary-source search found that the 2026 *Chains That See, Answers That Don't* paper already uses paired Qwen2.5-VL Video-MME accuracy including a no-video condition; any NExT-GQA adaptation would be a dataset-specific diagnostic, not a novel intervention ([arXiv:2606.22862](https://arxiv.org/abs/2606.22862)).

**Outcome-blind availability check.** The local test split contains 5,553 rows / 990 video IDs; 3,155 rows with complete five-choice/unique-answer structure also resolved to local media. I scanned all 14 local `nextgqa_samples_*.json` files and compared their video paths plus the available 2026-10-16 `_03` HMAC identity index (212 video-ID, 118 media-hash, and 225 video/QID tokens). After known sample-path and video-ID exclusions, 79 distinct test videos remained, but all 79 media hashes matched the 118-entry legacy/media set; no confidently new test media remained. On validation, 3,358 source rows yielded 234 valid local-media MC rows, 33 distinct candidates before the same media check, and all 33 media hashes also matched that set. The exact-QID HMAC test found no pair match among the first hash-ranked candidate questions, but this does not recover screening history. The index explicitly has `historical_completeness_authenticated=false`, and several media sources have unverified provenance. An independent agent is reproducing these counts; no sample IDs, paths, text, or answers were emitted.

**Decision and limits.** Do not spend GPU on this auxiliary run now: the available local videos are already represented in the known media-identity ledger, and reusing them would not meet the no-reuse requirement for a defensible new cohort. This is an exposure/readiness result, not an effect estimate. Direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored. The earliest direct route remains the review-only release freeze, which now has an audited runbook, followed by independently assigned human reviewers and actual browser/frame evidence. No head/layer/alpha search is resumed because the direct branch and this human-dependent path still have actionable work.

**Independent aggregate reproduction and scope correction.** A second audit rebuilt the `_03` identity sets from all 45 registered sources using the same mode-0600 32-byte key and reproduced the stored index byte-for-byte/object-for-object (212 video IDs, 225 video/QID pairs, 118 media hashes). It recomputed 14 manifests / 488 rows and applied stricter eligibility: nonempty question/five choices, exactly one normalized answer match, valid annotation interval, dataset-ID mapping, unique local MP4, and exact-byte media hash. This yields 3,132 eligible test rows / 500 videos before exclusions and 234 validation rows / 33 videos; after the full path/ID/pair/media-HMAC exclusion union, both have zero remaining videos, pairs, and media hashes. The earlier 3,155 test-row count was a less strict MC/local-media count, not this stricter interval-eligible denominator. The 118 legacy media hashes include unverified provenance, so the match is a conservative exclusion, not proof of prior exposure. The zero result is limited to the inspected local candidate files, manifests, and `_03` index; historical completeness is unauthenticated. No identifiers, paths, questions, answers, or media were exposed in the audit output. No GPU run was made.

**Repository update.** The research overview was updated to point to the review-only runbook and accurately state the uncreated trust root/receipt and 0/3 review/score status. Only `docs/research_overview.md` was committed as `e267fc8`; `git push origin main` succeeded and advanced the remote from `48a0e39`. A later read-only `ls-remote` attempt failed on transient DNS resolution, so no separate network readback was obtained; the successful push and local `origin/main` tracking ref both show `e267fc8`. The worktree's unrelated/private files remain unstaged.

**Next action.** The currently mounted WSL filesystem exposes no additional dataset mount; `/mnt/c` is read-only and is not a research-write target. Continue with local candidate discovery and human assignment only if they can materially unblock the first score. Do not relax the media exclusion after seeing these counts. A usable human review still needs an independently pinned freeze-owner/reviewer roster; the user has been asked whether these roles can be assigned. Direct faithfulness remains 0/3 reviewed, 0/3 scored, no effect/interval/test.

**Independent head-search disposition.** A separate source/result audit confirmed that the top-4 79-video confirmation is closed at −0.0756 nat (95% t CI [−0.1578,+0.0065], below the fixed +0.25 nat threshold), and the prior top-8 exploratory n=3 used the same sample hash as the existing head smoke. Its +0.0417 nat mean has a 95% t CI [−0.4327,+0.5160], with only one of three contrasts positive; it cannot justify further parameter search. The production head wrappers contain a fail-closed guard on the closed branch. There is no independent cohort, and varying K/layer/alpha on already evaluated videos after seeing these outcomes would be post-outcome search. The user's conditional allowance to resume head exploration is therefore not triggered: human handoff and dataset inventory remain useful unfinished work. No GPU job was run.

**Local package state independently rechecked.** The current candidate inventory contains 50 artifact roles and zero missing required slots, while `joint_freeze`, `review_started`, `review_completed`, and `score_ready` are all false and execution is `not_granted`. These are package-byte readiness counts only; they do not establish a trusted signer roster, reviewer completion, or score authorization.

## 2026-10-17 — add an integrity-only receipt check

The primary agent reviewed the new helper source and then independently ran the focused synthetic verifier suite (**17/17**) and existing synthetic inventory suite (**8/8**), plus compilation and CLI `--help`. A separate auditor checked the signature/roster/hash boundaries, private modes, exclusive output creation, symlinked output parents, and non-authorizing statuses. The only robustness issue found (memfd descriptor cleanup after a write failure) was fixed and covered by a regression test. No real receipt/key or private package was created/read by this implementation check. `docs/nextgqa_v6_human_review_runbook.md` now explicitly says the helper is not a release receipt and its success cannot open the package. A public synthetic-only implementation log was added. Direct faithfulness remains 0/3 reviewed, 0/3 scored; no GPU inference occurred.

The independent documentation audit found that the runbook's prior version exposed concrete local package and inventory paths; those were generalized. It also identified that the helper's three-role key roster does not bind the optional separate custodian. The final runbook now states that gap and requires the actual release procedure to bind either the owner-as-custodian assignment or a separate custodian identity/key. The modified 2026-10-16 log contains pre-existing user edits with other local filenames; it was deliberately not staged or changed. Only the four audited public files were committed as `2fa19e7e8948078a76983f5002f3ddb10de6193d`; push succeeded and a live `git ls-remote` independently confirmed `origin/main` at the same hash. Unrelated and private worktree files remain untouched.

**Primary-agent reproduction of alternate-cohort exclusion.** Rebuilt the `_03` HMAC index from all 45 registry-listed sources with the local mode-0600 key; both the resulting object and serialized bytes exactly match the stored index (212 video-ID, 225 video/QID, 118 media-hash tokens; historical completeness remains false). Independently invoked the existing test eligibility function and adapted the same checks for validation, without emitting IDs, paths, question/answer text, or hashes. The current `~` scan has 601 MP4 files and 599 last-two-component keys, one of which is ambiguous. Test has 5,552 usable five-choice rows; 3,155 have a unique local video before interval validation, and 3,132 also have a valid annotation interval. After sample-path exclusions, 85 test video candidates remain; six hit video-ID/pair HMAC exclusion, leaving 79, and all 85 match a media HMAC (including those six), so none survives the union. Validation has 3,358 five-choice rows, 234 interval-valid unique-media rows and 33 distinct candidate videos; all 33 match media HMAC exclusion. Thus both splits yield zero locally disjoint candidates under the conservative union. The 118 media set includes provenance-unverified legacy/custody sources, so HMAC equality is conservative linkage—not proof of actual exposure. Scope is only the searched home tree, 14 manifests, and `_03` index; history is unauthenticated. No new cohort GPU diagnostic was run.

## 2026-10-17 — synthetic nine-forward GPU capacity smoke

An individually elevated read-only GPU inventory found the RTX 5070 Ti, CUDA-enabled repository `.venv`, 14.66 GiB initially free VRAM, and about 13.5 GiB host/WSL MemAvailable. The relevant condition-builder, atomic-freezer, and scorer tests passed 38/38 under the correct `PYTHONPATH=scripts` invocation. The first invocation without that path failed at test-module imports only; no test body ran, and the corrected invocation passed.

Ran `PYTHONPATH=scripts .venv/bin/python scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py` with per-command elevated access. Synthetic-only three-row × three-condition scoring completed 9/9 forwards and 3/3 triplets in 30.158 s. Peak allocated/reserved VRAM was 10.744/12.473 GiB; minimum free memory before a forward was 1.921 GiB. After exit, the GPU had 14,914 MiB free and 1% utilization. The run reports `faithfulness_result=false`, source/review/clock authentication not performed, and execution not granted. A public synthetic-only experiment entry records the source hashes and limitations.

Faithfulness progress: direct generated-claim faithfulness is still 0/3 reviewed and 0/3 scored, with no direct estimate, interval, or test. The GPU path capacity question is now resolved for synthetic inputs only. The 3 human claim reviews, 15 browser-to-decoder frame-pair reviews and calibration, trusted joint freeze, and production all-three-row preflight remain gates. No head search or repeat of auxiliary gold-interval masking was warranted.

Independent smoke-code audit confirmed all nine CPU batches are constructed and frozen before model load; batch hashes are validated before first forward and again after transfer; inference is sequential; and a live free-VRAM guard is called immediately before every model forward. `faithfulness_result=false`, authentication `not_performed`, and authorization `not_granted` are emitted on success. The 30.158 s figure is scorer-only through CUDA synchronization, excluding CPU batch preparation and model loading. Peak stats reset after model load and include resident model allocation. The synthetic fixed-resolution workload does not establish headroom for real prompts/resolutions or concurrent GPU usage. No material auth/scoring bypass was found. Current run source SHA is `747fe4efe01183f9f41fbb4095d4d828bed6748d3f49652a9f7fbbdd20ef5c22` at source commit `2fa19e7e8948078a76983f5002f3ddb10de6193d`; an unrelated older log refers to an earlier smoke script SHA and was not used for this run.

## 2026-10-17 — implement review-only witnessed release-order gate

To reduce the nearest pre-review blocker without opening the review package, added a new isolated verifier (`scripts/audit_nextgqa_review_release_order.py`) and synthetic tests. It composes the unchanged candidate-integrity verifier, rechecks the current inventory-bound bytes, validates a second externally rooted five-role roster (owner, custodian, two reviewers, distinct release witness), and verifies the witness signature over exact receipt/roster/inventory/artifact bindings and an independently supplied ceremony ID. Custodian may share the owner key; witness and reviewers remain role-separated. Output permits only procedural human-review opening and keeps model load, score readiness, and execution authorization false.

Independent source audit found no bypass and required precision changes: success says `review_release_order_witness_attestation_valid`; it verifies the witness's signed statement, not actual chronology or human identity. It does not enforce file access or track consumed ceremony IDs/replay. Those limits are explicit in output and `docs/nextgqa_v6_human_review_runbook.md`. No actual package, root, key, or attestation was accessed or created.

Focused new suite 12/12, existing candidate verifier 17/17, primary rerun 29/29; `py_compile`, CLI help, whitespace checks pass. The positive tests remain synthetic and non-authorizing for scoring. Direct faithfulness remains 0/3 reviewed and 0/3 scored, with no direct effect/interval/test. Actual roster identities, out-of-band root pin, witnessed ceremony, human reviews, 15 capture/frame pairs, clock calibration, and production all-three-row preflight remain outstanding.

## 2026-10-17 — fresh NExT-GQA auxiliary GPU cohort (pre-score plan)

**Purpose and status.** Direct claim-specific faithfulness remains gated by the pending human review and calibration. To get a real-video empirical measurement sooner without relabeling that auxiliary measure, prepare a fresh, outcome-blind NExT-GQA answer-evidence cohort for the existing gold-interval margin endpoint. All materials and derived scores remain local; rights status is unverified.

**Frozen selection and endpoint.** Candidate videos come from the local NExTVideo ZIP and NExT-GQA test split. Eligibility requires a valid annotated interval, five nonempty choices with exactly one normalized gold match, an unambiguous ZIP media key, exclusion of known sample paths and exact prior video/pair/media HMAC matches, and no media SHA in the previous-pilot exclusion set. The separate private pair allowlist binds the eligible QA pairs. For the smoke, choose the first eligible video/pair under the runner's existing deterministic hash order, with 16 uniformly sampled frames and five fixed conditions (`full`, black gold-interval mask, black equal-count control, blur gold-interval mask, blur equal-count control). Freeze its hashes before seeing any score. If smoke passes, exclude that smoke media SHA and preflight the first 10 eligible distinct videos by the same fixed order; score exactly 10 (50 forwards), no replacements, using the same 16-frame/five-condition endpoint. Primary contrast is black equal-count control minus black gold-interval mask in answer log-margin. Blur/full conditions are secondary diagnostics only. No post-hoc subgroup, threshold, or tuning; no population CI or confirmatory p-value for this convenience cohort.

**Cost and stop rule.** Candidate preparation read and SHA-256 hashed the 24.25 GB ZIP, hashed eligible media, and extracted 490 unique videos (~2.33 GB). The smoke is expected to require 5 Qwen forwards after model load (~30–60 seconds scorer time; model loading/preparation additional), with roughly 9–11 GiB peak VRAM based on prior pilot/synthetic capacity logs. The main run is 50 forwards, expected roughly 2–4 minutes scorer time and similar VRAM; resource guard must pass before model load and each forward. If preflight, smoke, provenance, or resource guard fails, do not relax requirements or score a partial cohort.

**Verified preparation.** Builder synthetic tests (9/9) and runner synthetic tests (16/16) pass, as do compilation and `git diff --check`. Outcome-blind local build reports 5,553 CSV rows, 2,611 sample-path exclusions, 31 prior-video exclusions, 513 prior-media-matching rows (79 unique media hashes), 14 invalid intervals, 2,384 eligible QA pairs across 490 unique media files, zero ambiguous/missing ZIP media, and zero choice-ineligible rows after prior filters. These are eligibility counts, not model outcomes. Prior exposure index explicitly has `historical_completeness_authenticated=false`; unseen prior exposure cannot be ruled out.

**Faithfulness progress.** Direct generated-claim faithfulness remains **0/3 reviewed and 0/3 scored**, with no direct estimate, uncertainty interval, or test. This planned and prepared test is `補助診断のみ`: real-video QA answer-margin dependence on benchmark gold intervals, not claim-specific explanation faithfulness. Do not combine its result with the earlier 10-pair pilot because cohort identity and sample selection differ.

**Next action.** Run GPU outcome-blind preflight for the one-item smoke under per-command elevated access; if passed, freeze the exact hash-only cohort lock, then run the five-forward smoke. Inspect only technical gate/failure status before proceeding to the predeclared n=10 cohort. After the fixed n=10 score, independently verify all outputs/counts and add the estimate and finite-sample caveats here. No protected artifact or result is to be pushed.

## 2026-10-17 — fresh real-video GPU result and independent audit

**Execution.** The one-video smoke preflight passed (1/1 video, 16 frames, five conditions); its five-forward score completed with no failure and was excluded from the next cohort by its private media hash, as frozen in the pre-score rule. The fixed fresh cohort preflight then passed 10/10 distinct videos, 16/16 frames each, with 10 timing passes and zero rejects. The pinned Qwen2.5-VL-7B-Instruct checkpoint (revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`), Torch 2.11.0+cu128 / CUDA 12.8, 4-bit weights, BF16 compute, seed 20261015, and RTX 5070 Ti were used. The score attempted and completed all **50/50 forwards** and all **10/10 five-condition sets**, with zero failures, in 143.303 s; peak allocated VRAM was 7.443 GiB. The local private score is `data/private_nextgqa/answer_margin_fresh10_score_20261017_01.json`, mode 0600, SHA-256 `5837abae96ebdd460015d5e22e135e9bb166ac3b7dc5252000a8b8cf0797e0af`. The preflight and exact hash-only cohort lock are also mode 0600 and are bound by the score's recorded byte hashes. No raw QA text, video, IDs, paths, generation, or per-token logits were written to the result.

**Observed primary result.** On this deterministic, timing-eligible, non-probability sample of 10 videos, the predeclared primary contrast—black equal-count control gold-vs-best-foil next-token margin minus black gold-interval-mask margin—has mean **+1.025 nat**, sample SD **2.152 nat**, median **+0.500 nat**, range **[−2.125, +4.750]**, with 7 positive and 3 negative paired effects. Full-input restricted A–E top-choice correctness was 6/10 (60%); that is not general task accuracy. The secondary blur control-minus-evidence mean is +1.300 nat (SD 1.947), and the mean black-minus-blur gold-mask margin difference is +0.113 nat (SD 1.053). The runner records no inferential CI or p-value and uses no test; the independent audit's illustrative iid t interval `[−0.515, +2.565]` is post-hoc only and cannot be interpreted as population uncertainty because selection is deterministic and the cohort is not probabilistic. There is one predeclared primary contrast; secondary quantities are descriptive, with no multiplicity-adjusted claims.

**Interpretation and limits.** Positive primary mean means that, on average in these 10 selected QA/video pairs, masking the benchmark interval reduced the model's A–E gold-vs-best-foil next-token margin relative to an equal-count black control mask. It supports only exploratory benchmark gold-interval answer-evidence dependence. It does **not** measure model-generated explanation claims or claim-specific faithfulness. The 490-media local pool came from 2,384 eligible QA pairs; 479 timing-eligible candidates remained unscanned after selection. The cohort is deterministic, small, and not representative by design. Black masking can be OOD; blur is also not guaranteed in-distribution. Annotation player/source clock was not calibrated against decoder PTS. The HMAC exposure index is incomplete, so unseen prior exposure cannot be ruled out. Rights remain unverified; all protected data and derived results stay local.

**Independent audit and repair.** A separate auditor independently recomputed all 10 primary effects and the summary, verified 50 requested/attempted forwards, zero failures, 10 locked assignments, exact preflight/lock/pair-filter bindings, and the private artifact modes. It found no raw text/video/identifier leakage. It also found the runner's prior-media exclusion file was parsed and hashed in separate reads; that loader is now hardened to read and hash one verified 0600 byte snapshot, reject duplicate hashes/extra fields, and refuse symlinks. This repair is post-run; the old score remains bound to its recorded pre-repair runner SHA. The actual input exclusion-file digest matched the preflight and score record on independent recheck. Relevant suites pass 34/34, and the full script suite passes **1,299/1,299** after also classifying three existing synthetic/engineering model loaders in the static inventory test. Compilation and `git diff --check` pass.

**Faithfulness-progress assessment.** Direct generated-claim faithfulness is still **0/3 human-reviewed and 0/3 scored**; no direct effect estimate, CI, or test exists. This real-video GPU experiment is `補助診断のみ`. The direct branch remains blocked by human review, 15 frame screenshot/decoder correspondence reviews and clock calibration, trusted joint freeze, and an audited all-three-item score gate. Rights are authorized for local-only operation by the user but remain independently unverified. The closed head confirmation/search is not reopened: this auxiliary result provides no rationale for tuning heads.

**Next action.** Do not tune or subgroup on this result. Continue work that directly removes the remaining human review/calibration/freeze gates; while those remain external, use only separately predeclared exploratory checks with a clear diagnostic question. No GPU score, private artifact, or data-derived statistic is to be pushed.

## 2026-10-17 — post-hoc locked-cohort 32-frame sampling sensitivity (pre-run)

The fresh 16-frame result is heterogeneous (7 positive, 3 negative; SD 2.152 nat), while temporal frame sampling is a live measurement sensitivity. Before any 32-frame score, the next exploratory diagnostic is to re-evaluate the **same exact 10 media/QID assignments** bound by `answer_margin_fresh10_lock_20261017_01.json`, using 32 uniformly sampled frames and the same five conditions, model revision, seed, contrast, and preflight/score code. It holds record assignments and metric fixed while changing the frame sampler, frame-grid-derived evidence/control geometry, and model input length; these components are not separately identified. It does not select new records or alter the primary endpoint. The 16-vs-32 per-record paired differences and sign consistency will be descriptive; no p-value, population interval, or confirmation claim will be made. This check is post-hoc because the 16-frame result is already known.

Expected cost is 50 forwards, approximately 3–5 minutes scorer time plus preflight/model load, with roughly 9–10 GiB peak VRAM based on the prior 32-frame pilot; the fixed guard requires at least 12 GiB free before loading and before each forward. Stop if exact lock/preflight binding, CUDA/BF16, memory, or full 10-set completion fails; no replacement, retuning, or partial-result averaging. Protected files remain local.

### 32-frame sensitivity execution failure and failure-path repair

The 32-frame outcome-blind preflight passed 10/10 locked videos, 32/32 frames each, zero timing rejects, with the exact 10-row lock SHA matched and initial free VRAM 14.66 GiB. The score process later raised `torch.AcceleratorError: CUDA error: unknown error` at the final `torch.cuda.synchronize()` call. No score JSON was produced, so requested/attempted forward count, completed sets, and any effect are **unverified and not reportable**. A bounded elevated `nvidia-smi` check returned `Failed to initialize NVML: N/A`; this is consistent with the GPU management path becoming unavailable, but the exact driver/kernel cause is unknown. Do not retry inference while NVML is unavailable. The already completed 16-frame run remains a separate verified result; it does not rescue or fill in the failed 32-frame run.

The failure exposed a runner gap: a final asynchronous CUDA failure bypassed artifact writing entirely. Added guarded final synchronization; if it fails, all in-memory score records are discarded and a private `partial` failure artifact is written with unknown completed-condition state and null estimates/accuracy, never with partial scores. Peak allocation and device name are omitted on this branch. The independent audit confirmed the intended fail-closed behavior and prompted the explicit unknown/null fields. Synthetic regression coverage is in `test_final_cuda_sync_failure_discards_unverified_metrics`; it does not establish GPU recovery. The focused tests pass 36/36 and the complete script suite passes **1,301/1,301**. Generic runner/builder code and tests were committed as `c34c3c84177e053d967b2257544f53840ec3bdd9`, pushed, and live-verified on GitHub `main`; no private media, scores, manifests, or logs were included. A later bounded `nvidia-smi` check still returned `NVML: N/A`; the relevant kernel ring shows `dxgkio_escape/query_adapter_info: Ioctl failed: -22` after this run. That establishes WSL GPU management-path failure but not its underlying cause. Do not relaunch inference while this persists. No head search is warranted while direct review/calibration and another useful analysis path remain; do not claim the 32-frame sensitivity completed.

**Next action.** Continue CPU-only direct-branch audit and review-handoff improvements while the GPU management path is unavailable. Retry the already frozen 32-frame sensitivity only after a bounded elevated GPU check reports the adapter healthy and the preflight/resource gates pass. Direct faithfulness still requires the independent human reviews and clock calibration; do not replace these with synthetic results.

## 10:15 JST — recover review-inventory freshness and audit GPU-result source provenance

**GPU availability check.** A bounded elevated `nvidia-smi` query again failed with `Failed to initialize NVML: N/A`. A separate 4,096×4,096 BF16 CUDA matrix-product smoke using the existing `.venv` also failed before device selection: Torch 2.11.0+cu128 reported `torch.cuda.is_available() == false` and `cudaGetDeviceCount(): invalid argument`; no tensor or model forward ran. This independently confirms that the inference path is still unavailable in this WSL session. No 32-frame retry was attempted.

**16-frame score provenance audit.** The private score artifact remains hash-stable at SHA-256 `5837abae96ebdd460015d5e22e135e9bb166ac3b7dc5252000a8b8cf0797e0af`; it records 50 requested/attempted forwards, zero failures, and runner SHA-256 `a8b3f9b53c43a55acfe48c3e7f1be8c98dfc02f8b0e3201369bee23bb91a4e05`. The current hardened runner has SHA-256 `a8be815150699bffd66623c2f53ed49cb010ce59507751f6306fda25a39cd417`. I independently scanned every locally stored Git blob in the repository object database (1,837 blobs; 1,731 within 1 KiB–2 MB) and found no blob whose content SHA-256 equals the score's recorded runner hash. An independent auditor also confirmed that this run hash matches neither the available committed runner at the recorded source commit nor current HEAD, and that the 32-frame preflight's runner hash (`2d635336f9f2329f4118b2576e23c6282d2880f50cd08ac40c1490a1bfe223fa`) is likewise unavailable. The exact executed scorer source for the 16-frame result and exact preflight source for the 32-frame attempt therefore cannot now be recovered from local files/Git objects. Score, preflight, lock, filter, exclusion and dataset-input hashes are internally consistent, and the auditor independently reproduced the 10-record aggregates; however, strict end-to-end source provenance and the SSS bar remain unresolved. The score JSON does not directly record the candidate-filter hash (the preflight binds it), a secondary provenance limitation. These remain exploratory auxiliary results, not claim-faithfulness evidence.

**Review-inventory repair.** Independent handoff audit found that the prior 08:30 candidate inventory omitted the integrity-receipt and witnessed-release-order verifier source files, omitted its own builder and runbook, and pinned an earlier production-runner hash. The builder and runbook now include those four missing bindings, and a synthetic regression asserts their presence. The new local-only candidate inventory is `data/private_nextgqa/pre_review_inventory_v6_20261017_1020_01/pre_review_candidate_byte_inventory.json`, SHA-256 `70de3b76b351fba82572d4a5b4d7f8b5502990abdbcf9974bbd3422375e50e1a`, file mode 0600 / directory mode 0700; it records 54 present artifact roles and zero missing slots, while preserving `candidate_only`, `review_started=false`, `review_completed=false`, `joint_freeze=false`, `score_ready=false`, and `execution_authorization=not_granted`. The focused inventory suite passes 9/9, compilation and whitespace checks pass, and the independent auditor verified the inventory hashes against all five changed/current gate files and found no sensitive content fields. No review HTML or video was opened.

The three generic files were committed as `bb94c781060e1407fa6d1848fdfe791bbd09d1eb` (`Bind current review gate tools in pre-review inventory`), with only the runbook, builder, and synthetic tests staged. `git push` succeeded; direct `git ls-remote` then hit a transient DNS failure, so the GitHub API independently confirmed commit `bb94c78` is identical to `main` (0 ahead/behind) and returned file contents whose hashes matched all three local staged files. The private work log, inventory, local data, 16f score, and 32f preflight were not staged. The complete script test suite passes **1,302/1,302** after the inventory changes.

**Faithfulness progress and next action.** Direct generated-claim faithfulness remains 0/3 independently human-reviewed, 0/3 scored, with no direct estimate, CI, or test. The 10-video answer-margin result is auxiliary QA-gold-interval dependence only. The corrected inventory improves exact review-release readiness but does not create a trust-root pin, witnessed release, human labels, actual browser screenshots, 15 pair decisions, or clock calibration. Continue independent audit and source/protocol checks; retry only the frozen 32-frame sensitivity after a fresh bounded CUDA/NVML health check passes. Keep all media-derived artifacts local.

## 10:20 JST — require committed experiment code and refresh review candidate inventory

**Implementation.** Added `verify_committed_source_snapshot()` to the auxiliary margin runner. Before importing Torch or reaching any processor/model loading, it hashes the runner, decoder, and both project helper modules and compares each current byte sequence with `git show HEAD:<path>`. Any mismatch fails closed before CUDA/model work. Outcome-blind preflight records the source commit and four-file SHA map; score mode requires exact equality with that map, in addition to its prior runner/decoder/helper checks. The score now also carries candidate-pair-filter SHA and canonical digest directly rather than requiring the reader to follow only the preflight reference. A synthetic Git repository regression verifies exact committed-byte success and uncommitted mutation failure. Independent source audit approved the fail-before-model ordering and the preflight-to-score equality check; focused tests pass **20/20**, complete scripts pass **1,303/1,303**, and compile/whitespace checks pass. This is reproducibility engineering, not new faithfulness evidence.

**Publication and refreshed readiness.** Generic source/test changes were committed as `9dc6f1ab1160824ab011336716e0f255422ea38d` (`Require committed source snapshots for interval pilot`), pushed, and verified through GitHub API as identical to `main`. Because the commit changed the runner bytes, generated a fresh private candidate inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_1025_01/pre_review_candidate_byte_inventory.json`, SHA-256 `af79dbe6c72eea47f46b8a519f7ba4fd133bf4c7bed81f74c4740a4ff44b388b`, with directory/file modes 0700/0600. Independent audit verified 54 roles / 0 missing, current runner hash match, no sensitive content fields, and unchanged `candidate_only`, review incomplete, no joint freeze, `score_ready=false`, and authorization `not_granted`. The prior 10:20 candidate inventory predates the runner guard and must be treated as historical. Private inventory/log and all data/result artifacts remain local.

**GPU check and next run.** A fresh bounded elevated `nvidia-smi` still fails with `NVML: N/A`. Do not launch the old 32-frame preflight: its recorded source hash is unavailable and the newly committed source guard requires a fresh preflight with the new four-file source map. Once both NVML and CUDA device enumeration pass, rerun the same hash-locked 10-pair/32-frame outcome-blind preflight under a new unique output name, verify the same 10 media/QID assignments, then run its 50-forward score only if all gates pass. The exact previous 16-frame score source remains unavailable; the new guard prevents repeating that failure on future runs.

**Direct-faithfulness status.** Still 0/3 human-reviewed, 0/3 scored, no direct effect, interval, or statistical test. Auxiliary result remains n=10 with mean +1.025 nat on the benchmark gold-interval/control contrast; exploratory, finite deterministic cohort, no inferential test, and no generated-claim faithfulness interpretation. Human review, genuine browser capture, 15 correspondence decisions, clock calibration, and trusted freezes remain uncompleted.

## 10:22 JST — distinguish WSL GPU bridge failure from host GPU lost state

**Host diagnosis.** Elevated Windows `nvidia-smi.exe` reports `GPU is lost` and `No devices were found`, with the explicit recovery instruction to reboot the system. Windows `Win32_VideoController` and PnP still report the RTX 5070 Ti as `Status=OK` / `Problem=CM_PROB_NONE`; the disabled AMD integrated display adapter is unrelated to the requested NVIDIA device. A four-hour read-only System event query returned no matching NVDisplay/Display/WHEA warning rows. This does not identify why the NVIDIA device entered lost state. WSL `nvidia-smi` remains `NVML: N/A`, Torch device enumeration remains false, and the recent kernel ring records `dxgkio_query_adapter_info: Ioctl failed: -22`. WSL-to-Windows process interop subsequently failed with `UtilBindVsockAnyPort: socket failed 1`, so a scoped host device restart/reboot command cannot presently be launched from this session. No device reset, driver installation, or reboot was attempted.

**Decision.** GPU inference remains externally blocked by the host GPU lost state; RAM is not the limiting evidence (WSL reports about 14.5 GiB available). Do not retry scoring until the Windows host has been rebooted/recovered and both Windows and WSL CUDA health checks pass. After recovery, use the committed source guard and regenerate the 32-frame preflight for the same locked cohort; the old preflight is not valid for the new runner source contract. No direct faithfulness evidence was added; the human review/calibration gates are independently still open.

## 10:23 JST — verify no human-review handoff arrived during GPU outage

A filename/metadata-only inventory of the current private-data tree found 0 review-payload JSONs, 0 capture receipts, and 0 browser-clock observation files. It found 15 PNGs, all in the previously generated decoder-anchor export directory; those are decoder reference frames, not browser screenshots or human correspondence decisions. Therefore this checkpoint adds no claim reviews or 15-pair evidence. No live experiment process/job exists to poll: the previous 16-frame GPU job completed and the 32-frame attempt terminated in CUDA failure. WSL GPU check still fails `NVML: N/A`; Windows interop still fails with `UtilBindVsockAnyPort: socket failed 1`. The GPU lost-state and missing-human-evidence gates remain independently unresolved.

## 13:09–13:18 JST — resume check, powered-off host, and frozen-cohort mismatch

**GPU health.** An individually elevated `.venv` BF16 CUDA smoke after the user resumed work succeeded on the RTX 5070 Ti (PyTorch 2.11.0+cu128; 2048x2048 BF16 matmul; explicit synchronization; finite output; about 14.5 GiB free VRAM). This only establishes that a small CUDA context worked at that moment. The requested 32-frame pilot was not scored. Two new outcome-blind preflight attempts stopped before model loading: first, a mismatched exclusion/filter choice; second, despite restoring the old pair allowlist and prior-smoke exclusion from recorded hashes, the exact frozen cohort still failed closed with `cohort_lock_selected_assignment_mismatch`.

**Provenance discrepancy.** The old 16-frame preflight binds the 10-assignment lock SHA `3c56f569…`, pair-filter SHA `7013dc05…`, prior-smoke exclusion-file SHA `2036cfd9…`, candidate CSV/map/gsub hashes, and current sample-manifest digest. With the matching pair allowlist, the current dataset candidate builder yields zero eligible videos; without the allowlist it yields 85 videos but zero of the 10 locked media hashes. A local media-hash-only scan found those 10 lock media hashes in private reviewer-package MP4s, and none in the designated `~/nextgqa_sample` video root (533 MP4 files). These counts and hashes do not reveal identities, but they show that reusing reviewer-package copies would break the experiment's declared data-source binding. Therefore no fresh 32f preflight or score is valid from the current dataset root; do not search/match by copying review media into the inference cohort. No model or video inference ran in the two failed preflights.

**Host event audit.** Read-only Windows System log inspection after the user's forced power-off shows Event 6008 reporting the prior shutdown at 12:39:27 JST and Kernel-Power Event 41 on boot at 13:08:20. Event 41 data reports `BugcheckCode=0`, `PowerButtonTimestamp=0`, `LongPowerButtonPressDetected=false`, and `WHEABootErrorCount=0`. No WHEA-Logger, nvlddmkm, Display, or Event 1001 records appeared in the inspected four-hour window. This confirms an unclean shutdown, not its cause. The fan/unresponsive symptom plus earlier `GPU is lost`/WSL NVML failures remains consistent with a GPU/driver/host-level hang, but temperature, power delivery, and hardware condition are unverified. Keep expensive inference paused pending stable host diagnostics; no driver change, reset, or reinstall was attempted.

**Faithfulness status and next action.** Direct generated-claim faithfulness remains 0/3 independently human-reviewed and 0/3 scored, with no direct estimate, interval, or test. The prior 10-video answer-margin result remains auxiliary benchmark-interval dependence, and its executed source snapshot is unrecoverable. The attempted 32f sensitivity adds no valid result. Continue with CPU-only audit of the review/freeze readiness path and preserve all protected material locally; resume GPU inference only after the source-bound cohort is recoverable and the host has passed a stability check. Rights status remains unverified.

## 13:18–13:25 JST — recover source-bound cohort and complete 32-frame real-video sensitivity

**Archive recovery and provenance.** The designated extracted video root had no match for the 10 media hashes in the frozen assignment lock. The original local NExTVideo archive remained available; its full-file SHA-256 matched the previously recorded archive digest `1ffa3229…`. A read-only pass hashed all 5,440 MP4 members (22.89 GiB uncompressed, zero read errors) and found all 10 locked media hashes in the original archive. The matching bytes were extracted from those archive members into a mode-0700 temporary root, written mode 0600, and rehashed against the lock (10/10 exact). No reviewer-package copy was used. This resolves the source-byte mismatch while preserving the same locked assignments and original archive provenance.

**Technical smoke and preflight.** Using the newly committed current-source guard at commit `9dc6f1ab1160824ab011336716e0f255422ea38d`, an outcome-blind 32-frame one-item preflight passed and the technical smoke completed 5/5 forwards with final synchronization. Its score values were not inspected for the go/no-go decision. The complete fixed-cohort preflight then passed 10/10 distinct locked pairs, exactly 32 frames each, 10 timing passes, zero rejects, exact lock SHA match, and current four-source byte map; its private JSON is mode 0600. Immediately before the full run, RTX 5070 Ti telemetry was 43°C, 0% utilization, 24.76 W, and 14.66 GiB free VRAM. No threshold or cohort changed.

**Run and independent primary arithmetic.** The fixed 10-pair run completed all 50/50 forwards and 10/10 five-condition sets, with zero failures, successful final CUDA synchronization, and a mode-0600 score artifact `data/private_nextgqa/answer_margin_fresh10_32f_score_20261017_resume_01.json` (SHA-256 `5494c3a4de76c360f67b211cd10f78c0c3a9d04f33416a74fd5c1c9637eca739`). During inference, the GPU reached 63°C, 45% fan, about 141 W and 42% utilization, with 8,700 MiB allocated; the process exited normally. I independently recomputed the primary per-record contrast from all 10 record values and reproduced the runner summary: equal-count black control minus black gold-interval-mask answer-margin mean **+0.750 nat**, sample SD **1.858**, median **+0.125**, range **[−1.375,+5.250]**, 7 positive / 3 negative. No p-value, test, or CI was computed or reported. The corresponding 16f artifact has mean +1.025 nat (SD 2.152); paired 32f-minus-16f descriptive changes average −0.275 nat (SD 1.244; median −0.563; range [−2.250,+2.250], 4 positive / 6 negative). The lock hash matches exactly across these two artifacts, but the old 16f executed runner source remains unrecoverable, so the paired comparison inherits that provenance limitation and is exploratory only.

**Interpretation and limits.** This is a post-hoc, fixed-cohort sensitivity check of benchmark gold-interval QA answer-margin dependence. It does not measure generated-claim faithfulness, does not establish a population effect, and cannot separate frame density from the changed sample grid, mask/control geometry, and model input length. The convenience cohort is deterministic and small; black masking may be out of distribution, and the annotated player clock was not independently calibrated to decoder PTS. The old 16f source gap also prevents SSS treatment of the paired sensitivity. The source commit and every executed 32f source byte are now directly bound in the new preflight/score; the earlier old-source artifacts are not retroactively repaired.

**Faithfulness progress and next action.** Direct generated-claim faithfulness remains **0/3 independently human-reviewed and 0/3 scored**, with no direct effect estimate, interval, or test. This GPU run is auxiliary only and does not advance that numerical count. Host Event 41/WHEA checks still do not identify the earlier fan/freeze cause; a successful small run and normal temperatures now are not proof of long-term hardware stability. Keep all media and derived scores local. Do not reopen head/layer/alpha search. Continue CPU-only direct-branch audit and human-review readiness work while awaiting the genuinely external trust-root, reviewer, capture, 15-pair, and clock-calibration evidence; no experiment is justified on the available data in place of those inputs. Rights status remains unverified.

**Source-binding receipt.** Independent audit noted that score/preflight JSONs intentionally omit a local media-root path. To preserve a checkable local chain without adding any path or media bytes to the score, created mode-0600 `data/private_nextgqa/answer_margin_fresh10_32f_source_binding_20261017_resume_01.json` (SHA-256 `f7bc8c9fe0c042c24d627bef70e8aaf36f121403650abf1c678d36a9ce8171bc`). It binds the recorded full archive SHA, each of the 10 lock media hashes to its archive member and extracted-byte SHA, plus the exact lock/preflight/score SHAs and source commit. A second full ZIP read reproduced the recorded archive digest and all 10 member/extraction hashes. Independent re-audit is pending; until that review closes, treat the result as strongly hash-bound but do not label the experiment SSS. The score has no raw QA/generated text, video, or IDs; the local source receipt contains archive-relative member labels and stays protected/private. After inference, GPU returned to 45°C, 30% fan, 0% utilization, ~26 W, and 1.1 GiB used; no score process remained.

**Independent source/result audit closed.** A separate auditor verified the mode-0600 source-binding receipt against the lock, preflight, score, source commit, and all four committed source hashes. It also independently streamed the original archive, matched the full archive digest, counted 5,440 MP4 members with no read errors, and found each of the 10 locked media hashes exactly once at the receipt-bound member with matching extracted-byte hashes. The auditor therefore closed the missing source-content provenance issue. The receipt is post-run and cannot prove which filesystem path was opened, but the runner's exact lock/media-byte enforcement and the unique archive member content match bind the scored input bytes. This raises reproducibility confidence for the scoped 32f result; it does not change the post-hoc exploratory status, finite-cohort limits, rights uncertainty, or 0/3 direct claim-faithfulness status.

## 13:27 JST — post-hoc temporal-order × interval-mask interaction plan (before new outcomes)

**Question.** Does the fixed gold-interval versus equal-count-control masking contrast change when the *same 32 decoded frame pixels* are presented in reverse sequence order? The 32f contrast is already known, so this is exploratory/post-hoc. It tests only order sensitivity of this masking effect under the fixed cohort; it cannot validate generated-claim faithfulness or remove black-mask OOD concerns.

**Frozen diagnostic.** Reuse exactly the 10 media/QID pairs in the private lock SHA `3c56f569…`, same source archive/member bytes, 32-frame endpoint-inclusive sample, same Qwen revision/precision/seed, exact prompt and A–E gold-vs-best-foil next-token log-margin. Use six conditions per row: chronological full, gold-interval black mask, equal-count control black mask; and the corresponding three reversed conditions. Construct the two masked chronological frame lists first, then reverse each completed list. Thus evidence/control physical frame membership and mask fill are held fixed across orientation; only sequence order/positional placement changes. Keep processor timestamps and frame indices increasing, verify pixel tensors are exact reversals and grids/prompts otherwise agree, and counterbalance orientation/condition order by a fixed deterministic schedule. No fill/interval/cohort/metric tuning after scores. Any failed condition or missing row blocks the whole aggregate; no replacement.

**Primary quantities and stop rule.** For each row compute `E_chrono = control_margin − gold_mask_margin`, `E_reverse = reverse_control_margin − reverse_gold_mask_margin`, and `D = E_chrono − E_reverse`. Report only n, descriptive mean/sample SD/median/range/sign counts for each orientation and D; no p-value, CI, multiplicity claim, or population inference. Full-input reversal change is secondary descriptive context. First run a one-row six-forward real-video smoke and inspect only technical pass/failure status; if valid, regenerate the exact n=10 outcome-blind preflight, verify the lock and 10/10 six-grid/pixel gates, then run all 60 forwards. Expected scorer work is roughly 4–6 minutes plus model load/preflight; fixed resource guards apply. Keep all outputs private/local.

**Decision context.** This 2×3 factorial is more diagnostic of masking×order interaction than a simple full-video reversal pair, but reverse clips themselves are atypical and order/position are not separately identified. A positive D would mean the mask contrast is larger in chronology than in reverse order for this finite cohort, not proof that the model faithfully used the claim evidence. The direct generated-claim branch remains gated by human review and browser/frame calibration.


## 13:43 JST — temporal-order × mask factorial GPU result

**Execution and provenance.** Committed the generic runner and CPU contract tests in `38e39929d9ec1e2e2467810c14021f8325fa8fb1`; the working tree still has unrelated existing user files and they were not staged. The complete repository script suite passed (1,313 tests). Outcome-blind one-row six-condition smoke passed (6/6 forwards), then fixed-cohort preflight passed for 10/10 locked media and 60 planned forwards. Full run completed 10/10 complete six-cell sets, 60/60 model forwards, zero failures, and final CUDA synchronization passed. All run JSONs and receipts are mode 0600 under `data/private_nextgqa/`; no protected outputs were pushed. Score SHA-256: `33287181756839777e545f3a8c9739d05fb3e61cc1b68ab0d50a9f89c81579da`; private source-binding receipt SHA-256: `352399b2f4d6badaa51d0d2950a61d013ced756375e1a84a8d511d82c4e830ef`. Run source commit is `38e39929d9ec1e2e2467810c14021f8325fa8fb1`; cohort lock SHA-256 remains `3c56f5694372d690a016108a3c8b9e470ac983b422b9b6329e361f7141ad07cb`. Independent result audit is in progress.

**Observed fixed-cohort contrasts (n=10).** In chronological order, `control margin − gold-mask margin` mean +0.750 nat (sample SD 1.858; median +0.125; range −1.375 to +5.250; 7 positive, 3 negative). In reversed order it was +0.563 nat (SD 2.614; median −0.188; range −1.625 to +7.000; 4 positive, 6 negative). The prespecified interaction `D = E_chrono − E_reverse` was +0.187 nat (SD 1.792; median +0.437; range −4.625 to +2.125; 9 positive, 1 negative). These are descriptive calculations over the ten records only; no CI, p-value, multiplicity adjustment, or population claim was specified or computed.

**Interpretation and limits.** The point estimate is much smaller than the between-example spread, and signs vary. This run does not show a stable, practically large order interaction. It is compatible with the prior chronological mean of about +0.750 nat in this same locked cohort, while reverse order has a similarly positive but highly variable mean. Treat that as a finite-sample exploratory diagnostic, not evidence of generated-claim faithfulness. Reverse-order video is atypical; black masking may be out-of-distribution; model positional order and temporal order are not separated; the ten videos are a deterministic convenience cohort. Gold intervals are benchmark QA annotations, not explanations generated by the model. Rights status remains unverified and outputs remain local.

**GPU/runtime observation.** During the run the RTX 5070 Ti reached a sampled maximum of 74°C, 100% utilization, about 14.6 GiB VRAM use, 213 W and 44% fan. It later returned to idle around 56°C, 0% utilization and 1.1 GiB used. No CUDA failure occurred. This does not identify the earlier host freeze/fan incident's cause or guarantee stability under longer workloads.

**Faithfulness progress and next action.** Direct generated-claim faithfulness remains **0/3 independently human-reviewed, 0/3 scored**, with no direct estimate, CI, or test. This is a real-video auxiliary QA-interval dependence run and does not change that count or resolve the direct branch's reviewer/calibration gates. The primary technical result is a small, noisy, post-hoc interaction estimate. Preserve it as a negative/inconclusive diagnostic; do not tune heads or interventions from these ten rows. Next, complete the independent result audit, push only the generic committed runner/test code, and continue direct-branch readiness or another prespecified measurement that can proceed without inventing human labels.

**Independent result audit completed (13:45 JST).** A separate auditor re-read the private preflight and score, verified all source-file hashes against the committed code, recomputed every effect from stored margins, and matched the descriptive summaries above. It confirmed the preflight/source receipt/lock/score hash chain, 10/10 exact rows, frame counts, cyclic condition ordering, 60 forwards, zero failures, and successful final sync. The auditor also computed the secondary full-input chronological-minus-reversed margin: mean +0.313 nat, SD 0.826, median +0.125, range [−0.875,+2.125], 6 positive / 4 negative. The technical smoke reused the same media/QID assignment as one row in the 10-row cohort; only pass/failure status was inspected for its go/no-go, and its repeated score is not a separate sample. Generic runner/test commit `38e3992` was pushed and verified at live `origin/main`; no private logs, media, scores, or receipts were published.

## 14:02 JST — disjoint temporal-mask replication, cohort 2

**Frozen selection and provenance.** After cohort 1 showed substantial between-example spread, selected the next 10 assignments using the same deterministic candidate ordering, same 2,384-pair private allowlist, same 32-frame timing/control eligibility, and a strict union of 119 pre-existing media exclusions plus the 10 cohort-1 media hashes (129 total). Outcome values did not participate in the candidate selection. The exact new lock SHA-256 is `9af9a71a56a6922c9eab80e218423dd3c4c30c45da7b128169c582951f646c1c`; receipt SHA-256 is `c34c7d85dd4a18c88232a9dfa6633840b8d022e55b43d845805f10d513df2e39`. Recomputed the complete source ZIP SHA and matched the recorded value; each of the 10 selected media hashes matched exactly one archive MP4 member and the extracted bytes. Outcome-blind selection scanned 21 groups, skipped 11 excluded media, and found 10/10 timing-valid rows; decode failures, timing rejects, and missing interval/control pairs were all zero. The two 10-media locks are disjoint.

An initial lock attempt used a different, 79-item candidate-builder exclusion file; one assignment then overlapped the actual 119-item prior exclusion set. Its preflight was interrupted during CPU frame preparation before any model load or score. That lock and its partial attempt are not used. Cohort 2 was rebuilt under the correct 129-item exclusion union; the full outcome-blind preflight passed all 10 rows and six cells before scoring.

**Execution.** The corrected second-cohort run completed 10/10 rows and 60/60 forwards, with zero failures and successful final CUDA synchronization. Score SHA-256: `8d7c163c5705ad61ba126b5584cb2034f34bc348200a437fbdff2d3b9631ccfb`; post-run source-binding receipt SHA-256: `1bbee24f957e79a3c29dab018b56f6eb1615c008b4350a96dc25c05f8612e56a`. All artifacts remain mode 0600 under `data/private_nextgqa/`. A second independent result audit is pending.

**Observed second-cohort descriptive contrasts (n=10).** `E_chrono` mean +0.450 nat (sample SD 1.442; median +0.375; range [−1.625,+3.625]; 7 positive / 3 negative). `E_reverse` mean +0.100 nat (SD 0.820; median +0.063; range [−1.000,+1.750]; 5 positive / 4 negative / 1 zero). `D=E_chrono−E_reverse` mean +0.350 nat (SD 1.203; median +0.437; range [−1.750,+2.750]; 7 positive / 3 negative). No CI, p-value, or population inference was specified or computed.

**Combined descriptive view (n=20, exploratory only).** Across the two disjoint deterministic cohorts, `E_chrono` mean +0.600 nat (SD 1.626; median +0.250; range [−1.625,+5.250]; 14 positive / 6 negative), `E_reverse` mean +0.331 nat (SD 1.901; median −0.063; range [−1.625,+7.000]; 9 positive / 10 negative / 1 zero), and `D` mean +0.269 nat (SD 1.488; median +0.437; range [−4.625,+2.750]; 16 positive / 4 negative). The descriptive mean D remains small relative to its spread, and the most extreme negative value remains in cohort 1. The second cohort does not remove the atypical-reversal, positional-order, or black-mask OOD limitations. Cohort 2 was selected after seeing cohort-1 outcomes; although its deterministic selection rule and exclusions were fixed before cohort-2 outcomes, treat both the replication and combined n=20 view as exploratory, not confirmatory.

**GPU and faithfulness status.** During cohort 2, sampled GPU load reached 100%, ~14.6 GiB VRAM, 72°C, ~216 W and 42% fan; a later sample showed VRAM near 15 GiB, 65°C and 100% utilization. After completion the GPU returned to 56°C, 0%, about 1.1 GiB used. No CUDA failure occurred. Direct generated-claim faithfulness remains **0/3 independently reviewed, 0/3 scored**, with no direct estimate, interval, or test. These real-video runs measure benchmark QA-answer margin dependence on annotated intervals and temporal ordering only.


**Independent cohort-2 audit completed (14:04 JST).** The auditor independently matched lock, exclusion, preflight, score, receipt, source commit, source-file hashes, candidate-input hashes, and archive digest. It confirmed all ten archive members exist uniquely and their bytes match the locked/extracted hashes; all 10 assignments are disjoint from cohort 1 and all 129 exclusions. The preflight and score are complete, all effects recompute, and the six condition positions each occur ten times. The auditor also computed secondary full-input chronological-minus-reversed margin: mean −0.350 nat, SD 1.894, median −0.375, range [−4.375,+2.375], 4 positive / 6 negative. It noted the rejected `_01` preliminary lock and corrected `_02` lock differ by one assignment after the exclusion union grew from 89 to 129; `_01` has no preflight or score, and `_02` is the only analyzed cohort. The initial preflight was interrupted during CPU frame preparation before any model load or score. This correction occurred after cohort-1 results were known, reinforcing exploratory-only interpretation.

**Cohort-2 independent audit closed (14:05 JST).** The reviewer independently reproduced all cohort-2 E/D summaries, verified the 129-item disjointness and archive/source chain against the local ZIP, checked the exact commit source blobs, and confirmed 10/10 × 6 cells, 60/60 forwards, zero failures, final CUDA sync, finite values, and balanced inference order. The two cohorts' combined descriptive values above are my direct recomputation from both audited score files; no inferential test or CI was added. The only remaining critical gap is external human validation for direct generated-claim faithfulness.

## 14:24 JST — verify the real CSV source adapter and correct the protocol record

**Purpose.** Check whether actual NExT-GQA source-row binding is still missing, or whether only the protocol description is stale. This is a CPU/source-consistency audit; no claim review, model load, GPU forward, or score is involved.

**Observed result.** Ran `audit_nextgqa_actual_qa_source_binding.py` over the local test CSV, the three-row claim lineage, and the selected worksheet. It reported 5,553 CSV rows, 5,552 rows with a unique normalized gold choice, one ambiguous-gold row, 3/3 selected source rows, and 3/3 worksheet-question matches. It wrote no files and made zero model/score calls. The successful status is explicitly `local_source_binding_passed_unvalidated`: the adapter verifies exact CSV rows against caller-supplied lineage and worksheet data; it does not independently authenticate lineage, verify human claim support, or authorize scoring. Rights status remains unverified and all inputs stay local.

**Correction and checks.** Updated `docs/nextgqa_finite_claim_descriptive_protocol_v1.md`, which incorrectly said no real-CSV builder existed. It now names the existing actual-CSV adapter and aggregate auditor and scopes their limits. The focused source-adapter/auditor tests pass 9/9 with `PYTHONPATH=scripts`; `git diff --check` passes. An initial test invocation omitted that required import path and failed before test bodies ran; the corrected invocation passed. This is documentation accuracy and input-provenance progress, not direct faithfulness evidence.

**Faithfulness and next action.** Direct generated-claim faithfulness remains 0/3 independently reviewed, 0/3 scored, with no direct effect estimate, interval, or test. The GPU factorial remains a small, noisy exploratory QA-interval diagnostic, and another cohort would not remove the independent human-review or browser-to-decoder calibration gates. Continue by independently auditing the direct branch's atomic readiness checks and current inventory bindings; keep auxiliary GPU expansion closed unless a predeclared question and clean eligible cohort become available.

## 14:27 JST — begin closing the review candidate inventory's source set

**Audit finding.** Independent source review found that the candidate inventory bound the aggregate actual-CSV auditor but omitted its directly imported `nextgqa_actual_qa_source_adapter.py`. Thus the previous 54-role/zero-missing report did not cover all code on that documented source-binding path.

**Initial repair.** Added the adapter to `CODE_FILES` and to the synthetic regression's required role set. The focused inventory suite passed 9/9 and an interim 55-role inventory passed, but that inventory was superseded when recursive source-closure review found further omissions. No HTML, video, or claim text was opened for that audit.

**Scope.** This initial fix was incomplete; see the following source-closure cycle. The freezer/scorer still have no real-data orchestration caller, and a positive structural inventory cannot authorize scoring. The two audited auxiliary GPU cohorts do not unblock the direct branch. Direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored, without direct estimate, interval, or test.

## 14:29 JST — bind the complete local source closure in the review inventory

**Audit finding.** Independent recursive import review identified six additional local Python dependencies absent from the inventory: `annotation_clock_calibration.py`, `build_claim_input_manifest.py`, `claim_text_quality.py`, `necessity_choice_margin_utils.py`, `presentation_time_utils.py`, and `private_output_paths.py`. These affect calibration, claim construction/review rendering, frame sampling/masking, and private-output handling. I independently reproduced exactly these six omissions with an AST walk over the 40-file local Python import closure rooted at the inventory's source roles.

**Repair and result.** Added all six as explicit `CODE_FILES` roles and required them in the synthetic inventory regression, alongside the actual QA source adapter. The focused inventory suite passed 9/9. After the six-role expansion, the full `scripts/test_*.py` suite passed 1,313/1,313 again; `compileall`, whitespace checks, and the AST import-closure rerun (40 local Python source files, zero unmapped local imports) also passed. The authoritative private inventory is `data/private_nextgqa/pre_review_inventory_v6_20261017_1429_01/pre_review_candidate_byte_inventory.json`: 61 declared roles, zero missing slots, SHA-256 `edd8d839ef78ef1101fcb6953a4ef456f5cabf821ff4bbd60e912bbfe3aecc4f`, directory/file modes 0700/0600. An independent auditor matched all six added source hashes and sizes and confirmed the state remains `candidate_only`, all review/freeze/score booleans false, execution `not_granted`, and rights `unverified_local_only`. The interim output with the future-looking `_1445_01` label was generated during this turn, never used, and removed after verifying it contained only our superseded candidate inventory.

**Limitations and next step.** This improves byte-closure only; it does not authenticate any human identity/decision, actual browser screenshots, frame correspondence, clock calibration, or execution. The real-data production orchestration caller remains absent, and current structural components intentionally do not grant score authorization. The scoped code changes are still uncommitted because `.git` is mounted read-only (`git add` could not create `.git/index.lock`); no staged changes or index lock remain. Continue by auditing for any further source-closure omissions or a safe, non-authorizing integration improvement. Direct generated-claim faithfulness is still 0/3 reviewed and 0/3 scored, with no direct estimate, interval, or test. No GPU run was justified because another auxiliary cohort would not address the direct evidence gate.

## 21:08 JST — recheck GPU and Windows review environment

**Environment evidence.** Under individually elevated, read-only commands, `nvidia-smi` identified the RTX 5070 Ti at 55°C, 0% utilization, 1,189 MiB used / 14,807 MiB free VRAM, 27 W and 30% fan. The repository `.venv` reports Torch 2.11.0+cu128, CUDA build 12.8, `cuda.is_available()=True`, one device. An elevated PowerShell read-only query reports Edge 155.0.4283.45 and seven Edge processes. A corrected aggregate process/title query found zero Edge command lines containing the private review-page filename and zero visible windows titled `Private offline video review`. An initial `Get-CimInstance -Filter` check had malformed quoting and was discarded; the corrected queries succeeded. The normal sandbox still lacks `/dev/dxg`, and normal WSL-to-Windows process interop returned `UtilBindVsockAnyPort: socket failed 1`; individually elevated PowerShell was available. No Edge process was opened, closed, or controlled, and no browser capture occurred.

**Decision.** GPU hardware is currently available, but no authorized, scientifically valid direct claim-score run exists before independent review, real screenshot/frame evidence and calibration, trusted release/freeze, and production all-row gate. A further run of the prior QA-gold-interval endpoint would repeat an exploratory diagnostic without addressing those blockers. Keep the GPU idle for this checkpoint. The changed Windows interop provides a possible route for later browser work after the trusted release prerequisites; it does not by itself authorize exposing the review page. Direct faithfulness remains 0/3 independently reviewed, 0/3 scored, with no direct estimate, interval, or test. Rights remain unverified and protected outputs local.

## 21:15 JST — close the `from scripts import` source-closure edge case

**Finding and repair.** The independent inventory audit found that Python syntax `from scripts import local_module` was not handled by the static source-closure walk. No registered source currently uses that syntax, but an unregistered `scripts/local_module.py` could otherwise escape detection. The checker now fails closed for `from scripts import ...`, and a synthetic negative regression creates that exact local module/import pair and requires the inventory to retain `candidate_only` while reporting `python_source_import_closure` missing.

**Verification.** The focused inventory suite passed 13/13 with `PYTHONPATH=scripts`. The full `scripts/test_*.py` discovery passed 1,317/1,317 in 24.3 seconds; expected negative-test CLI diagnostics appeared in captured test output. `compileall` and `git diff --check` passed. One initial unittest invocation without `PYTHONPATH=scripts` failed during module discovery, before tests ran; the corrected invocation above passed. A fresh local-only inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_2114_01/pre_review_candidate_byte_inventory.json` has SHA-256 `65e008cdc1b34dc3e956405fd17f12bc7db0a3823472d79cc3ac2c24b148891a`, 61 roles and zero missing slots, with directory/file modes 0700/0600. It remains `candidate_only`; review started/completed, joint freeze, and score-ready are all false; authorization is `not_granted`; rights are `unverified_local_only`. Independent audit of this latest inventory and edge-case fix is pending.

**Scope and next action.** This closes one static import spelling, not every possible dependency path: dynamic imports, subprocess/runpy targets, non-Python assets, and modules outside the flat `scripts/*.py` layout still require explicit inventory roles. No claim/review material was opened and no GPU inference ran. The direct generated-claim faithfulness count remains **0/3 independently reviewed, 0/3 scored**, with no direct effect estimate, interval, or test. Continue the independent audit and then inspect the next production-readiness gap; do not treat source-byte closure as review or score authorization.

## 21:16 JST — close bare `import scripts` namespace access

**Finding and repair.** Reviewing the import-closure walk's known edge cases showed that bare `import scripts` can expose the namespace package and allow attribute-based access to local modules without an explicit `scripts.foo` import. No current registered source uses this form, but the checker previously ignored it. The walker now marks bare `import scripts` unmapped, and a synthetic regression verifies that an alias followed by local-module attribute access blocks the inventory while leaving it candidate-only and non-authorizing.

**Verification.** Focused inventory tests passed 14/14; full repository test discovery passed 1,318/1,318 in 24.1 seconds; `compileall` and `git diff --check` passed. The new inventory is `data/private_nextgqa/pre_review_inventory_v6_20261017_2115_01/pre_review_candidate_byte_inventory.json`, SHA-256 `4919871a17f70a1be179e4f74008fd3263dccca06a0eca4aa42f0d7b14a46288`, 61 roles, zero missing slots, directory/file modes 0700/0600. State remains `candidate_only`, no review/freeze/score readiness, execution `not_granted`, rights `unverified_local_only`. Independent audit is pending.

**Limitations and next action.** This still does not resolve dynamic imports, subprocess/runpy targets, local files outside the flat scripts directory, or non-Python runtime dependencies; the inventory docstring/run log describe those limits. It does close both `from scripts import ...` and namespace-package `import scripts` spellings. No human review or inference occurred, and direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored without an estimate, CI, or test. After audit, continue by tracing concrete runtime dependencies or another review-readiness defect, while preserving the human and calibration gates.

## 21:19 JST — resolve dynamic imports and repository-root module ambiguity

**Audit input.** The independent review of the 21:17 inventory passed all 61 roles and 40 declared code-file hashes, and confirmed the inventory was private, candidate-only, and non-authorizing. It also identified an ambiguity in Python import resolution: a bare import could resolve to either a registered `scripts/<name>.py` or a same-named module at the repository root when the root is on `sys.path`.

**Repair.** The static closure walk now fails closed on such root-name collisions before accepting the registered script. It resolves literal `__import__` and `importlib.import_module` arguments through the same local-module checks and blocks nonliteral dynamic import expressions. New synthetic cases cover a dynamic local module, variable-valued dynamic import, and collision between a registered script and repository-root module. A separate AST scan of all registered Python sources found no explicit imports resolving to repository-root Python modules outside `scripts/`.

**Verification.** Focused inventory tests passed 17/17. Full repository test discovery passed 1,321/1,321 in 28.1 seconds; compilation and whitespace checks passed. Fresh inventory: `data/private_nextgqa/pre_review_inventory_v6_20261017_2118_01/pre_review_candidate_byte_inventory.json`, SHA-256 `504c69e7baab58b06efaa5e35efd561eb05dee6d1032d20015378bac002c3520`, 61 roles / zero missing, directory/file modes 0700/0600, candidate-only and not-authorizing. Independent audit is pending.

**Limits and decision.** Static Python import scanning is still not a general runtime dependency resolver: subprocess/runpy targets, environment-driven imports, and non-Python assets must be inventoried explicitly. No direct claim scoring, review-page opening, or GPU inference occurred. Direct generated-claim faithfulness remains 0/3 reviewed and 0/3 scored with no direct effect estimate, CI, or test. The human review and 15-pair clock/frame correspondence plus trust-root/witnessed-release requirements still gate the real-video scoring branch.

## 21:20 JST — fail closed on aliased dynamic-import functions

**Independent finding.** Audit of the prior version found that assigning `importlib.import_module` or `builtins.__import__` to another variable before calling it bypassed the dynamic-import recognizer. No such alias chain occurs in the current declared source set, but it could hide a local module dependency in a future source revision.

**Repair and checks.** The AST walk now recognizes direct builtins and importlib dynamic calls, resolves literal module strings, and marks variable-valued calls or non-call references/aliases to importer functions as unmapped. Tests cover importlib module aliases, a renamed `import_module` function, a renamed `builtins.__import__`, variable arguments, and literal local-module resolution. Focused suite passed 17/17; full repository suite passed 1,321/1,321 in 25.6 seconds; `compileall` and `git diff --check` passed. New inventory `data/private_nextgqa/pre_review_inventory_v6_20261017_2119_01/pre_review_candidate_byte_inventory.json` has SHA-256 `ad77c05c67f8383e22d7a03e460c21dcb1344e54b89ce1966e3061dbcb64d1d2`, 61 roles / zero missing, 0700 directory / 0600 receipt, and remains candidate-only, not review- or score-ready, `execution_authorization=not_granted`, `rights_status=unverified_local_only`. Independent audit is pending.

**Remaining boundary.** Arbitrary reflection (`getattr`/`eval`), subprocess/runpy script targets, and environment-driven loading cannot be exhaustively proven absent by this AST walk. They remain explicit source-audit/inventory responsibilities; this code is a conservative static guard, not proof of all runtime behavior. Human review, independent trust/witness evidence, and 15 screenshot-to-decoder correspondences remain the practical gates. Direct faithfulness remains 0/3 reviewed and scored, with no direct estimate, interval, or test.

## 21:23 JST — bound reflected/relative dynamic imports and reject synthetic no-op pixel envelopes

**Independent source-closure audit and repair.** The review of the 21:19 inventory found two remaining recognized-importer forms: relative `import_module('.helper', package=...)` targets, and calling an importer retrieved through `getattr`. The checker now marks leading-dot targets incomplete, treats direct reflective retrieval of `import_module`/`__import__` as incomplete, and the docstring now explicitly limits its guarantee to recognized forms rather than arbitrary reflection. Regressions cover both cases. Arbitrary reflection such as `vars()`/`operator.attrgetter`, subprocess/runpy script targets, and non-Python runtime dependencies remain outside this static check; no current direct-chain source uses the specifically audited forms.

**Synthetic preflight consistency repair.** A separate source audit found that `run_nextgqa_claim_mask_feasibility_v2.py` validated only pixel-hash syntax while its positive test expected three distinct condition pixel hashes. The validator now rejects any duplicate of the three declared per-row pixel hashes before returning a structural pass. Tests cover all-three-equal and a two-equal/one-different pair; in both cases the whole candidate remains blocked with zero loads, zero forwards, zero score calls, null D3, `production_ready=false`, and authorization `not_granted`. This checks only declared synthetic hashes; it neither verifies processor bytes nor proves masking correctness. A real row whose masked pixels are already identical could also be rejected, which is consistent with using this synthetic guard to detect no-op condition inputs but must be reconsidered against actual hashed tensors before production integration.

**Verification.** Combined focused suites passed 30/30. Full repository discovery passed 1,322/1,322 in 25.8 seconds; `compileall` and `git diff --check` passed. Current private candidate inventory: `data/private_nextgqa/pre_review_inventory_v6_20261017_2123_01/pre_review_candidate_byte_inventory.json`, SHA-256 `610390aad0efe7b9a2b78a612063c7b1dd3df96f9bb1564654cedeb6c9c9ecf0`, 61 roles / zero missing, modes 0700/0600, candidate-only and not-authorizing. Independent review of the final source closure/inventory and the pixel-hash gate is pending.

**Research status.** These fixes strengthen readiness diagnostics only. No human review, trust-root/witnessed release, frame/clock calibration, or model scoring has occurred. Direct generated-claim faithfulness is still 0/3 reviewed and 0/3 scored, with no direct estimate, CI, or test. The two prior exploratory GPU cohorts remain the only numerical evidence on the auxiliary QA-interval endpoint; no GPU job is justified while it cannot address the missing direct-evidence gate.

## 21:25 JST — independently audit processor-level condition pixels

**Reason.** The synthetic envelope guard checks only caller-declared hash inequality. To test the concrete code that materializes Qwen batches, extended the CPU-only `build_exact_triplet_batches` output audit: after hashing all nine processed batches, it requires each row's three `pixel_values_videos` tensor digests to be distinct, while the existing check continues requiring every nonpixel input descriptor to match across conditions. Thus an incorrectly ignored mask that leaves the actual processor tensors identical now blocks construction before model loading/scoring.

**Test and result.** Added a fake processor mode that returns the same zero tensor for each condition; the batch builder raises `record_<n>_condition_pixel_tensors_not_distinct` after 9 batch constructions. The ordinary synthetic batches still pass all mask-rank, prompt, metadata, and pixel-reconstruction checks. The focused condition-builder and synthetic-preflight suites passed 20/20; full repository suite passed 1,323/1,323 in 27.0 seconds. Compilation and whitespace checks passed. No model was loaded and no model forward or score call occurred.

**Limits.** This verifies exact processed tensor inequality only for the synthetic fixtures. A real video whose masked frames are already all black, or whose processor maps distinct frames to identical tensors, would be blocked as a no-op; the rule does not establish semantic adequacy of a mask or claim. Independent audit of this change is pending. Latest inventory `data/private_nextgqa/pre_review_inventory_v6_20261017_2124_01/pre_review_candidate_byte_inventory.json` has SHA-256 `1447d60854759108053607c6c8823a24f2ead72c5e386870bd1444dbdc545596`, 61 roles, zero missing, mode 0700/0600, still candidate-only and non-authorizing.

**Faithfulness status.** There is no increase in direct claim-faithfulness evidence: still 0/3 human-reviewed and 0/3 scored, with no direct effect estimate, confidence interval, or test. The work hardens a pre-model quality gate; real review/custody, trust root and witnessed release, and 15 browser-clock/decoder-frame correspondences remain unresolved.

**21:26 JST independent audit closure.** The auditor matched the 2124 inventory digest `1447d60854759108053607c6c8823a24f2ead72c5e386870bd1444dbdc545596`, confirmed 61 roles/zero missing and all 40 current source hash/size pairs, plus private modes and non-authorizing flags. It independently verified that the processor-level check uses the exact `tensor_sha256` digest from typed tensor descriptors, including content bytes, dtype, and shape; the synthetic no-op test blocks after all nine processor calls and before any model/scoring path. It reiterated that equal tensors may occur for already-black masked regions, so this is a conservative no-op guard and not condition-validity evidence. The prior synthetic envelope audit was also confirmed: duplicate declared hashes block, with zero loads/forwards/scores, null D3, and authorization not granted.

## 21:28 JST — enforce distinct pixel tensors again at atomic freeze

**Motivation.** The processor-level builder rejects no-op conditions, but the atomic freezer can be called directly with a different producer. To prevent bypassing that useful invariant, the freezer now selects the known pixel field (`pixel_values_videos` in Qwen batches; `pixel_values` in its CPU fixture), rehashes each CPU snapshot's typed tensor, and requires three distinct per-row values before invoking the authenticated final-gate callback. Missing or ambiguous pixel fields also fail closed. This adds one additional tensor-only digest pass per condition after the existing complete-batch hash and preserves the existing precheck-before-materialization ordering.

**Verification.** A negative atomic-freeze test uses three identical condition tensors and recomputes their expected full-batch hashes; freeze rejects row 1 after its three condition snapshots, before the final gate, and returns no frozen bundle. The normal fixture now gives each condition distinct pixel values and retains all existing freeze/scorer integration checks. Focused atomic, condition-builder, and synthetic-validator suites passed 39/39. Full repository suite passed 1,324/1,324 in 26.5 seconds; compilation and whitespace checks passed. Fresh inventory `data/private_nextgqa/pre_review_inventory_v6_20261017_2128_01/pre_review_candidate_byte_inventory.json` has SHA-256 `8ce7c42d3672bb4e28a8b0ba72b008f8b0d175980695840bff2fe64ae26a714c`, 61 roles/zero missing, 0700/0600, candidate-only and not-authorizing. Independent audit completed: the 61-role inventory and 40 current source hash/size pairs matched; the auditor verified that production Qwen's `pixel_values_videos` is checked using the typed tensor digest, and that duplicate pixels block before the authenticated final-gate callback. The audit also confirms the intended limitation: legitimate identical tensors (for example, already-black evidence regions) are rejected; this remains a no-op guard, not proof of mask or claim validity.

**Limits.** This compares digest equality on exact CPU pixel tensors; it does not certify source media, mask ranks, evidence semantics, or score authorization. Already-black/effectively equal masked tensors will block the row. The change adds CPU hashing cost for three pixel tensors per row; no CUDA or model scoring was run. Direct generated-claim faithfulness remains 0/3 human-reviewed and 0/3 scored, with no direct estimate, interval, or test.

## 21:29–21:33 JST — bind frame verification to Qwen's configured resize policy

**Independent audit finding.** After the atomic freezer audit closed, an independent reviewer found a geometry gap in `nextgqa_condition_batch_builder.py`: the expected-frame reconstruction used the spatial shape returned by the vision utility as its own target shape. A modified or misconfigured utility could therefore resize the same frames to a different geometry and have that geometry validate itself. This was a code-path finding, not evidence that the pinned production utility currently does so.

**Repair.** Expected output geometry is now derived independently from the source frames and the installed `qwen_vl_utils.vision_process` policy: frame-list image preprocessing, `FRAME_FACTOR`, `VIDEO_MIN_TOKEN_NUM`, `VIDEO_MAX_TOKEN_NUM`, `MODEL_SEQ_LEN`, and `smart_resize`. The observed vision tensor must match those exact reconstructed pixels and shape. Added a regression where a 364×364 resize is otherwise a valid factor-28 geometry under the pixel budget but differs from the utility-derived target; it must fail before the processor is called.

**Verification.** The focused condition-builder suite passed **8/8**. The pinned local Qwen processor smoke passed **3 synthetic records / 9 batches**, exact frame-pixel identity verified, manifest verified, CUDA uninitialized, no weights loaded, **0 model forwards**, and authorization `not_granted` (Transformers 5.18.0, pinned model revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`). Full repository test discovery passed **1,325/1,325** in 27.2 seconds; `compileall` and `git diff --check` passed. The processor emitted its existing deprecation warning about `min_pixels`/`max_pixels`; this did not affect the smoke result.

**Private candidate inventory.** Rebuilt after source changes at `data/private_nextgqa/pre_review_inventory_v6_20261017_2132_01/pre_review_candidate_byte_inventory.json`; SHA-256 `7b09ad8b5e7fefadabb93c3b9aab6b897cb534c9969a3f03808afb90ed260476`, 61 roles, zero missing, directory/file modes 0700/0600. It remains `candidate_only`, review/freeze/score flags false, execution `not_granted`, rights `unverified_local_only`.

**Interpretation and next step.** This closes one preprocessing self-validation path and verifies the synthetic path against the locally cached Qwen processor; it is not real-video validation and supplies no faithfulness score. Direct generated-claim faithfulness remains **0/3 independently human-reviewed, 0/3 scored**, with no direct effect estimate, interval, or test. Human claim review, independent identity/custody evidence, witnessed trust-root release, and 15 browser-clock/decoder correspondences remain unresolved. No GPU run was justified by this geometry fix. Next, independently audit the newly bound resize calculation and determine whether the existing production preflight can consume a fixed shape contract without duplicating utility internals.

**Independent audit and publication (21:33 JST).** The auditor matched the installed `fetch_video` frame-list path for the actual fixed contract: 32 equal-size RGB PIL frames and only `sample_fps` metadata. The helper invokes the installed `smart_resize` directly, so it shares that version's rounding behavior. Audit caveat: if future prompt construction adds `resized_height`, `min_pixels`, `total_pixels`, or `max_pixels`, the reconstruction must bind those parameters too; today's internal prompt builder supplies none of them. Committed only the eight generic Python source/test files as `f96b6d7c89ce9feacb0496d1e81051d801c310ee` and pushed to `origin/main`; live `git ls-remote` returned the exact commit. All local data and logs were excluded. The private work log and inventory remain local. Worktree still contains unrelated user changes and untracked inputs, all left untouched.

## 21:34–21:36 JST — rerun the changed input path through bounded GPU integration

The geometry change affects tensors passed to the scorer, so after the CPU check I ran the existing synthetic nine-forward GPU smoke once to verify the changed builder and real pinned model remain integrated. Before load, an individually elevated read-only check found RTX 5070 Ti with 14,948 MiB free of 16,303 MiB, 54°C, and WSL `MemAvailable` 14,284,124 kB; `.venv` reported Torch 2.11.0+cu128, CUDA available, BF16 supported. This met the runner's 12 GiB load guard. The GPU script uses three generated RGB clips and invented QA, network disabled, local checkpoint revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16/4-bit BF16 compute.

**Observed result.** `PYTHONPATH=scripts .venv/bin/python scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py` passed: 3/3 synthetic triplets and 9/9 forwards, no failure code, 30.937 seconds scoring-phase time. Peak allocated/reserved VRAM was 10.744/12.473 GiB, minimum free before a model call 1.871 GiB, and free before load 14.657 GiB. After process exit, read-only `nvidia-smi` showed 14,867 MiB free and host `MemAvailable` 14,250,956 kB. The runner reports `faithfulness_result=false`, `execution_authorization=not_granted`; no per-record scores were emitted or analyzed. This was a bounded engineering integration check, not a real-data experiment.

The result is summarized without local paths or private content in `experiments/2026-10-17_qwen_preflight_geometry_gpu_smoke.md`. That public-safe summary is queued for scoped commit/push. Do not repeat the synthetic cohort: the condition-builder fix is now tested through CPU tests, the real pinned processor, and the GPU scorer. Direct generated-claim faithfulness remains **0/3 independently human-reviewed, 0/3 scored**, no estimate/interval/test; only the separate auxiliary QA-interval endpoint has earlier exploratory estimates.

**Provenance follow-up.** Recorded the installed qwen-vl-utils 0.0.14 `vision_process.py` SHA-256, torchvision 0.26.0+cu128, and effective resize constants (`MODEL_SEQ_LEN=128000`, 128/768 min/max video tokens per frame, frame factor 2) in the public summary. A source search also confirmed the condition builder is called by the synthetic CPU/GPU smoke scripts and tests only; no production real-media orchestration caller exists. This narrows the smoke's scope further: it validates the software integration fixture, not the production scoring path. The production caller must bind the concrete gates and pass a private real-media smoke once human review/capture evidence exists. Updated the safe public summary for a second scoped publication.

## 21:39–21:45 JST — add non-authorizing signed review-export envelope

**Plan and independent audit.** The v6 handoff validator checks the frozen policy, exact review/index byte bindings, row tokens, declarations and interval structure, but always returns `semantic_review_authenticated=false` and `execution_authorization=not_granted`. A design review found that the completed export itself had no reviewer signature, while the existing pre-review signatures do not bind post-review output. Preserve frozen v6 bytes and add only an optional detached Ed25519 envelope.

**Implementation and repair.** Added `scripts/nextgqa_completed_review_attestation.py` and synthetic tests. The first audit caught a gap: a minimal JSON object claiming `review_completed=true` could be signed despite missing all three review rows, and exponent overflow in unknown fields could parse as infinity. The module now requires the existing structural handoff audit to pass, checks exact worksheet/lineage/review/index parsed-byte equality, binds all four hashes plus the pre-review receipt digest and reviewer key ID/fingerprint into a domain-separated signature, and recursively rejects nonfinite values. Reports state that receipt, roster, source-media custody, person identity and semantic review remain unverified; execution remains denied. The module explicitly leaves receipt digest and reviewer key identity/fingerprint verification to an out-of-band trusted workflow.

**Verification and independent audit.** Focused tests passed 10/10; full script test discovery passed 1,335/1,335 in 25.5 seconds; compilation and whitespace checks passed. An independent auditor re-ran the 10 focused tests and verified the bindings and false authorization/authentication flags. Synthetic temporary keys and fixtures only; no private candidate, real reviewer export, existing key, protected video or GPU model was accessed. Published the scoped code/tests/public-safe log as commit `d1d9cfed5f0c6c184fd591be60fc7b1dc827becd`; live `git ls-remote` matched the commit, and the fetched remote log blob SHA-256 matched the local file (`45bfde2e2aa49e28a51224ca272048e0985cb358f114949af5845f214bb1522f`). No unrelated tracked or untracked files were staged.

**Faithfulness and limits.** This is review provenance plumbing only. It does not complete or replace human review, establish key-to-person identity, prove viewing, verify the separate on-disk media custody manifest, validate actual review semantics, or create the production all-row score-freeze gate. Direct generated-claim faithfulness remains **0/3 independently human-reviewed, 0/3 scored**, with no direct estimate, CI or test. The witnessed release and 15 browser-to-decoder correspondence gates remain unresolved; rights are unverified and private materials remain local. No GPU run was justified: scoring still cannot answer the direct research question validly before those gates pass.

**Next action.** Keep this helper non-authorizing. Before any real use, establish the reviewer key fingerprint and pre-review receipt through the trusted roster/release workflow, then run it only on a real structurally valid export after the witnessed human review. Continue independent preparation of the production gate and human handoff; do not score until the upstream evidence is complete.

## 21:46 JST — make the private attestation usable from the operator CLI

**Reason.** The API-only helper required a Python caller and could not directly support a reviewer/custodian's local handoff. Added `sign` and `verify` subcommands, both using the repository's no-symlink, owner/mode-checked private reader. The CLI requires 0600 input files and writes no-overwrite outputs under 0700 directories, prints sanitized status only, and retains all false authentication/authorization declarations. It accepts the expected receipt/key/lineage values as explicit external pins; it still does not verify their upstream trust chain.

**Checks.** Focused tests pass 11/11, including CLI round-trip, output mode, overwrite refusal, bad pins, malformed data, and gate failure. Full repository rerun is in progress. No real review, protected content, existing keys, GPU, or model were used. Direct generated-claim faithfulness is still 0/3 reviewed and 0/3 scored; no effect estimate, CI, or test.

**Next.** Finish full tests, independent audit of the CLI/permission behavior, update the public experiment summary, and publish scoped changes. This improves the eventual private review handoff but does not make the missing external human and witness actions happen.

**21:47–21:50 JST completion.** CLI-focused suite passed 12/12, including real CLI sign→verify on synthetic files, 0600 output, refusal to overwrite while preserving existing bytes, and final-component symlink rejection. Full repository discovery passed 1,337/1,337 in 25.4 seconds; compileall and whitespace checks passed. Independent audit confirmed input-to-API argument mapping, mode checks, no-overwrite behavior, sanitized summaries, and all authorization flags remain denied. Updated the privacy-safe public experiment log. No model/GPU run was performed because direct scoring remains invalid before human and trust evidence exists.

**Publication (21:50 JST).** Committed the CLI and privacy-safe experiment summary as `afb0818b2ab540ef2578cc6aa10c5c58a035c3d0`, pushed to `origin/main`, fetched the remote branch, confirmed `git ls-remote` returned the same commit, and verified the remote experiment-log blob SHA-256 equals local `e7109b906cafde3ee7735f91cabfabba0dd02d19f42496347e8023a6dd965266`. All unrelated preexisting modifications and local data remain unstaged.

## 21:50–21:53 JST — connect the reviewer signature handoff to freeze inputs

**Finding and repair.** The new attestation helper had not been named in the human-review runbook or the 61-role candidate inventory. The runbook now explains when to use `sign`/`verify`, that the reviewer key fingerprint and receipt digest must come from the trusted release workflow, and that this signature does not verify identity/viewing/custody or permit scoring. Added the attestation script to the inventory builder's required code closure and a regression assertion that the role remains included.

**Verification.** Inventory-builder tests passed 17/17; attestation tests passed 12/12. A fresh inventory was built at `data/private_nextgqa/pre_review_inventory_v6_20261017_2151_01/pre_review_candidate_byte_inventory.json`: 62 slots, zero missing, candidate-only, all review/freeze/score flags false, execution `not_granted`, rights `unverified_local_only`, mode 0700/0600. It binds the attestation script at 16,350 bytes and SHA-256 `caa492b3dc364abbe902239babd334cfd55209e7a29985051c2a47c8cdcf1c46`; independent audit matched that entry without opening private candidate payloads. Full script discovery passed 1,337/1,337 in 25.7 seconds; compilation and whitespace checks passed. No private review/media payloads or keys were opened.

**Publication and status.** The privacy-safe experiment log and handoff documents plus generic inventory-builder change were committed/pushed as `9f0dc2a96a79144dc986b71d021258b9a6d1ee40`. Live `git ls-remote` matched; the fetched public experiment-log hash equals local `33738eb5b1c28d12b8e6dc6c15deece6d245c6a47903e8df2453af0e7c196bf3`. The private candidate inventory is excluded from publication. Direct generated-claim faithfulness remains 0/3 human-reviewed and 0/3 scored; no direct effect estimate, CI, or test. No GPU job was justified because current inputs cannot pass the actual human/trust/timing gates.

## 21:53–22:00 JST — bind the reviewer signature to the signed role roster

**Plan and finding.** Independent audit verified that v1 of the completed-review envelope accepted caller-supplied reviewer ID/fingerprint; it authenticated a selected key, not its rostered claim-reviewer role. The existing pre-review receipt module already verifies a root-signed role roster against an externally pinned root, so use that established verifier and keep the frozen review package untouched.

**Implementation.** The new v2 envelope derives the claim-reviewer key ID and public key from the root-signed roster, checks the signing key pair against that entry, and binds roster bytes and root fingerprint into the signature. CLI inputs now require the roster, root public key, and root DER fingerprint; direct reviewer-ID/fingerprint inputs were removed. Verification reports roster-signature/role validity while retaining false flags for receipt verification, person identity, semantic review, and media custody; execution remains `not_granted`.

**Checks and independent audit.** Focused invocation passed 29/29 (12 attestation tests plus the 17 imported integrity-receipt tests). Full discovery passed 1,354/1,354 in 27.9 seconds, `compileall` and `git diff --check` passed. Independent reviewers read the code and tests, reproduced the focused pass, and found no critical defect. Their limitation is material: they did not inspect real release keys/artifacts, and the receipt digest remains unverified. The out-of-band root pin is still essential; a caller choosing both root and pin can self-create a roster.

**Private inventory.** Rebuilt after source changes at `data/private_nextgqa/pre_review_inventory_v6_20261017_2158_01/pre_review_candidate_byte_inventory.json`; 62 roles, zero missing, candidate-only, all review/freeze/score flags false, execution denied, rights unverified-local-only, mode 0700/0600. Inventory SHA-256 `dc4b1d4f587703b3a46daa816a7bf438d23e5ec2d281a87f591a053f7f88fcbe`. Its `completed_review_attestation` role binds 17,389 bytes, SHA-256 `3987052283985f59daf271ff9627292ee9940b8f7f76410cca6b4186687f06e0`. Private inventory remains unpublished.

**Faithfulness checkpoint.** Direct generated-claim faithfulness remains 0/3 independently reviewed and 0/3 scored, no effect estimate/CI/test. This change is trust-plumbing, not scientific evidence. No GPU experiment is currently justified before trusted human-review and frame-correspondence prerequisites. Next inspect the live release-order and browser/decoder capture path for a concrete, locally resolvable missing artifact; do not treat synthetic results as release evidence.

## 22:00 JST — establish whether real review prerequisites exist

**Independent metadata audits.** Three independent read-only audits examined the release verifier/runbook, private path metadata, and browser-clock/decoder pipeline. They agree: there is no externally pinned trust root, signed integrity or release roster, freeze-owner receipt, witnessed release attestation, or complete real artifact-source set. The 62-slot inventory from 21:58 is candidate-only, not evidence that a freeze occurred. Separately, the repository has 15 planned decoder PNGs and plan/provenance, but no screenshot, observation, capture receipt, crop, pair-review export, or signed evidence manifest; verified correspondences are 0/15. Human operator interaction with the active Windows browser and a separate human's blinded visual match review are required. The capture executable is in the repo, but its operation in the current WSL/desktop state was not established (prior validate/permission smokes do not count).

**Documentation/inventory follow-up.** Updated the human-review runbook to describe attestation v2's root-roster inputs accurately and clarify that its caller-supplied receipt digest is not receipt validation. Rebuilt the local inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_2206_01/`; it has 62 roles, zero missing, candidate-only, all review/freeze/score flags false, `execution_authorization=not_granted`, `rights_status=unverified_local_only`, 0700/0600. Inventory SHA-256 `2ba107edfff577ac466fd16f664ac686dfafe4b1e9fa45ea2c72c9c608994cb0`; attestation code source remains 17,389 bytes, SHA-256 `3987052283985f59daf271ff9627292ee9940b8f7f76410cca6b4186687f06e0`.

**Progress and blocker.** Direct generated-claim faithfulness is 0/3 human-reviewed, 0/3 scored, no estimate/CI/test. Frame correspondence is 0/15. No GPU run is scientifically justified while the unopened package has no witnessed release and human review data are absent. The immediate block is external: trusted human identities/keys and witnessed release must precede opening; then interactive capture and independent pair review must occur. Continue low-risk preparation and make the operator steps concrete, but never count synthetic work as research validation.

## 22:04 JST — audit the v2 construct against the thesis endpoint

**Verified design.** Read the v2 protocol and scorer directly: the generated claim is not part of the QA score prompt; it selects the frames for a black-mask intervention. The outcome is gold-minus-best-foil next-token margin. A complete result would therefore describe the effect of masking claim-selected visual evidence on QA margin for these three exact records, not whether the model used or faithfully expressed the claim. The foil can change by condition, black fill may be OOD, and the control does not match semantic information. The protocol records these caveats and correctly disallows inferential/population claims for n=3.

**Literature audit.** Checked primary ACL publications. Atanasova et al. (ACL 2023) propose tests that manipulate counterfactual reasons absent from an NLE and reconstruct input from generated NLE reasons; the object is the explanation content. Their ACL 2024 discussion distinguishes causal output dependence from whether explanation text faithfully represents useful prediction factors. Source links are recorded in the public construct-validity audit. These papers support the conceptual distinction, not an untested adapted metric.

**Decision and status.** No v2 outcomes exist, so this is pre-outcome, not result-driven. Keep current v2 unchanged only as a finite `claim-aligned visual-evidence dependence` feasibility study; if the thesis needs a direct explanation-faithfulness claim, draft and freeze a separately versioned outcome-blind protocol before score access. The target construct (textual rationale dependence, visual evidence attribution, or both) cannot be inferred from the current QA-margin outcome. Head search remains closed. Direct generated-explanation faithfulness is `not_measured`; claim review 0/3, scoring 0/3, no effect/CI/test. Human/trust release and frame correspondence remain external blockers (0/15).

**Publication.** Published the primary-source-backed construct audit and revised overview as `1fa5fe543bd35b43a23edf1151e2d3f43ed384e8`. Live `origin/main` matched, and both remote/local construct-audit blobs matched exactly. The privacy-safe audit is public; no raw claim, reviewer, video, or private inventory was staged. The v2 outcome protocol remains unchanged.

## 22:14–22:15 JST — primary recomputation of the auxiliary 20-video GPU result

**Audit method.** Re-opened only the private mode-0600 score, preflight, lock, and source-binding JSONs for the two temporal-mask factorial cohorts. Did not open any video, raw question/answer, claim, or media path. Checked score/preflight/lock hashes against each local source-binding receipt, confirmed the recorded runner source hashes against Git commit `38e39929d9ec1e2e2467810c14021f8325fa8fb1`, verified 10/10 records and 60/60 forwards in each run, zero failures, final CUDA synchronization, and null CI/p-value with test=`none`. Confirmed that the two lock files contain 20 unique media hashes. Independently recomputed effects from the stored condition margins, then checked them against each record's stored effect fields.

**Primary recomputation.** Cohort 1 (n=10): chronological control-minus-gold-interval-mask mean +0.750 nat (SD 1.858; median +0.125; range −1.375 to +5.250; 7 positive/3 negative); reversed-order mean +0.563 nat (SD 2.614; median −0.188; range −1.625 to +7.000; 4 positive/6 negative); interaction D mean +0.1875 nat (SD 1.792; median +0.438; range −4.625 to +2.125; 9 positive/1 negative). Cohort 2 (n=10): chronological +0.450 nat (SD 1.442; median +0.375; range −1.625 to +3.625; 7/3); reversed +0.100 nat (SD 0.820; median +0.063; range −1.000 to +1.750; 5 positive/4 negative/1 zero); D +0.350 nat (SD 1.203; median +0.438; range −1.750 to +2.750; 7/3). Combined exploratory n=20: chronological +0.600 nat (SD 1.626); reversed +0.331 nat (SD 1.901); D +0.26875 nat (SD 1.488; median +0.438; range −4.625 to +2.750; 16 positive/4 negative). No inferential test, p-value, confidence interval, or population inference was computed.

**Interpretation.** This confirms the arithmetic and local source-binding chain for the previously audited GPU runs. The observed interaction is small compared with row-to-row spread; signs are mixed. Cohort 2 was chosen after cohort 1 outcomes were seen, so the combined view is post-selection exploratory, not confirmatory. These runs measure benchmark gold-interval QA-margin dependence under chronological/reversed video order. They do not use model-generated explanations or claims and cannot establish explanation faithfulness. This is a primary audit, not new GPU evidence. Rights remain unverified and all source/result material stays local.

**Separate literature conclusion.** Primary ACL work supports tests that intervene on stated reasons and compare full output distributions. A new September 2026 EDCT-Bench arXiv preprint proposes verified visual counterfactual edits from VLM explanations and paired human validation, but it is not yet a peer-reviewed standard and image edit methods do not directly handle temporal video. The resulting design decision is not to run a claim-in-prompt sensitivity job under a faithfulness label; it would test text-conditioned behavior only. A stronger direct visual-faithfulness branch needs validated explanation propositions plus localized, validity-checked video edits, beyond the current three-video interval-mask setup. Direct generated-claim faithfulness remains 0/3 independently reviewed and 0/3 scored, with no direct effect, CI, or test; actual screenshot-to-decoder correspondence remains 0/15.

## 22:20 JST — independently recompute current candidate inventory bytes

**Purpose.** Close the remaining metadata-audit gap without opening review content or authorizing the study: verify that the 63 roles in the current candidate inventory still match the actual bytes they name through the builder's role mapping.

**Procedure.** Rebuilt a fresh private candidate inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_2220_01/pre_review_candidate_byte_inventory.json` with the current builder. Separately parsed its hash/size-only JSON, derived the expected role-to-path set from the registered private/source/code maps and exact three-media package layout, and independently recomputed SHA-256 and byte size for each target. Checked role uniqueness, expected role set, private-file modes, and the inventory state flags. Did not inspect video frames, question/answer/claim text, or reviewer exports.

**Observed result.** Inventory SHA-256 `0f91cb5395e6e2d7cbbebb8355f0d789cad7928009bb056734b76bd11e4f705b`; 63 rows and 63 expected roles; no duplicate/missing/extra roles; zero size/hash/mode mismatches across all 63 bound files; aggregate bound bytes 21,758,803. It records zero missing slots, `candidate_only`, `review_started=false`, `review_completed=false`, `joint_freeze=false`, `score_ready=false`, `execution_authorization=not_granted`, and `rights_status=unverified_local_only`.

**Interpretation and limit.** This independently confirms current byte identity of the inventoried inputs at the time of the check. It does not prove the correctness of upstream declared source/cohort digests, human identity or viewing, witnessed release chronology, screenshot-to-decoder correspondence, semantic claim support, or authorization. It is engineering readiness evidence only.

**Faithfulness checkpoint.** Direct generated-claim faithfulness remains **0/3 independently human-reviewed, 0/3 scored**, with no direct estimate, confidence interval, or test. Actual screenshot-to-decoder correspondence remains **0/15**. The research gate still needs an externally trusted role/key root and witnessed release, a real independent claim reviewer, interactive browser captures plus blinded frame-pair review, and a production frozen scorer receipt. No GPU job is justified until those prerequisites establish that the outcome represents the claim-specific target.

## 22:23–22:30 JST — lock the target construct and map actual visual-token time slices

**Research direction from the user.** The thesis endpoint is now clarified as selective prediction dependence on the video evidence indicated by the explanation, with improvement through a prespecified internal intervention as the primary metric candidate. A draft operational contrast was added to the internal-intervention plan: within baseline/treatment, compare the fixed original prediction margin drop after evidence-token-group ablation against an equal-count matched-control group ablation; then compare that selectivity between treatment and baseline. The effect formula, intervention, fixed answer/foil rule, controls, inclusion rule, sample size, and practical threshold are explicitly not frozen. This is a candidate construct, not a registered or approved test.

**Cohort provenance audit.** Added a local-only CLI that independently rebuilds the full three-record fingerprint payload from the saved generation rows, exact local video bytes, fresh 16-frame decode PTS, QA/choice hashes, evidence positions, and matched control positions. For the three exact stored records, recomputed digest `162fda51377768ae4e488fbfa1a8377f903ead3af7b277a17780b53e67b79a95` exactly matched the declared digest. The independent audit did not replay the full candidate scan/selection; digest matching therefore confirms the selected-record fingerprint only. It reports no question/answer/claim text, identity, or paths.

**Actual processor-to-frame layout diagnostic.** Extended the real-source CPU target candidate builder to record each contiguous Qwen video-token slice with its two source sampled-frame indices and exact decoder-relative PTS, using the checkpoint processor's temporal-patch/grid sizes and the exact token-ID positions. It rejects token-count, grid, frame-count, or contiguity mismatches. The first CLI attempt used an obsolete worksheet and failed before decode (`claim_sidecar_input_hashes_mismatch`, zero files written); comparing only hashes identified the worksheet field and no other compared field. Re-running with the exact sidecar-bound v6 worksheet processed all three videos × 32 sampled frames through the pinned processor. It emitted 3/3 diagnostic rows, each with 32/32 frames partitioned into 16 temporal grids, 16 contiguous video-token groups, and matching processor/source hashes. Per-row video-token counts were 4,224, 4,560, and 4,032. The mode-0600 local artifact SHA-256 is `599cdbb1262f27bbca7d0cb59d78d44e01c2b54495f4cf23c88a7defc8d276cf`. No model weights, CUDA, forward, score, or network were used; authorization remains `not_granted` and rights remain `unverified_local_only`.

**Limits.** This establishes the local processor's contiguous temporal-grid layout for these exact 32-frame tensors, not semantic evidence validity, model-layer mediation, browser-to-decoder timing, or claim faithfulness. The earlier cohort fingerprint was generated from 16-frame PTS; v2 scoring calls for a separately frozen 32-frame condition bundle. Real screenshot-to-decoder correspondence remains 0/15. Independent human review remains 0/3.

**Faithfulness checkpoint.** Direct generated-claim faithfulness is still **0/3 independently reviewed, 0/3 scored**, with no estimate, interval, or test. The ancillary QA-gold-interval GPU evidence remains an unrelated exploratory endpoint. No GPU intervention test was started: the treatment definition is not frozen and the human/trust/clock gates needed to assign actual reviewed evidence to these token groups remain open. Next: finish full tests, make sure the fresh private inventory binds the new mapper/auditor source hashes, and specify a single non-search internal intervention that could increase the candidate selective-dependence score without building its own outcome-selected target.

## 22:30–22:34 JST — close implementation audit and challenge the treatment candidate

**Verification.** Full repository script discovery completed successfully: 1,367/1,367 tests passed. The two focused suites passed 24/24 again after the final source edits; `compileall` and `git diff --check` also passed. An independent reviewer inspected both new code paths and reran the focused tests in `.venv`. The fingerprint helper's canonical digest, video key, inclusive PTS span, and minimum rank-distance control behavior agree with the source producer; a separate synthetic happy-path recomputation passed. The mapper validates temporal grid/frame/token dimensions and contiguous token positions before grouping processor slices with sampled-frame indices/PTS. No clear critical defect was found. Remaining implementation coverage gaps: the helper's end-to-end recompute path has no checked-in integration test, and the mapper's fake-processor tests alone do not prove checkpoint temporal semantics.

**Fresh local inventory.** Rebuilt after all code changes at `data/private_nextgqa/pre_review_inventory_v6_20261017_2233_01/pre_review_candidate_byte_inventory.json`. An independent role-to-path reconstruction and byte-hash pass matched 64/64 roles, 21,774,672 bytes total, with zero mismatches; inventory hash is `2e4948dee4470817807332a3e3222a49bb759a0ae8ae3a89fb3074f21cbfe37b`. Output directory/file modes are 0700/0600. It reports no missing slots but remains candidate-only, review incomplete, score unavailable, execution not granted, and rights unverified-local-only. This attests to bytes, not the truth of upstream declarations or human review.

**Intervention design challenge.** An independent design review proposed multiplying reviewed claim-span visual-token residuals by a fixed gain, then measuring ablation selectivity. Primary agent audit identified a serious circularity: treatment and outcome both target the same claim-selected tokens, so a positive `ΔF` may be built into the manipulation rather than demonstrate a useful intervention that naturally improves prediction faithfulness. Reject this as the primary efficacy treatment unless an independent discriminating rationale/control resolves the tautology; it may serve only as an engineering positive control. A less circular treatment must choose its target from an outcome-independent question/model signal and be fixed before confirmation, while the reviewed claim span remains evaluation-only. KFP/TA-AE-style methods still contain multiple choices and overlap prior video activation-engineering work; no intervention is frozen yet.

**Updated public construct log.** Added the user-selected target, candidate difference-in-differences interpretation, limitations, implementation checks, and current direct-evidence gates to the privacy-safe construct audit. No protected identifiers, video details, local artifacts, or private inventory were added to the public log.

**Status and next action.** Direct faithfulness remains 0/3 independently reviewed and 0/3 scored, no effect estimate/CI/test; actual screenshot-to-decoder correspondence remains 0/15. This cycle produced provenance and processor-layout readiness, not research efficacy progress. Continue pre-outcome intervention-design review and close the end-to-end test gap; do not run a faithfulness GPU job until the external human/trust/timing gates and a non-circular, frozen intervention pass.

## 22:34–22:36 JST — close cohort-auditor integration coverage; audit KFP reproducibility

**Implementation.** Added a synthetic end-to-end test for `recompute()`: three synthetic video byte sources, generation/lineage bindings, decoded PTS fixture, exact inclusive evidence spans, rank-distance control selection, and final cohort digest. Only decoder/pixel verification are mocked; the remaining audit path executes. Registered the test as a candidate-inventory source role. The test asserts that a matching digest remains non-authorizing. No private video/claim payload was used by this test.

**Independent audit and verification.** The independent reviewer reran the new fingerprint and inventory suites: 26/26 passed. It found no test or registration flaw; the synthetic test does not validate real PyAV decode/pixel semantics and has limited corruption coverage. Full repository discovery passed 1,369/1,369 in 29.8 seconds. A new private inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_2238_01/pre_review_candidate_byte_inventory.json` has 65/65 expected roles, zero missing, zero independently recomputed byte/hash/mode mismatches, 21,781,439 bound bytes, inventory SHA-256 `f304eabdaebae6c275635d64819f4da1bf5ece7d3302c1689216808f8f599af1`, directory/file mode 0700/0600, and still records review incomplete, score unavailable, execution not granted, rights unverified-local-only.

**Treatment source audit.** Rechecked the primary VERHallu/KFP paper. It describes layers 8–15, Gaussian `σ=1`, neighborhood width `m=5`, and `β=0.6`, but leaves `top-k` unspecified in the reviewed method text; its m/β ablations include LLaVA-NeXT, and layer outcomes trade off across task metrics. The official author repository page currently lists only README and LICENSE. Thus KFP is related and technically informative but not an exact reproducible fixed treatment from available source. Do not tune missing parameters on this cohort or call an invented fill-in a replication. Separately, direct multiplication of the reviewed claim-token residuals was rejected as primary because intervention and outcome would target the same tokens, risking a positive `ΔF` by construction.

**Status.** Direct generated-explanation faithfulness remains 0/3 reviewed and 0/3 scored, no effect/CI/test; actual browser/decoder correspondence remains 0/15. The test/inventory work improves auditability only. No valid efficacy run is currently ready. Next, compare an outcome-independent question-conditioned selector with the published KFP constraints, define a global-shift/sham control without outcome tuning, and keep all treatment parameters unfrozen until justified. GPU efficacy remains gated by external review/trust/time correspondence and a non-circular treatment definition.

## 22:36–22:37 JST — harden local audit reads and verify with actual source files

**Security repair.** A primary review of the new helper found that its first private-file reader checked mode with `lstat()` and then reopened by path, leaving a symlink/replacement race. Replaced it with a descriptor-based, no-follow read rooted at the repository's `data/private_nextgqa`, validating directory ownership/mode, file owner/type/link count/mode/size, and stable inode/size/timestamps during read. Added synthetic tests for successful private reading, symlink rejection, and wrong mode; the integration fixture also asserts that source-byte hash mismatch fails closed.

**Verification.** Focused cohort-auditor, inventory-builder, and temporal-mapper tests passed 44/44; `compileall` and `git diff --check` passed. Full repository test discovery passed 1,370/1,370 in 29.7 seconds. Re-ran the cohort audit through the hardened reader against the three exact local records and actual local video bytes/decoder: status `passed`, digest match true, 3/3 records, 16 frames each; full candidate scan remains unreplayed and human review/execution authorization remain false. This is source-fingerprint verification only. A new inventory at `data/private_nextgqa/pre_review_inventory_v6_20261017_2245_01/pre_review_candidate_byte_inventory.json` independently matched 65/65 expected roles and all bound file hashes/sizes/modes, zero missing/mismatch, 21,785,750 bytes total, inventory SHA-256 `61541b9f43837b81c4ff167d8e846aa512ba8596dd1c85efa14aba22fd572e02`, output modes 0700/0600. It remains candidate-only, review incomplete, score unavailable, execution not granted, rights unverified-local-only.

**Limits and next action.** The real audit did not independently replay cohort candidate selection, authenticate human review, verify screenshot/decoder correspondence, validate claim truth, or score the model. Direct faithfulness remains 0/3 reviewed and 0/3 scored with no estimate/CI/test; actual screenshot-to-decoder correspondences remain 0/15. Continue with a pre-outcome treatment design that does not target the same claim-selected tokens used by the outcome; do not start efficacy scoring until a fixed non-circular method and external review/trust/time gates pass.

## 22:37–22:39 JST — freeze interpretation boundaries for the candidate contrast

**Design clarification.** The same baseline-generated answer, explanation claim, and time interval must remain fixed across baseline/treatment scoring; otherwise `ΔF` compares different output objects. If post-treatment free-generation quality or grounding is evaluated, it is a separate secondary family and cannot substitute for the fixed-claim selective-dependence endpoint. For any KFP-style internal selector, select once from an ordinary full-video pass and reuse that key-frame choice across all ablation branches; condition-specific re-selection would confound evidence removal with selector movement. Added these interpretation constraints to the plan and privacy-safe construct audit.

**Verification/status.** Documentation-only changes passed `git diff --check`; no outcome data were accessed and no run was initiated. Direct faithfulness remains 0/3 human-reviewed, 0/3 scored, and no effect/CI/test; screenshot-to-decoder correspondence remains 0/15. This reinforces, but does not resolve, the design gate.

**Metric algebra clarification.** For the proposed nested contrast, `F_g=[m_g(full)-m_g(evidence-ablation)]-[m_g(full)-m_g(control-ablation)]` simplifies exactly to `m_g(control-ablation)-m_g(evidence-ablation)`. Hence `ΔF` isolates the internal-treatment × ablation-type interaction and cancels any common additive margin offset. This does not protect against condition-specific shifts, treatment-induced foil changes, or noncomparable token groups; the proposal fixes the answer/foil and requires matched token controls. The full-condition score remains a diagnostic outcome, not an extra term in the primary. The public construct audit and plan now record this identity and its limitations.
