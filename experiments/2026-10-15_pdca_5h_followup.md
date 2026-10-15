# Five-hour PDCA follow-up — 2026-10-15

## Plan and starting state

The requested work window began at 14:47:06 JST and has a hard deadline of
19:47:06 JST. Work completed before the start time is not counted. At start,
`HEAD` and the cached `origin/main` ref both named
`5ec542636c30174b6d0c18924d7a18ac08e1428b`; tracked files were clean. Existing
untracked NExT-GQA/sample files and the downloaded archive's zone identifier
were treated as protected and not read or changed.

The GPU/environment preflight was an individually elevated, read-only command.
It observed an RTX 5070 Ti with 14,821 MiB free of 16,303 MiB, utilization 4%,
PyTorch 2.11.0+cu128, CUDA 12.8, `torch.cuda.is_available() == True`, BF16
support, WSL-visible 13 GiB available of 15 GiB, and cgroup memory current
2.24 GiB with no configured maximum. The normal sandbox still reports CUDA
unavailable. GPU execution is not justified while the real-data scoring gates
remain closed.

The research gate remains `no_go`: no completed human review of model claims,
per-item media rights and dataset-agreement review are unresolved, inferential
target/SESOI are not frozen, and production decoder-pixel PTS mapping is not
calibrated to the annotation-player clock. The latest public-source audit says
NExT-GQA describes spans in seconds but does not establish the clock-origin
mapping to decoder PTS. The closed head-confirmation result is unchanged and no
head/layer/alpha search was reopened.

## Do — fail-closed execution-path repair

Source review found that `run_necessity_choice_margin_test_and_publish.py`
accepted an optional preflight path and checked only that it existed. Its
direct child scorer did not consume a gate report. An independent audit then
identified that merely checking `ready`/`granted` strings would allow a
hand-authored report and would not bind the report to timing calibration or
the run configuration. That audit finding was reproduced by inspecting both
entrypoints and the current cross-gate, which deliberately always returns
`blocked`/`not_granted`.

The correction is deliberately fail-closed: both the publisher and direct
scorer require a report before doing work, and
`scripts/require_execution_preflight.py` rejects every current schema-v1
cross-gate report, including a fabricated `ready`/`granted` document. Unknown
schemas are rejected as unsupported. The current protocol has no authenticated,
run-bound execution-grant format, so this scorer cannot load a model until a
future gate is implemented with exact cohort/sample and run-configuration
bindings plus timing/pixel, human-review, and rights checks. This is a safety
repair, not an experimental result.

Added `docs/annotation_clock_calibration_worksheet.md` as a blank private form.
It specifies the evidence a reviewer would need to bind annotation clock,
decoder PTS/time base, source-frame order and pixels, calibration error and
uncertainty, and the pre-outcome tolerance. It requires every frozen cohort
member to pass and leaves all measurements blank. It cannot close the gate by
itself. The contract links the worksheet and records that no current report
schema can authorize scoring.

## Check

Focused checks passed **14/14** across the new fail-closed preflight tests, the
necessity-run publisher tests, and the existing claim-mask cross-gate tests.
The direct CLI smoke uses a blocked report and a nonexistent sample path; it
exits at the preflight gate before reading sample data. Python byte-compilation
and `git diff --check` passed. One earlier focused run exposed stale assertions
from the intermediate permissive design; those tests were rewritten to verify
unconditional schema-v1 and unsupported-schema rejection, then passed.

The independent auditor verified the fabricated-report bypass before repair,
then confirmed the revised helper has no acceptance path. The auditor also
reviewed the worksheet as blank/private and found no authorization overclaim.
The full standalone script suite passed **803/803** after the final
fail-closed implementation and tests (`PYTHONPATH=scripts .venv/bin/python -m
unittest discover -s scripts -p 'test_*.py'`). Python byte-compilation and
`git diff --check` also passed. GitHub publication and final source hashes are
pending.

## Faithfulness progress at this checkpoint

| Quantity | This five-hour window | Running project total evidenced here |
|---|---:|---:|
| Requested direct faithfulness samples | 0 | 0 new in this window |
| Effective direct faithfulness samples | 0 | 0 new in this window |
| New direct faithfulness effect estimates | 0 | 0 new in this window |
| Model/dataset/intervention/score | Qwen2.5-VL / NExT-GQA / reviewed-claim frame mask vs equal-count temporal control / gold-vs-best-foil log-margin | Not run |

This checkpoint adds no accuracy, grounding, faithfulness, or causal evidence.
There are no effect intervals, hypothesis tests, multiplicity adjustments,
sample exclusions, or inference failures to report because scoring was not
performed. Existing historical head-confirmation evidence remains −0.0756 nat
(95% t CI [−0.1578, +0.0065], 79/79) against the prespecified +0.25 nat
threshold; it is unchanged and closed.

## Act / next steps

Continue the authorized window by independently auditing the remaining
pre-scoring gates and the other current inference entrypoints for a concrete
cross-path bypass. Re-run the complete suite after the fail-closed changes,
inspect exact staged paths and hashes, then publish only the scoped code,
blank worksheet, contract update, and experiment log if the privacy and review
checks pass. Do not run real-data scoring or synthetic GPU load merely to fill
time; reconsider GPU work only if an executable, discriminating protocol test
with a clear purpose becomes available.

### Expanded legacy-inference entrypoint audit — 15:02 JST

An independent inventory of Qwen model-loading scripts found nine additional
active command-line routes that could perform real-media scoring despite the
current no-go or closed-search decisions: post-hoc evidence-probe task-validity
QA, activation patching, activation steering, soft and hard head ablation,
unsupervised head selection, steering-layer survey, grounding proxy, and the
old faithfulness validation orchestrator. Source inspection confirmed each
route reads NExT-GQA sample/video inputs and/or initializes the Qwen model; the
orchestrator could create manifests before its already-guarded child stopped.

Each entrypoint now calls the shared fail-closed stop immediately after
argument parsing and before sample/manifest reads, media access, model loading,
or Git side effects. The static inventory now covers 26 retired real-data
inference CLIs and checks guard ordering plus explicit codes for all nine new
entries. `docs/research_overview.md` now records these closures. The
video-only NOAA runtime smoke and synthetic engineering diagnostics remain
separate; their usage neither computes QA/grounding/faithfulness outcomes nor
reopens the closed head search. I did not rerun GPU inference: the identical
NOAA smoke was already completed and its log says no repetition is needed, and
no new GPU diagnostic with a discriminating question follows from these CLI
stops.

Focused preflight/publisher/branch tests passed 10/10. An initial command
omitted `PYTHONPATH=scripts` and failed at test-module import; it made no code
or data changes. The documented invocation then passed. After the expanded
entrypoint set, the full suite passed **805/805** in 7.767 seconds; this
supersedes the earlier 803-test checkpoint. The existing failed-diagnostic
JSON lines printed during the suite are expected fixture outputs and did not
cause test failures. Direct faithfulness progress remains **0 requested / 0
effective / 0 new estimates**; no QA or grounding samples were run, so no
exclusions, effect intervals, p-values, or adjusted results exist for this
window.

The independent auditor verified guard placement for the first 17 routes and
then flagged these nine additional bypasses; those were closed. A final
independent audit of the integrated diff is pending. Next: re-run static and
compile checks, inspect every changed path and protected untracked item,
resolve any auditor findings, then stage and publish only the scoped research
code, contract/documentation, blank calibration worksheet, and this log.

### Runtime CLI stop verification — 15:04 JST

Ran each of the nine newly guarded legacy CLIs in a fresh `.venv` process with
nonexistent sample/output paths. All nine exited with argparse status 2 and the
shared `blocked[...] ... no inference was launched` diagnostic before opening
those paths. This supplements the AST order audit with runtime behavior. The
final inventory test now checks the literal stop code for all 26 guarded CLIs,
including the original 17, and focused gate/inventory tests pass 6/6 after that
hardening. Nine loader guards are also byte-compiled. This remains protocol
safety work: no model was loaded, no data scored, and direct faithfulness stays
0 requested / 0 effective / 0 estimates.

### Cross-gate to scorer integration check — 15:06 JST

A remaining verification seam was that the synthetic cross-gate test proved all
component structures could pass while the report remained blocked, and the
scorer consumer separately rejected schema-v1 reports, but no single test
connected those two assertions. Extended the fully synthetic integration test
to write its structurally complete cross-gate report and pass it to the scorer
preflight consumer. It is rejected with the explicit non-authorizing-schema
reason; no private or real cohort inputs are involved. The focused integration
test passes. This supports the fail-closed property across report production
and consumption, but still does not authenticate human/legal evidence or
authorize scoring. Faithfulness progress: 0 requested / 0 effective / 0
estimates in this window.

The integrated cross-gate assertion and current-log link passed the full
standalone script suite: **805/805** in 7.853 seconds. Transformers emitted
three non-failing deprecation warnings in legacy test fixtures; the existing
fixture failure-status JSON lines are expected test outputs. No test accessed
protected sample/media files or initialized CUDA. The current experiment
overview now links this follow-up log separately from the earlier same-date
PDCA record.

### Model-loader inventory closure — 15:08 JST

The fixed 26-entry guard list protects known retired CLIs but alone would not
flag a new model-loader script added later. Added a source-level inventory test
that scans every non-test Python script for `from_pretrained` calls and
requires explicit classification. It finds 30 loader scripts in this checkout:
17 retired scoring CLIs covered by the early-stop guard, the separate direct
choice-margin scorer covered by the execution-preflight consumer, and 12
allowlisted engineering/runtime diagnostics. The diagnostics are processor
boundary checks or label-free runtime/numerical smoke tools; the independent
scope audit found no QA, grounding, or faithfulness scoring claim in them.
The inventory and guard-order tests pass 2/2. This protects against silent
unclassified additions; it is a static source inventory, not a semantic proof
that future diagnostic changes preserve their scope. Faithfulness remains
0/0/0 for this window.

The new loader inventory test and all existing standalone tests passed
**806/806** in 8.450 seconds. Non-failing legacy processor deprecation
warnings and synthetic failure-path status lines appeared as expected. The
previous B=1/B=2 batch-shape branch is not being reopened: its same-clip
near-tie sensitivity and explicit stop are recorded in
`experiments/2026-10-15_pdca_3h.md` and
`experiments/2026-10-14_pdca_9h_gpu_cache.md`; additional repeats would not
answer a new question. Consequently no CUDA work is run in this follow-up.

The independent reviewer completed the loader-classification audit: all 17
retired loaders are in the guard inventory, the direct choice-margin loader
is preflight-controlled, and the 12 other loaders are documented engineering
diagnostics. No omitted direct scorer or overblocked reusable/synthetic path
was found. Two generic `spike_test_*attention.py` tools accept arbitrary media
and prompts, so their docstrings now explicitly restrict use to media cleared
for local processing and state that they report only attention wiring/resource
use, not QA, grounding, or faithfulness. The overview carries the same approved-
media caveat. These doc-only clarifications add no experimental evidence.

A separate design-support review confirmed the inventory test closes the
current direct-loader gap and preserves the diagnostic paths. Its AST scan
only detects syntactic `.from_pretrained(...)` calls; aliased/custom loader
wrappers or non-Hugging-Face inference APIs are outside its guarantee, and no
such current bypass was found in the scanned scripts. A single unified
filename-to-class mapping would be a minor maintainability improvement, but
there is no present mismatch; defer that refactor rather than add churn to the
verified guard set.

### WSL memory snapshot — 15:11 JST

A read-only snapshot found WSL `MemTotal` 15,920,824 KiB, `MemAvailable`
14,625,288 KiB, 1.2 GiB displayed used, and 2.6 GiB buff/cache. The visible
cgroup had `memory.current` 3,568,611,328 bytes, `memory.max=max`, and all
`memory.events` counters (including OOM) at zero. Its `memory.stat` attributed
about 2.53 GiB to file cache and 0.67 GiB to anonymous memory; memory PSI
`avg10/avg60/avg300` was zero. The process table showed no model process.
This demonstrates that this WSL workload was not using 30/32 GiB at this
checkpoint; it cannot identify other Windows processes or total host use.
No GPU process was launched. Direct faithfulness progress remains 0/0/0.

### Claim-review and rights gate boundary audit — 15:13 JST

The primary source review of `audit_claim_review_evidence_index.py` confirms
that it byte-binds the eligible, claim-input, mapping, and review-index files;
checks exact video coverage, unique review-record hashes, model/run/claim
references, and requires the protocol to declare review completed before
scoring. It explicitly returns `semantic_validation=not_machine_verifiable`
and `execution_authorization=not_granted`. This cannot establish that a human
actually reviewed the evidence, that the reviewer was independent or
qualified, or that a claim is visually supported. Those require examining the
private evidence and a trustworthy human attestation.

The primary source review of `audit_data_rights_evidence_index.py` confirms
separate byte/hash and exact-coverage checks for annotation terms, per-item
media conditions, and dataset access-agreement evidence, with intended uses
required to be permitted. It too always reports `execution_authorization=
not_granted`; its `structural_status=passed` is not a license opinion or
evidence authenticity check. The current fail-closed consumer rejects even a
fabricated ready structural report, so neither structural auditor can unlock
inference. No structural bypass was found in this pass; reviewer authority,
underlying source authenticity, actual license scope, and a human review remain
unresolved upstream gates. No private ledgers or media were opened. Faithfulness
remains 0 requested / 0 effective samples / 0 estimates.

An independent source-level audit additionally confirmed the known rights-index
boundary: the contract explicitly says the media/conditions bundle hash and
locator are syntax/count checked but not resolved or compared with each row's
`conditions_record_sha256`; it likewise documents that agreement locators are
not rehashed. This is not an undisclosed execution bypass, because the report
never authorizes and the current consumer rejects every schema-v12 report. A
future stronger evidence-binding revision could add per-record hashes and
verified file resolution, but that requires a private evidence-bundle interface
and human review procedure; adding fields without actual evidence resolution
would create false assurance. Defer implementation until that interface is
specified.

A second independent reviewer found no material structural bypass, and
corroborated the documented limit: these audits do not open locator targets or
prove semantic claim support, media identity, actual blinded/independent review,
reviewer authority, license applicability/condition fulfillment, or real-world
event ordering. The review specifically checked strict boolean review
declarations, claim/model/run bindings, row-level rights/use decisions, and
agreement applicability branches. Structural checks remain provenance aids,
not human review substitutes.

That reviewer also found a genuine overblock: `smoke_qwen_attention.py` built a
temporary synthetic MP4 but delegated the forward to the now-retired
`grounding_check.py` scorer, so the synthetic wiring smoke could no longer
run. Replaced that delegation with a self-contained, offline-only Qwen
attention forward over four sampled frames from a generated synthetic MP4.
It writes a mode-0600 aggregate artifact, records model/config/runner/media
hashes and runtime/VRAM, and emits no attention or answer scores. The closed
real-media scorer remains stopped. The expanded dynamic inventory now finds
31 pretrained-loader scripts: 17 retired scorers, one preflight-controlled
direct choice-margin scorer, and 13 explicitly classified engineering smoke
tools (including this synthetic one). Focused tests passed 4/4 and CLI help and
byte-compilation passed before the GPU smoke.

The bounded GPU smoke then completed 1/1 synthetic videos, 4/4 requested
frames, and one forward with finite logits and attention across 28 layers.
On RTX 5070 Ti / Qwen2.5-VL-7B-Instruct pinned revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
eager attention, seed 20261015, PyTorch 2.11.0+cu128 and Transformers 5.18.0,
the forward took 1.771 s; the runner's 11.291 s elapsed field begins after
synthetic-video creation and frame extraction, so it covers model load through
forward and is not full process wall time. PyTorch peak allocation was
7,096,033,280 bytes (6.61 GiB). Pre-load free VRAM was 15,738,077,184 bytes.
The result is
`results/qwen_synthetic_attention_smoke_20261015_01.json`, SHA-256
`a49726a3fa5d81ab08e7333b7b10fb0bbdcc105064c34a36a15c6c0ccdd68327`;
its runner hash is
`cdd3838677149ed27c6659e9c0af358c47c318a81ce307776e59199fe279188b`.
The post-run manifest
`results/qwen_synthetic_attention_smoke_20261015_01.manifest.json` binds that
hash to checkout commit `b233016b7027393967843b79528ea3db9ed5ddd0`, records
that the worktree was dirty at execution, and freezes the run settings; its
SHA-256 is `85480f4e1289ab102ac0185ccee71349e2df4b0ad2c4348b1df0658a2541a5b9`.
The independent artifact auditor reproduced the result and runner hashes and
all reported counts/resources; no QA text, IDs, paths, raw pixels, logits,
attention arrays or scores appear in the output. It also confirmed the
runner-hash/checkout relationship is the best provenance available because the
run happened from a dirty worktree; the companion manifest records the parent
commit and that dirty state. This demonstrates the restored synthetic
execution path only; it contributes zero direct faithfulness samples and zero
effect estimates.

The same audit found that the direct `.from_pretrained()` inventory did not
classify an unrelated tiny random-weight CPU constructor in
`audit_qwen_cache_positions_cpu.py`. This was not a real-checkpoint or dataset
bypass, but the test name overstated its coverage. Renamed the check to specify
literal pretrained loaders and added a separate explicit classification test
for the known synthetic CPU model constructor. This is still a source-level
inventory of known construction forms, not a proof against every future custom
or dynamic loader API.

The complete standalone suite was rerun after the pretrained-loader scope
clarification: **809/809 passed in 8.044 seconds**; `py_compile` and `git diff
--check` also passed. Only fixture-driven expected failure messages and
non-failing upstream processor deprecation warnings appeared. No GPU model was
loaded and no GPU work ran during this CPU test-suite run; the separate GPU
smoke above is recorded independently.

### Outcome-blind annotation-clock calibration design audit — 15:22 JST

The annotation-clock worksheet previously requested offset, drift, residuals,
and uncertainty but did not distinguish anchors used to fit the map from
anchors used to validate it. This leaves a measurement-design ambiguity: an
affine offset/drift map fit from two anchors can match those same two anchors
exactly, which does not establish accuracy at other times. Updated the private
worksheet and protocol contract to require disjoint, preselected fit and
validation anchor sets with frozen temporal coverage; the mapping and
coefficients must be frozen before validation values are examined, and
validation residuals cannot be used to refit. Counts, spacing, mapping form,
uncertainty method, and tolerance remain decisions that must be justified and
frozen before measurement; this change does not invent values for them.

This is a protocol design repair only. The worksheet remains blank, no media or
private calibration record was opened, and no item has passed clock
calibration. It does not unlock the gate. Direct faithfulness progress remains
0 requested / 0 effective / 0 estimates. No GPU run follows from this design
change.

An independent auditor confirmed that this does not set a sample-count or
tolerance value and preserves all execution refusals. The review caught a
wording risk: disjoint anchors within one video are not thereby statistically
independent. Updated the worksheet and contract to call them held-out anchors
and to require any uncertainty calculation to justify its within-video
dependence assumptions.

### Independent task-validity route audit — 15:27 JST

An earlier audit summary said `audit_evidence_probe_task_validity.py` could
still score NExT-GQA. Rechecking current source showed that the scorer itself
already stops immediately after argument parsing, before reading its sample,
probe, annotation, or map files or loading a processor/model. The remaining
gap was its publisher wrapper: `run_task_validity_audit_and_publish.py` first
resolved and checked input paths and read local Git identity before launching
the already-closed child. It could not reach inference or publication because
the child failed, but this violated the intended early-stop boundary and could
expose private paths in local diagnostics.

Added the same branch stop to the publisher immediately after parsing and
registered it in the guard-order inventory. A runtime invocation with
nonexistent input files exited with the expected argparse status 2 and
`blocked[legacy_evidence_probe_task_validity]`; the designated output was not
created. The focused branch inventory passes **3/3**, including checks that all
27 registered retired CLIs have the expected stop before file/model/process
side effects. No sample, probe, or media contents were read and no inference
ran. The independent source audit confirmed the updated guard, and also notes
that the allowlisted engineering scripts `spike_test_attention.py` and
`spike_test_video_attention.py` can accept arbitrary local image/video prompts
for 3B generation/attention smoke without a rights preflight. They do not
produce research faithfulness scores; if the policy later expands to prohibit
all arbitrary real-media model forwards, that scope needs an explicit gate.
This audit made no such scope change.

Faithfulness remains **0 requested / 0 effective / 0 new effect estimates**
for this window. No metric, cohort, intervention, exclusions, failures, CI,
significance test, or multiplicity-adjusted result was generated. This route
closure is operational hardening, not scientific evidence. The broader
claim-mask no-go remains unchanged.

### Claim-mask SESOI and contrast audit — 15:29 JST

Independently checked the schema-v12 contract, validator, and blank decision
worksheet against the older threshold review. The algebra and sign are
consistent: positive `d_i = S_i(control) - S_i(claim-mask)` means the
claim-linked mask lowers the gold-versus-best-foil log-margin more than the
equal-count control; `exp(d_i)` is the control/evidence ratio of those
condition-specific gold-to-best-foil ratios. `exp(mean(d))` is a geometric
mean of per-video ratios, not an answer-probability or accuracy change. The
success rule (lower endpoint of a two-sided 95% video-level t interval above
the fixed SESOI) matches the declaration.

The independent review found no basis for carrying the old `+0.25 nat`
head-ablation threshold into this distinct input-mask estimand. The worksheet
correctly leaves the value unresolved and asks for a practical interpretation
and independent pre-outcome review. This agrees with Lakens' sample-size
guidance, which distinguishes a smallest effect of interest from the effect
size needed for statistical significance or power and expects justification
to match the inferential goal ([Lakens, 2022](https://doi.org/10.1525/collabra.33267)).
No threshold value was proposed or frozen. The worksheet/validator hashes and
review timestamps are declarations, not authentication of the evidence or
reviewer independence; the docs already state this limit. Primary video
faithfulness progress remains 0/0/0, and no scoring follows from this review.

The complete standalone suite passed **809/809 in 9.173 seconds** after the
publisher guard change. The modified scripts also pass `py_compile`, and the
scoped diff passes `git diff --check`. The code fix and inventory update are in
commit `448de214f2c98e4d66a91425fd181625760e9fd7`; a live elevated
`git ls-remote` confirmed that exact commit as `origin/main`. Source SHA-256:
`scripts/run_task_validity_audit_and_publish.py` —
`b4ef19f8dd8fdeda47e1b79cfdfda9f5d15738eb237701386027fff293b32005`;
`scripts/test_closed_research_branches.py` —
`15eabd62bb0ef66d0855765a92c90624ac05c3b0ef9398636a0192c5e677876a`.

### NExT-GQA rights-lineage source check — 15:32 JST

Rechecked the public primary pages named by the private rights worksheet. The
NExT-GQA README says its test/validation questions are subsets of NExT-QA and
represents temporal spans in seconds, with a separate frame-id-to-seconds map
([NExT-GQA dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)).
The NExT-QA README says its videos come from VidOR and links annotations to
video paths using an ID map; its repository MIT text explicitly names software
and associated documentation files, not a blanket license for video media or
all annotations ([NExT-QA README](https://github.com/doc-doc/NExT-QA/blob/main/README.md),
[NExT-QA LICENSE](https://github.com/doc-doc/NExT-QA/blob/main/LICENSE)).
VidOR describes its videos as unprocessed YFCC100M items and exposes distinct
YFCC IDs, video hashes, and paths ([VidOR project page](https://xdshang.github.io/docs/vidor.html)).
The Multimedia Commons/YFCC page states that dataset access is subject to a
Webscope License Agreement and each original media item follows the
creator/uploader's Creative Commons selection in item metadata
([YFCC100M Core Dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)).

These mutable public pages corroborate the worksheet's multi-edge lineage and
separate annotation/software, media-license, and dataset-access reviews. They
do not identify the exact local cohort items, establish which YFCC records or
current files match them, prove any item license's scope/conditions, or verify
whether the relevant agreement was accepted/applicable. No local cohort,
archive, sample manifest, or media was opened. The private rights state remains
unresolved and scoring remains `no_go`; this source check contributes zero
faithfulness samples or effects.

### Retired-child publisher inventory regression test — 15:35 JST

The publisher gap suggests a testable inventory invariant: a Python list that
names a retired inference CLI must appear only in an early-stop-guarded file.
Added an AST regression that scans all non-test scripts for such literal list
references and requires the containing file to be a guarded entrypoint. This
would have caught `run_task_validity_audit_and_publish.py` before the runtime
audit; it also checks the existing evidence-control and head publishers. The
focused guard suite passes **4/4**. This conservative scan does not prove a
list is executed as a command and may flag a non-executed reference; it also
misses tuples, dynamic imports, aliases, shell-built commands, and arbitrary
loader APIs. Independent source review remains necessary.

The 15:32 source check did not resolve a single NExT-GQA video's license or
agreement applicability. NExT-GQA's seconds/frame-id documentation still does
not calibrate annotation-player time to the production decoded PTS origin.
Therefore rights, human claim mapping, inferential target/SESOI, and timing
remain independent blocking gates. Direct faithfulness this window remains
**0 requested / 0 effective / 0 new effect estimates**. The cumulative closed
head-confirmation result is still 79/79, −0.0756329 nat, 95% t CI
[−0.1577747, +0.0065089], and is not part of this window's progress.

At 15:34 JST, the full standalone script suite passed **810/810** in 8.945 s
after the AST test's name and scope description were made precise. `git diff
--check` also passed. The suite emitted expected diagnostics from negative-path
tests; overall exit status was zero. This verifies code/test integrity only,
not rights, annotations, or faithfulness.

### Gate-consumer and hash-binding audit — 15:36–15:38 JST

An independent adversarial source review found no way for the current structural
cross-gate or its scorer consumer to authorize inference. The producer always
emits `blocked`/`not_granted` and exits 1; the consumer rejects every current
schema/purpose report unconditionally (including fabricated `ready`/`granted`)
and rejects all other schemas. The direct scorer and publisher call that
consumer before model initialization. Duplicate JSON keys, non-finite constants,
and malformed component contracts also fail closed. This is source-level
evidence only; future execution authorization still needs a separate
authenticated, run-bound format. The main agent independently verified these
branches and the synthetic integration test in source.

The protocol hash functions bind the declared primary-estimand and inference
plan fields. Existing tests covered several mutations but not every leaf. Added
table-driven tests for every leaf included in each canonical hash payload; the
focused validator suite passes **52/52**, and the full suite passes
**812/812** in 9.338 s. These checks show that each declared field changes its
hash in the test fixture; they do not authenticate evidence referenced by the
hashes or clear any scientific gate.

### Synthetic PTS-to-Qwen processor integration — 15:36 JST

Ran `.venv/bin/python scripts/audit_qwen_pts_to_processor_cpu.py` against a
temporary lossless fixture and the pinned local processor/config revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. No project media, model weights,
CUDA, QA, or annotation were used. The fixture decoder returned 7 CFR source
frames at PTS 0–3000 with time base 1/1000; deterministic endpoint sampling
selected 4/4 at source indices `[0,2,4,6]` and relative times `[0,1,2,3]`
seconds. Distinct pixel signatures survived the Qwen utility; processor frame
indices were `[0,1,2,3]`, FPS was 1, and the temporal grid was `[2,2,2]`.
Installed Qwen RoPE temporal anchors `[15,19]` matched the fixed reference
calculation exactly. A nonuniform VFR fixture and an 8-to-4 endpoint-sampled
CFR fixture with nonuniform selected times were both rejected by the uniform-
FPS gate. The independent auditor reproduced these temporal calculations and
matched the runner, helper, checkpoint-config, and installed-source hashes.

Artifact `results/qwen_pts_processor_cpu_audit_20261015_1536.json` has SHA-256
`a921726c29d094be5a703c783e2dbad104782a8dca9c767496ee8bd420819760`; runner
`scripts/audit_qwen_pts_to_processor_cpu.py` has SHA-256
`a312a7b76ff0110f8c5485dd8d8aa512e8fa4868603c0773190e85295cfc24fd`. The
artifact's `source_commit` was `3f1bdb145994c9e1dbd67bfadd9e3b97a3323181`.
This synthetic result validates only the local PyAV-to-Qwen fixture boundary
for this stack. It does not test production media/decoder equivalence, codec
diversity, or annotation-player clock origin, so timing calibration remains
blocked. It measures no answer accuracy, faithfulness, or semantic validity.

At this checkpoint, direct faithfulness remains **0 requested / 0 effective /
0 new effect estimates** for the five-hour window; no inference failures,
exclusions, effect intervals, tests, or multiplicity results were generated.

### Sampling-configuration hash gap and schema-v13 repair — 15:39 JST

The independent hash audit found that `sampling.config_sha256` was in the
primary-estimand hash, but the protocol's explicit policy, claim-selection
flag, frame count, frame size, pixel cap, and per-frame-cap flag were not.
Changing one of those declared values while leaving the digest field unchanged
therefore left the primary hash unchanged. Since the exact sampled inputs are
part of the estimand, that was an avoidable ambiguity. The digest alone also
does not prove the referenced private configuration bytes match those declared
values.

Updated the canonical primary hash to include every explicit sampling field,
and allowed strict JSON booleans in the canonical hash payload (without
accepting integer aliases). Because this changes canonical hash semantics,
the protocol and its cross-gate/audit consumers were consistently advanced
from schema v12 to v13; v9–v12 inputs remain rejected. Updated current contract,
SESOI template, research overview, runtime messages, and synthetic fixtures.
The contract now requires a future runner to resolve the exact sampling
configuration, verify its digest, and check agreement with the declarations.
Historical v12 experiment entries remain unchanged.

The table-driven tests now mutate every primary-estimand and inference-plan
hash leaf. Each primary field mutation changes the computed digest and causes
the validator to report a stale-hash mismatch; explicit sampling fields are
included in that set. Focused validator, cross-gate, rights/review-audit, and
publisher tests pass **91/91**. The full script suite passed **812/812** in
8.328 s. An independent source audit verified every schema consumer and
synthetic fixture now expects v13, all seven sampling fields are included in
the hash, boolean values are canonical JSON booleans distinct from integers,
and the cross-gate and scorer consumer remain fail-closed. It found one
documentation sentence that mentioned only the sampling-config hash; that is
corrected to require agreement between the digest and explicit declarations.
`rg` found no active v12 schema consumer in scripts/docs; remaining v12
references identify legacy history or tests that reject old versions.
`git diff --check` passes. No test read protected sample/media data or used
CUDA. SSS is not claimed while human, rights, timing, inferential-target, and
SESOI gates remain unresolved. The cross-gate remains `blocked`/`not_granted`;
the repair authorizes no inference and changes no experimental outcome.
Faithfulness progress remains **0 requested / 0 effective / 0 new estimates**.

The schema-v13 hash update and sanitized synthetic result were published in
`79bff800933f715518062870ee3d83ba9b50b43c`; the exact-config cross-gate repair
and audit were published in `650414c3032deb7e30c951ae877d6199845cbbf8`.
`git ls-remote origin refs/heads/main` returned the latter commit, matching
local `HEAD`. The committed result JSON's SHA-256 matches
`a921726c29d094be5a703c783e2dbad104782a8dca9c767496ee8bd420819760`; no raw
media, protected sample manifests, personal locators, or local paths were
included.

### Exact sampling-config cross-gate binding — 15:43–15:45 JST

Followed the hash audit's remaining limitation by adding the CPU-only
`audit_claim_sampling_config.py` component to the schema-v13 cross-gate. It
checks the SHA-256 of the exact configuration bytes and strict-type equality
between the parsed config and each explicit protocol sampling value. A hash
match alone, value match alone, bool/int substitution, wrong schema, malformed
JSON, or duplicate-key config cannot be treated as a passed component. The
report emits only fixed status/error codes and always retains
`execution_authorization=not_granted`.

Synthetic tests verify exact match, byte-only mismatch, declaration mismatch,
bool/int mismatch, wrong-version refusal, and integrated cross-gate failure.
Focused tests pass **12/12**; the complete standalone script suite passes
**818/818** in 8.405 s. `--help`, Python byte-compilation, and
`git diff --check` pass. This does not read any project config/media or
authenticate rights/review evidence; future runners still have to use the
same exact private config and independently satisfy every gate. Direct
faithfulness remains **0 requested / 0 effective / 0 new estimates**.

An independent second audit found no cross-gate bypass and confirmed the test
suite covers the raw-byte/parser boundary. It noted that the first direct
value-mismatch unit test paired changed parsed values with old raw bytes; I
repaired it to serialize the changed values and update the protocol digest,
then assert that the value mismatch is still reported without a hash mismatch.
Integrated tests now also prove duplicate-key and non-finite sampling JSON is
rejected generically. The focused cross-gate/config suite passes **13/13** and
the full suite after these test changes passes **819/819** in 8.037 s. The
auditor independently inspected this correction and confirmed the gap is
closed. Authorization remains blocked; no data or CUDA was used.

### Per-video rights-evidence binding audit — 15:48–15:51 JST

An independent design review identified that the private data-rights index
asserted a media/conditions bundle hash and cohort coverage count, but did not
structurally bind each eligible video's ledger `media_evidence_sha256` and
`conditions_record_sha256` to its corresponding index record. Upgraded the
private index to schema v2 with one exact `(video_id, media_evidence_sha256,
conditions_record_sha256)` row per eligible item. The checker now enforces
one-to-one identity coverage, strict record shape/hash syntax, and hash equality
against that video's ledger row; row ordering is intentionally irrelevant.
Updated synthetic direct and integrated preflight fixtures and the human-review
worksheet. A matching structural fixture still yields `not_granted`.

Focused rights-index tests pass **15/15**, and cross-gate integration tests
pass **8/8**. The full CPU suite passes **820/820** in 9.028 s. The initial
full-suite attempt had one test-fixture failure because the row-reordering
test changed index bytes without recomputing the protocol's exact index hash;
the fixture was corrected, then the focused and full suites passed. Python
byte-compilation and `git diff --check` pass. No protected input, project
media, model, or CUDA was accessed. This is structural consistency only: it
does not dereference locators, re-hash bundle files, authenticate reviewer or
source evidence, judge rights, or authorize inference. Faithfulness progress
remains **0 requested / 0 effective / 0 new effect estimates**; no metric,
effect size, interval, significance test, multiplicity adjustment, or
inference-failure count is available for this research window.

Independent implementation audit found no fail-open and confirmed row-order
independence, but recommended a stronger identity mutation: swap two videos'
valid-shaped media and conditions hashes while also recomputing the index and
protocol hashes. Added this test; the direct rights-index tests remain **15/15**
and the complete suite rerun passes **820/820** in 8.514 s. This specifically
shows that exact per-row ID/hash binding rejects a valid-format cross-video
swap, while a pure row-order permutation passes after its outer exact-byte
hash is updated.

Next action: independent implementation audit, then inspect any remaining
executable protocol boundary with synthetic data. Do not run claim-mask scoring
until human frame mapping, item-level rights, production-clock calibration,
target/dependence, SESOI, and a run-bound authenticated gate are resolved.

### Synthetic annotation-clock calibration arithmetic — 15:51–15:53 JST

Implemented `scripts/annotation_clock_calibration.py` as an exact-rational,
CPU-only ordinary least-squares affine fit from PTS seconds to annotation
seconds, followed by held-out residual calculation with no refit. Fit and
held-out rows must use disjoint unique anchor IDs; malformed/empty inputs,
fewer than two fit anchors, degenerate fit PTS, nonpositive fitted drift, and
nonpositive tolerances fail closed. The caller supplies a positive rational
tolerance; the helper returns `uncertainty_status=not_evaluated` and
`execution_authorization=not_granted`. Documentation limits its use to cases
where this exact fit method was already fixed before observing anchors.

An independent code review verified the OLS formulas, exact arithmetic, and
fit/held-out separation. It recommended a non-collinear fit fixture to
distinguish OLS over all fit observations from a two-point line, and explicit
tolerance-boundary checks; both were added. Six synthetic unit tests pass,
covering known offset/drift, non-collinear OLS and inclusive threshold,
held-out outlier/no-refit, degenerate and reused anchors, negative drift, and
malformed tolerance/values. The complete CPU suite passes **826/826** in
8.656 s; `py_compile` and `git diff --check` pass. No actual calibration
anchors, videos, model, CUDA, or private project data were used. This utility
cannot establish anchor provenance, source identity, temporal coverage,
measurement chronology, uncertainty, decoder equivalence, or actual clock
validity. It does not resolve the production timing gate. Faithfulness remains
**0 requested / 0 effective / 0 new effect estimates**.

A post-implementation cross-document scan found that the protocol contract's
data-rights section still described the earlier limitation and did not mention
schema-v2 per-video binding. Corrected that summary and checked the worksheet
and current contract statements against the implementation. This is a
documentation consistency repair; it changes no result or gate.

### Related-work audit: video and intervention-based explanation faithfulness — 15:56–16:00 JST

Reviewed primary papers to check whether current framing overstates what the
proposed claim-to-span mask contrast can establish. This is a literature
comparison, not evidence about Qwen2.5-VL on this project's data. The review
covered the especially close visual counterfactual CCT study ([Menzat et al.,
arXiv:2609.06704v1](https://arxiv.org/abs/2609.06704v1), whose arXiv record
reports acceptance to Findings of EMNLP 2026), CCT
([Siegel et al., ACL 2024](https://aclanthology.org/2024.acl-short.49/)),
EDCT ([arXiv:2510.00047v1](https://arxiv.org/abs/2510.00047v1)), EDCT-Bench
([arXiv:2609.17953v1](https://arxiv.org/abs/2609.17953v1)), INFACT
([arXiv:2603.11481v1](https://arxiv.org/abs/2603.11481v1)), FIFA
([Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.555/)),
and *Causality Matters: How Temporal Information Emerges in Video Language
Models* ([AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37856)).
The arXiv links are pinned to the reviewed versions. Menzat et al.'s arXiv
record reports conference acceptance, while EDCT, EDCT-Bench, and INFACT are
identified as arXiv preprints here.

The papers operationalize different questions. Menzat et al. adapt CT/CCT to
single-object image removals on A-OKVQA and SNLI-VE: vCCT relates mention of an
edited object to the size of a candidate-normalized answer-distribution shift,
and includes no-op reconstruction controls. This is a close precedent for
behavioral explanation-faithfulness testing in VLMs, so the general idea of
testing whether visual evidence cited by an explanation tracks intervention
impact is not new. Its reported evaluation is image-based and uses edited
objects, explanation mentions, and distribution shifts rather than a
video-temporal claim interval, equal-count rank-matched frame-mask control,
and the project's frozen gold-versus-condition-specific-best-foil margin
contrast (the selected foil can differ across conditions). These
differences motivate only a possible distinct operational question; they do
not support a novelty or priority claim. Siegel et al.'s NLP CCT experiments
insert adjectives or adverbs into input text, then relate changes in the full
label distribution to whether the explanation mentions the inserted words.
EDCT and its 2026 benchmark use image edits tied to cited concepts and assess
answer/explanation consistency. EDCT-Bench comprises 300 manually curated
image-question pairs; its human edit-validity and CCS comparison covers a
128-intervention validation subset, not all 300 pairs. INFACT probes video-level
temporal reliability with evidence corruption and temporal shuffling or
reversal; the authors characterize these operators as proxies and do not
claim that they localize the cues driving a particular answer. FIFA assesses
video-text factual-support consistency by extracting atomic/event facts from
generated text, verifying them with a VideoQA model, and aggregating support
through a dependency graph. The AAAI paper studies model-internal temporal
pathways under positional-encoding and sequence interventions. None of these
observations supplies a measurement for this project's model/data or resolves
the validity of its planned human claim mappings.

The project's possible operational distinction in the frozen proposal is a
reviewer-checked answer claim fixed before scoring and mapped to an interval,
then a teacher-forced gold-versus-condition-specific-best-foil log-margin
contrast between masking that interval's
sampled frames and masking an equal-count temporally rank-matched control.
This would measure a post-hoc input-intervention behavior contrast under a
particular mask and score. It would not establish that the generated claim
faithfully reports the model's internal reasoning, prove the claim true, or
identify a causal mechanism. It is narrower than a full output-distribution
shift measure; the best-foil competitor may differ by condition, and masking
may create out-of-distribution inputs. The direct
visual CCT precedent substantially narrows any plausible contribution to the
video-temporal estimand and its controls; that difference alone does not
establish priority or novelty. Keep the primary protocol frozen and make no
novelty claim until a broader related-work and advisor review is complete.
The human, rights,
timing, target/SESOI, and authenticated-run gates remain unresolved; therefore
no scoring follows from this literature review.

An independent audit corrected three wording issues: EDCT-Bench's human
validation is limited to a 128-intervention subset; the reviewed NLP CCT
experiments insert adjectives/adverbs rather than using an unspecified text
perturbation; and FIFA is described as video-text factual-support consistency
rather than as a factual-precision-only metric. Versioned arXiv citations
were pinned above. The auditor agreed that the stated difference from the
frozen estimand and the no-novelty conclusion are appropriately limited.

Faithfulness progress remains **0 requested / 0 effective / 0 new direct
effect estimates** in this window. No cohort, exclusions, inference failures,
effect interval, hypothesis test, or multiplicity result was generated here.
The next useful work is an independent audit of this comparison against the
repository's frozen estimand and cited primary-source claims, then continue
with independent executable gate/reproducibility work. Do not start GPU
scoring or revise the primary metric based on this post-hoc literature scan.

### Bounded search for closer video-temporal faithfulness studies — 16:00–16:02 JST

An independent primary-source scan looked for work closer to the full
claim-interval and frame-intervention design. [Javadov et al. (arXiv
v2)](https://arxiv.org/html/2609.07596v2), whose record reports acceptance at
the VLM4RWD NeurIPS 2026 workshop, elicits VLM self-reports about modality
dependence and compares them with executed modality ablation/restoration
behavior. This establishes behavioral testing of self-explanations as a
neighboring direction; its interventions are modality/view-level, and its
outcomes are answer agreement/change and sufficiency measures rather than
claim-span masking or gold-vs-foil margins. [Cao et al. (arXiv
v2)](https://arxiv.org/html/2608.03160v2) hold a user-provided temporal-order
claim fixed and alter event order or sampling offset. Crucially, at a fixed
frame budget their paper also compares removing the event frames with removing
the same number of other frames. This is a close video-temporal evidence
control precedent and rules out a broad claim to be first to test fixed-claim
responses under temporal evidence removal with a matched-count frame control.
The claim is user-supplied, however; its intervention does not mask a
reviewed interval in a model-generated explanation, and its outcomes are
claim-acceptance/correction/sycophancy rates rather than a gold-answer
log-margin contrast.

Other adjacent sources include [Song et al.](https://arxiv.org/abs/2609.00830v1)
(arXiv record identifies EMNLP 2026), which evaluates attention-ranked
visual-token ablation through comprehensiveness/sufficiency;
[EG-VQA](https://arxiv.org/abs/2606.24797v2), which scores temporal/semantic
alignment of predicted textual evidence; [Evidence-Backed Video
QA](https://arxiv.org/abs/2607.11862), with answers plus temporal
segments/masklets (arXiv record lists ECCV 2026); and
[CausalPhys](https://arxiv.org/abs/2606.05966v1), which compares video/image
QA rationales to expert-annotated causal graphs (arXiv record reports KDD
2026 acceptance). [Kumar's timestamp-citation verifier](https://arxiv.org/abs/2608.15574v1)
is an auxiliary preprint on checking answer claims against cited frames, not a
direct intervention test of the generator. The withdrawn
[step-level grounding-faithfulness preprint](https://arxiv.org/abs/2603.06828v2)
was excluded as evidence for a supported result.

The limited search found no single study combining a model-generated,
reviewed answer claim's temporal interval, claim-frame masking, equal-count
rank-matched temporal control, and condition-specific gold-versus-best-foil
teacher-forced margin contrast. This is a bounded search result, not a
systematic review or proof of novelty/priority. It supports keeping claims
narrow and requiring advisor-level related-work review before any novelty
statement. No protocol or metric was changed.

The independent reviewer checked the source pages and paper descriptions and
explicitly distinguished verified estimands from the bounded-search
non-finding. No project data, model, CUDA, or faithfulness outcomes were
used. Direct faithfulness remains **0 requested / 0 effective / 0 new effect
estimates**.

### Common alternative model-weight loader audit — 16:01–16:03 JST

The earlier loader inventory recognized Hugging Face `.from_pretrained(...)`
calls but did not test common alternative weight APIs. Extended
`scripts/test_closed_research_branches.py` to classify `torch.load`,
`torch.jit.load`, safetensors `load_file`/`safe_open`, and `load_state_dict`,
including straightforward direct-import aliases. The current source scan found
31 `.from_pretrained`-using scripts: 17 retired real-data inference CLIs
stopped by the early guard, one direct scorer controlled by the fail-closed
preflight consumer, and 13 allowlisted engineering/runtime diagnostics. No
additional alternative weight-loading API was found in active scripts.

The first focused tests exposed a flaw in the new detector's handling of
unaliased dotted imports (`import torch.jit` and `import safetensors.torch`):
it incorrectly treated the submodule as the name Python binds. Corrected the
binding normalization and added regression fixtures for those forms plus
`from safetensors.torch import load_file`. The independent code reviewer
confirmed the reported alias gap is closed and found no further critical
alias defect. The helper remains a static, syntax-based inventory: it does
not follow assignment aliases, star imports, dynamic imports, `getattr`, or
custom loader wrappers, and suffix-matching `load_state_dict` can conservatively
flag unrelated objects with the same method name. Independent source review is
still required for those patterns.

Focused guard tests pass **6/6**; the full CPU script suite passes
**828/828** in 9.130 s. `py_compile` and `git diff --check` pass. Test fixture
diagnostics and Transformers deprecation warnings were non-failing. No model
weights, project data, or CUDA were used. This expands future regression
coverage but does not authorize or execute any inference. Faithfulness remains
**0 requested / 0 effective / 0 new estimates**.

The reviewed guard-inventory test source SHA-256 is
`cc386b9f7e36d5bbc46dbf69ec5b7f6e4e8f14e377c5ba6c70b70024b963de15`.

### Margin estimand algebra and interpretation audit — 16:04–16:05 JST

A formula-level review of the frozen gold-vs-condition-specific-best-foil
contrast found that the protocol stated that the foil may change by condition
but did not spell out how this affects interpretation. Writing the primary
contrast as `d_i = Δ_gold - Δ_best-foil`, where each delta is the
control-minus-claim-mask log-probability change, shows that positive `d_i`
may result from lower gold-label probability, higher best-foil probability, or
both. It is therefore a margin change, not necessarily a gold-probability
drop by itself.

An arithmetic-only example illustrates this: control gold/best-foil
log-probabilities `0.0/-1.0` give margin `1.0`; claim-mask values
`-0.1/+0.2` give margin `-0.3`, with the best foil changing. Thus
`d=1.3 = 0.1 - (-1.2)`: only `0.1` nat comes from the gold score decrease,
while `1.2` nat comes from the best-foil increase. These are deliberately
synthetic values, not model outputs or project results.

Added this decomposition and interpretation limit to the schema-v13 protocol
contract and the dated protocol-design note. The primary contrast, score,
cohort, and hypotheses are unchanged; no new metric or test was added. This
clarifies what future outcomes can support and prevents describing the
contrast as gold-label probability drop alone. No dataset, model, or GPU was
used. Faithfulness remains **0 requested / 0 effective / 0 new direct effect
estimates**.

An independent reviewer separately expanded the condition-specific margins
and confirmed the sign and arithmetic: `S_control - S_claimmask` equals
`(L_gold,control - L_gold,claimmask) - (L_bestfoil,control -
L_bestfoil,claimmask)`. For the synthetic example, the gold contribution is
`+0.1` nat and the foil contribution is `-(-1.2) = +1.2` nat, totaling
`+1.3` nat. The reviewer confirmed that `S_full` cancels, the primary score,
cohort, and hypotheses remain unchanged, and the added decomposition is
interpretive only.

### Independent execution-gate and timing-path audit — 16:05–16:09 JST

Two independent source-only reviews examined the pre-scoring gates and
repository-defined scoring entrypoints. I then checked the cited source paths
and ran the three focused CPU suites. The audits found no repository-defined
claim-mask CLI path that can reach real-data scoring under the current
protocol. In particular, `require_execution_preflight.py` accepts no report:
it rejects both the current schema-v1 cross-gate report and unknown/future
schemas, including a fabricated `ready`/`granted` document. The direct scorer
calls it before reading the sample file or initializing the video processor
and model. The publisher checks it before invoking its analysis runner.
The publisher's earlier existence check may inspect a path, but does not read
sample contents or score them.

The schema-v13 cross-gate remains fixed to `blocked` /
`not_granted`, even when structural components pass. The synthetic timing
preflight is not connected to that gate or the scorer and also returns
`execution_authorization=not_granted`; its pixel hashes are input witnesses,
not authenticated evidence from a production decode. The timing helpers and
prior PyAV→processor synthetic integration establish only that synthetic
code-path behavior, not source-video PTS, annotation-clock calibration, or
production pixel identity. Human claim review, per-item media/annotation
rights, inferential target/dependence/SESOI, production PTS/pixel provenance,
and an authenticated run-bound gate therefore remain separate blockers.

The source-level audits did not inspect protected untracked data, execute
inference, or run tests. My independent verification ran
`PYTHONPATH=scripts .venv/bin/python -m unittest
test_require_execution_preflight test_preflight_claim_mask_gates
test_preflight_claim_mask_timing -v`: **25/25 passed in 3.102 s**. These tests
cover fabricated/unsupported reports, the scorer's stop-before-sample-read
behavior, structural cross-gate refusal, and synthetic timing failure cases.
This is executable-gate verification only; it supplies no model or
faithfulness result. The audited source hashes are:

| File | SHA-256 |
|---|---|
| `scripts/require_execution_preflight.py` | `48154e66a769a6a49eccbe0f22a29b08bb4b9cd52abb53912062261bfa5f4cd5` |
| `scripts/necessity_choice_margin_test.py` | `66342b9169af203437ff14c526753af10e249d110da41d175d43a4b86cb3cffe` |
| `scripts/run_necessity_choice_margin_test_and_publish.py` | `936e43c7020ba4ccaeaa8ebeb889ff1cf14c5a8930bffaa6abf80cfcbb4b679d` |
| `scripts/preflight_claim_mask_gates.py` | `45cf2005926e1922e477fd5ad549514e4d0af089659272a96e10541daa1bb328` |
| `scripts/preflight_claim_mask_timing.py` | `1a3c9a02071334400438e01c3347dbfebecf1050595d25a9b52f5c259e698f15` |
| `scripts/presentation_time_utils.py` | `9aa1e737f985cffc781a0667d2e4cccb1e7bd1d8222aac743e55a5c3fd52f093` |
| `scripts/validate_claim_mask_protocol.py` | `1b4c072ca19624d303dd7c4bda7a2becc937f17a64cc08781289d83c5410319e` |

The remaining boundary is explicit: these checks establish behavior of the
current repository CLIs and tests, not a security boundary against arbitrary
Python code, manual model use, or future code changes. Do not implement a
permissive grant schema until the upstream scientific and evidence gates are
actually resolved. Faithfulness remains **0 requested / 0 effective / 0 new
direct effect estimates**; no GPU workload was scientifically justified by
this audit.

### Independent cluster/tail sensitivity artifact audit — 16:09–16:12 JST

To check whether the already published synthetic dependence analysis had
internally consistent provenance and summaries, I audited
`results/claim_mask_tail_coverage_sensitivity_20261015_02.json` against its
generator and dependency. The artifact declares 288 conditions and 5,760,000
synthetic datasets (80 videos per dataset, 20,000 repetitions per condition).
All 288 condition keys are unique and match the declared six layouts, six ICC
values, and component-distribution grid (at ICC zero, the inactive cluster
component has one canonical label). The artifact's generator SHA-256
`ad6a1b520da829f7f8125c46e381b0a0071ec423d18e7bf36c29281eedf33727` and
dependency SHA-256
`6fa22da3a6a57291f824250f8ca2eec735a467b69589b5d13d59a4a922a75ddf` match
the current source bytes. The artifact SHA-256 is
`182f3a8763ec7b0c4e80da610eba080dab4cefe2a3ecd9fe1a9da967c564e786`.

I independently recomputed all 288 coverage proportions, Bernoulli Monte
Carlo standard errors, Wilson intervals, random-intercept design effects,
and analytic sampling standard deviations; each matched to floating-point
precision. A separate reviewer also verified the declared mean CI width
against `2 * t(.975, N-1) * mean_naive_estimated_se` across the grid. Neither
the simulations nor the artifact were regenerated.

For the normal/normal ICC=.20 cells, the earlier 50,000-repetition Gaussian
grid and newer 20,000-repetition mixed-tail grid are numerically compatible:

| Fixed layout | Earlier coverage (Wilson 95% interval) | Newer coverage (Wilson 95% interval) | Difference / combined MCSE |
|---|---:|---:|---:|
| Balanced, 10 clusters | 0.79250 [0.78892, 0.79603] | 0.79745 [0.79182, 0.80296] | 1.47 |
| Unequal, 10 clusters | 0.77046 [0.76675, 0.77413] | 0.77260 [0.76674, 0.77836] | 0.61 |

The Wilson intervals overlap and analytic SDs agree for both layouts. The
grids use independent seeds and different repetition counts; this is an
internal consistency check, not paired replication or exact reproducibility.
Only these normal-random-intercept cells have an earlier comparable condition.
The full sensitivity grid assumes fixed cluster sizes and independent,
unit-variance random-intercept and residual components. ICC/distribution
values are scenarios, not estimates from project data; Wilson/Monte Carlo
uncertainty concerns simulation error only. These outputs do not establish
real-cohort interval coverage, dependence, SESOI, or an appropriate inference
method. Faithfulness remains **0 requested / 0 effective / 0 new direct
effect estimates**.
