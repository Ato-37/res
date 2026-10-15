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

### Overview/contract consistency and committed artifact provenance — 16:12–16:14 JST

Updated the high-level research overview to state the frozen primary's
interpretive limit: because each condition selects its own best foil, a
positive gold-vs-foil margin difference can include gold-score decline,
best-foil-score increase, or both. The overview now points to the protocol
contract and this log for the derivation. No score, cohort, estimand, or result
changed.

As a final provenance cross-check for the tail-sensitivity audit, the
committed result blob at `84959ba0251130204fc810448e958e8a4ebb759d`
(`Bind tail simulation dependency provenance`, 2026-10-15 12:20:58 JST) has
the same SHA-256 as the local artifact:
`182f3a8763ec7b0c4e80da610eba080dab4cefe2a3ecd9fe1a9da967c564e786`.
The generator and its directly imported dependency also match the hashes
embedded in that artifact. This explanatory update changes no measurement;
faithfulness remains **0 requested / 0 effective / 0 new direct effect
estimates**.

### Two-pass PyAV sampled-frame adapter and independent re-audit — 16:17–16:21 JST

Implemented `scripts/claim_mask_video_decoder.py` as a reusable, non-authorizing
video decode adapter. It decodes one video stream twice, requires valid unique
monotonic presentation timestamps and stable time base/pixel specification,
chooses endpoint-inclusive uniform decoded-frame indices, and refuses sampled
PTS grids outside the caller's declared tolerance. The second pass returns the
RGB PIL frame buffers after checking every PTS and the selected frame pixel
hashes against pass one. A separate boundary helper rechecks those frame
buffers immediately before a caller passes them onward. Metadata explicitly
states that annotation-clock calibration is false and execution authorization
is `not_granted`.

An independent review found that hashing a path and reopening it for each pass
could bind the source digest to different bytes if the path target changed
between decodes. The implementation now opens one seekable file handle, hashes
that handle before/after, and seeks/reuses it for both PyAV passes. A synthetic
regression test replaces the path after pass one and confirms that both passes
still decode the pinned original object and that its source hash is recorded.
The RGB sequence digest metadata key is explicitly named
`pass1_decoded_rgb_sequence_sha256`; pass two verifies all timestamps and the
selected inference frames, rather than claiming to hash-compare every frame's
pixels across both passes.

The independent re-audit confirmed the path-replacement repair and all seven
adapter tests. Remaining limits: in-place hostile mutation and restoration of
the same inode during decoding is not prevented; global cohort preflight and
one frozen tolerance across all items belong to a future caller; no production
annotation-clock calibration or Qwen scoring path is connected. No protected
manifests/media were opened, and no model or GPU was used. The combined
decoder/PTS/processor boundary tests passed **24/24**; the full script test
suite passed **835/835 in 9.678 s**. Faithfulness remains **0 requested / 0
effective / 0 new direct effect estimates**. This is a provenance and temporal
measurement repair validated only on synthetic media, not a faithfulness
result or evidence that real-video sampling is correct.

### Real-video PyAV→Qwen CPU/GPU feasibility experiment — 16:23–16:30 JST

In response to the user's explicit direction to perform an experiment with real
video and to set aside rights/inferential-design gates for this exploratory
work, fixed the already locally retained NOAA EX2503 dive03 source before
outcomes. The clip is public-domain per the prior verified NOAA source record;
this run used no dataset QA, answer text, annotation, or human-attestation
claim. It is an input-path feasibility experiment, not an efficacy test.

Added `scripts/run_claim_mask_real_video_gpu_smoke.py`. The runner verifies the
fixed media digest both before decode and against the digest of the same pinned
file handle used by PyAV, decodes twice, checks all 8,992 displayed-frame PTS,
and selects four endpoint-inclusive frames on a uniform grid with a frozen
1 ms tolerance. It passes those exact PIL buffers through the installed Qwen
video utility and processor, verifies returned frame count/order metadata,
FPS, total frame count, mean-color ordering signatures, video-token/grid
count, expected temporal-grid size, and seconds-per-grid. Mean-color signatures
are an ordering diagnostic that can collide; they are not an exact or
cryptographic pixel-identity proof after Qwen preprocessing. Annotation-clock
calibration remains false. The GPU branch then performs one Qwen forward with
finite logits and all attention layers, with no generation or scored answer.

Two initial CPU-only CLI attempts failed due runner defects (an unconditional
CUDA tensor transfer, then a model-token reference before model loading); both
left only privacy-safe failed-attempt records. After fixing those defects, the
real-video CPU processor preflight passed: 1/1 video, 4/4 frames, 8,992 source
frames, processor grid `[2,24,44]`, 528 video tokens, and
`seconds_per_grid=199.9998016357422`. The fixed source media SHA-256 is
`13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126`.

The bounded GPU run used RTX 5070 Ti, CUDA 12.8, torch 2.11.0+cu128, BF16,
4-bit quantization, eager attention, and the pinned Qwen2.5-VL-7B revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Pre-run free VRAM was 14,875 MiB.
It completed **1/1 video, 4/4 frames, 28/28 finite attention layers**, finite
logits, grid `[2,24,44]`, 528 video tokens, peak CUDA allocation **6.647 GiB**,
and **72.0 seconds** elapsed including the full two-pass CPU decode and model
load. Output is `results/claim_mask_real_video_gpu_smoke_20261015_01.json`,
SHA-256 `ee7020b3cc5f9a55defa5c3824d3cf2d45134d0e9684d8377fcce9b3915ad327`.
Both source hashes embedded in the output independently match the runner and
decoder files. The artifact contains no absolute local paths, raw media,
annotations, QA, generated text, or answer-level scores. CPU preflight summary
is `results/claim_mask_real_video_processor_cpu_20261015_03.json`, SHA-256
`f2d43900b487c1028a3a91de2ca6f36250adcc2be34ab9243e46c36395966b81`.
Post-run independent artifact audit is pending.

This establishes that this fixed real video traverses the new PyAV→Qwen path
and produces numerically finite tensors under this four-frame configuration.
It does not establish semantic frame correctness, annotation alignment,
grounding, QA accuracy, intervention behavior, faithfulness, or generality.
There is no treatment/control contrast and therefore no effect estimate,
confidence interval, hypothesis test, or multiplicity correction. Current
window faithfulness progress remains **0 requested / 0 effective direct
faithfulness samples / 0 new direct effect estimates**; the one real video is a
technical feasibility input, not an effect-estimation sample.

The independent artifact reviewer recomputed the frame-index/time mapping and
token count, confirmed artifact SHA-256 and embedded source hashes, and found
no raw media, annotation, answer, generated text, or absolute path. The
measured forward itself was 5.269 seconds. The runner's `gpu_free_before_gib`
is 14.6572 GiB (Torch CUDA memory API); a separate `nvidia-smi` sample
immediately before launch showed 14,875 MiB. These are separate reads taken at
different instants, differing by about 135 MiB, and are not treated as the
same snapshot. PyAV/FFmpeg versions and decoded time-base/deviation are not
embedded in this first GPU artifact; source hashes and the separately audited
CPU preflight bind its code path, but those fields should be added if this
runner is extended. The tracked HEAD at launch was
`43add2b1aba9a89c626bba0c5401ead979ebf88f`; the newly added runner/decoder
were untracked at that instant and are bound by their embedded SHA-256 values,
listed here pending their scoped commit:

| Source | SHA-256 at run |
|---|---|
| `scripts/claim_mask_video_decoder.py` | `c0226f23aa073317a34d982a40cdf199820e502ce6ae992f8fc19082991f6516` |
| `scripts/run_claim_mask_real_video_gpu_smoke.py` | `12be80b9d2661d213856c1ee1159a70fae4d462f393ceb295539c9def8961e43` |

An initial full test discovery found two expected-list omissions in the static
closed-branch loader guard because the new engineering runner had not yet been
classified. Added it to the guard's allowed engineering loaders (without
changing any closed research branch); the focused guard passed **6/6** and the
full script suite then passed **835/835 in 9.171 s**. The previous full run's
two guard failures therefore reflect this missing classification, now fixed.

### Fixed-spatial 4/8/16-frame real-video capacity sweep — in progress

To estimate whether the actual-video path can safely process longer temporal
inputs, added `scripts/run_claim_mask_real_video_capacity_sweep.py`. This is a
resource-feasibility diagnostic on the same fixed NOAA clip and pinned Qwen
revision. It performs one model load and one forward for each of 4, 8, and 16
endpoint-inclusive frames, in ascending order; uses 4-bit/BF16, fixed per-frame
pixel limits, does not request attention outputs, does not generate text, and
disables KV caching. It has predeclared host-memory and free-VRAM guards, and
requires at least 6 GiB free VRAM before the 16-frame condition. A failure
records only the stage and completed safe summaries. No QA, annotation,
treatment/control, or faithfulness score is read or produced.

An independent source audit found two missing guards before GPU execution:
the runner now rejects spatial-grid differences across conditions and checks
host available memory before every forward after model loading. The sampler
tolerance is fixed at half the source CFR frame period (1001/60000 s) solely to
allow frame-index quantization in this capacity diagnostic; it must not be
reused as a claim-mask measurement tolerance. Focused CPU tests pass 4/4.

The real-video CPU preflight passed all three conditions: each decoded 8,992
source frames; effective sample counts were 4/4, 8/8, 16/16; all spatial grids
were `[24,44]`; temporal grids were 2/4/8; and Qwen video-token counts were
528/1,056/2,112. Per-condition decode+CPU processor preparation took 31.06,
30.51, and 30.68 seconds. Maximum sampled-grid deviations were 0, 14.3, and
13.3 ms, within the frozen 16.68 ms diagnostic tolerance. Result:
`results/claim_mask_real_video_capacity_preflight_20261015_02.json`, SHA-256
`641afde174166f939aca3f0437192f27627b97bfa7ce3531497cd9ac92881d08`; the
runner SHA-256 at preflight was
`43fd3e8f1f2e60ebe3848bf5ec1be832fe2d92bb39e1b4c47665da8d3c902f89`.
These data confirm processor/grid consistency only. The bounded GPU resource
sweep is the next action; runtime, VRAM scaling, and 16-frame feasibility are
not yet observed. User-authorized per-command elevation will be used for the
GPU command after source commit and live resource recheck.

This branch measures input/resource feasibility, not relevance or explanation
faithfulness. Frame count is not evidence coverage, and one clip with one
forward per condition cannot establish QA performance, sampling quality, or
optimal frame count. Current faithfulness progress remains **0 requested / 0
effective direct faithfulness samples / 0 new direct effect estimates**.

### Real-video temporal input capacity GPU sweep — 16:42–16:44 JST

Ran the committed `76c4f0996081377b323f849c668a76dcb9b510fe` runner on the
same NOAA source and pinned Qwen2.5-VL-7B checkpoint. All 3/3 conditions
completed with finite logits and zero failures. The source was fully decoded
8,992 frames per condition; endpoint-inclusive effective counts were 4/4,
8/8, and 16/16. Spatial processor grid stayed `[24,44]`; temporal grid was
2/4/8 and video tokens 528/1,056/2,112. Total input token counts were
560/1,088/2,144. The CPU preflight and GPU artifact agree on these counts,
grids, and sampled indices.

Peak CUDA allocated memory was 6.049/6.236/7.237 GiB and peak reserved memory
6.375/6.797/8.102 GiB. Free VRAM immediately before forwards was
8.683/8.657/8.657 GiB; host `MemAvailable` was 12.991/11.934/11.915 GiB.
The 16-frame condition therefore passed the predeclared 6 GiB free-VRAM and
8 GiB host-memory guards. Single-forward durations were 1.928/0.743/1.875 s;
they are non-monotonic and each has only one observation, so they do not
support a latency scaling estimate. Sparse GPU utilization samples (2–4 per
condition) had mean 11.0/51.5/48.25%, maxima 18/76/99%, and mean memory-busy
time 1.25/17.5/18.75% (maxima 2/27/40%). These are descriptive telemetry, not
stable utilization estimates or proof of memory-bandwidth saturation.

End-to-end elapsed time was 103.08 s, including 9.69 s model load. The three
CPU decode/processor preparations took 29.26/29.60/29.25 s (88.11 s total),
while GPU forwards totalled 4.55 s. Thus this measured run spent far more time
repeatedly decoding/preparing the same clip than doing model forwards. This
supports optimizing shared video decode across frame-count conditions before
scaling multi-video experiments; it does not imply anything about task quality.
The artifact SHA-256 is
`16936ac563c27559cc980d9e155d427924d5cf0c096bad7f104129040a0b47e3`. Its
source commit equals the runner's pre-run HEAD, and runner/decoder/model/helper
hashes independently match. Reviewer re-computed the counts and token-grid
relationship, confirmed finite outputs and privacy-safe fields, and found no
material inconsistency. The artifact uses no QA, annotation, generated text,
score, or raw media.

Interpretation is limited to this one NOAA clip, this pinned model/configuration,
and one forward at each frame count. At 16 frames, this exact path fits within
the observed GPU/RAM headroom; evidence coverage, frame quality, answer
performance, optimal sampling, and explanation faithfulness remain untested.
No treatment/control contrast exists, so there is no effect estimate, CI,
hypothesis test, or multiplicity correction. Faithfulness progress remains **0
requested / 0 effective direct faithfulness samples / 0 new direct effect
estimates**. The final GPU artifact supersedes and permits removal of the
temporary ignored CPU preflight summary `_02.json` after publication.

### Workspace cleanup audit — 16:44–16:50 JST

The user requested directory cleanup because generated code/results have
accumulated. A read-only inventory found `results/` is a flat, Git-ignored
working area (23 MiB at inspection) and `scripts/__pycache__/` contained 10 MiB
of regenerable Python bytecode. I did not reorganize result paths because the
standalone experiment CLIs and logs rely on `results/` filenames, and moving
them would break reproducibility without reducing data volume. I removed only
clearly disposable local items: the two failed pre-Qwen CPU attempts
`claim_mask_real_video_processor_cpu_20261015_01/02.json` (both failures are
already described above; passing `_03` remains); superseded first capacity
preflight `_01.json` (the audited corrected preflight `_02` remains); exact
byte-identical duplicate `qwen_temporal_position_repeatability_20261015_03.json`
(identical to `_02`, same SHA-256); empty
`qwen_gpu_utilization_20261015_03.csv`; and regenerated `scripts/__pycache__/`.
These removals affect only ignored local artifacts. An independent read-only
inventory found other exact duplicate outputs, so I compared their hashes and
checked repository references before cleanup.
For exact duplicate untracked split/video-mapping summaries from 2026-10-14, I
kept the `_03` files cited by the current experiment log and removed `_01/_02`
duplicates; for thesis exports, I kept the `thesis_summary_final` CSV/SVG and
removed byte-identical untracked alternate names. Other separately numbered
memory and GPU measurements remain because distinct hashes alone do not show
redundancy. A follow-up full hash scan found the only remaining exact duplicate
was the tracked dated Ubuntu forest plot; after confirming no current
reference and that `thesis_summary_final_forest.svg` is the linked canonical
copy, I removed the duplicate tracked path while preserving its Git history.
No confirmatory raw/source result was removed. Both temporary shared-decoder CPU preflight
JSONs were removed after the schema-v2 GPU artifact, with matching grids and
counts, was pushed and verified live; their checksums and measurements remain
in this log.

### Shared multi-condition decoder optimization — in progress

The capacity measurement showed that 88.11 s of its three condition-level
CPU decode+processor stages were spent repeating the same video's two-pass
decoder. To remove that measured redundancy while preserving provenance,
generalized `decode_sampled_video` into a compatibility wrapper over
`decode_sampled_video_sets`. The shared API independently computes each
condition's endpoint-inclusive index set, sample rate and PTS tolerance,
decodes once for the complete PTS/RGB digest, then makes one second full pass
that rechecks the whole PTS/time-base/pixel-spec sequence and RGB hashes for
the deduplicated union of selected indices. It reconstructs each returned
condition in its own ordered index sequence and retains source-handle hashes
before and after both passes. The Qwen boundary still verifies each condition's
exact frame buffers. This removes repeated video scans across conditions, not
the two-pass integrity check or condition-specific processor work.

The actual NOAA source passed the revised CPU-only 4/8/16 preflight. It decoded
the 8,992-frame PTS/RGB sequence once across conditions and reproduced the
same per-condition frames, PTS grids, `[24,44]` spatial grids, temporal grids
2/4/8, and 528/1,056/2,112 video tokens. Shared two-pass decode measured
29.598 s in the first revised preflight and 29.051 s in a second preflight;
Qwen CPU processing took 0.085/0.032/0.092 s and 0.039/0.034/0.114 s,
respectively. The prior single run's 88.11 s covered three separate
decode+processor stages. The revised second run took 29.24 s for shared decode
plus all processor stages, a descriptive 66.8% reduction for this one clip.
This is two before/after observations, not a controlled repeated benchmark or
general performance guarantee. The latest local CPU artifact is
`results/claim_mask_real_video_capacity_shared_decode_preflight_20261015_02.json`,
with schema 2, decoder SHA-256
`e5ddd9e1e665a849add91d5cf010bbe4bf96a3609b8d7792b542de7288a5cd9c`, and runner
SHA-256 `19abfc9a91b3c7fc5fd4426344cfdc80bbc021e70936589bdfd865312cdd47dc`.

Focused decoder/capacity tests pass 15/15 and the full script suite passes
843/843. The independent auditor confirmed source hashes match the latest CPU
preflight, per-condition PTS checks, selected-union RGB checks, schema version
2 for success/failure records, and CPU artifact consistency. The latest CPU
preflight artifact SHA-256 is
`9a5dbc4c0a768be16b09e71032b0a6488ab6ed1f44a38d2c32c17ceb15f28a62`.

### Shared-decoder real-video GPU recheck — 16:54–16:55 JST

Ran the schema-v2 shared-decoder sweep on the committed source
`482cc032b39b954b95a3b3298de79cbd13da6a09`. All 3/3 GPU conditions passed;
logits were finite and the output grids/tokens/sample indices matched both
CPU preflight results. Peak allocated/reserved memory remained
6.049/6.375, 6.236/6.797, and 7.237/8.102 GiB for 4/8/16 frames. Free VRAM
before forwards was 8.683/8.657/8.657 GiB and host available memory
12.966/11.956/11.962 GiB. Thus 16 frames still fit the fixed 224-equivalent
spatial setup on this observed machine state.

The runner's shared two-pass decode took 28.989 s; total run time was 42.802 s
including 7.163 s model load. Compared with the previous 103.077 s run, the
observed elapsed time was 58.5% shorter, while the shared decoder alone
replaced three roughly 29-second decode/preparation stages. Forward times
were 3.137/0.734/1.519 s (one measurement each), so latency still does not
scale monotonically. Sparse utilization samples averaged 18.5/27.5/73.3% with
maxima 37/46/99%; memory-busy averages were 4.5/7.0/27.0% (max 13/13/39%).
This confirms GPU kernels ran and that the longest condition had higher sampled
activity, but the 2–6 samples per short forward cannot establish stable GPU
utilization or memory-bandwidth behavior. The prior and current runs are not a
controlled benchmark series, so the elapsed reduction is descriptive only.

The privacy-safe schema-v2 artifact SHA-256 is
`a480a921d820c9d3425b2bf61e8fc5bc495b38097b339d1270637c654da08ab1`; its
source commit and runner, decoder, model config, and helper hashes were
independently matched to the checked-out files and CPU preflight. The reviewer
verified the three conditions, finite logits, resource counters, and absence
of raw/private data. It contains no QA, annotations, generated text, scores,
or faithfulness estimate. This remains a feasibility measurement on one video
and is not evidence for temporal evidence quality, answer validity, or
faithfulness. The current session still has **0 direct faithfulness samples
and 0 direct faithfulness effect estimates**.

### Follow-up repository cleanup — 17:06 JST

Rechecked the repository after the earlier cleanup. All 279 tracked files in
`results/` have distinct SHA-256 hashes; no exact duplicate remains. The flat
paths are referenced by experiment logs and standalone runners, so I retained
the result directory layout. A read-only scan found four unreferenced standalone
scripts. Independent review showed `analyze_necessity_mask.py` was the only
legacy-schema analyzer with no tracked input artifact, tests, source imports,
experiment-log references, or current thesis output; its one-sample t summary is
not part of the current black/blur margin analysis pipeline. I initially removed
it, then checked the ignored local result area before considering cleanup done.
That direct inventory found local `necessity_mask_confirm_heldout_v2.json` and
`necessity_mask_smoke.json` files in the legacy schema. The analyzer is the
available reproduction path for those local results, so I restored it in a
follow-up commit. No local result or private input was removed. The other three
unreferenced scripts remain because one is the only checkpoint-fetch helper,
one is named in tracked host/guest memory result provenance, and one is the
only synthetic MP4 encode/decode smoke whose source hash is recorded in past
results. Regenerable cache files under the private untracked `NExT-GQA/` tree
were left untouched. After this cleanup, no additional safe tracked deletion
was identified. Faithfulness progress remains **0 new direct samples / 0 new
direct effect estimates**; the retained prior studies have their distinct
limits described above.

The full CPU unittest suite passed **843/843** after removal; `pytest` is not
installed in this `.venv`, so verification used the repository's unittest
discovery command. A separate metadata-only profile of the three already-local
NOAA source files found all had the same 8,992-frame 640×360 yuv420p stream,
time base `1/30000`, exact CFR PTS increment `1001/30000`, and
`30000/1001` average rate. Their media SHA-256 values matched the recorded
allowlist. Because the additional clips provide no distinct source-format or
PTS challenge and would still be only engineering feasibility, no extra GPU
capacity run was justified by this check. It would not add a faithfulness
measurement. This decision is outcome-blind and does not inspect QA or review
data.

### Targeted literature update — 17:10 JST

Checked three current primary sources to guide the next faithfulness branch.
INFACT is an ACL 2026 video-LLM benchmark with separate label-invariant
robustness operators and order-changing temporal interventions; its abstract
reports that clean accuracy does not determine reliability under those
interventions. The paper audits assumed label-preserving transformations with
three blinded reviewers on 200 cases per operator. This is useful evidence
that intervention semantics need their own validation, but its Resist Rate and
Temporal Sensitivity Score measure answer reliability/video dependence, not
whether a generated explanation faithfully reports the model's decision basis.
Primary source: https://aclanthology.org/2026.acl-long.2062/ .

ECCV 2026's E-VQA asks models to produce answers plus temporal spans and dense
masklets, with a human-verified grounding benchmark. That is adjacent to
evidence localization, but spatial/temporal grounding accuracy is not the same
estimand as selective dependence on a fixed explanation claim. Source:
https://arxiv.org/abs/2607.11862 . EDCT-Bench (September 2026 arXiv preprint)
derives visual edits from image-VLM explanations and tests answer/explanation
consistency; its image-only scope does not supply a ready video intervention.
Source: https://arxiv.org/abs/2609.17953 .

Decision: retain the claim-linked mask estimand as distinct from these
benchmark-level reliability and grounding outcomes. Do not add a temporal
shuffle experiment until order-sensitive items and the label-changing
counterfactual are human-verified before outcomes; this could be a separate
future estimand rather than a replacement selected after observing results.
The human claim-review artifact remains incomplete, so current direct
faithfulness progress is unchanged at **0 new samples / 0 new effect
estimates**. These papers guide protocol design; they are not measurements on
this project's model or data.

### Clarification: NExT-QA versus NExT-GQA and prior evidence — 17:27 JST

A user pointed out that NextQA may supply evidence intervals. The precise
distinction is that the original NExT-QA release documents question-answer
annotations, while the NExT-GQA extension adds per-question temporal spans in
`gsub_val.json` and `gsub_test.json`. The official NExT-GQA README shows the
video-ID/question-ID keyed span format. This repository already contains a
counts-only temporal-annotation audit (3,358/3,358 validation and 5,553/5,553
test QA rows joined to gsub) and previously ran NExT-GQA interval-based
necessity-mask experiments. Thus it was incorrect to say that no such
interval-annotated data or prior real-video measurements exist.

The prior 60-item NExT-GQA necessity-mask study is evidence of a limited
operational answer-margin response to annotated-interval masking, not a new
result in this follow-up and not proof that generated explanations faithfully
represent model reasoning. Its all-item estimates were black mask +0.296 nat
(95% bootstrap CI [−0.046,+0.615]) and blur +0.360 nat ([−0.035,+0.721]);
the intervals include zero. The baseline-correct subset (n=37) had black
+0.334 nat ([+0.111,+0.581]) and blur +0.456 nat ([+0.172,+0.760]), but these
are subgroup estimates without multiplicity correction, and the cohort is
selected by baseline correctness. Mask-fill OOD, interval-to-token mapping,
and the distinction between QA evidence intervals and claim-specific spans
remain limits. Do not present these as a new independent confirmation.

The independent source audit also found that E-VQA's paper includes a
200-sample evidentiality diagnostic: pixelate annotated evidence masks versus
same-procedure random non-evidence regions, then compare QA-accuracy drops.
The literature review was corrected to describe this intervention and its
boundary: it tests selective QA dependence on annotated regions, not a fixed
generated-claim-to-interval margin estimand. Targeted CPU contract checks
passed 24/24 after setting `PYTHONPATH=scripts`; the first invocation without
that repository import path failed at test discovery and ran no tests.

The local NExT-GQA tree and private sample manifests remain unopened in this
check, in accordance with the repository's protected-data instruction. The
tracked source, public README, prior logs, and published derived results
establish that the annotation format and evaluation path exist. Further work
should resume from those existing outputs and the open temporal-token mapping
question, rather than repeating the completed sample.

### Legacy preflight artifact privacy cleanup — 17:30 JST

The read-only audit of published run provenance found that the unreferenced
tracked `results/necessity_choice_replication_preflight.json` contained
per-sample IDs and absolute local video paths. It was not needed by any tracked
runner or documentation reference, and those fields are not appropriate for a
published derived summary. Removed the artifact from the current tree. Its
aggregate counts were 80 candidates, 72 valid, 7 without matched controls, and
1 without a sampled evidence frame; equivalent high-level candidate coverage
is already described in existing research records. No sample manifest, video,
annotation, or protected untracked file was opened or changed in this cleanup.
Git history was not rewritten, so earlier commits may still contain the
removed artifact; current-branch deletion limits future exposure without
force-pushing or altering published history.

### Tracked result privacy audit — 17:32 JST

A value-redacting scanner over all 265 tracked `results/*.json` files found 33
with sample/video/question identifiers or local paths. Affected artifacts
spanned older evidence-control, head-group, and necessity-mask outputs and
their run manifests; another 12 files had path-only diagnostics. Rewrote 45
tracked JSON artifacts to remove those identifiers, QA/annotation payload
fields, absolute/home paths, and raw command arrays while preserving aggregate
summaries, run counts, numerical per-item score records where available, and
non-local configuration. No protected untracked files were read or changed.
The result-consumer test suite still passes 843/843, including the thesis
summary and faithfulness snapshot consumers. A second scan found zero
remaining identity-key, numeric sample-ID-key, or absolute/home-path signals
among the 265 tracked result JSONs. `git diff --check` passes. Git history was
not rewritten; old published commits retain their original bytes, while the
current branch removes the avoidable identifiers and paths.

### NExT-GQA interval-grid coverage and legacy provenance limit — 17:35 JST

Re-read only the published counts-only annotation/time-grid audit. Validation
has 3,358/3,358 QA rows joined to a gsub span; 3,355 rows are structurally and
duration-valid, containing 3,926 spans, of which 3,854 (98.17%) intersect at
least one of the supplied 32-point `frame2time` timestamps. Test has 5,553/5,553
joins; 5,517 rows and 6,560 spans pass structural/duration checks, with 6,479
(98.77%) spans intersecting that timestamp grid. Median span duration is 4.5 s
on validation and 4.1 s on test. These are deterministic metadata coverage
counts, not model evidence or sampling-based estimates. The grid is the
dataset's 32-point `frame2time` metadata; it does not establish the display PTS
of the exact pixels historically passed to Qwen, nor does it calibrate the
annotation clock origin to decoder PTS.

Independent arithmetic checks over the now-redacted published 60/60 black and
blur replication records reproduce their saved means, SDs, and group counts;
the two conditions have the same record ordering and baseline-correct group
membership. The earlier original run was 55/60 with five failures, so it is not
the source of the 60/60 estimates. The replication's saved result and run
manifest do not contain a sample-manifest content hash; therefore exact cohort
selection cannot now be reverified without the protected private manifest.
This provenance gap limits confidence in the historical result's full
reproducibility even though the saved numerical summaries match their
per-record outputs. No new experiment or effect estimate was produced.

### Local NExT-GQA real-video GPU interval-margin pilot — 18:05 JST

The user authorized local-only operations and asked to move quickly into real
data. A counts-only local audit first found 85 previously unused test videos
with usable QA/interval metadata. The new runner
`scripts/run_nextgqa_gold_interval_margin_pilot.py` deterministically scans
those candidates, caps the cohort at six videos, requires a 1 ms sampled-PTS
uniformity gate and an equal-count non-evidence control, and freezes the QA,
answer labels, sampled pixels/PTS, and mask positions before reading model
scores. The outcome-blind preflight accepted 6/6 selected videos; the scan had
51 decode attempts (45 timing rejections, 6 timing passes), zero decode errors,
and zero selected videos lacking a mask/control pair. All six also passed Qwen
processor geometry and pixel-order checks. The six are a deterministic timing-
eligible subset, not a probability sample.

The local-only GPU run used the pinned Qwen2.5-VL-7B-Instruct checkpoint
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
CUDA 12.8, and the RTX 5070 Ti. It completed 18/18 requested forwards across
full-video, black gold-interval-mask, and equal-count control-mask conditions
for 6/6 videos, with zero failures. The fixed-cohort descriptive contrast was
`control-mask gold-vs-best-foil log-margin − gold-interval-mask margin`:
mean **+2.5833 nat**, sample SD **2.5195**, range **−0.25 to +7.00 nat**.
The full condition selected the correct answer label in 4/6 cases under the
restricted five-label next-token scoring rule. No inferential test, confidence
interval, or population generalization was computed. The pilot took 177.74 s;
peak allocated GPU memory was 7.76 GiB. The private derived JSON is kept only
at `/tmp/nextgqa_gold_interval_result_20261015_1805.json` (0600); its hash is
`8d577aa2601755f5e42f3a69d54dc9371b145a333c227a483b621c2880a57ed5`.
The matching preflight JSON is also local-only and 0600. Runner SHA-256 is
`eaaa9e2d3652ef7465f76efcfa42795f25f160cdd5e00acc7c2c920e5d5d53ac`, cohort
SHA-256 is `a1cc3abd5d16b0dee357688606963f0bb2214ddd9058bc46bb6f38a9e23682ef`,
and preflight file SHA-256 is
`b8b3966afd52793bb186c1b829f0f62b4be3b8ce49a31df8d09c810ac0ee767b`.

Interpretation is limited to an exploratory benchmark answer-margin response
under these two masks. It is **not claim-specific explanation faithfulness**:
no generated-claim grounding was evaluated, the user-confirmed human claim
review is incomplete, and current direct faithfulness progress therefore
remains **0 new samples / 0 new effect estimates**. The independent audit
identified a further material limit: decoder PTS is relative to the first
decoded frame and has not been independently calibrated to the annotation
player/source-edit clock. Regularity of the 16 sampled PTS and agreement with
dataset frame2time metadata do not alone establish that origin. Correction:
the pilot itself did not read or hash `frame2time_test.json`; the cited
proximity comes only from a separate 85-video census and is not a per-cohort
clock calibration. Black masking
can also produce out-of-distribution videos, and the best foil can change by
condition. The positive mean is an observation on six videos, not evidence of
causal localization or statistical significance.

The local runner was hardened to bind score mode to the preflight schema,
purpose, mode, runner and decoder hashes, candidate data hashes, exclusion
manifest digest, frame/tolerance settings, checkpoint revision, and frozen
cohort digest. It caps the cohort at six, checks host memory and free VRAM
before every forward, and writes summaries exclusively with mode 0600. The
complete repository test suite passes **843/843**, `py_compile` passes, and
`git diff --check` passes. No raw data, video, per-sample identifiers, QA text,
or model-generated text was placed in the result. No network publication was
performed.

Next: independently audit whether the six videos' gsub intervals and supplied
frame2time vectors support a defensible annotation-clock mapping, then design a
clock-robust interval membership sensitivity check before treating any mask
contrast as interval evidence. Keep the new pilot exploratory and do not reopen
the closed head/layer/alpha search. Continue separate claim-specific work only
after human claim-to-interval review is complete.

### Pilot cohort accounting correction and fixed-cohort replay — 18:20 JST

An independent audit found that the first preflight's `requested_videos=85`
and one limitation sentence did not reflect the runner's six-video cap. The
runner stopped after the first six eligible videos in deterministic hash order;
it decoded only the scanned prefix. The observed count decomposition is 85
available candidates = 51 scanned + 34 unscanned; among the 51 scanned, 45
failed the fixed 1 ms sampled-PTS regularity gate and six passed. The run had
zero decode failures (51 = 45 + 6) and none of the six lacked an equal-count
control. The initial private result's effect values are unchanged, but its
preflight wording must not be read as a census of all 85 videos.

The runner was corrected to report the cap, scanned/unscanned counts and exact
first-six selection rule, and to state that it does not read `frame2time`.
Score mode now also verifies preflight condition grids, installed package
versions and the pinned model config hash, in addition to the previous input,
cohort, runner/decoder, exclusion, and frame/tolerance bindings. It records
that local-only use was user-authorized without claiming independent rights
verification. The corrected preflight and a same-cohort GPU replay passed:
6/6 triplets and 18/18 forwards, zero failures. Independently compared to the
first run, all six per-video contrasts were exactly equal, so the mean/SD/range
again equal +2.5833/2.5195/[−0.25,+7.00] nat. This is a technical replay of
the same six videos, not an independent replication or added sample. Corrected
replay SHA-256 is
`d2e14a623af0491e91b86a589f5f5ba87a3dc864bd305f01f6e68ae5bbd3ad99`; its
preflight SHA-256 is
`b61b873ec5d10a45edd5b6f8eb1c96685e39592e9148e988230c3770eaff1448` and runner
SHA-256 is `a528cc42303d64ead9bbddd8db67fda8c8c5c3ae1dc26e0c86d87841c3156250`.
All three derived artifacts remain under `/tmp` with mode 0600. The full suite
again passed 843/843.

Direct claim-faithfulness progress remains **0 new samples / 0 new effect
estimates**. The benchmark interval-margin observation is local, exploratory,
and clock-unresolved; no generated explanation claims were tested. No internet
publication occurred.

### Frozen-cohort frame2time cross-check — 18:25 JST

To inspect the open clock-alignment issue without another model run, added
`scripts/audit_nextgqa_pilot_frame2time_alignment.py`. It reconstructs the
corrected six-video cohort, verifies its frozen preflight/cohort hash and
current runner hash, then emits only aggregate metadata counts under `/tmp`.
All six selected videos had one unambiguous video-map key and a 32-point
`frame2time` vector. For the seven gsub spans attached to the six selected
questions, 7/7 overlapped at least one `frame2time` point and 7/7 overlapped at
least one of the pilot's 16 sampled PTS. Across 96 nearest timestamp pairs,
the frame2time-to-sampled-PTS distance had median **0.181 s**, p95 **0.469 s**,
and maximum **0.700 s**. Input and cohort hashes are bound in the summary;
no identifiers, QA, span values, paths, or media are emitted. Local artifact
`/tmp/nextgqa_pilot_frame2time_alignment_20261015_1820.json` is mode 0600,
SHA-256 `e7d9b32af824586fc72b79dd2a24f9714058570abebf8ae446c00c68894afbc8`;
the audit script SHA-256 is
`b7430a63441310402da771824c88c83064c7c9afcae136786abd521ea39274c5`.

This confirms coarse overlap for this fixed cohort, not timestamp equality,
pixel identity between the two grids, or a shared annotation-player/source
edit. The 0.181 s median and 0.700 s maximum nearest-grid distances leave
substantial temporal quantization; overlap of both grids can occur even when
they select different frames. The interval-margin pilot therefore remains
exploratory benchmark dependence, not claim-specific faithfulness or a
confirmed evidence localization effect. Human claim review remains incomplete;
direct faithfulness progress is still **0 new samples / 0 new effect estimates**.

#### Full-frame PTS comparison for the same six videos — 18:35 JST

The initial 16-sample comparison above had a large nearest-grid distance
because it compared a 32-point metadata vector to only 16 pixels. Re-ran the
same frozen cohort against every decoded frame PTS (median 571 frames/video).
Across 192 `frame2time` values, nearest full-PTS error was median **8.2 ms**,
p95 **16.2 ms**, maximum **188.5 ms**. Four videos had all 32 points within
16.7 ms. In the remaining two, 31/32 points had median absolute residual
about 6 ms; the sole ~188.5 ms residual was the final `frame2time` point in
each. All seven gsub spans still overlapped both a supplied frame2time point
and one of the pilot's sampled PTS. This supports close dataset-metadata to
decoder-clock correspondence through most of each vector, while exposing an
end-point exception; it still does not independently verify the annotation
player/source edit. Counts-only artifact
`/tmp/nextgqa_pilot_frame2time_alignment_20261015_1835.json` is mode 0600,
SHA-256 `54220767890bc95cc45331437e5866c17b11b03ba6334741ad97a7029f3c58d1`;
the updated audit script SHA-256 is
`3a4e1005216cc07c5a3fd66c5cb9eb55527c31867adce36202ac7d734357689a`.

### Fresh-cohort 50 ms timing and mask-fill sensitivity — 18:47 JST

The separate full 85-video PTS census found a maximum 16-frame uniform-grid
deviation of 46.7 ms. As an explicitly exploratory technical-sensitivity
condition, the runner was changed from 1 ms to a fixed 50 ms tolerance and
limited to the first ten eligible videos in deterministic order. A local-only
private media-SHA exclusion file listed all six videos from the previous pilot;
an independent cohort reconstruction reproduced the new cohort hash and found
zero overlap with those six. This 50 ms rule was chosen after inspecting the
timing census and is not a confirmatory sample rule.

The outcome-blind preflight scanned 10/85 candidate videos: all ten passed the
50 ms timing and interval/control checks; 75 candidates were not scanned. No
prior pilot video appeared in the scanned prefix. The preflight fixed ten
videos, 16 frames, five conditions, package/model-config hashes, and the
exclusion-file hash. On the RTX 5070 Ti, Qwen2.5-VL-7B-Instruct (4-bit, BF16,
CUDA 12.8) completed **50/50 forwards** (10 full condition sets) without
failure. Peak allocated VRAM was **9.12 GiB**; available host RAM before model
load was 11.24 GiB; total elapsed time was 119.98 s. Full-condition
restricted-choice accuracy was 9/10. Result and preflight are local-only with
mode 0600. Result SHA-256 is
`c5a00dd2bcace8ffa4e28bd7d955642f81506a13cd73538390463a1e42a260ee`,
preflight SHA-256 is
`19647eb92f57c02c41235ed4f96af16683eb422d27cc05f44596f7fde355ad84`, and
runner SHA-256 is `0df9bd45e6209fead3eb110c25cef27f40d6a80f81fedd837ab402cb00f7fc7c`.
The private exclusion file's SHA-256 is
`dac8fe8191940f3def6a4458c20e40919eb7e0b4751c7229ac17288f73a45d98`.

The fixed-cohort black-fill contrast (equal-count control margin minus
gold-interval-mask margin) had mean **+0.700 nat**, sample SD **1.120**, and
range **−0.375 to +3.500**. Under the same videos and mask locations,
radius-12 Gaussian blur produced mean **−0.0625 nat**, SD **0.737**, and range
**−1.375 to +0.750**. The black gold-mask margin minus the blur gold-mask
margin averaged **−0.475 nat** (SD **0.851**, range **−2.875 to approximately
0**). No p-value, confidence interval, multiplicity correction, or population
inference was computed. The best foil can change by condition.

Observation: the direction and size of the interval-versus-control contrast
changed substantially between black and blur fill on this selected cohort.
The black-fill response is therefore not robust to this fill choice here. This
does not identify whether OOD masking, genuine evidence dependence, foil
switching, or another interaction caused the change. The 50 ms cohort is a
post-census technical sensitivity analysis; both it and the initial six-video
cohort are deterministic feasibility subsets. This remains benchmark
answer-margin dependence, not claim-specific explanation faithfulness. Direct
faithfulness progress remains **0 new samples / 0 new effect estimates**. No
network publication occurred.

#### Fresh 50 ms cohort repeat with the fixed five-condition grid — 18:52 JST

To check whether the fill-sensitivity pattern above repeats without reusing its
media, reconstructed its ten-video cohort from the private six-video exclusion
manifest and verified the exact cohort hash before creating a new private
exclusion list. Excluded media count is now 16 (six from the earlier 1 ms pilot
plus ten from the first 50 ms cohort); the exclusion JSON is mode 0600 and has
SHA-256 `5f66a9fdb1ae4ee74774484a103f7435a779e8bbe7825dbb0957a97d1b521f2c`.
The third cohort selected ten fresh videos after scanning 22/85 candidates;
63 remained unscanned. Every selected video met the same fixed 50 ms PTS rule
and had an equal-count disjoint control. This is an exploratory repeat chosen
after the earlier outcomes, not a confirmatory test or probability sample.

The outcome-blind preflight passed for ten videos × 16 frames × five
conditions, fixed cohort SHA-256
`87e26d501dbe5406cd1a91bf9bc4b0eab0edfc9da0ec4d6b75c4fbfe8bdb988c`, and
model/configuration bindings. Local GPU state before scoring showed 14.66 GiB
free VRAM and 13.45 GiB host-available RAM. Qwen2.5-VL-7B-Instruct (4-bit,
BF16, RTX 5070 Ti) completed **50/50 model forwards**, **10/10 full condition
sets**, zero failures, in 132.18 s; peak allocated VRAM was **8.92 GiB**.
Restricted-choice accuracy was 7/10. Result SHA-256 is
`8cafb52c23c5fb560ca26224003c519614a02dc91446011966622720eeb9e037`,
preflight SHA-256 is
`efef602dbce56fc4fe3d1d2d75f833c6dc624b2192df4288fbd29f7f54cef634`, and
the executed runner SHA-256 is
`d476f0626a4b7d8e1c5b368d36a81a17a725a9264cad501cbdb3e34005c2b747`.
The exact executed runner is archived privately under `/tmp` with mode 0600;
the current source only corrects its descriptive docstring after that run.

Independent record-level recomputation gives the black control-minus-gold
interval margin **−0.400 nat**, sample SD **1.187**, range **−3.000 to +1.250**
(4/10 positive); the matched blur contrast is **−0.400 nat**, SD **1.493**,
range **−3.375 to +2.125** (5/10 positive). Their paired difference averaged
approximately zero (**−0.00000008 nat**, SD **0.618**). The first fresh 50 ms
cohort instead had black +0.700 nat and blur −0.0625 nat. The sign reversal
between fresh cohorts means the earlier apparent fill sensitivity did not
repeat; neither cohort supports a stable direction for this small deterministic
sample. No p-value, CI, multiplicity correction, or population inference is
claimed. Since each condition's margin uses its own best foil, the score can
change through foil switching as well as gold-choice probability changes.

#### Third-cohort clock audit — 18:55 JST

Updated the local counts-only audit to bind the same private exclusion file and
preflight cohort hash instead of silently reconstructing a different cohort.
Across 320 `frame2time` points and every decoded PTS in ten videos (median
1,434 frames/video), nearest-clock error was median **8.3 ms**, p95 **16.2 ms**,
maximum **20.8 ms**. Fourteen of 15 gsub spans overlapped at least one
`frame2time` point; 12/15 overlapped at least one of the pilot's 16 sampled
PTS. This supports close metadata-to-decoder clock correspondence for the
fixed sample, but the sparse model sampling misses 3 annotated spans and this
does not establish annotation-player/source-edit calibration or claim
semantics. Private audit output SHA-256 is
`23bbeec7647fc5ac48229963d268b104cc958f6afe1a1ae39181b89bf5ae919f`.

The source's stale description of a 1 ms / three-condition protocol was
corrected after the run; the exact executed source is retained locally for
provenance. A separate stale limitation sentence saying “first six” was also
corrected to “first ten”; this is wording only and did not change selection or
scoring behavior. Direct claim-specific faithfulness progress remains **0 new
samples / 0 new effect estimates**. No network publication occurred.

### Thirty-two-frame real-video extension — 19:10 JST

The 10-video timing audit showed 14/15 annotated spans intersecting a
32-frame endpoint-inclusive grid versus 12/15 at 16 frames in the initial
sample (and full coverage at 64 frames). To see whether 32 frames can be run
within local GPU memory, extended the real-video smoke and the interval-margin
runner with explicit 16/32 frame configuration. A one-video 32-frame Qwen
smoke passed with finite logits, 4,256 input tokens, 4.19 s forward time, and
10.93 GiB peak allocated VRAM. Smoke artifact SHA-256 is
`6197134fbd871a5107c3dff685ab0128cf885a4fbaec056280b441938705ad1a`.

The 32-frame outcome-blind preflight selected the same ten media files as the
16-frame third-cohort run (22/85 scanned, 63 unscanned); **9/10 exact
video-question pairs matched**, while one question changed because the first
eligible question depends on the frame grid. This was determined before the
32-frame scores were read. The preflight froze 32 frames, five conditions,
50 ms timing tolerance, and cohort hash
`a6533afb092739d9020cf6aef5d830e6b9c661f638b9350d699dedf2ed99d49b`.
Preflight SHA-256 is
`cfa8e8822c076962a5fcf304f247bbfddc0c67102487c128148f9f63c1d12e60`.

On the RTX 5070 Ti, Qwen2.5-VL-7B-Instruct (4-bit, BF16, eager attention)
completed **50/50 forwards**, **10/10 full condition sets**, zero failures,
in **245.76 s**. Peak allocated VRAM reached **14.31 GiB** of 15.89 GiB
reported total; this is close to capacity and does not justify scaling the
same eager-attention batch beyond 32 frames. Full-condition restricted-choice
accuracy was 8/10. Result SHA-256 is
`c2a0c052aaaa9ae69b4a4eea8d7746c5127223a3c0f4550e2d82a390ab954436`, and the
exact run source (SHA-256
`7c9392be10590c0d9c629485783694cc14c43381e77b0d143080e3973b5355f8`) is
archived locally under `/tmp` with mode 0600.

Independent record recomputation gives the black control-minus-interval
margin **−0.313 nat**, SD **1.920**, range **−4.000 to +1.625** (6/10
positive); the blur contrast is **−0.350 nat**, SD **1.919**, range **−3.375
to +3.625** (3/10 positive). Their per-video paired difference averaged
**+0.038 nat**, SD **1.075**, range **−2.000 to +1.875**. The 16-frame run
on the same ten media had black −0.400 and blur −0.400, but one of ten
questions differs; these means are descriptive and not a complete paired
frame-count comparison. No test, interval, multiplicity correction, or
population inference was performed. Effects remain benchmark QA-margin
responses and do not measure explanation faithfulness.

The bound 32-frame clock audit matched all ten video keys and frame2time
vectors. Across 320 frame2time points and all decoded PTS, nearest-clock error
was median **8.3 ms**, p95 **16.2 ms**, maximum **20.8 ms**. For the ten
questions selected by the 32-frame rule, the 16-, 32-, and 64-frame grids
intersected 11/15, 14/15, and 15/15 gsub spans respectively; all ten videos
met the 50 ms uniform-grid criterion at each count. This is geometric
coverage only; the 32-frame Qwen input intersected 14/15 spans, and no model
score used 64 frames. Audit artifact SHA-256 is
`b62aadc06dfed4aa39944d83561104bb6b7675352eee28fe5f2460a358d5eeef`.

The current runner accepts `--frames 16|32`, records the count in its
preflight/result binding, and no longer assumes the private exclusion file
contains exactly six earlier media. The real-video smoke accepts explicit
frame count and timing tolerance and creates mode-0600 outputs. Python
compilation, `git diff --check`, and all **846/846** repository unit tests pass.
The local result files remain private; nothing was published online. Direct
claim-specific faithfulness progress remains **0 new samples / 0 new effect
estimates**.

#### Second fresh 32-frame cohort — 19:20 JST

To see if the first 32-frame cohort's negative mean repeated, froze another
outcome-blind cohort after excluding the previous 26 media hashes (the original
six-video run and both ten-video 50 ms cohorts). The private exclusion file
contains 26 unique SHA-256 values; its digest is
`d894ace9bacce60fdbaf977aeea6e0becf1fd3edb7da1cc65798f311185406ec`.
The preflight selected ten fresh videos after scanning 33/85 candidate videos;
52 were unscanned. Twenty-three excluded media hashes were encountered in
that scanned prefix before the ten eligible selections completed. All ten
passed the 32-frame, 50 ms, equal-count control assignment and Qwen processor
checks. Cohort SHA-256 is
`c8eff325d2b1fbd551da8a9e87dc4b60a8f1ecca32afb4b7227497363d92076d`;
preflight SHA-256 is
`a71aa7534e34fd6b67aa57d681dd6922cd3f407d7db13acd4889663fa7d40f67`.

The RTX 5070 Ti completed **50/50 forwards**, ten complete five-condition
sets, with zero failures in **224.74 s**. Peak allocated VRAM was **14.64
GiB**, leaving little headroom; do not increase this eager-attention frame
count or parallelize GPU jobs. Full-condition restricted-choice accuracy was
6/10. Result SHA-256 is
`b710ecbd9ca703d710f813313f15fe7e1405b3a2e417d8e7206d3b392f13813d`; result
mode is 0600 and it remains local.

Independent record-level recomputation gives black control-minus-interval
margin **+0.275 nat**, SD **1.684**, range **−2.375 to +4.250** (5/10
positive); blur contrast **+0.088 nat**, SD **1.382**, range **−1.875 to
+3.500** (4/10 positive). Black-minus-blur contrast averaged **+0.188 nat**,
SD **0.457**, range **−0.500 to +1.125**. This cohort's black estimate has
the opposite sign from the previous 32-frame cohort (−0.313 nat); both are
small, deterministic, exploratory samples with no p-value/CI or population
claim. The sign instability is evidence against treating this particular
benchmark-margin response as a stable effect. It is not a negative or positive
claim-faithfulness result.

The bound clock audit found 11/11 annotated spans intersected the actual
32-frame sample grid (and also the hypothetical 16- and 64-frame uniform
grids) for these selected questions. Across 320 frame2time points, median
nearest full-PTS error was **9.2 ms**, p95 **16.2 ms**, maximum **151.6 ms**;
the maximum was one final vector point, with the other points close to the
decoder clock. The sampled-grid audit cannot calibrate the annotation-player
source edit; one span in the prior cohort was missed by 32 frames, so 11/11 is
specific to this newly selected set. Audit artifact SHA-256 is
`03b40f107b114f5d68839f0566eecb6379bca037198277a2a370bf3868ec758f`.

#### Research decision at this checkpoint

Two exploratory 50 ms media cohorts contain 20 records scored at 16 frames.
Two 32-frame runs add 20 score sets: one revisits the second cohort’s same ten
media (9/10 questions match the 16-frame assignment), and one adds ten fresh
media. Descriptive black-fill contrasts vary across these four run/cohort
combinations (+0.700, −0.400, −0.313, +0.275 nat); the first value is the
first 16-frame 50 ms cohort. Blur-fill contrasts also vary (−0.063, −0.400,
−0.350, +0.088 nat). These are not independent identical replications
because timing tolerance, frame count, and eligibility differ; do not pool
them into a confirmatory estimate. No new GPU condition is
justified until the claim-level target and review plan are established. Direct
claim-specific faithfulness progress remains **0 new samples / 0 new effect
estimates**. All data, source archives, and summaries stayed local; nothing was
published online.

### Full local test-video timing audit — 19:22 JST

Expanded `scripts/audit_nextgqa_pts_alignment_local.py` to compare 4/8/16/32/64
endpoint-inclusive frame-index grids under both 1 ms and 50 ms uniformity
limits, report annotated-span coverage separately for each, and write its
counts-only summary exclusively with mode 0600. Ran it over all **85/85**
available distinct annotated local test videos (requested cap 100), with **0
decode failures** and **643/643** valid gsub spans intersecting at least one
fully decoded frame PTS. This is the full available local candidate set for
that script's deterministic, manifest-excluded cohort, not all NExT-GQA videos
globally.

Frame2time metadata was close to the decoder clock: 2,712 timestamp points had
nearest-PTS error median **8.5 ms**, p95 **16.7 ms**, maximum **33.3 ms**.
Uniform-index sampling gave span-hit rates 274/643 (**42.6%**) at 4 frames,
450/643 (**70.0%**) at 8, 584/643 (**90.8%**) at 16, 634/643 (**98.6%**) at
32, and 643/643 (**100%**) at 64. Every video passed the post-census 50 ms
uniform-grid criterion at all five sample counts; only 6/85 passed 1 ms at
16 frames and 1/85 at 32 and 64. The 50 ms threshold remains an exploratory
engineering tolerance selected after examining timing behavior, not a
confirmatory rule. Span/frame overlap is geometric only and says nothing about
whether pixels support the QA answer or a generated explanation.

All-85 summary SHA-256 is
`6b60f8c0787d4a509b25b54690d4d7a081114c1c29287d8d99d28f273dac49ab`; it is
mode 0600 under `/tmp`. The output binds dataset input hashes and a private
video-media fingerprint; no IDs, paths, QA text, intervals, or video data are
written. Python compilation, `git diff --check`, and the full **846/846** unit
suite passed after these changes.

This establishes that 32-frame sampling captures most supplied temporal
intervals geometrically in this local test set, while the 32-frame model run
uses nearly all available VRAM and its small-cohort QA-margin sign changed
across fresh selections. It supports neither a claim-specific faithfulness
result nor a benefit from masking. The next useful experiment is not another
unreviewed interval-margin repetition: define and review model-generated
answer claims against the supplied spans on a frozen local cohort, then freeze
an explicit claim-score/control estimand before any new GPU scoring. Per the
user's 2026-10-15 directive, local-only protected-data analysis may proceed
without a rights-verification gate; the rights status remains unverified and
no data or result has been shared online. Direct faithfulness progress stays
**0 new claim samples / 0 effect estimates**.

#### Independent reproduction of full-cohort counts — 19:23 JST

A separately written one-off CPU pass independently rebuilt the
manifest-excluded candidate join from the local CSV/JSON files, decoded all 85
videos with PyAV, and recomputed interval coverage and frame-grid hits without
calling the audit's aggregation function. It reproduced 85 decoded videos,
zero failures, 643 intervals, 643 full-PTS overlaps, 2,712 frame2time points,
and exact 4/8/16/32/64-frame coverage counts of 274/450/584/634/643. It also
reproduced 85/85 videos passing 50 ms at each count. Assertions compared these
counts to the private report and all passed. This closes the primary numerical
reproduction check for this geometric audit; it still does not verify
annotation meaning or explanation faithfulness.

#### Explicit frame2time end-of-video accounting — 19:25 JST

The first full audit's 2,712-point error distribution excluded finite
`frame2time` values after each video's final decoded presentation timestamp.
Added explicit accounting and reran all 85 videos: 2,720/2,720 points were
finite, none preceded the first PTS, and **8/2,720 (0.29%)** were after the
last PTS. The largest endpoint excess was **421.1 ms**. The remaining 2,712
in-range points retain median nearest-PTS error 8.5 ms, p95 16.7 ms, max
33.3 ms. Thus the 151.6 ms endpoint residual in the fresh 32-frame cohort
falls into this after-last-PTS category; it should not be described as a
typical within-video clock error. This still does not prove how the annotator
player/source edit was timed, and those eight endpoint values remain a data
handling limitation rather than evidence of annotation error.

The endpoint-explicit private audit SHA-256 is
`aff43294b4eb66277d948ceefca4e0b0175d9988d6c9457d2e4ba824cf40e009`, mode
0600. The summary continues to report span coverage separately from timestamp
range validity. Python compilation, diff checks, and **846/846** tests pass.

An independent direct pass over the same 85 local media and `frame2time`
vectors also reproduced 2,720 finite points, zero negative/out-of-range-start
values, eight values beyond the last decoded PTS, and the maximum **421.1 ms**
excess. This reconciles the selected-cohort endpoint exception with the
full-cohort in-range distribution.
