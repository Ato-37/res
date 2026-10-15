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
attention arrays or scores appear in the output. This demonstrates the
restored synthetic execution path only; it contributes zero direct faithfulness
samples and zero effect estimates.

The complete standalone suite was rerun after the synthetic-smoke repair:
**808/808 passed in 8.092 seconds**; `py_compile` and `git diff
--check` also passed. Only fixture-driven expected failure messages and
non-failing upstream processor deprecation warnings appeared. No GPU model was
loaded and no GPU work ran during this CPU test-suite run; the separate GPU
smoke above is recorded independently.
