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
