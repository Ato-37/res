# PDCA checkpoint: runtime, informal response, and direct-score gates (2026-10-21, local-only)

## Purpose

Revalidate the active state after the previous exposure-audit repair: check the
new user-supplied exploratory response without exposing names or free text,
verify current local GPU/runtime availability, and decide whether a nonduplicate
real-video GPU experiment can run without bypassing the frozen human/timing
gates. This checkpoint was initiated 2026-10-21 JST; no fixed-duration window
was specified.

## Provenance and checks

- Repository branch is `internal-ablation-wiring-20261018` at
  `67f1c6544b65a8efa76620f19888e4302a6e934f`, matching the live remote ref at
  the time of verification. The preceding scoped exposure-audit commit and
  generator pair-digest change are present on that ref. Numerous unrelated
  worktree edits and local data remain unstaged.
- Read the user-provided `exploratory_video_review_2026-10-19T09-49-47-676Z.json`
  using `scripts/inspect_nextgqa_quick_review_response.py`. Its schema is v5;
  3 respondent slots and 9 categorical item judgments validate against the
  stable IDs in the local v14 exploratory review page only in explicit legacy
  mode. It has no response fingerprint. The exact page version viewed is
  therefore unverified. It is not a formal review and does not cover the newer
  generated-claim v1–v4 pages; no per-item result is attributed to those GPU
  candidates.
- Independently reaggregated the response categories from the inspector's
  sanitized in-memory result. Per-item counts (three judgments each): item 1
  clarity unclear 2/3; answer fit no 1/3, yes 1/3, cannot tell 1/3; evidence
  insufficient 2/3, cannot tell 1/3. Item 2: clarity unclear 2/3; answer fit
  no 2/3; evidence insufficient/sufficient/cannot tell 1/3 each. Item 3:
  clarity clear 3/3, answer fit yes 3/3, evidence insufficient 2/3 and
  sufficient 1/3. These are descriptive responses from three self-labeled
  participants, not independent-authenticity evidence, a representative sample,
  claim truth labels, or a faithfulness estimate.
- `test_inspect_nextgqa_quick_review_response`: **16/16 passed**. The earlier
  focused exposure/generator/source-ledger/pool suites passed **46/46** and the
  exposure change passed an independent source review. No protected response
  fields were printed or changed.

## Runtime and GPU decision

- Normal sandbox check: WSL presents 15 GiB RAM with about 13.6 GiB available;
  `.venv` has PyTorch 2.11.0+cu128 but `torch.cuda.is_available()` is false
  because the normal sandbox cannot access NVML/device files.
- Individual elevated read-only check at this checkpoint: NVIDIA GeForce RTX
  5070 Ti, 16,303 MiB total / 14,332 MiB free, 1% GPU utilization, 12% memory
  utilization; CUDA 12.8 and BF16 are available in the existing `.venv`.
  `/proc/meminfo` showed 14,275,344 KiB available. Workspace filesystem had
  approximately 888 GiB free. No long-running experiment process was present.
- Current log and code audit found no justified, nonduplicate actual-video GPU
  endpoint before the human gates. Fixed QTA proxy cohorts, selector diagnostics,
  mask-specificity work, synthetic-event work, and head exploration are closed.
  Repeating them would not advance the direct estimand. The formal preload and
  transaction diagnostic modules explicitly have no model/CUDA/score or
  authorization path; a passing structural diagnostic still sets
  `model_load_permitted=false` and `authorized_forwards=0`.
- The next planned GPU transaction remains exactly three untreated baseline
  forwards, but it is not executable now. The valid path first needs the
  external-root-authenticated role/release roster and witnessed review-only
  handoff; signed Stage-1 claim reviews; all 15 captured and independently
  reviewed/calibrated clock/frame correspondences; frozen source, prompt,
  processor batch and code pins; and a production atomic pre-load gate. After
  the three baseline predictions freeze `p/q`, a distinct Stage-2 review is
  required before any later treatment/control forwards. No agent-generated or
  informal review can substitute.

## Faithfulness progress

- **Directly evaluated:** 0/3 claims; 0/15 clock/frame correspondences;
  0/3 prediction-link judgments; 0/3 direct scores. The model/intervention
  estimand has not been run on eligible reviewed inputs; there is no direct
  `DeltaF`, uncertainty interval, statistical test, or multiplicity result.
- **Auxiliary only:** the old v14 informal response provides 9 categorical
  impressions over three stable slots; legacy version binding is unverified.
  It does not count toward direct progress. The October 19 real-video GPU claim
  generation (3/3 records, 16 frames each) remains a separate review-workflow
  feasibility run, not a faithfulness result.
- **Status:** direct score path `no_go` pending external human/trust/timing
  inputs and production runner. Dataset/model rights remain unverified; all
  protected files and this detailed checkpoint stay local.

## Additional transaction-path verification

- A focused integration set covering the pre-baseline transaction coordinator,
  Stage-1 source pins, clock-row adapter, prescore mapping binding, and baseline
  p/q candidate validation completed **91/91 tests passed** in 42.1 seconds
  (`PYTHONPATH=scripts .venv/bin/python -m unittest
  test_qta_prebaseline_transaction_preflight
  test_qta_prescore_clock_row_adapter test_qta_prescore_mapping_binding
  test_qta_baseline_pq_candidate -v`). The tests use synthetic fixtures; this
  verifies fail-closed structural wiring, not real human authorization or
  direct faithfulness.
- An independent source audit confirmed there is still no real-data production
  runner that checks all actual review/timing gates before model loading and
  then scores exactly the frozen three rows. Existing clock, mapping, token,
  and transaction components validate separate structural bindings, but they
  deliberately do not authenticate human identity, semantic correctness,
  independent trust-root origin, or human frame correspondence. The GPU smoke
  uses synthetic inputs and is not a substitute.
- A first broad test invocation mistakenly used an unavailable `python`
  executable and then an incorrect unittest module name; the corrected run
  above is the authoritative pass. No source code or existing user artifacts
  were changed in this checkpoint.

## Human-review handoff verification

- Independent source audits confirmed the informal review page implements the
  three requested feedback axes: question clarity; whether the answer matches
  the video; and whether the explanation supplies enough video events,
  including sequence when needed. It keeps nickname-first entry and a separate
  nickname for each reviewer. The existing versioned artifact fingerprint
  prevents new responses from being silently joined to another wording
  version. These ratings remain exploratory impressions.
- The current informal browser E2E was run with elevated WSL-to-Windows Chrome
  access and **passed 1/1** in 1.99 seconds. It used three synthetic videos,
  drove three reviewer sessions, checked reset/attribution and a combined
  schema-v6 JSON export. The formal review flow E2E separately **passed 1/1**
  in 3.84 seconds with synthetic fixtures, checking two reviewer sessions,
  reset, combined envelope parsing and retry equality. Neither test opened
  protected user videos or response data; neither authenticates reviewers or
  verifies that a real person watched the videos.
- The associated CPU suites passed **61/61** for the quick artifact builder,
  response inspector, private importer and envelope, and **43 tests** for the
  formal renderer, prediction-link review/validation and formal workspace
  (42 passed, 1 browser E2E skipped in the normal sandbox because Windows
  interop is hidden; that exact E2E then passed under the individual elevated
  invocation above). This verifies current handoff mechanics only. Direct
  claim review, clock/frame review and scoring counts remain unchanged.
- The existing 2026-10-19 experiment log records a prior exact-v4 generated-
  claim page check (3/3 embedded media decoded and a synthetic three-reviewer
  combined export); that is historical evidence, not a run performed today.
  Today’s browser run checks the generic v14 quick-review builder and must not
  be cited as a new review of those v4 claims. The supplied older response
  remains unbound to v4 and cannot fill the formal claim-review gate.
- Improved the untracked Chrome E2E test so it waits for metadata, a nonempty
  current media source, and a decoded frame after each of the three item
  transitions. An independent test-design audit confirmed the asynchronous
  wait ordering and that only Boolean readiness is returned to the test
  driver. The enhanced synthetic test passed **1/1**; the same test against
  the exact private v4 generated-claim HTML passed **1/1** in 2.33 seconds.
  The v4 run used synthetic judgments and a temporary download directory; it
  verified video decoding/UI plumbing for all three slots, not claim truth or
  reviewer response. Existing primary byte-binding audit remains the evidence
  that those embedded clips match the private source package.
- Builder, inspector, importer and combined-envelope suites were rerun and
  passed **61/61**. No model/GPU inference or human review occurred. Direct
  counts remain 0/3, 0/15, 0/3 and 0/3.
- Scoped publication: committed only the E2E test and public-safe experiment
  summary as `1bdaaf96b5b8d951877ee6c6a2df2d7f1aa52ab8` and pushed the tracked
  branch. A live `git ls-remote` check returned that exact hash. The private
  checkpoint, response JSON, videos, and all unrelated worktree changes were
  excluded.

## Next action

Keep the user’s schema-v5 response as legacy exploratory feedback only. The
next action that advances official counts is the trusted human release and
completion of Stage 1 plus the 15-row capture/frame review and calibration.
The engineering candidate after those gates are authentic is an atomic
pre-load transaction that revalidates the exact source, 15-anchor crosswalk,
prompt/token bindings, signed phase-1 receipt, and same live frozen batches,
then permits only the three baseline forwards and freezes p/q. It must reject
today’s non-authorizing structural reports; synthetic test keys or a caller
boolean cannot enable model loading. Once those exact inputs arrive, build and
test that runner before launching the three-row GPU baseline. Until then, do
not rerun a closed proxy or claim-generation batch solely to occupy the
available GPU.
