# AGENTS.md — mandatory first read

Single source of truth for all coding agents in this repository. **Read it before reading or modifying source,
experiments, or result artifacts.** `claude.md` is only a compatibility pointer back to this file.

## 1. Continuation and autonomy

- **Stop-only rule (2026-10-18).** During a research/PDCA task, keep doing useful, authorized work until the user
  explicitly says `stop`. Do not end early because a branch is blocked, a scientific stop was reached, progress is slow,
  a checkpoint is complete or the turn feels complete. When a branch cannot proceed, switch to another independent,
  justified task. External blockers block only the dependent operation; document them and continue elsewhere. The only
  voluntary stop is token/context exhaustion (then report the work as incomplete, not complete). `stop` ends work at once.
- **Autonomy (2026-10-22).** Implement, run bounded GPU work, audit, commit and push scoped changes without asking
  permission at each step. Permission-type gates are treated as cleared and logged as
  `ASSUMED-CLEARED (pending user confirmation)`. Still in force: no fabricated labels or reviews, no force-push or hard
  reset, no publication of raw/private data, no unrun result reported as run.
- **Status requests are steering, not a stop.** Answer briefly and continue the loop. "確認して" is a workflow trigger
  (section 6): once verified, continue to the next useful action in the latest log. Pause only for a scientific stop, an
  external access blocker or a materially missing decision.
- **Durations.** If the user gives one, record the start time (the request's timestamp, else the current clock, stated as a
  conservative start) and the deadline in the user's timezone; re-read the clock at each checkpoint. A duration is a hard
  minimum work window: do not return early because results are sufficient or a branch stopped; pick another independent
  research, protocol, measurement or reproducibility task. Never imply unmeasured time was worked. If the turn must end for
  a platform reason, report the window as incomplete. Do not create activity or runs without a discriminating scientific
  purpose merely to fill time.

## 2. Repository and environment

`res` supports a Kumamoto University graduation thesis on video-VLM (Qwen2.5-VL) explanation faithfulness: research
scripts, sample manifests and dated experiment logs; not a packaged library. Read
[docs/research_overview.md](docs/research_overview.md) for plan, scope and model choices, and the latest relevant entry in
`experiments/` for the current state and next action (reassess it against newer results before acting).

- Dependencies: `scripts/requirements-spike.txt` (non-torch); install torch/torchvision separately with the CUDA-matched
  index in [docs/remote_run.md](docs/remote_run.md); check the experiment log for the GPU machine before assuming versions.
- Qwen2.5-VL: `torch_dtype=torch.bfloat16`, and for 4-bit `bnb_4bit_compute_dtype=torch.bfloat16` (float16 gives NaN/Inf on
  7B; see [experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md](experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md)).
- Scripts are standalone argparse CLIs: read the docstring/usage first. This Ubuntu/WSL machine may execute research: at the
  start of a relevant task check GPU, CUDA/PyTorch, storage and dataset availability rather than assuming they are remote or
  absent. A visible GPU does not imply CUDA dependencies in the active Python; identify the intended environment before
  installing anything.
- Hardware (user-reported): RTX 5070 Ti 16 GB, 2 x 16 GB RAM; WSL reports about 15 GiB RAM. Capacity is not availability:
  check free VRAM and OS/WSL memory for each run.
- **WSL / GPU access (2026-10-09).** Keep the existing WSL checkout and `.venv`; do ordinary edits, analysis and Git in the
  normal sandbox. The sandbox may hide `/dev/dxg`, so an `nvidia-smi` error there is not proof that the driver or GPU is
  unavailable. For a justified GPU command, request individual elevated execution for that exact command and state its
  purpose. Never switch permanently to Full access. If elevation is unavailable, report the blocker and continue CPU-only
  work; do not build a replacement environment. Do not migrate to native Windows, reinstall CUDA or rebuild `.venv` without
  explicit instruction, and do not write to a Windows checkout via the read-only `/mnt/c` mount.

## 3. Multi-agent work and audit

- Use sub-agents extensively for independent tasks and successive waves, within the platform's concurrency limit. Preferred
  model (2026-10-14): gpt-6-luna, reasoning_effort medium, with explicit task context when the platform needs
  `fork_turns none` for overrides; changing the primary session model needs a client-side selection and must not be claimed
  via delegation.
- At the start of each substantive PDCA cycle the primary assigns concrete deliverables and boundaries: at least one
  design/implementation role and, for any material experiment or code change, a separate **independent auditor** who does
  not rely on the executor's summary. Reports separate verified facts, proposals and open questions.
- Reassign agents promptly when tasks finish or priorities change; avoid duplicate experiments. Give each file one owner:
  no concurrent edits, competing Git operations or simultaneous GPU jobs. The primary controls integration, publication and
  the GPU workload.
- The primary is the overall auditor and stays accountable for provenance: verify important claims against source,
  manifests, original results and primary sources, investigate conflicts, and run proportionate checks before accepting
  delegated findings.
- **SSS bar** for every material result: verified provenance and hashes; independently reproduced key counts and estimates;
  meaningful code/protocol checks passed; alternative explanations and failure cases addressed; claims within tested scope
  with uncertainty and exclusions; audit issues resolved; next action justified by evidence and upstream gates. Do not label
  work SSS or complete while a criterion is unresolved: state the gap, repair it or audit further.

## 4. Research standards

**Claims.** Never report an experiment as run unless its output or a tracked result verifies it. Never present a
non-significant difference as significant; state effect size, interval, test and multiplicity correction. Paraphrase and cite
prior work, distinct from this project's measurements. For faithfulness metrics (Deletion/Insertion, Sensitivity-n, ROAR,
Sufficiency/Comprehensiveness, ...) state what they measure and their limits (masking OOD, metric disagreement). Scope claims
to the tested metric, dataset, model, intervention and conditions. Label hypotheses, exploratory and post-selection results
as such. A scientific stop of a prespecified branch is mandatory for that branch only.

**Reproducibility.** Record seeds, library versions, hardware, model/precision, data split, intervention and material
hyperparameters in a manifest or log. Keep tunable parameters and splits configurable. Keep runnable code for generated
figures/tables with generation date and data version. Every experiment or result-affecting code change updates
`experiments/YYYY-MM-DD_topic.md` with purpose, setup, verified results (requested/effective counts, failures),
interpretation, limitations and next action; separate observed numbers from inference.

**Execution and data.** Smoke-test the code path on a small sample before any long GPU job and state expected workload first.
Never use destructive Git operations; never silently overwrite outputs (use a new path or stop). Do not share raw or
identifiable video/images; check dataset/model licenses before redistribution. Directive (2026-10-15): user-authorized
local-only protected-data/model operations proceed without a rights gate; keep material and derived outputs local, state that
rights remain unverified, and note this does not authorize redistribution.

**Publication.** Publish inspectable result summaries to this GitHub repository as the default handoff instead of asking the
user to paste output. Use a publisher that writes to a unique `results/` path and stages only that artifact with its manifest
and analysis (never mix unrelated tracked changes; never `git add -f` a directory). Afterwards verify the pushed commit by
fetching and analyze it from the repository; a printed "wrote" is not proof. On failure preserve the local artifact and give a
safe recovery command; ask for a paste only if Git publication is blocked or unsafe. Publish the minimum derived information:
never commit raw videos, original annotation archives/pickles, model weights, credentials, personal identifiers, absolute
local paths or unnecessary raw question/answer text; check size, license, privacy and content first, and keep a summary local
if it could expose restricted data.

## 5. Label-free confirmatory design (2026-10-22)

- Confirmatory experiments are label-free: do not gate GPU runs, analysis or reporting on human review. Use endpoints from
  existing gold annotations (e.g. NExT-GQA QA time annotations) and the model's own outputs; human review is optional
  validation of construct validity on small samples. Such an endpoint measures dependence on the annotated interval, not
  explanation faithfulness, and reports must say so.
- Sample size is the main limitation. Before selecting a cohort prespecify N, the practical threshold and one primary analysis
  from a power calculation (video = unit; no interim looks; no batch-by-batch pooling), and report an equivalence test (TOST)
  so a null reads "effect below the threshold".
- Justify the practical threshold (delta_min) from data before selection, never by convention: (a) decision relevance (share of
  videos whose baseline gold-vs-foil margin lies within +/-delta of the decision boundary), (b) a standardized smallest effect of
  interest (d_z = 0.2 x the SD of the per-video contrast), (c) the measurement noise floor. For the answer-margin endpoint in the
  annotation-key-bias design use delta_min = 0.3 nat (345 videos: 5.5% of decisions lie within 0.3 nat; 0.2 x SD 1.69 = 0.34 nat;
  0.025 nat moved 0.0% of decisions and was too small). Re-derive it for a new endpoint.
- Use the fp32 `lm_head` readout wrapper `scripts/run_nextgqa_annotation_key_bias_fp32_readout.py` (bf16 logits quantize
  margins at 0.125 nat); never pool bf16 and fp32 readouts.
- Pair every intervention test with a prespecified **positive control** that must be able to move the endpoint; if it fails,
  nulls from that intervention are uninformative and must be reported so.

## 6. Workflow: result -> analysis -> implementation -> command

For every "確認して" and every newly supplied run/result, complete these steps in one response unless a real blocker needs user
input:

1. **Verify provenance first**: Git status/commit, result JSON, manifest, command/config, requested/effective counts,
   failures, completeness. Distinguish live GitHub from a cached `origin/main`; never imply a fetch or run that did not occur.
2. **Analyze**: intervention and score, direction/size, interval, test, multiplicity, exclusions, selection, limitations;
   observations separate from interpretation. Do not ask the user to rerun a published analysis just to print values already in
   the repository; if rerunning helps, default to a short human-readable summary with verbose output behind a flag; never paste
   large machine-readable dumps. Exact repeated pastes get a minimal reply.
3. **Choose the next scientifically useful action**: confirm, repair measurement, revise the hypothesis or stop. Never chase
   significance by changing metrics or subgroups after seeing outcomes (that is exploratory and needs independent
   confirmation). Do not start a costly GPU run merely to look active; do run the next prespecified experiment once
   prerequisites and smoke checks pass.
4. **Implement** instead of only describing: code, proportionate tests/static checks, dated log update. Do not claim an unrun
   GPU test passed. If stopping is sound, record why and avoid meaningless code.
5. **Commit/push scoped work** (authorized): check branch/worktree, stage only task files; for non-fast-forward use a safe
   fast-forward/rebase on a clean worktree, never force-push; if blocked, preserve work and give the exact blocker and recovery.
6. **Publish remote outputs by default** through a scoped, safe wrapper, then inspect the committed artifact.
7. **Give exact Ubuntu commands** for the next useful work (not a sync-only command; `git pull --ff-only` only as a prerequisite
   chained with `&&` to the real task). Costly inference: smoke first, then full; unique output name; guard against
   overwriting; prefer runners that write a manifest and publish safely. If no Ubuntu run is needed, say so and give the next
   CPU verification command.
8. **Explain in plain Japanese after the commands**: why the run follows from the result, which cohort/candidates and how
   selected, what it can and cannot establish, coverage, failures, effect sizes, intervals, adjusted results; state that
   proposed commands have not run; use a compact table for multiple conditions.

If the sound choice is to stop or revise direction, say so, record it and give a proportionate next action instead of inventing
another expensive run.

## 7. PDCA loop

Do not end a cycle by handing commands back to the user: execute authorized bounded work, analyze, repair, publish scoped safe
artifacts and continue without asking "continue?". An unresolved gate blocks only the dependent branch; continue independent
measurement repair, synthetic validation or protocol work. GitHub publication is the handoff, and it does not turn synthetic
checks into real-data evidence.

1. **Plan**: inspect Git/result provenance, latest log, dataset and Python environment, GPU/CUDA state and the next prespecified
   gate; pick the smallest experiment that discriminates live hypotheses.
2. **Implement**: analysis/intervention code and tests with explicit configs, splits, seeds, thresholds. Do not reopen a closed
   search (head/layer/alpha tuning included) without a rationale fixed before outcomes.
3. **Check**: CPU tests and a smoke sample first; before a full run state sample count, runtime/VRAM and cost. Proceed without
   another approval for bounded, reversible, in-scope research; pause only if cost, data exposure or scope materially exceeds
   the grant.
4. **Do**: run when gates pass. If the runtime or data is unavailable, never silently substitute a model, cohort or metric;
   repair the environment safely or record the blocker and continue elsewhere.
5. **Study**: verify output and manifest; counts, failures, effect and uncertainty, multiplicity, exclusions, limitations.
6. **Act**: update the log and next action, implement the next step, and publish privacy-safe artifacts with a scoped
   commit/push, verified by fetch.

Repetition is not progress. If a prespecified gate fails, stop that branch or revise the protocol before collecting more
outcomes. Never change a primary metric, subgroup or threshold after seeing results and call the same data confirmatory.
Negative and inconclusive results are valid outputs that guide the next design.

## 8. Current research status (update when results change)

- The preregistered head-faithfulness confirmation missed its +0.25 nat threshold (79/79; -0.0756 nat, 95% t CI
  [-0.1578, +0.0065]); do not resume head/layer/alpha search. The CLEVRER matched-pair branch is closed (158/12,974 pairs
  shared operator traces; 12/158 had token edit distance 2-3; not evidence of semantic validity).
- Earlier grounding/claim-fit counts are geometric/proxy audits only, not efficacy results.
- QA-annotation key bias (label-free, NExT-GQA val, fp32 readout; see `experiments/2026-10-22_annotation_key_bias_*` and `2026-10-23_*`, consolidated in
  `2026-10-23_annotation_key_bias_consolidated_summary.md`): last-row bias nulls (+ln2..+ln16) are uninformative because their suppression controls failed. With the
  multi-query scope (all post-video rows, all 28 layers) the suppression control passed in two fresh cohorts (N=149 p=0.001; N=196 p=0.031, fragile under rank tests);
  steering (+ln4, +ln16) was inconclusive after Holm in both and never improved the margin over baseline; contrasts are mostly sham-driven. Scoped SSS for these
  measurements only; none of this is explanation faithfulness (direct evaluation not done).
- Base-paper extension (Map the Flow, ICLR 2026): WP0 pathway-knockout replication on Qwen2.5-VL met all four prespecified criteria; WP1 (evidence-specific knockout, test N=112) supported its primary hypothesis
  (video->text, layers 6-20: evidence minus sham -0.68 nat, Holm p=4e-5, positive control passed; video->last equivalent to zero). Evidence-focusing Stage 2 (N=80) was not confirmed (+0.18 nat, random-keep also improves);
  Stage 2b/pooled confirmation and SGC are queued. See `experiments/2026-10-23_*` and `docs/thesis_base_paper_extension_plan.md`.
- Hardware note: RTX 5070 Ti, `.venv` and the pinned checkpoint as above; check runtime availability before each run.
