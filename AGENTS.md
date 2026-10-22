# AGENTS.md — mandatory first read

## Explicit stop-only continuation rule (user instruction, 2026-10-18)

- During an ongoing research/PDCA task, continue useful, authorized work until the
  user explicitly says `stop`. Do not end early because one branch is blocked,
  a scientific stop is reached, progress is slow, a checkpoint is complete, or
  the current turn feels complete. Switch to another independent, justified task
  when a branch cannot proceed. The only reason to voluntarily stop this ongoing
  work is token/context exhaustion; if that happens, report the work as incomplete
  and do not claim completion. A user `stop` instruction immediately ends work.
- Treat actual external blockers as blocking only the dependent operation. Continue
  useful in-scope work elsewhere while documenting the blocker; do not use a branch
  gate as a reason to stop the whole session.

This file is the single source of truth for all coding agents working in this
repository. **Read `AGENTS.md` before reading or modifying source, experiments,
or result artifacts.** `claude.md` is only a compatibility pointer for Claude Code
and directs back here; this file remains the sole policy source.

## Repository and research context

This repository (`res`) supports a Kumamoto University graduation thesis on
video-VLM (Qwen2.5-VL) explanation faithfulness. It contains research scripts,
sample manifests, and dated experiment logs; it is not a packaged software library.
Read [docs/research_overview.md](docs/research_overview.md) for the research plan,
motivation, scope, and model choices. Consult the latest relevant entry under
`experiments/` for the current state and next action.

### Environment

- `scripts/requirements-spike.txt` lists non-torch dependencies. Install torch and
  torchvision separately using the CUDA-matched index described in
  [docs/remote_run.md](docs/remote_run.md); check the experiment log for the GPU
  machine before assuming versions.
- Qwen2.5-VL must use `torch_dtype=torch.bfloat16` and, for 4-bit quantization,
  `bnb_4bit_compute_dtype=torch.bfloat16`. float16 can cause NaN/Inf failures on
  the 7B model. See
  [experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md](experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md).
- Scripts are standalone argparse CLIs. Read the script docstring/usage before
  invoking. Treat this Ubuntu environment as a potentially executable research
  environment: inspect current GPU, CUDA/PyTorch, storage, and dataset availability
  at the start of a relevant task instead of assuming those resources are remote or
  absent. A visible GPU does not imply that the active Python environment has CUDA
  dependencies installed; identify the intended environment before installing anything.

### WSL workspace and per-command GPU access (user instruction, 2026-10-09)

- Keep the existing WSL/Ubuntu checkout and `.venv`. Run ordinary code edits,
  analysis, and Git operations in the normal Codex sandbox.
- The normal sandbox may hide `/dev/dxg`; an observed `nvidia-smi` error there
  does not establish that the WSL host or GPU driver is unavailable. If a GPU
  operation is justified, request individual elevated execution for that exact
  command and state its purpose. Use the existing repository `.venv`.
- Never switch permanently to Full access for GPU work. If individual elevated
  execution is unavailable or rejected, report that blocker and continue with
  useful CPU-only work; do not create a replacement environment.
- Do not migrate this project to native Windows, reinstall CUDA, or rebuild
  `.venv` unless the user explicitly instructs it. Do not write to a Windows
  checkout from the current read-only `/mnt/c` mount.
- Read-only verification on 2026-10-09 confirmed that normal sandbox lacked
  `/dev/dxg` and reported GPU access blocked, while an individually elevated
  check in the same WSL checkout saw the RTX 5070 Ti and CUDA-enabled `.venv`.
  This is a Codex sandbox visibility boundary, not evidence of a WSL driver
  failure.

## Multi-agent research and independent audit (user instruction, 2026-10-09)

- The user explicitly requests extensive sub-agent deployment for research work.
  Use the available concurrency slots for independent, useful tasks, and deploy
  further waves as needed; do not exceed the platform's actual concurrency limit.
- The primary agent is the overall auditor: integrate findings, verify important
  claims against source code, original results, manifests, and primary sources,
  and personally investigate any doubtful or conflicting sub-agent summaries.
- Designate an independent auditor among the sub-agents. That auditor reviews
  implementation/protocol/evidence independently rather than accepting another
  sub-agent's summary as proof. Reports must distinguish verified facts from
  proposals and unresolved questions.
- Reassign sub-agents when their tasks finish or priorities change. Initial role
  assignments are not permanent: use follow-up tasks and successive waves to
  keep available resources focused on useful work, without redundant experiments
  or speculative busywork. Retain an independent audit role for each material
  implementation or experiment and have the primary agent verify integration.
- Assign clear ownership for edits. Avoid concurrent mutation of the same files,
  competing Git operations, or simultaneous GPU jobs. The primary agent controls
  final integration, publication, and the GPU workload.
- User-reported hardware: RTX 5070 Ti, 16 GB VRAM, physical RAM 16 GB x 2 = 32 GB.
  This is host hardware capacity, not a guarantee of runtime-available memory.
  Check GPU access, free VRAM, and OS/container/WSL memory limits for each run;
  the current WSL runtime has reported approximately 15 GiB total RAM.
- User model preference (2026-10-14): spawn new sub-agents with gpt-6-luna and
  reasoning_effort medium, using explicit task context when the platform requires
  fork_turns none for model overrides. Changing the primary session model requires
  a client-side model selection; do not claim to have changed it via delegation.

### Per-cycle role rotation (user clarification, 2026-10-10)

- At the start of every substantive PDCA cycle, the primary assigns concrete
  deliverables and boundaries to sub-agents. Use at least one design/implementation
  support role and a separate independent auditor when a material experiment or
  code change is involved; do not let the auditor rely only on the executor's
  summary.
- When a task completes, the primary promptly reassigns that agent to the next
  useful independent task or audit. Initial role assignments are temporary; keep
  available concurrency focused on useful work without duplicate experiments.
- The primary remains accountable for provenance and integration: inspect source,
  manifests, outputs, and disagreements directly, and independently run proportionate
  checks before accepting delegated findings.
- A user status/progress request during active research is steering, not a request
  to stop. Answer it briefly and continue the active PDCA loop without asking whether
  to proceed. Pause only for the explicit workflow gates above, a scientific stop,
  an external access blocker, or a materially missing decision.

### Sustained work and primary-agent SSS audit (user clarification, 2026-10-10)

- When the user specifies a work duration, record the requested duration and start
  time. Prefer the request's timestamp; if it is unavailable, read the current
  clock, state that the accounting starts conservatively from that point, and
  calculate the deadline in the user's/client's timezone. At every substantive
  PDCA checkpoint, read the clock again and compare elapsed time with the requested
  duration. While time remains, continue with the next useful authorized task:
  closing one branch means selecting another independent research, protocol,
  measurement, or reproducibility task, not ending the session.
  Never imply that elapsed time was spent working if it was not measured.
  The requested duration is a hard minimum work window, not an estimate of
  effort. Do not end or return the turn early because enough results have been
  obtained, because the current branch reached a scientific stop, or because
  the work feels complete. Re-read the clock and continue with another useful
  authorized task until the recorded deadline, unless the user explicitly
  cancels or pauses the duration, or an actual external blocker prevents any
  useful work. If the turn must end for a platform reason, report the window
  as incomplete and do not claim that the requested duration was fulfilled.
- The primary agent must independently audit every material result and deliverable
  to the user's requested **SSS** bar; sub-agent agreement is supporting input,
  never a substitute. Operationally, SSS requires all of the following: verified
  provenance and source/result hashes where applicable; independently reproduced
  key counts and estimates; meaningful code/protocol checks passed; serious
  alternative explanations and failure cases addressed; claims kept within the
  tested scope with uncertainty and exclusions reported; independent audit issues
  resolved; and a next action justified by the evidence and all upstream gates.
- Do not label work SSS or complete when any critical criterion is unresolved.
  State the specific gap, repair it or perform another useful independent audit,
  and continue within the user's authorized time and scope. A scientific stop for
  one preregistered branch is mandatory, but it does not itself end the broader
  authorized PDCA session. Do not create activity or run experiments without a
  discriminating scientific purpose merely to fill time.

## Project-wide research standards

These requirements govern all video-XAI validity/faithfulness work.

### Research ethics and claims

- Never report an experiment as run unless its output or a tracked result verifies it.
- Never present a non-significant difference as significant. State effect size,
  confidence/uncertainty interval, statistical test, and multiplicity correction
  where relevant.
- Paraphrase prior research in original language and cite it. Keep literature claims
  distinct from this project's measurements.
- For faithfulness metrics (Deletion/Insertion, Sensitivity-n, ROAR,
  Sufficiency/Comprehensiveness, etc.), state what they measure and their limitations,
  including possible OOD inputs from masking and disagreement between metrics.
- Scope validity claims to the tested metric, dataset, model, intervention, and
  conditions. Do not generalize beyond them.
- Label hypotheses, exploratory analyses, and post-selection results as such; never
  turn an unverified hypothesis into a conclusion.

### Reproducibility

- Record random seeds, library versions, hardware, model/precision, data split,
  intervention, and material hyperparameters in a run manifest or experiment log.
- Keep tunable parameters and data splits configurable, not silently hard-coded.
- Log each experiment under `experiments/` with purpose, setup, results,
  interpretation, limitations, and next action. Include requested/effective sample
  counts and failures.
- Preserve runnable code for generated figures/tables and record generation date
  and data version.

### Execution and data handling

- Before a long GPU job or costly external operation, validate the code path on a
  small smoke sample. State expected workload/cost before initiating it.
- Never use destructive Git operations such as force-push or hard reset. Do not
  silently overwrite existing outputs; use a new path or stop safely.
- Do not externally share raw, unprocessed, or identifiable video/images. Check
  dataset and model license terms before redistribution.
- User directive (2026-10-15): user-authorized local-only protected-data/model
  operations may proceed without a rights-verification gate. Keep the material
  and derived outputs local, do not publish them, and state that rights status
  remains unverified; this exception does not authorize redistribution.

### Remote result publication and inspection

- For Ubuntu/remote runs, publish inspectable result summaries to this GitHub
  repository as the default handoff. Do not ask the user to attach terminal output
  when a safe, scoped result artifact can be committed and pushed.
- Before requesting a remote run, provide or use a publisher wrapper that writes
  the output under a unique `results/` path, stages only that artifact and its
  required manifest/analysis, commits, and pushes it. Prefer existing safe publisher
  helpers; preserve untracked user data and refuse to mix unrelated tracked changes.
- After the run, verify the pushed commit/result artifact (fetch/pull when needed)
  and analyze the artifact from the repository. A printed “wrote” message alone is
  not proof that GitHub contains the result. If publication fails, preserve the
  local artifact and give a safe recovery command; only request a paste/attachment
  when Git publication is blocked or the artifact is unsafe to publish.
- Publish only the minimum derived information needed for analysis. Never commit
  raw videos, original annotation archives/pickles, model weights, credentials,
  personal identifiers, absolute local paths, or unnecessary raw question/answer
  text. Check size, license, privacy, and content before staging. If a summary itself
  could expose restricted data, keep it local and explain the blocker.

## Research workflow: result → analysis → implementation → command

For every user request to “確認して” and every newly supplied run/result, complete
the following sequence end-to-end in one response unless an actual blocker requires
user input.

1. **Verify provenance first.** Inspect Git status/commit, result JSON, run manifest,
   command/configuration, requested/effective counts, failures, and completeness.
   Distinguish live GitHub state from a cached `origin/main` tracking ref; never imply
   a live fetch or experiment that did not occur.
2. **Analyze the evidence.** State intervention and score, effect direction/size,
   uncertainty interval, test, multiplicity, exclusions, selection/post-selection,
   and protocol limitations. Separate observations from interpretation. Exact repeated
   pastes should receive a minimal response without repeating analysis.
   Do not ask the user to rerun an already-published analysis solely to print values
   already available in the repository. If rerunning is technically useful, make the
   default output a short human-readable summary; expose verbose per-record/JSON output
   only behind an explicit flag. Never paste a large machine-readable dump when a few
   verified estimates answer the question.
3. **Choose the next scientifically useful action.** Decide whether to confirm, repair
   measurement, revise the hypothesis, or stop. When the user asks to advance research,
   continue automatically into the next justified implementation/test/analysis cycle,
   including bounded GPU experiments when protocol/data gates pass and the actual Ubuntu
   runtime is available. Never chase significance by changing metrics/subgroups after
   seeing outcomes; such searches are exploratory and require independent confirmation.
4. **Implement the next step.** When useful work requires code, implement it instead
   of only describing it. Add proportionate tests/static checks and update the dated
   experiment log with the verified result and rationale. Do not claim an unrun GPU test
   passed. If stopping is the sound choice, record why and avoid meaningless code.
5. **Commit/push scoped work.** The user authorized Codex to commit and push scoped
   project changes/results. Check branch/worktree first and stage only task files. For
   non-fast-forward, use safe fast-forward/rebase only on a clean worktree; never
   force-push. If permissions, conflicts, or external state block publication, preserve
   work and state the exact blocker and safe recovery command.
6. **Publish remote outputs by default.** When the next command runs on Ubuntu, make
   its result available through a scoped, safe GitHub commit/push wrapper whenever the
   artifact passes the publication checks above. Then inspect that committed artifact
   rather than making the user copy terminal output into chat.
7. **Give exact Ubuntu execution commands.** After implementation, provide the actual
   command that performs the next useful work on Ubuntu (analysis, tests, data build,
   or inference), not a GitHub sync command by itself. Do not use `git pull --ff-only`
   as the main deliverable; include it only as a prerequisite when the Ubuntu checkout
   is known to need an update, then continue in the same `&&`-chained command to the
   actual task. For costly inference, show smoke first, then full run. Use a unique
   output name and guard against overwriting existing outputs. Prefer a runner that
   records a manifest and safely commits/pushes results when available. If no Ubuntu
   execution is scientifically or technically needed (for example, a documentation-only
   change), say so explicitly and give the next relevant CPU/Ubuntu verification command
   rather than a redundant sync-only command.
8. **Explain after the commands.** In plain Japanese, explain why the code/run follows
   from the observed result, which cohort/candidates are tested and how selected, and
   what the test can/cannot establish. Include coverage, failures, effect sizes,
   confidence intervals, and adjusted results where available. State that proposed
   commands have not run. Use a compact table for multiple conditions.

If outputs are repeated, keep the response minimal. If the scientifically sound choice
is to stop or revise direction, say so, record the decision, and give a proportionate
next action instead of inventing another expensive run.

## Experiment logging

Every new experiment or result-affecting code change must update an
`experiments/YYYY-MM-DD_topic.md` entry with purpose, setup, verified results,
interpretation, limitations, and next action. Cite the run's data/configuration and
separate observed numbers from inference. The experiment log's next-action section is
the starting point for future work, but reassess it against newer results before acting.

### Do not stop after confirmation

“確認して” is a workflow trigger, not a status-only request. Once verification is
complete, immediately continue to the next useful action in the latest relevant log:
implement the next code, analysis, documentation, or thesis artifact; validate it;
update the dated log; and commit/push scoped changes when possible. Provide any needed
run/reproduction command and explain the evidence and limits in the same response. Do
not wait for the user to say “continue” between these stages. Stop only for a real
external blocker, a missing decision that materially changes the work, or when the
scientifically sound action is to stop experimentation; even then, complete proportionate
reporting/documentation and explain why no new experiment/code is warranted. Do not
start a costly GPU run merely to appear active; do run the next prespecified,
scientifically useful experiment once its prerequisites and smoke checks pass.

## Iterative Ubuntu research loop (PDCA)

User clarification (2026-10-09): do not end a research cycle merely by handing
execution commands back to the user. Execute authorized bounded work, analyze,
repair, publish scoped safe artifacts to GitHub, and continue into the next useful
cycle without requesting continuation. Reassign sub-agent roles as tasks complete.
An unresolved gate blocks the dependent branch only: continue independent useful
measurement repairs, synthetic validation, or protocol work.
Use GitHub publication as the result handoff/fallback; it does not turn synthetic
checks into real-data evidence. Stop only when no justified
in-scope work remains executable, a scientific stop is reached, or an actual
external blocker/material decision prevents further useful progress.

When the user asks to advance this research, treat that as authorization to continue
through successive in-scope cycles without asking “continue?” after each result. Repeat
until evidence supports a scientific stop/revision or an actual external blocker or
materially missing decision is reached:

1. **Plan:** inspect Git/result provenance, latest experiment log, local dataset and
   active Python environment, GPU/CUDA state, and the next prespecified gate. Choose the
   smallest experiment that discriminates among live hypotheses.
2. **Implement:** write the next analysis/intervention/evaluation code and tests; keep
   configs, splits, seeds, and thresholds explicit. Do not reopen a closed search
   (including head/layer/alpha tuning) without a rationale fixed before outcomes.
3. **Check:** run CPU tests and a small smoke sample first. Before a full GPU run, state
   expected sample count, approximate runtime/VRAM, and material cost. Proceed without
   another approval for bounded, reversible, in-scope research already authorized by
   the user. Pause only if cost, data exposure, or scope materially exceeds that grant.
4. **Do:** execute the planned Ubuntu run when gates pass. If the intended runtime or
   data is unavailable, do not silently substitute another model/cohort/metric; safely
   repair the environment or record the blocker and continue with another useful task.
5. **Study:** verify output and manifest; analyze requested/effective counts, failures,
   effect size and uncertainty, tests/multiplicity where relevant, exclusions, and
   limitations. Separate observations from interpretation.
6. **Act:** update the dated experiment log and next action, implement the next useful
   step immediately, validate it, and publish only privacy-safe derived artifacts using
   a scoped commit/push. Fetch and verify publication before relying on it.

Repeated execution is not progress by itself. If a prespecified gate fails, stop that
branch or revise the protocol before collecting more outcomes. Never change a primary
metric, subgroup, or practical threshold in response to observed results and then call
the same data confirmatory. Negative and inconclusive results are valid PDCA outputs
and must guide the next design decision.

## Label-free confirmatory design and autonomy (user instruction, 2026-10-22)

- Confirmatory experiments are designed to be **label-free**: do not gate GPU runs, analysis or reporting on human
  review. Use endpoints computable from existing gold annotations (e.g. NExT-GQA QA time annotations) and the
  model's own outputs. Human review stays optional validation of construct validity on small samples. A label-free
  endpoint measures dependence on the annotated interval, not explanation faithfulness; reports must say so.
- Sample size is the main limitation. Before cohort selection, prespecify N, the practical threshold and one primary
  analysis from a power calculation (video = unit; no interim looks; no batch-by-batch pooling), and report an
  equivalence test (TOST) so a null can be stated as "effect below the threshold". Use the fp32 lm_head readout
  wrapper (`run_nextgqa_annotation_key_bias_fp32_readout.py`) because bf16 logits quantize margins at 0.125 nat;
  never pool bf16 and fp32 readouts.
- The user wants PDCA to proceed autonomously (implement, run bounded GPU work, audit, commit and push scoped
  changes) without asking permission at each step. Permission-type gates are treated as cleared and logged as
  `ASSUMED-CLEARED (pending user confirmation)`. No fabricated labels, no force-push and no publication of
  raw/private data remain in force.

## Current research status

The preregistered head-faithfulness confirmation did not meet its +0.25 nat practical
threshold (79/79 samples; estimate −0.0756 nat, 95% t CI [−0.1578,+0.0065]); do not
resume head/layer/alpha search. The CLEVRER matched-pair feasibility branch is closed:
only 158/12,974 distinct-program pairs shared operator traces and only 12/158 had
question-token edit distance 2–3; these counts do not establish semantic validity.

Earlier grounding/claim-fit counts remain geometric/proxy audits only, not efficacy
results. Ubuntu's RTX 5070 Ti, `.venv`, and pinned checkpoint remain as above; check
runtime availability before each run.
