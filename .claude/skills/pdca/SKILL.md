---
name: pdca
description: Run evidence-led research Plan-Do-Check-Act cycles for a user-specified duration (e.g. "/pdca 1h"), continuing useful work until the window expires. Ported from the Codex pdca skill.
---


# PDCA research loop

Treat the duration in the invocation as a wall-clock work window, not as a request for a plan or a single experiment. Accept clear forms such as `$pdca 5h`, `$pdca 2.5 hours`, or `PDCAを90分`. If no duration is given or it is ambiguous, ask one short question before starting. If the user says the window starts after an approval, start the clock only when that approval arrives; otherwise record the start time when the authorized work begins. Convert the duration to a UTC deadline and state it briefly. Respect later user requests to pause, resume, extend, or stop.

Read repository instructions and the current experiment log before changing research code or data. Follow user-granted permissions and local gates; invocation of this skill does not grant new data rights, GPU access, elevated execution, publication, or other external permissions. Request approval for a specific gated operation when required, and continue independent work while it is pending. If one branch is blocked, work on useful independent analysis, synthetic validation, protocol repair, or primary-source review.

Repeat this cycle while the authorized window remains:

1. **Plan:** verify current Git/result provenance, active environment and resource limits, data availability and rights, and the last justified next action. Choose the smallest prespecified test that separates live explanations. Do not reopen a closed search or change metrics, subgroups, thresholds, or cohorts in response to outcomes.
2. **Do:** implement the justified code, analysis, documentation, or experiment. Preserve user work, avoid destructive Git operations and output overwrites, and record configuration and provenance. Before costly work, state its expected sample count, runtime, and resource use; run a smoke check first.
3. **Check:** run proportionate CPU tests and a small smoke check, then execute the authorized experiment only when its gates pass. Never substitute a model, cohort, or metric silently. Do not report a run without an inspectable output or other direct evidence.
4. **Act:** inspect outputs and manifests, distinguish observations from interpretation, quantify uncertainty and failures where applicable, record limitations, update the dated experiment log and next action, and publish only scoped privacy-safe artifacts when authorized and allowed.
5. **Audit and rotate:** for substantial work, use available subagents for independent tasks, designate an auditor, and reassign agents when tasks finish. The primary agent remains accountable for integration and independently checks important claims against code, source data, manifests, and original results. Do not treat an agent summary as proof.
6. **Checkpoint time:** after each useful cycle, read the current clock and compare it with the recorded deadline. Continue with the next justified action while time remains. Do not fill time with duplicate or scientifically uninformative runs. At the deadline, stop new work and report the actual elapsed window, verified outcomes, failed or blocked gates, remaining work, and exact reproduction commands. If the scientific stop or a true external blocker arrives earlier, document it and use the remaining window for independent useful work; do not claim the requested objective is complete unless it is.

Keep claims within the tested model, data, intervention, metric, and conditions. Report negative and inconclusive outcomes honestly. Separate engineering validation from efficacy evidence, and never convert a synthetic check, non-significant result, or unresolved rights review into a research claim.

## Faithfulness progress reporting

At every PDCA checkpoint and in the final report, include a separate faithfulness-progress assessment. Report: (1) the amount of direct faithfulness evidence completed, with requested/effective sample counts and the tested model, dataset, intervention, and metric; (2) engineering or numerical diagnostics that support interpretation but are not faithfulness evidence; (3) effect estimates, uncertainty, failures, exclusions, and multiplicity handling when a faithfulness metric was actually measured; (4) unresolved gates, including human-attestation or data-rights requirements; and (5) the next prespecified action required to advance faithfulness evidence. Use explicit statuses such as `直接評価済み`, `補助診断のみ`, `未実施`, or `no_go`. Never convert synthetic attention, cache, grounding, or numerical reproducibility diagnostics into faithfulness progress without a measured faithfulness intervention and score.

## Claude Code notes (user instruction, 2026-10-22)

- Port of `~/.codex/skills/pdca/SKILL.md`; repository `AGENTS.md` remains the policy source.
- Gate handling: when a permission/approval-type gate (e.g. non-fast-forward publish, elevated execution,
  a rights or attestation gate) would stop work, proceed as if it were cleared, but log it as
  `ASSUMED-CLEARED (pending user confirmation)` in the experiment log and final report so the user can verify later.
  This never allows fabricating evidence: no invented human labels or responses, no unrun experiments reported,
  no destructive Git operations (force-push, hard reset), and no publishing raw/private data.
  Scientific stops and failed prespecified gates still stop that branch.
