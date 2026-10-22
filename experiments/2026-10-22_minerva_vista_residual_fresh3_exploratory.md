# Minerva residual intervention: fresh 3-video exploratory run (2026-10-22, local)

## Purpose
Test whether the earlier single-video (previously exposed) contrast, anchor-pair vs
farthest temporal-sham residual steering on object-name token log-prob, appears on
videos not used before. Exploratory, 3 rows, descriptive only; no significance claim.

## Setup (fixed before outcomes)
- Cohort: `minerva_metadata_proxy_anchor_cohort_v2_followup_20261022_01.json`
  (frozen 2026-10-22T06:27Z, requested 3 rows, one video per participant, 12 exposed videos excluded).
- Official MP4/time-map downloaded to the protected local area and verified by MD5; official media/time-map
  CPU preflight (`preflight_minerva_grounding_cohort.py`): 3/3 pass.
- Runner `scripts/run_minerva_vista_residual_smoke.py --fresh-generation` (new mode: baseline sentence is
  generated greedily in-run; it must pass `_generated_target`; failures kept, no replacement).
  Qwen2.5-VL-7B (revision cc59489...), 4-bit, bf16 compute, 64 frames, block 14, 10% RMS dose, seed 20261022.
- Hardware: RTX 5070 Ti, peak VRAM ~11.3 GB, ~20 s GPU section per row.
- Rights status unverified; local-only. Raw receipts stay in `data/private_minerva/`.

## Results (verified from the GPU receipts)
Requested 3 / effective 2. Contrast = (anchor - baseline) - (sham - baseline), nats per object-name token (1 token each).

| row | status | anchor-baseline | sham-baseline | anchor - sham |
|---|---|---|---|---|
| 0 | failed `generated_target_unclear` (generation done, target not unambiguous; retained, not replaced) | - | - | - |
| 1 | completed | -0.01304 | -0.00483 | -0.00821 |
| 2 | completed | +0.03634 | +0.03146 | +0.00488 |

Mean of 2 rows: -0.0017 nat/token; opposite signs; n=2, so no CI or test is meaningful.
The `_01` GPU attempts failed from an implementation bug (`NoneType.get` after generation, before any
score); their failure receipts are retained and superseded by `_02` (CPU receipts rebuilt for the new runner hash).

## Interpretation and limits
- Observation only: effects are tiny (|x| < 0.04 nat/token), inconsistent in sign, and far below the 0.25 nat
  practical threshold used earlier. No evidence that anchor steering beats the temporal sham; n=2 also cannot detect an effect in either direction. The 0.25 nat figure is a loose reference from the earlier head-faithfulness metric, not a gate for this metric.
- Anchor times are a 30 fps proxy; whole-pair mean-fill is OOD; sham is farthest-endpoint, not exchangeable.
- Likelihood of an object name is not explanation faithfulness; no human support labels were collected.
- Selection effect (independent audit): `_generated_target` requires the model's own sentence to name the metadata object, so effective rows are conditioned on that. Each effective row scores a single object-name token (noisy).
- Video-fresh but not participant-fresh. Row 0 exclusion is a post-hoc-visible failure; it is retained.

## Next action
Do not tune layer/dose on these results. Direct faithfulness still needs human claim-support responses
bound to the cohort11 v3 artifact. Remaining independent option: a labeled full-BF16 precision check
(24 GB GPU) per `docs/lab_24gb_gpu_experiment_queue_20261022.md`, The privacy-minimized derived summary exists: `results/minerva_vista_fresh3_summary_20261022_01.json` (built by `scripts/summarize_minerva_vista_fresh_cohort.py`; independent audit reproduced all numbers and bindings, found no identifiers).

## PDCA checkpoint (2026-10-22 07:32-08:32Z window, `/pdca 1h`)
- ASSUMED-CLEARED (pending user confirmation): publication gate. `git push` was non-fast-forward (remote branch
  carries re-hashed copies of earlier commits plus new ones) and the local worktree is dirty. Per the user's
  instruction to proceed, the two run commits were cherry-picked cleanly onto `origin/internal-ablation-wiring-20261018`
  in a separate clean worktree and pushed as a normal fast-forward (`db76a5f`, verified by fetch). The local branch
  remains diverged and was not rewritten; no force-push.
- Added `scripts/test_summarize_minerva_vista_fresh_cohort.py` (5/5 pass): contrast recomputation, tamper,
  cohort-binding, non-fresh receipt and hook-cleanup rejection.
- Decision: no additional fresh rows (9 eligible videos remain after exclusions). The earlier log already says not
  to run more rows merely to raise n for a method with no selective benefit, and n=2 single-token scores cannot
  discriminate; revisit only if a human-labelled support anchor or a prespecified 24 GB BF16 check is available.
- Human-response gate (cohort11 v3): not fabricated; it stays `no_go` for direct faithfulness. "Assumed cleared"
  applies to permission-type gates only and was not used to invent labels.
