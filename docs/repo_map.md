# Repository map (written 2026-10-26)

Graduation-thesis research repository (Kumamoto University): Qwen2.5-VL-7B video-VLM, how video information reaches the answer and what corrections are possible. Policy and status live in `AGENTS.md`; this file only says where things are.

## Layout
| path | what | tracked |
| --- | --- | --- |
| `AGENTS.md`, `claude.md` | policy, workflow, current research status (single source of truth) | yes |
| `docs/` | plans, protocols, thesis drafts, correction planning, `script_index.md` (generated), this map, `interim_presentation_20261026.pptx` | yes |
| `experiments/` | one dated log per experiment or decision: purpose, setup, verified results, interpretation, limits, next action. Plans are written before runs (`*_plan.md`, `*_prespec.md`) | yes |
| `scripts/` | flat on purpose (receipts, queues and logs reference file names): runners `run_*`, analyses `analyze_*`, tests `test_*`, libraries `qta_*`; catalogue in `docs/script_index.md` | yes |
| `scripts/queues/` | sanitized copies of the shell queues used for the 2026-10-23..26 GPU runs | yes |
| `results/` | privacy-safe summaries (no IDs, no raw text). Ignored by `.gitignore` pattern `results/`; add single files with `git add -f <file>` (never a directory) | files only |
| `data/private_nextgqa/` | local-only receipts, cohort locks, manifests, partial JSONL, logs, decoded-video roots | no (ignored) |
| `data/private_minerva/`, `data/gpu_telemetry/`, `NExT-GQA/`, `datasets/` | local datasets and telemetry | no |

## Where to start
1. `AGENTS.md` section 8 (status), then the newest `experiments/2026-10-2x_*` logs.
2. Result summaries: `experiments/2026-10-25_sgc_val_result.md`, `experiments/2026-10-24_csl_plan.md` (CSL result appended), `experiments/2026-10-24_evidence_focusing_pooled_verdict.md`.
3. Interim material: `docs/interim_presentation_20261026.pptx`.

## Branches and publication
- Published branch: `origin/internal-ablation-wiring-20261018` (updated through the clean worktree `~/.res_publish_wt`, fast-forward only; never force-push).
- The main checkout's local branch is stale (ahead 13, behind 76 relative to that origin branch, diverged). The 13 local commits are preserved as `origin/backup/local-head-20261026`. The main checkout also holds uncommitted edits (43 modified files) that were not published; do not reset it without checking them.
- Published copies of 136 previously local-only logs/protocols/scripts have absolute local paths replaced; the local originals are unchanged.

## Housekeeping tools
- `scripts/evict_page_cache.py --loop 300`: evicts clean file cache (videos, model weights); WSL2 otherwise keeps it as VM memory that Windows counts as used.
- `scripts/build_script_index.py`: regenerates `docs/script_index.md` (also lists scripts with no reference, for review only).
- Git health on 2026-10-26: `git fsck` clean; 13.7 GB of interrupted-push temp packs removed from `.git` (19 GB -> 4.5 GB).
