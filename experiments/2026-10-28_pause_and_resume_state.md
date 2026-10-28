# Pause state and how to resume (2026-10-28 13:20 UTC, paused at the user's request)

Stopped: all GPU jobs and the 30-minute monitoring cron job. Nothing is running except the page-cache eviction loop (scripts/evict_page_cache.py --loop 300, harmless). All results so far are committed and pushed (latest log commit 122cb4b plus this note).

## Paused run: TVBench pathway knockout on Qwen2.5-VL-7B (tag u)
- Partial file: data/private_nextgqa/tvb0_full_20261028u.json.partial.jsonl, 153 of 400 records (action_sequence 100, scene_transition 53), 0 corrupt lines, 0 duplicate videos. No final receipt exists yet.
- **Resume was tested**: restarting with --resume re-read the partial file and appended a new record (152 -> 153) before the test run was stopped; the cohort is deterministic (hash-ranked), so the resumed run continues with the same videos.
- Resume command (run from the repository root; GPU command, needs elevated execution in this sandbox):
  `.venv/bin/python scripts/run_tvbench_pathway.py --mode full --n 400 --per-task 100 --tvbench-root data/private_tvbench --frames 8 --user-authorized-local-only --resume --output data/private_nextgqa/tvb0_full_20261028u.json`
  then, after the final receipt appears: `cd scripts && ../.venv/bin/python analyze_pathway_knockout.py --receipt ../data/private_nextgqa/tvb0_full_20261028u.json --output ../results/tvb0_summary_20261028u.json`
- Remaining workload at the measured speed (about 50 s per video, 247 videos left): about 3.5 hours; run no other GPU job at the same time (concurrent GPU jobs on this machine are about 10x slower).

## Not run (deferred or blocked)
- WP5 val and WP6 val (planned in experiments/2026-10-28_wp5_wp6_input_level_and_stability_plan.md): not run; WP5/WP6 test and WP7 are done.
- Official Map the Flow code with the official checkpoint (experiments/2026-10-28_map_the_flow_reproduction_plan.md): the permission layer denied running the cloned third-party code; waiting for the user's explicit permission. Prepared: isolated virtualenv ~/external/mtf-venv (python 3.12, torch 2.11 cu128, transformers 4.40.0, GPU verified), checkpoints in ~/external/models (LLaVA-NeXT-7B-Video-FT, llava-v1.6-vicuna-7b-hf), the repository clone in ~/external/map-the-flow (config_dataset.py still holds the placeholder dataset path; to be set to data/private_tvbench).
- Full own re-implementation of the paper's remaining analyses (Table 2/B, Figs 7-9, windows, failure cases, pruning): not started.

## Interim presentation (next week)
Advisor-facing explanation document: Claude Docs "研究計画と土台論文のご説明(指導教員向け 1〜2 ページ)"; progress document: "中間発表 進捗確認と要点"; deck docs/interim_presentation_20261028.pptx (13 slides; needs the base-paper explanation moved to the front, WP7 and TVBench results, author name; not visually rendered). The advisor has not yet been told about the base paper and plan (per the user).
