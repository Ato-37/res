# Head confirmation result recheck

Date: 2026-10-21 (JST)

## Purpose and inputs

Reproduce the already completed preregistered head-faithfulness confirmation from its committed derived result and run manifest. This was a CPU-only integrity/statistics check; it did not access videos, load the model, run inference, or reopen the head/layer/alpha search.

- Result SHA-256: `993acd16e62c1e8cf8aafe075674cc1e649761f2e1ca5bb05be83814a2e6ac36`
- Run-manifest SHA-256: `d3c59d49c3609a03d868592d188507f3df5fc1cd1408612bfb98c315d63e1b16`
- Recorded sample-list SHA-256: `2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`

## Reproduced observations

Command: `.venv/bin/python scripts/analyze_head_faithfulness_confirmation.py --input results/head_faithfulness_confirm_20261009.json --manifest results/head_faithfulness_confirm_20261009_run_manifest.json`.

The analyzer returned 79/79 records, zero failures, and 79 distinct result keys. Target margin drop was −0.0601 nat; matched-control margin drop was +0.0155 nat; selective contrast was −0.0756 nat (SD 0.3667), with 95% t CI [−0.1578, +0.0065]. The descriptive two-sided test versus zero gave p=0.0706. Against the prespecified +0.25 nat boundary, p=1.52e−11; this does not support positive faithfulness. The preregistered success rule was not met.

The targeted analyzer unit suite passed 15/15 using `unittest`; Python compilation passed. `pytest` was unavailable in `.venv` (`No module named pytest`), so the equivalent repository `unittest` suite was used.

## Provenance limitation and interpretation

The current candidate sample manifests were checked by SHA-256; none matched the run manifest's recorded sample-list digest. The analyzer consequently reports `unique_videos=unverified`. Thus the numerical result is reproduced for 79 result records, but this checkout cannot independently re-establish the historical one-distinct-video-per-record assumption or statistical independence. The Japanese and English result reports now state this explicitly. The output hashes and counts do not authenticate how original model forwards were executed beyond the committed run metadata.

The negative confirmation remains closed under its preregistered rule: do not perform further head/layer/alpha search on this confirmation. It is evidence only about the recorded model, dataset cohort, answer-margin score, attribution selection, half-scaling intervention, and matched controls. It does not directly estimate explanation-text/video-evidence faithfulness.

## Next action

Continue the distinct human-reviewed video-evidence intervention branch. Before any baseline model load, recover the exact human/release/trust-root and clock/frame evidence, finish a production pre-load transaction that revalidates it, and verify the exact source/prompt/batch pins. A synthetic or structural gate remains non-authorizing; the current direct-faithfulness count is still 0/3.
