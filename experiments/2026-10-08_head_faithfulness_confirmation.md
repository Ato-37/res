# Independent head-faithfulness confirmation: result and closure

Date: 2026-10-08

## Purpose and protocol

Complete the one independent test-set confirmation selected by the CPU design review. The target is the four heads with the largest gradient×activation attribution over all 28 layers. Their final-prompt-position outputs are scaled by `alpha=0.5`; five disjoint controls are matched on head count, layer histogram, and approximate activation norms. For each video, freeze the model's pre-intervention top A–E choice and its strongest foil, and define

`d_i = (baseline margin - target-intervention margin) - mean_r(baseline margin - control_r margin)`.

The estimand is the mean of `d_i`, one independent video per observation. The preregistered success criterion is that the lower endpoint of the two-sided 95% t interval exceed `+0.25 nat`. No secondary subgroup or alternative dose is used to replace this primary test.

## Provenance and observed result

- Result artifact: `results/head_faithfulness_confirm_20261009.json`; manifest: `results/head_faithfulness_confirm_20261009_run_manifest.json`.
- The run artifacts were published with confirmation commit `a80ec61`. The later analyzer/documentation commit `8f22581` is present in the local checkout; live GitHub could not be queried during the current follow-up because DNS resolution for `github.com` failed.
- Requested/effective/failure counts: 79/79/0; all 79 sample keys are unique. The manifest records sample SHA-256 `2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`.
- Configuration: Qwen2.5-VL-7B-Instruct, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit, RTX 5070 Ti, all 28 layers, top-4, alpha `0.5`, five controls, 16 frames, pixel cap enabled, `max_pixels=200704`, seed `20261010`.
- Mean target margin drop: `−0.060127 nat`; mean control margin drop: `+0.015506 nat`; mean selective contrast: `−0.075633 nat` (SD `0.366724`).
- Two-sided 95% t interval: `[−0.157775, +0.006509] nat`. It includes zero and its upper endpoint is below the preregistered `+0.25 nat` threshold. Descriptive two-sided t test vs zero: `p=0.070604`; test vs threshold: `p=1.5211e−11`. The latter is evidence against a mean as large as the threshold in this sample, not evidence of positive faithfulness.
- Baseline choice accuracy was 56/79 (descriptive). The primary analysis includes all 79 examples regardless of correctness. No secondary multiplicity family was tested here.

The local CPU analyzer `scripts/analyze_head_faithfulness_confirmation.py` recomputes each record's control mean and target-minus-control contrast, the primary statistics, and protocol/manifest invariants. Its six focused tests and a combined 17-test related suite pass; `git diff --check` and Python compilation pass.

## Interpretation, limits, and next action

The preregistered practical-effect criterion was not met. For this model, dataset, answer-margin score, attribution rule, and half-scaling intervention, the test does not show that selected heads support the frozen answer more selectively than matched controls. The negative estimate does not prove a universal zero effect, but the interval is inconsistent with the planned `+0.25 nat` success boundary. Do not reuse this confirmation set to change the selected heads, alpha, subgroup, or metric in pursuit of a favorable result.

Close this specific head-faithfulness primary and report it as a negative confirmation. No further GPU search is justified under this hypothesis. A genuinely different research question may be designed on CPU with a new independently justified target, control, estimand, and confirmation plan; it is not approved by this result. The human-grounded mask experiments and randomization sanity checks remain distinct evidence and do not turn this result into a general claim about explanation faithfulness.

Reproduction uses only the committed JSON and manifest (no video/model inference):

```bash
python scripts/analyze_head_faithfulness_confirmation.py \
  --input results/head_faithfulness_confirm_20261009.json \
  --manifest results/head_faithfulness_confirm_20261009_run_manifest.json
```
