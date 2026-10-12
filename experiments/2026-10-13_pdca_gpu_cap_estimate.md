# 2026-10-13: high-resolution pixel-cap GPU estimate

## Purpose and scope

This cycle tests whether the newly exposed processor pixel cap changes the
actual Qwen2.5-VL input grid at `frame_size=1024`, and records a small
necessity-choice margin estimate under the cap. This is an engineering and
preprocessing diagnostic. It is not direct explanation-faithfulness evidence,
and it does not reopen the closed head/layer/alpha search.

## Setup

- Existing `.venv`, Qwen2.5-VL-7B-Instruct, 4-bit quantization, BF16 compute,
  eager attention, seed `20261013`.
- Local three-sample NExT-GQA smoke cohort; 16 uniformly sampled frames per
  video; `max_pixels=200704`.
- Cap comparison: one sample at `frame_size=1024` with and without
  `--cap-pixels-per-frame`.
- Estimate: three samples with the cap enabled; the runner's prespecified
  baseline-correct primary cohort and bootstrap interval are retained.

## Verified results

| condition | requested/effective | failures | primary n | primary mean (nat) | 95% interval |
|---|---:|---:|---:|---:|---:|
| cap, 3-sample smoke | 3/3 | 0 | 2 | 1.5000001 | [0.2500001, 2.7500000] |
| cap, one-sample shape witness | 1/1 | 0 | 1 | 0.2500001 | not estimated |
| no cap, one-sample shape witness | 1/1 | 0 | 1 | 0.1250 | not estimated |
| no cap, 3-sample smoke | 3/3 | 0 | 2 | 0.2500000 | [0.1250, 0.3750] |
| cap + blur, 3-sample smoke | 3/3 | 0 | 2 | 1.2500001 | [0.0000001, 2.5000001] |

The cap-enabled high-resolution input grid was `[8, 40, 26]` with
`input_ids_shape=[1,2161]`; the uncapped comparison was `[8,38,24]` with
`input_ids_shape=[1,1905]`. Thus the cap is operational at this setting.
The one-sample score difference is a diagnostic only: it is confounded with
the changed processor grid and has no uncertainty estimate.

Raw artifact hashes are preserved only in the local run records. Privacy-safe
derived summaries are
`results/necessity_choice_margin_cap_frame1024_smoke_derived_20261013_01.json`
`results/necessity_choice_margin_cap_frame1024_smoke_derived_20261013_02.json`,
`results/necessity_choice_margin_cap_frame1024_blur_smoke_derived_20261013_01.json`,
`results/necessity_choice_cap_black_vs_blur_paired_20261013_01.json`,
`results/necessity_choice_cap_vs_uncap_paired_20261013_01.json`,
and `results/necessity_choice_frame1024_cap_comparison_20261013_01.json`.

## Interpretation and limitations

The implementation now records `frame_size`, `max_pixels`, cap state,
`input_ids_shape`, and `video_grid_thw`, making future runs auditable. The
runner now also records the source Git commit and input-manifest SHA without
publishing manifest contents; a fresh 1/1 provenance smoke recorded commit
`f7027f874fcec70de7ac28fc61b135c6ccd0552a` and completed with zero failures.
Three-sample black, blur, and uncapped reruns under this provenance-aware
runner all completed 3/3 with zero failures. The strict paired comparator
accepted the shared model/seed/frame/processor settings and reproduced the
black-minus-blur and cap-minus-uncapped estimates above. Their privacy-safe
outputs are `results/necessity_choice_cap_black_vs_blur_paired_20261013_02.json`
and `results/necessity_choice_cap_vs_uncap_paired_20261013_02.json`.
three-sample primary estimate is unstable (n=2) and the interval is a smoke
diagnostic; it cannot support a faithfulness or efficacy claim. Masked-input
OOD effects, choice-margin calibration, answer-label validity, and semantic
claim alignment remain unresolved. The NOAA human-attestation gate remains
`no_go`.

The black-versus-blur primary means (1.50 vs 1.25 nat) are descriptive only:
the cohorts are tiny, the intervals overlap, and no between-mask hypothesis was
registered. They should not be interpreted as a treatment effect.

To prevent cohort-matching mistakes, `scripts/compare_necessity_smoke_conditions.py`
now validates identical sample-ID sets and shared model/seed/frame/processor
settings internally, and publishes only aggregate paired differences. For black
minus blur, the paired primary estimate was
`+0.24999999 nat` (n=2; bootstrap 95% interval `[+0.24999995,
+0.25000003]`). This degenerate small-sample interval is a reproducibility
check, not inferential evidence.

The paired cap-minus-uncapped primary difference across the same three records
was `+1.2500001 nat` (n=2; bootstrap 95% interval `[+0.1250002,
+2.3750000]`). Because the processor grids differ, this is evidence that the
preprocessing setting materially changes the measured margin, not evidence for
or against explanation faithfulness.

## Next action

Keep this cap configuration fixed for the next prespecified claim-selective
intervention/control design. Do not tune the cap or search heads on these
outcomes. Before any larger cohort, run the existing CPU tests and a fresh
privacy/provenance audit of the derived artifact.
