# 2026-10-09: temporal grounding data feasibility audit

## Purpose

Implement the first CPU-only validity gate for the proposed internal-intervention study: verify the NExT-GQA QA-to-gsub join, interval structure and duration bounds, previously used-video overlap, and the shape of local frame-to-time arrays. This is a data audit, not an intervention or faithfulness experiment.

## Setup

- Inputs: local NExT-GQA validation/test CSV, gsub annotation JSON, video map, frame2time JSON, and all local `data/nextgqa_samples*.json` manifests.
- Analysis: `scripts/audit_grounding_plan_data.py`; counts-only JSON output. The runner `scripts/run_grounding_plan_audit_and_publish.py` writes a unique dated result/manifest, refuses overwrites and tracked dirty state, and stages only those two artifacts.
- Validation: four synthetic unit tests and Python byte-compilation. No random sampling or GPU/model execution.
- Candidate exclusion: canonicalized video-map paths from the prior sample manifests; raw IDs/paths were used locally but are not emitted.
- Actual command is recorded in the audit result manifest; source dataset hashes are in the result JSON.

## Verified observations

| Split | CSV rows | gsub join | Structurally valid interval rows | Duration-valid interval rows | Malformed interval rows | Past-duration rows | Unique eligible videos after old-video exclusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 3,358 | 3,358 | 3,358 | 3,355 | 0 | 3 | 567 |
| test | 5,553 | 5,553 | 5,517 | 5,517 | 36 | 0 | 575 |

- No duplicate video-question rows were observed in either split.
- All joined interval values used the documented list-of-pairs container. Thirty-six test rows contained at least one malformed interval value; their raw annotation values are intentionally not reproduced here.
- The prior manifests covered 415 unique canonical video paths. Validation had 0 overlapping rows; test had 2,611 overlapping rows. Eligibility counts are video-level after excluding those prior paths and invalid/duration-overrun rows.
- The local frame2time maps had 570 validation and 999 test vectors; every vector had exactly 32 finite timestamps. No vector was non-monotonic. Adjacent repeated timestamps occurred in 63 validation and 123 test vectors.
- The validation interval-span median was 4.5 s (range 0.1–82.6 s); test median was 4.1 s (range 0.2–91.8 s). These are descriptive summaries, not sampling-based estimates.
- Tests: 4/4 passed; `py_compile` passed. The CPU audit completed on the full local annotation tables.

## Interpretation

The tables support a usable *candidate pool* for a later split-level protocol: validation has 567 previously unused eligible videos and test has 575. They do not establish independent confirmation feasibility by themselves; video-to-question assignment, claim/span semantic fit, duplicate-timestamp consequences, and a prespecified sample-size requirement remain to be checked.

The 32-point timestamp arrays establish a coarse time grid in the local metadata, not a proven mapping to Qwen2.5-VL visual tokens or intervention chunks. Repeated adjacent times indicate non-unique timestamps in some arrays, not necessarily invalid data. The 36 malformed test rows and 3 validation duration overruns should be excluded or handled by a rule fixed before model outputs are examined; they must not be silently dropped after outcome inspection.

No effect size, confidence interval, hypothesis test, or multiplicity adjustment applies: this was a deterministic structural audit, not an outcome comparison.

## Limitations

- Counts and interval validation cannot show that a human interval supports a generated model claim.
- The source annotations and prior manifests are local/untracked, so this result must be interpreted alongside the stored input hashes; no raw text, IDs, paths, or video was published.
- Frame2time-to-Qwen chunk correspondence remains unverified. No video decoding, Qwen inference, internal intervention, or faithfulness evaluation was run.
- Video-level eligibility counts are upper bounds on a usable cohort; question-type balance and one-question-per-video assignment still need a prespecified, outcome-blind rule.
- Dataset licensing/redistribution and the actual remote GPU environment have not been re-audited here.

## Next action

Continue with a CPU-only time-mapping gate: verify key coverage between the gsub/video map/frame2time metadata without emitting keys, quantify interval-to-grid coverage and effective temporal resolution, and inspect the local Qwen2.5-VL video preprocessing code to determine whether its temporal chunks can be aligned to those timestamps. In parallel, check annotation/question-type semantics from dataset documentation. Do not start GPU inference until these mapping and claim-validity checks are resolved. Do not resume the closed head/layer/alpha search.
