# Full-archive candidate availability and continuous review handoff (2026-10-19)

## Purpose

Recheck whether the configured partial NExTVideo directory is limiting a
future human-review cohort, and rerun the synthetic browser test for the
requested sequential-review / one-JSON handoff. This is an outcome-blind data
availability census and software check; it is not a model experiment or a
faithfulness result.

## Setup

The pilot source uses `NExT-GQA/datasets/nextgqa/test.csv`,
`map_vid_vidorID.json`, `gsub_test.json`, and the default video root
`~/nextgqa_sample/NExTVideo`. The local `~/NExTVideo.zip` was inspected through
its central directory, then MP4 bytes were streamed only to compute aggregate
SHA-256 overlap with the existing local `exclude_*.json` manifests. Candidate
metadata checks reproduced the pilot's required answer/span fields without
printing or persisting question text, video identifiers, paths, or hashes.
No video was decoded and no model was loaded. All protected inputs and derived
counts remain local; rights status is unverified.

## Observations

- The pilot's current `test.csv` has 5,553 rows and 990 distinct video IDs;
  the configured root has 533 MP4 files and 500 matching test-video keys.
- The full local ZIP index contains 5,440 MP4 entries. Of these, metadata
  checks found 575 test videos with valid answer/span structure after applying
  the existing `nextgqa_samples_*.json` video exclusions.
- The 575 candidate MP4s (2,350,614,261 bytes streamed) were compared against
  the union of 13 existing private media-exclusion manifests. 86 matched an
  explicitly listed prior hash; 489 did not match those supplied hashes.
- The prior-exposure index is documented as partial and unauthenticated.
  Therefore the 489 unmatched videos are **not certified fresh**; this only
  shows that the small default video root is not the full archive and that the
  current exclusion files do not list those exact bytes.
- The previously noted 85-row candidate count applied to the default-root
  geometry census. It is not the full-archive metadata count and does not
  establish which candidates pass the pilot's decoded timing/control gate.
- The independently run Windows Chrome E2E passed **1/1** (2.04 s). It used
  synthetic videos/items, advanced two reviewers in sequence, downloaded one
  combined JSON, and validated the exact schema and ordered responses. It did
  not open the protected review artifact or create a human response.

## Interpretation and limits

The current three-item informal artifact's continuous flow is operational:
one click starts the next participant, and the final button downloads all
completed responses together. The combined-export behavior is software-tested
but supplies no new human or faithfulness evidence. Its three-item, inline-video
contract cannot be assumed to scale to a larger cohort without updating and
testing the builder, inspector, importer, payload-size bounds, translations,
and browser workflow.

Direct faithfulness remains **0/3 formal claim reviews, 0/15 verified
clock/frame correspondences, 0/3 prediction-link reviews, and 0/3 direct
scores**. There is no direct `DeltaF`, confidence interval, or hypothesis test.
GPU/CUDA was available at the check (RTX 5070 Ti; 14,746 MiB free VRAM; PyTorch
2.11.0+cu128; 13.89 GiB WSL `MemAvailable`), but no direct GPU run was valid:
the trusted formal reviews and frame-correspondence gates remain open. The
fixed candidate proxy cohorts remain closed and were not repeated.

## Next action

Use the current three-item artifact for additional informal respondents if
desired; its tested flow already supports sequential people and one combined
download. Before creating any larger review artifact or spending GPU on another
claim-generation batch, freeze a new cohort-selection and review protocol,
resolve the partial prior-exposure limitation as far as the available records
allow, confirm Japanese translation coverage, and test realistic HTML/JSON
size bounds. For the direct thesis endpoint, the next eligible GPU transaction
remains the fixed three untreated baseline forwards after trusted Stage-1 review
and all 15 clock/frame correspondences pass.
