# Discrete frame-index claim-mask pilot

## Purpose

Explore whether frames selected by a machine as support for a generated claim
affect Qwen2.5-VL QA-answer margins. This is a machine-localization sensitivity
measure, not a direct claim-faithfulness result. Human-review collection was
waived for local exploration; no human review was performed.

## Setup

- Dataset: private local NExT-GQA test split; 5,553 rows scanned by the
  candidate-pool builder. Existing-history filtering excluded 594 prior media;
  14 rows had invalid intervals and 2,852 were excluded by sample-path rules.
- A 20-video outcome-blind candidate pool was first built. Its pair-hash
  allowlist was incompatible with the existing claim selector and selected no
  records, so it was not used. The 20 selected media hashes were added to the
  existing 93-media exclusion set (113 unique exclusions total). A fixed
  selector then found 85 eligible candidate videos and selected 3 unique media;
  timing checks passed for all selected videos. The generated packet's three
  media hashes were disjoint from all 113 exclusions.
- Qwen2.5-VL-7B generated claims with `v2_short` on 16 sampled frames;
  structured claim generation succeeded for 3/3 rows. The packet is private,
  mode 0600, SHA-256
  `bf5673870ff08d499297943b306bd9d70521dbadc80928ff40b3392a95008b32`.
- For each record, Qwen2.5-VL-3B selected one inclusive contiguous span of 1–4
  frame indices from a labeled 4×4 contact sheet. Qwen2.5-VL-7B scored the
  frozen gold-versus-best-foil next-token QA margin under full video, black
  masking of the machine-proposed span, and black masking of an equal-count
  nearest-rank temporal control. Seed 20261021; BF16 compute, 4-bit model
  loading; eager attention; local pinned model revisions; offline inference.
- GPU: RTX 5070 Ti, 16,303 MiB total; run started with 14.657 GiB free VRAM
  and 13.714 GiB host-available memory. Peak allocated VRAM was 12.483 GiB.
  Runtime was 48.948 s for the full run. The receipt pins source/helper hashes,
  inference-package versions, model revisions/config hashes, prompt hash,
  packet hash, seed, and execution precision/quantization.
- CPU validation at first-run freeze: focused suite passed 7/7; `py_compile` and scoped
  `git diff --check` passed. Smoke authorization now explicitly requires requested,
  attempted, and completed scorer-forward counts all equal five; focused tests
  exercise each count independently and currently pass 10/10.

## Verified result

The smoke completed 3/3 scorer forwards. The full run completed 3/3 rows and
9/9 scorer forwards, with status `passed` and no partial rows. For each row,
the primary contrast is
`matched_control_black QA margin - claim_interval_black QA margin`:

| Quantity | First cohort: unconstrained | Second cohort: finite JSON grammar |
| --- | ---: | ---: |
| Mean contrast | +1.1667 nat | +0.6667 nat |
| Sample SD | 1.5877 | 0.4018 |
| Range | +0.2500 to +3.0000 nat | +0.3750 to +1.1250 nat |
| Positive / negative / near-zero rows | 3 / 0 / 0 | 3 / 0 / 0 |

The first cohort receipt was independently audited: packet/source/exclusion
hashes and mode-0600 permissions matched; all nine forwards and the three
contrasts were recomputed. The second cohort was independently audited against
its disjoint packet, exclusion set, smoke/full pins, and arithmetic. These are
descriptive summaries over deterministically selected records; no confidence
interval, hypothesis test, or population estimate is reported.
The mode-0600 full receipt SHA-256 is
`f4bfeb1ae3f1105953936a6aae695d49984ed5efaeb7a9c247270dbd3d541ac5`.

## Disjoint follow-up cohort

To check whether the direction persisted, a second three-video cohort was
generated after adding the first cohort's three media hashes to the 113-media
exclusion set. It was disjoint from both the previous pool and the first
packet. Claim generation again produced structured records for 3/3 videos.
After strict-format failures in exploratory runs, this cohort used a finite
token grammar constraining the localizer to the 58 canonical JSON strings for
all valid 1–4 frame spans. Its smoke passed 3/3 and its full run passed 9/9
forwards (3/3 rows), with no partial rows. Current CPU validation passes 9/9.

| Cohort / localizer decoding | Mean black contrast | Sample SD | Range |
| --- | ---: | ---: | ---: |
| First 3 videos, unconstrained decoding + unique-object parser | +1.1667 nat | 1.5877 | +0.2500 to +3.0000 |
| Disjoint 3 videos, finite JSON-token grammar | +0.6667 nat | 0.4018 | +0.3750 to +1.1250 |

All six observed contrasts were positive. Because the localizer decoding
protocol changed between the cohorts, these are reported separately; the
six signs are only descriptive cross-version agreement, not a pooled effect or
confirmatory estimate. The second private full receipt SHA-256 is
`4aff13446296d1faa3c07c3db583fa46d8e88ba854f84150dea183f4200b932e`.

## Blur-intervention cohort

A third disjoint test cohort was selected after adding both earlier cohorts to
the private exclusion set (119 excluded media). The finite JSON-token grammar
was unchanged. This protocol scored five conditions per record: full video,
candidate interval black/blur, and matched-count temporal control black/blur.
Smoke passed 5/5 forwards; full passed 15/15 forwards over 3/3 rows with no
partial rows. The 0600 full receipt SHA-256 is
`3c64789ac95c072563424b0e0d04a732039a68fa712396a76985ed0af59bba0f`.

| Contrast | Mean | Sample SD | Range | Positive rows |
| --- | ---: | ---: | ---: | ---: |
| Matched-control black minus candidate-interval black | +0.3750 nat | 0.2165 | +0.2500 to +0.6250 | 3/3 |
| Matched-control blur minus candidate-interval blur | +0.5417 nat | 0.1909 | +0.3750 to +0.7500 | 3/3 |

These are separate descriptive contrasts for this third cohort. Positive blur
contrasts make a black-fill-only explanation less sufficient, but blur remains
an intervention with distribution shift, and neither candidate interval has
human adequacy verification. The third artifact was independently audited:
packet/source/exclusion hashes, disjointness, 5/5 smoke and 15/15 full forwards,
model/runtime pins, and both contrast calculations matched; no material issues
were found beyond adding an explicit smoke attempted-forward assertion, now fixed.

## Fourth disjoint five-condition cohort

The fourth cohort was generated after excluding 122 media hashes from the prior
three full cohorts and earlier candidate pool. The packet contains 3/3 unique
videos with zero overlap; packet, exclusion manifest, smoke and full receipt are
local mode 0600. Claim generation succeeded 3/3. Smoke passed all 5/5 requested,
attempted and completed scorer forwards. Full passed 3/3 rows and 15/15 forwards;
elapsed time was 68.399 s, with 12.49 GiB initial free VRAM and 12.48 GiB peak
allocated VRAM. Both models used pinned local revisions, BF16 compute and 4-bit
loading.

| Contrast | Per-video values (nat) | Mean | Sample SD | Positive / negative / near-zero |
| --- | --- | ---: | ---: | ---: |
| Matched-control black minus candidate-interval black | −0.1250, +2.8750, +0.0000 | +0.9167 | 1.6971 | 2 / 1 / 0 |
| Matched-control blur minus candidate-interval blur | +0.5000, +2.7500, +0.2500 | +1.1667 | 1.3769 | 3 / 0 / 0 |

The black result is heterogeneous, including one negative and one numerically
near-zero contrast; blur contrasts are positive in all three videos. This fourth
cohort does not establish a stable effect: n=3, deterministic selection, no
confidence interval or inferential test. It weakens a broad assertion that the
black-mask effect consistently favors the machine-selected interval. Blur
remains an exploratory distribution-shift intervention. Independent audit
confirmed packet/source/exclusion hashes, disjointness from all prior three
cohorts, smoke/full pins, counts and arithmetic. Private full receipt SHA-256:
`f8111345666bd0037067b8e6d0b9f53adcf0566e1540b7ceecee83535bb152ba`.
Human review was waived for this local exploratory run and was not performed;
direct claim-faithfulness scoring was not claimed.

## Interpretation and limitations

Observed: the 7B QA margin was lower under the interval selected by the 3B
model than under the equal-count temporal control in all three records.
Inference is limited to this finite exploratory cohort and this machine-selected
interval-dependence endpoint. It does not show that the interval actually
supports the generated claim, that the claim is correct, or that explanations
are faithful. The localizer and scorer share the Qwen model family. A 16-frame
contact sheet reduces spatial resolution, span lengths are restricted to 1–4
frames, temporal controls need not be semantically matched, and black masking
can create out-of-distribution inputs. Human review remains 0/3; direct
claim-faithfulness scores remain 0/3.

The work included protocol/debugging failures that produced no valid full
result: an incompatible pair allowlist selected zero records; strict JSON
format failures and VRAM guards stopped multiple runs before complete scoring.
The parser, finite constrained decoder, and CUDA cache handling were repaired
and CPU-tested before the second cohort's successful full receipt. Failed runs
are not observations and their incomplete forwards are not counted.

## Next prespecified check

The proposed test-split continuation was not run: after excluding the 125
previous media hashes, the frozen selector found only 1 eligible video rather
than 3. No model inference was started; this is neither a scored failure nor an
observation. Do not substitute that single record into the failed three-row
plan. To continue without reusing the same test cohort, the next pre-outcome
feasibility check is a separately labeled validation-split cohort: exclude the
union of source-video hashes recorded in all 394 owner-only private JSON
artifacts (1,032 unique hashes), then ask whether the fixed selector can
provide three unique eligible videos. The CPU feasibility check found 0/33
validation candidates after the known-media exclusion. Thus this branch is
closed for lack of a disjoint local cohort; no model inference ran, and we do
not weaken N or replace records. Human review is waived as a prerequisite and
remains not performed.

## Frozen ancillary construct-validity check

With no further disjoint cohort available, use the saved machine-localized
indices and dataset QA intervals from the four completed cohorts for a
separate outcome-independent analysis. Before reading interval or margin
values, freeze this diagnostic: map selected frame indices to their saved
relative presentation times; obtain QA-interval frame indices with the
repository's strict interval/time helper; report the machine interval's
frame-set Jaccard overlap with that benchmark interval and its within-row rank
among all contiguous 1–4-frame spans in the fixed 16-frame grid. Keep cohorts
separate (the first localizer decoding differed), include all rows, and report
no inferential test or pooled faithfulness estimate. This is alignment with
question-level dataset timing annotations, not validation that the interval
supports the generated claim. All artifacts and row-level intermediates remain
private.

The analysis runner is `scripts/analyze_claim_frame_index_qa_overlap.py`; its
focused synthetic suite passes 3/3, and the combined runner/analysis suite
passes 13/13. Static compilation and scoped whitespace checks pass. An initial
implementation incorrectly expected five-condition/15-forward receipts only;
the gate was corrected to accept complete three-condition/9-forward cohorts
or five-condition/15-forward cohorts, and all four pairs then ran without
failures. The independent audit matched the same timing map, Jaccards and ranks
for all rows.

| Cohort / localizer version | n | Mean Jaccard | Median Jaccard | Mean same-length percentile | Rows with positive overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / unconstrained decoding | 3 | 0.0000 | 0.0000 | 0.6538 | 0/3 |
| 2 / finite JSON grammar | 3 | 0.0000 | 0.0000 | 0.6974 | 0/3 |
| 3 / finite grammar + black/blur | 3 | 0.2963 | 0.2222 | 0.7316 | 3/3 |
| 4 / finite grammar + black/blur | 3 | 0.4722 | 0.5000 | 0.9524 | 3/3 |

This is an outcome-independent ancillary analysis of already completed runs,
but remains exploratory and post hoc relative to the GPU study. The shift from
zero overlap in the first two cohorts to partial overlap in the later two may
reflect changed sample composition, localizer decoding version, QA interval
geometry, or chance; the design cannot separate these explanations. The
same-length percentile is only a deterministic rank among 1–4-frame contiguous
spans, not a significance test or null probability. In particular, when
Jaccard is zero its inclusive upper-tail fraction is 1 because all scores are
nonnegative; this must not be interpreted as strong alignment. QA timing is
question-level and does not establish claim-specific evidence. Direct
faithfulness remains 0/3; no human review was performed.
