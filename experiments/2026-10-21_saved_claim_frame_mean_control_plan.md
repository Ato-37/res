# Exploratory frame-mean visual-structure control (2026-10-21)

## Why this follow-up

The prior fixed-three-row original-video versus constant-gray diagnostic had
positive differences for all three claims. Because constant-gray replacement
is a severe distribution shift and removes both spatial structure and global
color, a separate follow-up will preserve each sampled frame's per-channel
mean RGB while removing spatial arrangement. This is explicitly a post-result
robustness exploration, not an independent or confirmatory replication.

## Frozen contrast

- Reuse exactly the same three fixed validation rows, original 16 decoded
  frames, source prompt, saved parsed claim strings, tokenizer and pinned model
  revision as the gray-video run; no row replacement.
- For each frame independently, convert to RGB, calculate per-channel
  arithmetic means over all pixels, and map each mean with `floor(mean + 0.5)`
  clipped to `[0,255]`; replace the frame
  with a same-size uniform RGB image at those three values. Keep frame count,
  timestamps, question, choices, target text, processor, and attention settings
  fixed.
- Score exactly two teacher-forced branches per row: original video and
  per-frame RGB-mean video. Primary descriptive contrast is the row-level
  original-minus-mean-video mean target-token log likelihood (nat/token),
  reported with all rows, finite-cohort mean/median/range. No inferential test
  or population interval for this selected n=3 cohort.
- Run one fixed-row smoke first; proceed only if both conditions complete and
  all nonvisual input tensors/geometry/timing match. Then run the two remaining
  fixed rows. Preserve failures; do not substitute rows or modify the transform
  after seeing scores.

The independent design audit approved this as an exploratory follow-up and
emphasized that original-video scores are shared with the gray comparison,
so the two contrasts are correlated and this is not a replication. It also
warned that local variance, edges, texture, and other image statistics change
along with spatial layout. Source receipts will record exact per-frame RGB
means and the deterministic rounding rule.

## Interpretation limits

This is post-hoc exploration selected to assess a specific alternative to the
previous flat-gray result. A positive contrast is consistent with sensitivity
to spatial arrangement beyond each frame's mean RGB. It does not show claim
truth, semantic evidence support, frame localization, causal evidence use, or
faithfulness improvement. Uniform mean-color frames remain out-of-distribution;
low-level statistics other than channel means change. Model-generated saved
claims, deterministic re-tokenization, and three selected validation rows
limit interpretation. Human review remains unperformed and waived only as a
prerequisite for local exploratory work. Rights remain unverified and all
protected material remains local.

## Execution and audited result

Implemented `scripts/run_saved_claim_frame_mean_control.py` and the summary
analyzer `scripts/analyze_saved_claim_frame_mean_control.py`. CPU transform and
analyzer tests passed **4/4**; `py_compile`, CLI help, and scoped whitespace
checks passed. The runner matched prompts, geometry, timestamp metadata, tensor
shapes, all nonvisual tensor values, and fixed visual-key positions before
scoring; it required processor-expanded pixels to differ. Each invocation ran
on the pinned Qwen2.5-VL-7B-Instruct revision in 4-bit/BF16 eager mode and
completed 2/2 forwards. All rows succeeded; no retry, row replacement, or
protocol change was needed.

| Fixed row | Re-tokenized target length | Original minus per-frame RGB mean (nat/token) |
|---|---:|---:|
| Smoke row | 7 | +0.68657609 |
| Remaining row 0 | 13 | +0.36391159 |
| Remaining row 1 | 15 | +1.64926488 |

Finite-cohort mean **+0.89991752 nat/token**, median **+0.68657609**, range
**[+0.36391159, +1.64926488]**; 3/3 positive. These are descriptions only;
no confidence interval, p-value, or population inference is reported.
Reproduce the aggregation with:

```bash
.venv/bin/python scripts/analyze_saved_claim_frame_mean_control.py
```

Primary and independent audit both verified six successful forwards, all
three unique row tokens, source/packet/claim/video hashes, per-row likelihood
arithmetic, all 16×3 rounded frame RGB means, frame/timing geometry, processor
pixel differences, baseline hook counts, and all five source hashes in every
receipt. The actual video bytes matched the frozen packet video hashes. The
receipts are owner-only (`0600`) and contain no raw claim, question ID, or
source path. As with the gray diagnostic, model weights are identified by
pinned revision but individual weight shard hashes are not present in the
receipts. The exact runner source hash is recorded in each receipt.

This comparison shares the original-video branch scores with the earlier
gray-condition comparison, so the two sets of contrasts are correlated. It
is a post-result exploratory sensitivity follow-up, not an independent
replication. It is consistent with the model's claim-token likelihood
depending on local/spatial image structure in addition to framewise mean
colors under this test. It cannot isolate spatial arrangement from every
other altered pixel statistic, and it does not establish semantic evidence
support or faithfulness.

The overall direct-faithfulness result remains **0/3**. Human reviews were
not performed; the user's waiver applies only to proceeding with exploratory
local scoring. Clock/frame correspondence remains **0/15**, prediction-link
reviews **0/3**, and there is no direct `DeltaF`, interval, or confirmatory
test. Dataset/model rights remain unverified; all protected inputs and
derived results remain local.

Next: do not further perturb these same three claims merely to accumulate
positive sensitivity results. The useful branch is to obtain actual review
or independent annotated evidence for a fresh, prespecified cohort, or to
build outcome-blind candidate selection and preregistration for that cohort.
If access to those inputs is unavailable, report the current sensitivity
diagnostics as groundwork and avoid converting them into an efficacy claim.
