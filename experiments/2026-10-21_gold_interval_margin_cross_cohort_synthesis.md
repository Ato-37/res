# Cross-cohort check of benchmark gold-interval QA-margin sensitivity (2026-10-21)

## Question

After a three-video validation pilot and a contradictory three-video test pilot,
a separate outcome-blind ten-video test cohort was scored under the same
benchmark QA gold-interval versus equal-count control masking design. This
local synthesis independently re-joins the three completed receipts to assess
whether this auxiliary endpoint is stable enough to keep expanding.

## Recomputed finite-cohort results

| Fixed cohort | Videos | Black control − gold-interval margin mean (nat) | Sample SD | Full-video accuracy |
|---|---:|---:|---:|---:|
| Validation pilot | 3 | −0.2500 | not calculated | not retained in current summary |
| Test pilot A | 3 | +0.5833 | 1.1342 | 0/3 |
| Test pilot B | 10 | −0.2625 | 1.2183 | 8/10 |

The 16 underlying row contrasts were independently re-read from private
receipts and recomputed: arithmetic pooled mean −0.1016 nat, sample SD 1.1799,
range [−3.2500,+1.8750], with 6 positive, 1 exact zero, and 9 negative rows.
This pooled value is only a descriptive sum over three deterministic cohorts;
it is not a population estimate, inferential CI, or hypothesis test. The two
test-split cohorts are disjoint by exact known-history path/media exclusions.
The three cohort means span 0.8458 nat from lowest to highest, illustrating
substantial between-cohort sensitivity relative to their small average
contrasts. The ten-video blur sensitivity mean is −0.0125 nat (SD 0.8589),
also near zero with mixed row signs.

Receipt hashes:

- Validation n=3: `4e8736558c639204fd7a7c34d36343db2dab6ab24c503e8313493492d9a6a0cf`
- Test pilot A n=3: `53d17bd72fee6ac0f92bfcf44388a1baa5372bafe3948995887feb1eb8260767`
- Test pilot B n=10: `7f8026adc2b40ccbd133360ffcb3a0fd0ca45cf0e13495de58d99823a6c394f4`

## Interpretation and decision

The sign and size vary by cohort; the blur-control contrast is close to zero
in the larger test cohort. These data provide no stable evidence that masking
the benchmark's QA interval uniquely reduces the Qwen answer margin more than
an equal-count temporal control. The endpoint is especially limited because
the selected interval belongs to the QA annotation, not necessarily to the
generated explanation, and black/blur masks alter the input distribution.
Therefore, stop expanding this QA-gold-interval endpoint. Further cohorts
would add GPU cost without resolving the thesis's direct claim-to-evidence
question.

This synthesis does not change direct faithfulness counts: current-claim human
review 0/3, clock/frame correspondence 0/15, Stage-2 prediction link 0/3,
direct generated-claim score 0/3. The human-review collection waiver permits
local exploratory work; it does not create missing observations. The next
experiment must use a claim-conditioned evidence anchor and be explicitly
identified as machine-proposed and unreviewed.

## Verification

Primary recomputation joined all 16 row effects from the three mode-0600 local
receipts. An independent audit of the ten-video run confirmed every condition
contrast, aggregate statistic, 50/50 forward, 10/10 completed row, 8/10 full
accuracy, receipt/preflight/lock/source pin, and no execution failure. No raw
question, answer, claim, media path, or identity is present in this synthesis.
