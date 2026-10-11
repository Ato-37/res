# Primary faithfulness estimand: CPU design review (2026-10-08)

> **Post-confirmation update:** The planned 79-video independent confirmation has since completed (79/79, 0 failures). Its primary estimate was `−0.0756 nat` (SD `0.3667`, 95% t CI `[−0.1578,+0.0065]`), below the preregistered `+0.25 nat` success threshold. The recommendation below to run confirmation is historical and has been superseded: close this particular head-faithfulness hypothesis; do not launch additional head/layer/dose searches on this test set. See the [confirmation result log](2026-10-08_head_faithfulness_confirmation.md) and [result interpretation](../docs/head_faithfulness_confirmation_results.md).

## Decision

Keep the already-approved head-group estimand unchanged and move toward one independent, preregistered test-set confirmation. Do not run further layer/head searches or adjust K, alpha, controls, or score using the validation pilot. The cumulative layer-randomization result is a completed sanity check, not a reason to switch the estimand.

## Locked estimand

For each unique, previously unused NExT-GQA test video, let `ŷ` be Qwen2.5-VL's highest-log-probability A–E choice before intervention and hold that choice fixed in every condition. Select the four heads with largest `|z_h · ∂m/∂z_h|`, where `m` is the pre-intervention chosen-answer margin and `z_h` is the final-prompt-token head output. Scale those head outputs by `alpha=0.5`; compare their margin drop with the mean margin drop from five controls matched to the selected set's layer counts, head count, and approximate activation magnitudes. For video `i`, define `d_i = [m_i(full)-m_i(target)] - mean_r[m_i(full)-m_i(control_r)]`, and estimate `theta` as the mean of `d_i` over videos, one QA per video.

The single primary success rule remains: the lower bound of the two-sided 95% t confidence interval for `theta` must exceed `0.25` nat. The selected answer is fixed before intervention; the choice is not changed to the post-intervention winner. `K=4`, `alpha=0.5`, five controls, the scoring rule, and the threshold are not retuned after the pilot. Other doses, correctness strata, question types, and temporal alignment remain secondary; secondary families use Holm correction. The confirmatory video set must be independent of development data and exclude the three test-side engineering-smoke videos and all previously used sample IDs.

## Evidence considered and decision rationale

- The 30-video validation-development pilot completed 30/30 with no failures. Its mean selective margin drop was `−0.0225` nat (SD `0.3774`, 95% t interval `[−0.1634, +0.1184]`); the interval lies below the predeclared practical threshold `+0.25` nat. This is not a confirmatory result and does not prove a zero effect, but it makes the current method's expected practical effect doubtful.
- The three-video cumulative decoder randomization completed 3/3. After all 28 blocks were reset, mean head-rank Spearman correlation was `−0.286`, top-4 overlap was `0/4` in every video, and the selected answer changed in all three. This shows sensitivity to a severe network-destruction perturbation; it does not show that the selected heads are faithful.
- Earlier head/group patch screens did not identify a stable head group. They are not used to select a new subgroup or replace the primary outcome.

Given the original plan's explicit separation of validation pilot from independent test, the protocol-consistent next step is one locked independent test rather than another exploratory GPU search. A null or imprecise confirmation must be reported as such; there will be no post-hoc rescue by changing the method or reusing the same test examples.

## Operational gate before GPU inference

Implemented `scripts/run_head_faithfulness_confirmation.py`. It refuses existing output names, checks the repository is ready, builds disjoint raw manifests outside Git using all `data/nextgqa_samples*.json` plus the earlier test smoke as exclusions, runs a 3/3 smoke, and stops before confirmation if the smoke has any failure. If smoke passes, it builds 79 fresh videos excluding the smoke and runs the locked confirmation. The publisher commits only anonymous derived results and manifests. It does not replace failed cases after inference; it reports completion only when at least 71/79 videos succeeded.

Ubuntu command (not yet run):

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_head_faithfulness_confirmation.py \
  --tag 20261009 \
  --seed 20261009 \
  --exclude-manifest ~/nextgqa_sample/faithfulness_smoke_samples_20261008.json
```

Budget roughly 15–25 minutes for two model loads, a 3-video smoke, and up to 79 test videos, using the previous 30-video pilot's approximately 250-second sample-loop runtime as a rough linear reference. This is only an estimate; the runner records actual timings. Raw question/video paths remain in `~/nextgqa_sample/`, never in the repository. Afterward, inspect the published result and manifest on GitHub before interpreting any effect.

No new GPU job was run in this CPU review. The 71-video confirmation is potentially long-running and should be estimated from the new smoke before launch. Detailed definitions remain in [the research plan](../docs/faithfulness_research_plan.md); current evidence and caveats are in [the implementation log](2026-10-08_head_faithfulness_implementation.md).
