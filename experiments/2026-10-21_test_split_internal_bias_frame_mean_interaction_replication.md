# Exploratory cross-cohort test: global visual-key bias × frame-mean sensitivity (2026-10-21)

## Question and status

Does a global visual-key bias make a model-generated video explanation claim more dependent on intact spatial video structure? This is a post-result exploratory cross-cohort replication of the earlier three-row frame-mean interaction, not a confirmatory test. The control replaces each of 16 sampled frames with that frame's per-channel RGB mean. This is a severe out-of-distribution intervention that also removes spatial layout and other image statistics.

**Result status:** 3 rows were frozen before generation; 2 generated claims passed the fixed sentence-shape check, 1 failed. The failed row was retained, not replaced. Both valid claims received four fresh teacher-forced scores. Because one locked row failed, no full-cohort mean, confidence interval, p-value, or population estimate is reported.

## Cohort and provenance

- Dataset: local NExT-GQA test split and local `NExTVideo`; 16 uniformly sampled frames, one question/video, existing QA interval and equal-count temporal-control eligibility rule.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, pinned revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights, BF16 compute, eager attention; seed 20261021.
- Outcome-blind lock: three distinct local media hashes, selected after scanning 79 candidate groups; 76 known-hash exclusions, three decode/timing/control passes, zero timing/decode/control failures among selected rows. Lock and filters are private mode 0600.
- Local exposure heuristic: 321 recognized generation/scoring JSONs and 122 distinct media hashes; the three locked media did not overlap that included set. History completeness is unauthenticated, pretrained-model exposure cannot be excluded, the split is the same test split, and selection followed the earlier observed result. Therefore this is not a fresh, independent, or confirmatory cohort.
- The generation packets contain protected question/video/claim material and remain local. No raw identifiers, claim text, paths, or packet/receipt hashes are included in this public-safe log. Rights remain unverified; no protected data or receipt was published.

## Intervention and endpoint

For each fixed claim, measure mean conditional log-probability of the same re-tokenized claim tokens in four conditions: intact video and per-frame-mean video, each without bias and with `+ln(2)` bias to all expanded visual keys at layers 5–16 for claim-token prediction rows. The row statistic is

`(LL(intact,bias) − LL(mean,bias)) − (LL(intact,baseline) − LL(mean,baseline))`.

Positive values mean that this global bias increased claim-token likelihood sensitivity to the intact-video versus frame-mean contrast on that row. This does not test claim truth, claim-specific support, temporal evidence localization, or explanation faithfulness. No human claim review was performed.

## Observed outcome

| Frozen ordinal | Generation | Scoring | Difference-in-differences (nat/token) |
|---:|---|---:|---:|
| 0 | valid | 4/4 forwards | −0.00904447 |
| 1 | invalid sentence shape | 0/4 forwards | not measured |
| 2 | valid | 4/4 forwards | +0.04469058 |

The two measurable rows have opposite signs. They are shown rowwise; no survivor-only mean or inferential test is used. The earlier exploratory three-row cohort's mean was +0.03927387 nat/token, so the new outcomes do not reproduce a uniform positive direction across rows/cohorts. These finite-sample observations do not establish that the intervention helps or harms faithfulness.

The failed row's `claim_sentence_shape_invalid` outcome is a generation-format failure, not a negative score. It was not retried or replaced. Two earlier runner wiring attempts were stopped before any forward (0 completed), and subsequent fixes exposed a hook-key mismatch (1 completed forward) and then a receipt-serialization bug (4 completed forwards, no valid result receipt). All failure receipts were preserved locally; the usable row-0 result was rerun under the corrected, source-hashed runner. The final usable scoring receipts verify 4/4 forwards, finite scores, baseline hook counts 0 and biased hook counts 1 at each of 12 layers. Peak allocated VRAM was 7.27 GiB (ordinal 0) and 7.26 GiB (ordinal 2); measured scoring-run times were 23.0 s and 31.9 s.

## Faithfulness-progress assessment

- **Direct faithfulness:** `未実施`, 0/3 claim-support judgments, 0/3 claim-specific fidelity scores. The measured endpoint is conditional claim-token likelihood sensitivity to a synthetic frame-mean control, not a direct faithfulness metric.
- **Auxiliary/internal-intervention evidence:** 2/3 locked rows scored; one fixed row failed generation format. Row effects are −0.00904447 and +0.04469058 nat/token. No CI, p-value, or multiplicity-adjusted result is applicable to this incomplete post-result exploratory cohort.
- **Interpretation:** mixed rowwise direction weakens the proposed broad global-bias mechanism. The earlier positive three-row proxy and this mixed result can motivate a thesis on the limits of indiscriminate visual-key steering, but not a claim that explanations became more faithful.
- **Unresolved:** semantic claim-to-video support, independently reviewed evidence intervals, incomplete exposure history, and rights verification. Local-only use was user-authorized; redistribution remains out of scope.

## Engineering validation and failures

The dedicated runner checks the frozen test-split lock, full selected-pair filter order, batch-specific generation pair-filter hash/digest/count, packet row order, exact claim prompt version (`v2_short`), source media/QID binding, all four condition forwards, finite target scores, expected layer-hook counts, and private exclusive receipt writes. The analyzer independently revalidates source/packet/receipt hashes and difference-in-differences arithmetic and refuses to report a complete cohort estimate when any locked row failed.

- Runner and prior-interaction focused tests: 23/23 passed; Python compilation and scoped `git diff --check` passed.
- Cohort summary tests: 2/2 passed. The real local lock was independently re-frozen and all three assignment/filter joins reproduced.
- Exact source SHA-256: runner `1702d3b02fcf9b298642165633355a29353533b9327e09f6189bbb6baf07fe12`; analyzer `5f687de276137b86e374e52d29fba6d0adafbe6539a5c0cdaeda5990f47b0b81`.
- Analysis command (prints hashes and rowwise derived values only; protected files remain local):

```bash
PYTHONPATH=scripts .venv/bin/python scripts/analyze_frozen_frame_mean_attention_bias_interaction.py \
  --lock data/private_nextgqa/internal_bias_replication_lock_20261021_01.json \
  --pair-filter data/private_nextgqa/internal_bias_replication_pairs_20261021_01.json \
  --smoke-packet data/private_nextgqa/internal_bias_replication_claim_smoke_20261021_01.json \
  --smoke-pair-filter data/private_nextgqa/internal_bias_replication_smoke_pair_20261021_01.json \
  --remaining-packet data/private_nextgqa/internal_bias_replication_claim_remaining_20261021_01.json \
  --remaining-pair-filter data/private_nextgqa/internal_bias_replication_remaining_pairs_20261021_01.json \
  --row0-receipt data/private_nextgqa/internal_bias_replication_interaction_row0_retry03_20261021_01.json \
  --row1-receipt data/private_nextgqa/internal_bias_replication_interaction_row1_invalid_final02_20261021_01.json \
  --row2-receipt data/private_nextgqa/internal_bias_replication_interaction_row2_20261021_01.json
```

## Next action

Do not tune layer or bias strength on these rows. Revise the mechanism question: test whether steering to benchmark temporal-evidence keys is more selective than an equal-budget matched-sham key set, with a matched evidence-ablation margin and a new outcome-unseen cohort. Treat NExT-GQA intervals as QA evidence proxies, not claim-specific support. Before any direct faithfulness claim, obtain genuine independent claim-to-frame support judgments; informal review impressions are insufficient. If no new eligible local cohort exists, develop/test the protocol on synthetic fixtures and record the data blocker rather than reusing these rows as confirmation.
