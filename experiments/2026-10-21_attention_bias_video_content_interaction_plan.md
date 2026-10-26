# Post-result visual-attention-bias × video-content interaction (2026-10-21)

## Research question

The thesis asks whether internal intervention can improve explanation
faithfulness. The fixed claim-text sensitivity checks show positive
original-video versus content-ablated differences, but do not establish
semantic support. The earlier attention-bias × temporal-order interaction was
near zero. A different, explicitly post-result exploratory contrast now asks
whether that same fixed global visual-key bias selectively increases
conditional likelihood under original video versus constant-gray video.

## Frozen estimand and execution

- Fixed cohort: the same three unique validation rows used in the already-run
  video-versus-constant-gray test. This is a post-result follow-up, not a
  confirmatory or independent sample.
- Reuse the exact completed baseline receipts and original-video `+ln(2)`
  visual-key-bias scores from the matched prior attention receipt. Bind every
  join by row-token, video, packet, claim and re-tokenized target-ID hashes,
  pinned model revision, frame/processor geometry, bias layer/key/query rules,
  and matching original/baseline log likelihood.
- Add only the missing condition: constant-gray video plus the same
  `+ln(2)` key bias on all expanded video-token positions in layers 5–16.
  This requires one new GPU forward per row (3 total); do not rerun original
  branches. Same 16 decoded frames/timestamps and fixed prompt/tokenizer.
- Per row, compute
  `(original_biased − gray_biased) − (original_baseline − gray_baseline)`
  in mean target-token log likelihood (nat/token). Positive values indicate
  the bias increases the original-versus-gray contrast under this particular
  diagnostic.
- Report every row plus finite-cohort mean/median/range and forward/failure
  counts. No population confidence interval, hypothesis test, or significance
  claim for this selected n=3 cohort. No row replacement or post-score tuning.

## Interpretation and limitations

This is a post-result mechanism-sensitivity diagnostic using same-model
generated sentences and a severe out-of-distribution gray-video control.
Its two factors are an internal global attention-logit bias and a content
replacement; neither is claim-specific, and a positive interaction would
not mean explanations became more truthful, grounded, or faithful. The
baseline comparisons are reused across prior local receipts, so source and
cross-receipt consistency must be checked before GPU load. Human review is
waived as a collection prerequisite for local exploratory runs only; no review
will be claimed as performed. Rights remain unverified; everything remains
local.

## Gate and next action

Implement and CPU-test strict cross-receipt joins and the interaction formula.
Independently audit the protocol and input bindings. Run one fixed-row
one-forward GPU smoke; only proceed to two remaining rows if it passes all
hook/query/key and receipt checks. If an existing baseline does not exactly
join to its corresponding original-video baseline receipt, stop this branch
without substituting or recapturing baselines.

## Verified finite-cohort result (2026-10-21)

The three successful one-forward receipts were selected by frozen row token:
`attention_bias_content_interaction_20261021_smoke_retry_01.json`,
`attention_bias_content_interaction_20261021_remaining_00_retry_01.json`,
and `attention_bias_content_interaction_20261021_remaining_01.json`. The initial
smoke receipt failed after its forward due to a missing post-run hash helper;
the first remaining-00 attempt failed before a forward because it joined the
wrong frozen baseline. Neither failed receipt contributes an estimate. The
corrected receipts completed one new forward per row, and the analyzer
recomputed all three contrasts from the exact saved baseline and treatment
scores. Analyzer and runner regressions passed **6/6** using `unittest`.

| Row | Baseline original−gray | Biased original−gray | Difference-in-differences |
|---|---:|---:|---:|
| smoke | +0.645801 | +0.595798 | −0.050003 |
| remaining 00 | +0.392772 | +0.426939 | +0.034167 |
| remaining 01 | +1.629914 | +1.634302 | +0.004388 |
| **mean** | **+0.889496** | **+0.885680** | **−0.003816** |

The interaction median is +0.004388 nat/token, range [−0.050003,+0.034167],
with 2/3 positive rows. This finite three-row result is near zero and mixed;
it gives no evidence that the tested global visual-key bias increases claim
likelihood's dependence on original visual content. The baseline original−gray
contrast is reused from the earlier content-dependence experiment, so this is
not an independent replication. No CI, hypothesis test, or population claim is
appropriate. This endpoint still does not measure semantic claim support or
direct faithfulness. Human review collection was waived for exploratory local
work; actual human review remains unperformed. Rights status remains
unverified and all receipts stay local.

Receipt SHA-256 values: smoke retry `14b8895c216deb8a3f72f15e8be5780a7587f6f8ead0bcc924b8e2c0e33d1087`,
remaining 00 retry `06b18df4810a24fb82a50d317c1857fc1b321d87605b804463eefe49c14402f9`,
remaining 01 `018581d98170d9d31198328320f0f1a61050a92bf1db1bb2aa99bee941b2a249`.
Reproduction command:

```bash
.venv/bin/python scripts/analyze_attention_bias_content_interaction.py \
  data/private_nextgqa/attention_bias_content_interaction_20261021_smoke_retry_01.json \
  data/private_nextgqa/attention_bias_content_interaction_20261021_remaining_00_retry_01.json \
  data/private_nextgqa/attention_bias_content_interaction_20261021_remaining_01.json
```

**Next action:** close this selected-cohort interaction branch. Do not add more
conditions or repeat these rows. Resume outcome-blind cohort construction and
measurement design for a fresh exploratory sample; do not call test candidates
fresh while known exposure history is incomplete.
