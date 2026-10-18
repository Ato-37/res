# QTA v4 query-to-score position integration check

## Purpose

Close one human-free engineering gap before considering later QTA scoring: verify
that the v4 row runner applies its key intervention on the final prompt query,
that only the intended decoder layers and query row change, and that the resulting
four branch logits reach the A–E next-token margin adapter with the same prompt
length. This is a synthetic CPU wiring test, not a model experiment or a
faithfulness result.

## Setup and provenance

- Date: 2026-10-18 (Asia/Tokyo); repository HEAD before this local test/log was
  `aaef12a7216ae71d1ae7d27f109259e95557bd62`.
- Test: `scripts/test_qta_v4_query_score_position_integration.py`, SHA-256
  `dfdf8cc7359f4cca7577707ca9cc8ca56e69a83e157f47283e3677a14d462bb6`.
- Production runner: `scripts/run_nextgqa_qta_v4_union_scorer.py`, SHA-256
  `9bbc08b8f2bc98bf5e02ed1b0035691f75d9fd161566e4823f801027880e0ef8`.
- Forward path: `scripts/run_nextgqa_validation_qta_annotation_proxy.py`,
  SHA-256 `0039e66f063a08c81a44eb16097e45d60b7a5adee53c278781ee9f925d485063`.
- Real hook helpers: `scripts/qwen_internal_attention_ablation.py`, SHA-256
  `a96175d796ee4cfab0ffc40dbf8666d7313bf72a6d4c50555222aa4b94ab4109`, and
  `scripts/qwen_visual_attention_boost.py`, SHA-256
  `f162b6e30af7f77b552db5c8deb3f1d8c7db8049f6f346d077cf55c804fe7b41`.
- Score adapter: `scripts/fixed_annotation_union_dependence.py`, SHA-256
  `90f680f6853c37022d6c6c4bb78b2dab5a0dc2cffd85b28ca7b4bfc7202554b5`.
- Synthetic prompt length 7; two one-token temporal groups; four production
  `score_row` conditions; one forward requested/attempted/completed per branch.
  Decoder output and geometry preparation were fixture-injected. The real
  QTA-v4 scorer, `_forward_score`, attention pre-hook composition, baseline
  attention capture/selection, and A–E score adapter ran. CUDA capacity checks
  were mocked; no CUDA, model weights, video, dataset row, or protected review
  artifact was loaded or read.

## Verified results

Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_qta_v4_query_score_position_integration \
  test_run_nextgqa_qta_v4_union_scorer \
  test_fixed_annotation_union_dependence \
  test_qwen_internal_attention_ablation \
  test_qwen_visual_attention_boost -v
```

All **20/20** relevant tests passed. In the new integration fixture:

- all four branches received the same prompt object, used `use_cache=False`, and
  requested `logits_to_keep=1`;
- actual hook installation used query position `6 = prompt_length - 1`;
- only the configured intervention layers changed; at those layers, only the
  final query row changed, with the expected `ln(2)` bias and/or `-inf` evidence
  key mask; all non-target layers and all other rows remained equal to the
  original synthetic zero mask;
- the actual scorer selected the intended temporal group and the real A–E
  adapter returned the expected condition margins: `3`, `2`, `3+ln(2)`, and
  `2+ln(2)` nats. Both synthetic mask drops were `1` nat and their difference
  was `0` nat, as dictated by the fixture.

The final test also passed `py_compile` and `git diff --check`.

## Interpretation and limits

This closes a Python-level wiring gap: under the synthetic attention module and
seven-token prompt, the production row runner sends the intervention to the
same final query position whose next-token A–E logits are scored. The fixture's
logits are deliberately generated from the actual post-hook mask, so it checks
the mask-to-score path rather than merely asserting mocked call arguments.

This does not validate Qwen's real attention implementation, the video
processor, temporal mapping on a real clip, the semantic validity of annotation
intervals, reviewer correspondence, or any faithfulness construct. No real
faithfulness score, effect estimate, CI, inferential test, or GPU result was
produced. Existing QTA v4 media/mask measurements remain annotation-mask
prediction-dependence proxies, not direct generated-claim evidence.

## Faithfulness progress and next action

Direct generated-claim evaluation remains **0/3** human-reviewed claims,
**0/15** independently verified player-clock/decoder-PTS frame pairs,
**0/3** prediction-link judgments, and **0/3** claim-linked scores. No real
`ΔF`, uncertainty interval, or test exists. This work is engineering-only.
The review-opening branch still needs an independently pinned trust root,
actual distinct role/key assignments, signed review-only rosters/receipt, and a
witnessed unopened handoff. The next human-free work must address a distinct
known gap without repeating the pHash or QTA v4 outcome analyses; direct
scoring remains gated on human review and source/timing evidence.
