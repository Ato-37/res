# QTA choice-token contract hardening (2026-10-18)

## Purpose

The independent scorer audit found that the validation-QTA kernel duplicated
the A–E continuation-token check and accepted malformed token IDs that the
shared resolver already rejects. The v4 specificity scorer then indexed logits
before checking token-ID type, sign, and uniqueness. This change aligns the
kernel with the shared resolver and makes the scorer reject invalid IDs before
indexing.

## Implementation

`run_nextgqa_validation_qta_annotation_proxy._choice_token_ids` now delegates to
`qta_choice_token_binding.resolve_choice_continuation_ids`, which validates the
prompt prefix, one-token continuation, nonnegative integer IDs, and distinct
A–E tokens. The v4 specificity `_choice_logps` checks for exactly five distinct
nonnegative plain integers before indexing vocabulary logits.

CPU synthetic tests exercise the actual kernel wrapper, malformed negative
token IDs, bool/duplicate/missing/out-of-range scorer IDs, the scorer's
single-position logits interface, and the four-condition adapter's final
prompt position. A multi-position synthetic case also makes the intervention's
top choice differ from the baseline and confirms that the adapter retains the
baseline-full answer/foil for all four conditions.

## Verification

Command:

    PYTHONPATH=scripts .venv/bin/python -m unittest test_run_nextgqa_validation_qta_annotation_proxy.ChoiceTokenKernelTests test_run_nextgqa_qta_v4_mask_specificity.SpecificityRunnerContractTests test_fixed_annotation_union_dependence -v

Result: **13 tests passed**. `py_compile` also passed on the two modified
production scripts and three test files. The first draft of the new position
test supplied full-sequence logits to `_choice_logps`, whose contract receives
the already sliced `[1,1,vocab]` output; the test was corrected to exercise
that interface, while final-position extraction is independently checked at
the four-condition adapter where full-sequence logits are accepted.

An independent code audit reported no blocking arithmetic or position issue;
it independently ran 33 focused tests successfully. It confirmed all four
condition scores use the final sequence position and the p/q pair remains
frozen from baseline-full. The separate complete focused command above ran 42
tests successfully.

SHA-256 of the resulting code and test artifacts:

| File | SHA-256 |
|---|---|
| `scripts/run_nextgqa_validation_qta_annotation_proxy.py` | `0039e66f063a08c81a44eb16097e45d60b7a5adee53c278781ee9f925d485063` |
| `scripts/run_nextgqa_qta_v4_mask_specificity.py` | `4db93fd2beecd1a9747eedbd84b433a304095255d0132f91bafd6d45352bbdcb` |
| `scripts/test_run_nextgqa_validation_qta_annotation_proxy.py` | `36be5abe61e2944fa20c62366e9fc52d0f8e5d13a49f76711fc02b71cd5c6226` |
| `scripts/test_run_nextgqa_qta_v4_mask_specificity.py` | `ce1f68642488836376b29156df8c2e63bed13b2d4bb59e5ef1c7bb18507789cf` |
| `scripts/test_fixed_annotation_union_dependence.py` | `7c34159150f7a1264716dddeca6d8ee3ca0deb1c048f89d125b23f01cbce0651` |

No data files, protected review content, media, checkpoint, CUDA device, or
model forward were used. No faithfulness score was computed.

## Interpretation and limitations

This closes a software validation gap for A–E token continuation and scorer
indexing. It does not establish that the pinned real tokenizer and multimodal
processor agree for any reviewed example, that the videos support claims, or
that QTA improves explanation faithfulness. The earlier mask-specificity run
is a separate exploratory finite-cohort diagnostic; these edits occurred after
that run and do not change its observed result or turn it into faithfulness
evidence. The runner binds tokenizer continuations of the rendered prompt, but
does not compare those IDs against a separately rendered multimodal
processor-output continuation or actual model answer formatting. Therefore the
new checks establish tokenizer-level continuation semantics, not processor-
bound alignment.

## Next action

Complete independent code review and the focused suite. Keep real claim-linked
scoring blocked until the trust-root, role roster, review-only receipt,
witnessed handoff, claim reviews, and verified browser-to-decoder frame pairs
are complete. Continue with independent CPU-only contract checks that do not
reuse the unreviewed cohort's outcomes.
