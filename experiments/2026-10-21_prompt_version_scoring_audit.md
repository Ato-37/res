# Prompt-version scoring audit and repair (2026-10-21)

## Finding

An audit of the two completed interaction branches found that both scoring
runners imported the legacy `PROMPT_SUFFIX` (v1) directly, while their frozen
three-row claim cohort is logged as `v2_short`. The QA-interval × attention-bias
branch has three completed receipts (3 rows, 12 forwards). The content ×
attention-bias branch has three completed treatment receipts (3 rows, 3 new
forwards) and reuses baseline scores from a separate receipt chain. Two failed
attempts contribute no score. Receipt source hashes and runner code corroborate
the legacy constant; the receipts did not record prompt version or suffix hash.

## Consequence

Do not treat the finite-cohort estimates `−0.018260 nat/token` and
`−0.003816 nat/token` as valid measurements for the logged `v2_short` scoring
condition. In the second branch, the new gray+bias score also cannot be
subtracted from reused baselines as a prompt-matched interaction. Preserve the
receipts as historical wrong-prompt diagnostics; do not overwrite them or
retroactively call them faithful/unfaithful. The source prompt defect does not
invalidate claim generation. Earlier original-versus-gray sensitivity results
were produced by a separate runner with recorded-version resolution; they
remain a content-sensitivity diagnostic only, not direct faithfulness.

The earlier qualitative user review is tied by source-text hashes to a v10
candidate lineage, but its JSON has no artifact fingerprint. It does not bind to
the current artifact or furnish claim/frame support labels, and the associated
diagnostic receipts had zero forwards and score calls. Formal direct-review and
direct-faithfulness counts therefore remain 0/3.

## Repair and verification

Both interaction runners now resolve `generation_prompt_version` through
`PROMPT_SUFFIX_BY_VERSION`, reject unknown versions, and write the resolved
version plus suffix SHA-256 into new receipts. A unit regression checks v2_short
resolution and unknown-version rejection. Four relevant runner/analyzer test
modules passed: **14/14 tests**. No GPU rerun was made: the estimates are
post-result diagnostics on a selected n=3 cohort, and repeating these endpoints
would not establish claim support or direct faithfulness. No GPU experiment is
currently justified on this cohort.

## Limits and next action

Current direct faithfulness evidence is still **0 reviewed claims / 0 direct
intervention scores**. QA intervals are question-level annotations and cannot
stand in for claim-specific evidence. Rights status remains unverified; data
and derived receipts remain local. Next, ensure every claim-likelihood runner
records and uses its row's prompt version, then prioritize an outcome-blind,
independently reviewed claim-to-frame target and a fresh cohort before another
GPU efficacy run. Do not reopen the closed head/layer/alpha search.
