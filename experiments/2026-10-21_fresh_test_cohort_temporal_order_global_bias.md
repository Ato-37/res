# Outcome-blind test-cohort check: global visual-key bias × temporal order (2026-10-21)

## Prespecified question and estimand

On three test-split videos selected without new cohort scores, does the fixed
global visual-key attention bias change the conditional likelihood sensitivity
of a generated claim to natural versus temporally block-reversed video order?
This is a finite-cohort exploratory replication of an automated claim-likelihood
proxy, not a direct faithfulness experiment. The claim-to-video and
claim-to-frame relationship will remain unreviewed.

For each row, with a saved claim held fixed, calculate
`DID = (LL(natural,bias) − LL(reversed,bias)) −
(LL(natural,baseline) − LL(reversed,baseline))` in nat/token. The only
primary aggregate is the mean of all three locked row values if all three
claims are structurally valid and all 12 scorer forwards complete with matching
source bindings. Also report the median, range, row values, and failures. No
CI, p-value, threshold tuning, row replacement, survivor-only mean, or
population inference. Positive values indicate only increased sensitivity to
this specific temporal-order transform under a global visual-key bias.

## Frozen cohort, target and intervention

- Lock: `data/private_nextgqa/temporal_order_claim_cohort_20261021_05.json`,
  SHA-256 `4b5d51335927a3646c7683d39f8a1adfd7d9ebe6b2cfb5e7279dba39fc0899c6`.
- Pair allowlist: `data/private_nextgqa/temporal_order_claim_pair_filter_20261021_05.json`,
  SHA-256 is recorded by the lock as
  `fd04a478e9ac10533c12fdbe597ba0b7ad7b30cf35bc3da8d72b1df3627d5162`.
- Frozen before generation: three distinct test-split media/QID assignments,
  with 82 candidates scanned; 3/3 decode and timing passes; 0 known overlap
  against the current 125-media local claim/scoring digest set and the
  117-media, 240-source-file exposure-union manifest 30. This is known local
  artifact non-overlap only; model pretraining exposure is unknown.
- Generate one claim per frozen row using the existing v1 prompt, 16 frames,
  deterministic decoding, pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; v1 matches the scorer's pinned
  prompt suffix. No generation outcome is used to select or replace rows.
- Compare natural video with the deterministic reversal of temporal-patch-sized
  frame blocks, preserving within-block frame order. Baseline and fixed
  `+ln(2)` visual-key bias at layers 5–16. Four forwards per row; the bias is
  global over visual keys, not claim-support-key steering.

## Interpretation boundaries

Temporal reversal can be out of distribution; the global intervention is not
claim-specific; claims are generated and scored by the same model; historical
exposure is incomplete; the cohort is small and chosen after earlier exploratory
results. A positive result would be an internal intervention signal for
temporal-order sensitivity, not evidence that the claim is true, that the
relevant interval supports it, or that semantic faithfulness improved. Rights
remain unverified; all protected input and detailed output stay local.

## Execution plan

Run focused CPU contract tests and verify exact lock/packet/pair hashes before
model loading. Generate all three claims on the pinned local GPU. Score locked
row 0 first (four forwards) as a smoke; continue to rows 1–2 only if claim,
prompt, frame/timestamp, hook, and source checks pass. Expected scoring cost is
12 teacher-forced forwards across three model loads, around 3–6 minutes from
prior runs, peak allocated VRAM below 12 GiB; generation adds one model load.
Record actual runtime and resource use. Preserve errors and incomplete rows.

## Verified results

The independent lock audit and a second local recomputation confirmed 3 unique
media/pair rows, source and exposure manifest hash bindings, zero overlap with
the union of 125 locally found prior claim/scoring media, all three lock-to-
generation row joins, and 3/3 timing/decode passes. The audit found that the
lock does not store a `pair_sha256` beside each per-row assignment or the
video-root path; therefore row pairing can only be re-established through the
generation packet's matching ordered records and the frozen allowlist. The
lock and generation packet were joined row by row before scoring; all three
matched media hash, QID hash, and pair hash. This is a provenance limitation,
not a row mismatch.

Claim generation completed **3/3 structured claims**, 16 frames per row,
using the local RTX 5070 Ti, pinned revision, v1 prompt and 4-bit/BF16. It took
18.16 seconds after model setup, with peak allocated VRAM 7.48 GiB. The output
packet has mode `0600`, SHA-256
`fb9518a3427a3a6a8a1851b1b8368d90df6ee6ed498ebb10af472c2f3a2b305d`.
No row was replaced. The scoring runner accepted all three rows and the v1
prompt binding before GPU execution.

All locked rows completed **12/12 scorer forwards**, zero failed or excluded
rows. Pinned model revision, RTX 5070 Ti, 4-bit weights/BF16 compute and eager
attention were used; per-row peak allocated VRAM was 7.38, 7.23 and 7.36 GiB.
Scoring runtime was 21.74, 22.96 and 29.39 seconds, excluding each model-load
overhead. Independent arithmetic recomputation from the four branch token-log
likelihood means per receipt produced:

| Frozen row ordinal | Bias × natural-vs-reversed interaction (nat/token) |
|---:|---:|
| 0 | −0.022759 |
| 1 | +0.008539 |
| 2 | −0.016455 |
| **Mean** | **−0.010225** |

Median is −0.016455; range [−0.022759,+0.008539]; 1/3 rows are positive. No
CI, hypothesis test or population inference was computed. This does not
reproduce a positive average interaction as a robust effect; it is a small,
mixed exploratory finite-cohort result. The endpoint remains a proxy for
conditional likelihood sensitivity to temporal reversal under a global bias.
It does not test claim truth, claim-support interval validity, support-specific
steering, hallucination, or direct faithfulness. Human reviews remain 0/3 and
direct faithfulness remains 0/3; the user's waiver permitted this local run but
did not create labels. Rights remain unverified.

Private receipt SHA-256:

- Lock: `2732335cd03131a0a83709f569fdc2aff12e6f488558d2c9e29b902c1254a183`
- Pair filter: `fd04a478e9ac10533c12fdbe597ba0b7ad7b30cf35bc3da8d72b1df3627d5162`
- Row 0: `001a9e1ec5c63528985303689adc9859b2aeb4a86c5a05fb1e3cb80cb95f862e`
- Row 1: `bcb51566efa4b3e23626adfe8b28c124c321345f1d63f3315c1fdf7469b598bc`
- Row 2: `0308221b577a9c06ae6b71136b6595cc29de3522e39aeb15ce9ffa613b0cda49`

The output packets and row receipts remain private (`0600`), with no raw media
or text published. An independent audit recomputed all four branch means and
all three interactions; it confirmed 12/12 forwards, no failures, unique
videos, exact packet/lock/pair/video/claim/source bindings, 12-layer hook
counts, and matching pinned checkpoint configuration. Its values matched the
primary recomputation. The auditor also highlighted that this result is
distinct from the earlier validation cohort and must not be pooled with it.

Added `scripts/analyze_temporal_order_global_bias_cohort.py` to recheck the
private packet, lock, pair filter, per-row assignments, receipt hashes and DID
arithmetic without printing raw question/claim/media data. An initial independent
code review found that the first analyzer could accept path traversal and
unrelated source maps, so its provenance flag was not trusted. The analyzer now
rejects private-path traversal and symlink parents, requires the exact frozen
source set and current source hashes, and checks lock source bindings. A second
independent review found directory-mode, replacement-race, and malformed-nested-
schema gaps. A further review found source-path replacement, duplicate JSON key,
and non-finite-constant cases. The analyzer now opens private and source paths
by directory descriptors with `O_NOFOLLOW`, requires owner-only private
directories/files, rejects malformed nested rows, duplicate JSON keys and
non-finite constants, and emits only stable error codes. The expanded suite
passes **10/10**, including these negative cases; reanalysis of the real packet
reproduced the table with source hashes and row/pair bindings verified. Python
compilation and `git diff --check` passed. No confirmatory claim is made.

## Interpretation and next action

The previously observed v2_short validation/temporal-order mean
(+0.038378 nat/token, 2/3 positive) used the wrong v1 prompt suffix and is
invalid for that condition. Its corrected, outcome-exposed rerun is
+0.021637 (3/3 positive); the new outcome-blind-within-pool test cohort is a
separate v1 cohort at −0.010225 (1/3 positive). Do not pool them. This still
does not establish a reliable improvement: it is a small proxy result on
different prompt versions, and the corrected validation estimate is
post-selection repair rather than independent confirmation.

Do not repeat this closed three-row cohort. A useful next implementation is a
claim-token-conditioned temporal-group selector with equal-budget deterministic
sham and an explicitly machine-proposed, unreviewed target. Its primary remains
an automated proxy unless later supported by actual claim/frame judgments; if
the available candidate pool is exhausted or the estimator fails its CPU
contract, stop that experiment rather than substitute rows.
