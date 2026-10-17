# QTA checkpoint hash smoke and construct-link audit (2026-10-17)

## Purpose

Close the checkpoint-byte provenance gap in the synthetic internal-score-path
smoke, and respond to an independent construct audit asking whether the human
reviewed claim actually explains the answer prediction used in the faithfulness
score. No real-data branch was run.

## Setup and verified result

The existing synthetic one-record QTA-Bias/internal visual-key-ablation path
was rerun with five local Qwen2.5-VL-7B-Instruct checkpoint shards hashed by
streaming SHA-256 before model load. Pinned revision:
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. All five computed digests equal
the content-addressed Hugging Face cache blob names. The run completed 1/1
synthetic record and 9/9 forwards in 47.239 seconds on an RTX 5070 Ti,
bf16/4-bit with bf16 compute; peak allocated/reserved VRAM was 10.7549/12.4395
GiB. The private run artifact SHA-256 is
`7dd89c84d799b49e1749255d53f1ffa35e4c07c8f60d014926893dca7fa69933`.

The synthetic target and sham `ΔF` values were each approximately
`−5.96e−8` nat, with a zero difference. They are floating-point residue on
synthetic groups and have no scientific interpretation. The sanitized,
content-hash summary is
[`results/synthetic_qta_checkpoint_hash_smoke_20261018.json`](../results/synthetic_qta_checkpoint_hash_smoke_20261018.json), SHA-256
`9e2e29f9d146a639a9e0b7383d112adbf78270040d8abf295324740f1f4d6cdf`.

The evidence-to-token mapper also received a CPU regression test for an
interval whose legacy contiguous control does not exist but whose union-level
rank-distance control does. The multi-occurrence mapping path now maps each
occurrence without prematurely requiring the legacy control, then applies the
protocol's union-level rank-distance control. The focused mapper suite passed
12/12 tests and the focused QTA composition suite passed 6/6. The final
repository suite passed 1,479/1,479 after the dirfd output hardening and
prediction-link validator changes; Python compilation and `git diff --check`
also passed.

An independent protocol auditor found a construct-link gap: the reviewed
answer-like claim was not bound to the fixed baseline `p/q` predictions later
scored. The draft now requires a post-baseline, pre-treatment adjudication that
the reviewed claim supplies a video-grounded reason for choosing `p` over `q`.
It also harmonizes interval endpoint wording with review policy v6. The
current review packet does not contain the supplemental judgment, and no
real-data score is authorized by this engineering work.

To make this future gate executable, `scripts/nextgqa_prediction_link_review.py`
now hashes and parses the exact baseline-manifest bytes, verifies the expected
cohort token sequence and externally pinned question/options/claim digests,
recomputes `p/q` from the five baseline A–E log probabilities, and binds the
receipt to the displayed question/options/claim/p/q digest and protocol hash.
It fails closed on missing, extra, reordered, or changed rows; its result only
reports the count of supporting self-reported judgments and whether every
expected row has one. Six CPU tests pass. The caller must source expected
hashes/tokens independently;
the validator does not authenticate that source, render a reviewer packet,
authenticate reviewer identity, verify semantic correctness, or authorize scoring.

The independent code audit also found that the smoke output was temporarily
world-readable during write and could chmod a shared parent. The runner now
opens the verified `data/private_nextgqa` root and a fresh run directory via
directory descriptors, creates the run directory as 0700, and uses exclusive
0600 file creation with cleanup of partial output. Seven focused tests cover
modes, overwrite, symlink/outside paths, shared-directory protection, open
failure, and serialization cleanup. This change affects local result handling
only; no protected review or score output was generated.

## Faithfulness progress and limits

Direct real-data faithfulness progress remains **0/3 reviews, 0/3 prediction-link
adjudications, 0/3 scored, and 0/15 verified player-to-decoder correspondences**.
There is no real `ΔF`, confidence interval, or hypothesis test. The rerun
improves checkpoint reproducibility and synthetic execution coverage only; it
does not improve empirical faithfulness evidence. The reviewed-claim mapping
and predictor-link construct gate are prerequisites, not evidence of an effect.
Rights status remains unverified; protected inputs and the private artifact
remain local.

## Next action

Independently audit the new receipt validator and implement a separate human
packet renderer. Then bind their results and all source/release receipts into
fail-closed cohort preflight. Keep treatment branches unexecuted until review,
clock, prediction-link, and release-order gates pass.
