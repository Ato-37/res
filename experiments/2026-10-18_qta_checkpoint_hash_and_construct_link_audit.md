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

## Offline prediction-link review packet

The function-only renderer was not operational for a reviewer, so it now has a
CLI that reads an exact manifest file and creates an exclusive new HTML file
directly inside a mode-0700 private root. It opens the directory and children
with directory-relative, no-follow operations; refuses non-regular or oversized
input, refuses overwrite, writes output as mode 0600, and removes a partial
output on write failure. The page is static, uses a restrictive CSP with no
network/image/media access, HTML-escapes source text, and exports only the
self-reported receipt with `execution_authorization=not_granted`. The browser's
download flow for that receipt still needs an operator to verify and secure its
destination; the page itself does not guarantee that downloaded receipt's mode.

Five synthetic CPU regression tests passed: hostile markup escaping and frozen
P/Q display, hash/schema/prediction fail-closed behavior, private output mode and
exclusive creation, symlink/unsafe-root rejection, and partial-write cleanup.
The renderer also passed `node --check` on its emitted inline JavaScript. Full
repository tests passed 1,484/1,484 in 29.391 seconds; `py_compile` and
`git diff --check` passed. The CLI pins its root to
`data/private_nextgqa`, requires current-user ownership and private root/input
modes, and accepts only direct-child names. It has not been exercised in a
browser and no real manifest or reviewer receipt was created. Independent audit
found that the original function-only implementation lacked a practical safe
CLI; after the additions, it confirmed the path, mode, no-follow, size, escaping,
and cleanup controls. Residual limitations are that power loss/process kill can
leave an incomplete exclusively-created HTML file, and the browser's receipt
download destination/mode is not controlled by the page; the reviewer must
verify and secure that local file before custody validation.

## Faithfulness progress and next action

Direct empirical faithfulness remains **0/3 video-claim reviews, 0/3
prediction-link adjudications, 0/3 scored records, 0/15 player-to-decoder
correspondences, and no real-data ΔF, CI, or test**. The renderer and its tests
improve the readiness of the review handoff only. They do not establish that an
explanation predicts or supports a model answer, nor that the internal
intervention improves faithfulness. Rights status remains unverified; the
renderer is designed to keep its HTML inside local protected storage, and no
protected content was published.

Next, build fail-closed preflight binding the existing release/trust-root
ceremony, first-stage human video review, browser-to-decoder clock receipts, and
post-baseline prediction-link receipt. Do not score real data before all these
gates and cohort bindings pass.

The future local invocation (not run because no frozen baseline manifest exists)
is:

```bash
.venv/bin/python scripts/render_nextgqa_prediction_link_review.py \
  --manifest-name <frozen-baseline-manifest.json> \
  --output-name <new-private-review-page.html> \
  --expected-manifest-sha256 <64-hex-frozen-manifest-hash> \
  --protocol-sha256 <64-hex-frozen-protocol-hash>
```
