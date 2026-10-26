# Exploratory QTA generated-text / annotation-dependence protocol and CPU primitives (2026-10-19)

## Purpose

Define a human-free, generated-text likelihood proxy that avoids the mechanical
coupling in the first proposal (ablating the exact keys receiving the QTA bias).
Evaluation ablations are now fixed by official `gsub` timing groups and a
matched control; QTA-selected groups remain a separate frozen treatment and
their overlap is explicitly recorded. This remains a benchmark-annotation
dependence proxy, not direct faithfulness or a hallucination judgment.

## Implementation and checks

- Added `scripts/qta_multiquery_video_key_intervention.py` for fixed multi-query
  edits to Qwen's eager additive attention mask and compact target-logit-window
  scoring. It validates causal positions, key visibility, exact shapes, and
  finite values; ablation takes precedence if a key is also biased. Mock module
  tests verify hook application/cleanup, and a raw generation-suffix helper
  checks the exact prompt prefix before freezing target IDs and EOS status.
  Exact equal flattened evidence/control key budgets and separate selector
  overlap reporting are checked by pure helpers. Twelve intended production
  layers (5–16) are tested with mock modules.
- Added `scripts/freeze_qta_generated_text_candidate_cohort.py`, which read 343
  local private JSON files, gathered 602 known media hashes, excluded 83/500
  local test-video matches, and froze 417 candidates in deterministic
  outcome-blind order. The resulting lock is private mode 0600 and ignored by
  Git; it explicitly says historical exposure is not exhaustive.
- Added CPU-only geometry preflight and a local GPU runner. The preflight
  passed for four locked candidates: 4/4 effective, 0 failures, 4 candidates
  attempted, 16 frames/video, and equal flattened annotation/control key counts
  for all four. The first candidate is reserved for an engineering smoke and
  the next three for a separate descriptive proxy run.
- Inspected the installed Transformers Qwen2.5-VL `forward` implementation:
  integer `logits_to_keep=L+1` selects the trailing `L+1` logit rows. The scorer
  consumes the first `L` rows in that window, aligned to absolute queries
  `P−1 … P+L−2`; the extra final row is not scored.
- Added synthetic tests for query-row and key locality, overlap precedence,
  causal/visibility rejection, target logit row offset, mean target-token
  log-likelihood, and the four-condition difference-in-differences arithmetic.
- Initial focused test run: **17/17 passed**; after the runner failure-accounting,
  input-index, selector-call, and partition fixes the suite is **25/25 passed**.
  Python compilation and scoped
  `git diff --check` passed.
- Two individually authorized local GPU smoke attempts loaded the pinned 4-bit
  BF16 model on the RTX 5070 Ti (observed peak allocation 7.22 GiB) but produced
  **no completed score**. Attempt 1 stopped before inference on a missing `csv`
  import; the index-loading helper and a real-shape unit test now cover that
  path. Attempt 2 reached the selector stage but raised `ValueError`; its
  private failure receipt records 1 selector attempt / 0 completion and 0/6
  scoring forwards, but that version did not preserve the error code. A later
  private-receipt change now retains a truncated local failure code for diagnosis.
- A CPU diagnostic on the frozen smoke candidate confirmed 2,152 input tokens,
  19 contiguous question tokens, 8 temporal groups, and exact match to the
  geometry digest. Attempt 3 recorded `captured_attention_shape_invalid`.
  Inspection of the producer showed its compact rows are `[heads, question,
  sequence]` (3D), while the new aggregator incorrectly required a 4D batch
  axis. The aggregator and fixture now match the actual producer shape;
  **25/25** focused tests pass. Neither failed attempt is an efficacy result.

## Interpretation and limits

The earlier sentence packet lacks raw continuation IDs, so its decoded text
cannot be treated as the exact generated sequence. This runner generates once
on each frozen candidate and preserves the exact raw suffix IDs. The four
candidate hashes do not match the current local private JSON inventory, but
complete historical exposure remains unknown. `gsub` is QA timing, not verified
support for a particular claim. Any result describes only conditional model
likelihood sensitivity to annotation-key removal.

## Faithfulness status and next action

- Direct NExT-GQA faithfulness remains **0/3 claims scored**; formal claim
  review **0/3**, clock/frame correspondence **0/15**, prediction-link review
  **0/3**. No direct `ΔF`, confidence interval, or hypothesis test exists.
- Historical head-intervention confirmation remains negative (79/79;
  −0.0756 nat; 95% t CI [−0.1578,+0.0065]); the head/layer/alpha search remains
  closed.
- Independent audit cleared the source-manifest, partition, and synthetic
  failure-accounting blockers. After fixing the captured-row dimensionality,
  source-bound preflight `_11` passed 4/4 candidates, 0 failures, with 13
  source hashes matching current files. The row-0 GPU smoke completed:
  selector 1/1, deterministic generation 1/1 (9 raw tokens, EOS observed),
  and all six score conditions 6/6, with 0 failures. Peak allocated VRAM was
  7.33 GiB; elapsed time was 53.37 seconds. The separate indices 1–3 run
  completed selector/generation 3/3 and all 18 score forwards, with 0
  failures, in 94.23 seconds and peak allocated VRAM 7.34 GiB. Its private
  receipt and all four dataset hashes and 13 current source hashes were
  independently checked; per-row contrasts were recomputed from the stored
  log-probabilities. A mode-0600 supplementary linkage audit now records the
  smoke, finite-run, preflight, cohort-lock, source and dataset hashes without
  altering either original run receipt.
- Observed finite-row values: baseline `A=-0.025585`, QTA `A=-0.015209`,
  `ΔA=+0.010375 nat/token`; QTA-selected 252 visual keys overlapped neither
  evidence nor control keys. Both A values are negative. With n=1 this is only
  a wiring/finite-row observation; no interval, p-value, or efficacy inference
  is warranted. The existing full-cohort QTA/answer-margin proxies will not be
  rerun.
- For rows 1–3, the observed ΔA values were `−0.018309`, `−0.003430`, and
  `+0.022023 nat/token`; their mean was `+0.0000948`, median `−0.003430`, and
  range `[−0.018309,+0.022023]`. Signs were negative for 2/3 and positive for
  1/3. No interval, p-value, or population inference was computed. QTA's 252
  selected keys overlapped the control set completely (252/252) in two rows
  and not at all in one; overlap with evidence was 0/3. Since ablation takes
  precedence, the two complete overlaps alter the intended selective contrast.
  Do not exclude those rows post hoc: treat the pooled estimate as a weakly
  interpretable exploratory feasibility description, not clean selective
  dependence or efficacy evidence.
- The finite3 receipt does not itself record the smoke receipt's name/hash.
  Supplementary local linkage manifest
  `data/private_nextgqa/qta_generated_text_annotation_dependence_local_20261019_linkage_audit_01.json`
  binds both receipts by their hashes and shared preflight/source hashes, and
  verifies all current source/data hashes. It is mode 0600 and must remain
  local. This repairs handoff traceability without rewriting a result.
- Next: stop this endpoint pending a new, pre-outcome protocol that prevents
  QTA/control overlap or estimates overlap interactions with a separately
  justified design. Do not run another GPU cohort on the current post-outcome
  specification. Continue the distinct direct-faithfulness / human-review
  handoff branch; the simple nickname-to-question workspace is only an
  informal check, and no formal review response JSON is present.

## Prospective v2 control design (no outcome run)

Following the independent audit, added the separate prospective design
[`docs/qta_generated_text_annotation_dependence_protocol_v2_exploratory.md`](../docs/qta_generated_text_annotation_dependence_protocol_v2_exploratory.md).
It does not alter or reinterpret v1. A new pure utility
`scripts/qta_disjoint_control_design.py` selects equal-count rank-distance
controls while excluding both evidence and QTA groups, rejects treatment/evidence
overlap, and fails closed when no eligible control exists. An exhaustive small
state-space oracle test checks the optimum and lexicographic tie rule; malformed
indices and infeasible cases are covered. The new helper plus the unchanged v1
QTA primitives pass **19/19** focused tests, `py_compile`, and `git diff --check`.
The helper is not yet wired into a production runner, and no v2 selector-only
or scoring GPU run has been performed. The next technical step is a selector-only
GPU feasibility receipt on a prespecified fresh hash-order candidate pool,
before any generation or target scoring, followed by an independent audit.

## Informal human-check handoff verification (2026-10-19)

The local quick-check artifact and its Japanese launcher were checked against
the user's requested low-friction flow: nickname, then embedded video and
question. It is explicitly a three-item impression check, not a full hallucination
judgment or formal blinded review. There is no file picker, frame/time entry, or
network submission in the rendered page. The artifact builder's focused tests
passed **18/18** when invoked with `PYTHONPATH=scripts`; an initial invocation
without that import path failed at test-module import and is not a product-test
failure. The protected HTML is mode 0600 and about 30.7 MB because the three
videos are embedded for offline use. No `exploratory_video_review_*.json`
response is currently present. A nickname and downloaded answers do not verify
identity, independent review, completed viewing, or claim truth.

The local launcher is `docs/nextgqa_human_review_workspace_ja.html`; from this
checkout it opens with `xdg-open docs/nextgqa_human_review_workspace_ja.html`.
The underlying protected artifact and any response JSON must remain local. This
informal check does not fill the formal review gates: claim review 0/3,
clock/frame correspondence 0/15, prediction-link review 0/3, and direct
faithfulness score 0/3.

Rights status remains unverified. All protected data and derived results stay
local; no publication is part of this exploratory work.
