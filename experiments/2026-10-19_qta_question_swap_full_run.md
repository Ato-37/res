# QTA same-video question-swap diagnostic: full local run (2026-10-19)

## Purpose

Measure whether changing the question changes the temporal groups selected by
the question-token attention selector more than repeating the same question on
the same fixed video. This is a selector-mechanism diagnostic. It does not
measure whether a selected group contains the evidence, whether the answer
depends on that evidence, or whether an internal intervention improves
faithfulness.

## Frozen setup and provenance

- Cohort: all 80 eligible local NExT-GQA test-associated videos in the frozen
  finite cohort, after excluding the 3 previously used repeatability/wiring
  videos. No replacement was allowed. These are not a fresh independent sample.
- Question-pair eligibility and selection used only distinct question text,
  QIDs, and video IDs. The historical annotation-validity selector was used
  only to reconstruct/exclude the three known prior videos; it did not affect
  new candidate eligibility or question-pair choice.
- Per video: identical 32-frame video inputs for two distinct questions;
  exact same processor batch reused for A1/A2 and B1/B2. Primary contrast is
  `d = J_repeat - J_cross`, where repeat overlap is the mean of A1/A2 and
  B1/B2 Jaccard and cross overlap is the mean of all four A-to-B Jaccards.
- Model: pinned Qwen2.5-VL-7B revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16 weights, 4-bit weights
  with BF16 compute, eager attention, seed 20261018; RTX 5070 Ti, CUDA 12.8,
  PyTorch 2.11.0+cu128.
- Full result: private local artifact
  `data/private_nextgqa/qta_question_swap_full_20261019_0030.json`, mode 0600
  under the mode-0700 private root. It contains row-level selector groups,
  question token counts and video fingerprints; it remains local and is not
  included in a publication. Artifact SHA-256:
  `44b79f7170b1a05b6199524e9d8f2baef85ada08aae10deb6cdbeb838cd3e4d3`.
  Pinned model snapshot SHA-256:
  `feda551cbe8222314cfbafcd58213f74543519f24162c2f237a4af37a50e06d6`.
  Rights remain unverified.
- The run artifact records source hashes for 15 code/runtime modules and the
  pinned model snapshot hash before and after execution. Independent readback
  reproduced all 15 source hashes and confirmed the before/after model hashes
  agree with the current pinned snapshot. Its plan digest resolves to an
  identical-byte private freeze manifest; 5 same-content copies were found
  locally. The plan records 80 unique media hashes, 3 explicit prior-video
  exclusions, and exact-byte overlap of 0/147, 0/147, and 0/490 with the three
  named local snapshots. These are local-snapshot comparisons, not proof of
  historical freshness or source/event independence.

## Execution and verified results

- A first smoke did not enter forward because the pre-forward free-VRAM check
  failed; its failure handler also had an out-of-scope accounting import. A
  subsequent run completed 3/4 forwards, then found 0 free bytes before the
  fourth. The failure artifact captured the incomplete denominator. The runner
  was repaired to release freed CUDA allocator blocks after each forward and
  to keep failure accounting available to the failure handler.
- The repaired four-forward smoke passed, then passed again after adding
  persisted capture-shape/dtype/layer-call evidence. Independent audit verified
  exact visual tensor equality, the repeated-batch digests, 4/4 receipts, BF16
  normalization tolerance, and source/model/plan provenance.
- Full run: **80/80 videos, 320/320 requested/attempted/completed forwards,
  0 failed videos/forwards, 0 unattempted forwards**. Every capture receipt
  reports the expected 12 layers and 28 heads; all 320 captures used BF16 and
  verified one call per expected layer. Peak attention-row normalization
  deviation was 0.003228, under the BF16 tolerance 0.0041. The runner's
  measured interval from post-load start to completion was 1,596.8 s, with
  1,191.4 s in forward/capture. This excludes the earlier full-cohort media
  preflight and model load.
- Descriptive finite-cohort results, independently recomputed from all 80
  row-level contrasts:

  | Quantity | Result |
  |---|---:|
  | Mean `d` | 0.2542 |
  | Sample SD | 0.3567 |
  | Median | 0 |
  | Range | [0, 1] |
  | Positive / zero / negative | 28 / 52 / 0 |
  | Same-question repeat selections identical | 80/80 |

  All row-level `J_repeat`, `J_cross`, and `d` arithmetic matched the aggregate
  summary. No p-value or population interval was computed. The frozen cohort is
  test-associated and source/event clustering and prior exposure are unresolved.

### Post-outcome alternative-explanation check (exploratory)

Using only the local row aggregates, the absolute question-token-count gap
averaged **5.00** tokens (median 4, range 0–12) in the 28 pairs with `d>0`,
versus **2.81** tokens (median 2.5, range 0–8) in the 52 pairs with `d=0`.
Across all 80 rows, descriptive Pearson `r` between absolute token-count gap
and `d` was **+0.4355**. An independent reader reproduced these values. This
is a post-outcome exploratory description with no p-value or interval. Token
count gap is only a coarse proxy; question meaning, lexical content, and
tokenization are confounded. It does not explain the selector changes or
support a causal/population claim and was not used to tune the protocol.

## Interpretation and limitations

In this exact setup, all repeated same-question runs selected the same groups;
28 of 80 distinct-question pairs selected groups with lower cross-question
overlap, while 52 pairs selected the same groups. The finite-cohort mean
contrast therefore reflects question-conditioned selector changes beyond the
observed same-question repeat variation. It can also reflect wording,
tokenization, and lexical differences. It does not establish semantic evidence
quality, explanation-video correspondence, answer prediction dependence,
internal-intervention benefit, or faithfulness. These results cannot support a
population claim or replace human judgments.

Direct faithfulness remains 0/3 reviewed claims, 0/15 verified
browser-to-decoder frame correspondences, 0/3 prediction-link judgments, and
0/3 direct scores; no direct `DeltaF`, interval, or test exists. The human
review packet is candidate-only: a trusted root fingerprint, signed role
assignments and release rosters, review-only freeze receipt, and witnessed
unopened handoff remain missing. The role-assignment template is ready, but no
reviewer identities or keys were supplied in this workspace.

## Human-review readiness update

After the full run, the local hash-only pre-review inventory was refreshed at
`data/private_nextgqa/pre_review_inventory_candidate_20261019_0024/`.
Independent structural readback confirmed **82 unique slots, 0 missing**,
SHA-256 formatting for every slot, directory/file modes 0700/0600, and flags
`candidate_only`, `review_started=false`, `review_completed=false`,
`joint_freeze=false`, `score_ready=false`, `execution_authorization=not_granted`.
Its local SHA-256 is
`e8a7613d278bc53dc9eb7c0e38f9af2bf30583759601df124bd5c1a4ed3b3767`.
The builder suite passed 20/20. This is an inventory refresh, not a review,
identity verification, protocol freeze, or scoring authorization. Reviewer
identities/keys and an independently pinned trusted root remain unavailable.

## Next action

Do not tune the selector using this result. Keep the protected cohort, review
inventory, and full result local. Continue independent, human-free feasibility
work on the preplanned internal-intervention scorer and its software bindings
without converting this diagnostic into a faithfulness claim. Once the
independent role/root inputs are available, resume the human review ceremony
and score only after its required audits pass.
