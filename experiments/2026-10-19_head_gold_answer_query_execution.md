# Gold-answer-conditioned head-query exploration execution (2026-10-19)

## Purpose and fixed setup

To use otherwise idle GPU time on a distinct, explicitly exploratory head
question, this scan added the mean temporal attention at gold-answer token
positions to the existing generation-prefix and question-body query summaries.
The protocol is
`experiments/2026-10-19_head_gold_answer_query_protocol.md` (SHA-256
`bc89608fd2694aa010d406a63d9756ce2688d7f793817645ea1bc651d8f3cd36`). It
uses 8 sampled frames, four temporal bins, the pinned local Qwen2.5-VL-7B
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16
compute, eager attention, and the fixed 61-row ordered manifest.

Gold answer text was not emitted or used as a score. The runner appended the
tokenized answer IDs after the assistant generation prefix in the same
forward, then summarized attention rows at the final prefix position, exact
question subtokens, and appended answer tokens. This measures attention after
answer tokens have been supplied; it is not the attention used to predict
those same tokens.

## Execution and provenance

- A 3-row smoke succeeded after one engineering correction. The first attempt
  exposed that `mm_token_type_ids` also needs text-type zeros appended along
  with answer IDs and `attention_mask`; it aborted before writing an artifact.
  The corrected smoke passed 3/3, 5.048 seconds inference, 8.537 GiB peak
  allocated VRAM.
- Full GPU scan requested 61 rows and completed 58. Three fixed frame-decode
  failures were excluded; no inference failure was converted into an
  exclusion. Inference took 62.423 seconds and peaked at 8.542 GiB allocated
  VRAM on the RTX 5070 Ti. The 58 successful rows and three failures were
  disjoint and covered all 61 manifest rows.
- Private summary SHA-256:
  `1da8afd4c4efafa82bbc533c3fda5e73b6cde0ba1e38496bcdfffc6fa7c228af`.
  Per-row metric NPZ SHA-256:
  `cff1c7fcd3f4612a7fd52ba13ec9e540b9302360519556025e9839453eba6e3a`.
  Both are mode `0600` under mode-`0700` `data/private_nextgqa/`.
- Executed runner source is preserved at
  `scripts/run_head_gold_answer_query_exploration_executed_20261019_v1.py`,
  SHA-256 `5e5259b5642ee41471af9e3add8066b2245ee796313680498437a4a657542caf`,
  matching the hash embedded in the run summary. The maintained runner adds
  an all-row answer-token preflight helper after the run; 3 new helper tests
  and 8 paired-query tests passed (11/11), and Python compilation passed.
- An independent audit confirmed the dataset schema (61 nonempty answers),
  privacy properties, and Qwen's appended `mm_token_type_ids` interface. One
  process deviation remains: the runner version used for the full run did
  not check all answer tokenizations before model loading. A separate local
  CPU pass checked all 61 while inference was underway and found zero empty
  tokenizations. The maintained runner now checks this before loading weights;
  the preserved executed source and the deviation are kept explicit here.
- Primary-side independent CPU recomputation from the private arrays
  re-derived all chord elbows, sink survivors, top-20 sets, overlaps, and
  average-tie Spearman values. It matched the stored rankings and comparisons.

## Observed comparisons

| Query compared with generation-prefix | Eligible heads (prefix / query) | Top-20 intersection / union | Jaccard | Shared eligible heads | Entropy-rank Spearman |
|---|---:|---:|---:|---:|---:|
| Question-body mean | 15 / 56 | 0 / 35 | 0.000 | 5 | 0.700 |
| Gold-answer mean | 15 / 38 | 1 / 34 | 0.029 | 5 | 0.700 |

These are descriptive, post-selection quantities on the same 58 videos.
Because only five eligible heads were shared, the rank correlation is based
on a very small intersection. There is no p-value, interval, independent
validation cohort, or basis to select a head from these results.

## Interpretation, faithfulness status, and next action

The attention-shape top set changed almost completely across all three query
definitions. This reinforces that the current heuristic is query-position
sensitive; it does not show that answer-conditioned attention is more
faithful, grounded, or causally predictive. The formal head-faithfulness
confirmation remains closed and negative (79/79, estimate `-0.0756` nat,
95% t CI `[-0.1578, +0.0065]`, below the `+0.25` nat practical threshold).
Direct faithfulness evidence remains **0/3** formal claim reviews, **0/15**
verified clock/frame correspondences, **0/3** prediction-link judgments, and
**0/3** direct scores; there is no direct `ΔF`, interval, or test.

Do not continue changing query definitions or selecting heads from this same
cohort. Keep the new gold-answer query mode as a descriptive code path only;
return GPU priority to a prespecified intervention-based faithfulness test
once its independent-review and execution gates are ready. Rights status for
local media remains unverified; all video-derived arrays stay local and are
not committed or published.
