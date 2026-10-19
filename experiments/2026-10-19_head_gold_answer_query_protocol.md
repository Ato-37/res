# Gold-answer-conditioned head-query exploration protocol (2026-10-19)

## Question and scope

Earlier scans compared video attention from the final generation-prefix query
and from question-body subtokens. This fixed, post-selection exploratory scan
adds one distinct query definition: the mean attention of the gold-answer
tokens when appended after the assistant generation prefix. It asks whether
attention-shape ranking is sensitive to conditioning on answer tokens. Gold
answers are used only to define this query position; they are not used as a
faithfulness outcome, head target, or intervention criterion. This is not a
confirmatory test and does not reopen the closed head-faithfulness branch.

## Frozen design

- Use all 61 ordered rows of `data/nextgqa_samples_big.json`, once each, with
  Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, local-only 4-bit weights,
  BF16 compute, eager attention, 8 sampled frames, and `max_pixels=200704`.
- For each row, make one ordinary question + video input with a generation
  prefix, append the row's exact tokenized gold answer, and do one forward.
  Record per-head temporal video mass, normalized temporal entropy, and one
  peak temporal bin for (a) final generation-prefix query, (b) exact
  question-body subtokens, and (c) appended gold-answer subtokens. Require the
  answer-token subsequence to match exactly once after the prefix and question
  spans; otherwise abort the row/run rather than repair offsets heuristically.
- Reuse the fixed ranking rule: descending mean video mass, chord elbow
  (`min_keep=10`), sink fraction `<0.5`, then ascending mean temporal
  entropy, ties by descending mass/layer/head; report top 20. Do not tune any
  parameter after seeing results.
- Primary descriptive comparison: gold-answer query versus generation-prefix
  query and question-body query, using top-20 intersection/Jaccard and
  average-tie Spearman entropy ranks among shared eligible heads. No p-values,
  intervals, or causal claims. Same rows and same forward must underlie all
  three summaries.
- CPU-preflight every row before loading weights. Exclude only fixed sampled
  frame decode failures. Any other preflight/inference error aborts. Keep raw
  row indices and per-head/video arrays owner-only in
  `data/private_nextgqa/`; publish no video-derived arrays or answer text.

## Resource and interpretation limits

Run a 3-row smoke first. Prior 8-frame paired query capture peaked at 8.53 GiB
and took about 53 seconds for 58 successful rows; budget about 2–4 minutes
plus model load and smoke, with current resource checks. A differing ranking
shows only query-conditioning sensitivity on this previously inspected cohort;
similarity shows only heuristic stability. Neither establishes grounding,
prediction dependence, or faithfulness. Rights status remains unverified.
