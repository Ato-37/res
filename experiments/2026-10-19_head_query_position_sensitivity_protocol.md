# Head query-position sensitivity exploration protocol (2026-10-19)

## Question and status

This post-selection exploratory measurement asks whether the existing
attention-shape head ranking depends on measuring attention from the final
input/generation-prefix query versus averaging queries over the question's
body-token span. The earlier full-sample candidates have already been seen, so
this is hypothesis-generating only; it is not a confirmatory head search and
does not reopen the closed head-faithfulness test.

## Frozen design

- Cohort: all 61 rows of the exact `data/nextgqa_samples_big.json` manifest, in
  manifest order. Process each row once at 16 uniformly sampled frames, with
  the locally pinned Qwen2.5-VL-7B-Instruct commit
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
  eager attention, and `max_pixels=200704`.
- For each successful row, measure both query definitions from the same single
  forward and same decoded/processed inputs:
  1. final input/generation-prefix query at `input_ids` position `-1`; and
  2. the equal-token-weight mean of attention from every exact question-body
     subtoken span in `input_ids`.
- Find the question span by exact subsequence matching of tokenizer IDs for the
  original question. It must occur exactly once, be non-empty, and start after
  the complete video-token span. Missing, duplicate, overlapping, or malformed
  spans fail that row; no heuristic offsets or outcome-based repair.
- For each query definition, first sum attention mass over spatial tokens
  within each of the eight temporal bins, then average those per-bin values
  equally over question subtokens. From that per-video/head vector, compute
  video mass, normalized temporal entropy, and one peak bin. Average each
  video's head summaries equally across successful videos; sink frequency is
  computed from one per-video peak, not token-level pseudo-replicates.
- Both rankings use the pre-existing fixed rule: descending mean video mass,
  chord elbow with `min_keep=10`, exclude sink fraction `>=0.5`, then ascending
  mean temporal entropy with ties broken by descending mass, layer, and head;
  report top 20. Do not tune query mask, frame count, filter, threshold, or K
  after inspecting the output.
- Primary comparison is descriptive: top-20 intersection/Jaccard and
  average-tie Spearman correlation of mean temporal entropy among shared
  non-sink survivors. No p-value, confidence interval, or causal/effect claim.
  Coverage and all failures are reported. Both query summaries must have the
  exact same successful row set because they come from one forward.
- Keep row indices, per-video attention metrics, and all results under
  `data/private_nextgqa/`, owner-only; never publish video/question data or
  derived per-head tables. No labels, generated answer text, or faithfulness
  score enter selection or analysis.
- Before model load, the runner performs a CPU processor/token-span preflight for
  every requested row. Only the two fixed frame-decode failure codes may be
  excluded. Any malformed/ambiguous question span, geometry mismatch, processor
  failure, or changed successful row set aborts the run before it writes an
  analysis artifact. During GPU inference any CUDA OOM or other inference
  exception aborts the entire run; it is never converted into a partial result.
- Before writing, both query arrays must have paired identical dimensions,
  finite values, video mass in `[0,1]`, entropy in `[0, ln(8)]` within numeric
  tolerance, and peak bins in `0..7`. Record PyTorch, Transformers, tokenizer,
  qwen-vl-utils, OpenCV, Pillow, and NumPy versions in the private summary.

## Cost and interpretation limits

An engineering smoke on the first 12 manifest rows precedes the full run; it
must exercise non-empty survivor rankings and the paired comparison, and is
excluded from the full-run analysis and denominator.
Prior 16-frame attention capture peaked at 15.25 GiB allocated VRAM and took
about 312 seconds for the 61-row scan (58 successful); expect a similar peak and
roughly 3–6 minutes for the paired query capture, with live pre-run VRAM and
host-memory checks. Stop if the existing validated resource guard is not met.

A difference indicates sensitivity of this attention-shape heuristic to query
position. Similar rankings indicate stability only across these two definitions
on this fixed successful cohort. Neither outcome establishes evidence grounding,
head function, model prediction dependence, explanation quality, or faithfulness.
The candidate method was chosen after inspecting earlier exploratory outputs;
the preregistered head-faithfulness test remains closed and negative.
